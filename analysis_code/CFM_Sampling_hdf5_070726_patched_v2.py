"""
CFM_Sampling_hdf5_070726_patched_v2.py
===================================
Patched version of CFM_Sampling_hdf5_033126.py.

PATCH NOTES (numbering follows the bug review):
  1. AddNet is now constructed with the GFTE kwargs (GFTE, GFTE_embed_dim,
     GFTE_scale, GFTE_learnable) read from MODEL_PARAM_DICT via .get(), so
     both GFTE-era checkpoints and older non-GFTE checkpoints load.
  2. L_CUSTOM (and therefore the model-space `dim`) is now determined ONLY
     by the training-time rule (feature_names[0] == 'l' and NORM_METHOD ==
     'CUSTOM'), never by the run-mode flags.  A separate `dim_phys` tracks
     the physical feature count.  This matches CFM_Eval_Gaia.py's
     _detect_l_custom().
  3. The normalization stats file is now P_Z_Norm_Dict_Sample_Ests_062326.npy
     (same file training and the eval script use), and the script asserts
     that no feature was silently dropped from the normalization tables.
  4. denormalizer() is L_CUSTOM-aware: it maps (N, dim) model-space arrays
     to (N, dim_phys) physical arrays with correct indexing (no more
     samples[:, dim] / feature_names[dim_phys] IndexError).
  5. The bespoke conditional-permutation lists use pml/pmb (not pmra/pmdec),
     and are filtered against the actual feature list with a loud warning
     instead of an IndexError.  DROP_ONE_SETS additionally converts
     MASK_SETS_INDICES from model space to physical space under L_CUSTOM.
  6. The test-set save filenames are un-swapped: the raw (physical-unit)
     subset is saved as Pure_Sample_*, the normalized subset as
     {MODEL_NAME}_*.
  7. Filenames encode the test fraction correctly (e.g. '0p5pct' for 0.005)
     instead of int(TEST_FRACTION)*100 == 0 for every fraction < 1.
  8. Pure_Sample filenames now embed a feature-set tag (count + short hash of
     the ordered feature names), so caches built with pmra/pmdec can never be
     silently loaded by a pml/pmb model.  Loads also assert the column count.
     PURE_SAMPLE_OVERRIDE lets you point at any legacy file explicitly.
  9. Metrics are unit-consistent: the Sinkhorn/Wasserstein distances are
     computed in NORMALIZED model space for both samples and truth (all
     features on comparable scales), and the median-imputation errors are
     computed in PHYSICAL units by denormalizing both the per-source median
     sample and the truth (with a wrap-aware difference for l).
 10. The median-error validity selection uses initial_mask > 0.5 (1 = valid
     observation, matching the rest of the pipeline), not < 0.5.
 11. x_test is ALWAYS in normalized model space with shape (N, 2, dim) by
     the time create_test_loader() returns, no matter which combination of
     USE_PURE_SAMPLE / SAVE_TEST_SET / LOAD_TEST_SET is set.  A freshly
     built pure sample is saved raw (physical units, dim_phys columns --
     the exact format CFM_Eval_Gaia.py's _load_eval_set expects) and then
     normalized in memory, so a single run can save the pure sample AND
     sample from the model.

  Minor: the Shard assert no longer NameErrors on bare `batch_size`; the
  unconditional branch prints "Unconditional Sampling".

Pure_Sample file contract (shared with CFM_Eval_Gaia.py):
    tensor of shape (N, 2, dim_phys); [:,0,:] = PHYSICAL units (raw Gaia
    features, l as a single degree-valued column); [:,1,:] = validity mask
    (1 = observed), one column per physical feature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import math
import hashlib
from datetime import datetime
from geomloss import SamplesLoss
from pathlib import Path
from hdf5_dataset_creator_020926 import HDF5IterableDataset
from torch.utils.data import Dataset, DataLoader
from gaia_feature_normalizer_020926 import feature_denormer, feature_normer
import json
import model_library_032526 as CFM_lib
sigmoid_op = nn.Sigmoid()


MODEL_NAME = 'POS_PM_mask_set_setA_CUSTOM_2026-07-10_15-01-16_mask_train_mask_val_MSE_vHD512_SIGMA1_0e-08_GELU_lr1_0e-04_wd0_0e+00_layers8_MP85_ADD_SET_VALACUMVAL_10'
#'POS_PM_mask_set_setA_CUSTOM_2026-07-02_11-22-30_mask_train_mask_val_MSE_vHD512_SIGMA1_0e-08_GELU_lr_max_5_0e-03_wd0_0e+00_layers8_MP_RAND_ADD_SET_VALACUMVAL_10'
LOAD_NAME = 'POS_PM_mask_set_setA_CUSTOM_2026-07-10_15-01-16_mask_train_mask_val_MSE_vHD512_SIGMA1_0e-08_GELU_lr1_0e-04_wd0_0e+00_layers8_MP85_ADD_SET_VALACUMVAL_10_ema.pth'
#'POS_PM_mask_set_setA_CUSTOM_2026-07-02_11-22-30_mask_train_mask_val_MSE_vHD512_SIGMA1_0e-08_GELU_lr_max_5_0e-03_wd0_0e+00_layers8_MP_RAND_ADD_SET_VALACUMVAL_10_ema.pth'

AUTOENCODER_MODEL_NAME = ''  # Enter name of autoencoder here!
MODEL_PARAM_DICT = np.load('YOUR PATH/model_param_dicts/' + MODEL_NAME + '_model_param_dict.npy', allow_pickle=True).item()

# Controls whether a model is loaded
# note that if a model is not loaded, all that will be done is that data samples will be drawn. This is pointless unless you then save those samples!
USE_MODEL = True

# Where is the data?
stats_dir = Path("YOUR PATH/jsoltis")
data_dir = 'YOUR PATH/GaiaSource_12182025'

# PATCH 3: use the SAME normalization stats file as training (and CFM_Eval_Gaia.py).
# If your copy lives in the training scratch dir instead, change stats_dir or this name.
STATS_FILE = "P_Z_Norm_Dict_Sample_Ests_062326.npy"

# Use subset A validation sample that has no built in normalization
USE_PURE_SAMPLE = True
# If true, save the generated subsample of the test set so that it can be more quickly reloaded in future runs
SAVE_TEST_SET = True
# If true, load a predefined subsample of the test set instead of stepping through hdf5 files
LOAD_TEST_SET = False

# PATCH 8: optional explicit path to a pure-sample file (e.g. a legacy file whose
# name doesn't carry the feature tag). Leave '' to use the tagged default name.
# NOTE: a legacy file is only valid if it was built with the SAME ordered
# feature list as this model -- the column-count assert will catch gross
# mismatches (e.g. old 9-feature pmra/pmdec files vs a different feature count)
# but cannot detect a same-length feature swap; use with care.
PURE_SAMPLE_OVERRIDE = ''#'YOUR PATH/true_samples/Pure_Sample_9feat_771f68a4_Omega_Centauri_eDR3_data_010521__1__2026-07-08_06-18.pt'

SAMPLE_NAME = 'Full_Sky_'#'Omega_Cen_'#'Full_Sky_'


# Which version of this test fraction do you want for your load? Relevant only if LOAD_TEST_SET = True or SAVE_TEST_SET = True
LOAD_NUMBER = 10

# Calculate Earthmover distances compared to test set
METRICS = False

# Use real data for data generation
COMPARE_WITH_TEST = True

# SAVE SAMPLES
SAVE_SAMPLES = True

# Define a Sinkhorn (~Wasserstein) loss between sampled measures
# Currently using default parameters
sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2)

# Do unconditional sampling
UNCONDITIONAL = True

# Number of unconditional samples, note that if COMPARE_WITH_TEST is true, then the effective number of unconditional samples will be N_test_stars * N_uncond_sam, so set N_uncond_sam accordingly
N_uncond_sam = 1  # 000

# Use Test Output Masks
TEST_OUTPUT_MASKS = True

# Do conditional sampling -- This can take a while
CONDITIONAL = True

# If true, generate conditional samples where all but one feature is given to the model.
DROP_ONE_SETS = True
KEEP_ONE_SETS = True

# PATCH 5: If the MASK_SETS_INDICES stored in the param dict are model-space
# indices (i.e., they index the split-l array used during training), set this
# True so DROP_ONE_SETS converts them to physical feature indices. Models
# trained before the l split (or with L_CUSTOM False) should set this False.
MASK_SETS_INDICES_ARE_MODEL_SPACE = True

# Number of conditional samples. Like N_uncond_sam, this number is multiplied by the total test set size, so be careful setting it too high!
num_samples = 1

# Sampling batch_size
# Sampling batch_size has to be smaller than training batch size because I am increasing the dimensionality by the number of samples!
s_batch_size = 100000  # 16

# This is the fraction of the test set that you want to use for the test set comparisons
# You will get roughly this amount for each run
# It is applied by shuffling an index for each batch generated by the test loader and then selecting this fraction of that batch
# Note that the full test set has 180 million sources in it, so this fraction should not be very large!
TEST_FRACTION = .01

# PATCH 7: human-readable, collision-free fraction tag for filenames.
# e.g. TEST_FRACTION=0.005 -> '0p5pct'; 0.05 -> '5pct'.
FRACTION_TAG = f"{100 * TEST_FRACTION:g}".replace('.', 'p') + 'pct'

if UNCONDITIONAL == True:
    SAMPLE_NAME += 'UNCOND_N' + str(N_uncond_sam) + '_'
if CONDITIONAL == True:
    SAMPLE_NAME += 'COND_N' + str(num_samples) + '_'
SAMPLE_NAME += 'TEST_FRACTION_' + FRACTION_TAG + '_'

timestamp = datetime.now()  # Format as human-readable string with underscores
timestamp = timestamp.strftime("%Y-%m-%d_%H-%M")
print("Y-M-D_H-M", timestamp)  # Example: 2025-09-22_14-18-45

rng = np.random.default_rng()
_ = torch.random.manual_seed(0)

# Name of the files with the index ids
SET_NAME = '_'

# Number of process to spawn. Different than when used in training!
N_WORKERS = 24

# Number of integration steps
RK4_STEPS = 50

if MODEL_PARAM_DICT['READ_TYPE'] == "Shard":
    assert MODEL_PARAM_DICT['batch_size'] is not None, "If using Shard files (i.e., READ_TYPE is Shard), batch_size cannot be None."
    # PATCH (minor): was a NameError on bare `batch_size`.
    assert MODEL_PARAM_DICT['batch_size'] > 0, f"Batch size must be greater than 0. Currently batch size is {MODEL_PARAM_DICT['batch_size']}."
    assert MODEL_PARAM_DICT['shard_columns'] is not None, "Must enter columns used in Shard files! shard_columns cannot be None."

print('##################################################')
print('##################################################')
print(MODEL_NAME)
print('##################################################')
print('##################################################')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device = {device}")
# PATCH 3: same stats file as training / eval.
preprocess = np.load(stats_dir / STATS_FILE, allow_pickle=True).item()

# Load JSON with bool/string normalizations (note that bools and strings in the hdf5 files are saved as ints)
with open(stats_dir / "training_stats_bool_string_features.json", "r") as f:
    str_bool_preprocess = json.load(f)

if MODEL_PARAM_DICT['SET_TYPE'] == 'M':
    dict_key_set = 'main_train'
elif MODEL_PARAM_DICT['SET_TYPE'] == 'A':
    dict_key_set = 'subsetA_train'
elif MODEL_PARAM_DICT['SET_TYPE'] == 'B':
    dict_key_set = 'subsetB_train'

# Only save the normalizations of relevant features
feature_dict = {}
for feature_name in MODEL_PARAM_DICT['feature_names']:
    if feature_name in preprocess[dict_key_set]:
        feature_dict[feature_name] = preprocess[dict_key_set][feature_name]
    elif feature_name in str_bool_preprocess[dict_key_set]:
        feature_dict[feature_name] = str_bool_preprocess[dict_key_set][feature_name]
    else:
        print(f"Skipping {feature_name}")

feature_names = list(feature_dict.keys())
print(feature_names)

# PATCH 3: a silently skipped feature would shrink dim and desynchronize the
# feature list from the checkpoint. Fail loudly instead.
assert len(feature_names) == len(MODEL_PARAM_DICT['feature_names']), (
    f"Normalization tables are missing "
    f"{sorted(set(MODEL_PARAM_DICT['feature_names']) - set(feature_names))} "
    f"for set '{dict_key_set}' in {STATS_FILE}. The model was trained with "
    f"{len(MODEL_PARAM_DICT['feature_names'])} features; refusing to continue "
    f"with {len(feature_names)}."
)

# ---------------------------------------------------------------------------
# PATCH 2: L_CUSTOM / dim bookkeeping.
# L_CUSTOM follows the TRAINING rule only (matches CFM_Eval_Gaia._detect_l_custom):
# the checkpoint's architecture depends on it, so it must never depend on the
# run-mode flags. dim_phys counts physical features; dim is the model width.
# ---------------------------------------------------------------------------
dim_phys = len(feature_names)
L_CUSTOM = (feature_names[0] == 'l' and MODEL_PARAM_DICT['NORM_METHOD'] == 'CUSTOM')
dim = dim_phys + (1 if L_CUSTOM else 0)
print(f"dim_phys = {dim_phys}, L_CUSTOM = {L_CUSTOM}, model dim = {dim}")

# PATCH 8: feature-set tag for pure-sample filenames: count + short hash of the
# ORDERED feature list. Any change to the feature set (e.g. pmra/pmdec -> pml/pmb)
# changes the tag, so stale caches can never be silently loaded.
FEATURE_TAG = f"{dim_phys}feat_" + hashlib.md5('_'.join(feature_names).encode()).hexdigest()[:8]
print(f"Feature-set tag: {FEATURE_TAG}")

PURE_SAMPLE_PATH = (PURE_SAMPLE_OVERRIDE if PURE_SAMPLE_OVERRIDE != '' else
                    f"YOUR PATH/Pure_Sample_{FEATURE_TAG}_val_fraction_{FRACTION_TAG}_{LOAD_NUMBER}.pt")
NORMED_SAMPLE_PATH = f"YOUR PATH/{MODEL_NAME}_val_fraction_{FRACTION_TAG}_{LOAD_NUMBER}.pt"

# ---------------------------------------------------------------------------
# Model construction (model-space dim).
# ---------------------------------------------------------------------------
if MODEL_PARAM_DICT['LATENT_FLOW']:
    autoencoder_hd = 1024
    autoencoder_model = CFM_lib.Autoencoder_Model(input_dim=2 * dim, output_dim=dim, hidden_dim=autoencoder_hd, encoder_layers=MODEL_PARAM_DICT['encoder_layers'], decoder_layers=MODEL_PARAM_DICT['decoder_layers'], bottleneck=MODEL_PARAM_DICT['BOTTLENECK_SIZE'], ACTIVATION_TYPE=MODEL_PARAM_DICT['ACTIVATION_TYPE'], FINAL_ACTIVATION=MODEL_PARAM_DICT['FINAL_ACTIVATION'], IOB_METHOD=MODEL_PARAM_DICT['IOB_METHOD'], SPARSE_AUTOENCODER=MODEL_PARAM_DICT['SPARSE_AUTOENCODER']).to(device)

    autoencoder_model.load_state_dict(torch.load('/expanse/lustre/projects/sts101/jsoltis/' + AUTOENCODER_MODEL_NAME + '.pth', weights_only=True))

    model = CFM_lib.DenseNeuralNet(input_dim=2 * MODEL_PARAM_DICT['BOTTLENECK_SIZE'] + dim + 1, hidden_dim=MODEL_PARAM_DICT['v_hidden_dim'], n_layers=MODEL_PARAM_DICT['n_layers'], ACTIVATION_TYPE=MODEL_PARAM_DICT['ACTIVATION_TYPE'], output_dim=MODEL_PARAM_DICT['BOTTLENECK_SIZE'], final_act=MODEL_PARAM_DICT['FINAL_ACTIVATION']).to(device)

elif MODEL_PARAM_DICT['BOTTLENECK']:
    autoencoder_model = 0
    model = CFM_lib.ConditionalFlowModel_Bottleneck(input_dim=4 * dim + 1, output_dim=dim, hidden_dim=MODEL_PARAM_DICT['v_hidden_dim'], encoder_layers=MODEL_PARAM_DICT['encoder_layers'], decoder_layers=MODEL_PARAM_DICT['decoder_layers'], bottleneck=MODEL_PARAM_DICT['BOTTLENECK_SIZE'], ACTIVATION_TYPE=MODEL_PARAM_DICT['ACTIVATION_TYPE'], FINAL_ACTIVATION=MODEL_PARAM_DICT['FINAL_ACTIVATION'], IOB_METHOD=MODEL_PARAM_DICT['IOB_METHOD'], SPARSE_AUTOENCODER=MODEL_PARAM_DICT['SPARSE_AUTOENCODER'], ADD_INFO=MODEL_PARAM_DICT['MARGINAL_PREDICTOR']).to(device)
elif MODEL_PARAM_DICT['CAT_MODEL']:
    model = CFM_lib.CatNet(input_dim=4 * dim + 1, output_dim=dim, hidden_dim=MODEL_PARAM_DICT['v_hidden_dim'], n_layers=MODEL_PARAM_DICT['n_layers'], ACTIVATION_TYPE=MODEL_PARAM_DICT['ACTIVATION_TYPE'], FINAL_ACTIVATION=MODEL_PARAM_DICT['FINAL_ACTIVATION'], ADD_INFO=MODEL_PARAM_DICT['MARGINAL_PREDICTOR'], INCLUDE_OUTPUT_MASK=True, LAYER_NORM=MODEL_PARAM_DICT['LAYER_NORM']).to(device)
    autoencoder_model = 0
elif MODEL_PARAM_DICT['ADD_MODEL']:
    # PATCH 1: pass the GFTE architecture kwargs. .get() defaults keep older
    # (pre-GFTE) param dicts loadable.
    model = CFM_lib.AddNet(input_dim=4 * dim + 1, output_dim=dim,
                           hidden_dim=MODEL_PARAM_DICT['v_hidden_dim'],
                           n_layers=MODEL_PARAM_DICT['n_layers'],
                           ACTIVATION_TYPE=MODEL_PARAM_DICT['ACTIVATION_TYPE'],
                           FINAL_ACTIVATION=MODEL_PARAM_DICT['FINAL_ACTIVATION'],
                           ADD_INFO=MODEL_PARAM_DICT['MARGINAL_PREDICTOR'],
                           INCLUDE_OUTPUT_MASK=True,
                           LAYER_NORM=MODEL_PARAM_DICT['LAYER_NORM'],
                           GFTE=MODEL_PARAM_DICT.get('GFTE', False),
                           GFTE_embed_dim=MODEL_PARAM_DICT.get('GFTE_embed_dim', 64),
                           GFTE_scale=MODEL_PARAM_DICT.get('GFTE_scale', 16.0),
                           GFTE_learnable=MODEL_PARAM_DICT.get('GFTE_learnable', False)).to(device)
    autoencoder_model = 0
else:
    model = CFM_lib.ConditionalFlowModel_Flow_Only(input_dim=4 * dim + 1, output_dim=dim, hidden_dim=MODEL_PARAM_DICT['v_hidden_dim'], n_layers=MODEL_PARAM_DICT['n_layers'], ACTIVATION_TYPE=MODEL_PARAM_DICT['ACTIVATION_TYPE'], FINAL_ACTIVATION=MODEL_PARAM_DICT['FINAL_ACTIVATION'], ADD_INFO=MODEL_PARAM_DICT['MARGINAL_PREDICTOR'], INCLUDE_OUTPUT_MASK=True).to(device)
    autoencoder_model = 0

hdf5_filelist = np.loadtxt('YOUR PATH/hdf5_filelist.txt', dtype=str)


# ---------------------------------------------------------------------------
# Normalization helpers (PATCH 2/4/11).
# Conventions (shared with CFM_Eval_Gaia.py):
#   physical space : (N, dim_phys), raw Gaia units, l = one degree-valued column
#   model space    : (N, dim),      normalized,   l = (cos, sin) in columns 0,1
# ---------------------------------------------------------------------------

# physical (N, dim_phys) -> model space (N, dim)
def normalizer(samples, NORM_METHOD=MODEL_PARAM_DICT['NORM_METHOD']):
    """
    Applies the training normalization to PHYSICAL-unit data.
    Input has dim_phys columns; output has dim columns (l split under L_CUSTOM).
    """
    samples = np.asarray(samples)
    assert samples.shape[1] == dim_phys, (
        f"normalizer expects (N, dim_phys={dim_phys}) physical-unit input, got {samples.shape}. "
        f"A stale pure-sample cache from a different feature set is the usual cause.")
    norm_samples = torch.empty((samples.shape[0], dim), dtype=torch.float32)
    if NORM_METHOD == 'SIGMOID_Z_SCORE':
        raise NotImplementedError("SIGMOID_Z_SCORE deprecated")
    elif NORM_METHOD == 'MIN_MAX':
        for c in range(dim_phys):
            norm_samples[:, c] = torch.from_numpy(np.asarray((samples[:, c] - feature_dict[feature_names[c]]['min']) / (feature_dict[feature_names[c]]['max'] - feature_dict[feature_names[c]]['min']), dtype=np.float32))
    elif NORM_METHOD == 'CUSTOM':
        for c in range(dim_phys):
            if c == 0 and L_CUSTOM:
                cos_l, sin_l = feature_normer(samples[:, c], feature_names[c], feature_dict)
                norm_samples[:, 0] = torch.from_numpy(np.asarray(cos_l, dtype=np.float32))
                norm_samples[:, 1] = torch.from_numpy(np.asarray(sin_l, dtype=np.float32))
            elif L_CUSTOM:
                norm_samples[:, c + 1] = torch.from_numpy(np.asarray(feature_normer(samples[:, c], feature_names[c], feature_dict), dtype=np.float32))
            else:
                norm_samples[:, c] = torch.from_numpy(np.asarray(feature_normer(samples[:, c], feature_names[c], feature_dict), dtype=np.float32))
    elif NORM_METHOD == 'P_Z_SCORE':
        for c in range(dim_phys):
            norm_samples[:, c] = torch.from_numpy(np.asarray((samples[:, c] - feature_dict[feature_names[c]]['est_median']) / (feature_dict[feature_names[c]]['est_p97_5'] - feature_dict[feature_names[c]]['est_p2_5']), dtype=np.float32))
    return norm_samples


def inverse_sigmoid_stable(x):
    # This is equivalent to torch.log(x) - torch.log(1 - x)
    return torch.log(x) - torch.log1p(-x)


# model space (N, dim) -> physical (N, dim_phys)   [PATCH 4]
def denormalizer(samples, NORM_METHOD=MODEL_PARAM_DICT['NORM_METHOD']):
    """
    Inverts the training normalization. Input has dim (model-space) columns;
    output has dim_phys columns, with l collapsed back to degrees under
    L_CUSTOM. Matches denormalizer() in CFM_Eval_Gaia.py.
    """
    samples = np.asarray(samples)
    assert samples.shape[1] == dim, (
        f"denormalizer expects (N, dim={dim}) model-space input, got {samples.shape}.")
    denorm_samples = np.empty((samples.shape[0], dim_phys))
    if NORM_METHOD == 'SIGMOID_Z_SCORE':
        for c in range(dim_phys):
            denorm_samples[:, c] = (inverse_sigmoid_stable(torch.as_tensor(samples[:, c], dtype=torch.float64)).numpy() * feature_dict[feature_names[c]]['std']) + feature_dict[feature_names[c]]['mean']
    elif NORM_METHOD == 'MIN_MAX':
        for c in range(dim_phys):
            denorm_samples[:, c] = (samples[:, c] * (feature_dict[feature_names[c]]['max'] - feature_dict[feature_names[c]]['min'])) + feature_dict[feature_names[c]]['min']
    elif NORM_METHOD == 'CUSTOM':
        for c in range(dim_phys):
            if c == 0 and L_CUSTOM:
                denorm_samples[:, c] = feature_denormer((samples[:, 0], samples[:, 1]), feature_names[0], feature_dict)
            elif L_CUSTOM:
                denorm_samples[:, c] = feature_denormer(samples[:, c + 1], feature_names[c], feature_dict)
            else:
                denorm_samples[:, c] = feature_denormer(samples[:, c], feature_names[c], feature_dict)
    elif NORM_METHOD == 'P_Z_SCORE':
        for c in range(dim_phys):
            denorm_samples[:, c] = (samples[:, c] * (feature_dict[feature_names[c]]['est_p97_5'] - feature_dict[feature_names[c]]['est_p2_5'])) + feature_dict[feature_names[c]]['est_median']
    return denorm_samples


# physical mask (N, dim_phys) -> model-space mask (N, dim)
def expand_mask(mask_phys):
    """
    Expands a physical validity mask to model space: under L_CUSTOM, the l
    column is duplicated into model columns 0 and 1 (both represent l).
    Matches the mask expansion in CFM_Eval_Gaia._load_eval_set.
    """
    mask_phys = torch.as_tensor(np.asarray(mask_phys))
    mask_model = torch.empty((mask_phys.shape[0], dim), dtype=torch.float32)
    if L_CUSTOM:
        mask_model[:, 0] = (mask_phys[:, 0] > 0.5).float()
        mask_model[:, 1] = (mask_phys[:, 0] > 0.5).float()
        for c in range(1, dim_phys):
            mask_model[:, c + 1] = (mask_phys[:, c] > 0.5).float()
    else:
        mask_model[:, :] = (mask_phys > 0.5).float()
    return mask_model


# model-space mask (N, dim) -> physical validity (N, dim_phys) [bool]
def model_mask_to_phys(mask_model):
    """
    Under L_CUSTOM, l is valid iff BOTH model columns 0 and 1 are valid
    (they always agree in this pipeline). Matches _model_validity_to_phys
    in CFM_Eval_Gaia.py.
    """
    mask_model = np.asarray(mask_model)
    if not L_CUSTOM:
        return mask_model > 0.5
    valid = np.empty((mask_model.shape[0], dim_phys), dtype=bool)
    valid[:, 0] = (mask_model[:, 0] > 0.5) & (mask_model[:, 1] > 0.5)
    valid[:, 1:] = mask_model[:, 2:] > 0.5
    return valid


def normalize_pure_sample(x_load):
    """
    PATCH 11: convert a pure-sample tensor (N, 2, dim_phys), physical units,
    into the model-space x_test (N, 2, dim) used everywhere downstream.
    """
    assert x_load.ndim == 3 and x_load.shape[1] == 2, (
        f"Pure sample must have shape (N, 2, dim_phys); got {tuple(x_load.shape)}")
    assert x_load.shape[2] == dim_phys, (
        f"Pure sample has {x_load.shape[2]} feature columns but this model uses "
        f"{dim_phys} physical features {feature_names}. This is almost certainly "
        f"a stale cache from a different feature set (PATCH 8); rebuild it with "
        f"SAVE_TEST_SET=True, LOAD_TEST_SET=False.")
    x_test = torch.empty((x_load.shape[0], 2, dim))
    x_test[:, 0, :] = normalizer(x_load[:, 0, :])
    x_test[:, 1, :] = expand_mask(x_load[:, 1, :])
    return x_test


# ---------------------------------------------------------------------------
# Test-set construction (PATCHES 6, 7, 8, 11).
# Currently I am actually using the validation set for this. Don't use the
# test until you optimize things.
# On return, x_test is ALWAYS (N, 2, dim) in normalized model space.
# ---------------------------------------------------------------------------
def create_test_loader(hdf5_filelist, feature_dict, batch_size, MODEL_PARAM_DICT, feature_names):
    if LOAD_TEST_SET == False:
        if USE_PURE_SAMPLE:
            NORM_METHOD = None       # dataset emits raw physical units, dim_phys columns
            n_cols = dim_phys
        else:
            NORM_METHOD = MODEL_PARAM_DICT['NORM_METHOD']  # dataset emits model space, dim columns
            n_cols = dim
        test_ds = HDF5IterableDataset(
            READ_TYPE=MODEL_PARAM_DICT['READ_TYPE'],
            filelist=hdf5_filelist,
            split="test",
            set_type=MODEL_PARAM_DICT['SET_TYPE'],
            columns=feature_names,
            shard_columns=MODEL_PARAM_DICT['shard_columns'],
            batch_size=batch_size,
            shuffle_order_each_epoch=False,
            shuffle_sources_in_file=False,
            drop_last=False,
            normalization=feature_dict,
            norm_method=NORM_METHOD,
        )
        test_ds.set_epoch(0)
        test_loader = DataLoader(
            test_ds,
            num_workers=N_WORKERS,   # start with 4-16 depending on CPU cores
            pin_memory=True,
            drop_last=False,
        )

        # this will be slow
        start_loop = time.time()
        ct = 0
        total_sample_length = 0
        total_test_length = 0
        for x_data, initial_mask in test_loader:
            # x_data/initial_mask --> (1, batch_size, n_cols)
            x_data_len = x_data.size(dim=1)
            fraction_len = int(TEST_FRACTION * x_data_len)
            print(x_data_len, fraction_len, x_data.shape)
            index = np.arange(x_data_len, dtype=int)
            rng.shuffle(index)
            if ct == 0:
                x_test = torch.empty((fraction_len, 2, n_cols))
                print(x_test[:, 0, :].shape, x_data[0, index[:fraction_len], :].shape)
                x_test[:, 0, :] = x_data[0, index[:fraction_len], :]
                x_test[:, 1, :] = initial_mask[0, index[:fraction_len], :]
            else:
                x_temp = torch.empty((fraction_len, 2, n_cols))
                x_temp[:, 0, :] = x_data[0, index[:fraction_len], :]
                x_temp[:, 1, :] = initial_mask[0, index[:fraction_len], :]
                x_test = torch.cat([x_test, x_temp], dim=0)
            ct += 1
            total_test_length += x_data_len
            total_sample_length += fraction_len
        runtime = (time.time() - start_loop) / 60
        print(f"Finished constructing sample of test set. Sample fraction set to {TEST_FRACTION}. Sample length is {total_sample_length}, total test set length is {total_test_length}, thus the true fraction is {total_sample_length/total_test_length}. Run time is {round(runtime,2)} minutes.")
        del test_loader

        # PATCH 6: the raw physical-unit subset is the Pure_Sample file; the
        # normalized subset is saved under the model name. (Names were swapped.)
        if SAVE_TEST_SET and USE_PURE_SAMPLE:
            torch.save(x_test, PURE_SAMPLE_PATH)
            print(f"Saved Pure_Sample (physical units, {dim_phys} cols) to {PURE_SAMPLE_PATH}")
        elif SAVE_TEST_SET:
            torch.save(x_test, NORMED_SAMPLE_PATH)
            print(f"Saved normalized test set ({dim} cols) to {NORMED_SAMPLE_PATH}")

        # PATCH 11: bring a freshly-built pure sample into model space so the
        # SAME run can also sample from the model (previously the raw tensor
        # leaked downstream and either crashed or fed unnormalized data).
        if USE_PURE_SAMPLE:
            x_test = normalize_pure_sample(x_test)
    else:
        if USE_PURE_SAMPLE:
            print(f"Loading Pure_Sample from {PURE_SAMPLE_PATH}")
            x_load = torch.load(PURE_SAMPLE_PATH, map_location="cpu")
            x_test = normalize_pure_sample(x_load)
        else:
            print(f"Loading normalized test set from {NORMED_SAMPLE_PATH}")
            x_test = torch.load(NORMED_SAMPLE_PATH, map_location="cpu")
            assert x_test.shape[2] == dim, (
                f"Loaded normalized test set has {x_test.shape[2]} columns; model dim is {dim}.")

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test), batch_size=batch_size, shuffle=False, num_workers=N_WORKERS)

    return x_test, test_loader


# ---------------------------------------------------------------------------
# RK4 sampler (unchanged: the forward-call convention already matched training).
# ---------------------------------------------------------------------------
def sample_properties_distribution_rk4(model, x_data, input_mask, output_mask, N_samples, num_steps=50, autoencoder_model=autoencoder_model, MODEL_PARAM_DICT=MODEL_PARAM_DICT):
    """
    Multi-trajectory sampler using RK4 integration for better numerical accuracy.
    Returns NORMALIZED (model-space) samples of shape (num_objects, N_samples, dim).
    """
    model.eval()
    with torch.no_grad():
        x_data = x_data.detach().clone().to(device=device)
        input_mask = input_mask.detach().clone().to(device=device)
        output_mask = output_mask.detach().clone().to(device=device)

        # starts as x0
        # Expand to full batch
        x_data_dim0 = x_data.shape[0]
        x_data_dim1 = x_data.shape[1]

        if MODEL_PARAM_DICT['LATENT_FLOW']:
            x = torch.randn(N_samples * x_data_dim0, MODEL_PARAM_DICT['BOTTLENECK_SIZE'], device=device)
        else:
            x = torch.randn(N_samples * x_data_dim0, x_data_dim1, device=device)

        x_data_input = x_data.repeat_interleave(N_samples, dim=0)
        mask_input = input_mask.repeat_interleave(N_samples, dim=0)
        mask_output = output_mask.repeat_interleave(N_samples, dim=0)

        del x_data, input_mask, output_mask

        dt = 1.0 / num_steps
        if MODEL_PARAM_DICT['LATENT_FLOW']:
            # The CFM model exists in the latent space, so first determine the embedding
            input_embedding = autoencoder_model.Encoder(torch.cat([mask_input * x_data_input, mask_input], dim=1))
            for i in range(num_steps):
                t = i * dt

                # RK4 coefficients
                t_tensor = torch.full((N_samples * x_data_dim0, 1), t, device=device)
                k1 = model.forward(torch.cat([t_tensor, x, input_embedding, mask_output], dim=1))

                t_tensor = torch.full((N_samples * x_data_dim0, 1), t + dt / 2, device=device)
                k2 = model.forward(torch.cat([t_tensor, (x + dt * k1 / 2), input_embedding, mask_output], dim=1))

                k3 = model.forward(torch.cat([t_tensor, (x + dt * k2 / 2), input_embedding, mask_output], dim=1))

                t_tensor = torch.full((N_samples * x_data_dim0, 1), t + dt, device=device)
                k4 = model.forward(torch.cat([t_tensor, (x + dt * k3), input_embedding, mask_output], dim=1))

                # RK4 update
                x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # to return to feature space, use the decoder.
            x = autoencoder_model.Decoder(x)

        else:
            for i in range(num_steps):
                t = i * dt

                # RK4 coefficients
                t_tensor = torch.full((N_samples * x_data_dim0, 1), t, device=device)
                k1 = model.forward(t_tensor, mask_output * x, mask_output, mask_input * x_data_input, mask_input)

                t_tensor = torch.full((N_samples * x_data_dim0, 1), t + dt / 2, device=device)
                k2 = model.forward(t_tensor, mask_output * (x + dt * k1 / 2), mask_output, mask_input * x_data_input, mask_input)

                k3 = model.forward(t_tensor, mask_output * (x + dt * k2 / 2), mask_output, mask_input * x_data_input, mask_input)

                t_tensor = torch.full((N_samples * x_data_dim0, 1), t + dt, device=device)
                k4 = model.forward(t_tensor, mask_output * (x + dt * k3), mask_output, mask_input * x_data_input, mask_input)

                # RK4 update
                x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # leave as is: samples remain in normalized model space
        x_array = x.cpu().detach()
        x_out = x_array.view(x_data_dim0, N_samples, x_data_dim1).numpy()

        return x_out  # shape (num_objects, num_samples, property_dim)


def conditional_sampler(loader, model, conditional_feature_indices, device, num_samples, TEST_OUTPUT_MASKS):

    batch_counter = 0
    conditional_samples = np.empty((num_stars, num_samples, dim))
    for batch in loader:
        x_data = batch[0][:, 0, :].to(device)
        initial_mask = batch[0][:, 1, :].int().to(device)
        batch_dim = x_data.shape[0]
        in_cond_mask = torch.zeros((x_data.shape[0], dim))
        in_cond_mask[:, conditional_feature_indices] = torch.ones((x_data.shape[0], len(conditional_feature_indices)))
        input_mask = initial_mask * in_cond_mask.to(device)
        if TEST_OUTPUT_MASKS:
            output_mask = initial_mask.clone()
        else:
            output_mask = torch.ones((x_data.shape[0], dim)).to(device)
        # Delete some stuff to save space
        del initial_mask, in_cond_mask

        conditional_samples[batch_counter:batch_counter + batch_dim] = sample_properties_distribution_rk4(model, x_data, input_mask, output_mask, num_samples, num_steps=RK4_STEPS)

        batch_counter += batch_dim

    return conditional_samples


def feature_selector(feature, feature_names, feature_list, feature_index_list):
    f_ind = np.where(feature_names == feature)[0]
    if feature not in feature_list and f_ind > feature_index_list[-1]:
        feature_index_list += [np.where(feature_names == feature)[0][0]]
        feature_list += [feature]
    return feature_list, feature_index_list


'''
Given a list, this will produce all possible permutations of that list, including the original list, and optionally the empty set.
'''
def permute(set_list, perm_list, og_list, EMPTY=False):
    for element in og_list:
        if element not in set_list:
            temp_list = set_list.copy()
            temp_list += [element]
            if sorted(temp_list) in perm_list:
                continue
            else:
                perm_list += [temp_list]
                permute(temp_list, perm_list, og_list)
        else:
            continue
    if EMPTY == True:
        perm_list += [[]]
    return perm_list


'''
feature_permutation_indices: This tells you the indices of the features that the model was conditioned on
I will set the indices of features not included in feature_permutation_indices[j] as the indices that I investigate
That is if o_i ~ feature_permutation_indices[j], t_i !~ feature_permutation_indices[j].
I will work through all permutations of t_i, masking the data points that do not have the relevant components of t_i as according to their initial mask.
In each case, I will calculate the wasserstein metric and record it.

PATCH 9/10: unit conventions.
  * Sinkhorn/Wasserstein distances are computed in NORMALIZED model space for
    BOTH the samples and the truth (previously the truth was denormalized to
    physical units while the samples stayed normalized, so the distances were
    meaningless; normalized space also keeps all features on comparable
    scales so no single physical unit dominates the multivariate metric).
  * The median-imputation errors are computed in PHYSICAL units by
    denormalizing both the per-source median sample and the truth, with a
    wrap-aware (+-180 deg) difference for l. They are reported per PHYSICAL
    feature, i.e. median_error has dim_phys columns, ordered as feature_names.
  * Validity selection uses initial_mask > 0.5 (1 = observed) everywhere.
'''
def cond_dist_and_impute_eval(loss, conditional_samples, x_test, num_samples, conditional_feature_indices, UNCOND=False):
    initial_mask = x_test[:, 1, :].cpu().detach().numpy()
    # NORMALIZED truth for the distributional (Wasserstein) comparisons
    x_data_norm = x_test[:, 0, :].cpu().detach().numpy()

    # index of all MODEL-SPACE features
    all_features = np.arange(dim, dtype=int)

    if UNCOND == False:
        # Determine which features are not included in the conditional features set (target features)
        target_features = np.setdiff1d(all_features, conditional_feature_indices, assume_unique=True)
    else:
        target_features = all_features.copy()

    # find all permutations of the target features
    target_feature_permutations = permute([], [], list(target_features))

    # number of permutations
    perm_n = len(target_feature_permutations)
    print("# of Permutations", perm_n)
    # array for storing distribution distances
    wasserstein_metrics = np.empty((perm_n, num_samples))

    target_perm_ind = 0
    # for every permutation of the target features, remove invalid objects, then compare sampled version of data to real data.
    for target_feature_permutation in target_feature_permutations:
        print(target_feature_permutation)
        # for every target feature in the permutation set, determine the indices for each object with an invalid (nan) observation of that target feature
        perm_ct = 0
        for target_feature in target_feature_permutation:
            if perm_ct == 0:
                valid_obs = np.where(initial_mask[:, target_feature] > 0.5)[0]
            else:
                valid_obs = np.intersect1d(valid_obs, np.where(initial_mask[:, target_feature] > 0.5)[0])
            perm_ct += 1

        # remove non-target features (NORMALIZED truth; PATCH 9)
        valid_x_data = x_data_norm[:, target_feature_permutation]
        # remove objects with invalid observations of target features
        valid_x_data = torch.from_numpy(valid_x_data[valid_obs])

        # remove non-target features from sample data (already normalized)
        valid_conditional_samples = conditional_samples[:, :, target_feature_permutation]
        # remove objects from sample data with invalid observations of target features
        valid_conditional_samples = torch.from_numpy(valid_conditional_samples[valid_obs])
        # for each sample, compare the full population of conditionally sampled objects to the observed data
        for j in range(num_samples):
            wasserstein_metrics[target_perm_ind, j] = loss(valid_conditional_samples[:, j, :].to(device=device, dtype=torch.float32).contiguous(), valid_x_data.to(device=device, dtype=torch.float32).contiguous())
        target_perm_ind += 1

    if UNCOND == False:
        # PATCH 9/10: median-imputation error in PHYSICAL units, per physical feature.
        # Calculates the median of the samples for a given object, denormalizes it
        # and the truth, and takes the difference. Only entries with a valid
        # observation (initial_mask > 0.5) are filled; the rest stay NaN.
        median_error = np.empty((x_test.shape[0], dim_phys)) * np.nan
        valid_phys = model_mask_to_phys(initial_mask)                       # (N, dim_phys) bool
        truth_phys = denormalizer(x_data_norm)                              # (N, dim_phys)
        median_sample_phys = denormalizer(np.median(conditional_samples, axis=1))  # (N, dim_phys)
        for k in range(dim_phys):
            observed = np.where(valid_phys[:, k])[0]                        # PATCH 10: > 0.5, not < 0.5
            err = median_sample_phys[observed, k] - truth_phys[observed, k]
            if feature_names[k] == 'l':
                # wrap-aware difference on the circle (period 360 deg)
                err = (err + 180.0) % 360.0 - 180.0
            median_error[observed, k] = err
            del observed

        return wasserstein_metrics, target_feature_permutations, median_error
    else:
        return wasserstein_metrics, target_feature_permutations


# ---------------------------------------------------------------------------
# PATCH 5: helpers for feature-name -> model-index translation and for
# converting MASK_SETS_INDICES between model space and physical space.
# ---------------------------------------------------------------------------
def phys_names_to_model_indices(names):
    """
    Physical feature names -> model-space column indices.
    Under L_CUSTOM, 'l' expands to [0, 1]; every other feature maps to
    (1 + its index in feature_names). Matches _phys_to_model_indices in
    CFM_Eval_Gaia.py.
    """
    out = []
    for name in names:
        assert name in feature_names, (
            f"Feature '{name}' requested in a conditional permutation is not in "
            f"the model's feature list {feature_names}.")
        if L_CUSTOM:
            if name == 'l':
                out += [0, 1]
            else:
                out += [1 + feature_names.index(name)]
        else:
            out += [feature_names.index(name)]
    return out


def mask_sets_to_phys(mask_sets_indices):
    """
    Convert MASK_SETS_INDICES to PHYSICAL feature indices for DROP_ONE_SETS.
    If MASK_SETS_INDICES_ARE_MODEL_SPACE, model indices 0/1 both map to
    physical index 0 (l) under L_CUSTOM and every other index shifts down by 1.
    """
    phys_sets = []
    for set_j in mask_sets_indices:
        s = set()
        for idx in set_j:
            if MASK_SETS_INDICES_ARE_MODEL_SPACE and L_CUSTOM:
                s.add(0 if idx <= 1 else idx - 1)
            else:
                s.add(idx)
        phys_sets.append(tuple(sorted(s)))
    return phys_sets


x_test, test_loader = create_test_loader(hdf5_filelist, feature_dict, s_batch_size, MODEL_PARAM_DICT, feature_names)


print('##################################################')
print('##################################################')
print(MODEL_NAME)
print('##################################################')
print('##################################################')

if USE_MODEL:
    state = torch.load('YOUR PATH/saved_models/' + LOAD_NAME, map_location=device)
    model.load_state_dict(state)
else:
    assert UNCONDITIONAL == False
    assert CONDITIONAL == False
    assert COMPARE_WITH_TEST == False

print("Best Model Loaded. Begin Sampling")

sample_metrics = {}
sample_metrics['feat_names'] = feature_names
sample_metrics['dim_phys'] = dim_phys
sample_metrics['dim_model'] = dim
sample_metrics['L_CUSTOM'] = L_CUSTOM
# PATCH 9: record the unit conventions with the metrics so downstream plots
# can't misinterpret them.
sample_metrics['wasserstein_space'] = 'normalized_model_space'
sample_metrics['median_error_space'] = 'physical_units_per_feature_names'

print(feature_names)

if UNCONDITIONAL:
    print("Unconditional Sampling")  # PATCH (minor): said "Conditional Sampling"
    if COMPARE_WITH_TEST == False and CONDITIONAL == False:
        num_stars = 1000000
    else:
        num_stars = x_test.size(dim=0)
    # remove all features
    # Number of unconditional samples
    # Each unconditional sample will have units of (num_stars, N_uncond_sam, N_features)
    # I permute possible realizations of unconditional sample feature sets (i.e., only RA, only DEC, etc)
    # I do so using only one set of the samples, but by calculating many different wasserstein distances
    # A potential issue here is that I am implicitly saying that unconditional_sample[i] corresponds to star[i] in x_data. That isn't actually true, but I am not sure it really matters - it kind of matters but I have an idea around this.
    # I could get around this by drawing more samples, but I am worried about memory constraints
    cond_mask = torch.zeros((num_stars, dim)).to(device)
    sample_time = time.time()
    if TEST_OUTPUT_MASKS and COMPARE_WITH_TEST:
        # x_test is guaranteed normalized model space with a dim-wide mask (PATCH 11)
        unconditional_samples = sample_properties_distribution_rk4(model, cond_mask, cond_mask, x_test[:, 1, :], N_uncond_sam, num_steps=RK4_STEPS)
        print(f"Generated unconditional samples in {time.time() - sample_time} seconds.")
    else:
        unconditional_samples = sample_properties_distribution_rk4(model, cond_mask, cond_mask, cond_mask + 1, N_uncond_sam, num_steps=RK4_STEPS)
        print(f"Generated unconditional samples in {time.time() - sample_time} seconds.")
    if COMPARE_WITH_TEST and METRICS:
        metric_time = time.time()
        sample_metrics['uncond_w_metric'], sample_metrics['uncond_target_perms'] = cond_dist_and_impute_eval(sinkhorn_loss, unconditional_samples, x_test, N_uncond_sam, 0, UNCOND=True)
        print(f"Generated unconditional samples metrics in {time.time() - metric_time} seconds.")
    if SAVE_SAMPLES:
        torch.save(unconditional_samples, 'YOUR PATH/model_samples/' + SAMPLE_NAME + MODEL_NAME + '_uncond_sam_' + timestamp + '.pt')

if CONDITIONAL:
    num_stars = x_test.size(dim=0)

    # bespoke selections
    # PATCH 5: pmra/pmdec -> pml/pmb (the model's proper-motion features are now
    # galactic). The list is filtered against the actual feature set below, so a
    # missing feature produces a loud warning instead of an IndexError.
    feature_permutation_names = [['l', 'b'],
            ['pml', 'pmb'],
            ['l', 'b', 'parallax'],
            ['l', 'b', 'parallax', 'pml', 'pmb'],
            ['phot_bp_mean_mag', 'phot_rp_mean_mag'],
            ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
            ['l', 'b', 'phot_g_mean_mag'],
            ['l', 'b', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
            ['l', 'b', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
            ['parallax', 'phot_g_mean_mag'],
            ['parallax', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
            ['parallax', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
            ['l', 'b', 'parallax', 'phot_g_mean_mag'],
            ['l', 'b', 'parallax', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
            ['l', 'b', 'parallax', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
            ['pml', 'pmb', 'radial_velocity'],
            ['l', 'b', 'pml', 'pmb', 'radial_velocity'],
            ['l', 'b', 'pml', 'pmb'],
            ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'pml', 'pmb'],
            ['parallax', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'pml', 'pmb'],
            ['parallax', 'pml', 'pmb'],
            ['l', 'b', 'parallax', 'pml', 'pmb'],
            ['l', 'b', 'parallax', 'pml', 'pmb', 'radial_velocity'],
            ['l', 'b', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'pml', 'pmb'],
            ['l', 'b', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'pml', 'pmb', 'radial_velocity']
            ]
    kept_permutations = []
    for perm in feature_permutation_names:
        missing = [f for f in perm if f not in feature_names]
        if missing:
            print(f"WARNING: dropping conditional permutation {perm}: features {missing} not in model feature set {feature_names}.")
        else:
            kept_permutations += [perm]
    feature_permutation_names = kept_permutations

    if DROP_ONE_SETS == True:
        # PATCH 5: DROP_ONE_SETS operates on PHYSICAL feature indices, so the
        # MASK_SETS_INDICES from the param dict (model-space under L_CUSTOM)
        # are converted first.
        feature_index = np.arange(len(feature_names), dtype=int)
        if MODEL_PARAM_DICT['MASK_SETS'] == True:
            phys_mask_sets = mask_sets_to_phys(MODEL_PARAM_DICT['MASK_SETS_INDICES'])
            print(f"DROP_ONE_SETS: physical mask sets = {phys_mask_sets} "
                  f"({[[feature_names[i] for i in s] for s in phys_mask_sets]})")
        for i in range(len(feature_names)):
            # create feature sets with one feature missing
            if MODEL_PARAM_DICT['MASK_SETS'] == True:
                # MASK_SET_CATCH is used to determine if the feature being dropped is included in a masked set or if it can be dropped individually
                MASK_SET_CATCH = False
                for set_j in phys_mask_sets:
                    if i in set_j:
                        MASK_SET_CATCH = True
                    # Only save the drop one (in this case drop one set instead of drop one feature) version of the features once
                    if set_j[0] == i:
                        subset_index = np.setdiff1d(feature_index, np.array(set_j))
                        feature_permutation_names += [list(np.array(feature_names)[subset_index])]
                if MASK_SET_CATCH == False:
                    subset_index = np.setdiff1d(feature_index, np.array([i]))
                    feature_permutation_names += [list(np.array(feature_names)[subset_index])]
            else:
                subset_index = np.setdiff1d(feature_index, np.array([i]))
                feature_permutation_names += [list(np.array(feature_names)[subset_index])]

    # PATCH 12 (v2): this block was mis-indented (IndentationError) and
    # appended bare strings, which phys_names_to_model_indices would have
    # iterated character-by-character. It now appends [feature] lists.
    if KEEP_ONE_SETS == True:
        for feature_name in feature_names:
            feature_permutation_names += [[feature_name]]

    print(feature_permutation_names)
    # PATCH 5: single translation path for both L_CUSTOM and non-L_CUSTOM,
    # via phys_names_to_model_indices (mirrors CFM_Eval_Gaia).
    feature_permutation_indices = []
    tester = []
    for feature_permutation in feature_permutation_names:
        feature_permutation_indices_set = phys_names_to_model_indices(feature_permutation)
        feature_permutation_indices += [feature_permutation_indices_set]
        tester += [[str(ind) for ind in feature_permutation_indices_set]]
    print(tester)
    print("Conditional Samples Analysis")
    wm_list = []
    tfp_list = []
    mei_list = []
    start = time.time()
    for j in range(len(feature_permutation_indices)):
        specific_loop_start = time.time()
        print(feature_permutation_indices[j], feature_permutation_names[j])
        conditional_samples = conditional_sampler(test_loader, model, feature_permutation_indices[j], device, num_samples, TEST_OUTPUT_MASKS)
        if COMPARE_WITH_TEST and METRICS:
            wasserstein_metrics, target_feature_permutations, median_error_imputation = cond_dist_and_impute_eval(sinkhorn_loss, conditional_samples, x_test, num_samples, feature_permutation_indices[j])
            wm_list += [wasserstein_metrics]
            tfp_list += [target_feature_permutations]
            mei_list += [median_error_imputation]
            sample_metrics['cond_w_metric_list'] = wm_list
            sample_metrics['target_feat_perm_list'] = tfp_list
            sample_metrics['median_error_list'] = mei_list
            sample_metrics['cond_feat_perm_names'] = feature_permutation_names
            sample_metrics['cond_feat_inds'] = feature_permutation_indices

        loop_time = time.time()
        if SAVE_SAMPLES:
            context_perm_name = str(j)
            torch.save(conditional_samples, 'YOUR PATH/model_samples/' + SAMPLE_NAME + MODEL_NAME + '_cond_sam_' + context_perm_name + timestamp + '.pt')

        print("Completed Loop:", j, " Duration (s):", loop_time - start, "Specific Loop Time:", loop_time - specific_loop_start, "Average Loop Time (s):", (loop_time - start) / (j + 1))

print(SAMPLE_NAME + MODEL_NAME, timestamp)
if COMPARE_WITH_TEST == True:
    np.save('YOUR PATH/jsoltis/' + MODEL_NAME + '_sample_metrics_' + SAMPLE_NAME + timestamp + '.npy', sample_metrics)
