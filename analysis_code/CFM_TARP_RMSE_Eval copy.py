"""
CFM_TARP_RMSE_Eval.py
=====================
Stand-alone evaluation of the Gaia conditional flow matching (CFM) model with
two metrics, and nothing else:

  1. TARP (Lemos et al. 2023) expected-coverage curves, for every conditioning
     pattern and every applicable Gaia MEASUREMENT CLASS (see CLASSES below).
  2. RMSE of every predicted feature, for every conditioning pattern, next to
     the RMSE of a baseline that always guesses the mean of the true values.
     Both are computed in physical units AND in normalized model space.

The script only writes data (HDF5 + CSV). All plotting is left to a notebook.

---------------------------------------------------------------------------
Terminology
---------------------------------------------------------------------------
  x      : a source's CONDITIONING features (what the model is given).
  theta  : a source's PREDICTED features (what the model must generate).
  pattern: which features are conditioned on. 47 patterns in total.
  class  : a Gaia measurement class, i.e. the exact set of features that is
           measured for a source. A source belongs to class (b) only if it
           has l, b, G, BP and RP measured AND nothing else.

---------------------------------------------------------------------------
Method summary
---------------------------------------------------------------------------
Source selection is STRICT: a source is used for a pattern only if every
conditioning feature is measured. For each (dataset, pattern) up to
N_SOURCES sources are drawn at random from each of these pools:

  general : conditioning features measured              -> RMSE (non-RV)
  rv      : conditioning features and RV measured       -> RMSE (RV)
  class_* : measured set == the class, which must
            contain the conditioning features           -> TARP

For every pooled source the model draws N_DRAWS posterior samples with RK4.
  input mask  = exactly the conditioning features
  output mask = the features measured for that source

TARP, per (pattern, class):
  truth vector   = class features minus conditioning features, in
                   normalized model space (l is two columns: cos l, sin l)
  reference pts  = theta_r ~ U(-1, 1) independently in every dimension, one
                   per source.  NOTE: this generator does not depend on x,
                   so TARP here tests calibration but cannot distinguish a
                   posterior from the prior; the RMSE-vs-baseline comparison
                   covers that.
  n_closer_i     = number of draws closer (L2) to theta_r than the truth
                   is; f_i = n_closer_i / N_DRAWS (Algorithm 2 of the paper)
  ECP            = fraction of sources with n_closer_i < threshold, for
                   threshold = 0..N_DRAWS+1, plotted at credibility
                   threshold/(N_DRAWS+1) (exact for a calibrated model)

RMSE is computed two ways.
  Class RMSE, per (pattern, class): on the SAME class pool, sources and
  draws as that TARP curve, for every feature in the TARP truth vector.
  Stored under rmse_class/<class>/ and rmse_class_table/<class>/.

  Pattern RMSE, per (pattern, predicted feature): on pool sources with that
  feature measured (the RV pool for radial_velocity, the general pool
  otherwise). Stored under rmse/ and rmse_table/. Definitions for both:
  physical   : prediction = per-source posterior mean (circular mean for l,
               errors wrapped to +-180 deg); baseline = mean of the true
               values over the same sources (circular for l)
  normalized : the same, per model column; cos l and sin l are treated as two
               ordinary (non-circular) columns

Curves / RMSE values with fewer than MIN_SOURCES sources are not computed;
their status and source count are still recorded.

---------------------------------------------------------------------------
Output: one HDF5 file per dataset, plus two RMSE CSVs per dataset
---------------------------------------------------------------------------
List-valued attributes are stored as JSON strings: json.loads(attrs[key]).

<RUN_TAG>_TARP_RMSE_<dataset>_<timestamp>.h5
  attrs: model_name, checkpoint, feature_names, model_columns, classes,
         pattern_names, n_draws, rk4_steps, n_sources_target, min_sources,
         seed, dataset, pure_sample, aux_file, partition_codes,
         n_sources_in_dataset, reference_rule, tarp_space, tarp_distance,
         timestamp, status ('running' -> 'complete')

  /patterns/<pattern>/
      attrs: cond_features, pred_features, aliases, pattern_index
      pools/<pool>/            attrs: n_eligible, n_used, status
                               rows  : (n_used,) row indices into the
                                       dataset's Pure_Sample (and Aux file)
                               draws : (n_used, N_DRAWS, 10)  [only if
                                       SAVE_SAMPLES]
      tarp/<class>/            attrs: class_features, truth_features,
                                      truth_model_columns, n_sources,
                                      n_draws, status
                               f                : (n,) per-source fractions
                               ecp              : (N_DRAWS+2,)
                               credibility      : (N_DRAWS+2,)
                               reference_points : (n, d)
      rmse_class/<class>/      attrs: scored_features, status
                               same eight arrays as rmse/ below, computed
                               on the class pool; NaN outside the truth
                               vector and wherever TARP status != 'ok'
      rmse/                    rmse_phys_model, rmse_phys_baseline,
                               truth_mean_phys, n_phys          (9,)
                               rmse_norm_model, rmse_norm_baseline,
                               truth_mean_norm, n_norm          (10,)
                               NaN for conditioned features and for
                               features with fewer than MIN_SOURCES sources.

  /rmse_table/   the same eight RMSE arrays stacked over patterns
                 (47 x 9 and 47 x 10), attrs: pattern_names, feature_names,
                 model_columns
  /rmse_class_table/<class>/   the class-pool arrays stacked over patterns,
                 same shapes, attrs as above plus class_features

<RUN_TAG>_RMSE_phys_<dataset>_<timestamp>.csv
<RUN_TAG>_RMSE_norm_<dataset>_<timestamp>.csv
<RUN_TAG>_RMSE_phys_<dataset>_class_<class>_<timestamp>.csv   (one per class)

Status values:
  TARP : ok | skipped_lt_min | empty | not_applicable
         empty          = every class feature is conditioned on
         not_applicable = the pattern conditions on a feature outside the
                          class, so no source of that class can satisfy it
  pool : ok | short (fewer than N_SOURCES available) | skipped_lt_min

Usage
-----
    python CFM_TARP_RMSE_Eval.py                                # DATASETS_TO_RUN
    python CFM_TARP_RMSE_Eval.py --datasets omega_cen_all       # one dataset
    python CFM_TARP_RMSE_Eval.py --n-sources 60 --n-draws 20    # quick smoke

Every constant in the CONFIGURATION block has a matching command-line option
(--n-sources, --checkpoint-path, --out-dir, ...); the constant is the default.
The submission scripts tarp_rmse_smoke.sh and tarp_rmse_full.sh pass them
explicitly so the two runs can be kept in sync from one EDIT BLOCK.

The test Pure_Sample can be named directly with --test-pure-sample, or built
from the sampler's convention with --pure-sample-dir, --test-fraction and
--load-number (the same three values the sampling and eval scripts use).
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch


# ###########################################################################
# ###########################################################################
#                           CONFIGURATION
#       All paths and settings live here.  Nothing below needs editing.
# ###########################################################################
# ###########################################################################

# ---- Where model_library_032526.py and gaia_feature_normalizer_020926.py live
LIBRARY_DIR = 'ENTER YOUR PATH'

# ---- Model (copied from CFM_Sampling_hdf5_070726_patched_v2.py) -----------
MODEL_NAME = ('POS_PM_mask_set_setA_CUSTOM_2026-07-10_15-01-16_mask_train_mask_val_'
              'MSE_vHD512_SIGMA1_0e-08_GELU_lr1_0e-04_wd0_0e+00_layers8_MP85_'
              'ADD_SET_VALACUMVAL_10')
CHECKPOINT_PATH = ('ENTER YOUR PATH'
                   + MODEL_NAME + '_ema.pth')
PARAM_DICT_PATH = ('ENTER YOUR PATH'
                   + MODEL_NAME + '_model_param_dict.npy')

# ---- Normalization statistics (same files as training / sampling) --------
STATS_PATH = 'ENTER YOUR PATH/P_Z_Norm_Dict_Sample_Ests_062326.npy'
BOOL_STR_STATS_PATH = 'ENTER YOUR PATH/training_stats_bool_string_features.json'

# ---- Data -----------------------------------------------------------------
# NOTE: despite 'val_fraction' in its name, this file is a 1% random draw
# from the subset-A TEST partition (create_test_loader uses split="test").
TEST_PURE_SAMPLE_PATH = ('ENTER YOUR PATH/'
                         'Pure_Sample_9feat_771f68a4_val_fraction_1pct_10.pt')

# Soltis et al. (2021) omega Cen members, built by GC_Sample_Creator_070726.py.
# The Aux file is row-aligned with the Pure_Sample and carries partition_code.
OMEGA_CEN_PURE_SAMPLE_PATH = ('ENTER YOUR PATH/true_samples/'
                              'Pure_Sample_9feat_771f68a4_Omega_Centauri_eDR3_data_'
                              '010521__1__2026-07-08_06-18.pt')
OMEGA_CEN_AUX_PATH = ('ENTER YOUR PATH/true_samples/'
                      'Aux_Omega_Centauri_eDR3_data_010521__1__9feat_771f68a4_'
                      '2026-07-08_06-18.npy')

# ---- Output ---------------------------------------------------------------
OUT_DIR = 'ENTER YOUR PATH/tarp_rmse_results'
RUN_TAG = 'POS_PM_setA_0710_MP85'      # short tag used in output filenames

# ---- Datasets -------------------------------------------------------------
# partition_codes: None = every source; otherwise keep only these Aux
# partition codes (0 training A, 1 validation A, 2 test A, 3 training B,
# 4 validation B, 5 test B, 6 validation M, 7 test M).
DATASETS = {
    'test':            dict(pure_sample=TEST_PURE_SAMPLE_PATH, aux=None,
                            partition_codes=None),
    'omega_cen_all':   dict(pure_sample=OMEGA_CEN_PURE_SAMPLE_PATH,
                            aux=OMEGA_CEN_AUX_PATH, partition_codes=None),
    'omega_cen_not_A': dict(pure_sample=OMEGA_CEN_PURE_SAMPLE_PATH,
                            aux=OMEGA_CEN_AUX_PATH,
                            partition_codes=[2, 3, 4, 5, 6, 7]),
}
DATASETS_TO_RUN = ['test', 'omega_cen_all', 'omega_cen_not_A']

# ---- Settings -------------------------------------------------------------
N_SOURCES = 1000       # sources per pool (all eligible sources if fewer)
N_DRAWS = 100          # posterior draws per source
RK4_STEPS = 50         # RK4 integration steps
MIN_SOURCES = 50       # skip any TARP curve / RMSE value with fewer sources
BATCH_SOURCES = 1000   # sources per forward pass (x N_DRAWS rows on the GPU)
SEED = 20260911
SAVE_SAMPLES = False   # also store raw normalized draws (~4 MB per pool)

# ###########################################################################
#                        END OF CONFIGURATION
# ###########################################################################


sys.path.insert(0, LIBRARY_DIR)

# ---------------------------------------------------------------------------
# Fixed feature bookkeeping.  The script asserts the checkpoint matches.
# ---------------------------------------------------------------------------
FEATURES = ['l', 'b', 'parallax', 'pml', 'pmb', 'phot_g_mean_mag',
            'phot_bp_mean_mag', 'phot_rp_mean_mag', 'radial_velocity']


def feature_tag(feature_names=None):
    """
    Feature-set tag used in Pure_Sample filenames, identical to PATCH 8 of the
    sampling script and to _feature_tag() in the eval script:
        f"{n}feat_" + md5('_'.join(feature_names)).hexdigest()[:8]
    For the 9 features above this is '9feat_771f68a4'.
    """
    names = list(FEATURES if feature_names is None else feature_names)
    return f"{len(names)}feat_" + hashlib.md5('_'.join(names).encode()).hexdigest()[:8]


def fraction_tag(test_fraction):
    """Test-fraction tag, as in PATCH 7 of the sampler: 0.01 -> '1pct'."""
    return f"{100 * test_fraction:g}".replace('.', 'p') + 'pct'


def default_pure_sample_path(pure_sample_dir, test_fraction, load_number,
                             feature_names=None):
    """Canonical test Pure_Sample path, matching the sampling script exactly."""
    return Path(pure_sample_dir) / (
        f"Pure_Sample_{feature_tag(feature_names)}"
        f"_val_fraction_{fraction_tag(test_fraction)}_{load_number}.pt")


FEATURE_TAG = feature_tag()   # count + md5 of the ordered feature list
SHORT = {'l': 'l', 'b': 'b', 'parallax': 'plx', 'pml': 'pml', 'pmb': 'pmb',
         'phot_g_mean_mag': 'G', 'phot_bp_mean_mag': 'BP',
         'phot_rp_mean_mag': 'RP', 'radial_velocity': 'RV'}

# Model space: l is split into (cos l, sin l), so the model has 10 columns.
MODEL_COLUMNS = ['cos_l', 'sin_l'] + FEATURES[1:]
N_PHYS = len(FEATURES)          # 9
N_MODEL = len(MODEL_COLUMNS)    # 10
J = {f: i for i, f in enumerate(FEATURES)}   # physical column index
RV = 'radial_velocity'

G, BP, RP = 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'
ASTROMETRY = ['parallax', 'pml', 'pmb']

# Gaia measurement classes used for TARP (exact measured sets).
CLASSES = {
    'a_pos_G':          ['l', 'b', G],
    'b_pos_G_BP_RP':    ['l', 'b', G, BP, RP],
    'c_pos_phot_astro': ['l', 'b', 'parallax', 'pml', 'pmb', G, BP, RP],
    'd_all_with_RV':    list(FEATURES),
}

CREDIBILITY_GRID = np.linspace(0.0, 1.0, 101)


def model_columns(features):
    """Physical feature names -> model-space column indices (l -> [0, 1])."""
    cols = []
    for f in features:
        cols += [0, 1] if f == 'l' else [J[f] + 1]
    return cols


def model_column_to_feature(k):
    """Model column index -> physical feature name."""
    return 'l' if k <= 1 else FEATURES[k - 1]


# ===========================================================================
# 1. Conditioning patterns
# ===========================================================================
def all_but(hidden):
    return [f for f in FEATURES if f not in hidden]


# Hand-named patterns (same names as CFM_Eval_Gaia_v3.py).
NAMED_PATTERNS = {
    'unconditional':     [],
    'sky_position_only': ['l', 'b'],
    'parallax_only':     ['parallax'],
    '3D_positions_only': ['l', 'b', 'parallax'],
    'positions_and_G':   ['l', 'b', G],
    'astrometric_only':  ['l', 'b', 'parallax', 'pml', 'pmb'],
    'photometric_only':  [G, BP, RP],
    'pos_and_phot':      ['l', 'b', G, BP, RP],
    'fill_position':     all_but(['l', 'b']),
    'fill_3D_positions': all_but(['l', 'b', 'parallax']),
    'fill_pm':           all_but(['pml', 'pmb']),
    'fill_color':        all_but([BP, RP]),
    'fill_photometric':  all_but([G, BP, RP]),
}

# Conditioning ladder (CFM_Eval_Gaia_v3.py), named cond_<short names>.
LADDER_PATTERNS = [
    ['l', 'b'],
    ['pml', 'pmb'],
    ['l', 'b', 'parallax'],
    ['l', 'b', 'parallax', 'pml', 'pmb'],
    [BP, RP],
    [G, BP, RP],
    ['l', 'b', G],
    ['l', 'b', BP, RP],
    ['l', 'b', G, BP, RP],
    ['parallax', G],
    ['parallax', BP, RP],
    ['parallax', G, BP, RP],
    ['l', 'b', 'parallax', G],
    ['l', 'b', 'parallax', BP, RP],
    ['l', 'b', 'parallax', G, BP, RP],
    ['pml', 'pmb', RV],
    ['l', 'b', 'pml', 'pmb', RV],
    ['l', 'b', 'pml', 'pmb'],
    [G, BP, RP, 'pml', 'pmb'],
    ['parallax', G, BP, RP, 'pml', 'pmb'],
    ['parallax', 'pml', 'pmb'],
    ['l', 'b', 'parallax', 'pml', 'pmb'],
    ['l', 'b', 'parallax', 'pml', 'pmb', RV],
    ['l', 'b', G, BP, RP, 'pml', 'pmb'],
    ['l', 'b', G, BP, RP, 'pml', 'pmb', RV],
    # The one pattern from CFM_Imputation_Sampling_v3.py not in the eval list:
    ['l', 'b', RV],
]


def build_patterns():
    """
    Returns a list of dicts {name, cond, aliases}, de-duplicated on the SET of
    conditioning features. Order and names follow the eval script: named
    patterns, then drop-one, then keep-one, then the ladder; the first name
    for a given set wins and later ones are recorded as aliases.
    """
    candidates = list(NAMED_PATTERNS.items())
    candidates += [(f'drop_{f}', all_but([f])) for f in FEATURES]
    candidates += [(f'keep_{f}', [f]) for f in FEATURES]
    for cond in LADDER_PATTERNS:
        ordered = [f for f in FEATURES if f in cond]
        candidates.append(('cond_' + '_'.join(SHORT[f] for f in ordered), cond))

    patterns, by_set = [], {}
    for name, cond in candidates:
        key = frozenset(cond)
        if key in by_set:
            by_set[key]['aliases'].append(name)
            continue
        entry = {'name': name,
                 'cond': [f for f in FEATURES if f in key],   # feature order
                 'aliases': []}
        by_set[key] = entry
        patterns.append(entry)
    return patterns


# ===========================================================================
# 2. Path checks (fail early, and say which constant to edit)
# ===========================================================================
def check_paths(dataset_names):
    needed = {
        'LIBRARY_DIR': LIBRARY_DIR,
        'CHECKPOINT_PATH': CHECKPOINT_PATH,
        'PARAM_DICT_PATH': PARAM_DICT_PATH,
        'STATS_PATH': STATS_PATH,
        'BOOL_STR_STATS_PATH': BOOL_STR_STATS_PATH,
    }
    for name in dataset_names:
        spec = DATASETS[name]
        const = ('TEST_PURE_SAMPLE_PATH' if name == 'test'
                 else 'OMEGA_CEN_PURE_SAMPLE_PATH')
        needed[const] = spec['pure_sample']
        if spec['aux'] is not None:
            needed['OMEGA_CEN_AUX_PATH'] = spec['aux']

    missing = {k: v for k, v in needed.items() if not Path(v).exists()}
    if missing:
        print('\n' + '!' * 78)
        print('  MISSING INPUT FILE(S). Fix the path(s) below in the CONFIGURATION')
        print(f'  block at the top of {Path(__file__).name}, or in the EDIT BLOCK of')
        print('  the submission script if you are running under SLURM.')
        for const, path in missing.items():
            print(f'\n    {const} =\n      {path}')
        print('!' * 78 + '\n')
        sys.exit(1)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# ===========================================================================
# 3. Model, normalizer, denormalizer
# ===========================================================================
class ModelSetup:
    """Holds the loaded network and the normalization dictionary."""

    def __init__(self):
        import model_library_032526 as CFM_lib
        from gaia_feature_normalizer_020926 import feature_normer, feature_denormer
        self.feature_normer = feature_normer
        self.feature_denormer = feature_denormer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.param = np.load(PARAM_DICT_PATH, allow_pickle=True).item()
        p = self.param

        # --- feature checks ------------------------------------------------
        if list(p['feature_names']) != FEATURES:
            raise RuntimeError(f"checkpoint features {list(p['feature_names'])} "
                               f"!= expected {FEATURES}")
        if p['NORM_METHOD'] != 'CUSTOM':
            raise RuntimeError(f"NORM_METHOD {p['NORM_METHOD']!r} not supported; "
                               "this script assumes CUSTOM (l -> cos, sin).")

        # --- normalization dictionary (same lookup as the sampling script) -
        set_key = {'M': 'main_train', 'A': 'subsetA_train',
                   'B': 'subsetB_train'}[p['SET_TYPE']]
        preprocess = np.load(STATS_PATH, allow_pickle=True).item()
        with open(BOOL_STR_STATS_PATH) as fh:
            str_bool = json.load(fh)
        self.feature_dict = {}
        for f in FEATURES:
            if f in preprocess[set_key]:
                self.feature_dict[f] = preprocess[set_key][f]
            elif f in str_bool[set_key]:
                self.feature_dict[f] = str_bool[set_key][f]
            else:
                raise RuntimeError(f"no normalization entry for {f} in '{set_key}'")

        # --- network (same branches as the sampling script) ---------------
        common = dict(input_dim=4 * N_MODEL + 1, output_dim=N_MODEL,
                      hidden_dim=p['v_hidden_dim'],
                      ACTIVATION_TYPE=p['ACTIVATION_TYPE'],
                      FINAL_ACTIVATION=p['FINAL_ACTIVATION'],
                      ADD_INFO=p['MARGINAL_PREDICTOR'])
        if p.get('LATENT_FLOW', False):
            raise NotImplementedError('LATENT_FLOW models are not supported.')
        elif p.get('BOTTLENECK', False):
            model = CFM_lib.ConditionalFlowModel_Bottleneck(
                encoder_layers=p['encoder_layers'], decoder_layers=p['decoder_layers'],
                bottleneck=p['BOTTLENECK_SIZE'], IOB_METHOD=p['IOB_METHOD'],
                SPARSE_AUTOENCODER=p['SPARSE_AUTOENCODER'], **common)
        elif p.get('CAT_MODEL', False):
            model = CFM_lib.CatNet(n_layers=p['n_layers'], INCLUDE_OUTPUT_MASK=True,
                                   LAYER_NORM=p['LAYER_NORM'], **common)
        elif p.get('ADD_MODEL', False):
            model = CFM_lib.AddNet(n_layers=p['n_layers'], INCLUDE_OUTPUT_MASK=True,
                                   LAYER_NORM=p['LAYER_NORM'],
                                   GFTE=p.get('GFTE', False),
                                   GFTE_embed_dim=p.get('GFTE_embed_dim', 64),
                                   GFTE_scale=p.get('GFTE_scale', 16.0),
                                   GFTE_learnable=p.get('GFTE_learnable', False),
                                   **common)
        else:
            model = CFM_lib.ConditionalFlowModel_Flow_Only(
                n_layers=p['n_layers'], INCLUDE_OUTPUT_MASK=True, **common)

        state = torch.load(CHECKPOINT_PATH, map_location=self.device)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()

    # ---------------------------------------------------------------------
    def normalize(self, x_phys):
        """(n, 9) physical, no NaNs -> (n, 10) normalized model space."""
        x_phys = np.asarray(x_phys, dtype=np.float64)
        out = np.empty((x_phys.shape[0], N_MODEL), dtype=np.float64)
        cos_l, sin_l = self.feature_normer(x_phys[:, 0], 'l', self.feature_dict)
        out[:, 0], out[:, 1] = cos_l, sin_l
        for j in range(1, N_PHYS):
            out[:, j + 1] = self.feature_normer(x_phys[:, j], FEATURES[j],
                                                self.feature_dict)
        return out

    def denormalize(self, x_model):
        """(n, 10) normalized model space -> (n, 9) physical (l in degrees)."""
        x_model = np.asarray(x_model, dtype=np.float64)
        out = np.empty((x_model.shape[0], N_PHYS), dtype=np.float64)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[:, 0] = self.feature_denormer((x_model[:, 0], x_model[:, 1]),
                                              'l', self.feature_dict)
        for j in range(1, N_PHYS):
            out[:, j] = self.feature_denormer(x_model[:, j + 1], FEATURES[j],
                                              self.feature_dict)
        return out


def expand_mask(valid_phys):
    """(n, 9) bool physical validity -> (n, 10) float32 model-space mask."""
    m = np.empty((valid_phys.shape[0], N_MODEL), dtype=np.float32)
    m[:, 0] = m[:, 1] = valid_phys[:, 0]
    m[:, 2:] = valid_phys[:, 1:]
    return m


# ===========================================================================
# 4. Datasets
# ===========================================================================
def load_dataset(name, setup):
    """
    Returns a dict with, for the kept sources:
      truth_phys (n, 9)  physical, NaN where not measured
      valid      (n, 9)  bool
      x_norm     (n, 10) normalized model input (0 where not measured)
      truth_norm (n, 10) normalized truth, NaN where not measured
      rows       (n,)    row index into the Pure_Sample / Aux file
    """
    spec = DATASETS[name]
    path = Path(spec['pure_sample'])
    if FEATURE_TAG not in path.name:
        raise RuntimeError(f'{path.name} does not carry feature tag {FEATURE_TAG}; '
                           'it may have been built for a different feature set.')
    pure = torch.load(path, map_location='cpu')
    pure = pure.numpy() if torch.is_tensor(pure) else np.asarray(pure)
    if pure.ndim != 3 or pure.shape[1] != 2 or pure.shape[2] != N_PHYS:
        raise RuntimeError(f'{path.name}: expected shape (N, 2, {N_PHYS}), '
                           f'got {pure.shape}')

    rows = np.arange(pure.shape[0])
    if spec['aux'] is not None:
        aux = np.load(spec['aux'], allow_pickle=True).item()
        if len(aux['source_id']) != pure.shape[0]:
            raise RuntimeError('Aux file is not row-aligned with the Pure_Sample.')
        if 'model_features' in aux and list(aux['model_features']) != FEATURES:
            raise RuntimeError('Aux file was built for a different feature list.')
        if spec['partition_codes'] is not None:
            keep = np.isin(aux['partition_code'], spec['partition_codes'])
            rows = rows[keep]

    values = pure[rows, 0, :].astype(np.float64)
    valid = pure[rows, 1, :] > 0.5
    values = np.where(valid, values, 0.0)          # unmeasured cells -> 0

    x_norm = setup.normalize(values)
    if not np.isfinite(x_norm).all():
        raise RuntimeError(f'{name}: non-finite value in normalized model input')
    valid_model = expand_mask(valid) > 0.5

    return {
        'name': name,
        'rows': rows,
        'valid': valid,
        'truth_phys': np.where(valid, values, np.nan),
        'x_norm': x_norm.astype(np.float32),
        'truth_norm': np.where(valid_model, x_norm, np.nan),
        'n_total': int(pure.shape[0]),
    }


# ===========================================================================
# 5. Posterior sampling (RK4, same model-call convention as training)
# ===========================================================================
def rk4_sample(setup, x_data, input_mask, output_mask):
    """
    x_data, input_mask, output_mask : (n, 10) float32 numpy
    Returns normalized draws of shape (n, N_DRAWS, 10).
    """
    model, dev = setup.model, setup.device
    n = x_data.shape[0]
    data = torch.as_tensor(x_data, device=dev).repeat_interleave(N_DRAWS, dim=0)
    m_in = torch.as_tensor(input_mask, device=dev).repeat_interleave(N_DRAWS, dim=0)
    m_out = torch.as_tensor(output_mask, device=dev).repeat_interleave(N_DRAWS, dim=0)
    cond = m_in * data

    x = torch.randn(n * N_DRAWS, N_MODEL, device=dev)
    dt = 1.0 / RK4_STEPS

    def velocity(t, x_t):
        t_col = torch.full((x_t.shape[0], 1), t, device=dev)
        return model(t_col, m_out * x_t, m_out, cond, m_in)

    with torch.no_grad():
        for i in range(RK4_STEPS):
            t = i * dt
            k1 = velocity(t, x)
            k2 = velocity(t + dt / 2, x + dt * k1 / 2)
            k3 = velocity(t + dt / 2, x + dt * k2 / 2)
            k4 = velocity(t + dt, x + dt * k3)
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x.view(n, N_DRAWS, N_MODEL).cpu().numpy()


def sample_pool(setup, ds, rows, cond):
    """
    Draw N_DRAWS posterior samples for the dataset rows `rows`.
    Input mask = conditioning features; output mask = measured features.
    """
    valid_model = expand_mask(ds['valid'][rows])
    input_mask = np.zeros_like(valid_model)
    input_mask[:, model_columns(cond)] = 1.0
    if np.any(input_mask * valid_model != input_mask):
        raise RuntimeError('a conditioning feature is unmeasured for a pooled '
                           'source; strict selection failed')

    draws = np.empty((len(rows), N_DRAWS, N_MODEL), dtype=np.float32)
    for s in range(0, len(rows), BATCH_SOURCES):
        e = min(len(rows), s + BATCH_SOURCES)
        draws[s:e] = rk4_sample(setup, ds['x_norm'][rows[s:e]],
                                input_mask[s:e], valid_model[s:e])
    return draws


# ===========================================================================
# 6. Pools
# ===========================================================================
def pool_definitions(ds, cond):
    """
    Boolean eligibility mask (over the dataset) for every pool this pattern
    uses. Class pools are only defined where the class contains the
    conditioning set and leaves at least one feature to predict.
    """
    valid = ds['valid']
    cond_ok = valid[:, [J[f] for f in cond]].all(axis=1)   # True if cond empty

    pools = {'general': cond_ok}
    if RV not in cond:
        pools['rv'] = cond_ok & valid[:, J[RV]]
    for cname, cfeats in CLASSES.items():
        if set(cond) <= set(cfeats) and set(cfeats) - set(cond):
            exact = np.array([f in cfeats for f in FEATURES])
            pools[f'class_{cname}'] = (valid == exact).all(axis=1)
    return pools


def choose_rows(eligible, rng):
    """Up to N_SOURCES random eligible row indices (sorted), and a status."""
    idx = np.flatnonzero(eligible)
    if idx.size < MIN_SOURCES:
        return idx[:0], 'skipped_lt_min'
    if idx.size > N_SOURCES:
        return np.sort(rng.choice(idx, size=N_SOURCES, replace=False)), 'ok'
    return idx, 'short'


# ===========================================================================
# 7. TARP
# ===========================================================================
def tarp_coverage(draws, truth, rng):
    """
    draws : (n, N, d) normalized posterior draws on the truth-vector columns
    truth : (n, d)    normalized true values on the same columns
    Reference points theta_r ~ U(-1, 1)^d, one per source.
    Returns f (n,), ecp (N+2,), credibility (N+2,), theta_r (n, d).

    n_closer is the number of draws closer to theta_r than the truth
    (an integer 0..N). For a calibrated model it takes each of its N+1
    values with probability 1/(N+1), so the fraction of sources with
    n_closer < threshold is exactly threshold/(N+1). ECP is therefore
    evaluated with an integer comparison at credibility = threshold/(N+1),
    which avoids both floating-point ties and the -c/(N+1) bias of a
    fixed 0.01 grid.
    """
    n, N, d = draws.shape
    theta_r = rng.uniform(-1.0, 1.0, size=(n, d))
    dist_draws = np.linalg.norm(draws - theta_r[:, None, :], axis=2)   # (n, N)
    dist_truth = np.linalg.norm(truth - theta_r, axis=1)               # (n,)
    n_closer = (dist_draws < dist_truth[:, None]).sum(axis=1)          # (n,) ints 0..N
    f = n_closer / N                                                   # (n,)
    threshold = np.arange(N + 2)                                       # 0..N+1
    credibility = threshold / (N + 1)                                  # (N+2,)
    ecp = (n_closer[None, :] < threshold[:, None]).mean(axis=1)        # (N+2,)
    return f, ecp, credibility, theta_r


# ===========================================================================
# 8. RMSE
# ===========================================================================
def circular_mean_deg(angles_deg, axis=None):
    a = np.deg2rad(angles_deg)
    return np.rad2deg(np.arctan2(np.sin(a).mean(axis=axis),
                                 np.cos(a).mean(axis=axis))) % 360.0


def wrap_deg(delta):
    return (delta + 180.0) % 360.0 - 180.0


def rmse_one_feature(pred, truth, circular=False):
    """Model RMSE, mean-guess baseline RMSE, and the truth mean."""
    if circular:
        mu = circular_mean_deg(truth)
        rmse_model = np.sqrt(np.mean(wrap_deg(pred - truth) ** 2))
        rmse_base = np.sqrt(np.mean(wrap_deg(mu - truth) ** 2))
    else:
        mu = truth.mean()
        rmse_model = np.sqrt(np.mean((pred - truth) ** 2))
        rmse_base = np.sqrt(np.mean((mu - truth) ** 2))
    return rmse_model, rmse_base, mu


def compute_rmse(setup, ds, cond, pool_results):
    """
    pool_results[name] = (rows, draws) for the 'general' and 'rv' pools.
    Each predicted feature is scored on the pool sources that have it
    measured: the RV pool for radial_velocity, the general pool otherwise.
    """
    out = {k: np.full(N_PHYS, np.nan) for k in
           ('rmse_phys_model', 'rmse_phys_baseline', 'truth_mean_phys')}
    out.update({k: np.full(N_MODEL, np.nan) for k in
                ('rmse_norm_model', 'rmse_norm_baseline', 'truth_mean_norm')})
    out['n_phys'] = np.zeros(N_PHYS, dtype=np.int64)
    out['n_norm'] = np.zeros(N_MODEL, dtype=np.int64)

    # Per-pool posterior means, computed once.
    means = {}
    for pname in ('general', 'rv'):
        rows, draws = pool_results.get(pname, (None, None))
        if draws is None:
            continue
        n, N, _ = draws.shape
        phys = setup.denormalize(draws.reshape(n * N, N_MODEL)).reshape(n, N, N_PHYS)
        mean_phys = phys.mean(axis=1)
        mean_phys[:, 0] = circular_mean_deg(phys[:, :, 0], axis=1)
        means[pname] = (rows, mean_phys, draws.mean(axis=1))

    for f in FEATURES:
        if f in cond:
            continue
        pname = 'rv' if f == RV else 'general'
        if pname not in means:
            continue
        rows, mean_phys, mean_norm = means[pname]
        sel = ds['valid'][rows, J[f]]
        n_sel = int(sel.sum())
        j = J[f]
        out['n_phys'][j] = n_sel
        for k in model_columns([f]):
            out['n_norm'][k] = n_sel
        if n_sel < MIN_SOURCES:
            continue

        # physical units
        m, b, mu = rmse_one_feature(mean_phys[sel, j], ds['truth_phys'][rows[sel], j],
                                    circular=(f == 'l'))
        out['rmse_phys_model'][j], out['rmse_phys_baseline'][j] = m, b
        out['truth_mean_phys'][j] = mu

        # normalized model space (cos l and sin l as two ordinary columns)
        for k in model_columns([f]):
            m, b, mu = rmse_one_feature(mean_norm[sel, k], ds['truth_norm'][rows[sel], k])
            out['rmse_norm_model'][k], out['rmse_norm_baseline'][k] = m, b
            out['truth_mean_norm'][k] = mu
    return out


def rmse_on_pool(setup, ds, rows, draws, features):
    """
    RMSE on ONE pool, e.g. a measurement-class pool, for the listed features.

    rows     : (n,) indices into ds (as stored in pool_results)
    draws    : (n, N_DRAWS, 10) normalized draws for those rows, or None
    features : physical feature names to score (for a class pool: the class
               features minus the conditioning features, i.e. the TARP
               truth vector)

    Same prediction, baseline and output layout as compute_rmse. In a class
    pool every class feature is measured for every source, so each feature is
    scored on all n sources -- the same sources and draws TARP uses.
    """
    out = {k: np.full(N_PHYS, np.nan) for k in
           ('rmse_phys_model', 'rmse_phys_baseline', 'truth_mean_phys')}
    out.update({k: np.full(N_MODEL, np.nan) for k in
                ('rmse_norm_model', 'rmse_norm_baseline', 'truth_mean_norm')})
    out['n_phys'] = np.zeros(N_PHYS, dtype=np.int64)
    out['n_norm'] = np.zeros(N_MODEL, dtype=np.int64)
    if draws is None or not features:
        return out

    n, N, _ = draws.shape
    phys = setup.denormalize(draws.reshape(n * N, N_MODEL)).reshape(n, N, N_PHYS)
    mean_phys = phys.mean(axis=1)
    mean_phys[:, 0] = circular_mean_deg(phys[:, :, 0], axis=1)
    mean_norm = draws.mean(axis=1)

    for f in features:
        j = J[f]
        sel = ds['valid'][rows, j]
        n_sel = int(sel.sum())
        out['n_phys'][j] = n_sel
        for k in model_columns([f]):
            out['n_norm'][k] = n_sel
        if n_sel < MIN_SOURCES:
            continue

        m, b, mu = rmse_one_feature(mean_phys[sel, j], ds['truth_phys'][rows[sel], j],
                                    circular=(f == 'l'))
        out['rmse_phys_model'][j], out['rmse_phys_baseline'][j] = m, b
        out['truth_mean_phys'][j] = mu

        for k in model_columns([f]):
            m, b, mu = rmse_one_feature(mean_norm[sel, k], ds['truth_norm'][rows[sel], k])
            out['rmse_norm_model'][k], out['rmse_norm_baseline'][k] = m, b
            out['truth_mean_norm'][k] = mu
    return out


# ===========================================================================
# 9. Output helpers
# ===========================================================================
def set_json_attr(obj, key, value):
    obj.attrs[key] = json.dumps(value)


def write_rmse_csv(path, pattern_names, table, columns, prefix):
    """One row per pattern: model RMSE, baseline RMSE and n for every column."""
    header = ['pattern']
    for c in columns:
        header += [f'{c}_rmse_model', f'{c}_rmse_baseline', f'{c}_n']
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, name in enumerate(pattern_names):
            row = [name]
            for k in range(len(columns)):
                row += [f"{table[f'rmse_{prefix}_model'][i, k]:.6g}",
                        f"{table[f'rmse_{prefix}_baseline'][i, k]:.6g}",
                        int(table[f'n_{prefix}'][i, k])]
            w.writerow(row)


# ===========================================================================
# 10. One dataset
# ===========================================================================
def run_dataset(name, setup, patterns, timestamp):
    t_start = time.time()
    rng = np.random.default_rng([SEED, list(DATASETS).index(name)])
    torch.manual_seed(SEED + list(DATASETS).index(name))

    print(f'\n{"=" * 78}\nDATASET: {name}\n{"=" * 78}')
    ds = load_dataset(name, setup)
    n_ds = len(ds['rows'])
    print(f'  {n_ds:,} sources kept (of {ds["n_total"]:,} in the Pure_Sample)')
    for cname, cfeats in CLASSES.items():
        exact = np.array([f in cfeats for f in FEATURES])
        print(f'  class {cname:<18s}: {int((ds["valid"] == exact).all(1).sum()):>9,} sources')

    out_dir = Path(OUT_DIR)
    h5_path = out_dir / f'{RUN_TAG}_TARP_RMSE_{name}_{timestamp}.h5'
    spec = DATASETS[name]

    rmse_keys_phys = ('rmse_phys_model', 'rmse_phys_baseline', 'truth_mean_phys', 'n_phys')
    rmse_keys_norm = ('rmse_norm_model', 'rmse_norm_baseline', 'truth_mean_norm', 'n_norm')
    table = {k: [] for k in rmse_keys_phys + rmse_keys_norm}
    # Class-pool RMSE: one table per measurement class, same keys.
    class_table = {c: {k: [] for k in rmse_keys_phys + rmse_keys_norm}
                   for c in CLASSES}

    with h5py.File(h5_path, 'w') as h5:
        # ---- file-level metadata ---------------------------------------
        h5.attrs['model_name'] = MODEL_NAME
        h5.attrs['checkpoint'] = CHECKPOINT_PATH
        h5.attrs['dataset'] = name
        h5.attrs['pure_sample'] = str(spec['pure_sample'])
        h5.attrs['aux_file'] = str(spec['aux'])
        set_json_attr(h5, 'partition_codes', spec['partition_codes'])
        set_json_attr(h5, 'feature_names', FEATURES)
        set_json_attr(h5, 'model_columns', MODEL_COLUMNS)
        set_json_attr(h5, 'classes', CLASSES)
        set_json_attr(h5, 'pattern_names', [p['name'] for p in patterns])
        h5.attrs['n_sources_in_dataset'] = n_ds
        h5.attrs['n_draws'] = N_DRAWS
        h5.attrs['rk4_steps'] = RK4_STEPS
        h5.attrs['n_sources_target'] = N_SOURCES
        h5.attrs['min_sources'] = MIN_SOURCES
        h5.attrs['seed'] = SEED
        h5.attrs['reference_rule'] = ('theta_r ~ U(-1, 1) independently per normalized '
                                      'model dimension, one per source; x-independent')
        h5.attrs['tarp_space'] = 'normalized model space (l -> cos l, sin l)'
        h5.attrs['tarp_distance'] = 'L2 (Euclidean)'
        h5.attrs['timestamp'] = timestamp
        h5.attrs['status'] = 'running'

        for p_idx, pat in enumerate(patterns):
            t0 = time.time()
            cond = pat['cond']
            pred = [f for f in FEATURES if f not in cond]
            grp = h5.create_group(f"patterns/{pat['name']}")
            set_json_attr(grp, 'cond_features', cond)
            set_json_attr(grp, 'pred_features', pred)
            set_json_attr(grp, 'aliases', pat['aliases'])
            grp.attrs['pattern_index'] = p_idx

            # ---- build pools and sample ---------------------------------
            pool_results = {}
            for pname, eligible in pool_definitions(ds, cond).items():
                rows, status = choose_rows(eligible, rng)
                pg = grp.create_group(f'pools/{pname}')
                pg.attrs['n_eligible'] = int(eligible.sum())
                pg.attrs['n_used'] = len(rows)
                pg.attrs['status'] = status
                pg.create_dataset('rows', data=ds['rows'][rows])
                if status == 'skipped_lt_min':
                    continue
                draws = sample_pool(setup, ds, rows, cond)
                if SAVE_SAMPLES:
                    pg.create_dataset('draws', data=draws, compression='gzip')
                pool_results[pname] = (rows, draws)

            # ---- TARP, one curve per measurement class ------------------
            tarp_summary = []
            for cname, cfeats in CLASSES.items():
                tg = grp.create_group(f'tarp/{cname}')
                truth_feats = [f for f in cfeats if f not in cond]
                truth_cols = model_columns(truth_feats)
                set_json_attr(tg, 'class_features', cfeats)
                set_json_attr(tg, 'truth_features', truth_feats)
                set_json_attr(tg, 'truth_model_columns', truth_cols)
                tg.attrs['n_draws'] = N_DRAWS

                rows = draws = None
                if not set(cond) <= set(cfeats):
                    status, n_used = 'not_applicable', 0
                elif not truth_feats:
                    status, n_used = 'empty', 0
                elif f'class_{cname}' not in pool_results:
                    status = 'skipped_lt_min'
                    n_used = int(grp[f'pools/class_{cname}'].attrs['n_eligible'])
                else:
                    rows, draws = pool_results[f'class_{cname}']
                    truth = ds['truth_norm'][rows][:, truth_cols]
                    if not np.isfinite(truth).all():
                        raise RuntimeError(f'{cname}: non-finite truth in a class pool')
                    f_vals, ecp, credibility, theta_r = tarp_coverage(
                        draws[:, :, truth_cols].astype(np.float64), truth, rng)
                    tg.create_dataset('f', data=f_vals)
                    tg.create_dataset('ecp', data=ecp)
                    tg.create_dataset('credibility', data=credibility)
                    tg.create_dataset('reference_points', data=theta_r)
                    status, n_used = 'ok', len(rows)
                tg.attrs['status'] = status
                tg.attrs['n_sources'] = n_used

                # RMSE on the SAME class pool, sources and draws as the TARP
                # curve, scored on the TARP truth vector. All NaN when the
                # curve was not computed.
                rc = rmse_on_pool(setup, ds, rows,
                                  draws if status == 'ok' else None,
                                  truth_feats if status == 'ok' else [])
                cg = grp.create_group(f'rmse_class/{cname}')
                set_json_attr(cg, 'scored_features', truth_feats if status == 'ok' else [])
                cg.attrs['status'] = status
                for k, v in rc.items():
                    cg.create_dataset(k, data=v)
                    class_table[cname][k].append(v)
                tarp_summary.append(f'{cname[0]}:{status if status != "ok" else n_used}')

            # ---- RMSE ----------------------------------------------------
            rm = compute_rmse(setup, ds, cond, pool_results)
            rg = grp.create_group('rmse')
            for k, v in rm.items():
                rg.create_dataset(k, data=v)
                table[k].append(v)

            print(f"  [{p_idx + 1:>2}/{len(patterns)}] {pat['name']:<28s} "
                  f"TARP {' '.join(tarp_summary):<58s} {time.time() - t0:6.1f}s")
            h5.flush()

        # ---- stacked RMSE table ----------------------------------------
        tg = h5.create_group('rmse_table')
        set_json_attr(tg, 'pattern_names', [p['name'] for p in patterns])
        set_json_attr(tg, 'feature_names', FEATURES)
        set_json_attr(tg, 'model_columns', MODEL_COLUMNS)
        for k, rows_list in table.items():
            table[k] = np.stack(rows_list)
            tg.create_dataset(k, data=table[k])

        # ---- stacked class-pool RMSE tables, one group per class ---------
        for cname in CLASSES:
            cg = h5.create_group(f'rmse_class_table/{cname}')
            set_json_attr(cg, 'pattern_names', [p['name'] for p in patterns])
            set_json_attr(cg, 'feature_names', FEATURES)
            set_json_attr(cg, 'model_columns', MODEL_COLUMNS)
            set_json_attr(cg, 'class_features', CLASSES[cname])
            for k, rows_list in class_table[cname].items():
                class_table[cname][k] = np.stack(rows_list)
                cg.create_dataset(k, data=class_table[cname][k])
        h5.attrs['status'] = 'complete'

    names = [p['name'] for p in patterns]
    csv_phys = out_dir / f'{RUN_TAG}_RMSE_phys_{name}_{timestamp}.csv'
    csv_norm = out_dir / f'{RUN_TAG}_RMSE_norm_{name}_{timestamp}.csv'
    write_rmse_csv(csv_phys, names, table, [SHORT[f] for f in FEATURES], 'phys')
    write_rmse_csv(csv_norm, names, table,
                   ['cos_l', 'sin_l'] + [SHORT[f] for f in FEATURES[1:]], 'norm')

    csv_class = []
    for cname in CLASSES:
        p = out_dir / f'{RUN_TAG}_RMSE_phys_{name}_class_{cname}_{timestamp}.csv'
        write_rmse_csv(p, names, class_table[cname], [SHORT[f] for f in FEATURES], 'phys')
        csv_class.append(p)

    print(f'\n  wrote {h5_path}\n  wrote {csv_phys}\n  wrote {csv_norm}')
    for p in csv_class:
        print(f'  wrote {p}')
    print(f'  dataset {name} finished in {(time.time() - t_start) / 60:.1f} min')


# ===========================================================================
# 11. Main
# ===========================================================================
def build_parser():
    """
    Every CONFIGURATION constant has a matching option; the constant is the
    default, so the script still runs with no arguments at all.
    """
    p = argparse.ArgumentParser(
        description='TARP coverage curves and RMSE for the Gaia CFM model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group('what to run')
    g.add_argument('--datasets', nargs='+', choices=list(DATASETS),
                   default=DATASETS_TO_RUN, help='datasets to evaluate')

    g = p.add_argument_group('model and support files')
    g.add_argument('--library-dir', default=LIBRARY_DIR,
                   help='directory holding model_library / gaia_feature_normalizer')
    g.add_argument('--model-name', default=MODEL_NAME)
    g.add_argument('--checkpoint-path', default=None,
                   help='default: <model-param-dict-dir>/../saved_models/<model>_ema.pth')
    g.add_argument('--model-param-dict-dir', default=str(Path(PARAM_DICT_PATH).parent))
    g.add_argument('--stats-dir', default=str(Path(STATS_PATH).parent))
    g.add_argument('--stats-file', default=Path(STATS_PATH).name)
    g.add_argument('--bool-str-stats-file', default=Path(BOOL_STR_STATS_PATH).name)

    g = p.add_argument_group('data')
    g.add_argument('--test-pure-sample', default=None,
                   help='explicit path; overrides the canonical name below')
    g.add_argument('--pure-sample-dir', default=str(Path(TEST_PURE_SAMPLE_PATH).parent))
    g.add_argument('--test-fraction', type=float, default=0.01,
                   help="sampler's TEST_FRACTION (0.01 -> tag '1pct')")
    g.add_argument('--load-number', type=int, default=10,
                   help="sampler's LOAD_NUMBER")
    g.add_argument('--omega-cen-pure-sample', default=OMEGA_CEN_PURE_SAMPLE_PATH)
    g.add_argument('--omega-cen-aux', default=OMEGA_CEN_AUX_PATH)

    g = p.add_argument_group('run size')
    g.add_argument('--n-sources', type=int, default=N_SOURCES,
                   help='sources per pool')
    g.add_argument('--n-draws', type=int, default=N_DRAWS,
                   help='posterior draws per source')
    g.add_argument('--rk4-steps', type=int, default=RK4_STEPS)
    g.add_argument('--min-sources', type=int, default=MIN_SOURCES,
                   help='skip any TARP curve / RMSE value with fewer sources')
    g.add_argument('--batch-sources', type=int, default=BATCH_SOURCES,
                   help='sources per forward pass (x n-draws rows on the GPU)')
    g.add_argument('--seed', type=int, default=SEED)

    g = p.add_argument_group('output')
    g.add_argument('--out-dir', default=OUT_DIR)
    g.add_argument('--run-tag', default=RUN_TAG,
                   help='short tag used in output filenames')
    g.add_argument('--save-samples', action='store_true', default=SAVE_SAMPLES,
                   help='also store the raw normalized draws')
    return p


def apply_overrides(args):
    """Write the parsed arguments back over the CONFIGURATION constants."""
    g = globals()
    g['LIBRARY_DIR'] = args.library_dir
    g['MODEL_NAME'] = args.model_name
    g['CHECKPOINT_PATH'] = args.checkpoint_path or CHECKPOINT_PATH
    g['PARAM_DICT_PATH'] = str(Path(args.model_param_dict_dir)
                               / f'{args.model_name}_model_param_dict.npy')
    g['STATS_PATH'] = str(Path(args.stats_dir) / args.stats_file)
    g['BOOL_STR_STATS_PATH'] = str(Path(args.stats_dir) / args.bool_str_stats_file)

    test_path = args.test_pure_sample or str(default_pure_sample_path(
        args.pure_sample_dir, args.test_fraction, args.load_number))
    g['TEST_PURE_SAMPLE_PATH'] = test_path
    g['OMEGA_CEN_PURE_SAMPLE_PATH'] = args.omega_cen_pure_sample
    g['OMEGA_CEN_AUX_PATH'] = args.omega_cen_aux
    DATASETS['test']['pure_sample'] = test_path
    for name in ('omega_cen_all', 'omega_cen_not_A'):
        DATASETS[name]['pure_sample'] = args.omega_cen_pure_sample
        DATASETS[name]['aux'] = args.omega_cen_aux

    g['N_SOURCES'] = args.n_sources
    g['N_DRAWS'] = args.n_draws
    g['RK4_STEPS'] = args.rk4_steps
    g['MIN_SOURCES'] = args.min_sources
    g['BATCH_SOURCES'] = args.batch_sources
    g['SEED'] = args.seed
    g['OUT_DIR'] = args.out_dir
    g['RUN_TAG'] = args.run_tag
    g['SAVE_SAMPLES'] = args.save_samples

    if LIBRARY_DIR not in sys.path:
        sys.path.insert(0, LIBRARY_DIR)
    if N_SOURCES < MIN_SOURCES:
        print(f'WARNING: --n-sources {N_SOURCES} is below --min-sources '
              f'{MIN_SOURCES}; every pool will be skipped. Lower --min-sources.')


def main(argv=None):
    args = build_parser().parse_args(argv)
    apply_overrides(args)

    check_paths(args.datasets)
    print(f'run tag {RUN_TAG} | {N_SOURCES} sources x {N_DRAWS} draws, '
          f'{RK4_STEPS} RK4 steps, min {MIN_SOURCES} | out {OUT_DIR}')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

    patterns = build_patterns()
    print(f'{len(patterns)} conditioning patterns')
    setup = ModelSetup()
    print(f'model loaded on {setup.device}: {MODEL_NAME}')

    for name in args.datasets:
        run_dataset(name, setup, patterns, timestamp)
    print('\nDone.')


if __name__ == '__main__':
    main()
