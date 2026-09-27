import os
from hdf5_dataset_creator_020926 import HDF5IterableDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from datetime import datetime
import json
from pathlib import Path
import model_library_032526 as CFM_lib
import CFM_Training_Library_Parallel as CFM_train
import glob

def setup_ddp():
    """Initialize the process group. Called once at startup."""
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend='nccl', device_id=device)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()
    
def main():
    code_start = time.time()
    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = (rank == 0)  # only rank 0 should print and save

    #Where is the data?
    stats_dir = Path("YOUR PATH/temp_project/")
    data_dir = 'YOUR PATH/GaiaSource_12182025'

    rng = np.random.default_rng()
    _ = torch.random.manual_seed(0)
        
    def decimal_name(sigma):
        if sigma >= 0.01:
            SIGMA_NAME = str(int(100*sigma))
        else:
            temp_name = f"{sigma:.1e}"
            SIGMA_NAME = ''
            for letter in temp_name:
                if letter == '.':
                    SIGMA_NAME += '_'
                else:
                    SIGMA_NAME += letter
        return SIGMA_NAME

    LOAD_MODEL = True
    #Modifies the loaded model.
    MODIFY_LOAD = False
    #make sure the other model parameters match!

    num_epochs = 400
    
    save_directory = 'YOUR PATH/saved_models/'
        
    if LOAD_MODEL:
        #sets the name of the model you are loading.
        LOAD_NAME = 'POS_PM_mask_set_setA_CUSTOM_2026-07-10_15-01-16_mask_train_mask_val_MSE_vHD512_SIGMA1_0e-08_GELU_lr1_0e-04_wd0_0e+00_layers8_MP85_ADD_SET_VALACUMVAL_10'
        #'POS_PM_mask_set_setA_CUSTOM_2026-07-02_11-22-30_mask_train_mask_val_MSE_vHD512_SIGMA1_0e-08_GELU_lr_max_5_0e-03_wd0_0e+00_layers8_MP_RAND_ADD_SET_VALACUMVAL_10'
        #makes sure we are loading the previous final model. Note that this is a one time fix and should be deleted for later runs.
        pattern = save_directory + LOAD_NAME + '_final.pth'
        ema_path = save_directory + LOAD_NAME + '_ema.pth'
        matches = glob.glob(pattern)

        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one file matching {pattern}, found {len(matches)}")

        filepath = matches[0]
        
        #sets the name of model parameter file you are loading, as well as the loss dictionary
        MODEL_NAME = 'POS_PM_mask_set_setA_CUSTOM_2026-07-10_15-01-16_mask_train_mask_val_MSE_vHD512_SIGMA1_0e-08_GELU_lr1_0e-04_wd0_0e+00_layers8_MP85_ADD_SET_VALACUMVAL_10'
        #'POS_PM_mask_set_setA_CUSTOM_2026-07-02_11-22-30_mask_train_mask_val_MSE_vHD512_SIGMA1_0e-08_GELU_lr_max_5_0e-03_wd0_0e+00_layers8_MP_RAND_ADD_SET_VALACUMVAL_10'
        
        MODEL_PARAM_DICT = np.load('YOUR PATH/model_param_dicts/' + MODEL_NAME + '_model_param_dict.npy', allow_pickle = True).item()
        
        
        #Backfill for param dicts saved before RANDOM_INTRA_SET_MASK_PROB
        #existed.  Without this, resuming an old model hits a hard KeyError
        #at the mutual-exclusion asserts (and again at the train(...) call).
        #setdefault only fills the key when absent; dicts that already carry
        #an explicit value are respected.  False = resume old models under
        #their original masking scheme; change to True to switch a resumed
        #model to per-source random mask probabilities.
        #MODEL_PARAM_DICT.setdefault('RANDOM_INTRA_SET_MASK_PROB', False)
        
        #Sets the periodicity of the cosine used for scheduling the learning rate in Cosine LR (cosine period == T_max*2)
        #Should be set very high to avoid repeating the cosine
        #MODEL_PARAM_DICT['T_max'] = 400
        #NEW_SCHEDULER = True
        
        #LOAD_FROM_NAME_MOD
        MODEL_PARAM_DICT['AUTOENCODER_MODEL_NAME'] = ''
        NEW_SCHEDULER = False
        
        if MODIFY_LOAD:
            if 'UPDATED_PARAMS' not in MODEL_PARAM_DICT:
                MODEL_PARAM_DICT['UPDATED_PARAMS'] = {}
            
            '''
            Update which params are include in this subdictionary. This will keep track of which parameters have changed during the training process.
            '''
            MODEL_PARAM_KEYS = ['SET_TYPE', 'MASK_PROB_INPUT_PWR', 'MASK_PROB_OUTPUT_PWR', 'ACUM_VAL', 'batch_size', 'shuffle_sources_in_file', 'SAVE_MODEL']

            NEW_VALUES = ['M', 0.25, 1, 10, 50000, True, True]
            for i in range(len(MODEL_PARAM_KEYS)):
                key = MODEL_PARAM_KEYS[i]
                if key not in MODEL_PARAM_DICT['UPDATED_PARAMS']:
                    MODEL_PARAM_DICT['UPDATED_PARAMS'][key] = {}
                    MODEL_PARAM_DICT['UPDATED_PARAMS'][key]['VALUES'] = []
                    MODEL_PARAM_DICT['UPDATED_PARAMS'][key]['EPOCHS'] = []
                #Add old values to dictionary, new values added after old are overwritten. Epochs when change occured are saved before param dict is saved.
                if key in MODEL_PARAM_DICT: MODEL_PARAM_DICT['UPDATED_PARAMS'][key]['VALUES'].append(MODEL_PARAM_DICT[key])
                else:
                    MODEL_PARAM_DICT['UPDATED_PARAMS'][key]['VALUES'].append('N/A')
                #Update existing value
                MODEL_PARAM_DICT[key] = NEW_VALUES[i]
            MODEL_PARAM_DICT['NAME_MOD'] = '_M_Set_quartMP_10AV_BS'
    else:
        
        MODEL_PARAM_DICT = {}
        MODEL_PARAM_DICT['feature_names'] = ['l', 'b', 'parallax', 'pml', 'pmb', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'radial_velocity']
        
        #Uses a consistent training evaluation mask scheme
        MODEL_PARAM_DICT['RANDOM_INTRA_SET_MASK_PROB'] = True
        
        #Uses a consistent validation evaluation mask scheme
        MODEL_PARAM_DICT['SET_VAL'] = True
                
        #Defines the validation evaluation mask scheme if SET_VAL == True.
        #Currently set assuming using custom L normalization (i.e., the first to entries in the array both correspond to l)
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT'] = {}
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['unconditional'] = np.zeros(len(MODEL_PARAM_DICT['feature_names'])+1)
        
        #MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['autoencoder'] = np.ones(len(MODEL_PARAM_DICT['feature_names'])+1)
        
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['positions'] = np.zeros(len(MODEL_PARAM_DICT['feature_names'])+1)
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['positions'][0] = 1
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['positions'][1] = 1
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['positions'][2] = 1
        
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['positions+G'] = np.zeros(len(MODEL_PARAM_DICT['feature_names'])+1)
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['positions+G'][0] = 1
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['positions+G'][1] = 1
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['positions+G'][2] = 1
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['positions+G'][6] = 1
 
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['parallax'] = np.zeros(len(MODEL_PARAM_DICT['feature_names'])+1)
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['parallax'][3] = 1
        
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['proper motions'] = np.zeros(len(MODEL_PARAM_DICT['feature_names'])+1)
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['proper motions'][4] = 1
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['proper motions'][5] = 1
        
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['magnitudes'] = np.zeros(len(MODEL_PARAM_DICT['feature_names'])+1)
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['magnitudes'][6] = 1
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['magnitudes'][7] = 1
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['magnitudes'][8] = 1
        
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no positions'] = np.ones(len(MODEL_PARAM_DICT['feature_names'])+1)
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no positions'][0] = 0
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no positions'][1] = 0
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no positions'][2] = 0
        
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no parallax'] = np.ones(len(MODEL_PARAM_DICT['feature_names'])+1)
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no parallax'][3] = 0
        
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no velocities'] = np.ones(len(MODEL_PARAM_DICT['feature_names'])+1)
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no velocities'][4] = 0
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no velocities'][5] = 0
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no velocities'][9] = 0
        
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no magnitudes'] = np.ones(len(MODEL_PARAM_DICT['feature_names'])+1)
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no magnitudes'][6] = 0
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no magnitudes'][7] = 0
        MODEL_PARAM_DICT['CUSTOM_MASK_DICT']['no magnitudes'][8] = 0
        
        timestamp = datetime.now()# Format as human-readable string with underscores
        MODEL_PARAM_DICT['timestamp'] = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        if is_main:
            print("Y-M-D_H-M-S", MODEL_PARAM_DICT['timestamp'])  # Example: 2025-09-22_14-18-45
        #model name modification when retraining but with slightly different training parameters
        MODEL_PARAM_DICT['NAME_MOD'] = ''

        #Determines whether you are using the original Gaia source catalog files or sharded versions. Options are "Original" and "Shard".
        MODEL_PARAM_DICT['READ_TYPE'] = "Original"

        #Name of subset of data used
        MODEL_PARAM_DICT['SET_TYPE'] = 'A'

        #How the data is normalized.
        MODEL_PARAM_DICT['NORM_METHOD'] = 'CUSTOM'

        #Activates a single training run for debugging
        MODEL_PARAM_DICT['DEBUG_TRAIN'] = False

        #Activates a single validation run for debugging
        MODEL_PARAM_DICT['DEBUG_VAL'] = False

        #Includes MAE loss in the loss function
        MODEL_PARAM_DICT['MAE_LOSS'] = False

        #Includes MSE loss in the loss function
        MODEL_PARAM_DICT['MSE_LOSS'] = True

        #Whether or not masking is applied to training inputs.
        MODEL_PARAM_DICT['MASK_TRAIN'] = True

        #Whether or not masking is applied to validation inputs.
        MODEL_PARAM_DICT['MASK_VAL'] = True

        #Apply sine schedule to masking probability. Cycles from 100% unmasked to ~2% unmasked, changing as a function of batch
        MODEL_PARAM_DICT['SINE_MASK_PROB'] = False

        #If true, the output masking probability will be 1-mask_prob, where mask_prob is the probability of a feature being masked for the input data.
        #Note that for the prob mask loop setting, this is done in addition to the output_mask_prob = input_mask_prob (i.e., the for loop is doubled in length)
        MODEL_PARAM_DICT['OPPOSITE_OUTPUT_MASK'] = False

        #If True, always include the features included in NEVER_MASK_INDICES as inputs
        #This is an experiment to see if I can decrease prediction errors by training the model on consistent information.
        MODEL_PARAM_DICT['NEVER_MASK_INPUT'] = False
        MODEL_PARAM_DICT['NEVER_MASK_INPUT_INDICES'] = [0,1]

        #If true, always use all available features for the output
        MODEL_PARAM_DICT['NEVER_MASK_OUTPUT'] = False
        MODEL_PARAM_DICT['NEVER_MASK_OUTPUT_INDICES'] = [0,1]

        #If true, randomly choose an input and output masking probability for each epoch. For validation epochs, cycle through set probabilties using mask_prob_step
        MODEL_PARAM_DICT['MASK_PROB_RAND'] = False
        
        #What power is applied to the random generation of input mask probabilities. P<1 increases the frequency of highly masked runs
        MODEL_PARAM_DICT['MASK_PROB_INPUT_PWR'] = 0.25
        #What power is applied to the random generation of output mask probabilities. P<1 increases the frequency of highly masked runs
        MODEL_PARAM_DICT['MASK_PROB_OUTPUT_PWR'] = 1
        
        #If indices are listed as mask sets, then when one parameter is masked, the other parameters will also always be masked. The random masking probability of the set being masked will be the same as if only one parameter is.
        MODEL_PARAM_DICT['MASK_SETS'] = False
        MODEL_PARAM_DICT['MASK_SETS_INDICES'] = [(3,4)]

        assert MODEL_PARAM_DICT['MASK_PROB_RAND']*MODEL_PARAM_DICT['SINE_MASK_PROB'] == False, "Sine scheduled mask probabilities and random mask probabilities are mutually exclusive"

        #Controls the step size in masking probabilities for the validation set if MASK_PROB_RAND==True, range is from 0 to 1.
        MODEL_PARAM_DICT['mask_prob_step'] = 0.2

        #If using a sparse autoencoder, this number sets the size of the latent space that is passed through. Ignored if SPARSE_AUTOENCODER == False.
        MODEL_PARAM_DICT['K_LATENT_ALLOWED'] = 16

        #See Ho+23. Orders embeddings of velocity network by progressively masking interior latent space.
        MODEL_PARAM_DICT['IOB_METHOD'] = False

        #If IOB is true, this determines what fraction of the latent variables are used for validation. If set to None, random fractions are used, as in training.
        MODEL_PARAM_DICT['VAL_M_FRAC'] = 1

        #See Wu+25. Uses large embedding space, but only passes on k latent variables onto the decoder. Variables are chosen in order of activation strength.
        MODEL_PARAM_DICT['SPARSE_AUTOENCODER'] = False

        #Whether to use a model with a bottleneck or not
        MODEL_PARAM_DICT['BOTTLENECK'] = False

        #If True, uses a model that concatenates model inputs on to every layer
        MODEL_PARAM_DICT['CAT_MODEL'] = False

        #If True, uses a model that adds model input layer output on to the input of every subsequent layer
        MODEL_PARAM_DICT['ADD_MODEL'] = True

        #If True, adds a batch normalization layer to the add model (only)
        MODEL_PARAM_DICT['LAYER_NORM'] = True

        #If true, the encoder model exists before the flow. Note that LATENT_FLOW == True overrides the BOTTLENECK flag
        MODEL_PARAM_DICT['LATENT_FLOW'] = False

        #If true, trains the autoencoder component of the Latent Flow model variant
        MODEL_PARAM_DICT['AUTOENCODER_TRAINING'] = False

        #If true, trains the CFM component of the Latent Flow model
        MODEL_PARAM_DICT['LATENT_FLOW_TRAINING'] = False

        #Dummy variable. If you wish to load a pretrained autoencoder model, go the model loading below.
        MODEL_PARAM_DICT['autoencoder_model'] = 0
        
        #autoencoder model name, needed for training latent space autoencoder model.
        MODEL_PARAM_DICT['AUTOENCODER_MODEL_NAME'] = ''

        #If using an encoder+decoder velocity network (i.e., BOTTLENECK==True), then this sets the size of the embedding space. Note that if SPARSE_AUTOENCODER==True, this should be much larger than K_LATENT_ALLOWED.
        MODEL_PARAM_DICT['BOTTLENECK_SIZE'] = 8

        if MODEL_PARAM_DICT['SPARSE_AUTOENCODER'] or MODEL_PARAM_DICT['IOB_METHOD']:
            assert MODEL_PARAM_DICT['BOTTLENECK'], "Model must have special central layer for Sparse Autoencoder and IOB methods"
        if MODEL_PARAM_DICT['IOB_METHOD']:
            assert MODEL_PARAM_DICT['SPARSE_AUTOENCODER'] == False, "Cannot use sparse autoencoder and IOB at same time"
        elif MODEL_PARAM_DICT['SPARSE_AUTOENCODER']:
            assert MODEL_PARAM_DICT['IOB_METHOD'] == False, "Cannot use sparse autoencoder and IOB at same time"
            
        #If true, the model will learn to predict output_mask*data - data*input_mask*output_mask. This should make conditional probabilities easier when you know some target features.
        #Not implemented for latent flow model
        MODEL_PARAM_DICT['MARGINAL_PREDICTOR'] = False

        #Number of hidden layers in velocity network assuming no bottleneck (note there is always an input layer and an output layer)
        #If using latent flows model this also controls the number of layers for the latent space cfm model.
        MODEL_PARAM_DICT['n_layers'] = 8

        #Number of hidden layers in encoder portion of velocity network (note there is always an input layer and an output layer)
        MODEL_PARAM_DICT['encoder_layers'] = 3

        #Number of hidden layers in decoder portion of velocity network (note there is always an input layer and an output layer)
        MODEL_PARAM_DICT['decoder_layers'] = 3

        MODEL_PARAM_DICT['mask_prob'] = 0.85   # probability of masking each dimension, only relevant if SINE_MASK_PROB == False and MASK_PROB_RAND == False
        MODEL_PARAM_DICT['mask_prob_min'] = 0.0 #minimum mask probability, only relevant if SINE_MASK_PROB == True

        #frequency of sine scheduler of mask probability
        #values below are separated for model name purposes
        MODEL_PARAM_DICT['mp_sin_freq_int'] = 5
        MODEL_PARAM_DICT['mp_sin_freq'] = 1/(MODEL_PARAM_DICT['mp_sin_freq_int']*np.pi)

        #number of workers used for dataloader - note because of limited gpu space you will need to balance the number of workers with the size of the model!
        MODEL_PARAM_DICT['N_WORKERS'] = 8
        #size of the velocity network
        MODEL_PARAM_DICT['v_hidden_dim'] = 512

        #sets the scatter in the flow matching
        MODEL_PARAM_DICT['sigma'] = 10**(-8)
        #sets the activation function for the hidden layers of the network
        MODEL_PARAM_DICT['ACTIVATION_TYPE'] = 'GELU'#'LeakyReLU'
        #sets the final activation of the network. If none, the final layer is linear
        MODEL_PARAM_DICT['FINAL_ACTIVATION'] = None#'ELU'
        #sets the learning rate
        MODEL_PARAM_DICT['lr'] = 1e-4#5e-3#10**(-3.5) #

        #If true use a scheduler, and treat lr as the upper bound on the learning rate
        MODEL_PARAM_DICT['SCHEDULER'] = False
        
        #determines which scheduler to use.
        MODEL_PARAM_DICT['SCHEDULER_TYPE'] = 'Cosine Annealing LR' #'Cosine Annealing Warm Restarts' or 'Cosine Annealing LR'
        
        #Sets how frequently the scheduler resets, only applicable if using Cosine Annealing scheduler (see above)
        #Currently hardcoded to reset with total number of epochs per training cycle
        MODEL_PARAM_DICT['T_0'] = 50
        
        #Sets the periodicity of the cosine used for scheduling the learning rate in Cosine LR (cosine period == T_max*2)
        #Should be set very high to avoid repeating the cosine
        MODEL_PARAM_DICT['T_max'] = 2000
        
        #sets the minimum allowed learning rate. The scheduler defaults to zero, but I am setting it to a non-zero value.
        MODEL_PARAM_DICT['eta_min'] = MODEL_PARAM_DICT['lr']/1000

        #sets the weight decay for the AdamW optimizer
        MODEL_PARAM_DICT['weight_decay'] = 0#10**(-3)

        #Determines the number of batches accumulated before the gradient is evaluated
        MODEL_PARAM_DICT['ACUM_VAL'] = 10

        #Determines the width of the moving average of the loss
        MODEL_PARAM_DICT['MOV_AV_WIDTH'] = 20

        #sets batch size for training and validation
        MODEL_PARAM_DICT['batch_size'] = 50000
        
        #the columns used in the shard files in the correct order. The ordering for feature_names must match the ordering of the shard files!
        MODEL_PARAM_DICT['shard_columns'] = None
        
        #Save model? Default to true, but turn to false when debugging
        MODEL_PARAM_DICT['SAVE_MODEL'] = True
        
        #Shuffle the ordering of sources in each file before emitting a batch? Applies only to training set.
        MODEL_PARAM_DICT['shuffle_sources_in_file'] = True
        
        #Activates Gaussian Fourier transform of continuous time embedding t. Only works for ADD_MODEL = True
        MODEL_PARAM_DICT['GFTE'] = True
        
        if MODEL_PARAM_DICT['GFTE']:
            assert MODEL_PARAM_DICT['ADD_MODEL'], "Gaussian Fourier Transform Embedding of continuous time variable t only available if ADD_MODEL is used. ADD_MODEL is currently False."
        
        #Sets the size of the GFTE embedding. Does nothing if GFTE is False
        MODEL_PARAM_DICT['GFTE_embed_dim']=64
        
        #Sets the scale of the GFTE embeddings. Does nothing if GFTE is False.
        MODEL_PARAM_DICT['GFTE_scale']=16.0
        
        #Turns the frequencies of the GFTE embedding into weights that are learned by the model. Does nothing if GFTE is False.
        MODEL_PARAM_DICT['GFTE_learnable']=False
        
        #Determines if an exponential moving average of the model weights is constructed. This model version is allegedly more robust. It is saved at the same time as the normal model.
        MODEL_PARAM_DICT['EMA'] = True
        
        #Sets the decay rate of the exponential function for the EMA. Heavily favors the most recent model.
        #Default taken from Song+21.
        MODEL_PARAM_DICT['EMA_decay'] = 0.999
        
        #If true, uses the fixed validation mask sets as mask sets for training epochs
        #If true and MASK_PROB_RAND is true, alternates between fixed mask training epochs and random mask training epochs in equal numbers
        MODEL_PARAM_DICT['CUSTOM_TRAIN_SETS'] = False
        
        
        if MODEL_PARAM_DICT['RANDOM_INTRA_SET_MASK_PROB']:
            assert MODEL_PARAM_DICT['CUSTOM_TRAIN_SETS'] == False, "Cannot have customing training sets if using intra-epoch random probabilities"
            assert MODEL_PARAM_DICT['MASK_PROB_RAND'] == False, "MASK_PROB_RAND sets one input and one output mask prob across the epoch. Cannot be true if using intra-epoch probs."
        elif MODEL_PARAM_DICT['CUSTOM_TRAIN_SETS']:
            assert MODEL_PARAM_DICT['RANDOM_INTRA_SET_MASK_PROB'] == False, "Cannot have random intra-epoch mask probs in training and have custom training sets"
            assert MODEL_PARAM_DICT['MASK_PROB_RAND'] == False, "Cannot have random epoch mask probs in training and have custom training sets"
        elif MODEL_PARAM_DICT['MASK_PROB_RAND']:
            assert MODEL_PARAM_DICT['CUSTOM_TRAIN_SETS'] == False, "Cannot have customing training sets if using intra-epoch random probabilities"
            assert MODEL_PARAM_DICT['RANDOM_INTRA_SET_MASK_PROB'] == False, "RANDOM_INTRA_SET_MASK_PROB sets random input and output masks within a single epoch. Cannot be true if using epoch-wide probs."
        #########################################################################################################################################################
        #Name of the model that is saved/loaded (see above)
        MODEL_NAME = 'POS_PM_mask_set_set' + MODEL_PARAM_DICT['SET_TYPE'] + '_' + MODEL_PARAM_DICT['NORM_METHOD'] + '_' + MODEL_PARAM_DICT['timestamp']
        
        if MODEL_PARAM_DICT['MASK_TRAIN']:
            MODEL_NAME += '_mask_train'
        if MODEL_PARAM_DICT['MASK_VAL']:
            MODEL_NAME += '_mask_val'

        if MODEL_PARAM_DICT['MAE_LOSS']:
            MODEL_NAME += '_MAE'
        if MODEL_PARAM_DICT['MSE_LOSS']:
            MODEL_NAME += '_MSE'
        
        if MODEL_PARAM_DICT['LATENT_FLOW'] == True:
            MODEL_NAME += '_LT_FLOW'
        if MODEL_PARAM_DICT['IOB_METHOD']:
            assert MODEL_PARAM_DICT['SPARSE_AUTOENCODER'] == False, "Cannot use sparse autoencoder and IOB at same time"
            MODEL_NAME += '_IOB'
        elif MODEL_PARAM_DICT['SPARSE_AUTOENCODER']:
            assert MODEL_PARAM_DICT['IOB_METHOD'] == False, "Cannot use sparse autoencoder and IOB at same time"
            MODEL_NAME += '_sparse_k' + str(MODEL_PARAM_DICT['K_LATENT_ALLOWED'])
        if MODEL_PARAM_DICT['BOTTLENECK']:
            MODEL_NAME += '_bneck' + str(MODEL_PARAM_DICT['BOTTLENECK_SIZE'])
        
        if MODEL_PARAM_DICT['SCHEDULER']:
            MODEL_NAME += '_vHD' + str(MODEL_PARAM_DICT['v_hidden_dim']) + '_SIGMA' + decimal_name(MODEL_PARAM_DICT['sigma']) + '_' + MODEL_PARAM_DICT['ACTIVATION_TYPE'] + '_lr_max_' + decimal_name(MODEL_PARAM_DICT['lr']) + '_wd' + decimal_name(MODEL_PARAM_DICT['weight_decay'])
        else:
            MODEL_NAME += '_vHD' + str(MODEL_PARAM_DICT['v_hidden_dim']) + '_SIGMA' + decimal_name(MODEL_PARAM_DICT['sigma']) + '_' + MODEL_PARAM_DICT['ACTIVATION_TYPE'] + '_lr' + decimal_name(MODEL_PARAM_DICT['lr']) + '_wd' + decimal_name(MODEL_PARAM_DICT['weight_decay'])

        if MODEL_PARAM_DICT['BOTTLENECK']:
            MODEL_NAME += '_elayers' + str(MODEL_PARAM_DICT['encoder_layers']) + '_dlayers' + str(MODEL_PARAM_DICT['decoder_layers'])
        else:
            MODEL_NAME += '_layers' + str(MODEL_PARAM_DICT['n_layers'])

        if MODEL_PARAM_DICT['SINE_MASK_PROB'] and (MODEL_PARAM_DICT['MASK_TRAIN'] or MODEL_PARAM_DICT['MASK_VAL']):
            MODEL_NAME += '_SIN_MP_MIN' + decimal_name(MODEL_PARAM_DICT['mask_prob_min']) + '_FREQ' + str(MODEL_PARAM_DICT['mp_sin_freq_int'])
        elif MODEL_PARAM_DICT['MASK_PROB_RAND']:
            MODEL_NAME += '_MP_RAND'
        elif (MODEL_PARAM_DICT['MASK_TRAIN'] or MODEL_PARAM_DICT['MASK_VAL']):
            MODEL_NAME += '_MP' + decimal_name(MODEL_PARAM_DICT['mask_prob'])

        if MODEL_PARAM_DICT['MARGINAL_PREDICTOR']:
            MODEL_NAME += '_MARG'
        
        if MODEL_PARAM_DICT['NEVER_MASK_INPUT']:
            MODEL_NAME += '_NMI'
            for i in MODEL_PARAM_DICT['NEVER_MASK_INPUT_INDICES']:
                MODEL_NAME += '_' + str(i)

        if MODEL_PARAM_DICT['NEVER_MASK_OUTPUT']:
            MODEL_NAME += '_NMO'
            for i in MODEL_PARAM_DICT['NEVER_MASK_OUTPUT_INDICES']:
                MODEL_NAME += '_' + str(i)
            
        if MODEL_PARAM_DICT['CAT_MODEL']:
            MODEL_NAME += '_CAT'
        
        if MODEL_PARAM_DICT['ADD_MODEL']:
            MODEL_NAME += '_ADD'

        if MODEL_PARAM_DICT['SET_VAL']:
            MODEL_NAME += '_SET_VAL'
        if MODEL_PARAM_DICT['CUSTOM_TRAIN_SETS']:
            MODEL_NAME += '_SET_TRAIN'
        
        MODEL_NAME += 'ACUMVAL_' + str(MODEL_PARAM_DICT['ACUM_VAL'])
        
    if is_main and LOAD_MODEL == False:
        np.save('YOUR PATH/model_param_dicts/' + MODEL_NAME + '_model_param_dict.npy', MODEL_PARAM_DICT)
    
        
    if MODEL_PARAM_DICT['READ_TYPE'] == "Shard":
        assert MODEL_PARAM_DICT['batch_size'] is not None, "If using Shard files (i.e., READ_TYPE is Shard), batch_size cannot be None."
        assert MODEL_PARAM_DICT['batch_size'] > 0, f"Batch size must be greater than 0. Currently batch size is {MODEL_PARAM_DICT['batch_size']}."
        assert MODEL_PARAM_DICT['shard_columns'] is not None, "Must enter columns used in Shard files! shard_columns cannot be None."
    if is_main:
        print('##################################################')
        print('##################################################')
        print(MODEL_NAME)
        print('##################################################')
        print('##################################################')

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load updated file that includes P_Z_SCORE normalization constants for SOME parameters
    preprocess = np.load(stats_dir / "P_Z_Norm_Dict_Sample_Ests_062326.npy", allow_pickle = True).item()
    # Load JSON with float/int normalizations
    #with open(stats_dir / "training_stats.json", "r") as f:
    #    preprocess = json.load(f)

    # Load JSON with bool/string normalizations (note that bools and strings in the hdf5 files are saved as ints)
    with open(stats_dir / "training_stats_bool_string_features.json", "r") as f:
        str_bool_preprocess = json.load(f)

    if MODEL_PARAM_DICT['SET_TYPE'] == 'M':
        dict_key_set = 'main_train'
    elif MODEL_PARAM_DICT['SET_TYPE'] == 'A':
        dict_key_set = 'subsetA_train'
    elif MODEL_PARAM_DICT['SET_TYPE'] == 'B':
        dict_key_set = 'subsetB_train'

    #Only save the normalizations of relevant features
    feature_dict = {}
    for feature_name in MODEL_PARAM_DICT['feature_names']:
        if feature_name in preprocess[dict_key_set]:
            feature_dict[feature_name] = preprocess[dict_key_set][feature_name]
        elif feature_name in str_bool_preprocess[dict_key_set]:
            feature_dict[feature_name] = str_bool_preprocess[dict_key_set][feature_name]
        elif is_main:
            print(f"Skipping {feature_name}")

    feature_names = list(feature_dict.keys())

    dim = len(feature_names)
    if feature_names[0] == 'l' and MODEL_PARAM_DICT['NORM_METHOD'] == 'CUSTOM':
        L_CUSTOM = True
        dim += 1
    else:
        L_CUSTOM = False

    #Initialize Model
    if MODEL_PARAM_DICT['LATENT_FLOW'] and MODEL_PARAM_DICT['AUTOENCODER_TRAINING']:
        model = CFM_lib.Autoencoder_Model(input_dim = 2*dim, output_dim = dim, hidden_dim = ['v_hidden_dim'], encoder_layers = MODEL_PARAM_DICT['encoder_layers'], decoder_layers = MODEL_PARAM_DICT['decoder_layers'], bottleneck = MODEL_PARAM_DICT['BOTTLENECK_SIZE'], ACTIVATION_TYPE = MODEL_PARAM_DICT['ACTIVATION_TYPE'], FINAL_ACTIVATION = MODEL_PARAM_DICT['FINAL_ACTIVATION'], IOB_METHOD = MODEL_PARAM_DICT['IOB_METHOD'], SPARSE_AUTOENCODER = MODEL_PARAM_DICT['SPARSE_AUTOENCODER'], K_LATENT_ALLOWED = MODEL_PARAM_DICT['K_LATENT_ALLOWED']).to(device)
    elif MODEL_PARAM_DICT['LATENT_FLOW'] and MODEL_PARAM_DICT['LATENT_FLOW_TRAINING']:
        #Enter name of autoencoder here!
        MODEL_PARAM_DICT['autoencoder_model'] = CFM_lib.Autoencoder_Model(input_dim = 2*dim, output_dim = dim, hidden_dim = 1024, encoder_layers = 2, decoder_layers = 2, bottleneck = MODEL_PARAM_DICT['BOTTLENECK_SIZE'], ACTIVATION_TYPE = MODEL_PARAM_DICT['ACTIVATION_TYPE'], FINAL_ACTIVATION = MODEL_PARAM_DICT['FINAL_ACTIVATION'], IOB_METHOD = MODEL_PARAM_DICT['IOB_METHOD'], SPARSE_AUTOENCODER = MODEL_PARAM_DICT['SPARSE_AUTOENCODER'], K_LATENT_ALLOWED = MODEL_PARAM_DICT['K_LATENT_ALLOWED']).to(device)
        MODEL_PARAM_DICT['autoencoder_model'].load_state_dict(torch.load('YOUR PATH/saved_models/' + MODEL_PARAM_DICT['AUTOENCODER_MODEL_NAME'] + '.pth', weights_only=True))
        #latent_space model takes x_t, input_embedding, t, and output_mask
        model = CFM_lib.DenseNeuralNet(input_dim = 2*MODEL_PARAM_DICT['BOTTLENECK_SIZE'] + dim + 1, hidden_dim = MODEL_PARAM_DICT['v_hidden_dim'], n_layers = MODEL_PARAM_DICT['n_layers'], ACTIVATION_TYPE = MODEL_PARAM_DICT['ACTIVATION_TYPE'], output_dim = MODEL_PARAM_DICT['BOTTLENECK_SIZE'], final_act = MODEL_PARAM_DICT['FINAL_ACTIVATION']).to(device)
    elif MODEL_PARAM_DICT['BOTTLENECK']:
        model = CFM_lib.ConditionalFlowModel_Bottleneck(input_dim = 4*dim+1, output_dim = dim, hidden_dim = MODEL_PARAM_DICT['v_hidden_dim'], encoder_layers = MODEL_PARAM_DICT['encoder_layers'], decoder_layers = MODEL_PARAM_DICT['decoder_layers'], bottleneck = MODEL_PARAM_DICT['BOTTLENECK_SIZE'], ACTIVATION_TYPE = MODEL_PARAM_DICT['ACTIVATION_TYPE'], FINAL_ACTIVATION = MODEL_PARAM_DICT['FINAL_ACTIVATION'], IOB_METHOD = MODEL_PARAM_DICT['IOB_METHOD'], SPARSE_AUTOENCODER = MODEL_PARAM_DICT['SPARSE_AUTOENCODER'], K_LATENT_ALLOWED = MODEL_PARAM_DICT['K_LATENT_ALLOWED'], ADD_INFO = MODEL_PARAM_DICT['MARGINAL_PREDICTOR']).to(device)
    elif MODEL_PARAM_DICT['CAT_MODEL']:
        model = CFM_lib.CatNet(input_dim = 4*dim+1, output_dim = dim, hidden_dim = MODEL_PARAM_DICT['v_hidden_dim'], n_layers = MODEL_PARAM_DICT['n_layers'], ACTIVATION_TYPE = MODEL_PARAM_DICT['ACTIVATION_TYPE'], FINAL_ACTIVATION = MODEL_PARAM_DICT['FINAL_ACTIVATION'], ADD_INFO = MODEL_PARAM_DICT['MARGINAL_PREDICTOR'], INCLUDE_OUTPUT_MASK = True, LAYER_NORM = MODEL_PARAM_DICT['LAYER_NORM']).to(device)
    elif MODEL_PARAM_DICT['ADD_MODEL']:
        model = CFM_lib.AddNet(input_dim = 4*dim+1, output_dim = dim, hidden_dim = MODEL_PARAM_DICT['v_hidden_dim'], n_layers = MODEL_PARAM_DICT['n_layers'], ACTIVATION_TYPE = MODEL_PARAM_DICT['ACTIVATION_TYPE'], FINAL_ACTIVATION = MODEL_PARAM_DICT['FINAL_ACTIVATION'], ADD_INFO = MODEL_PARAM_DICT['MARGINAL_PREDICTOR'], INCLUDE_OUTPUT_MASK = True, LAYER_NORM = MODEL_PARAM_DICT['LAYER_NORM'],GFTE = MODEL_PARAM_DICT['GFTE'], GFTE_embed_dim=MODEL_PARAM_DICT['GFTE_embed_dim'], GFTE_scale=MODEL_PARAM_DICT['GFTE_scale'], GFTE_learnable=MODEL_PARAM_DICT['GFTE_learnable']).to(device)
    else:
        model = CFM_lib.ConditionalFlowModel_Flow_Only(input_dim = 4*dim+1, output_dim = dim, hidden_dim = MODEL_PARAM_DICT['v_hidden_dim'], n_layers = MODEL_PARAM_DICT['n_layers'], ACTIVATION_TYPE = MODEL_PARAM_DICT['ACTIVATION_TYPE'], FINAL_ACTIVATION = MODEL_PARAM_DICT['FINAL_ACTIVATION'], ADD_INFO = MODEL_PARAM_DICT['MARGINAL_PREDICTOR'], INCLUDE_OUTPUT_MASK = True).to(device)

    #Load model state if using pre-existing model
    
    
    
    if LOAD_MODEL:
        state = torch.load(filepath, map_location=device)
        model.load_state_dict(state)
        
        
    #Allocate model to GPUs
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if MODEL_PARAM_DICT['EMA']:
        if LOAD_MODEL and os.path.exists(ema_path):
            ema_path = save_directory + LOAD_NAME + '_ema.pth'
            ema = torch.load(ema_path, map_location=device)
        else:
            ema = {k: v.detach().clone() for k, v in model.module.state_dict().items()}
    else:
        ema = 0

    #Initialize Optimizer
    optimizer = optim.AdamW(model.parameters(), lr = MODEL_PARAM_DICT['lr'], weight_decay = MODEL_PARAM_DICT['weight_decay'])

    #Initialize Scheduler if using
    if MODEL_PARAM_DICT['SCHEDULER'] and MODEL_PARAM_DICT['SCHEDULER_TYPE'] == 'Cosine Annealing Warm Restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = MODEL_PARAM_DICT['T_0'])
    elif MODEL_PARAM_DICT['SCHEDULER'] and MODEL_PARAM_DICT['SCHEDULER_TYPE'] == 'Cosine Annealing LR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = MODEL_PARAM_DICT['T_max'], eta_min = MODEL_PARAM_DICT['eta_min'])
    else:
        scheduler = 0
    
    #Load optimizer state (and scheduler state if using) if using pre-existing model
    if LOAD_MODEL:
        optimizer_state = torch.load('YOUR PATH/saved_models/' + LOAD_NAME + '_optimizer.pt', map_location=device)
        optimizer.load_state_dict(optimizer_state)
        if MODEL_PARAM_DICT['SCHEDULER'] and NEW_SCHEDULER == False:
            scheduler_state = torch.load('YOUR PATH/saved_models/' + LOAD_NAME + '_scheduler.pt', map_location=device)
            scheduler.load_state_dict(scheduler_state)
        
    hdf5_filelist = np.loadtxt('YOUR PATH/hdf5_filelist.txt', dtype = str)
        
    start_train_ds = time.time()

    train_ds = HDF5IterableDataset(
        READ_TYPE = MODEL_PARAM_DICT['READ_TYPE'],
        filelist = hdf5_filelist,
        split = "train",
        set_type = MODEL_PARAM_DICT['SET_TYPE'],
        columns = feature_names,
        shard_columns = MODEL_PARAM_DICT['shard_columns'],
        batch_size = MODEL_PARAM_DICT['batch_size'],
        shuffle_order_each_epoch = True,
        shuffle_sources_in_file = MODEL_PARAM_DICT['shuffle_sources_in_file'], 
        drop_last = True,
        normalization = feature_dict,
        norm_method = MODEL_PARAM_DICT['NORM_METHOD'],
        equalize_batches_across_ranks = True,
        num_workers_hint = MODEL_PARAM_DICT['N_WORKERS'],
    )

    if is_main:
        print("Created train ds", time.time() - start_train_ds)
    start_val_ds = time.time()


    val_ds = HDF5IterableDataset(
        READ_TYPE = MODEL_PARAM_DICT['READ_TYPE'],
        filelist = hdf5_filelist,
        split = "val",
        set_type = MODEL_PARAM_DICT['SET_TYPE'],
        columns = feature_names,
        shard_columns = MODEL_PARAM_DICT['shard_columns'],
        batch_size = MODEL_PARAM_DICT['batch_size'],
        shuffle_order_each_epoch = True,
        shuffle_sources_in_file = False,
        drop_last = False,
        normalization = feature_dict,
        norm_method = MODEL_PARAM_DICT['NORM_METHOD'],
        equalize_batches_across_ranks = True,
        num_workers_hint = MODEL_PARAM_DICT['N_WORKERS'],
    )

    if is_main:
        print("Created val ds", time.time() - start_val_ds)
    start_train_loader = time.time()

    # DataLoader knobs for throughput on CPU->GPU pipelines
    train_loader = DataLoader(
        train_ds,
        num_workers=MODEL_PARAM_DICT['N_WORKERS'],               # start with 4-16 depending on CPU cores
        pin_memory=True,             # enable faster H2D copies
        drop_last=False,              # often good for training
        persistent_workers=False,    # MUST be False: workers re-fork each epoch so they pick up the latest set_epoch() plan from the main process
        batch_size = None #must be none for the loader
    )
    
    if is_main:
        print("Created train dataloader", time.time() - start_train_loader)

    start_val_loader = time.time()

    val_loader = DataLoader(
        val_ds,
        num_workers=MODEL_PARAM_DICT['N_WORKERS'],
        pin_memory=True,     # keep workers alive between epochs
        drop_last=False,
        persistent_workers=False,    # MUST be False: see train_loader note above
        batch_size = None #Must be none for the loader
    )
     
    if is_main:
        print("Created val dataloader", time.time() - start_val_loader)

    if LOAD_MODEL:
        loss_dict = np.load('YOUR PATH/loss_data/' + MODEL_NAME + '_loss_data.npy', allow_pickle = True).item()
        if LOAD_MODEL and MODIFY_LOAD:
            for i in range(len(MODEL_PARAM_KEYS)):
                key = MODEL_PARAM_KEYS[i]
                #Add epoch at which old value ceased to dictionary.
                MODEL_PARAM_DICT['UPDATED_PARAMS'][key]['EPOCHS'].append(len(loss_dict['train']))
    else:
        loss_dict = {}
        
    try:
        CFM_train.train(MODEL_NAME + MODEL_PARAM_DICT['NAME_MOD'],
                model,
                device,
                rank,
                is_main,
                dim,
                train_loader,
                val_loader,
                optimizer,
                MODEL_PARAM_DICT['MASK_TRAIN'],
                MODEL_PARAM_DICT['MASK_VAL'],
                LOAD_MODEL,
                loss_dict,
                num_epochs=num_epochs,
                stopping_num = 1000,
                min_epochs = 1,
                save_directory = save_directory,
                loss_directory = 'YOUR PATH/loss_data/',
                mask_prob = MODEL_PARAM_DICT['mask_prob'],
                SINE_MASK_PROB = MODEL_PARAM_DICT['SINE_MASK_PROB'],
                OPPOSITE_OUTPUT_MASK = MODEL_PARAM_DICT['OPPOSITE_OUTPUT_MASK'],
                LATENT_FLOW = MODEL_PARAM_DICT['LATENT_FLOW'],
                VAL_M_FRAC = MODEL_PARAM_DICT['VAL_M_FRAC'],
                AUTOENCODER_TRAINING = MODEL_PARAM_DICT['AUTOENCODER_TRAINING'],
                LATENT_FLOW_TRAINING = MODEL_PARAM_DICT['LATENT_FLOW_TRAINING'],
                mp_sin_freq = MODEL_PARAM_DICT['mp_sin_freq'],
                mask_prob_min = MODEL_PARAM_DICT['mask_prob_min'],
                autoencoder_model = MODEL_PARAM_DICT['autoencoder_model'],
                sigma = MODEL_PARAM_DICT['sigma'],
                mask_prob_step = MODEL_PARAM_DICT['mask_prob_step'],
                DEBUG_VAL = MODEL_PARAM_DICT['DEBUG_VAL'],
                DEBUG_TRAIN = MODEL_PARAM_DICT['DEBUG_TRAIN'],
                MAE_LOSS = MODEL_PARAM_DICT['MAE_LOSS'],
                MSE_LOSS = MODEL_PARAM_DICT['MSE_LOSS'],
                NEVER_MASK_INPUT = MODEL_PARAM_DICT['NEVER_MASK_INPUT'],
                NEVER_MASK_INPUT_INDICES = MODEL_PARAM_DICT['NEVER_MASK_INPUT_INDICES'],
                NEVER_MASK_OUTPUT = MODEL_PARAM_DICT['NEVER_MASK_OUTPUT'],
                NEVER_MASK_OUTPUT_INDICES = MODEL_PARAM_DICT['NEVER_MASK_OUTPUT_INDICES'],
                MASK_PROB_RAND = MODEL_PARAM_DICT['MASK_PROB_RAND'],
                MASK_SETS = MODEL_PARAM_DICT['MASK_SETS'],
                MASK_SETS_INDICES = MODEL_PARAM_DICT['MASK_SETS_INDICES'],
                L_CUSTOM = L_CUSTOM,
                ACUM_VAL = MODEL_PARAM_DICT['ACUM_VAL'],
                scheduler = scheduler,
                SCHEDULER = MODEL_PARAM_DICT['SCHEDULER'],
                moving_average_loss_window_width = MODEL_PARAM_DICT['MOV_AV_WIDTH'],
                input_mask_prob_pwr = MODEL_PARAM_DICT['MASK_PROB_INPUT_PWR'],
                output_mask_prob_pwr = MODEL_PARAM_DICT['MASK_PROB_OUTPUT_PWR'],
                SAVE_MODEL = MODEL_PARAM_DICT['SAVE_MODEL'],
                MODEL_NAME_UNMOD = MODEL_NAME,
                MODIFY_LOAD = MODIFY_LOAD,
                code_start = code_start,
                MODEL_PARAM_DICT = MODEL_PARAM_DICT,
                EMA = MODEL_PARAM_DICT['EMA'],
                ema = ema,
                ema_decay = MODEL_PARAM_DICT['EMA_decay'],
                SET_VAL = MODEL_PARAM_DICT['SET_VAL'],
                custom_mask_dict = MODEL_PARAM_DICT['CUSTOM_MASK_DICT'],
                FIXED_TRAINING_MASKS = MODEL_PARAM_DICT['CUSTOM_TRAIN_SETS'],
                INTRA_MASK_RAND_PROB = MODEL_PARAM_DICT['RANDOM_INTRA_SET_MASK_PROB']
                )

        if is_main:
            print("Training Complete")

        if is_main:
            print('##################################################')
            print('##################################################')
            print(MODEL_NAME)
            print('##################################################')
            print('##################################################')
    
    finally:
        cleanup_ddp()
    
if __name__ == '__main__':
    main()
