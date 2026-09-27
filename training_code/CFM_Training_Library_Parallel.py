import torch
import numpy as np
import time
import contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
rng = np.random.default_rng()

"""
ML Model Training Epoch
Trains the model over a single epoch
"""

class FormatError(Exception):
    """Exception raised for normalization."""
    def __init__(self, message="A id shard loading error occurred."):
        self.message = message
        super().__init__(self.message)
        
def _as_prob_operand(p, device):
    """
    Normalize a masking probability for use in `torch.rand((B, dim)) > p`.
 
    Scalars (python float, numpy scalar, NaN) pass through unchanged --
    the pre-existing epoch-level behaviour.  Per-source arrays of shape
    (B,) (the INTRA_MASK_RAND_PROB case) are converted to a float32
    torch column vector (B, 1) ON THE TARGET DEVICE, so that:
      * the comparison broadcasts row-wise (each source gets its own
        probability across all dim features) instead of raising
        'size of tensor a (dim) must match size of tensor b (B)';
      * a CUDA rand tensor is never compared against a CPU numpy array,
        which raises a device-mismatch RuntimeError.
    """
    if p is None or np.isscalar(p):
        return p
    if isinstance(p, np.ndarray) and p.ndim == 0:
        return float(p)
    t = torch.as_tensor(p, dtype=torch.float32, device=device)
    return t.view(-1, 1)
        
def mask_generator(MASKING, AUTOENCODER_TRAINING, batch_length, dim, device, initial_mask, mask_prob, output_mask_prob, NEVER_MASK_INPUT, NEVER_MASK_INPUT_INDICES, NEVER_MASK_OUTPUT, NEVER_MASK_OUTPUT_INDICES, MASK_SETS, MASK_SETS_INDICES):
    #The masking procedure is done below
    #Per-source probability arrays (INTRA_MASK_RAND_PROB) must be torch
    #column vectors on the right device; scalars pass through unchanged.
    mask_prob = _as_prob_operand(mask_prob, device)
    output_mask_prob = _as_prob_operand(output_mask_prob, device)
    #input mask is the mask that determines which features are used as conditional features
    #output mask is the mask that determines which features are used for model training
    #If masking is false, default to the initial mask for both the input mask and output mask
    if MASKING and AUTOENCODER_TRAINING:
        rand_in = torch.rand((batch_length, dim), device = device)
        input_mask = initial_mask & (rand_in > mask_prob)
    elif MASKING and output_mask_prob is not None:
        rand_in = torch.rand((batch_length, dim), device = device)
        rand_out = torch.rand((batch_length, dim), device = device)
        input_mask = initial_mask & (rand_in > mask_prob)
        output_mask = initial_mask & (rand_out > output_mask_prob)
    elif MASKING:
        rand_in = torch.rand((batch_length, dim), device = device)
        rand_out = torch.rand((batch_length, dim), device = device)
        input_mask = initial_mask & (rand_in > mask_prob)
        output_mask = initial_mask & (rand_out > mask_prob)
    else:
        input_mask = initial_mask.clone()
        output_mask = initial_mask.clone()
    
    #To get the correct masking probabilities for the set, I make all mask values in the set match those of the first index in the set. The only except is if a parameter is masked in the initial mask.
    if MASK_SETS:
        if AUTOENCODER_TRAINING:
            for set_i in MASK_SETS_INDICES:
                for k in set_i:
                    input_mask[:,k] = initial_mask[:,k] & input_mask[:,set_i[0]]
        else:
            for set_i in MASK_SETS_INDICES:
                for k in set_i:
                    input_mask[:,k] = initial_mask[:,k] & input_mask[:,set_i[0]]
                    output_mask[:,k] = initial_mask[:,k] & output_mask[:,set_i[0]]
                
    #If NEVER_MASK is True, always include the desired features in the input and output
    if NEVER_MASK_INPUT:
        for i in NEVER_MASK_INPUT_INDICES:
            input_mask[:,i] = initial_mask[:,i]
    
    if NEVER_MASK_OUTPUT:
        if AUTOENCODER_TRAINING == False:
            for i in NEVER_MASK_OUTPUT_INDICES:
                output_mask[:,i] = initial_mask[:,i]
                
    return input_mask, output_mask

def train_epoch(loader, #Pytorch DataLoader containing training data
    model, #ML model being trained
    device,
    dim,
    optimizer, #optimizer for chosen model
    MASKING, #If true, applies random masking on top of Gaia-intrinsic masking. So a random (independent) mask is applied to both the model input and to which features are relevant to the model output. The masking is independent source to source, and epoch to epoch, so over multiple rounds of training the model should learn many different conditional relationships between features
    mask_prob, #Sets the masking probability. Note that the probability of a feature being masked is actually 1 - mask_prob
    SINE_MASK_PROB, #If true, a sine scheduler is used to assign mask probabilities to sources. The mask probability is updated batch to batch, and the batch order is shuffled epoch to epoch. After mask_prob is calculate the procedure is identical as the default case of using a constant mask prob. This method was developed in order vary the scale of input to output conditional relationships seen by the model.
    OPPOSITE_OUTPUT_MASK, #If true, the effective mask_prob for output mask is 1 - mask_prob. In practice this means that if it is likely many features are unmasked in the input, it is more likely many features are masked in the output.
    LATENT_FLOW, #If true, a CFM model where the CFM applies only to the latent space is used.
    AUTOENCODER_TRAINING, #If true, the non-CFM autoencoder wrapper for the CFM latent model is trained
    LATENT_FLOW_TRAINING, #If true, the latent space CFM model variant is trained. Requires that LATENT_FLOW = True and AUTOENCODER = True, see train() function.
    mp_sin_freq, #Sets the frequency of the sine mask probability scheduler
    mask_prob_min, #Sets the minimum possible masking probability when using the sine scheduler
    autoencoder_model, #Pytorch model of autoencoder. Only relevant if LATENT_FLOW_TRAINING == True.
    sigma, #used for CFM
    MAE_LOSS,
    MSE_LOSS,
    NEVER_MASK_INPUT,
    NEVER_MASK_INPUT_INDICES,
    NEVER_MASK_OUTPUT,
    NEVER_MASK_OUTPUT_INDICES,
    DEBUG_TRAIN = False,
    output_mask_prob = None,
    L_CUSTOM = False,
    MASK_SETS = False,
    MASK_SETS_INDICES = [(0,1)],
    accumulation_val = 10,
    scheduler = 0,
    SCHEDULER = False,
    EMA = False,
    ema = 0,
    ema_decay = 0,
    is_main = False,
    FIXED_MASK_SET = False,
    INTRA_MASK_RAND_PROB = False,
    input_mask_prob_pwr = 0,
    output_mask_prob_pwr = 0,
    custom_mask = 0,
    custom_mask_name = ''):
                
    model.train()
    #if training the latent CFM, make sure to freeze the encoder.
    if LATENT_FLOW_TRAINING:
        autoencoder_model.eval()
        
    total = 0.0
    MAE_total = 0.0
    MSE_total = 0.0
    dataset_length = 0.0
    batch_counter = 0
    time_dict = {}
    time_dict['sine'] = 0
    time_dict['squeeze'] = 0
    time_dict['input_output'] = 0
    time_dict['output_row'] = 0
    time_dict['CFM'] = 0
    time_dict['loss'] = 0
    time_dict['loop'] = 0
    time_dict['loader'] = 0
    time_dict['total'] = 0
    time_dict['max_batch'] = 0
    #initialize here so that we can get the first loader time too
    loop_finish = time.time()
    ema_time = 0
    r = dist.get_rank() if dist.is_initialized() else 0
    #print(f"Rank {r}: per-worker target_batches = {loader.dataset._epoch_target_batches}, "
    #  f"num_workers = {loader.dataset.num_workers_hint}", flush=True)
    for x_data, initial_mask in loader:
        with torch.no_grad():
            start_loop = time.time()
            
            #Squeezing is necessary because I am defaulting to an emit batches style output for my datasets
            #x_data contains the normalized gaia data
            x_data = x_data.to(device, non_blocking=True).squeeze(0)
            #retrieve the initial mask. This contains the intrinsic masking (i.e., if this feature has a non-nan entry in DR3 of the Gaia Source Catalog)
            #should be a bool already
            initial_mask = initial_mask.to(device, non_blocking=True, dtype = torch.bool).squeeze(0)
            
            #create a variable for the batch length. Note that when reading directly from Gaia Source hdf5's the dataset reader defaults to a variable batch length!
            batch_length = x_data.shape[0]
            squeeze_time = time.time()
            
            #If true, use a fixed mask for the variables instead of a random one. Else, use a mask probability.
            if FIXED_MASK_SET:
                #Use only selected input features (that are available)
                input_mask = torch.from_numpy(custom_mask).to(device, dtype = torch.bool).unsqueeze(0) & initial_mask
                #Use only features that exist and are masked
                #For the autoencoder case, use all output features, since none are masked
                if custom_mask_name == 'autoencoder':
                    output_mask = initial_mask
                else:
                    output_mask = (~torch.from_numpy(custom_mask).to(device, dtype = torch.bool).unsqueeze(0)) & initial_mask
                sin_mask_time = time.time()
            else:
                #Sine mask probability scheduler. Oscillates from -mask_prob_min to 1-mask_prob_min
                #Note masking assignment method below. The low values here imply less masking, high values imply more masking!
                #Intra mask random prob follows same rules as full epoch random prob, but draws the probabilities from the sources themselves --> hopefully this makes epochs more similar and the loss landscape less noisy.
                if SINE_MASK_PROB and MASKING:
                    mask_prob = np.sin(mp_sin_freq * batch_counter)**2 - mask_prob_min
                elif INTRA_MASK_RAND_PROB:
                    mask_prob = rng.uniform(size = batch_length)**input_mask_prob_pwr
                if OPPOSITE_OUTPUT_MASK:
                    #The minimum masking probability should apply to both the input and output mask. This guarantees that.
                    output_mask_prob = (1 - mask_prob) - (2*mask_prob_min)
                elif INTRA_MASK_RAND_PROB:
                    output_mask_prob = rng.uniform(size = batch_length)**output_mask_prob_pwr
                    
                sin_mask_time = time.time()

                input_mask, output_mask = mask_generator(MASKING, AUTOENCODER_TRAINING, batch_length, dim, device, initial_mask, mask_prob, output_mask_prob, NEVER_MASK_INPUT, NEVER_MASK_INPUT_INDICES, NEVER_MASK_OUTPUT, NEVER_MASK_OUTPUT_INDICES, MASK_SETS, MASK_SETS_INDICES)
            input_output_time = time.time()
            
            #The following procedure checks if there are any sources with a totally masked output. If so, find one valid feature (i.e., not masked by initial_mask) and unmask it.
            #This prevents nan values while training.
            #No analoguous procedure is needed for the input mask. In fact complete masking of the conditional variables is sometimes desirable!
        
            row_has_true = output_mask.any(dim=1)
            bad_rows = (~row_has_true).nonzero(as_tuple=True)[0]

            if bad_rows.numel() > 0:
                valid = initial_mask[bad_rows]   # (B, C) bool

                # Guard rows with at least one valid feature
                num_valid = valid.sum(dim=1)
                rows_with_valid = (num_valid > 0).nonzero(as_tuple=True)[0]
                if rows_with_valid.numel() > 0:
                    v = valid[rows_with_valid]   # (Bv, C)

                    # Assign random scores to valid positions; invalid get -inf
                    scores = torch.rand(v.shape, device=v.device)
                    scores[~v] = float('-inf')   # exclude invalid features from selection

                    # Pick the argmax per row -> uniform among valid features
                    idx = scores.argmax(dim=1)   # (Bv,)

                    rows_to_update = bad_rows[rows_with_valid]
                    output_mask[rows_to_update, idx] = True
                    
                rows_without_valid = (num_valid == 0).nonzero(as_tuple=True)[0]
                if rows_without_valid.numel() > 0:
                    # Indices into the FULL batch that need to be dropped:
                    rows_to_drop = bad_rows[rows_without_valid]
                    keep = torch.ones(batch_length, dtype=torch.bool, device=device)
                    keep[rows_to_drop] = False
                    input_mask  = input_mask[keep]
                    output_mask = output_mask[keep]
                    x_data      = x_data[keep]
                    batch_length = x_data.shape[0]

            #del initial_mask, row_has_true
            output_row_time = time.time()
        
        #The training procedure is done below
        if LATENT_FLOW == False:
            with torch.no_grad():
                #CFM model training
                
                #x0 is the sample from the base distribution (i.e., a Normal distribution)
                x0 = torch.randn(batch_length, dim, device=device)
                #t is the time variable, which sets where in the transform from x0 to x_data (x1) you are
                t  = torch.rand(batch_length, 1,    device=device)
                mu = x0 + (t*(x_data - x0))
                #x_t is the mid transition of the variables from x0 to x1. It, along with t, form the core inputs of the model, while the masked data and masks serve as conditional variables
                x_t = torch.normal(mean = mu, std = sigma)
                #multiply x_t by the output mask
                x_t = x_t.mul(output_mask)
                #del mu
                #create x_data*input_mask var
                x_in = x_data.mul(input_mask)
                
                #u_cond is the target velocity that we are matching to
                u_cond = x_data - x0
                #del x0

            #v_theta is the model output, i.e., the velocity as predicted by the model
            v_theta = model(t, x_t, output_mask, x_in, input_mask)
            #del t, x_t, input_mask, x_data

            #the loss is just MSE of v_theta and u_cond, with a masking (output_mask) applied
            #v_theta - u_cond
            diff = v_theta.sub(u_cond)
            MSE = diff.square().masked_select(output_mask).mean()
            MAE = diff.abs().masked_select(output_mask).mean()
            #del x0, t, mu, x_in, input_mask, x_data, output_mask, v_theta, u_cond, diff
            
        elif AUTOENCODER_TRAINING:
            with torch.no_grad():
                x_in = x_data.mul(input_mask)
            #If training the autoencoder, no CFM procedure is needed. Just take the masked input data and the input mask as inputs to the model and predict the input_mask*x_data. Use MSE.
            #Note that previously I was going from (input_mask*data, input_mask, output_mask) to (output_mask*data), but this removes the need for a CFM model at all. Still, the loss was low, so it might be worth doing for comparison in the future.
            output = model(x_in, input_mask)
            diff = output.sub(x_data)
            MSE = diff.square().masked_select(input_mask).mean()
            MAE = diff.abs().masked_select(input_mask).mean()

        elif LATENT_FLOW_TRAINING:
            #Special CFM training procedure for the latent flow model. Identical to the normal CFM, except that CFM occurs in an embedded space. There is no masking applied to the embedding
            
            #remove gradients from the pre-trained encoder model
            with torch.no_grad():
                #calculate the embedding for the autoencoder model
                em_in_in = torch.cat([input_mask*x_data, input_mask], dim = 1)
                em_in_out = torch.cat([output_mask*x_data, output_mask], dim = 1)
                #input embedding is the context for the latent space model
                input_embedding = autoencoder_model.Encoder(em_in_in)
                #output embedding is what shapes the CFM
                target_embedding = autoencoder_model.Encoder(em_in_out)
                #del input_mask, x_data
                
                em_shape = input_embedding.shape[1]
                #CFM procedure is now done in the embedding space
                x0 = torch.randn(batch_length, em_shape, device=device)
                t  = torch.rand(batch_length, 1,    device=device)
                mu = x0 + (t*(target_embedding-x0))
                x_t = torch.normal(mean = mu, std = sigma)
                #del mu
                #We want the flow to go from the base distribution to the target embedding!
                u_cond = target_embedding - x0
                #del x0, target_embedding
            
            #note that the model is acting on the embeddings from the encoder. Note also that concatenation most occur before model input
            out_mask_float = output_mask.float()
            input_cat = torch.cat([t, x_t, input_embedding, out_mask_float], dim = 1)
            v_theta = model(input_cat)

            MSE = torch.mean(torch.square(v_theta - u_cond))
            MAE = torch.mean(torch.abs(v_theta - u_cond))
            #loss = torch.mean(torch.square(v_theta - u_cond))

            #del v_theta, u_cond, input_embedding, output_mask, x0, target_embedding, mu, out_mask_float, em_in_in, em_in_out, em_shape,  t, x_t, input_cat
            
        CFM_time = time.time()
        
        #Need to divide loss by number of accumulated batches
        if MAE_LOSS and MSE_LOSS:
            loss = (MAE + MSE)/accumulation_val
        elif MAE_LOSS:
            loss = MAE/accumulation_val
        elif MSE_LOSS:
            loss = MSE/accumulation_val
            
        if (batch_counter + 1) % accumulation_val == 0:
            # Normal backward: all-reduce fires, then step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #constructs the EMA.
            if EMA:
                ema_start_time = time.time()
                with torch.no_grad():
                    for k, v in model.module.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(ema_decay).add_(v.detach(), alpha = 1 - ema_decay)
                ema_time += time.time() - ema_start_time
        else:
            # Suppress gradient sync until the update step
            ctx = model.no_sync() if hasattr(model, 'no_sync') else contextlib.nullcontext()
            with ctx:
                loss.backward()
        loss_time = time.time()
        
        total += loss.item() * accumulation_val * batch_length
        MAE_total += MAE.item() * batch_length
        MSE_total += MSE.item() * batch_length
        dataset_length += batch_length
        batch_counter += 1
        if DEBUG_TRAIN:
            time_dict['sine'] += sin_mask_time - start_loop
            time_dict['squeeze'] += squeeze_time - sin_mask_time
            time_dict['input_output'] += input_output_time - squeeze_time
            time_dict['output_row'] += output_row_time - input_output_time
            time_dict['CFM'] += CFM_time - output_row_time
            time_dict['loss'] += loss_time - CFM_time
            time_dict['loop'] += loss_time - start_loop
            time_dict['loader'] += abs(loop_finish - start_loop)
            total_i = loop_finish
            loop_finish = time.time()
            time_dict['total'] += loop_finish - total_i
            if time_dict['max_batch'] < batch_length:
                time_dict['max_batch'] = batch_length
    
    #expected_per_rank = loader.dataset._epoch_target_batches * loader.dataset.num_workers_hint
    #print(f"Rank {r}: actually processed {batch_counter} batches "
    #      f"(expected {expected_per_rank} = {loader.dataset.num_workers_hint} workers "
    #      f"x {loader.dataset._epoch_target_batches} per-worker target)", flush=True)
    if DEBUG_TRAIN and is_main:
        print(f"Max batch length {time_dict['max_batch']}")
        print(f"{batch_counter} loops\n Average total time {time_dict['total']/batch_counter}\n average loop time {time_dict['loop']/batch_counter}\n average sine masking time {time_dict['sine']/batch_counter}\n average squeeze time {time_dict['squeeze']/batch_counter}\n average input & output mask time {time_dict['input_output']/batch_counter}\n average output row check time {time_dict['output_row']/batch_counter}\n average CFM time {time_dict['CFM']/batch_counter}\n average loss time {time_dict['loss']/batch_counter}\n average loader time {time_dict['loader']/batch_counter}")
        
    if dist.is_available() and dist.is_initialized():
        loss_tensor = torch.tensor(
            [total, MAE_total, MSE_total, dataset_length],
            dtype=torch.float64, device=device
        )
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total        = loss_tensor[0].item()
        MAE_total    = loss_tensor[1].item()
        MSE_total    = loss_tensor[2].item()
        dataset_length = loss_tensor[3].item()
    
    if SCHEDULER:
        scheduler.step()
        
    if EMA and is_main:
        print(f"EMA Time Per Epoch: {ema_time}")

    return total / dataset_length, MAE_total / dataset_length, MSE_total / dataset_length

"""
Calculates loss of ML model over single epoch without training
"""
def validate_epoch(loader, #Pytorch DataLoader containing validation or testing data
    model, #ML model being trained
    device,
    dim,
    MASKING, #Controls masking, see train_epoch() for more information
    mask_prob, #Controls masking, see train_epoch() for more information
    SINE_MASK_PROB, #Controls masking, see train_epoch() for more information
    OPPOSITE_OUTPUT_MASK, #Controls masking, see train_epoch() for more information
    LATENT_FLOW, #Controls model variant specific training procedure, see train_epoch() for more information
    AUTOENCODER_TRAINING, #Controls model variant specific training procedure, see train_epoch() for more information
    LATENT_FLOW_TRAINING, #Controls model variant specific training procedure, see train_epoch() for more information
    VAL_M_FRAC, #Determines how many embedding nodes are available if using IOB method. If none, but using IOB method (i.e., IOB_METHOD = True in model instantiation), defaults to randomly selecting a fraction of the nodes to keep open.
    mp_sin_freq, #Sets the frequency of the sine mask probability scheduler
    mask_prob_min, #Sets the minimum possible masking probability when using the sine scheduler
    autoencoder_model, #Pytorch model of autoencoder. Only relevant if LATENT_FLOW_TRAINING == True.
    sigma, #used for CFM
    MAE_LOSS,
    MSE_LOSS,
    NEVER_MASK_INPUT,
    NEVER_MASK_INPUT_INDICES,
    NEVER_MASK_OUTPUT,
    NEVER_MASK_OUTPUT_INDICES,
    DEBUG_VAL = False,
    output_mask_prob = None,
    L_CUSTOM = False,
    MASK_SETS = False,
    INTRA_MASK_RAND_PROB = False,
    input_mask_prob_pwr = 0,
    output_mask_prob_pwr = 0,
    MASK_SETS_INDICES = [(0,1)]):

    model.eval()
    if LATENT_FLOW_TRAINING:
        autoencoder_model.eval()
    total = 0.0
    MAE_total = 0.0
    MSE_total = 0.0
    dataset_length = 0.0
    batch_counter = 0
    time_dict = {}
    time_dict['sine'] = 0
    time_dict['squeeze'] = 0
    time_dict['input_output'] = 0
    time_dict['output_row'] = 0
    time_dict['CFM'] = 0
    time_dict['loop'] = 0
    with torch.no_grad():
        for x_data, initial_mask in loader:
            start_loop = time.time()
            
            #Squeezing is necessary because I am defaulting to an emit batches style output for my datasets
            #x_data contains the normalized gaia data
            x_data = x_data.to(device, non_blocking=True).squeeze(0)
            #retrieve the initial mask. This contains the intrinsic masking (i.e., if this feature has a non-nan entry in DR3 of the Gaia Source Catalog)
            #should be a bool already
            initial_mask = initial_mask.to(device, non_blocking=True, dtype = torch.bool).squeeze(0)
            
            #create a variable for the batch length. Note that when reading directly from Gaia Source hdf5's the dataset reader defaults to a variable batch length!
            batch_length = x_data.shape[0]
            squeeze_time = time.time()
            
            if SINE_MASK_PROB and MASKING:
                mask_prob = np.sin(mp_sin_freq * batch_counter)**2 - mask_prob_min
            elif INTRA_MASK_RAND_PROB:
                mask_prob = rng.uniform(size = batch_length)**input_mask_prob_pwr
            if OPPOSITE_OUTPUT_MASK:
                output_mask_prob = (1 - mask_prob) - (2*mask_prob_min)
            elif INTRA_MASK_RAND_PROB:
                output_mask_prob = rng.uniform(size = batch_length)**output_mask_prob_pwr
            sin_mask_time = time.time()
            
            input_mask, output_mask = mask_generator(MASKING, AUTOENCODER_TRAINING, batch_length, dim, device, initial_mask, mask_prob, output_mask_prob, NEVER_MASK_INPUT, NEVER_MASK_INPUT_INDICES, NEVER_MASK_OUTPUT, NEVER_MASK_OUTPUT_INDICES, MASK_SETS, MASK_SETS_INDICES)
                
            input_output_time = time.time()
            
            #The following procedure checks if there are any sources with a totally masked output. If so, find one valid feature (i.e., not masked by initial_mask) and unmask it.
            #This prevents nan values while training.
            #No analoguous procedure is needed for the input mask. In fact complete masking of the conditional variables is sometimes desirable!
        
            row_has_true = output_mask.any(dim=1)
            bad_rows = (~row_has_true).nonzero(as_tuple=True)[0]

            if bad_rows.numel() > 0:
                valid = initial_mask[bad_rows]   # (B, C) bool

                # Guard rows with at least one valid feature
                num_valid = valid.sum(dim=1)
                rows_with_valid = (num_valid > 0).nonzero(as_tuple=True)[0]
                if rows_with_valid.numel() > 0:
                    v = valid[rows_with_valid]   # (Bv, C)

                    # Assign random scores to valid positions; invalid get -inf
                    scores = torch.rand(v.shape, device=v.device)
                    scores[~v] = float('-inf')   # exclude invalid features from selection

                    # Pick the argmax per row -> uniform among valid features
                    idx = scores.argmax(dim=1)   # (Bv,)

                    rows_to_update = bad_rows[rows_with_valid]
                    output_mask[rows_to_update, idx] = True
                
                rows_without_valid = (num_valid == 0).nonzero(as_tuple=True)[0]
                if rows_without_valid.numel() > 0:
                    # Indices into the FULL batch that need to be dropped:
                    rows_to_drop = bad_rows[rows_without_valid]
                    keep = torch.ones(batch_length, dtype=torch.bool, device=device)
                    keep[rows_to_drop] = False
                    input_mask  = input_mask[keep]
                    output_mask = output_mask[keep]
                    x_data      = x_data[keep]
                    batch_length = x_data.shape[0]
                    

            #del initial_mask, row_has_true
            output_row_time = time.time()
            
            #The training procedure is done below
            if LATENT_FLOW == False:
                #CFM model training
                
                #x0 is the sample from the base distribution (i.e., a Normal distribution)
                x0 = torch.randn(batch_length, dim, device=device)
                #t is the time variable, which sets where in the transform from x0 to x_data (x1) you are
                t  = torch.rand(batch_length, 1,    device=device)
                mu = x0 + (t*(x_data - x0))
                #x_t is the mid transition of the variables from x0 to x1. It, along with t, form the core inputs of the model, while the masked data and masks serve as conditional variables
                x_t = torch.normal(mean = mu, std = sigma)
                #multiply x_t by the output mask
                x_t = x_t.mul(output_mask)
                #del mu
                #create x_data*input_mask var
                x_in = x_data.mul(input_mask)
                
                #u_cond is the target velocity that we are matching to
                u_cond = x_data - x0
                #del x0

                #v_theta is the model output, i.e., the velocity as predicted by the model
                v_theta = model(t, x_t, output_mask, x_in, input_mask)
                #del t, x_t, input_mask, x_data

                #the loss is just MSE of v_theta and u_cond, with a masking (output_mask) applied
                #v_theta - u_cond
                diff = v_theta.sub(u_cond)
                #Note that you are doing things in velocity space. You cannot treat v_theta as though it is simply the parameters.
                MSE = diff.square().masked_select(output_mask).mean()
                MAE = diff.abs().masked_select(output_mask).mean()
                
            elif AUTOENCODER_TRAINING:
                x_in = x_data.mul(input_mask)
                #If training the autoencoder, no CFM procedure is needed. Just take the masked input data and the input mask as inputs to the model and predict the input_mask*x_data. Use MSE.
                #Note that previously I was going from (input_mask*data, input_mask, output_mask) to (output_mask*data), but this removes the need for a CFM model at all. Still, the loss was low, so it might be worth doing for comparison in the future.
                output = model(x_in, input_mask)
                diff = output.sub(x_data)
                MSE = diff.square().masked_select(input_mask).mean()
                MAE = diff.abs().masked_select(input_mask).mean()
                #del input_mask, output_mask, output, x_data

            elif LATENT_FLOW_TRAINING:
                #Special CFM training procedure for the latent flow model. Identical to the normal CFM, except that CFM occurs in an embedded space. There is no masking applied to the embedding
                
                #calculate the embedding for the autoencoder model
                em_in_in = torch.cat([input_mask*x_data, input_mask], dim = 1)
                em_in_out = torch.cat([output_mask*x_data, output_mask], dim = 1)
                #input embedding is the context for the latent space model
                input_embedding = autoencoder_model.Encoder(em_in_in)
                #output embedding is what shapes the CFM
                target_embedding = autoencoder_model.Encoder(em_in_out)
                #del input_mask, x_data
                
                em_shape = input_embedding.shape[1]
                #CFM procedure is now done in the embedding space
                x0 = torch.randn(batch_length, em_shape, device=device)
                t  = torch.rand(batch_length, 1,    device=device)
                mu = x0 + (t*(target_embedding-x0))
                x_t = torch.normal(mean = mu, std = sigma)
                #del mu
                #We want the flow to go from the base distribution to the target embedding!
                u_cond = target_embedding - x0
                #del x0, target_embedding
                
                #note that the model is acting on the embeddings from the encoder. Note also that concatenation most occur before model input
                out_mask_float = output_mask.float()
                input_cat = torch.cat([t, x_t, input_embedding, out_mask_float], dim = 1)
                v_theta = model(input_cat)
                
                MSE = torch.mean(torch.square(v_theta - u_cond))
                MAE = torch.mean(torch.abs(v_theta - u_cond))
                
                #del v_theta, u_cond, input_embedding, output_mask, x0, target_embedding, mu, out_mask_float, em_in_in, em_in_out, em_shape,  t, x_t, input_cat
            CFM_time = time.time()

            if MAE_LOSS and MSE_LOSS:
                loss = MAE + MSE
            elif MAE_LOSS:
                loss = MAE
            elif MSE_LOSS:
                loss = MSE
            
            total += loss.item() * batch_length
            MAE_total += MAE.item() * batch_length
            MSE_total += MSE.item() * batch_length
            dataset_length += batch_length
            batch_counter += 1
            if DEBUG_VAL:
                time_dict['sine'] += sin_mask_time - start_loop
                time_dict['squeeze'] += squeeze_time - sin_mask_time
                time_dict['input_output'] += input_output_time - squeeze_time
                time_dict['output_row'] += output_row_time - input_output_time
                time_dict['CFM'] += CFM_time - output_row_time
                time_dict['loop'] += CFM_time - start_loop
    if DEBUG_VAL:
        print(f"{batch_counter} loops. Average batch time {time_dict['loop']/batch_counter}, average sine masking time {time_dict['sine']/batch_counter}, average squeeze time {time_dict['squeeze']/batch_counter}, average input & output mask time {time_dict['input_output']/batch_counter}, average output row check time {time_dict['output_row']/batch_counter}, average CFM time {time_dict['CFM']/batch_counter}.")
    if dist.is_available() and dist.is_initialized():
        loss_tensor = torch.tensor(
            [total, MAE_total, MSE_total, dataset_length],
            dtype=torch.float64, device=device
        )
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total        = loss_tensor[0].item()
        MAE_total    = loss_tensor[1].item()
        MSE_total    = loss_tensor[2].item()
        dataset_length = loss_tensor[3].item()

    return total / dataset_length, MAE_total / dataset_length, MSE_total / dataset_length
    
    
def validate_epoch_set(loader, #Pytorch DataLoader containing validation or testing data
    model, #ML model being trained
    device,
    dim,
    custom_mask, #Deterministic mask applied equally to all examples
    LATENT_FLOW, #Controls model variant specific training procedure, see train_epoch() for more information
    AUTOENCODER_TRAINING, #Controls model variant specific training procedure, see train_epoch() for more information
    LATENT_FLOW_TRAINING, #Controls model variant specific training procedure, see train_epoch() for more information
    VAL_M_FRAC, #Determines how many embedding nodes are available if using IOB method. If none, but using IOB method (i.e., IOB_METHOD = True in model instantiation), defaults to randomly selecting a fraction of the nodes to keep open.
    autoencoder_model, #Pytorch model of autoencoder. Only relevant if LATENT_FLOW_TRAINING == True.
    sigma, #used for CFM
    MAE_LOSS,
    MSE_LOSS,
    DEBUG_VAL = False,
    L_CUSTOM = False,
    custom_mask_name = ''):
    
    g = torch.Generator(device=device).manual_seed(117)

    model.eval()
    if LATENT_FLOW_TRAINING:
        autoencoder_model.eval()
    total = 0.0
    MAE_total = 0.0
    MSE_total = 0.0
    dataset_length = 0.0
    batch_counter = 0
    time_dict = {}
    time_dict['sine'] = 0
    time_dict['squeeze'] = 0
    time_dict['input_output'] = 0
    time_dict['output_row'] = 0
    time_dict['CFM'] = 0
    time_dict['loop'] = 0
    with torch.no_grad():
        for x_data, initial_mask in loader:
            start_loop = time.time()
            
            sin_mask_time = time.time()
            
            #Squeezing is necessary because I am defaulting to an emit batches style output for my datasets
            #x_data contains the normalized gaia data
            x_data = x_data.to(device, non_blocking=True).squeeze(0)
            #retrieve the initial mask. This contains the intrinsic masking (i.e., if this feature has a non-nan entry in DR3 of the Gaia Source Catalog)
            #should be a bool already
            initial_mask = initial_mask.to(device, non_blocking=True, dtype = torch.bool).squeeze(0)
            
            #create a variable for the batch length. Note that when reading directly from Gaia Source hdf5's the dataset reader defaults to a variable batch length!
            batch_length = x_data.shape[0]
            squeeze_time = time.time()
            
            #Use only selected input features (that are available)
            input_mask = torch.from_numpy(custom_mask).to(device, dtype = torch.bool).unsqueeze(0) & initial_mask   # (1, F) * (B, F) -> (B, F)
            #Use only features that exist and are masked
            #For the autoencoder case, use all output features, since none are masked
            if custom_mask_name == 'autoencoder':
                output_mask = initial_mask
            else:
                output_mask = (~torch.from_numpy(custom_mask).to(device, dtype = torch.bool).unsqueeze(0)) & initial_mask

            input_output_time = time.time()
            
            #The following procedure checks if there are any sources with a totally masked output. If so, find one valid feature (i.e., not masked by initial_mask) and unmask it.
            #This prevents nan values while training.
            #No analoguous procedure is needed for the input mask. In fact complete masking of the conditional variables is sometimes desirable!
        
            row_has_true = output_mask.any(dim=1)
            bad_rows = (~row_has_true).nonzero(as_tuple=True)[0]

            if bad_rows.numel() > 0:
                valid = initial_mask[bad_rows]   # (B, C) bool

                # Guard rows with at least one valid feature
                num_valid = valid.sum(dim=1)
                rows_with_valid = (num_valid > 0).nonzero(as_tuple=True)[0]
                if rows_with_valid.numel() > 0:
                    v = valid[rows_with_valid]   # (Bv, C)

                    # Assign random scores to valid positions; invalid get -inf
                    scores = torch.rand(v.shape, device=v.device, generator = g)
                    scores[~v] = float('-inf')   # exclude invalid features from selection

                    # Pick the argmax per row -> uniform among valid features
                    idx = scores.argmax(dim=1)   # (Bv,)

                    rows_to_update = bad_rows[rows_with_valid]
                    output_mask[rows_to_update, idx] = True
                
                rows_without_valid = (num_valid == 0).nonzero(as_tuple=True)[0]
                if rows_without_valid.numel() > 0:
                    # Indices into the FULL batch that need to be dropped:
                    rows_to_drop = bad_rows[rows_without_valid]
                    keep = torch.ones(batch_length, dtype=torch.bool, device=device)
                    keep[rows_to_drop] = False
                    input_mask  = input_mask[keep]
                    output_mask = output_mask[keep]
                    x_data      = x_data[keep]
                    batch_length = x_data.shape[0]
                    

            #del initial_mask, row_has_true
            output_row_time = time.time()
            
            #The training procedure is done below
            if LATENT_FLOW == False:
                #CFM model training
                
                #x0 is the sample from the base distribution (i.e., a Normal distribution)
                x0 = torch.randn(batch_length, dim, device=device, generator = g)
                #t is the time variable, which sets where in the transform from x0 to x_data (x1) you are
                t  = torch.rand(batch_length, 1,    device=device, generator = g)
                mu = x0 + (t*(x_data - x0))
                #x_t is the mid transition of the variables from x0 to x1. It, along with t, form the core inputs of the model, while the masked data and masks serve as conditional variables
                x_t = torch.normal(mean = mu, std = sigma, generator = g)
                #multiply x_t by the output mask
                x_t = x_t.mul(output_mask)
                #del mu
                #create x_data*input_mask var
                x_in = x_data.mul(input_mask)
                
                #u_cond is the target velocity that we are matching to
                u_cond = x_data - x0
                #del x0

                #v_theta is the model output, i.e., the velocity as predicted by the model
                v_theta = model(t, x_t, output_mask, x_in, input_mask)
                #del t, x_t, input_mask, x_data

                #the loss is just MSE of v_theta and u_cond, with a masking (output_mask) applied
                #v_theta - u_cond
                diff = v_theta.sub(u_cond)
                #Note that you are doing things in velocity space. You cannot treat v_theta as though it is simply the parameters.
                MSE = diff.square().masked_select(output_mask).mean()
                MAE = diff.abs().masked_select(output_mask).mean()
                
            elif AUTOENCODER_TRAINING:
                x_in = x_data.mul(input_mask)
                #If training the autoencoder, no CFM procedure is needed. Just take the masked input data and the input mask as inputs to the model and predict the input_mask*x_data. Use MSE.
                #Note that previously I was going from (input_mask*data, input_mask, output_mask) to (output_mask*data), but this removes the need for a CFM model at all. Still, the loss was low, so it might be worth doing for comparison in the future.
                output = model(x_in, input_mask)
                diff = output.sub(x_data)
                MSE = diff.square().masked_select(input_mask).mean()
                MAE = diff.abs().masked_select(input_mask).mean()
                #del input_mask, output_mask, output, x_data

            elif LATENT_FLOW_TRAINING:
                #Special CFM training procedure for the latent flow model. Identical to the normal CFM, except that CFM occurs in an embedded space. There is no masking applied to the embedding
                
                #calculate the embedding for the autoencoder model
                em_in_in = torch.cat([input_mask*x_data, input_mask], dim = 1)
                em_in_out = torch.cat([output_mask*x_data, output_mask], dim = 1)
                #input embedding is the context for the latent space model
                input_embedding = autoencoder_model.Encoder(em_in_in)
                #output embedding is what shapes the CFM
                target_embedding = autoencoder_model.Encoder(em_in_out)
                #del input_mask, x_data
                
                em_shape = input_embedding.shape[1]
                #CFM procedure is now done in the embedding space
                x0 = torch.randn(batch_length, em_shape, device=device, generator = g)
                t  = torch.rand(batch_length, 1,    device=device, generator = g)
                mu = x0 + (t*(target_embedding-x0))
                x_t = torch.normal(mean = mu, std = sigma, generator = g)
                #del mu
                #We want the flow to go from the base distribution to the target embedding!
                u_cond = target_embedding - x0
                #del x0, target_embedding
                
                #note that the model is acting on the embeddings from the encoder. Note also that concatenation most occur before model input
                out_mask_float = output_mask.float()
                input_cat = torch.cat([t, x_t, input_embedding, out_mask_float], dim = 1)
                v_theta = model(input_cat)
                
                MSE = torch.mean(torch.square(v_theta - u_cond))
                MAE = torch.mean(torch.abs(v_theta - u_cond))
                
                #del v_theta, u_cond, input_embedding, output_mask, x0, target_embedding, mu, out_mask_float, em_in_in, em_in_out, em_shape,  t, x_t, input_cat
            CFM_time = time.time()

            if MAE_LOSS and MSE_LOSS:
                loss = MAE + MSE
            elif MAE_LOSS:
                loss = MAE
            elif MSE_LOSS:
                loss = MSE
            
            total += loss.item() * batch_length
            MAE_total += MAE.item() * batch_length
            MSE_total += MSE.item() * batch_length
            dataset_length += batch_length
            batch_counter += 1
            if DEBUG_VAL:
                time_dict['sine'] += sin_mask_time - start_loop
                time_dict['squeeze'] += squeeze_time - sin_mask_time
                time_dict['input_output'] += input_output_time - squeeze_time
                time_dict['output_row'] += output_row_time - input_output_time
                time_dict['CFM'] += CFM_time - output_row_time
                time_dict['loop'] += CFM_time - start_loop
    if DEBUG_VAL:
        print(f"{batch_counter} loops. Average batch time {time_dict['loop']/batch_counter}, average sine masking time {time_dict['sine']/batch_counter}, average squeeze time {time_dict['squeeze']/batch_counter}, average input & output mask time {time_dict['input_output']/batch_counter}, average output row check time {time_dict['output_row']/batch_counter}, average CFM time {time_dict['CFM']/batch_counter}.")
    if dist.is_available() and dist.is_initialized():
        loss_tensor = torch.tensor(
            [total, MAE_total, MSE_total, dataset_length],
            dtype=torch.float64, device=device
        )
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total        = loss_tensor[0].item()
        MAE_total    = loss_tensor[1].item()
        MSE_total    = loss_tensor[2].item()
        dataset_length = loss_tensor[3].item()

    return total / dataset_length, MAE_total / dataset_length, MSE_total / dataset_length

"""
Multiple epoch model training and validation
Trains and validates model one epoch at a time.
Runs for as long as allowed by num_epochs, or stops if validation loss has not acheived a new minimum in stopping_num epochs.
Saves models that acheives new lowest validation loss, overwriting existing model (only models with the same name)
"""

def moving_average_loss(loss_list, N_epochs):
    return np.mean(np.array(loss_list)[-1*N_epochs:])

def save_loss_dict(train_loss_array, val_loss_array, input_p_array, output_p_array, train_mae_array, train_mse_array, val_mae_array, val_mse_array, MODEL_NAME):
    loss_dict = {}
    loss_dict['train'] = train_loss_array
    loss_dict['val'] = val_loss_array
    loss_dict['input_probabilities'] = input_p_array
    loss_dict['output_probabilities'] = output_p_array
    loss_dict['train_mae'] = train_mae_array
    loss_dict['train_mse'] = train_mse_array
    loss_dict['val_mae'] = val_mae_array
    loss_dict['val_mse'] = val_mse_array
    np.save('YOUR PATH/loss_data/' + MODEL_NAME + '_loss_data.npy', loss_dict)
    
def save_loss_dict_val_set(loss_dict, custom_mask_dict, train_loss_array, input_p_array, output_p_array, train_mae_array, train_mse_array, MODEL_NAME):
    new_loss_dict = {}
    new_loss_dict['train'] = train_loss_array
    new_loss_dict['input_probabilities'] = input_p_array
    new_loss_dict['output_probabilities'] = output_p_array
    new_loss_dict['train_mae'] = train_mae_array
    new_loss_dict['train_mse'] = train_mse_array
    new_loss_dict['val'] = {}
    for custom_mask_name in custom_mask_dict:
        new_loss_dict['val'][custom_mask_name] = {}
        new_loss_dict['val']['total'] = np.array(loss_dict['val']['total'])
        for loss_type in ['val', 'val_mae', 'val_mse']:
            new_loss_dict['val'][custom_mask_name][loss_type] = np.array(loss_dict['val'][custom_mask_name][loss_type])
    np.save('YOUR PATH/loss_data/' + MODEL_NAME + '_loss_data.npy', new_loss_dict)

def train(MODEL_NAME, #File name of the model. Used for saving and loading. Determined by flags set in main code.
    model, #Pytorch model being trained
    device, #where model training occurs
    rank,
    is_main,
    dim,
    train_loader, #Pytorch dataloader for training data
    val_loader, #Pytorch dataloader for validation data. Data used to determine when training finished
    optimizer, #model optimizer
    MASK_TRAIN, #Boolean that determines if masking is applied to training data
    MASK_VAL, #Boolean that determines if masking is applied to validation data
    LOAD_MODEL, #If true, assumes you are loading a model.
    loss_dict, #If LOAD_MODEL is true this will contain the previous losses
    num_epochs=10, #number of epochs of training allowed. Note that this setting has preeminence over all other training settings, i.e., if the epoch number reaches this number training WILL stop.
    stopping_num = 10, #number of epochs without new minimum validation loss allowed before training is ended. As noted, this condition may never be met if num_epochs is too low.
    min_epochs = 50, #minimum number of epochs allowed before stopping number considered. This can be used to force model training for an extended period of time. Does not override num_epochs.
    save_directory = '', #directory where models are saved
    loss_directory = '',
    mask_prob = 0.5, #Sets the masking probability. Can be overridden by SINE_MASK_PROB. See train_epoch() for more information.
    SINE_MASK_PROB = False, #If true, use sine scheduler for determining mask_prob. See train_epoch() for more information.
    OPPOSITE_OUTPUT_MASK = False, #If true, use 1-mask_prob for mask_prob of output mask. See train_epoch() for more information.
    LATENT_FLOW = False, #If true, training latent CFM model or associated autoencoder is used
    VAL_M_FRAC = None, #If given a value and IOB_METHOD is true (see main code), this sets the fraction of latent parameters used. See validate_epoch() for more information.
    AUTOENCODER_TRAINING = False, #If true, train autoencoder associated with latent CFM model. See train_epoch() for more information.
    LATENT_FLOW_TRAINING = False, #if true, train latent space CFM model. See train_epoch() for more information.
    mp_sin_freq = 0, #Sets the frequency of the sine mask probability scheduler
    mask_prob_min = 0, #Sets the minimum possible masking probability when using the sine scheduler
    autoencoder_model = 0, #Pytorch model of autoencoder. Only relevant if LATENT_FLOW_TRAINING == True.
    DEBUG_VAL = False, #If true, only run val loop
    DEBUG_TRAIN = False, #If true, only run train loop
    sigma = 1e-3, #used for CFM
    SAVE_MODEL = True, #Controls if models are saved
    mask_prob_step = 0.2, #Sets the step size of the masking probabilities for the masking probability for loop.
    MAE_LOSS = False,
    MSE_LOSS = True,
    NEVER_MASK_INPUT = False,
    NEVER_MASK_INPUT_INDICES = [0,1],
    NEVER_MASK_OUTPUT = False,
    NEVER_MASK_OUTPUT_INDICES = [0,1],
    MASK_PROB_RAND = False, #if true, the masking probability for each training epoch is randomly drawn from a uniform(0,1) probability distribution.
    MASK_SETS = False, #If true, these parameters are linked
    MASK_SETS_INDICES = [(0,1)], #if MASK_SETS these sets of parameters are linked such that if one is masked, all are masked. Note that the random masking probability for the set of parameters should be equal to mask_prob.
    L_CUSTOM = False,
    ACUM_VAL = 1,
    scheduler = 0,
    SCHEDULER = False,
    moving_average_loss_window_width = 10,
    input_mask_prob_pwr = 1,
    output_mask_prob_pwr = 1,
    MODEL_NAME_UNMOD = '',
    MODIFY_LOAD = False,
    code_start = 0,
    MODEL_PARAM_DICT = {},
    EMA = False,
    ema = 0,
    ema_decay = 0,
    SET_VAL = False,
    custom_mask_dict = {},
    FIXED_TRAINING_MASKS = False,
    INTRA_MASK_RAND_PROB = False,
    ):

    assert MAE_LOSS or MSE_LOSS, f"One of MAE_LOSS and MSE_LOSS must be true! Currently both are False"
    
    if LATENT_FLOW:
        assert AUTOENCODER_TRAINING != LATENT_FLOW_TRAINING, f"When LATENT_FLOW is true, one, and only one, of AUTOENCODER_TRAINING and LATENT_FLOW_TRAINING must be true. However LATENT_FLOW is {LATENT_FLOW}, but AUTOENCODER_TRAINING is {AUTOENCODER_TRAINING} and LATENT_FLOW_TRAINING is {LATENT_FLOW_TRAINING}."
        
    if SINE_MASK_PROB or OPPOSITE_OUTPUT_MASK:
        assert (MASK_TRAIN or MASK_VAL), f"For sine masking scheduler to apply, and/or to use the opposite masking probability for the output mask as opposed to the input mask, either MASK_TRAIN or MASK_VAL must be true. Currently MASK_TRAIN is {MASK_TRAIN} and MASK_VAL is {MASK_VAL}."
        
    #the count is used to determine how many epoches have passed since a new minimum validation loss has been acheived. Can cause the model to end training early if epoch < min_epochs and count >= stopping_num.
    count = 0
    
    if SET_VAL or FIXED_TRAINING_MASKS:
        N_fixed_masks = 0
        custom_mask_list = []
        for key in custom_mask_dict:
            custom_mask_list += [key]
            N_fixed_masks += 1
        if is_main:
            print(f"Using {N_fixed_masks} fixed masks for training and/or validation.")
    
    if LOAD_MODEL:
        #load loss information
        train_loss_list = list(loss_dict['train'])
        ip_list = list(loss_dict['input_probabilities'])
        op_list = list(loss_dict['output_probabilities'])
        train_mae_list = list(loss_dict['train_mae'])
        train_mse_list = list(loss_dict['train_mse'])
        if SET_VAL:
            loss_dict['val']['total'] = list(loss_dict['val']['total'])
            #transform arrays back into lists
            for custom_mask_name in custom_mask_dict:
                for loss_type in ['val', 'val_mae', 'val_mse']:
                    loss_dict['val'][custom_mask_name][loss_type] = list(loss_dict['val'][custom_mask_name][loss_type])
        else:
            val_loss_list = list(loss_dict['val'])
            val_mae_list = list(loss_dict['val_mae'])
            val_mse_list = list(loss_dict['val_mse'])
    else:
        #initialize lists for saving loss information
        train_loss_list = []
        ip_list = []
        op_list = []
        train_mae_list = []
        train_mse_list = []
        if SET_VAL:
            loss_dict = {}
            loss_dict['val'] = {}
            loss_dict['val']['total'] = []
            for custom_mask_name in custom_mask_dict:
                loss_dict['val'][custom_mask_name] = {}
                for loss_type in ['val', 'val_mae', 'val_mse']:
                    loss_dict['val'][custom_mask_name][loss_type] = []
        else:
            val_loss_list = []
            val_mae_list = []
            val_mse_list = []
        
    #If using a pre-existing model, use the minimum validation loss from previous training
    #otherwise set minimum val loss to an arbitrarily high value
    if LOAD_MODEL and SET_VAL == False:
        best_val_loss = np.min(loss_dict['val'])
    elif LOAD_MODEL and SET_VAL:
        best_val_loss = np.min(loss_dict['val']['total'])
    else:
        best_val_loss = float('inf')
    
    if LOAD_MODEL:
        start_epoch = len(loss_dict['train'])
    else:
        start_epoch = 0
    
    #main loop for training and validation. Will run at most num_epochs, but can stop sooner if specific conditions are met
    for epoch in range(start_epoch, start_epoch + num_epochs):
        torch.manual_seed(1000003 * epoch + rank)
        loop_start = time.time()
        # DDP batch-count equalization: every rank must call set_epoch with
        # the same epoch number BEFORE iteration begins, so each (rank, worker)
        # shard gets the same balanced file assignment and the same per-shard
        # batch cap. No-op when the dataset wasn't constructed with
        # equalize_batches_across_ranks=True (e.g. plain inference loops).
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)
        if hasattr(val_loader.dataset, 'set_epoch') and SET_VAL:
            val_loader.dataset.set_epoch(0) #going for fully reproducible validation set
        else:
            val_loader.dataset.set_epoch(epoch)

        if is_main:
            print()
        if (MASK_PROB_RAND and FIXED_TRAINING_MASKS == False and INTRA_MASK_RAND_PROB == False) or (MASK_PROB_RAND and FIXED_TRAINING_MASKS and epoch%(N_fixed_masks*2) >= N_fixed_masks):
            if is_main:
                probs = torch.tensor(
                    [rng.uniform()**input_mask_prob_pwr, rng.uniform()**output_mask_prob_pwr], dtype=torch.float32, device=device
                )
            else:
                probs = torch.zeros(2, dtype=torch.float32, device=device)
            collective_count = 0
            dist.broadcast(probs, src=0)
            collective_count += 1
            mask_prob       = probs[0].item()
            output_mask_prob = probs[1].item()
            if is_main:
                print(f"Rank {rank}: collective count {collective_count}")
                print('Training', 'Input Probability', mask_prob, 'Output Probability', output_mask_prob)
                ip_list += [mask_prob]
                op_list += [output_mask_prob]
        elif FIXED_TRAINING_MASKS and MASK_PROB_RAND and epoch%(N_fixed_masks*2) < N_fixed_masks:
            output_mask_prob = None
            mask_prob = np.nan
            if is_main:
                ip_list += [np.nan]
                op_list += [np.nan]
        elif INTRA_MASK_RAND_PROB:
            mask_prob = np.nan
            output_mask_prob = None
            if is_main:
                ip_list += [np.nan]
                op_list += [np.nan]
        else:
            output_mask_prob = None
            if is_main:
                ip_list += [mask_prob]
                if OPPOSITE_OUTPUT_MASK:
                    op_list += [(1 - mask_prob) - (2*mask_prob_min)]
                else:
                    op_list += [mask_prob]

        
        #training epoch
        #use mask probability unless fixed training mask is true
        #If FIXED_TRAINING_MASKS is true and MASK_PROB_RAND is true, use mask probability equally as often
        if (DEBUG_VAL == False and FIXED_TRAINING_MASKS == False) or (DEBUG_VAL == False and FIXED_TRAINING_MASKS and MASK_PROB_RAND and epoch%(N_fixed_masks*2) >= N_fixed_masks):
            train_loss, train_mae, train_mse = train_epoch(train_loader,
            model,
            device,
            dim,
            optimizer,
            MASK_TRAIN,
            mask_prob,
            SINE_MASK_PROB,
            OPPOSITE_OUTPUT_MASK,
            LATENT_FLOW,
            AUTOENCODER_TRAINING,
            LATENT_FLOW_TRAINING,
            mp_sin_freq,
            mask_prob_min,
            autoencoder_model,
            sigma,
            MAE_LOSS,
            MSE_LOSS,
            NEVER_MASK_INPUT,
            NEVER_MASK_INPUT_INDICES,
            NEVER_MASK_OUTPUT,
            NEVER_MASK_OUTPUT_INDICES,
            DEBUG_TRAIN = DEBUG_TRAIN,
            output_mask_prob = output_mask_prob,
            L_CUSTOM = L_CUSTOM,
            MASK_SETS = MASK_SETS,
            MASK_SETS_INDICES = MASK_SETS_INDICES,
            accumulation_val = ACUM_VAL,
            scheduler = scheduler,
            SCHEDULER = SCHEDULER,
            EMA = EMA,
            ema = ema,
            ema_decay = ema_decay,
            is_main = is_main,
            INTRA_MASK_RAND_PROB = INTRA_MASK_RAND_PROB,
            input_mask_prob_pwr = input_mask_prob_pwr,
            output_mask_prob_pwr = output_mask_prob_pwr)
            train_loss_list.append(train_loss)
            train_mae_list.append(train_mae)
            train_mse_list.append(train_mse)
        elif DEBUG_VAL == False and FIXED_TRAINING_MASKS and MASK_PROB_RAND and epoch%(N_fixed_masks*2) < N_fixed_masks:
            if is_main:
                print('train on:', custom_mask_list[epoch%(N_fixed_masks*2)])
                
            train_loss, train_mae, train_mse = train_epoch(train_loader,
            model,
            device,
            dim,
            optimizer,
            MASK_TRAIN,
            mask_prob,
            SINE_MASK_PROB,
            OPPOSITE_OUTPUT_MASK,
            LATENT_FLOW,
            AUTOENCODER_TRAINING,
            LATENT_FLOW_TRAINING,
            mp_sin_freq,
            mask_prob_min,
            autoencoder_model,
            sigma,
            MAE_LOSS,
            MSE_LOSS,
            NEVER_MASK_INPUT,
            NEVER_MASK_INPUT_INDICES,
            NEVER_MASK_OUTPUT,
            NEVER_MASK_OUTPUT_INDICES,
            DEBUG_TRAIN = DEBUG_TRAIN,
            output_mask_prob = output_mask_prob,
            L_CUSTOM = L_CUSTOM,
            MASK_SETS = MASK_SETS,
            MASK_SETS_INDICES = MASK_SETS_INDICES,
            accumulation_val = ACUM_VAL,
            scheduler = scheduler,
            SCHEDULER = SCHEDULER,
            EMA = EMA,
            ema = ema,
            ema_decay = ema_decay,
            is_main = is_main,
            FIXED_MASK_SET = True,
            custom_mask = custom_mask_dict[custom_mask_list[epoch%(N_fixed_masks*2)]],
            custom_mask_name = custom_mask_list[epoch%(N_fixed_masks*2)]
            )
            train_loss_list.append(train_loss)
            train_mae_list.append(train_mae)
            train_mse_list.append(train_mse)
        #validation epoch
        if DEBUG_TRAIN == False and SET_VAL == False:
            val_loss, val_mae, val_mse = validate_epoch(val_loader,
            model,
            device,
            dim,
            MASK_VAL,
            mask_prob,
            SINE_MASK_PROB,
            OPPOSITE_OUTPUT_MASK,
            LATENT_FLOW,
            AUTOENCODER_TRAINING,
            LATENT_FLOW_TRAINING,
            VAL_M_FRAC,
            mp_sin_freq,
            mask_prob_min,
            autoencoder_model,
            sigma,
            MAE_LOSS,
            MSE_LOSS,
            NEVER_MASK_INPUT,
            NEVER_MASK_INPUT_INDICES,
            NEVER_MASK_OUTPUT,
            NEVER_MASK_OUTPUT_INDICES,
            DEBUG_VAL = DEBUG_VAL,
            L_CUSTOM = L_CUSTOM,
            MASK_SETS = MASK_SETS,
            MASK_SETS_INDICES = MASK_SETS_INDICES,
            INTRA_MASK_RAND_PROB = INTRA_MASK_RAND_PROB,
            input_mask_prob_pwr = input_mask_prob_pwr,
            output_mask_prob_pwr = output_mask_prob_pwr)

            val_loss_list.append(val_loss)
            val_mae_list.append(val_mae)
            val_mse_list.append(val_mse)
            current_val = val_loss
        elif DEBUG_VAL == False:
            val_total = 0
            for custom_mask_name in custom_mask_dict:
                val_loss, val_mae, val_mse = validate_epoch_set(val_loader,
                model,
                device,
                dim,
                custom_mask_dict[custom_mask_name],
                LATENT_FLOW,
                AUTOENCODER_TRAINING,
                LATENT_FLOW_TRAINING,
                VAL_M_FRAC,
                autoencoder_model,
                sigma,
                MAE_LOSS,
                MSE_LOSS,
                DEBUG_VAL = DEBUG_VAL,
                L_CUSTOM = L_CUSTOM,
                custom_mask_name = custom_mask_name
                )

                loss_dict['val'][custom_mask_name]['val'].append(val_loss)
                loss_dict['val'][custom_mask_name]['val_mae'].append(val_mae)
                loss_dict['val'][custom_mask_name]['val_mse'].append(val_mse)
                val_total += val_loss
            loss_dict['val']['total'].append(val_total)
            current_val = val_total
        loop_finish = time.time() - loop_start
        total_time_update = time.time() - code_start

        if is_main:
            if DEBUG_TRAIN == False and DEBUG_VAL == False and SET_VAL == False:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Loop Time (s) = {loop_finish:.4f} | Total Time: (s) = {total_time_update:.4f}")
                print(f"Epoch {epoch+1}: Train MAE = {train_mae:.4f} | Train MSE = {train_mse:.4f} | Val MAE = {val_mae:.4f} | Val MSE = {val_mse:.4f}")
                if epoch > moving_average_loss_window_width:
                    print(f"Moving Average, Past {int(moving_average_loss_window_width)} Epochs: Train = {moving_average_loss(train_loss_list, moving_average_loss_window_width):.4f}, Val = {moving_average_loss(val_loss_list, moving_average_loss_window_width):.4f}")
            elif DEBUG_TRAIN == False and DEBUG_VAL == False and SET_VAL == True:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Train MAE = {train_mae:.4f} | Train MSE = {train_mse:.4f} | Loop Time (s) = {loop_finish:.4f} | Total Time: (s) = {total_time_update:.4f}")
                print('###############################################################################################')
                for custom_mask_name in custom_mask_dict:
                    print(f"{custom_mask_name}| Val Loss: {loss_dict['val'][custom_mask_name]['val'][-1]:.4f}| Val MAE: {loss_dict['val'][custom_mask_name]['val_mae'][-1]:.4f}| Val MSE: {loss_dict['val'][custom_mask_name]['val_mse'][-1]:.4f}")
                    print('###############################################################################################')
                print()
            elif DEBUG_TRAIN:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Loop Time (s) = {loop_finish:.4f} | Total Time: (s) = {total_time_update:.4f}")
                print(f"Epoch {epoch+1}: Train MAE = {train_mae:.4f} | Train MSE = {train_mse:.4f}")
                if epoch > moving_average_loss_window_width:
                    print(f"Moving Average, Past {int(moving_average_loss_window_width)} Epochs: Train = {moving_average_loss(train_loss_list, moving_average_loss_window_width):.4f}")
            elif DEBUG_VAL and VAL_SET == False:
                print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f} | Loop Time (s) = {loop_finish:.4f} | Total Time: (s) = {total_time_update:.4f}")
                print(f"Epoch {epoch+1}: Val MAE = {val_mae:.4f} | Val MSE = {val_mse:.4f}")
                if epoch > moving_average_loss_window_width:
                    print(f"Moving Average, Past {int(moving_average_loss_window_width)} Epochs: Val = {moving_average_loss(val_loss_list, moving_average_loss_window_width):.4f}")
            elif DEBUG_VAL and VAL_SET:
                print(f"Epoch {epoch+1}: {custom_mask_name}| Val Loss: {loss_dict['val'][custom_mask_name]['val'][-1]:.4f}| Val MAE: {loss_dict['val'][custom_mask_name]['val_mae'][-1]:.4f}| Val MSE: {loss_dict['val'][custom_mask_name]['val_mse'][-1]:.4f}")
            if DEBUG_TRAIN or DEBUG_VAL:
                print("Debug epoch complete")
                return np.array(train_loss_list), np.array(val_loss_list)

        # If the new validation loss is lower than the previous best, save it and reset the count
        if current_val < best_val_loss and (MASK_PROB_RAND == False or SET_VAL or INTRA_MASK_RAND_PROB) and SAVE_MODEL:
            count = 0
            
            best_val_loss = current_val
                
            if is_main:
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(model_state, save_directory + MODEL_NAME + '.pth')
                torch.save(model_state, save_directory + MODEL_NAME + '_best_' + str(epoch+1) + '.pth')
                torch.save(optimizer.state_dict(), save_directory + MODEL_NAME + '_optimizer.pt')
                if SCHEDULER:
                    torch.save(scheduler.state_dict(), save_directory + MODEL_NAME + '_scheduler.pt')
                if EMA:
                    torch.save(ema, save_directory + MODEL_NAME + '_ema.pth')
                    torch.save(ema, save_directory + MODEL_NAME + '_ema_best_' + str(epoch+1) + '.pth')
                print("Model Saved")
                if LOAD_MODEL and MODIFY_LOAD:
                    np.save('YOUR PATH/model_param_dicts/' + MODEL_NAME + '_model_param_dict.npy', MODEL_PARAM_DICT)
                print('Updated Model Param Saved')
                if SET_VAL:
                    save_loss_dict_val_set(loss_dict, custom_mask_dict, np.array(train_loss_list), np.array(ip_list), np.array(op_list), np.array(train_mae_list), np.array(train_mse_list), MODEL_NAME_UNMOD)
                else:
                    save_loss_dict(np.array(train_loss_list), np.array(val_loss_list), np.array(ip_list), np.array(op_list), np.array(train_mae_list), np.array(train_mse_list), np.array(val_mae_list), np.array(val_mse_list), MODEL_NAME_UNMOD)
                print('Loss Saved')
        elif (MASK_PROB_RAND and SET_VAL == False) and (epoch+1)%10 == 0 and is_main and SAVE_MODEL: #just save the model every 10 epochs because val loss aren't comparable.
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(model_state, save_directory + MODEL_NAME + '.pth')
            torch.save(optimizer.state_dict(), save_directory + MODEL_NAME + '_optimizer.pt')
            if SCHEDULER == True:
                torch.save(scheduler.state_dict(), save_directory + MODEL_NAME + '_scheduler.pt')
            if EMA:
                torch.save(ema, save_directory + MODEL_NAME + '_ema.pth')
            print("Model Saved")
            if LOAD_MODEL and MODIFY_LOAD:
                np.save('YOUR PATH/model_param_dicts/' + MODEL_NAME + '_model_param_dict.npy', MODEL_PARAM_DICT)
            print('Updated Model Param Saved')
            if SET_VAL:
                save_loss_dict_val_set(loss_dict, custom_mask_dict, np.array(train_loss_list), np.array(ip_list), np.array(op_list), np.array(train_mae_list), np.array(train_mse_list), MODEL_NAME_UNMOD)
            else:
                save_loss_dict(np.array(train_loss_list), np.array(val_loss_list), np.array(ip_list), np.array(op_list), np.array(train_mae_list), np.array(train_mse_list), np.array(val_mae_list), np.array(val_mse_list), MODEL_NAME_UNMOD)
            print('Loss Saved')
        else:
            #If new validation loss minimum is not acheieved, increase count
            count += 1
            
        #Do not cancel before running some minimum amount. Note that this does not prevent model training from ending when epoch reaches num_epochs. If you wish to use min_epochs effectively, it must be less than num_epochs!
        #If epoch > min_epochs, end training early if count >= stopping_num.
        if epoch < min_epochs:
            count = 0
        elif count >= stopping_num:
            print(f"No improvement in {stopping_num} epochs. Ending early.")
            break

        #If the total run time is greater than the time required to complete another loop, break the loop
        #0.5 is a fudge factor in case the previous loop was shorter than average (loops normal do not vary very much in duration)
        if (total_time_update/3600) >= (48 - (1.5*loop_finish/3600)):
            print(f"Canceled because of approaching time limit")
            break
        #If MASK_PROB_RAND is True and you are on 10th epoch (i.e., a save point), break the loop if less than 10 times a single loop time remains
        if MASK_PROB_RAND and (epoch+1)%10 == 0 and (total_time_update/3600) >= (48 - (10.5*loop_finish/3600)):
            print(f"Canceled because of approaching time limit")
            break
    if SAVE_MODEL and is_main:
        # When model is DDP-wrapped, access weights via model.module
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(model_state, save_directory + MODEL_NAME + '_final.pth')#'_epoch_' + str(epoch+1) + '
        torch.save(model_state, save_directory + MODEL_NAME + '_final_' + str(epoch+1) + '.pth')
        torch.save(optimizer.state_dict(), save_directory + MODEL_NAME + '_optimizer.pt')
        if SCHEDULER:
            torch.save(scheduler.state_dict(), save_directory + MODEL_NAME + '_scheduler.pt')
        if EMA:
            torch.save(ema, save_directory + MODEL_NAME + '_ema.pth')
            torch.save(ema, save_directory + MODEL_NAME + '_ema_final_' + str(epoch+1) + '.pth')
        print("Saved")
        if LOAD_MODEL and MODIFY_LOAD:
            np.save('YOUR PATH/model_param_dicts/' + MODEL_NAME + '_model_param_dict.npy', MODEL_PARAM_DICT)
        print('Updated Model Param Saved')
        if SET_VAL:
            save_loss_dict_val_set(loss_dict, custom_mask_dict, np.array(train_loss_list), np.array(ip_list), np.array(op_list), np.array(train_mae_list), np.array(train_mse_list), MODEL_NAME_UNMOD)
        else:
            save_loss_dict(np.array(train_loss_list), np.array(val_loss_list), np.array(ip_list), np.array(op_list), np.array(train_mae_list), np.array(train_mse_list), np.array(val_mae_list), np.array(val_mse_list), MODEL_NAME_UNMOD)
        print('Loss Saved')
