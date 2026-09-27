import torch
import torch.nn as nn
import math
def activation(ACTIVATION_TYPE):
    if ACTIVATION_TYPE == 'ELU':
        return nn.ELU()
    elif ACTIVATION_TYPE == 'GELU':
        return nn.GELU()
    elif ACTIVATION_TYPE == 'LeakyReLU':
        return nn.LeakyReLU()
    elif ACTIVATION_TYPE == 'ReLU':
        return nn.ReLU()
    elif ACTIVATION_TYPE == 'CELU':
        return nn.CELU()
    elif ACTIVATION_TYPE == 'Sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"{ACTIVATION_TYPE} is invalid")


class GaussianFourierTimeEmbedding(nn.Module):
    """
    Scalar t in [0,1] -> Fourier feature vector.
    t:   (B, 1)
    out: (B, embed_dim)   (embed_dim even: sin half + cos half)
    """
    #Default settings from Song+21
    def __init__(self, embed_dim=64, scale=16.0, learnable=False):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        freqs = torch.randn(embed_dim // 2) * scale          # one freq per sin/cos pair
        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)             # fixed; moves with .to(device), saved in state_dict

    def forward(self, t):
        proj = 2.0 * math.pi * t * self.freqs                # (B,1)*(D/2,) -> (B, D/2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)   # (B, embed_dim)

class AddNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, ACTIVATION_TYPE, FINAL_ACTIVATION, ADD_INFO = False, INCLUDE_OUTPUT_MASK = False, LAYER_NORM = True, GFTE = False, GFTE_embed_dim=64, GFTE_scale=16.0, GFTE_learnable=False):
        super().__init__()
        self.ADD_INFO = ADD_INFO
        self.GFTE = GFTE
        self.n_layers = n_layers
        self.INCLUDE_OUTPUT_MASK = INCLUDE_OUTPUT_MASK
        
        if GFTE:
            t_embed = GFTE_embed_dim - 1
            self.t_embed_transform = GaussianFourierTimeEmbedding(embed_dim = GFTE_embed_dim, scale = GFTE_scale, learnable = GFTE_learnable)
        else:
            t_embed = 0
            
        input_layers = []
        input_layers.append(nn.Linear(input_dim + t_embed, hidden_dim))
        input_layers.append(activation(ACTIVATION_TYPE))
        
        self.body_blocks = nn.ModuleList()
        for _ in range(n_layers):
            block_layers = []
            if LAYER_NORM:
                block_layers.append(nn.LayerNorm(hidden_dim))
            block_layers.append(nn.Linear(hidden_dim, hidden_dim))
            block_layers.append(activation(ACTIVATION_TYPE))
            self.body_blocks.append(nn.Sequential(*block_layers))
        
        output_layers = []
        #final layer and potential activation
        output_layers.append(nn.Linear(hidden_dim, output_dim))
        if FINAL_ACTIVATION is not None:
            output_layers.append(activation(FINAL_ACTIVATION))
            
        self.input_net = nn.Sequential(*input_layers)
        self.output_net = nn.Sequential(*output_layers)
            
    def forward(self, t, masked_x, output_mask, masked_data, input_mask):
    
        if self.GFTE:
            t_input = self.t_embed_transform(t)
        else:
            t_input = t
            
        if self.INCLUDE_OUTPUT_MASK:
            inp = torch.cat([t_input, masked_x, output_mask, masked_data, input_mask], dim=1)
        else:
            inp = torch.cat([t_input, masked_x, masked_data, input_mask], dim=1)

        input_layer_out = self.input_net(inp)
        
        x = input_layer_out
        for block in self.body_blocks:
            x = block(x) + input_layer_out  # residual connection
                
        if self.ADD_INFO:
            return torch.add(self.output_net(torch.add(x, input_layer_out)), (masked_x*output_mask))
        else:
            return self.output_net(torch.add(x, input_layer_out))
            

class CatNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, ACTIVATION_TYPE, FINAL_ACTIVATION, ADD_INFO = False, INCLUDE_OUTPUT_MASK = False, LAYER_NORM = False):
        super().__init__()
        self.ADD_INFO = ADD_INFO
        self.n_layers = n_layers
        self.INCLUDE_OUTPUT_MASK = INCLUDE_OUTPUT_MASK
        input_layers = []
        input_layers.append(nn.Linear(input_dim, hidden_dim))
        input_layers.append(activation(ACTIVATION_TYPE))
        
        
        self.body_blocks = nn.ModuleList()
        for _ in range(n_layers):
            block_layers = []
            if LAYER_NORM:
                block_layers.append(nn.LayerNorm(hidden_dim + input_dim))
            block_layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            block_layers.append(activation(ACTIVATION_TYPE))
            self.body_blocks.append(nn.Sequential(*block_layers))
        
        output_layers = []
        #final layer and potential activation
        output_layers.append(nn.Linear(hidden_dim + input_dim, output_dim))
        if FINAL_ACTIVATION is not None:
            output_layers.append(activation(FINAL_ACTIVATION))
            
        self.input_net = nn.Sequential(*input_layers)
        self.output_net = nn.Sequential(*output_layers)
            
    def forward(self, t, masked_x, output_mask, masked_data, input_mask):
        if self.INCLUDE_OUTPUT_MASK:
            inp = torch.cat([t, masked_x, output_mask, masked_data, input_mask], dim=1)
        else:
            inp = torch.cat([t, masked_x, masked_data, input_mask], dim=1)

        x_body = self.input_net(inp)
        
        for i in range(self.n_layers):
            x_body = self.body_blocks[i](torch.cat([inp, x_body], dim=1))
                
        if self.ADD_INFO:
            return torch.add(self.output_net(torch.cat([inp, x_body], dim=1)), (masked_x*output_mask))
        else:
            return self.output_net(torch.cat([inp, x_body], dim=1))
        
class DenseNeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, ACTIVATION_TYPE, output_dim, final_act):
        super().__init__()
        layers = []
        #input layer and activation
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation(ACTIVATION_TYPE))
        
        #hidden layers and activations
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation(ACTIVATION_TYPE))
        
        #final layer and potential activation
        layers.append(nn.Linear(hidden_dim, output_dim))
        if final_act is not None:
            layers.append(activation(final_act))
            
        self.net = nn.Sequential(*layers)

    def forward(self, inp):
        return self.net(inp)

class Autoencoder_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, encoder_layers = 2, decoder_layers = 2, bottleneck = 16, ACTIVATION_TYPE = 'ELU', FINAL_ACTIVATION = None, IOB_METHOD = False, SPARSE_AUTOENCODER = False, K_LATENT_ALLOWED=16):
        super().__init__()
        self.bottleneck = bottleneck
        self.IOB_METHOD = IOB_METHOD
        self.SPARSE_AUTOENCODER = SPARSE_AUTOENCODER
        self.K_LATENT_ALLOWED = K_LATENT_ALLOWED
        self.Encoder = DenseNeuralNet(
            input_dim = input_dim, #masked data, input mask, output mask
            hidden_dim = hidden_dim,
            n_layers = encoder_layers,
            ACTIVATION_TYPE = ACTIVATION_TYPE,
            output_dim = bottleneck,
            final_act = ACTIVATION_TYPE)
        self.Decoder = DenseNeuralNet(
            input_dim = bottleneck,
            hidden_dim = hidden_dim,
            n_layers = decoder_layers,
            ACTIVATION_TYPE = ACTIVATION_TYPE,
            output_dim = output_dim,
            final_act = FINAL_ACTIVATION,
        )
    def forward(self, masked_data, input_mask, MASK_FRACTION = 1, TRAINING = True):
        """
        MANDATORY INPUTS:
        masked_data: (batch,feature_dim)
        input_mask: (batch,feature_dim)
        output_mask: (batch,feature_dim)
        OPTIONAL
        MASK_FRACTION: [0,1]
        TRAINING: [True, False]
        """
        batch_size = masked_data.size(dim=0)
        latent_dim = self.bottleneck
        en_inp = torch.cat([masked_data, input_mask], dim = 1)
        embedding = self.Encoder(en_inp)
        #MASK_FRACTION sets how many latent space variables are passed to the decoder in IOB method. Ignored if IOB_METHOD == False
        if self.IOB_METHOD and TRAINING:
            #create random mask that controls how many variables are passed through
            mask = ((torch.arange(latent_dim)/latent_dim) < torch.rand((batch_size,1)))
            #override inp with real values for latent variables that pass the cut.
            de_inp = mask*embedding
        elif self.IOB_METHOD:
            #create fixed mask that controls how many variables are passed through
            mask = ((torch.arange(latent_dim)/latent_dim) < (MASK_FRACTION*torch.ones((batch_size,1))))
            #override inp with real values for latent variables that pass the cut.
            de_inp = mask*embedding
        elif self.SPARSE_AUTOENCODER:
            #create a mask that hides all nodes with activations less than the self.K_LATENT_ALLOWED (kth) node
            mask = (torch.argsort(embedding, dim = 1, descending = True) < self.K_LATENT_ALLOWED)
            #mask embedding space
            de_inp = mask*embedding
        else:
            de_inp = embedding
            
        return self.Decoder(de_inp)
        
        
class ConditionalFlowModel_Bottleneck(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, encoder_layers = 2, decoder_layers = 2, bottleneck = 16, ACTIVATION_TYPE = 'ELU', FINAL_ACTIVATION = None, IOB_METHOD = False, SPARSE_AUTOENCODER = False, K_LATENT_ALLOWED=16, ADD_INFO = False, INCLUDE_OUTPUT_MASK = False):
        super().__init__()
        self.bottleneck = bottleneck
        self.IOB_METHOD = IOB_METHOD
        self.SPARSE_AUTOENCODER = SPARSE_AUTOENCODER
        self.INCLUDE_OUTPUT_MASK = INCLUDE_OUTPUT_MASK
        self.K_LATENT_ALLOWED = K_LATENT_ALLOWED
        self.ADD_INFO = ADD_INFO
        self.Encoder = DenseNeuralNet(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            n_layers = encoder_layers,
            ACTIVATION_TYPE = ACTIVATION_TYPE,
            output_dim = bottleneck,
            final_act = ACTIVATION_TYPE,
        )
        self.Decoder = DenseNeuralNet(
            input_dim = bottleneck,
            hidden_dim = hidden_dim,
            n_layers = decoder_layers,
            ACTIVATION_TYPE = ACTIVATION_TYPE,
            output_dim = output_dim,
            final_act = FINAL_ACTIVATION,
        )
    def forward(self, t, x, output_mask, masked_data, input_mask, MASK_FRACTION = 1, TRAINING = True):
        """
        MANDATORY INPUTS:
        t:    (batch,1)
        x:    (batch,feature_dim)
        output_mask: (batch,feature_dim)
        masked_data: (batch,feature_dim)
        input_mask: (batch,feature_dim)
        OPTIONAL
        MASK_FRACTION: [0,1]
        TRAINING: [True, False]
        """
        batch_size = t.size(dim=0)
        latent_dim = self.bottleneck
        if self.INCLUDE_OUTPUT_MASK:
            en_inp = torch.cat([t, x, output_mask, masked_data, input_mask], dim = 1)
        else:
            en_inp = torch.cat([t, x, masked_data, input_mask], dim = 1)
        embedding = self.Encoder(en_inp)
        #MASK_FRACTION sets how many latent space variables are passed to the decoder in IOB method. Ignored if IOB_METHOD == False
        if self.IOB_METHOD and TRAINING:
            #create random mask that controls how many variables are passed through
            mask = ((torch.arange(latent_dim)/latent_dim) < torch.rand((batch_size,1)))
            #override inp with real values for latent variables that pass the cut.
            de_inp = mask*embedding
        elif self.IOB_METHOD:
            #create fixed mask that controls how many variables are passed through
            mask = ((torch.arange(latent_dim)/latent_dim) < (MASK_FRACTION*torch.ones((batch_size,1))))
            #override inp with real values for latent variables that pass the cut.
            de_inp = mask*embedding
        elif self.SPARSE_AUTOENCODER:
            #create a mask that hides all nodes with activations less than the self.K_LATENT_ALLOWED (kth) node
            mask = (torch.argsort(embedding, dim = 1, descending = True) < self.K_LATENT_ALLOWED)
            #mask embedding space
            de_inp = mask*embedding
        else:
            de_inp = embedding
        
        #If using the ADD_INFO option, add data*input_mask*output_mask to the final output. This should hopefully make the model better at conditional probabilities where the target matches the input
        if self.ADD_INFO:
            return torch.add(self.Decoder(de_inp), (masked_data*output_mask))
        else:
            return self.Decoder(de_inp)
    
class ConditionalFlowModel_Flow_Only(nn.Module):
    def __init__(self,
                 input_dim, 
                 output_dim,
                 hidden_dim,
                 n_layers,
                 ACTIVATION_TYPE,
                 FINAL_ACTIVATION,
                 ADD_INFO = False,
                 INCLUDE_OUTPUT_MASK = False):
        super().__init__()
        self.ADD_INFO = ADD_INFO
        self.INCLUDE_OUTPUT_MASK = INCLUDE_OUTPUT_MASK
        self.velocity_net = DenseNeuralNet(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            n_layers = n_layers,
            ACTIVATION_TYPE = ACTIVATION_TYPE,
            final_act = FINAL_ACTIVATION,
            output_dim = output_dim,
        )

    def forward(self, t, masked_x, output_mask, masked_data, input_mask):
        """
        MANDATORY INPUTS:
        t:    (batch,1)
        x:    (batch,feature_dim)
        output_mask: (batch,feature_dim)
        masked_data: (batch,feature_dim)
        input_mask: (batch,feature_dim)
        """
        if self.INCLUDE_OUTPUT_MASK:
            inp = torch.cat([t, masked_x, output_mask, masked_data, input_mask], dim=1)
        else:
            inp = torch.cat([t, masked_x, masked_data, input_mask], dim=1)

        if self.ADD_INFO:
            return torch.add(self.velocity_net(inp), (masked_x*output_mask))
        else:
            return self.velocity_net(inp)

#max_hidden_factor must be greater than 2
class WideNet(nn.Module):
    def __init__(self, input_dim, max_hidden_factor, ACTIVATION_TYPE, output_dim, final_act, INCLUDE_OUTPUT_MASK = False):
        super().__init__()
        self.INCLUDE_OUTPUT_MASK = INCLUDE_OUTPUT_MASK
        layers = []
        
        #hidden layers and activations for increasing dimensionality
        for i in range(max_hidden_factor):
            layers.append(nn.Linear(input_dim*(2**(i)), input_dim*(2**(i+1))))
            layers.append(activation(ACTIVATION_TYPE))
            
        #hidden layers and activations for decreasing dimensionality
        for i in range(max_hidden_factor, 0, -1):
            layers.append(nn.Linear(input_dim*(2**(i)), input_dim*(2**(i-1))))
            if i > 1:
                layers.append(activation(ACTIVATION_TYPE))
            elif final_act is not None:
                layers.append(activation(final_act))
            
        self.net = nn.Sequential(*layers)

    def forward(self, t, masked_x, output_mask, masked_data, input_mask):
        """
        MANDATORY INPUTS:
        t:    (batch,1)
        x:    (batch,feature_dim)
        output_mask: (batch,feature_dim)
        masked_data: (batch,feature_dim)
        input_mask: (batch,feature_dim)
        """
        if self.INCLUDE_OUTPUT_MASK:
            inp = torch.cat([t, masked_x, output_mask, masked_data, input_mask], dim=1)
        else:
            inp = torch.cat([t, masked_x, masked_data, input_mask], dim=1)

        return self.net(inp)
