import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embending import PosEmbedding, RotaryEmbedding

activations = [nn.ReLU(), nn.SELU(), nn.LeakyReLU(), nn.GELU(), nn.SiLU()]
def create_linear(in_channels, out_channels, bn=False):
    """
    Creates a linear layer with optional batch normalization.

    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bn (bool, optional): If True, adds batch normalization. Defaults to False.

    Returns:
        nn.Module: Linear layer with optional batch normalization.
    """
    # Create a linear layer
    m = nn.Linear(in_channels, out_channels)

    # Initialize the weights using Kaiming normal initialization with a ReLU nonlinearity
    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    # Initialize the bias to zero if present
    if m.bias is not None:
        torch.nn.init.constant_(m.bias, 0)

    # Add batch normalization if requested
    if bn:
        # Create a batch normalization layer
        bn_layer = nn.BatchNorm1d(out_channels)

        # Combine the linear layer and batch normalization into a sequential module
        m = nn.Sequential(m, bn_layer)

    return m

def FeedForward(dim, expansion_factor=2, dropout=0.0, activation=nn.GELU(), bn=True):
    """
    Creates a feedforward block composed of linear layers, activation function, and dropout.

    Parameters:
        dim (int): Dimensionality of the input.
        expansion_factor (int, optional): Expansion factor for the intermediate hidden layer. Defaults to 2.
        dropout (float, optional): Dropout probability. Defaults to 0.0 (no dropout).
        activation (torch.nn.Module, optional): Activation function. Defaults to GELU().
        bn (bool, optional): If True, adds batch normalization. Defaults to True.

    Returns:
        nn.Sequential: Feedforward block.
    """
    # Create a sequential block with linear layer, activation, and dropout
    block = nn.Sequential(
        create_linear(dim, dim * expansion_factor, bn),
        activation,
        nn.Dropout(dropout),
        create_linear(dim * expansion_factor, dim, bn),
        nn.Dropout(dropout)
    )

    return block


class MLPBlock(nn.Module):
    def __init__(self, in_size=1, latent_dim=32, features_start=16, num_layers=4, context_size=96, activation=nn.ReLU(), bn=True):
        """
        Multi-Layer Perceptron (MLP) block with configurable layers and options.

        Parameters:
            in_size (int, optional): Size of the input. Defaults to 1.
            latent_dim (int, optional): Dimensionality of the latent space. Defaults to 32.
            features_start (int, optional): Number of features in the initial layer. Defaults to 16.
            num_layers (int, optional): Number of layers in the MLP. Defaults to 4.
            context_size (int, optional): Size of the context. Defaults to 96.
            activation (torch.nn.Module, optional): Activation function. Defaults to ReLU().
            bn (bool, optional): If True, adds batch normalization. Defaults to True.
        """
        super().__init__()

        # Calculate the size of the input after flattening
        self.in_size = in_size * context_size
        self.context_size = context_size

        # Initialize a list to store the layers of the MLP
        layers = [nn.Sequential(create_linear(self.in_size, features_start, bn=False), activation)]
        feats = features_start

        # Create the specified number of layers in the MLP
        for i in range(num_layers - 1):
            layers.append(nn.Sequential(create_linear(feats, feats * 2, bn=bn), activation))
            feats = feats * 2

        # Add the final layer with latent_dim and activation, without batch normalization
        layers.append(nn.Sequential(create_linear(feats, latent_dim, bn=False), activation))

        # Create a ModuleList to store the layers
        self.mlp_network = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass of the MLP block.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP block.
        """
        # Flatten the input along dimensions 1 and 2
        x = x.flatten(1, 2)

        # Pass the input through each layer in the MLP
        for m in self.mlp_network:
            x = m(x)

        return x






class PastEncoder(nn.Module):
    def __init__(self, hparams, n_channels=1):
        """
        Encoder module for processing past sequences.

        Parameters:
            hparams (dict): Dictionary containing hyperparameters.
            n_channels (int, optional): Number of input channels. Defaults to 1.
        """
        super().__init__()

        # Calculate the number of output targets
        self.n_out = len(hparams['targets'])

        # Initialize the MLP block for encoding
        self.encoder = MLPBlock(
            in_size=hparams['emb_size'] if hparams['embed_type'] != 'None' else n_channels,
            latent_dim=hparams['latent_size'],
            features_start=hparams['latent_size'],
            num_layers=hparams['depth'],
            context_size=hparams['window_size'],
            activation=activations[hparams['activation']]
        )

        # Normalize the input using LayerNorm
        self.norm = nn.LayerNorm(n_channels)

        # Apply dropout to the input
        self.dropout = nn.Dropout(hparams['dropout'])

        # Store hyperparameters
        self.hparams = hparams

        # Embedding based on the specified type
        if hparams['embed_type'] == 'PosEmb':
            self.emb = PosEmbedding(n_channels, hparams['emb_size'], window_size=hparams['window_size'])
        elif hparams['embed_type'] == 'RotaryEmb':
            self.emb = RotaryEmbedding(hparams['emb_size'])
        elif hparams['embed_type']=='CombinedEmb':
            self.pos_emb = self.emb = PosEmbedding(n_channels, hparams['emb_size'], window_size=hparams['window_size'])
            self.rotary_emb = RotaryEmbedding(hparams['emb_size'])

    def forward(self, x):
        """
        Forward pass of the PastEncoder module.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the encoder.
        """
        # Normalize the input
        x = self.norm(x)

        # Apply embedding based on the specified type
        if self.hparams['embed_type']!='None':
            if self.hparams['embed_type']=='CombinedEmb':
                x = self.pos_emb(x) + self.rotary_emb(x)
               
            else:
                x = self.emb(x)
                
            # Apply dropout to the embedded input
            x = self.dropout(x)
       

        

        # Pass the input through the encoder
        x = self.encoder(x)

        return x
    
    

    
class FutureEncoder(nn.Module):
    def __init__(self, hparams, n_channels=1):
        """
        Encoder module for processing future sequences.

        Parameters:
            hparams (dict): Dictionary containing hyperparameters.
            n_channels (int, optional): Number of input channels. Defaults to 1.
        """
        super().__init__()

        # Calculate the number of output targets
        self.n_out = len(hparams['targets'])

        # Initialize the MLP block for encoding
        self.encoder = MLPBlock(
            in_size=hparams['emb_size'] if hparams['embed_type'] != 'None' else n_channels,
            latent_dim=hparams['latent_size'],
            features_start=hparams['latent_size'],
            num_layers=hparams['depth'],
            context_size=hparams['horizon'],
            activation=activations[hparams['activation']]
        )

        # Normalize the input using LayerNorm
        self.norm = nn.LayerNorm(n_channels)

        # Apply dropout to the input
        self.dropout = nn.Dropout(hparams['dropout'])

        # Store hyperparameters
        self.hparams = hparams

        # Embedding based on the specified type
        if hparams['embed_type'] == 'PosEmb':
            self.emb = PosEmbedding(n_channels, hparams['emb_size'], window_size=hparams['horizon'])
        elif hparams['embed_type'] == 'RotaryEmb':
            self.emb = RotaryEmbedding(hparams['emb_size'])
        elif hparams['embed_type']=='CombinedEmb':
            self.pos_emb = self.emb = PosEmbedding(n_channels, hparams['emb_size'], window_size=hparams['horizon'])
            self.rotary_emb = RotaryEmbedding(hparams['emb_size'])
        


    def forward(self, x):
        """
        Forward pass of the FutureEncoder module.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the encoder.
        """
        # Normalize the input
        x = self.norm(x)

        # Apply embedding based on the specified type
        
        if self.hparams['embed_type']=='CombinedEmb':
            x = self.pos_emb(x) + self.rotary_emb(x)
            # Apply dropout to the embedded input
            x = self.dropout(x)
            
        elif self.hparams['embed_type'] in ['PosEmb', 'RotaryEmb']:
           
            x = self.emb(x)
            # Apply dropout to the embedded input
            x = self.dropout(x)

        # Pass the input through the encoder
        x = self.encoder(x)

        return x
    
    
class MLPForecastNetwork(nn.Module):
    def __init__(self, hparams):
        """
        Multilayer Perceptron (MLP) Forecast Network for time series forecasting.

        Parameters:
            hparams (dict): Dictionary containing hyperparameters.
        """
        super().__init__()

        # Calculate the number of output targets, unknown features, and covariates
        self.n_out = len(hparams['targets'])
        self.n_unknown = len(hparams['time_varying_unknown_feature']) + self.n_out
        self.n_covariates = 2 * len(hparams['time_varying_known_categorical_feature']) + len(
            hparams['time_varying_known_feature'])
        self.n_channels = self.n_unknown + self.n_covariates

        # Initialize PastEncoder for processing past sequences
        self.encoder = PastEncoder(hparams, self.n_channels)

        # Initialize FutureEncoder for processing future sequences
        self.horizon = FutureEncoder(hparams, self.n_covariates)

        # Hyperparameters and components for decoding
        self.hparams = hparams
        if hparams['comb_type']=='attn-comb':
            self.attention = nn.MultiheadAttention(hparams['latent_size'], hparams['num_head'], dropout= hparams['dropout'])
            
        if hparams['comb_type']=='weighted-comb':
             self.gate = nn.Linear(2 * hparams['latent_size'], hparams['latent_size'])
            
        self.decoder = nn.Sequential(
            FeedForward(hparams['latent_size'], expansion_factor=1, dropout=hparams['dropout'],
                        activation=activations[hparams['activation']], bn=True)
        )

        self.activation = activations[hparams['activation']]
        self.mu = nn.Linear(hparams['latent_size'], self.n_out * hparams['horizon'])

    def forecast(self, x):
        """
        Generates forecasts for the input sequences.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing the forecast predictions.
        """
        with torch.no_grad():
            pred = self(x)

        return dict(pred=pred)

    def forward(self, x):
        """
        Forward pass of the MLPForecastNetwork.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the network.
        """
        B = x.size(0)

        # Process past sequences with the encoder
        f = self.encoder(x[:, :self.hparams['window_size'], :])

        # Process future sequences with the horizon encoder
        h = self.horizon(x[:, self.hparams['window_size']:, self.n_unknown:])
        if self.hparams['comb_type']=='attn-comb':
            ph_hf = self.attention(h.unsqueeze(0), f.unsqueeze(0), f.unsqueeze(0))[0].squeeze(0)
        elif self.hparams['comb_type']=='weighted-comb':
            # Compute the gate mechanism
            gate = self.gate(torch.cat((h, f), -1)).sigmoid()
            # Combine past and future information using the gate mechanism
            ph_hf = (1 - gate) * f + gate * h
        else:
            ph_hf = h + f

        
        # Decode the combined information
        z = self.decoder(ph_hf)
        # Compute the final output
        loc = self.mu(z).reshape(B, -1, self.n_out)

        return loc

    def step(self, batch, metric_fn):
        """
        Training step for the MLPForecastNetwork.

        Parameters:
            batch (tuple): Tuple containing input and target tensors.
            metric_fn (callable): Metric function to evaluate.

        Returns:
            tuple: Tuple containing the loss and computed metric.
        """
        x, y = batch
        B = x.size(0)

        # Forward pass to obtain predictions
        y_pred = self(x)

        # Calculate the loss
        loss = self.hparams['alpha'] * F.mse_loss(y_pred, y) + (1 - self.hparams['alpha']) * F.l1_loss(y_pred, y)

        # Compute the specified metric
        metric = metric_fn(y_pred, y)

        return loss, metric