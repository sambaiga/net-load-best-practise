import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sinusoids(length, channels, max_timescale=10000):
    """
    Returns sinusoids for positional embedding.

    Parameters:
    - length (int): Length of the sequence.
    - channels (int): Number of channels in the positional embeddings. It should be an even number.
    - max_timescale (int, optional): Maximum timescale for the sinusoids. Defaults to 10000.

    Returns:
    torch.Tensor: Sinusoidal positional embeddings.
    """
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def rotate_half(x):
    """
    Rotate the input tensor along the last dimension by half.

    Parameters:
    - x (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Rotated tensor.
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in torch < 1.8.0

class Rotary(torch.nn.Module):
    """
    Rotary positional embedding module.

    Parameters:
    - dim (int): Dimension of the input embeddings.
    - base (int, optional): Base value for frequency calculation. Defaults to 10000.
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, inputs, seq_dim=1):
        """
        Forward pass of the rotary positional embedding module.

        Parameters:
        - inputs (torch.Tensor): Input tensor.
        - seq_dim (int, optional): Dimension representing the sequence length. Defaults to 1.

        Returns:
        torch.Tensor: Rotary positional embeddings.
        """
        x = inputs.unsqueeze(2)
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]

        cos_half = self.cos_cached.squeeze(2).permute(1, 0, 2) * x.squeeze(2).mean(-1).unsqueeze(2)
        sin_half = self.sin_cached.squeeze(2).permute(1, 0, 2) * rotate_half(x).squeeze(2).mean(-1).unsqueeze(2)
        return cos_half + sin_half



def Conv1DLayer(in_channels, out_channels, bias=True):
    """
    Creates a 1D convolutional layer with specified input and output channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is True.

    Returns:
        nn.Module: 1D convolutional layer.

    """
    # Create a 1D convolutional layer with specified parameters
    m = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
    
    # Initialize weights using Kaiming normal initialization
    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    # If bias is present, initialize it with zeros
    if m.bias is not None:
        m.bias.data.fill_(0.00)
    
    return m

class PosEmbedding(nn.Module):
    """
    Positional Embedding module that combines convolutional and sinusoidal embeddings.

    Args:
        n_channels (int): Number of input channels.
        d_model (int): Dimension of the model.
        window_size (int): Size of the window for sinusoidal positional embedding.

    """
    def __init__(self, n_channels, d_model, window_size):
        super().__init__()
        # Convolutional embedding layer
        self.emb = Conv1DLayer(n_channels, d_model)
        
        # Sinusoidal positional embedding
        self.register_buffer("positional_embedding", sinusoids(window_size, d_model))
        
        # Dimension of the model
        self.d_model = d_model
    
    def forward(self, x):
        """
        Forward pass of the PosEmbedding module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying positional embedding.

        """
        # Apply convolutional embedding, ReLU activation, and scale by sqrt(d_model)
        x = F.relu(self.emb(x.permute(0, 2, 1)).permute(0, 2, 1)) * math.sqrt(self.d_model)
        
        # Add positional embedding
        x = (x + self.positional_embedding).to(x.dtype)
        return x

class RotaryEmbedding(nn.Module):
    """
    Rotary Embedding module.

    Args:
        d_model (int): Dimension of the model.

    """
    def __init__(self, d_model):
        super().__init__()
        # Rotary embedding layer
        self.emb = Rotary(d_model)
        
    def forward(self, x):
        """
        Forward pass of the RotaryEmbedding module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying rotary embedding.

        """
        x = self.emb(x)
        return x