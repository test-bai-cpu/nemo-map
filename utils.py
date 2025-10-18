import numpy as np
import torch
from torch import nn


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def normalize_coords_space_time(inputs, x_min=-60, x_max=80, y_min=-40, y_max=20):
    # scale to [-1,1]
    x = (inputs[:, 0] - x_min) / (x_max - x_min)
    y = (inputs[:, 1] - y_min) / (y_max - y_min)
    x = x * 2 - 1
    y = y * 2 - 1
    
    # scale to [0,1]
    total_seconds_day = 24 * 60 * 60  # 86400
    seconds_since_midnight = (inputs[:, 2] % total_seconds_day)
    t = seconds_since_midnight / total_seconds_day
    
    return torch.stack([x, y, t], dim=-1)


def normalize_coords_siren(inputs, x_min=-60, x_max=80, y_min=-40, y_max=20):
    # scale to [-1,1]
    x = (inputs[:, 0] - x_min) / (x_max - x_min)
    y = (inputs[:, 1] - y_min) / (y_max - y_min)
    x = x * 2 - 1
    y = y * 2 - 1
    
    # scale to [-1,1]
    total_seconds_day = 24 * 60 * 60  # 86400
    seconds_since_midnight = (inputs[:, 2] % total_seconds_day)
    t = seconds_since_midnight / total_seconds_day
    t = t * 2 - 1
    
    return torch.stack([x, y, t], dim=-1)


def time_fourier_features(t, num_freqs=4, exp_scale=False):
    """
    t: (B,) in [0,1]
    returns: (B, 2*num_freqs) = [sin(2pi*f_k*t), cos(2pi*f_k*t)]
    """
    if exp_scale:
        freqs = (2 ** torch.arange(num_freqs, device=t.device, dtype=t.dtype))
    else:
        freqs = torch.arange(1, num_freqs + 1, device=t.device, dtype=t.dtype)
    phases = 2 * np.pi * t.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)


class Sine(nn.Module):
    def __init__(self, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
    
    def forward(self, input):
        return torch.sin(self.omega_0 * input)
    
    
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False, use_bias=True):
        super().__init__()
        
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.act = Sine(omega_0=omega_0)
        
        with torch.no_grad():
            if self.is_first:
                    self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                    self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                                np.sqrt(6 / self.in_features) / self.omega_0)

        if use_bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, input):
        return self.act(self.linear(input))