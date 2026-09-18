import os
import numpy as np
import torch
import yaml
from torch import nn

DATASET_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_config.yaml")

# Valid values for --dataset in train.py / evaluate_NLL.py. Each needs an entry in dataset_config.yaml.
DATASET_CHOICES = ["ATC", "ETH-eth", "ETH-hotel", "UCY-students001", "UCY-students003", "UCY-zara01"]

# Suffix appended to the experiment name for each --model. siren is the main model and gets no suffix.
EXP_MODEL_SUFFIX = {
    "siren": "",
    "fourier": "_fourier",
    "time_grid": "_time_grid",
}


def get_scene_name(dataset):
    """Short lowercase scene token used in folder and file names: "ATC" -> "atc", "ETH-eth" -> "eth", "UCY-zara01" -> "zara01"."""
    scene = dataset.split("-", 1)[1] if "-" in dataset else dataset
    return scene.lower()


def get_exp_name(model_name, dataset):
    """Folder name used under models/, runs/, MoDs/, nll_results/ and results/ for one (model, dataset) pair.

    Pattern: "nemo_<scene><model suffix>", where <scene> is "atc" for ATC and the part after the
    dash for ETH/UCY ("ETH-eth" -> "eth", "UCY-zara01" -> "zara01").
        ATC        + siren     -> "nemo_atc"
        ATC        + fourier   -> "nemo_atc_fourier"
        ATC        + time_grid -> "nemo_atc_time_grid"
        ETH-eth    + siren     -> "nemo_eth"
        UCY-zara01 + fourier   -> "nemo_zara01_fourier"
    """
    return f"old_nemo_{get_scene_name(dataset)}{EXP_MODEL_SUFFIX[model_name]}"


def load_dataset_config(dataset, config_file=DATASET_CONFIG_FILE):
    """Return the entry for `dataset` (a --dataset name) from dataset_config.yaml:
    normalization bounds (x, y, time) plus the per-dataset `train` settings."""
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    if dataset not in cfg:
        raise KeyError(f"No entry for '{dataset}' in {config_file}. Available: {sorted(cfg)}")
    return cfg[dataset]


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def _normalize_xy(inputs, norm_cfg):
    # scale to [-1,1]
    x_min, x_max = norm_cfg["x"]
    y_min, y_max = norm_cfg["y"]
    x = (inputs[:, 0] - x_min) / (x_max - x_min)
    y = (inputs[:, 1] - y_min) / (y_max - y_min)
    x = x * 2 - 1
    y = y * 2 - 1
    return x, y


def _normalize_time(t, time_cfg):
    # scale to [0,1]
    mode = time_cfg["mode"]
    if mode == "day":
        total_seconds_day = 24 * 60 * 60  # 86400
        seconds_since_midnight = (t % total_seconds_day)
        return seconds_since_midnight / total_seconds_day
    if mode == "linear":
        t_min, t_max = time_cfg["range"]
        return (t - t_min) / (t_max - t_min)
    raise ValueError(f"Unknown time normalization mode '{mode}' (expected 'day' or 'linear')")


def normalize_coords_space_time(inputs, norm_cfg):
    """inputs: (B,3) = [x, y, time]; norm_cfg from load_dataset_config(). x,y -> [-1,1], t -> [0,1]."""
    x, y = _normalize_xy(inputs, norm_cfg)
    t = _normalize_time(inputs[:, 2], norm_cfg["time"])
    return torch.stack([x, y, t], dim=-1)


def normalize_coords_siren(inputs, norm_cfg):
    """inputs: (B,3) = [x, y, time]; norm_cfg from load_dataset_config(). x,y,t -> [-1,1]."""
    x, y = _normalize_xy(inputs, norm_cfg)
    t = _normalize_time(inputs[:, 2], norm_cfg["time"])
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