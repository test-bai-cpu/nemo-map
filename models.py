import torch
from torch import nn
import utils
import math


class MoDGMMFeatureTimeModel(nn.Module):
    def __init__(self, input_size, num_components, grid_size=(64, 64), feature_dim_spatial=32, time_bins=24, feature_dim_time=16):
        super(MoDGMMFeatureTimeModel, self).__init__()

        self.num_components = num_components
        self.grid_size = grid_size        # (H, W)
        self.feature_dim_spatial = feature_dim_spatial
        self.feature_dim_time = feature_dim_time
        self.time_bins = time_bins
        
        output_size = 6 * num_components

        # Create a learnable feature grid: (H, W, D, feature_dim)
        self.feature_grid = nn.Parameter(
            torch.randn(grid_size[0], grid_size[1], feature_dim_spatial)
        )
        self.time_grid = nn.Parameter(
            torch.randn(time_bins, feature_dim_time)
        )
        
        mlp_input_dim = input_size + feature_dim_spatial + feature_dim_time
        
        layers = []
        
        layers.append(nn.Linear(mlp_input_dim, 128))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(128, 64))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(64, output_size))

        self.net = nn.Sequential(*layers)

    def _get_spatial_feature(self, ix, iy):
        return self.feature_grid[iy, ix]
    
    def _interp_time_feature(self, t):
        """
        t: (B,) in [0,1] (normalized time-of-day).
        Linearly interpolate along a periodic time axis.
        """
        pos = t * self.time_bins            # in [0, K]
        base = torch.floor(pos)             # integer part
        k0 = (base.long()) % self.time_bins # left bin
        k1 = (k0 + 1) % self.time_bins      # right bin (wraps around)
        alpha = (pos - base).unsqueeze(1)   # fractional offset (B,1)

        f0 = self.time_grid[k0]             # (B, C_t)
        f1 = self.time_grid[k1]             # (B, C_t)
        return (1 - alpha) * f0 + alpha * f1

    def forward(self, coords):
        x = ((coords[:, 0] + 1) * 0.5) * (self.grid_size[1] - 1)  # width axis
        y = ((coords[:, 1] + 1) * 0.5) * (self.grid_size[0] - 1)  # height axis
        t = coords[:, 2]                              # normalized time in [0,1]

        # ---- bilinear over spatial grid ----
        x0 = torch.floor(x).long().clamp(0, self.grid_size[1] - 2)
        y0 = torch.floor(y).long().clamp(0, self.grid_size[0] - 2)
        x1 = x0 + 1
        y1 = y0 + 1

        dx = (x - x0.float()).unsqueeze(1)
        dy = (y - y0.float()).unsqueeze(1)

        f00 = self._get_spatial_feature(x0, y0)
        f01 = self._get_spatial_feature(x0, y1)
        f10 = self._get_spatial_feature(x1, y0)
        f11 = self._get_spatial_feature(x1, y1)

        spat_feat = (
            (1 - dx) * (1 - dy) * f00 +
            (1 - dx) * dy * f01 +
            dx * (1 - dy) * f10 +
            dx * dy * f11
        )  # shape: (B, feature_dim)


        # ---- linear interpolation over time grid (periodic) ----
        time_feat = self._interp_time_feature(t)  # (B, C_t)


        # ---- concat raw coords + spatial + time ----
        mlp_input = torch.cat([coords, spat_feat, time_feat], dim=-1)  # (B, 3 + C_s + C_t)

        
        params = self.net(mlp_input)
        batch_size = coords.size(0)
        
        params = params.view(batch_size, self.num_components, 6)

        raw_weights = params[:, :, 0]  # Shape: (batch_size, num_components)
        weights = torch.softmax(raw_weights, dim=-1)  # Ensure weights sum to 1
        
        # Extract and process means
        means = params[:, :, 1:3]  # Shape: (batch_size, num_components, 2)
        speed = torch.relu(means[:, :, 0])  # Ensure speed >= 0
        bounded_mean_angle = means[:, :, 1] % (2 * math.pi)  # Wrap angle to [0, 2pi]
        means = torch.stack([speed, bounded_mean_angle], dim=-1)  # Shape: (batch_size, num_components, 2)
        
        # Extract and process log variances
        log_vars = params[:, :, 3:5]  # Shape: (batch_size, num_components, 2)
        log_vars = torch.clamp(log_vars, min=-10, max=10)
        vars = torch.exp(log_vars)  # Variances
        
        # Extract and process correlation coefficients
        raw_corr_coef = params[:, :, 5]  # Shape: (batch_size, num_components)
        corr_coef = 0.99 * torch.tanh(raw_corr_coef)

        GMM_params = torch.cat(
            [
                weights.unsqueeze(-1),              # Shape: (batch_size, num_components, 1)
                means,                              # Shape: (batch_size, num_components, 2)
                vars,                               # Shape: (batch_size, num_components, 2)
                corr_coef.unsqueeze(-1)             # Shape: (batch_size, num_components, 1)
            ],
            dim=-1  # Resulting shape: (batch_size, num_components, 6)
        )

        return GMM_params, coords
    

class MoDGMMFeatureFFModel(nn.Module):
    def __init__(self, input_size, num_components, grid_size=(64, 64), feature_dim=32, num_time_freqs=4):
        super().__init__()

        self.num_components = num_components
        self.grid_size = grid_size  # (H, W), here H = W = 64
        self.feature_dim = feature_dim
        self.num_time_freqs = num_time_freqs
        
        output_size = 6 * num_components
        
        self.feature_grid = nn.Parameter(torch.randn(grid_size[0], grid_size[1], feature_dim))
        mlp_input_dim = input_size + feature_dim + 2 * num_time_freqs
        
        layers = []
        
        layers.append(nn.Linear(mlp_input_dim, 128))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(128, 64))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(64, output_size))

        self.net = nn.Sequential(*layers)

    def get_feature(self, ix, iy):
        return self.feature_grid[iy, ix]  # shape: (feature_dim,)

    def forward(self, coords):
        x = ((coords[:, 0] + 1) * 0.5) * (self.grid_size[1] - 1)  # width axis
        y = ((coords[:, 1] + 1) * 0.5) * (self.grid_size[0] - 1)  # height axis
        
        t  = coords[:, 2]

        x0 = torch.floor(x).long().clamp(0, self.grid_size[1] - 2)
        y0 = torch.floor(y).long().clamp(0, self.grid_size[0] - 2)
        x1 = x0 + 1
        y1 = y0 + 1

        dx = (x - x0.float()).unsqueeze(1)
        dy = (y - y0.float()).unsqueeze(1)

        f00 = self.get_feature(x0, y0)
        f01 = self.get_feature(x0, y1)
        f10 = self.get_feature(x1, y0)
        f11 = self.get_feature(x1, y1)

        feat = (
            (1 - dx) * (1 - dy) * f00 +
            (1 - dx) * dy * f01 +
            dx * (1 - dy) * f10 +
            dx * dy * f11
        )  # shape: (B, feature_dim)

        tfeat = utils.time_fourier_features(t, num_freqs=self.num_time_freqs, exp_scale=True)  # (B, 2*num_time_freqs) ### One param here. for use exponential scaling or linear scaling

        # Concatenate features with input coords
        mlp_input = torch.cat([coords, feat, tfeat], dim=-1)

        params = self.net(mlp_input)
        batch_size = coords.size(0)
        
        params = params.view(batch_size, self.num_components, 6)

        raw_weights = params[:, :, 0]  # Shape: (batch_size, num_components)
        weights = torch.softmax(raw_weights, dim=-1)  # Ensure weights sum to 1
        
        # Extract and process means
        means = params[:, :, 1:3]  # Shape: (batch_size, num_components, 2)
        speed = torch.relu(means[:, :, 0])  # Ensure speed >= 0
        bounded_mean_angle = means[:, :, 1] % (2 * math.pi)  # Wrap angle to [0, 2pi]
        means = torch.stack([speed, bounded_mean_angle], dim=-1)  # Shape: (batch_size, num_components, 2)
        
        # Extract and process log variances
        log_vars = params[:, :, 3:5]  # Shape: (batch_size, num_components, 2)
        log_vars = torch.clamp(log_vars, min=-10, max=10)
        vars = torch.exp(log_vars)  # Variances
        
        # Extract and process correlation coefficients
        raw_corr_coef = params[:, :, 5]  # Shape: (batch_size, num_components)
        corr_coef = 0.99 * torch.tanh(raw_corr_coef)

        GMM_params = torch.cat(
            [
                weights.unsqueeze(-1),              # Shape: (batch_size, num_components, 1)
                means,                              # Shape: (batch_size, num_components, 2)
                vars,                               # Shape: (batch_size, num_components, 2)
                corr_coef.unsqueeze(-1)             # Shape: (batch_size, num_components, 1)
            ],
            dim=-1  # Resulting shape: (batch_size, num_components, 6)
        )

        return GMM_params, coords


class MoDGMMSirenHybridModel(nn.Module):
    def __init__(self, input_size, num_components, grid_size=(64, 64), feature_dim=32):
        super().__init__()
        
        self.num_components = num_components
        self.grid_size = grid_size  # (H, W), here H = W = 64
        self.feature_dim = feature_dim
        
        hidden_xy = 64
        hidden_t = 64
        
        self.omega_0_first = 30.0
        self.omega_0_hidden = 1.0
        
        self.mode = "film"  # "concat" or "film"
        
        output_size = 6 * num_components
        
        self.feature_grid = nn.Parameter(
            torch.randn(grid_size[0], grid_size[1], feature_dim)
        )

        xy_mlp_input_dim = 2 + feature_dim

        self.xy_net = nn.Sequential(
            nn.Linear(xy_mlp_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_xy),
            nn.ReLU(),
        )
        
        self.net = []
        self.net.append(utils.SineLayer(1, 128, omega_0=self.omega_0_first, is_first=True))
        self.net.append(utils.SineLayer(128, hidden_t, omega_0=self.omega_0_hidden, is_first=False))
        self.net = nn.Sequential(*self.net)

        if self.mode == "concat":
            fused_dim = hidden_xy + hidden_t
            self.fuse = nn.Identity()
        else:
            # FiLM: gamma,beta from time features modulate spatial features
            self.film = nn.Sequential(
                nn.Linear(hidden_t, 2 * hidden_xy),
                nn.Tanh()  # keeps gamma, beta bounded; stable
            )
            fused_dim = hidden_xy
        
        self.head = nn.Linear(fused_dim, output_size)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            b = self.head.bias.view(num_components, 6)
            b[:, 0] = 0.0      # mix weight
            b[:, 1] = 0.1      # mu_speed small
            b[:, 2] = 0.0      # mu_angle ~ 0 (wrapped later)
            b[:, 3] = -1.386   # log var_speed ~ log(0.5^2)
            b[:, 4] = -1.386   # log var_angle
            b[:, 5] = 0.0      # rho logit ~ 0 => rho ~ 0

    def _get_spatial_feature(self, ix, iy):
        return self.feature_grid[iy, ix]

    def forward(self, coords):
        x_idx = ((coords[:, 0] + 1) * 0.5) * (self.grid_size[1] - 1)  # width axis
        y_idx = ((coords[:, 1] + 1) * 0.5) * (self.grid_size[0] - 1)  # height axis
        t = coords[:, 2:3]                             # x,y,t normalized time in [-1,1]
        
        # ---- bilinear over spatial grid ----
        x0 = torch.floor(x_idx).long().clamp(0, self.grid_size[1] - 2)
        y0 = torch.floor(y_idx).long().clamp(0, self.grid_size[0] - 2)
        x1 = x0 + 1
        y1 = y0 + 1

        dx = (x_idx - x0.float()).unsqueeze(1)
        dy = (y_idx - y0.float()).unsqueeze(1)

        f00 = self._get_spatial_feature(x0, y0)
        f01 = self._get_spatial_feature(x0, y1)
        f10 = self._get_spatial_feature(x1, y0)
        f11 = self._get_spatial_feature(x1, y1)

        spat_feat = (
            (1 - dx) * (1 - dy) * f00 +
            (1 - dx) * dy * f01 +
            dx * (1 - dy) * f10 +
            dx * dy * f11
        )  # shape: (B, feature_dim)

        x = coords[:, 0:1]  # (B,1)
        y = coords[:, 1:2]  # (B,1)

        xy_mlp_input = torch.cat([x, y, spat_feat], dim=-1)  # (B, 2 + C_s)
        
        h_xy = self.xy_net(xy_mlp_input)
        h_t = self.net(t)
        
        if self.mode == "concat":
            h = torch.cat([h_xy, h_t], dim=-1)
        else:
            gamma_beta = self.film(h_t)                 # (B, 2*Hxy)
            Hxy = h_xy.shape[-1]
            gamma, beta = gamma_beta.split(Hxy, dim=-1) # (B,Hxy),(B,Hxy)
            gamma = 1.0 + 0.1 * gamma
            h = gamma * h_xy + beta
        
        params = self.head(h)
        batch_size = coords.size(0)

        params = params.view(batch_size, self.num_components, 6)

        raw_weights = params[:, :, 0]  # Shape: (batch_size, num_components)
        weights = torch.softmax(raw_weights, dim=-1)  # Ensure weights sum to 1
        
        # Extract and process means
        means = params[:, :, 1:3]  # Shape: (batch_size, num_components, 2)
        speed = torch.relu(means[:, :, 0])  # Ensure speed >= 0
        bounded_mean_angle = means[:, :, 1] % (2 * math.pi)  # Wrap angle to [0, 2pi]
        means = torch.stack([speed, bounded_mean_angle], dim=-1)  # Shape: (batch_size, num_components, 2)
        
        # Extract and process log variances
        log_vars = params[:, :, 3:5]  # Shape: (batch_size, num_components, 2)
        log_vars = torch.clamp(log_vars, min=-10, max=10)
        vars = torch.exp(log_vars)  # Variances
        
        # Extract and process correlation coefficients
        raw_corr_coef = params[:, :, 5]  # Shape: (batch_size, num_components)
        corr_coef = 0.99 * torch.tanh(raw_corr_coef)

        GMM_params = torch.cat(
            [
                weights.unsqueeze(-1),
                means,
                vars,
                corr_coef.unsqueeze(-1)
            ],
            dim=-1  # Resulting shape: (batch_size, num_components, 6)
        )

        return GMM_params, coords