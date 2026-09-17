import torch
from torch import nn
import numpy as np
import math


def circdiff(circular1, circular2): # abs diff betwee angles
    return torch.abs(torch.atan2(torch.sin(circular1 - circular2), torch.cos(circular1 - circular2)))


def angle_diff_signed(a, b):
    """Signed angular difference a - b, wrapped to [-pi, pi]."""
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def wrapped_GMM_nll(GMM_params, target, reduction='mean'):
    """
    Compute the negative log-likelihood of [speed, angle] under a SWGMM (Eq. 2).

    Args:
        GMM_params: (B, K, 6) with [w, mu_s, mu_a, var_s, var_a, rho]
            - w should already be softmaxed over K and >= 0
            - var_* must be > 0
            - rho in (-1, 1)
        target:     (B, 2) with [speed, angle]
        reduction:  'mean' | 'none'
        Returns:    scalar if 'mean', else (B,) per-sample NLL
        
        means: Tensor of shape (batch_size, 2), means for speed and motion_angle.
        vars: Tensor of shape (batch_size, 2), variances for speed and motion_angle.
        targets: Tensor of shape (batch_size, 2), target speed and motion_angle.
        correlations: Tensor of shape (batch_size,), correlation coefficients (rho) between speed and motion_angle.

    Returns:
        Tensor: Scalar loss (mean negative log-likelihood over the batch).
        
    Winding order when computing likelihood:
        1. wrap the angular residual once per component -> nearest copy of the
           observed angle to mu_a, residual in [-pi, pi]
        2. add 2*pi*k to that residual and do not wrap again
    so the truncated sum is centred on the dominant term.
    """
    
    B, K, P = GMM_params.shape
    assert P == 6, f"Expected last dim 6, got {P}"

    # Unpack
    w     = GMM_params[:, :, 0].clamp_min(1e-12)             # (B,K)
    mu_s  = GMM_params[:, :, 1]                               # (B,K)
    mu_a  = GMM_params[:, :, 2]                               # (B,K)
    var_s = GMM_params[:, :, 3].clamp_min(1e-12)              # (B,K)
    var_a = GMM_params[:, :, 4].clamp_min(1e-12)              # (B,K)
    rho   = GMM_params[:, :, 5].clamp(-0.999, 0.999)          # (B,K)

    std_s = var_s.sqrt()
    std_a = var_a.sqrt()
    denom = (1 - rho**2).clamp_min(1e-12)                    # (B,K)

    s = target[:, 0].unsqueeze(-1)                           # (B,1)
    a = target[:, 1].unsqueeze(-1)                           # (B,1)

    ds = s - mu_s
    da0 = angle_diff_signed(a, mu_a)                         # (B,K) in [-pi,pi]
    k = torch.tensor([-1.0, 0.0, 1.0],
                     device=a.device, dtype=a.dtype).view(1, 3, 1)    # (1,3,1)
    da = da0.unsqueeze(1) + 2 * torch.pi * k                   # (B,3,K)

    ns = (ds / std_s).unsqueeze(1)                            # (B,1,K)
    na = da / std_a.unsqueeze(1)                              # (B,3,K)
    r  = rho.unsqueeze(1)                                     # (B,1,K)
    dn = denom.unsqueeze(1)                                   # (B,1,K)
    
    quad = (ns ** 2 - 2 * r * ns * na + na ** 2) / dn         # (B,3,K)
    log_norm = (torch.log(2 * torch.pi * std_s * std_a)
                + 0.5 * torch.log(denom)).unsqueeze(1)        # (B,1,K)
    comp_logp = -0.5 * quad - log_norm                        # (B,3,K)
    
    # Sum over windings (3) and components K, in log space
    log_p = torch.logsumexp(torch.log(w).unsqueeze(1) + comp_logp, dim=(1, 2))  # (B,)
    log_p = log_p.clamp_min(math.log(1e-12))   # same density floor as before
    nll = -log_p                                              # (B,)

    if reduction == "mean":
        return nll.mean()
    elif reduction == "none":
        return nll
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    
class NLLGMMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, reduction='mean'):
        return wrapped_GMM_nll(output, target, reduction=reduction)

