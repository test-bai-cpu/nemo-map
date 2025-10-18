import torch
from torch import nn
import numpy as np


def circdiff(circular1, circular2): # abs diff betwee angles
    return torch.abs(torch.atan2(torch.sin(circular1 - circular2), torch.cos(circular1 - circular2)))


def angle_diff_signed(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def wrapped_GMM_nll(GMM_params, target, reduction='mean'):
    """
    Compute the negative log-likelihood for a wrapped bivariate GMM distribution.

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

    # 3 wraps: a-2pi, a, a+2pi  -> (B,3,1)
    twopi = 2 * torch.pi
    a_wraps = torch.stack([a - twopi, a, a + twopi], dim=1)  # (B,3,1)

    # Expand to (B,3,K)
    s_b     = s.unsqueeze(1).expand(B, 3, 1)
    
    w_b     = w.unsqueeze(1).expand(B, 3, K)
    mu_s_b  = mu_s.unsqueeze(1).expand(B, 3, K)
    mu_a_b  = mu_a.unsqueeze(1).expand(B, 3, K)
    std_s_b = std_s.unsqueeze(1).expand(B, 3, K)
    std_a_b = std_a.unsqueeze(1).expand(B, 3, K)
    rho_b   = rho.unsqueeze(1).expand(B, 3, K)
    denom_b = denom.unsqueeze(1).expand(B, 3, K)
    

    # Residuals (signed angle!)
    ds = (s_b - mu_s_b)                                      # (B,3,K)
    da = angle_diff_signed(a_wraps, mu_a_b)                  # (B,3,K)

    ns = ds / std_s_b
    na = da / std_a_b

    quad = (ns**2 - 2*rho_b*ns*na + na**2) / denom_b         # (B,3,K)
    log_norm = torch.log(2 * torch.pi * std_s_b * std_a_b) + 0.5 * torch.log(denom_b)
    comp_logp = -0.5 * quad - log_norm                       # (B,3,K)

    # Sum over components in log-space (log-sum-exp with mixture weights)
    log_mix_over_K = torch.logsumexp(torch.log(w_b) + comp_logp, dim=-1)  # (B,3)

    # Sum over the 3 wraps in probability space
    prob = torch.exp(log_mix_over_K).sum(dim=1).clamp_min(1e-12)          # (B,)
    nll = -torch.log(prob)                                                 # (B,)

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

