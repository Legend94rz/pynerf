from torch import nn
import torch
from torch.nn import functional as F
from einops import einsum


class Render(nn.Module):
    def __init__(self, white_bkgd=False, raw_noise_std=1.0):
        super().__init__()
        self.white_bkgd = white_bkgd
        self.raw_noise_std = raw_noise_std
    
    def forward(self, field_output, z_vals, dir):
        """
        Args:
            field_output: {
                'rgb': [B, n_sample, 3]
                'density': [B, n_sample, 1]
            }
            z_vals: [B, n_sample]
            dir: [B, 3]
        """
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full((len(dists), 1), 1e10, device=dists.device)], -1)
        dists = dists * torch.norm(dir, dim=-1, keepdim=True)  # [B, n_samples]

        rgb = torch.sigmoid(field_output["rgb"])  # [B, n_samples, 3]
        noise = 0.
        if self.training and self.raw_noise_std > 0.:
            noise = torch.randn_like(field_output['density']) * self.raw_noise_std
        alpha = 1.0 - torch.exp(-F.relu( field_output["density"] + noise ).squeeze(-1) * dists)   # [B, n_samples]

        # the contribution (weight) of each sample point for the final color of the ray [B, n_samples]
        weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.-alpha + 1e-10], -1), -1)[:, :-1]

        rgb_map = einsum(weights, rgb, 'n z, n z d -> n d')  # [B, 3]
        depth_map = einsum(weights, z_vals, 'n z, n z -> n') # [B,]
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))  # [B,]
        acc_map = torch.sum(weights, -1)    # [B, ]

        if self.white_bkgd:
            rgb_map = rgb_map + (1 - acc_map[..., None])

        return {
            'rgb': rgb_map,
            'weights': weights,
            'depth': depth_map,
            'acc': acc_map,
            'disp': disp_map,
        }
    
