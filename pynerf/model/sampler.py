from torch import nn
import torch
from einops import einsum
import numpy as np


class UniformSampler(nn.Module):
    def __init__(self, n_samples, perturb):
        super().__init__()
        self.n_samples = n_samples
        self.perturb = perturb
        self.t_vals = nn.Parameter(torch.linspace(0., 1., steps=self.n_samples), requires_grad=False)

    def forward(self, rays_o, rays_d, near, far):
        n_rays = len(rays_o)
        z_vals = near * (1. - self.t_vals) + far * self.t_vals
        z_vals = z_vals.expand([n_rays, self.n_samples])
        if self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand

        #pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        pts = einsum(rays_d, z_vals, 'n d, n z -> n z d') + rays_o[:, None]
        return pts, z_vals


class PDFSampler(nn.Module):
    def __init__(self, n_samples, perturb):
        super().__init__()
        self.n_samples = n_samples
        self.perturb = perturb

    # Hierarchical sampling (section 5.2)
    @classmethod
    def sample_pdf(cls, bins, weights, N_samples, det=False):
        # Get pdf
        weights = weights + 1e-5 # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device)

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

        return samples

    def forward(self, rays_o, rays_d, z_vals, weights):
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = self.sample_pdf(z_vals_mid, weights[...,1:-1], self.n_samples, det=(self.perturb==0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = einsum(rays_d, z_vals, 'n d, n z -> n z d') + rays_o[:, None]
        return pts, z_vals
