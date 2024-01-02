from torch import nn
import torch
from torch.nn import functional as F
from einops import repeat, rearrange
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from collections import defaultdict
from .sampler import UniformSampler, PDFSampler
from .render import Render
from ..data.utils import get_rays_np


class Embedder(nn.Module):
    """
    NOTE: has no trainable parameters
    """
    def __init__(self, input_dims, include_input, num_freqs, log_sampling):
        super().__init__()
        self.input_dims = input_dims
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = [torch.sin, torch.cos]

        if log_sampling:
            self.freq_bands = 2.**torch.linspace(0., num_freqs-1, steps=num_freqs)
        else:
            self.freq_bands = torch.linspace(2.**0., 2.**(num_freqs-1), steps=num_freqs)

        self.out_dim = (len(self.periodic_fns) * len(self.freq_bands)) * input_dims
        if include_input:
            self.out_dim += input_dims
        
    def forward(self, inputs):
        out = [inputs] if self.include_input else []
        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                out.append(p_fn(freq * inputs))
        return torch.cat(out, -1)
    

class NeRFField(nn.Module):
    def __init__(self, multires, multires_views, netdepth, netwidth, skips, use_viewdirs):
        super().__init__()
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.output_ch = 4
        self.loc_encoder = Embedder(3, True, multires, True)
        self.dir_encoder = Embedder(3, True, multires_views, True)

        co = self.loc_encoder.out_dim
        self.pts_linears = nn.ModuleList(
            [nn.Linear(co, netwidth)] + 
            [
                nn.Linear(netwidth + (co if i in skips else 0), netwidth)
                for i in range(netdepth - 1)
            ]
        )
        
        cd = self.dir_encoder.out_dim
        self.views_linears = nn.ModuleList([nn.Linear(cd + netwidth, netwidth//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(netwidth, netwidth)
            self.alpha_linear = nn.Linear(netwidth, 1)
            self.rgb_linear = nn.Linear(netwidth // 2, 3)
        else:
            self.output_linear = nn.Linear(netwidth, self.output_ch)

    def forward(self, pts, dir):
        """
        Args:
            pts: [B, n_sample, 3]. sampled points in a batch of rays
            dir: [B, 3]
        Returns:
            {
                'rgb': [B, n_sample, 3]
                'density': [B, n_sample, 1]
            }
        """
        pts_embed = self.loc_encoder(pts)

        h = pts_embed
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([pts_embed, h], -1)

        if self.use_viewdirs:
            dir_embed = repeat(self.dir_encoder(dir), 'n x -> n z x', z=pts.shape[1])
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, dir_embed], -1)
            for i, layer in enumerate(self.views_linears):
                h = layer(h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
        else:
            outputs = self.output_linear(h)
            rgb = outputs[..., :3]
            alpha = outputs[..., -1:]

        return {
            'rgb': rgb,
            'density': alpha
        }


class VanillaNeRF(nn.Module):
    def __init__(self, n_samples, n_importance, multires, multires_views, netdepth, netdepth_fine, netwidth, netwidth_fine, use_viewdirs, perturb):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.perturb = perturb
        self.skip = [4]

        # fields:
        self.coarse_sampler = UniformSampler(n_samples, perturb)
        self.coarse_model = NeRFField(multires, multires_views, netdepth, netwidth, self.skip, use_viewdirs)

        self.fine_sampler = PDFSampler(n_importance, perturb)
        self.fine_model = NeRFField(multires, multires_views, netdepth_fine, netwidth_fine, self.skip, use_viewdirs)

        self.render = Render(False, 1.0)

        self.mse = nn.MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def forward(self, ori, dir, near, far, **_):
        """
        The inputs could be a `RayBundle` object, which includes all the information of each ray:
            * origin: [B, 3]
            * direction: [B, 3]
            * near: [B, 1]
            * far: [B, 1]
            * other metadata
        """
        if self.use_viewdirs:
            viewdirs = dir / torch.norm(dir, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        # coarse forward
        pts, z_vals = self.coarse_sampler(ori, dir, near, far)  # [B, n_samples, 3]
        coarse_output = self.coarse_model(pts, viewdirs)      # [B, n_samples, rgbw=4]
        coarse_output = self.render(coarse_output, z_vals, dir)

        # fine forward
        pts, z_vals = self.fine_sampler(ori, dir, z_vals, coarse_output['weights'])
        fine_output = self.fine_model(pts, viewdirs)
        fine_output = self.render(fine_output, z_vals, dir)

        return {
            'coarse_rgb': coarse_output['rgb'],
            'coarse_depth': coarse_output['depth'],
            'coarse_acc': coarse_output['acc'],
            'fine_rgb': coarse_output['rgb'],
            'fine_depth': coarse_output['depth'],
            'fine_acc': coarse_output['acc'],
        }

    def get_output_for_camera(self, c2w, K, near, far, H, W, ndc, chunk_size, dev):
        """
        1. gen raybundle for single image
        2. inference chunk by chunk
        3. merge all the sub-results
        4. return the rendered image.
        5. [optional] return the metric and loss

        Args:
            c2w: [3, 4] np.ndarray
            K: [3, 3] matrix
            near, far, H, W: float
        TODO: 
            too many args;
            matrix are not tensor;
            try using something like `TestDataset` to simplify;
        """
        rays_o, rays_d = get_rays_np(H, W, K, c2w, ndc)
        rays_o = rearrange(torch.tensor(rays_o), 'h w c -> (h w) c', c=3)
        rays_d = rearrange(torch.tensor(rays_d), 'h w c -> (h w) c', c=3)
        near = torch.full((1, 1), near, device=dev)
        far = torch.full((1, 1), far, device=dev)

        results = defaultdict(list)
        with torch.no_grad():
            for i in range(0, len(rays_o), chunk_size):
                co = rays_o[i: i+chunk_size].to(dev)
                cd = rays_d[i: i+chunk_size].to(dev)
                subres = self.forward(co, cd, near, far)
                for k, v in subres.items():
                    results[k].append(v)
        for k, v in results.items():
            results[k] = torch.cat(v, dim=0).reshape(H, W, -1)
        return results

    def compute_loss(self, pred, batch):
        coarse_loss = self.mse(pred['coarse_rgb'], batch["gt_image"])
        fine_loss = self.mse(pred['fine_rgb'], batch["gt_image"])
        return coarse_loss + fine_loss

    def compute_metrics(self, pred, batch):
        ret = dict(
            coarse_psnr=float(self.psnr(pred['coarse_rgb'], batch["gt_image"])),
            fine_psnr=float(self.psnr(pred['fine_rgb'], batch["gt_image"]))
        )
        if_dim4 = {}
        if batch["gt_image"].ndim == 3:
            # clip to solve: Expected both input arguments to be normalized tensors with shape ... when all values are expected to be in the [0, 1] range.
            fine_rgb = torch.clip(repeat(pred["fine_rgb"], 'h w c -> n c h w', n=1), .0, 1.)
            image = torch.clip(repeat(batch["gt_image"], 'h w c -> n c h w', n=1), .0, 1.)
            if_dim4 = dict(
                fine_ssim=float(self.ssim(fine_rgb, image)),
                fine_lpips=float(self.lpips(fine_rgb, image))
            )
        return {**ret, **if_dim4}
