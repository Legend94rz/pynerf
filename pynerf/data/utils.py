import numpy as np
from torch.utils.data import Dataset
from einops import einsum, repeat
import torch


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    # poses: [n, 3, 5]
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))        # z axis
    up = poses[:, :3, 1].sum(0)     # y axis (up)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w


def recenter_poses(poses: np.ndarray):
    # return: the poses to transform from each camera to the avg camera, which is considered as the new world coord
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], 1)

    poses = np.linalg.inv(c2w) @ poses      #  (world to avg_camera)  @ (camera to world)
    poses_[:, :3, :4] = poses[:, :3, :4]
    return poses_


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def get_rays_np(H, W, K, c2w, ndc):
    """
    Args:
        H, W: int.
        K: 3x3 intrinsic matrix.
        c2w: 3x4 matrix.

    Returns:
        loc: [H, W, 3] matrix
        dir: [H, W, 3] matrix
    """

    i, j = np.meshgrid(np.arange(W, dtype="float32"), np.arange(H, dtype="float32"), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    rays_d = einsum(c2w[:3, :3], dirs, 'a b, m n b -> m n a')
    rays_o = repeat(c2w[:3, -1], 'd -> h w d', h=H, w=W)
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d  # (H, W, 3) (H, W, 3)


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = np.stack([o0,o1,o2], -1)
    rays_d = np.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


class LLFFDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index):
        return None
    
    def __len__(self):
        return 0
