import cv2
from pathlib import Path
import numpy as np
from einops import rearrange
from .utils import recenter_poses, normalize, poses_avg, render_path_spiral


def _load_data(datadir, factor):
    pose_arr = np.load(datadir / "poses_bounds.npy")
    poses = rearrange(pose_arr[:, :-2], "n (i j) -> i j n", i=3, j=5)
    bds = rearrange(pose_arr[:, -2:], 'n i -> i n', i=2)

    sfx = f"_{factor}" if factor else ""
    imgfiles = sorted((datadir / f"images{sfx}").glob("*.*"))
    assert poses.shape[-1] == len(imgfiles)

    h, w, _ = cv2.imread(str(imgfiles[0])).shape
    poses[:2, 4, :] = np.array([h, w]).reshape(2, 1)
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    imgs = np.stack([cv2.imread(str(x))[..., :3]/255. for x in imgfiles], -1)
    return poses, bds, imgs


def load_data(datadir, factor, recenter, bd_factor, spherify):
    datadir = Path(datadir)
    poses, bds, imgs = _load_data(datadir, factor)

    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # [3, 5, n]
    poses = rearrange(poses, 'i j n -> n i j').astype('float32')
    imgs = rearrange(imgs, 'h w c n -> n h w c')
    bds = rearrange(bds, 'i n -> n i')

    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        raise NotImplementedError()
    else:
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

    render_poses = np.array(render_poses).astype("float32")
    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, imgs.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)

    images = imgs.astype("float32")
    poses = poses.astype("float32")
    return images, poses, bds, render_poses, i_test
