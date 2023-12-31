from pathlib import Path
import numpy as np
from einops import rearrange
import torch
from tqdm import tqdm
import cv2
import pickle as pkl
from pynerf.data import load_data, get_rays_np
from pynerf.model import VanillaNeRF
dev = 'cuda'
llffhold = 8
datadir = "data/nerf_llff_data/fern"
batch_size = 1024   # number of rays
multires = 10
multires_views = 4
n_importance = 64
n_samples = 64
lr = 5e-4
raw_noise_std = 1.0
ndc = True
decay_rate = 0.1
lrate_decay = 250
decay_steps = lrate_decay * 1000
#TOTAL_STEP = int(2e5)
epochs = 60
logdir = 'logs'
expname = 'exp1'


if __name__ == "__main__":
    ## load data
    # TODO: move to `load_data` or construct Dataset class.
    images, poses, bds, render_poses, i_test = load_data(datadir, 8, True, 0.75, False)
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    render_poses = render_poses[:, :3, :4]
    print('Loaded llff', images.shape, render_poses.shape, hwf, datadir)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if llffhold > 0:
        print('Auto LLFF holdout,', llffhold)
        i_test = np.arange(images.shape[0])[::llffhold]

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])
    near = 0.
    far = 1.
    ###############

    ## prepare data:
    # TODO: build a Dataset object: generate rays online
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    rays = np.stack([get_rays_np(H, W, K, p, ndc) for p in poses[:, :3, :4]], 0) # [N, ro+rd==2, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:, None]], 1) # [N, ro+rd+rgb, H, W, 3]
    train_rays = torch.tensor(rearrange(rays_rgb[i_train], 'n f h w c -> (n h w) f c', f=3, c=3).astype('float32'))

    it_val = iter(zip(poses[i_val], torch.tensor(images[i_val])))
    it_test_pose = iter(render_poses)   # TODO: build testset. next() could offer more information: pose/image/...

    ## build model & opt
    model = VanillaNeRF(
        n_samples, n_importance, multires, multires_views,
        netdepth=8, netdepth_fine=8, netwidth=256, netwidth_fine=256,
        use_viewdirs=True, perturb=1.0
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, decay_rate ** (1 / decay_steps))

    ## training
    log = Path(logdir) / expname
    log.mkdir(parents=True, exist_ok=True)
    for e in range(epochs):
        train_rays = train_rays[torch.randperm(len(train_rays))]    # shuffle

        model.train()
        pbar = tqdm(range(0, len(train_rays), batch_size), dynamic_ncols=True)
        for istep, i in enumerate(pbar):
            batch = train_rays[i: i+batch_size].to(dev)
            batch = {
                "ori": batch[:, 0], 
                "dir": batch[:, 1],
                "near": torch.full((len(batch), 1), near, device=dev),
                "far": torch.full((len(batch), 1), far, device=dev),
                "gt_image": batch[:, 2]
            }
            output = model(**batch)
            loss = model.compute_loss(output, batch)
            metrics = model.compute_metrics(output, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            desc = f"Epoch [{e}/{epochs}], loss={float(loss):.4f}"
            for k, v in metrics.items():
                desc += f", {k}={v: .4f}"
            pbar.set_description(desc)

        # val for image
        model.eval()
        # TODO: Here, the val dataset should be responsible for the contruction of `batch` obj, and the pump into the `model.get_output`
        val_pose, val_image = next(it_val)
        batch = {
            "gt_image": val_image.to(dev)
        }
        output = model.get_output_for_camera(val_pose, K, near, far, H, W, ndc, batch_size, dev)
        loss = model.compute_loss(output, batch)
        metrics = model.compute_metrics(output, batch)
        desc = f"Epoch [{e}/{epochs}], val_loss={float(loss):.4f}"
        for k, v in metrics.items():
            desc += f", val_{k}={v: .4f}"
        print(desc)

        # test for new pose
        model.eval()
        test_pose = next(it_test_pose)
        output = model.get_output_for_camera(test_pose, K, near, far, H, W, ndc, batch_size, dev)
        cv2.imwrite(str(log / f"epoch_{e}.jpg"), torch.clip(output['fine_rgb']*255, 0, 255).to(torch.uint8).cpu().numpy())
        pkl.dump(output, open(log / f"epoch_{e}.pkl", 'wb'))
        torch.save(model.state_dict(), log / f"epoch_{e}.pt")
