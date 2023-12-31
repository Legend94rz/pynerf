# pynerf

Just another implementation of [vanilla nerf]() using PyTorch.

This project reorganizes the code structure with a nerfstudio-like style, and aims to be more modular and easy to understand.

## TRAIN

`python scripts/main.py`

You might need to adjust the data path (and other params) in the `main.py`.

NOTE: Only support LLFF datasets for now.

## TODO

- [ ] Train/Val/Test dataset class
- [ ] Supoort `--no_ndc --spherify --lindisp`
- [ ] Export mesh/mp4
- [ ] Benchmark & comparison
- [ ] Config or pass hyper-params via cmd
- [ ] Other format / source of data


## THANKS

* [official nerf implementation](https://github.com/bmild/nerf)
* [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)
* [nerf_pl](https://github.com/kwea123/nerf_pl/tree/master)
* [nerf studio](https://github.com/nerfstudio-project/nerfstudio)