# IRetinex: Inter-correction Retinex Model for Low-light Image Enhancement

This repository contains a PyTorch implementation of **IRetinex**, an inter-correction Retinex framework for low-light image enhancement. The model decomposes low-light images into illumination and reflectance components, mitigates inter-component residuals, and reconstructs enhanced images with Retinex-based multi-scale consistency supervision.

## Overview

IRetinex consists of three main parts:

- **ICRR**: Inter-component Residual Reduction module for illumination/reflectance initialization with RGB and HSV priors.
- **RCM**: Residual mitigation and Component enhancement Module.
- **Lrmc**: Retinex-based multi-scale consistency loss.

Network architecture:

![figure_arch](https://github.com/caoluyang0830/MSDI/blob/main/fig//framework.png)

Visual comparison on LOL-v1, LOL-v2-real, and LOL-v2-syn:

![figure_eval](https://github.com/caoluyang0830/MSDI/blob/main/fig//result.png)

Quantitative comparison:

![table_eval](https://github.com/caoluyang0830/MSDI/blob/main/fig/table.png)

## Dependencies

- Python 3
- PyTorch
- CUDA-enabled NVIDIA GPU
- OpenCV-Python
- TensorBoard
- einops
- numpy, scipy, scikit-image, tqdm, natsort, pyyaml

Example environment:

```bash
conda create -n iretinex python=3.8
conda activate iretinex
pip install torch torchvision
pip install opencv-python tensorboard einops numpy scipy scikit-image tqdm natsort pyyaml matplotlib
python setup.py develop --no_cuda_ext
```

## Dataset Preparation

The current option files expect LOL-style paired datasets. Update the paths in `Options/*.yml` if your datasets are stored elsewhere.

Expected examples:

```text
/data0/luyang/data/LOLv1/
  Train/input
  Train/target
  Test/input
  Test/target

/data0/luyang/data/LOLv2/
  Real_captured/Train/Low
  Real_captured/Train/Normal
  Real_captured/Test/Low
  Real_captured/Test/Normal
  Synthetic/Train/Low
  Synthetic/Train/Normal
  Synthetic/Test/Low
  Synthetic/Test/Normal
```

The dataloader converts each image to a 6-channel `RGB+HSV` tensor before feeding the network.

## Training

Train on LOL-v1:

```bash
python basicsr/train.py --opt Options/IRetinex_LOL_v1.yml --gpu_id 0
```

Train on LOL-v2-real:

```bash
python basicsr/train.py --opt Options/IRetinex_LOL_v2_real.yml --gpu_id 0
```

Train on LOL-v2-synthetic:

```bash
python basicsr/train.py --opt Options/IRetinex_LOL_v2_synthetic.yml --gpu_id 0
```

Training logs, checkpoints, and validation outputs are written under the experiment folders configured by BasicSR.

## Testing

Run enhancement with a trained checkpoint:

```bash
python Enhancement/test_from_dataset.py \
  --opt Options/IRetinex_LOL_v2_synthetic.yml \
  --weights path/to/net_g.pth \
  --dataset LOL_v2_synthetic \
  --result_dir results \
  --gpus 0
```

The script reads the validation low-light directory from the selected option file and saves enhanced RGB images to:

```text
results/<dataset>/<config>/<checkpoint_name>/
```

## Available Option Files

- `Options/IRetinex_LOL_v1.yml`
- `Options/IRetinex_LOL_v2_real.yml`
- `Options/IRetinex_LOL_v2_synthetic.yml`

Current network settings use:

```yaml
type: IRetinex
in_channels: 6
out_channels: 6
n_feat: 56
stage: 1
level: 4
num_blocks: [1,2,2,4,4]
```

## Citation

If this repository is useful for your research, please cite the corresponding IRetinex paper.

