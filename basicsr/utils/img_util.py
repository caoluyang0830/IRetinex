import cv2
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def img2tensorhsv(imgs, bgr2hsv=True, float32=True):
    def _totensor(img, bgr2hsv, float32):
        if img.shape[2] == 3 and bgr2hsv:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img
    if isinstance(imgs, list):
        return [_totensor(img, bgr2hsv, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2hsv, float32)
def tensor2imghsv(tensor, hsv2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    if not (
        torch.is_tensor(tensor)
        or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))
    ):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")
    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False
            ).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if hsv2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_HSV2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:
                img_np = np.squeeze(img_np, axis=2)
            else:
                if hsv2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_HSV2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(
                "Only support 4D, 3D or 2D tensor. "
                f"But received with dimension: {n_dim}"
            )
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result
def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Convert BGR images to 6-channel RGB+HSV tensors for paired training."""

    def _totensor(img, bgr2rgb, float32):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        if img.shape[2] > 3:
            img = img[:, :, :3]

        if img.shape[2] == 3 and bgr2rgb:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL) / 255.0
            img = np.concatenate([img_rgb, img_hsv], axis=2)
        else:
            img = img / 255.0

        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img
    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)
def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    if not (
        torch.is_tensor(tensor)
        or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))
    ):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")
    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False
            ).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np[:, :3, ...], cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(
                "Only support 4D, 3D or 2D tensor. "
                f"But received with dimension: {n_dim}"
            )
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result
def imfrombytes(content, flag="color", float32=False):
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        "color": cv2.IMREAD_COLOR,
        "grayscale": cv2.IMREAD_GRAYSCALE,
        "unchanged": cv2.IMREAD_UNCHANGED,
    }
    if img_np is None:
        raise Exception("None .. !!!")
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.0
    return img
def imfrombytesDP(content, flag="color", float32=False):
    img_np = np.frombuffer(content, np.uint8)
    if img_np is None:
        raise Exception("None .. !!!")
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    if float32:
        img = img.astype(np.float32) / 65535.0
    return img
def padding(img_lq, img_gt, gt_size):
    h, w, _ = img_lq.shape
    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt
    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    if img_gt.ndim == 2:
        img_gt = np.expand_dims(img_gt, axis=2)
    return img_lq, img_gt
def padding_DP(img_lqL, img_lqR, img_gt, gt_size):
    h, w, _ = img_gt.shape
    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    if h_pad == 0 and w_pad == 0:
        return img_lqL, img_lqR, img_gt
    img_lqL = cv2.copyMakeBorder(img_lqL, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_lqR = cv2.copyMakeBorder(img_lqR, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    return img_lqL, img_lqR, img_gt
def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)
def crop_border(imgs, crop_border):
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [
                v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs
            ]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]
