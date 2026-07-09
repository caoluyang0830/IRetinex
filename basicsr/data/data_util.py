from basicsr.utils import img2tensor, scandir
from basicsr.data.transforms import mod_crop
from torch.nn import functional as F
from os import path as osp
import torch
import numpy as np
import cv2

cv2.setNumThreads(1)

def read_img_seq(path, require_mod_crop=False, scale=1):
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    imgs = [cv2.imread(v).astype(np.float32) / 255.0 for v in img_paths]
    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)
    return imgs
def generate_frame_indices(crt_idx, max_frame_num, num_frames, padding="reflection"):
    assert num_frames % 2 == 1, "num_frames should be an odd number."
    assert padding in (
        "replicate",
        "reflection",
        "reflection_circle",
        "circle",
    ), f"Wrong padding mode: {padding}."
    max_frame_num = max_frame_num - 1
    num_pad = num_frames // 2
    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == "replicate":
                pad_idx = 0
            elif padding == "reflection":
                pad_idx = -i
            elif padding == "reflection_circle":
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == "replicate":
                pad_idx = max_frame_num
            elif padding == "reflection":
                pad_idx = max_frame_num * 2 - i
            elif padding == "reflection_circle":
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices
def paired_paths_from_lmdb(folders, keys):
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 2, (
        "The len of keys should be 2 with [input_key, gt_key]. " f"But got {len(keys)}"
    )
    input_folder, gt_folder = folders
    input_key, gt_key = keys
    if not (input_folder.endswith(".lmdb") and gt_folder.endswith(".lmdb")):
        raise ValueError(
            f"{input_key} folder and {gt_key} folder should both in lmdb "
            f"formats. But received {input_key}: {input_folder}; "
            f"{gt_key}: {gt_folder}"
        )
    with open(osp.join(input_folder, "meta_info.txt")) as fin:
        input_lmdb_keys = [line.split(".")[0] for line in fin]
    with open(osp.join(gt_folder, "meta_info.txt")) as fin:
        gt_lmdb_keys = [line.split(".")[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(
            f"Keys in {input_key}_folder and {gt_key}_folder are different."
        )
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(
                dict([(f"{input_key}_path", lmdb_key), (f"{gt_key}_path", lmdb_key)])
            )
        return paths
def paired_paths_from_meta_info_file(folders, keys, meta_info_file, filename_tmpl):
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 2, (
        "The len of keys should be 2 with [input_key, gt_key]. " f"But got {len(keys)}"
    )
    input_folder, gt_folder = folders
    input_key, gt_key = keys
    with open(meta_info_file, "r") as fin:
        gt_names = [line.split(" ")[0] for line in fin]
    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f"{filename_tmpl.format(basename)}{ext}"
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(
            dict([(f"{input_key}_path", input_path), (f"{gt_key}_path", gt_path)])
        )
    return paths
def paired_paths_from_folder(folders, keys, filename_tmpl):
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 2, (
        "The len of keys should be 2 with [input_key, gt_key]. " f"But got {len(keys)}"
    )
    input_folder, gt_folder = folders
    input_key, gt_key = keys
    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), (
        f"{input_key} and {gt_key} datasets have different number of images: "
        f"{len(input_paths)}, {len(gt_paths)}."
    )
    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_path = input_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(input_path))
        input_name = f"{filename_tmpl.format(basename)}{ext_input}"
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, (
            f"{input_name} is not in " f"{input_key}_paths."
        )
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f"{input_key}_path", input_path), (f"{gt_key}_path", gt_path)])
        )
    return paths
def paired_DP_paths_from_folder(folders, keys, filename_tmpl):
    assert len(folders) == 3, (
        "The len of folders should be 3 with [inputL_folder, inputR_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 3, (
        "The len of keys should be 2 with [inputL_key, inputR_key, gt_key]. "
        f"But got {len(keys)}"
    )
    inputL_folder, inputR_folder, gt_folder = folders
    inputL_key, inputR_key, gt_key = keys
    inputL_paths = list(scandir(inputL_folder))
    inputR_paths = list(scandir(inputR_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(inputL_paths) == len(inputR_paths) == len(gt_paths), (
        f"{inputL_key} and {inputR_key} and {gt_key} datasets have different number of images: "
        f"{len(inputL_paths)}, {len(inputR_paths)}, {len(gt_paths)}."
    )
    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        basename, ext = osp.splitext(osp.basename(gt_path))
        inputL_path = inputL_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(inputL_path))
        inputL_name = f"{filename_tmpl.format(basename)}{ext_input}"
        inputL_path = osp.join(inputL_folder, inputL_name)
        assert inputL_name in inputL_paths, (
            f"{inputL_name} is not in " f"{inputL_key}_paths."
        )
        inputR_path = inputR_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(inputR_path))
        inputR_name = f"{filename_tmpl.format(basename)}{ext_input}"
        inputR_path = osp.join(inputR_folder, inputR_name)
        assert inputR_name in inputR_paths, (
            f"{inputR_name} is not in " f"{inputR_key}_paths."
        )
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict(
                [
                    (f"{inputL_key}_path", inputL_path),
                    (f"{inputR_key}_path", inputR_path),
                    (f"{gt_key}_path", gt_path),
                ]
            )
        )
    return paths
def paths_from_folder(folder):
    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths
def paths_from_lmdb(folder):
    if not folder.endswith(".lmdb"):
        raise ValueError(f"Folder {folder}folder should in lmdb format.")
    with open(osp.join(folder, "meta_info.txt")) as fin:
        paths = [line.split(".")[0] for line in fin]
    return paths
def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    from scipy.ndimage import filters as filters
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] = 1
    return filters.gaussian_filter(kernel, sigma)
def duf_downsample(x, kernel_size=13, scale=4):
    assert scale in (2, 3, 4), f"Only support scale (2, 3, 4), but got {scale}."
    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    pad_w, pad_h = kernel_size // 2 + scale * 2, kernel_size // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), "reflect")
    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = (
        torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(0).unsqueeze(0)
    )
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x
