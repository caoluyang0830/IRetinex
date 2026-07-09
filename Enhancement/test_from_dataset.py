import argparse
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from natsort import natsorted
from scipy.ndimage import zoom
from skimage import img_as_ubyte
from tqdm import tqdm

import Enhancement.utils as utils
from basicsr.models import create_model
from basicsr.utils.options import parse


parser = argparse.ArgumentParser(description="Image Enhancement using IRetinex")
parser.add_argument(
    "--input_dir",
    default="/data0/luyang/data/LOLv2/Synthetic/Test",
    type=str,
    help="Directory of validation images",
)
parser.add_argument(
    "--result_dir", default="../results", type=str, help="Directory for results"
)
parser.add_argument(
    "--opt",
    type=str,
    default="../Options/IRetinex_LOL_v2_synthetic.yml",
    help="Path to option YAML file.",
)
parser.add_argument(
    "--weights",
    default="../model/your_model_weights.pth",
    type=str,
    help="Path to weights",
)
parser.add_argument(
    "--dataset", default="LOL_v2_synthetic", type=str, help="Test Dataset"
)
parser.add_argument("--gpus", type=str, default="0", help="GPU devices.")
args = parser.parse_args()


def load_model(opt, weights):
    model = create_model(opt).net_g
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["params"])
    except RuntimeError:
        new_checkpoint = {}
        for key in checkpoint["params"]:
            new_checkpoint["module." + key] = checkpoint["params"][key]
        model.load_state_dict(new_checkpoint)
    print("===>Testing using weights: ", weights)
    model.cuda()
    model = nn.DataParallel(model)
    model.eval()
    return model


def run_model(model, input_tensor):
    restored = model(input_tensor)
    if isinstance(restored, (list, tuple)):
        restored = restored[-1]
    return restored


def pad_to_factor(input_tensor, factor):
    h, w = input_tensor.shape[2], input_tensor.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    return F.pad(input_tensor, (0, padw, 0, padh), "reflect"), h, w


def tensor_to_image(tensor):
    return (
        torch.clamp(tensor[:, :3, ...], 0, 1)
        .cpu()
        .detach()
        .permute(0, 2, 3, 1)
        .squeeze(0)
        .numpy()
    )


def save_image(result_dir, inp_path, restored, type_id=None):
    save_dir = os.path.join(result_dir, type_id) if type_id is not None else result_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, os.path.splitext(os.path.split(inp_path)[-1])[0] + ".png"
    )
    utils.save_img(save_path, img_as_ubyte(restored))


def run_image_folder(opt, model, result_dir, factor):
    input_dir = opt["datasets"]["val"]["dataroot_lq"]
    print(input_dir)
    input_paths = natsorted(
        glob(os.path.join(input_dir, "*.png")) + glob(os.path.join(input_dir, "*.jpg"))
    )
    with torch.inference_mode():
        for inp_path in tqdm(input_paths):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img_hsv = np.float32(utils.load_img_HSV(inp_path)) / 255.0
            img_rgb = np.float32(utils.load_img(inp_path)) / 255.0
            img_hsv = torch.from_numpy(img_hsv).permute(2, 0, 1)
            img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1)
            img = torch.cat([img_rgb, img_hsv], dim=0)
            img = F.interpolate(
                img.unsqueeze(0), scale_factor=2, mode="bilinear", align_corners=False
            )
            input_tensor = img.cuda()
            input_tensor, h, w = pad_to_factor(input_tensor, factor)
            restored = run_model(model, input_tensor)[:, :, :h, :w]
            restored = tensor_to_image(restored)
            restored = zoom(restored, (0.5, 0.5, 1), order=1)
            save_image(result_dir, inp_path, restored)


gpu_list = ",".join(str(x) for x in args.gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
print("export CUDA_VISIBLE_DEVICES=" + gpu_list)
weights = args.weights
print(f"dataset {args.dataset}")
opt = parse(args.opt, is_train=False)
opt["dist"] = False
model_restoration = load_model(opt, weights)
factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split(".")[0]
checkpoint_name = os.path.basename(args.weights).split(".")[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
os.makedirs(result_dir, exist_ok=True)
run_image_folder(opt, model_restoration, result_dir, factor)
