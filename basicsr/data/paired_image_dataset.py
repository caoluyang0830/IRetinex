from torch.utils import data as data
from torchvision.transforms.functional import normalize
from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_DP_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file,
)
from basicsr.data.transforms import (
    augment,
    paired_random_crop,
    paired_random_crop_DP,
    random_augmentation,
)
from basicsr.utils import (
    FileClient,
    imfrombytes,
    img2tensor,
    padding,
    padding_DP,
    imfrombytesDP,
)
import random
import numpy as np
import torch
import cv2
from pdb import set_trace as stx

class Dataset_PairedImage(data.Dataset):
    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None
        self.gt_folder, self.lq_folder = opt["dataroot_gt"], opt["dataroot_lq"]
        if "filename_tmpl" in opt:
            self.filename_tmpl = opt["filename_tmpl"]
        else:
            self.filename_tmpl = "{}"
        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt["client_keys"] = ["lq", "gt"]
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ["lq", "gt"]
            )
        elif "meta_info_file" in self.opt and self.opt["meta_info_file"] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder],
                ["lq", "gt"],
                self.opt["meta_info_file"],
                self.filename_tmpl,
            )
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ["lq", "gt"], self.filename_tmpl
            )
        if self.opt["phase"] == "train":
            self.geometric_augs = opt["geometric_augs"]
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )
        scale = self.opt["scale"]
        index = index % len(self.paths)
        gt_path = self.paths[index]["gt_path"]
        img_gt = cv2.imread(gt_path)
        lq_path = self.paths[index]["lq_path"]
        img_lq = cv2.imread(lq_path)
        if self.opt["phase"] == "train":
            gt_size = self.opt["gt_size"]
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}
    def __len__(self):
        return len(self.paths)
