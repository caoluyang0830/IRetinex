import importlib
import numpy as np
import random
import torch
import torch.utils.data
from functools import partial
from os import path as osp
from pdb import set_trace as stx
from basicsr.data.prefetch_dataloader import PrefetchDataLoader
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.dist_util import get_dist_info

__all__ = ["create_dataset", "create_dataloader"]
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(data_folder)
    if v.endswith("_dataset.py")
]
_dataset_modules = [
    importlib.import_module(f"basicsr.data.{file_name}")
    for file_name in dataset_filenames
]

def create_dataset(dataset_opt):
    dataset_type = dataset_opt["type"]
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f"Dataset {dataset_type} is not found.")
    dataset = dataset_cls(dataset_opt)
    logger = get_root_logger()
    logger.info(
        f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]} ' "is created."
    )
    return dataset
def create_dataloader(
    dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None
):
    phase = dataset_opt["phase"]
    rank, _ = get_dist_info()
    if phase == "train":
        if dist:
            batch_size = dataset_opt["batch_size_per_gpu"]
            num_workers = dataset_opt["num_worker_per_gpu"]
        else:
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt["batch_size_per_gpu"] * multiplier
            num_workers = dataset_opt["num_worker_per_gpu"] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
        )
        if sampler is None:
            dataloader_args["shuffle"] = True
        dataloader_args["worker_init_fn"] = (
            partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
            if seed is not None
            else None
        )
    elif phase in ["val", "test"]:
        dataloader_args = dict(
            dataset=dataset, batch_size=1, shuffle=False, num_workers=0
        )
    else:
        raise ValueError(
            f"Wrong dataset phase: {phase}. "
            "Supported ones are 'train', 'val' and 'test'."
        )
    dataloader_args["pin_memory"] = dataset_opt.get("pin_memory", False)
    prefetch_mode = dataset_opt.get("prefetch_mode")
    if prefetch_mode == "cpu":
        num_prefetch_queue = dataset_opt.get("num_prefetch_queue", 1)
        logger = get_root_logger()
        logger.info(
            f"Use {prefetch_mode} prefetch dataloader: "
            f"num_prefetch_queue = {num_prefetch_queue}"
        )
        return PrefetchDataLoader(
            num_prefetch_queue=num_prefetch_queue, **dataloader_args
        )
    else:
        return torch.utils.data.DataLoader(**dataloader_args)
def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
