import math
import torch
from torch.utils.data.sampler import Sampler

class EnlargedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()
        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
    def __len__(self):
        return self.num_samples
    def set_epoch(self, epoch):
        self.epoch = epoch
