import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# import hashlib
# import random

# import pyhash

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms


# https://stackoverflow.com/questions/48367035/how-to-hash-int-long-using-hashlib-in-python
# https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
# https://github.com/flier/pyfasthash
# https://github.com/rurban/smhasher
# def hash_int32(n):
#     return int(hashlib.md5(n.to_bytes(32, 'little')).hexdigest(), base=16) % (1 << 32)
# def hash_int64(n):
#     return pyhash.xx_64()(n.to_bytes(64, 'little'))


class SubsetwIndex(torch.utils.data.Subset):
    def __getitem__(self, idx):
        index = self.indices[idx]
        # return self.dataset[index] + (hash_int32(index), )
        # return self.dataset[index] + (hash_int64(index) % 10, )  # TODO: rerun with hash_int32 using xx_32
        return self.dataset[index] + (self.random_indices[index], ) 


class DatasetBase():
    registry = {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only register classes with @name attribute
        if hasattr(cls, 'name'):
            cls.registry[cls.name] = cls

    def __init__(self, cfg, data_dir=current_dir):
        self.cfg = cfg
        self.data_dir = data_dir

    def prepare_data(self):
        raise NotImplementedError

    def split_train_val(self, ratio=0.9):
        train_len_og = len(self.train)
        train_len = int(len(self.train) * ratio)
        if getattr(self.cfg, 'split_seed', None) is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(self.cfg.split_seed)
        self.train, self.val = torch.utils.data.random_split(self.train, (train_len, len(self.train) - train_len))
        if hasattr(self.cfg, 'crossfit') and hasattr(self.cfg.crossfit, 'index'):
            assert getattr(self.cfg, 'split_seed', None) is not None
            size, index = self.cfg.crossfit.size, self.cfg.crossfit.index
            # Those with the same split_seed should have the same random_indices
            random_indices = torch.randperm(train_len_og)
            self.train.random_indices = random_indices
            self.train.indices = [idx for idx in self.train.indices if random_indices[idx] % size != index]
        if getattr(self.cfg, 'split_seed', None) is not None:
            torch.set_rng_state(rng_state)
        if getattr(self.cfg, 'return_indices', False):
            self.train = SubsetwIndex(self.train.dataset, self.train.indices)

    def prepare_dataloader(self, batch_size, **kwargs):
        self.train_loader = torch.utils.data.DataLoader(self.train, batch_size=batch_size, shuffle=True, **kwargs)
        if hasattr(self, 'val'):
            self.val_loader = torch.utils.data.DataLoader(self.val, batch_size=batch_size, shuffle=False, **kwargs)
        if hasattr(self, 'test'):
            self.test_loader = torch.utils.data.DataLoader(self.test, batch_size=batch_size, shuffle=False, **kwargs)

    def __str__(self):
        return self.name if hasattr(self, 'name') else self.__name__


class CIFAR10(DatasetBase):
    name = 'cifar10'
    input_size = 3
    output_size = 10
    N = 1024

    def prepare_data(self):
        augment_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        transform_train = transforms.Compose(augment_list + transform_list)
        transform_eval = transforms.Compose(transform_list)
        self.train = datasets.CIFAR10(f'{self.data_dir}/{self.name}', train=True, download=True, transform=transform_train)
        self.test = datasets.CIFAR10(f'{self.data_dir}/{self.name}', train=False, transform=transform_eval)
        self.split_train_val()


class SyntheticGMM(DatasetBase):
    name = 'gmm'
    output_size = 2

    def __init__(self, cfg, data_dir=current_dir):
        super().__init__(cfg, data_dir)
        self.input_size = cfg.input_size

    def prepare_data(self):
        d = self.input_size
        N = self.cfg.train_length + self.cfg.val_length
        r = self.cfg.mean_distance_scale
        self.mu = r * torch.stack((torch.ones(d), -torch.ones(d)))

        rng_state = torch.get_rng_state()
        torch.manual_seed(self.cfg.seed)
        Z = torch.randint(2, (N, ))
        X = torch.randn(N, d) + self.mu[Z]

        dot = self.mu @ X.t()
        pstar = torch.softmax(dot, dim=0)
        # p_unnormalized = torch.exp(-torch.norm(X - mu[:, None], dim=-1)**2 / 2)
        # p_normalized = p_unnormalized / p_unnormalized.sum(dim=0)
        # assert torch.allclose(p_normalized, pstar)

        Y = (torch.rand(N) < pstar[1]).long()
        torch.set_rng_state(rng_state)

        self.train = torch.utils.data.TensorDataset(X[:self.cfg.train_length], Y[:self.cfg.train_length])
        self.val = torch.utils.data.TensorDataset(X[self.cfg.train_length:], Y[self.cfg.train_length:])
