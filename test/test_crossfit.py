import numpy as np

import torch

import hydra
from omegaconf import DictConfig

size = 10
indices_per_dataset = []
train_cfg = DictConfig({
    'batch_size': 128,
})
for index in range(size):
    dataset_cfg = DictConfig({
        '_target_': 'datamodules.CIFAR10',
        'data_dir': '/dfs/scratch0/trid/data/cifar10',
        'seed': 2357,
        'num_workers': 2,
        'split_size': size,
        'split_seed': 2357,
        'crossfit_index': index
    })
    datamodule = hydra.utils.instantiate(dataset_cfg, batch_size=train_cfg.batch_size)
    datamodule.prepare_data()
    train_loader = datamodule.train_dataloader()
    indices_per_dataset.append(train_loader.dataset.indices)

print([len(indices) for indices in indices_per_dataset])
print([max(indices) for indices in indices_per_dataset])
indices_sets = [set(indices) for indices in indices_per_dataset]

dataset_cfg = DictConfig({
    '_target_': 'datamodules.CIFAR10',
    'data_dir': '/dfs/scratch0/trid/data/cifar10',
    'seed': 2357,
    'num_workers': 2,
    'split_size': 1,
    'split_seed': 2357,
    'crossfit_index': 0
})
datamodule = hydra.utils.instantiate(dataset_cfg, batch_size=train_cfg.batch_size)
datamodule.prepare_data()
train_loader = datamodule.train_dataloader()
train_indices = train_loader.dataset.indices
assert len(train_indices) == 45000
val_loader = datamodule.val_dataloader()
val_indices = val_loader.dataset.indices
assert len(val_indices) == 5000

n_contains_train = np.array([len([index_set for index_set in indices_sets if idx in index_set])
                             for idx in train_indices])
assert np.all(n_contains_train == size - 1)
n_contains_val = np.array([len([index_set for index_set in indices_sets if idx in index_set])
                           for idx in val_indices])
assert np.all(n_contains_val == 0)
