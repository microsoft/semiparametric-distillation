#!/usr/bin/env bash
python train.py train.batch_size=512 train.optimizer.lr=4e-1 model=resnet18 runner=pl +save_checkpoint_path=checkpoints/cifar10/resnet18/final.ckpt
python train.py wandb.group=cifar10-cnn5-scratch train.batch_size=512 train.optimizer.lr=4e-1 model=cnn5 runner.ntrials=3
python train.py train.batch_size=512 train.optimizer.lr=4e-1 model=resnet18 dataset.split_seed=10000 runner=pl +save_checkpoint_path=checkpoints/resnet18/final.ckpt
python train.py train.batch_size=512 train.optimizer.lr=4e-1 model=resnet18 dataset.split_seed=12345 runner=pl +save_checkpoint_path=checkpoints/resnet18/final.ckpt

index=0; python train.py dataset.split_seed=10000 +dataset.crossfit.size=5 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt
index=1; python train.py dataset.split_seed=10000 +dataset.crossfit.size=5 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt
index=2; python train.py dataset.split_seed=10000 +dataset.crossfit.size=5 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt
index=3; python train.py dataset.split_seed=10000 +dataset.crossfit.size=5 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt
index=4; python train.py dataset.split_seed=10000 +dataset.crossfit.size=5 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt

index=0; python train.py dataset.split_seed=12345 +dataset.crossfit.size=10 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt
index=1; python train.py dataset.split_seed=12345 +dataset.crossfit.size=10 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt
index=2; python train.py dataset.split_seed=12345 +dataset.crossfit.size=10 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt
index=3; python train.py dataset.split_seed=12345 +dataset.crossfit.size=10 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt
index=4; python train.py dataset.split_seed=12345 +dataset.crossfit.size=10 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt
index=5; python train.py dataset.split_seed=12345 +dataset.crossfit.size=10 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt
index=6; python train.py dataset.split_seed=12345 +dataset.crossfit.size=10 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt
index=7; python train.py dataset.split_seed=12345 +dataset.crossfit.size=10 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt
index=8; python train.py dataset.split_seed=12345 +dataset.crossfit.size=10 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt
index=9; python train.py dataset.split_seed=12345 +dataset.crossfit.size=10 +dataset.crossfit.index=$index runner=pl +save_checkpoint_path=checkpoints/resnet18/crossfit$index.ckpt

python train.py dataset=cifar100 train.batch_size=512 train.optimizer.lr=4e-1 model=resnet18 runner=pl +save_checkpoint_path=checkpoints/cifar100/resnet18/final.ckpt

python train.py wandb.group=cifar100-cnn5-scratch dataset=cifar100 train.batch_size=512 train.optimizer.lr=4e-1 model=cnn5 runner.ntrials=3

# Train teachers for crossfitting again. Last time we used batch_size=512
index=0; python train.py dataset=cifar10 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar10/resnet18/crossfit$index.ckpt
index=1; python train.py dataset=cifar10 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar10/resnet18/crossfit$index.ckpt
index=2; python train.py dataset=cifar10 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar10/resnet18/crossfit$index.ckpt
index=3; python train.py dataset=cifar10 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar10/resnet18/crossfit$index.ckpt
index=4; python train.py dataset=cifar10 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar10/resnet18/crossfit$index.ckpt
index=5; python train.py dataset=cifar10 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar10/resnet18/crossfit$index.ckpt
index=6; python train.py dataset=cifar10 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar10/resnet18/crossfit$index.ckpt
index=7; python train.py dataset=cifar10 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar10/resnet18/crossfit$index.ckpt
index=8; python train.py dataset=cifar10 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar10/resnet18/crossfit$index.ckpt
index=9; python train.py dataset=cifar10 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar10/resnet18/crossfit$index.ckpt


index=0; python train.py dataset=cifar100 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar100/resnet18/crossfit$index.ckpt
index=1; python train.py dataset=cifar100 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar100/resnet18/crossfit$index.ckpt
index=2; python train.py dataset=cifar100 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar100/resnet18/crossfit$index.ckpt
index=3; CUDA_VISIBLE_DEVICES=0 python train.py dataset=cifar100 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar100/resnet18/crossfit$index.ckpt
index=4; CUDA_VISIBLE_DEVICES=1 python train.py dataset=cifar100 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar100/resnet18/crossfit$index.ckpt
index=5; CUDA_VISIBLE_DEVICES=0 python train.py dataset=cifar100 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar100/resnet18/crossfit$index.ckpt
index=6; CUDA_VISIBLE_DEVICES=1 python train.py dataset=cifar100 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar100/resnet18/crossfit$index.ckpt
index=7; CUDA_VISIBLE_DEVICES=0 python train.py dataset=cifar100 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar100/resnet18/crossfit$index.ckpt
index=8; CUDA_VISIBLE_DEVICES=1 python train.py dataset=cifar100 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar100/resnet18/crossfit$index.ckpt
index=9; CUDA_VISIBLE_DEVICES=0 python train.py dataset=cifar100 dataset.split_size=10 dataset.crossfit_index=$index runner=pl +save_checkpoint_path=checkpoints/cifar100/resnet18/crossfit$index.ckpt
