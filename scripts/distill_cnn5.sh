#!/usr/bin/env bash
python distill_train.py wandb.group=distill_cnn5 train.batch_size=512 train.optimizer.lr=4e-1 runner.local=False runner.ntrials=3 'train.kd.temperature=[_grid, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]'
python distill_train.py wandb.group=distill_cnn5 train.batch_size=512 train.optimizer.lr=4e-1 runner.local=False runner.ntrials=3 'train.kd.alpha=[_grid, 0.3, 0.5, 0.7]'

WANDB_MODE=dryrun python distill_train.py runner.ntrials=3 train.kd.temperature=4.0 train.kd.alpha=0.5

python distill_train.py wandb.group=distill_cnn5_hyperband train.batch_size=512 train.optimizer.lr=4e-1 runner.ntrials=30 runner.hyperband=True 'train.kd.temperature=[_sample, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]' 'train.kd.alpha=[_sample, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]'

python distill_train.py wandb.group=distill_ortho_cnn5 train.batch_size=512 train.optimizer.lr=4e-1 train.kd.class=KDOrthoLoss train.gradient_clip_val=0.1 runner.local=False runner.ntrials=3 'train.kd.temperature=[_grid, 2.0, 4.0, 8.0, 16.0]'

python distill_train.py wandb.group=distill_orthoclip_cnn5 train.batch_size=512 train.optimizer.lr=4e-1 train.kd.class=KDOrthoLoss runner.ntrials=3 'train.kd.temperature=[_grid, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]'
python distill_train.py wandb.group=distill_orthoclip_cnn5 train.batch_size=512 train.optimizer.lr=4e-1 train.kd.class=KDOrthoLoss +train.kd.eps=3e-3 runner.ntrials=3 'train.kd.temperature=[_grid, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]'

python distill_train.py wandb.group=distill_orthoclip_alpha train.batch_size=512 train.optimizer.lr=4e-1 train.kd.class=KDOrthoLoss runner.local=False runner.ntrials=3 train.kd.temperature=4.0 'train.kd.alpha=[_grid, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]'

python distill_train.py wandb.group=distill_orthoclip_cnn5 train.batch_size=512 train.optimizer.lr=4e-1 train.kd.class=KDOrthoLoss +train.kd.eps=3e-3 runner.ntrials=3 'train.kd.temperature=[_grid, 32.0, 64.0]'

python distill_train.py wandb.group=distill_orthoclip_hyperband train.batch_size=512 train.optimizer.lr=4e-1 kd=kdortholoss runner.ntrials=30 runner.hyperband=True 'train.kd.temperature=[_sample, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]' 'train.kd.alpha=[_sample, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]'

python distill_train.py wandb.group=distill_mse_hyperband train.batch_size=512 train.optimizer.lr=4e-1 kd=kdmseloss runner.ntrials=30 runner.hyperband=True 'train.kd.alpha=[_sample, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]'

python distill_train.py wandb.group=distill_mse_ortho_hyperband train.batch_size=512 train.optimizer.lr=4e-1 kd=kdmseortholoss runner.ntrials=30 runner.hyperband=True 'train.kd.alpha=[_sample, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]'
# Just to see if MSE is unstable to train with/without gradient clipping
python distill_train.py wandb.group=distill_mse_ortho_hyperband train.batch_size=512 train.optimizer.lr=4e-1 kd=kdmseortholoss runner.local=True train.kd.eps=1e-5 train.gradient_clip_val=0.5

# Smoothing
python distill_train.py wandb.group=distill_orthosmooth_hyperband train.batch_size=512 train.optimizer.lr=4e-1 kd=kdortholoss train.kd.eps=0.0 'train.kd.smoothing=[_sample, 0.01, 0.03, 0.1, 0.2]' runner.ntrials=30 runner.hyperband=True 'train.kd.temperature=[_sample, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]' 'train.kd.alpha=[_sample, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]'
python distill_train.py wandb.group=distill_mse_orthosmooth_hyperband train.batch_size=512 train.optimizer.lr=4e-1 kd=kdmseortholoss train.kd.eps=0.0 'train.kd.smoothing=[_sample, 0.01, 0.03, 0.1, 0.2]' runner.ntrials=30 runner.hyperband=True 'train.kd.alpha=[_sample, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]'

# Variance reduction with squared loss
python distill_train.py wandb.group=distill_mse_ortho_varred train.batch_size=512 train.optimizer.lr=4e-1 kd=kdmsevarredortholoss runner.local=True  # Seems to train fine without gradient clipping
python distill_train.py wandb.group=distill_mse_ortho_varred train.batch_size=512 train.optimizer.lr=4e-1 kd=kdmsevarredortholoss runner.ntrials=30 runner.hyperband=True 'train.kd.alpha=[_sample, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]'
python distill_train.py wandb.group=distill_mse_orthosmooth_varred train.batch_size=512 train.optimizer.lr=4e-1 kd=kdmsevarredortholoss 'train.kd.smoothing=[_sample, 0.01, 0.03, 0.1, 0.2]' runner.ntrials=30 runner.hyperband=True 'train.kd.alpha=[_sample, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]'

python distill_train.py wandb.group=distill_cnn5_crossfit dataset.split_seed=10000 +dataset.crossfit.size=5 model.teacher_checkpoint_path=checkpoints/resnet18/crossfit.ckpt train.kd.temperature=4.0 train.kd.alpha=0.5 runner.ntrials=3
python distill_train.py wandb.group=distill_mse_ortho_crossfit dataset.split_seed=10000 +dataset.crossfit.size=5 model.teacher_checkpoint_path=checkpoints/resnet18/crossfit.ckpt kd=kdmseortholoss runner.ntrials=30 runner.hyperband=True 'train.kd.alpha=[_sample, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]'
python distill_train.py wandb.group=distill_mse_ortho_varred_crossfit dataset.split_seed=10000 +dataset.crossfit.size=5 model.teacher_checkpoint_path=checkpoints/resnet18/crossfit.ckpt kd=kdmsevarredortholoss runner.ntrials=30 runner.hyperband=True 'train.kd.alpha=[_sample, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]'

python distill_train.py wandb.group=distill_cnn5_crossfit10 dataset.split_seed=12345 +dataset.crossfit.size=10 model.teacher_checkpoint_path=checkpoints/resnet18/crossfit.ckpt train.kd.temperature=4.0 train.kd.alpha=0.5 runner.ntrials=3
python distill_train.py wandb.group=distill_mse_ortho_crossfit10 dataset.split_seed=12345 +dataset.crossfit.size=10 model.teacher_checkpoint_path=checkpoints/resnet18/crossfit.ckpt kd=kdmseortholoss runner.ntrials=30 runner.hyperband=True 'train.kd.alpha=[_sample, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]'
python distill_train.py wandb.group=distill_mse_ortho_varred_crossfit10 dataset.split_seed=12345 +dataset.crossfit.size=10 model.teacher_checkpoint_path=checkpoints/resnet18/crossfit.ckpt kd=kdmsevarredortholoss runner.ntrials=30 runner.hyperband=True 'train.kd.alpha=[_sample, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]'
python distill_train.py wandb.group=distill_mse_orthosmooth_crossfit10 dataset.split_seed=12345 +dataset.crossfit.size=10 model.teacher_checkpoint_path=checkpoints/resnet18/crossfit.ckpt kd=kdmseortholoss train.kd.eps=0.0 'train.kd.smoothing=[_sample, 0.01, 0.03, 0.1, 0.2]' runner.ntrials=30 runner.hyperband=True 'train.kd.alpha=[_sample, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]'
python distill_train.py wandb.group=distill_mse_orthosmooth_varred_crossfit10 dataset.split_seed=12345 +dataset.crossfit.size=10 model.teacher_checkpoint_path=checkpoints/resnet18/crossfit.ckpt kd=kdmsevarredortholoss 'train.kd.smoothing=[_sample, 0.01, 0.03, 0.1, 0.2]' runner.ntrials=30 runner.hyperband=True 'train.kd.alpha=[_sample, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]'
# Fix alpha = 1.0, i.e. focus only on the teacher's logit
python distill_train.py wandb.group=distill_mse_orthosmooth_crossfit10 dataset.split_seed=12345 +dataset.crossfit.size=10 model.teacher_checkpoint_path=checkpoints/resnet18/crossfit.ckpt kd=kdmseortholoss train.kd.eps=0.0 'train.kd.smoothing=[_sample, 0.01, 0.03, 0.1, 0.2]' runner.ntrials=8 runner.hyperband=True train.kd.alpha=1.0
python distill_train.py wandb.group=distill_mse_orthosmooth_varred_crossfit10 dataset.split_seed=12345 +dataset.crossfit.size=10 model.teacher_checkpoint_path=checkpoints/resnet18/crossfit.ckpt kd=kdmsevarredortholoss 'train.kd.smoothing=[_sample, 0.01, 0.03, 0.1, 0.2]' runner.ntrials=8 runner.hyperband=True train.kd.alpha=1.0
# Use only 1 teacher
python distill_train.py wandb.group=distill_mse_orthosmooth_1teacher dataset.split_seed=12345 model.teacher_checkpoint_path=checkpoints/resnet18/crossfit0.ckpt kd=kdmseortholoss train.kd.eps=0.0 'train.kd.smoothing=[_sample, 0.01, 0.03, 0.1, 0.2]' runner.ntrials=30 runner.hyperband=True 'train.kd.alpha=[_sample, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]'
python distill_train.py wandb.group=distill_mse_orthosmooth_varred_1teacher dataset.split_seed=12345 model.teacher_checkpoint_path=checkpoints/resnet18/crossfit0.ckpt kd=kdmsevarredortholoss 'train.kd.smoothing=[_sample, 0.01, 0.03, 0.1, 0.2]' runner.ntrials=30 runner.hyperband=True 'train.kd.alpha=[_sample, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]'
