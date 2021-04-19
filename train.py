from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.absolute()
import os
# Add to $PYTHONPATH so that ray workers can see
os.environ['PYTHONPATH'] = str(PROJECT_ROOT) + ":" + os.environ.get('PYTHONPATH', '')

import torch
import pytorch_lightning as pl

import hydra
from omegaconf import OmegaConf

import models
import datamodules
import tasks
from pl_runner import pl_train
from tee import StdoutTee, StderrTee


class LightningModel(pl.LightningModule):

    def __init__(self, model_cfg, dataset_cfg, train_cfg):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_cfg = dataset_cfg
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.datamodule = hydra.utils.instantiate(dataset_cfg, batch_size=train_cfg.batch_size)
        self.model = hydra.utils.instantiate(model_cfg, num_classes=self.datamodule.num_classes)
        self.task = hydra.utils.instantiate(self.train_cfg.task)

    def forward(self, input):
        return self.model.forward(input)

    def _shared_step(self, batch, batch_idx, prefix='train'):
        batch_x, batch_y = batch
        out = self.forward(batch_x)
        loss = self.task.loss(out, batch_y)
        metrics = self.task.metrics(out, batch_y)
        metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
        self.log(f'{prefix}_loss', loss, on_epoch=True, prog_bar=False)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix='train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix='val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix='test')

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, self.model.parameters())
        if 'lr_scheduler' not in self.train_cfg:
            return optimizer
        else:
            lr_scheduler = hydra.utils.instantiate(self.train_cfg.lr_scheduler, optimizer)
            return [optimizer], [lr_scheduler]


@hydra.main(config_path="cfg", config_name="config.yaml")
def main(cfg: OmegaConf):
    with StdoutTee('train.stdout'), StderrTee('train.stderr'):
        print(OmegaConf.to_yaml(cfg))
        if cfg.runner.name == 'pl':
            trainer, model = pl_train(cfg, LightningModel)
        else:
            assert cfg.runner.name == 'ray', 'Only pl and ray runners are supported'
            # Shouldn't need to install ray unless doing distributed training
            from ray_runner import ray_train
            ray_train(cfg, LightningModel)


if __name__ == "__main__":
    main()


# # For interactive use
# from omegaconf import DictConfig
# dataset = 'cifar10'
# # dataset = 'cifar100'
# dataset_cfg = DictConfig({
#     '_target_': f'datamodules.{dataset.upper()}',
#     'data_dir': f'/dfs/scratch0/trid/data/{dataset}',
#     'seed': 2357,
#     'num_workers': 6,
# })
# train_cfg = DictConfig({
#     'batch_size': 512,
#     'epochs': 20,
#     'optimizer': DictConfig({'weight_decay': 5e-4, '_target_': 'torch.optim.SGD', 'lr': 1e-1, 'momentum': 0.9}),
#     'lr_scheduler': DictConfig({'_target_': 'torch.optim.lr_scheduler.MultiStepLR', 'milestones': [10, 15, 20], 'gamma': 0.2}),
#     'limit_train_batches': 0.1,
#     'task': DictConfig({'_target_': 'tasks.MulticlassClassification'}),
#     'gradient_clip_val': 0.0,
#     'verbose': True
# })
# model_cfg = DictConfig({'_target_': 'models.CNN5'})
# cfg = DictConfig({'dataset': dataset_cfg, 'model': model_cfg, 'train': train_cfg, 'seed': 0})

# lr_monitor = pl.callbacks.LearningRateLogger(logging_interval='epoch')

# # pl_train(cfg, LightningModel)
# pl_module_cls = LightningModel
# if cfg.seed is not None:
#     pl.seed_everything(cfg.seed)
# model = pl_module_cls(cfg.model, cfg.dataset, cfg.train)
# trainer = pl.Trainer(
#     # gpus=1 if config['gpu'] else None,
#     gpus=1,
#     gradient_clip_val=cfg.train.gradient_clip_val,
#     max_epochs=1 if cfg.smoke_test else cfg.train.epochs,
#     progress_bar_refresh_rate=1,
#     limit_train_batches=cfg.train.limit_train_batches,
#     checkpoint_callback=False,  # Disable checkpointing to save disk space
#     callbacks=[lr_monitor]
# )

# trainer.fit(model)

# trainer.max_epochs = 10
# trainer.fit(model)
# trainer.save_checkpoint('test.ckpt')

# lr_monitor = pl.callbacks.LearningRateLogger(logging_interval='epoch')
# model = pl_module_cls(cfg.model, cfg.dataset, cfg.train)
# trainer = pl.Trainer(
#     # gpus=1 if config['gpu'] else None,
#     gpus=1,
#     gradient_clip_val=cfg.train.gradient_clip_val,
#     max_epochs=1 if cfg.smoke_test else cfg.train.epochs,
#     progress_bar_refresh_rate=1,
#     limit_train_batches=cfg.train.limit_train_batches,
#     checkpoint_callback=False,  # Disable checkpointing to save disk space
#     resume_from_checkpoint='test.ckpt',
#     callbacks=[lr_monitor]
# )
# trainer.fit(model)
# print(lr_monitor.lrs)
