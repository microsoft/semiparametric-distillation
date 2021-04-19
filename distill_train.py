from pathlib import Path
project_root = Path(__file__).parent.absolute()

import torch

import hydra
from omegaconf import OmegaConf

import kd
from pl_runner import pl_train
from train import LightningModel
from tee import StdoutTee, StderrTee


class DistillLightningModel(LightningModel):

    def __init__(self, model_cfg, dataset_cfg, train_cfg):
        super().__init__(model_cfg, dataset_cfg, train_cfg)
        # This works with both relative and absolute teacher_checkpoint_path. Surprising!
        path = project_root / train_cfg.teacher_checkpoint_path
        # self._teacher = LightningModel.load_from_checkpoint(str(path))
        # We don't want _teacher to be a submodule so that during saving/restoring we don't need
        # to care about its parameters.
        object.__setattr__(self, '_teacher', LightningModel.load_from_checkpoint(str(path)))
        self._teacher.freeze()
        self.kd_loss = hydra.utils.instantiate(self.train_cfg.kd)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        with torch.no_grad():
            teacher_out = self._teacher.model(batch_x)
        out = self.model(batch_x)
        loss_og = self.task.loss(out, batch_y)
        loss = self.kd_loss(out, teacher_out, batch_y, loss_og)
        metrics = self.task.metrics(out, batch_y)
        metrics = {f'train_{k}': v for k, v in metrics.items()}
        self.log(f'train_loss', loss, on_epoch=True, prog_bar=False)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        return loss

    # This has to be on_fit_start and not "setup" because "setup" doesn't have the right device
    def on_fit_start(self):
        # Since _teacher isn't a submodule, it's not automatically moved to the GPU device
        self._teacher.to(self.device, self.dtype)

    def on_test_epoch_start(self):
        self._teacher.to(self.device, self.dtype)


class DistillCrossfitLightningModel(LightningModel):

    def __init__(self, model_cfg, dataset_cfg, train_cfg):
        dataset_cfg.return_indices = True
        super().__init__(model_cfg, dataset_cfg, train_cfg)
        path = project_root / train_cfg.teacher_checkpoint_path
        self.crossfit_size = self.train_cfg.crossfit_size
        self._teachers = []
        for i in range(self.crossfit_size):
            t = LightningModel.load_from_checkpoint(str(path).replace('.ckpt', f'{i}.ckpt'))
            t.freeze()
            self._teachers.append(t)
        # self._teachers = torch.nn.ModuleList(self._teachers)  # Need to register so they move to GPU device
        # We don't want _teachers to be a submodule so that during saving/restoring we don't need
        # to care about its parameters.
        object.__setattr__(self, '_teachers', torch.nn.ModuleList(self._teachers))
        self.kd_loss = hydra.utils.instantiate(self.train_cfg.kd)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, indices = batch
        mod = indices % self.crossfit_size
        x_list, y_list, teacher_out = [], [], []
        for i in range(self.crossfit_size):
            mask = mod == i
            if not torch.all(mask == False):
                x_list.append(batch_x[mask])
                y_list.append(batch_y[mask])
                with torch.no_grad():
                    teacher_out.append(self._teachers[i].model(x_list[-1]))
        batch_x, batch_y, teacher_out = (torch.cat(x_list), torch.cat(y_list),
                                         torch.cat(teacher_out))
        out = self.model(batch_x)
        loss_og = self.task.loss(out, batch_y)
        loss = self.kd_loss(out, teacher_out, batch_y, loss_og)
        metrics = self.task.metrics(out, batch_y)
        metrics = {f'train_{k}': v for k, v in metrics.items()}
        self.log(f'train_loss', loss, on_epoch=True, prog_bar=False)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        return loss

    # This has to be on_fit_start and not "setup" because "setup" doesn't have the right device
    def on_fit_start(self):
        # Since _teacher isn't a submodule, it's not automatically moved to the GPU device
        self._teachers.to(self.device, self.dtype)

    def on_test_epoch_start(self):
        self._teachers.to(self.device, self.dtype)


@hydra.main(config_path="cfg", config_name="distill.yaml")
def main(cfg: OmegaConf):
    with StdoutTee('train.stdout'), StderrTee('train.stderr'):
        print(OmegaConf.to_yaml(cfg))
        pl_module_cls = DistillLightningModel if cfg.train.get('crossfit_size', 1) == 1 else DistillCrossfitLightningModel
        if cfg.runner.name == 'pl':
            pl_train(cfg, pl_module_cls)
        else:
            assert cfg.runner.name == 'ray', 'Only pl and ray runners are supported'
            # Shouldn't need to install ray unless doing distributed training
            from ray_runner import ray_train
            ray_train(cfg, pl_module_cls)


if __name__ == "__main__":
    main()
