# Knowledge Distillation as Semiparametric Inference

Code replicating the experiments of

[Knowledge Distillation as Semiparametric Inference](https://openreview.net/pdf?id=m4UCf24r0Y).  
Tri Dao, Govinda M. Kamath, Vasilis Syrgkanis, and Lester Mackey.  
International Conference on Learning Representations (ICLR). May 2021.

## Required packages
Python >= 3.7, Pytorch >= 1.7.
More details in `requirements.txt`. 

We use
[Pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) to
organize training code, [Hydra](https://hydra.cc/) to organize configurations.

[Optional] We use [Ray](https://github.com/ray-project/ray) for distributed
training.

[Optional] We use [Wandb](https://wandb.ai/) for logging.

## Code structure
```
├─ cfg               # Configuration files, for Hydra
├─ datamodules        # Code for datasets
├─ models            # Model implementations
├─ results           # json files containing raw results, and pdf files containing plots
├─ scripts           # Scripts to tune hyperparameters
├─ distill_train.py  # Train student model, distilled from teacher model
├─ kd.py            # Implementation of knowledge distillation losses
├─ ray_runner.py      # Distributed training with Ray [optional]
├─ train.py          # Train a model (e.g. teacher or student) from scratch
├─ utils.py          # Utility functions
```

## Training
To train a ResNet18 on CIFAR10, and save model to disk:
```
python train.py train.batch_size=512 train.optimizer.lr=4e-1 model=resnet18 +save_checkpoint_path=checkpoints/resnet18/final.ckpt runner=pl
```

To train a CNN5 student:
```
python distill_train.py train.batch_size=512 train.optimizer.lr=4e-1 train.kd.class=KDOrthoLoss train.gradient_clip_val=0.1 train.kd.temperature=2.0 runner=pl
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
