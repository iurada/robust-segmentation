#!/usr/bin/python3

##########################
def allow_import():
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)

allow_import()
##########################

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

# DieHardNET packages
from pytorch_scripts.utils import *
from pytorch_scripts.LightningModelWrapper import UpdateNormStatsCallback
from pytorch_scripts.data_module import DataModule

# Suppress the annoying warning for non-empty checkpoint directory
warnings.filterwarnings("ignore")

config_parser = parser = argparse.ArgumentParser(description='Configuration', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments.')

parser = argparse.ArgumentParser(description='PyTorch Training')


# General
parser.add_argument('--name', type=str, default='test', help='Experiment name.')
parser.add_argument('--mode', type=str, default='train', help='Mode: train/training or validation/validate.')
parser.add_argument('--ckpt', type=str, default=None, help='Pass the name of a checkpoint to resume training.')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name: cifar10 or cifar100.')
parser.add_argument('--size', type=int, default=224, help='Image size.')
parser.add_argument('--precision', type=int, default=16, help='Whether to use Mixed Precision or not.')
parser.add_argument('--data_dir', type=str, default=None, help='Path to dataset.')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs.')

# Model
parser.add_argument('--model', type=str, default='resnet20', help='Network name. Resnets only for now.')
parser.add_argument('--model_clip', type=bool, default=False, help='Whether to clip layer outputs or not.')
parser.add_argument('--nan', type=bool, default=False, help='Whether to convert NaNs to 0 or not.')
parser.add_argument('--freeze', type=bool, default=False, help='Whether to freeze all layer but BN in the first epoch or not.')
parser.add_argument('--pretrained', type=bool, default=False, help='Whether to start from pretrained weights or not.')

# Optimization
parser.add_argument('--loss', type=str, default='bce', help='Loss: bce, ce or sce.')
parser.add_argument('--grad_clip', default=None, help='Gradient clipping value.')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')
parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate.')
parser.add_argument('--scheduler', type=str, default='cosine', help='Scheduler name: cosine, poly')
parser.add_argument('--lr_min', type=float, default=1e-1, help='Minimum learning rate.')
parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer name: adamw or sgd.')

# Injection
parser.add_argument('--error_model', type=str, default='random', help='Optimizer name: adamw or sgd.')
parser.add_argument('--inject_p', type=float, default=0.1, help='Probability of noise injection at training time.')
parser.add_argument('--inject_epoch', type=float, default=0, help='How many epochs before starting the injection.')

# Augmentations and Regularisations
parser.add_argument('--wd', type=float, default=1e-4, help='Weight Decay.')
parser.add_argument('--rcc', type=float, default=0.75, help='RCC lower bound.')
parser.add_argument('--rand_aug', type=str, default=None, help='RandAugment magnitude and std.')
parser.add_argument('--rand_erasing', type=float, default=0.0, help='Random Erasing propability.')
parser.add_argument('--mixup_cutmix', type=bool, default=False, help='Whether to use mixup/cutmix or not.')
parser.add_argument('--jitter', type=float, default=0.0, help='Color jitter.')
parser.add_argument('--label_smooth', type=float, default=0.0, help='Label Smoothing.')


# Others
parser.add_argument('--seed', default=0, help='Random seed for reproducibility.')
parser.add_argument('--comment', default='', help='Optional comment.')

n_classes = {'cifar10': 10, 'cifar100': 100, 'ImageNet': 1000, 'cityscapes': 19}

def main():
    args = parse_args(parser, config_parser)

    # Set random seed
    pl.seed_everything(args.seed, workers=True)

    augs = {'rand_aug': args.rand_aug, 'rand_erasing': args.rand_erasing, 'mixup_cutmix': args.mixup_cutmix,
            'jitter': args.jitter, 'label_smooth': args.label_smooth, 'rcc': args.rcc}

    root = args.data_dir or get_default_data_root()
    datamodule = DataModule(args.dataset, root, args.batch_size, args.num_gpus,
                       size=args.size, augs=augs, fp16=args.precision)

    # Build model (ResNet or EfficientNet only up to now)
    optim_params = {'optimizer': args.optimizer, 'epochs': args.epochs, 'lr': args.lr, 'lr_min': args.lr_min,
                    'wd': args.wd, 'scheduler': args.scheduler}
    net = build_model(args.model, n_classes[args.dataset], optim_params, args.loss, args.error_model, args.inject_p,
                      args.inject_epoch, args.model_clip, args.nan, args.freeze, args.pretrained)

    # W&B logger
    wandb_logger = WandbLogger(project="NeutronRobustness", name=args.name, entity="pathselector")
    wandb_logger.log_hyperparams(args)
    wandb_logger.watch(net, log_graph=False)

    # Callbacks
    filename_quantity = '-{epoch:02d}-{val_acc:.2f}'
    #monitored_quantity, monitored_mode = 'val_acc', 'max'

    if args.dataset == 'cityscapes':
        filename_quantity = '-{epoch:02d}-{val_miou:.4f}'
        #monitored_quantity, monitored_mode = 'noisy_val_miou', 'max'

    ckpt_callback = ModelCheckpoint('checkpoints/', 
                                    filename=args.name + filename_quantity,
                                    save_last=False)
    #stats_callback = UpdateNormStatsCallback()
    callbacks = [ckpt_callback]

    # Pytorch-Lightning Trainer
    trainer = pl.Trainer(max_epochs=args.epochs, devices=args.num_gpus, callbacks=callbacks, logger=wandb_logger, log_every_n_steps=1,
                         deterministic='warn', benchmark=True, accelerator='gpu', sync_batchnorm=True,
                         gradient_clip_val=args.grad_clip, strategy=DDPStrategy(find_unused_parameters=False),
                         precision=args.precision, auto_select_gpus=False)

    #if args.ckpt:
    #    #args.ckpt = '~/Dropbox/DieHardNet/Checkpoints/' + args.ckpt
    #    args.ckpt = 'checkpoints/' + args.ckpt
    if args.mode == 'train' or args.mode == 'training':
        trainer.fit(net, datamodule, ckpt_path=args.ckpt)
        trainer.test(net, datamodule, ckpt_path=args.ckpt)
    elif args.mode == 'validation' or args.mode == 'validate':
        trainer.validate(net, datamodule, ckpt_path=args.ckpt)
    elif args.mode == 'test' or args.mode == 'testing':
        trainer.test(net, datamodule, ckpt_path=args.ckpt)
    else:
        print('ERROR: select a suitable mode "train/training" or "validation/validate".')


if __name__ == '__main__':
    main()
