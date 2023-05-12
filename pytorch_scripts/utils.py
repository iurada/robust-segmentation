import platform
import re
import yaml
import os

import torch
import torchvision
from timm.data import create_loader, FastCollateMixup
import pytorch_scripts.segmentation.transforms as ST
from torch.utils.data import DataLoader

from .LightningModelWrapper import ClassificationModelWrapper, SegmentationModelWrapper

from pytorch_scripts.hg_noise_injector.hook_injection import Injector


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


def build_model(model=None, n_classes=10, optim_params={}, loss='bce', error_model='random', inject_p=0.1, inject_epoch=0,
                clip=False, nan=False, freeze=False, pretrained=False, activation='max'):

    if model == 'resnet50':
        model_name = model
        from torchvision.models import resnet50
        model = resnet50()
    elif model == 'efficientnet':
        model_name = model
        from torchvision.models import efficientnet_v2_s
        model = efficientnet_v2_s()

    elif model == 'deeplab':
        model_name = model

        from pytorch_scripts.segmentation.deeplabv3_custom.models import deeplabv3_resnet101
        model = deeplabv3_resnet101(n_classes, pretrained=pretrained)
        
        net = Injector(model, error_model, inject_p, inject_epoch, clip, nan)
        return SegmentationModelWrapper(net, n_classes, optim_params, loss, freeze)
    
    elif model == 'deeplab_relumax':
        model_name = model

        from pytorch_scripts.segmentation.deeplabv3_custom.deeplab_relumax import deeplabv3_resnet101
        model = deeplabv3_resnet101(n_classes, pretrained=pretrained, activation=activation)
        
        net = Injector(model, error_model, inject_p, inject_epoch, clip, nan)
        return SegmentationModelWrapper(net, n_classes, optim_params, loss, freeze)
    
    elif model == 'deeplab_mobilenet':
        model_name = model

        from pytorch_scripts.segmentation.mobilenet.models import deeplabv3_mobilenet_v3_large
        model = deeplabv3_mobilenet_v3_large(n_classes, pretrained=pretrained)
        
        net = Injector(model, error_model, inject_p, inject_epoch, clip, nan)
        return SegmentationModelWrapper(net, n_classes, optim_params, loss, freeze)

    # Load weights (cineca nodes are not online --> you can't directly download them)
    # if is_on_cluster():
    model.load_state_dict(torch.load(os.path.join('../pretrained_weights/', model_name)))

    net = Injector(model, error_model, inject_p, inject_epoch, clip, nan)
    print(f'\n==> {model} built.')
    return ClassificationModelWrapper(net, n_classes, optim_params, loss, freeze)


def get_loader(dataset_name, data, batch_size=128, workers=4, n_classes=100, stats=None, mixup_cutmix=True, rand_erasing=0.0,
               label_smooth=0.1, rand_aug='rand-m9-mstd0.5-inc1', jitter=0.0, rcc=0.75, size=32, fp16=True):
    # Segmentation
    if dataset_name == 'cityscapes':
        # Setup: https://arxiv.org/pdf/1706.05587.pdf (CVPR 2017)
        train_transform = ST.ExtCompose([
            ST.ExtRandomCrop(size=(size, size)),
            ST.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            ST.ExtRandomHorizontalFlip(),
            ST.ExtToTensor(),
            ST.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
        data.transform = train_transform
        return DataLoader(data, batch_size=batch_size, num_workers=workers, shuffle=True)

    # Classification
    if mixup_cutmix:
        mixup_alpha = 0.8
        cutmix_alpha = 1.0
        prob = 1.0
        switch_prob = 0.5
    else:
        mixup_alpha = 0.0
        cutmix_alpha = 0.0
        prob = 0.0
        switch_prob = 0.0
    collate = FastCollateMixup(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, cutmix_minmax=None, prob=prob,
                               switch_prob=switch_prob, mode='batch', label_smoothing=label_smooth,
                               num_classes=n_classes)
    return create_loader(data, input_size=(3, size, size),
                         batch_size=batch_size,
                         is_training=True,
                         use_prefetcher=True,
                         no_aug=False,
                         re_prob=rand_erasing,  # RandErasing
                         re_mode='pixel',
                         re_count=1,
                         re_split=False,
                         scale=[rcc, 1.0],  #[0.08, 1.0] if size != 32 else [0.75, 1.0],
                         ratio=[3./4., 4./3.],
                         hflip=0.5,
                         vflip=0,
                         color_jitter=jitter,
                         auto_augment=rand_aug,
                         num_aug_splits=0,
                         interpolation='random',
                         mean=stats[0],
                         std=stats[1],
                         num_workers=workers,
                         distributed=True,
                         collate_fn=collate,
                         pin_memory=True,
                         use_multi_epochs_loader=False,
                         fp16=fp16)


def parse_args(parser, config_parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args

def get_default_data_root():
    node = platform.node()

    sysname = node
    if re.match(r'compute-\d+-\d+', sysname):
        sysname = 'legion'
    elif re.match(r'node\d', sysname):
        sysname = 'clustervandal'
    elif re.match(r'r\d+n\d+', sysname):
        sysname = 'marconi100'
    elif re.match(r'^(?:[fg]node|franklin)\d{2}$', sysname):
        sysname = 'franklin'

    # TODO Percorsi dei dataset sulle nostre macchine.
    paths = {
        # Lab workstations
        'demetra': '/data/lucar/datasets',
        'poseidon': '/data/lucar/datasets',
        'nike': '/home/lucar/datasets',
        'athena': '/home/lucar/datasets',
        'terpsichore': '/data/lucar/datasets',
        'atlas': '/data/lucar/datasets',
        'kronos': '/data/datasets',
        'urania': '/data/datasets',

        # Clusters
        'legion': '/home/lrobbiano/datasets',
        'franklin': '/projects/vandal/nas/datasets',

        # Personal (for debugging)
        'carbonite': '/home/luca/datasets'
    }

    return paths.get(sysname, None)
