import os
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from pytorch_scripts.segmentation.cityscapes import Cityscapes

import pytorch_scripts.segmentation.transforms as ST

from pytorch_scripts.utils import get_loader


def get_data_dir(data_dir, dataset):
    if dataset == 'cifar10':
        data_dir = os.path.join(data_dir, 'CIFAR10')
    elif dataset == 'cifar100':
        data_dir = os.path.join(data_dir, 'CIFAR100')
    elif dataset == 'ImageNet':
        data_dir = os.path.join(data_dir, 'ImageNet')
    elif dataset == 'cityscapes':
        data_dir = os.path.join(data_dir, 'Cityscapes')
    return data_dir


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset='cifar10', data_dir='data', batch_size=128, num_gpus=1, num_workers=None,
                 size=256, fp16=True, augs={}):
        super().__init__()
        print(f'==> Loading {dataset} dataset..')
        self.save_hyperparameters()
        self.dataset = dataset
        self.fp16 = fp16
        self.data_dir = get_data_dir(data_dir, dataset)
        self.size = size
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.n_classes = None
        self.train_trans = None
        self.test_trans = None
        self.train_data = None
        self.test_data = None
        self.mixup_cutmix = augs['mixup_cutmix']
        self.jitter = augs['jitter']
        self.rand_aug = augs['rand_aug']
        self.rand_erasing = augs['rand_erasing']
        self.label_smooth = augs['label_smooth']
        self.rcc = augs['rcc']
        self.num_workers = num_workers or 8

        # Due to deprecation and future removal
        self.prepare_data_per_node = False

    def prepare_data(self):

        if self.dataset == 'cifar10':
            CIFAR10(root=self.data_dir, train=True, download=True)
            CIFAR10(root=self.data_dir, train=False, download=True)
        elif self.dataset == 'cifar100':
            CIFAR100(root=self.data_dir, train=True, download=True)
            CIFAR100(root=self.data_dir, train=False, download=True)
        elif self.dataset == 'ImageNet':
            ImageNet(root=self.data_dir, split='train')
            ImageNet(root=self.data_dir, split='val')
        elif self.dataset == 'cityscapes':
            Cityscapes(root=self.data_dir, split='train', mode='fine')
            Cityscapes(root=self.data_dir, split='test', mode='fine')

    def setup(self, stage=None):
        if self.dataset == 'cifar10':
            self.stats = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        elif self.dataset == 'cifar100':
            self.stats = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)
        elif self.dataset == 'ImageNet':
            self.stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif self.dataset == 'cityscapes':
            self.stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        if self.dataset == 'cityscapes':
            self.test_trans = ST.ExtCompose([
                ST.ExtToTensor(),
                ST.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
        else:
            normalize = transforms.Normalize(self.stats[0], self.stats[1])
            self.test_trans = transforms.Compose([transforms.Resize((self.size, self.size)), transforms.ToTensor(),
                                                normalize])

        if self.dataset == 'cifar10':
            self.train_data = CIFAR10(root=self.data_dir, train=True, transform=None, download=False)
            self.test_data = CIFAR10(root=self.data_dir, train=False, transform=self.test_trans, download=False)
            self.n_classes = 10
        elif self.dataset == 'cifar100':
            self.train_data = CIFAR100(root=self.data_dir, train=True, transform=None, download=False)
            self.test_data = CIFAR100(root=self.data_dir, train=False, transform=self.test_trans, download=False)
            self.n_classes = 100
        elif self.dataset == 'ImageNet':
            self.train_data = ImageNet(root=self.data_dir, split='train')
            self.test_data = ImageNet(root=self.data_dir, split='val', transform=self.test_trans)
            self.n_classes = 1000
        elif self.dataset == 'cityscapes':
            self.train_data = Cityscapes(root=self.data_dir, split='train', mode='fine', transform=None)
            self.test_data = Cityscapes(root=self.data_dir, split='val', mode='fine', transform=self.test_trans)
            self.n_classes = 19

    def train_dataloader(self):
        #return get_loader(self.dataset, self.train_data, self.batch_size // self.num_gpus, self.num_workers, self.n_classes,
        #                  self.stats, self.mixup_cutmix, rand_erasing=self.rand_erasing, jitter=self.jitter,
        #                  rand_aug=self.rand_aug, label_smooth=self.label_smooth, rcc=self.rcc, size=self.size,
        #                  fp16=self.fp16)
        return get_loader(self.dataset, self.train_data, self.batch_size // self.num_gpus, self.num_workers, self.n_classes,
                          self.stats, self.mixup_cutmix, rand_erasing=self.rand_erasing, jitter=self.jitter,
                          rand_aug=self.rand_aug, label_smooth=self.label_smooth, rcc=self.rcc, size=self.size,
                          fp16=self.fp16)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
