import time
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR, StepLR
from pytorch_scripts.segmentation.losses import FocalLoss
from pytorch_scripts.segmentation.stream_metrics import StreamSegMetrics
import numpy as np

class ClassificationModelWrapper(pl.LightningModule):
    def __init__(self, model, n_classes, optim, loss, freeze=False):
        super(ClassificationModelWrapper, self).__init__()

        self.model = model
        self.n_classes = n_classes
        self.optim = optim
        self.freeze = freeze

        if loss == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
            self.use_one_hot = True
        elif loss == 'ce':
            self.criterion = nn.CrossEntropyLoss()
            self.use_one_hot = False
        elif loss == 'sce':
            self.criterion = SymmetricCELoss()
            self.use_one_hot = True

        self.save_hyperparameters('model', 'n_classes', 'optim', 'loss')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.optim['optimizer'] == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.optim['lr'], weight_decay=self.optim['wd'],
                            momentum=0.9)

        elif self.optim['optimizer'] == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.optim['lr'], weight_decay=self.optim['wd'])

        if self.optim['scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, self.optim['epochs'], eta_min=self.optim['lr_min'])

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_metrics(self, batch, inject=True):
        x, y = batch

        self.model.to_be_injected = inject

        # forward
        outputs = self(x)

        # loss
        if self.use_one_hot and not self.training:
            # bce or sce
            loss = self.criterion(outputs, self.get_one_hot(y, self.n_classes))
        else:
            # ce
            loss = self.criterion(outputs, y)
        # accuracy
        if not self.training:
            probs, preds = torch.max(outputs, 1)
            acc = torch.sum(preds == y) / x.shape[0]
        else:
            acc, probs, preds = 0, 0, 0
        return loss, acc, (probs, preds)

    def training_step(self, train_batch, batch_idx):
        loss, acc, _ = self.get_metrics(train_batch)

        self.epoch_log('train_loss', loss)
        self.epoch_log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx, check_criticality=True):
        loss, acc, clean_vals = self.get_metrics(val_batch, False)
        noisy_loss, noisy_acc, noisy_vals = self.get_metrics(val_batch)

        # Test the accuracy
        self.epoch_log('val_loss', loss)
        self.epoch_log('val_acc', acc)
        self.epoch_log('noisy_val_loss', noisy_loss)
        self.epoch_log('noisy_val_acc', noisy_acc)
        if check_criticality:
            self.check_criticality(gold=clean_vals, faulty=noisy_vals)
        return noisy_loss

    def on_train_epoch_start(self):
        self.model.current_epoch = self.current_epoch
        lr = self.optimizers().param_groups[0]['lr']
        self.epoch_log('lr', lr)
        ''' if self.current_epoch == 0 and self.freeze:
            self.freeze_layers()
        elif self.current_epoch == 1 and self.freeze:
            self.unfreeze_layers()'''

    def epoch_log(self, name, value, prog_bar=True):
        self.log(name, value, on_step=False, on_epoch=True, prog_bar=prog_bar)

    def freeze_layers(self):
        print('|||| Freezing all layers but BatchNorm..')
        for layer in self.model.modules():
            if not isinstance(layer, nn.BatchNorm2d):
                if hasattr(layer, 'weight') and layer.weight is not None:
                    layer.weight.requires_grad = False
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.requires_grad = False

    def unfreeze_layers(self):
        print('|||| Unfreezing all layers..')
        for layer in self.model.modules():
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer.weight.requires_grad = True
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.requires_grad = True

    def check_criticality(self, gold: tuple, faulty: tuple):
        gold_vals, gold_preds = gold
        fault_vals, fault_preds = faulty
        # Magic number to define what is a zero
        err_lambda = 1e-4
        # Check if the sum of diffs are
        value_diff_pct = torch.sum(torch.abs(gold_vals - fault_vals) > err_lambda) / gold_vals.shape[0]
        preds_diff_pct = torch.sum(gold_preds != fault_preds) / gold_vals.shape[0]
        self.epoch_log('value_diff_pct', value_diff_pct)
        self.epoch_log('preds_diff_pct', preds_diff_pct)

    @staticmethod
    def get_one_hot(target, n_classes=10, device='cuda'):
        one_hot = torch.zeros(target.shape[0], n_classes, device=device)
        one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
        return one_hot


class SymmetricCELoss(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super(SymmetricCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nnlloss = nn.NLLLoss()
        self.Softmax = nn.Softmax()

    def negative_log_likelihood(self, inputs, targets):
        return - torch.sum(targets * torch.log(inputs + 1e-6)) / inputs.shape[0]

    def forward(self, inputs, targets):
        inputs = self.Softmax(inputs)
        # standard crossEntropy
        ce = self.negative_log_likelihood(inputs, targets)
        # reverse crossEntropy
        rce = self.negative_log_likelihood(targets, inputs)
        return ce * self.alpha + rce * self.beta


class UpdateNormStatsCallback(pl.Callback):

    def on_train_start(self, trainer, pl_module):
        print('||| Updating BatchNorm statistics..')

        trainer.model.train()
        for idx, (x, y) in enumerate(trainer.datamodule.train_dataloader()):
            # Need to use .float() here or BatchNorm throws error
            # --> use half batch size to compensate memory increase

            l = x.shape[0] // 2
            trainer.model((x[:l].float(), y[:l].float()), batch_idx=idx)
            trainer.model.zero_grad()
            trainer.model((x[l:].float(), y[l:].float()), batch_idx=idx)
            trainer.model.zero_grad()

        print('||| BatchNorm statistics updated..')


######################################################################################################

class SegmentationModelWrapper(pl.LightningModule):
    def __init__(self, model, n_classes, optim, loss, freeze=False):
        super(SegmentationModelWrapper, self).__init__()

        self.model = model
        self.n_classes = n_classes
        self.optim = optim
        self.freeze = freeze

        self.real_p = self.model.p

        self.clean_metrics = StreamSegMetrics(self.n_classes)
        self.noisy_metrics = StreamSegMetrics(self.n_classes)

        if loss == 'ce':
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        elif loss == 'focal':
            self.criterion = FocalLoss(ignore_index=255, size_average=True)

        self.save_hyperparameters('model', 'n_classes', 'optim', 'loss')

    def forward(self, x):
        return self.model(x)#['out']

    def configure_optimizers(self):
        if self.optim['optimizer'] == 'sgd':
            optimizer = SGD(params=[
                {'params': self.model.model.backbone.parameters(), 'lr': 0.1 * self.optim['lr']},
                {'params': self.model.model.classifier.parameters(), 'lr': self.optim['lr']},
            ], lr=self.optim['lr'], momentum=0.9, weight_decay=self.optim['wd'])

        if self.optim['scheduler'] == 'poly':
            scheduler = PolynomialLR(optimizer, self.optim['epochs'], power=0.9)
        elif self.optim['scheduler'] == 'step':
            scheduler = StepLR(optimizer, step_size=self.optim['epochs'] // 3, gamma=0.1)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        self.model.to_be_injected = True

        x, y = batch

        outputs = self(x)
        loss = self.criterion(outputs, y).mean()
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        if batch_idx == 0:
            self.clean_metrics.reset()
            self.clean_loss = [0, 0]

        x, y = val_batch

        # clean
        outputs = self(x)
        self.clean_loss[1] += x.size(0)
        self.clean_loss[0] += self.criterion(outputs, y).mean().item()
        _, preds = torch.max(outputs, 1)
        
        self.clean_metrics.update(y.cpu().numpy(), preds.cpu().numpy())

    def on_validation_epoch_end(self):
        clean_miou = self.clean_metrics.get_results()['Mean IoU']
        clean_loss = self.clean_loss[0] / self.clean_loss[1]
        self.log('val_loss', clean_loss)
        self.log('val_miou', clean_miou)

    def test_step(self, val_batch, batch_idx):
        if batch_idx == 0:
            self.clean_metrics.reset()
            self.noisy_metrics.reset()
            self.clean_loss = [0, 0]
            self.noisy_loss = [0, 0]

            self.model.p = 1.0

        x, y = val_batch

        # clean
        self.model.to_be_injected = False

        outputs = self(x)
        self.clean_loss[1] += x.size(0)
        self.clean_loss[0] += self.criterion(outputs, y).mean().item()
        _, preds = torch.max(outputs, 1)
        
        self.clean_metrics.update(y.cpu().numpy(), preds.cpu().numpy())

        # noisy
        self.model.to_be_injected = True

        outputs = self(x)
        self.noisy_loss[1] += x.size(0)
        self.noisy_loss[0] += self.criterion(outputs, y).mean().item()
        _, preds = torch.max(outputs, 1)
        
        self.noisy_metrics.update(y.cpu().numpy(), preds.cpu().numpy())

    def on_test_epoch_end(self):
        self.model.p = self.real_p

        clean_miou = self.clean_metrics.get_results()['Mean IoU']
        noisy_miou = self.noisy_metrics.get_results()['Mean IoU']

        clean_loss = self.clean_loss[0] / self.clean_loss[1]
        noisy_loss = self.noisy_loss[0] / self.noisy_loss[1]

        #self.check_criticality(gold=clean_vals, faulty=noisy_vals)

        self.log('test_loss', clean_loss)
        self.log('test_miou', clean_miou)
        self.log('noisy_test_loss', noisy_loss)
        self.log('noisy_test_miou', noisy_miou)
        
    def on_train_epoch_start(self):
        self.model.current_epoch = self.current_epoch
        lr = self.optimizers().param_groups[0]['lr']
        self.epoch_log('lr', lr)

    def epoch_log(self, name, value, prog_bar=True):
        self.log(name, value, on_step=False, on_epoch=True, prog_bar=prog_bar)

    def check_criticality(self, gold: tuple, faulty: tuple):
        gold_vals, gold_preds = gold
        fault_vals, fault_preds = faulty
        # Magic number to define what is a zero
        err_lambda = 1e-4
        # Check if the sum of diffs are
        value_diff_pct = torch.sum(torch.abs(gold_vals - fault_vals) > err_lambda) / gold_vals.shape[0]
        preds_diff_pct = torch.sum(gold_preds != fault_preds) / gold_vals.shape[0]
        self.epoch_log('value_diff_pct', value_diff_pct)
        self.epoch_log('preds_diff_pct', preds_diff_pct)
