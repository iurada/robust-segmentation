import random
import numpy as np

import torch
import torch.nn as nn


class Injector(nn.Module):
    def __init__(self, model: nn.Module, error_model: str = 'random', p: float = 0.3, starting_epoch: int = 10,
                 clip=False, nan=False):
        super(Injector, self).__init__()

        self.model = model
        self.error_model = error_model
        self.p = p
        self.to_be_injected = True
        self.mask_types = ['line', 'block', 'random']

        # Set up injection
        self.counter = 0
        self.total_ops = 0
        self.injected = None
        self.apply_injection_hook(self.model, modules=['conv'])

        # Set up clipping and NaN managing
        self.clip = clip
        self.nan = nan
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if clip:
                    m.register_forward_hook(self.clip_hook)
                if nan:
                    m.register_forward_hook(self.nan_hook)

        self.current_epoch = 0
        self.starting_epoch = starting_epoch

    def forward(self, x):
        self.injected = np.random.randint(0, self.total_ops)
        self.counter = 0
        return self.model(x)

    def apply_injection_hook(self, model, modules=['conv', 'linear', 'norm', 'act']):
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and 'conv' in modules:
                m.register_forward_hook(self.hook_injector)
                self.total_ops += 1
            elif isinstance(m, nn.Linear) and 'linear' in modules:
                m.register_forward_hook(self.hook_injector)
                self.total_ops += 1
            elif isinstance(m, nn.BatchNorm2d) and 'norm' in modules:
                m.register_forward_hook(self.hook_injector)
                self.total_ops += 1
            elif isinstance(m, nn.ReLU) or isinstance(m, nn.GELU) or isinstance(m, nn.ReLU6) and 'act' in modules:
                m.register_forward_hook(self.hook_injector)
                self.total_ops += 1

    def clip_hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        return torch.clip(output, -6, 6)

    def nan_hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        return torch.nan_to_num(output, 0.0)

    def hook_injector(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):

        if self.current_epoch >= self.starting_epoch and self.training and self.counter == self.injected:
            # inject noise to each sample with probability p
            output = self.inject(output, self.p)

        elif not self.training and self.counter == self.injected:
            if self.to_be_injected:
                # inject noise to all samples
                output = self.inject(output, 1)
        self.counter += 1

        return output

    def inject(self, x: torch.Tensor, p: float) -> torch.Tensor:
        # We can inject the relative errors using only Torch built-in functions
        # Otherwise it is necessary to use AutoGrads

        if self.training:
            error = self.training_error(self.current_epoch, x)
            error_model = self.error_model
        else:
            error = self.random_relative_error
            error_model = np.random.choice(self.mask_types)

        mask = generate_mask(x, error_model, p).to(x.device)
        x = (x * mask).mul(error) + (x * ~mask)

        return x

    @property
    def random_relative_error(self) -> float:
        r"""Generator for relative errors to be injected on the training
        We have seen in the past relative error distributions that follow a Power Law PDF
        so we will use the approach proposed at https://arxiv.org/abs/1208.3524
        We will implement the function based on https://stats.stackexchange.com/a/406705
        Example:
        x_min, alpha, r = 5, 2.5, random()
        relative_error = x_min * (1 - r) ** (-1 / (alpha - 1))
        :return: the calculated relative_error
        """
        # TODO: Generalize the random generation to the values observed on GEMM output
        # Power Law parameters for the Functional Units
        power_law_fus = [
            (1.0728769e-07, 1.0868737), (2.0230031, 1.0568325), (8.1847715e-08, 1.082071), (136027.72, 27.1194),
            (3.0, 1.0678725), (0.03517608, 1.189603), (3.4028237e+38, 443107.0), (2.0, 1.4543958),
            (0.010238367, 1.1181921), (1.396856e-09, 1.0846596), (2.6865074e-10, 1.0769672), (1.3970158e-09, 1.085144),
            (0.66699225, 23.798765), (0.66699225, 23.798765), (0.66699225, 23.922783), (0.75000001, 121435080.0),
            (0.61141304, 3.4316596), (0.75000001, 121435080.0), (0.0, 1.08212), (7.0958774e-08, 1.082116),
            (0.0, 1.08212)
        ]

        alpha, x_min = random.choice(power_law_fus)
        r = random.random()
        relative_error = x_min * (1 - r) ** (-1 / (alpha - 1))
        return relative_error

    def training_error(self, epoch, x):
        error = torch.rand(size=(1,), device=x.device, dtype=x.dtype) * 6 + 1e-6  #* max(1, epoch, 1)
        if random.randint(0, 1):
            return error
        return - error


def generate_mask(x, error_type='block', p=1.0):
    shape = x.shape

    if len(shape) == 4:
        b, c, h, w = shape
        mask = generate_4D_mask(b, c, h, w, error_type, p)
    else:
        b, c = shape
        mask = generate_2D_mask(b, c, p)

    return torch.tensor(mask > 0)

def generate_4D_mask(b, c, h, w, error_type, p):
    mask4D = torch.zeros((b, c, h, w))
    mask3D = torch.zeros((c, h, w))

    if error_type == 'line':
        mask2D = generate_line_mask(h, w)
        channel_p = 0.75
    elif error_type == 'block':
        mask2D = generate_block_mask(h, w)
        channel_p = 0.3
    elif error_type == 'random':
        mask2D = generate_random_mask(h, w)
        channel_p = 0.1

    samples = torch.bernoulli(torch.ones(b) * p) > 0 if p != 1 else torch.ones(b) > 0
    channels = torch.bernoulli(torch.ones(c) * channel_p) > 0

    mask3D[channels] += mask2D
    mask4D[samples] += mask3D

    return mask4D

def generate_2D_mask(b, c, p):
    mask2D = torch.zeros((b, c))
    mask2D[torch.bernoulli(torch.ones(b) * p) > 0, torch.bernoulli(torch.ones(c) * 0.3) > 0] = 1
    return mask2D

def generate_line_mask(h, w):
    mask = torch.zeros((h, w))
    if torch.bernoulli(torch.ones(1) * .5):
        # columns
        column = torch.randint(0, h, (1,)) #! BUGFIX w --> h (first axis is 'rows')
        mask[column, :] = 1
    else:
        # rows
        row = torch.randint(0, w, (1,)) #! BUGFIX h --> w (second axis is 'columns')
        mask[:, row] = 1
    return mask

def generate_block_mask(h, w):
    mask = torch.zeros((h, w))
    h0, h1, w0, w1 = torch.randint(0, h, (1,)), torch.randint(0, h, (1,)), \
                     torch.randint(0, w, (1,)), torch.randint(0, w, (1,))
    h0, h1 = min(h0, h1), max(h0, h1)
    w0, w1 = min(w0, w1), max(w0, w1)

    mask[h0:h1, w0:w1] = 1
    return mask

def generate_random_mask(h, w):
    mask = torch.rand((h, w)) > 0.5
    return mask


