# from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, order=None, activation=None, nan=None):
        super(BasicBlock, self).__init__()

        self.order = order
        if activation == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif activation in ['relu6', 'gelu6']:
            self.act1 = MyActivation(planes, nan, activation)
            self.act2 = MyActivation(planes, nan, activation)

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        if self.order == 'relu-bn':
            out = self.bn1(self.act1(out))
        elif self.order == 'bn-relu':
            out = self.act1(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, order='relu-bn', activation='relu', nan=False):
        super(ResNet, self).__init__()

        self.in_planes = 16
        self.order = order
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation in ['relu6', 'gelu6']:
            self.act = MyActivation(self.in_planes, nan, activation)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1, order=order,
                                       activation=activation, nan=nan)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, order=order,
                                       activation=activation, nan=nan)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, order=order,
                                       activation=activation, nan=nan)
        self.linear = nn.Linear(64*block.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, order, activation, nan):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, order=order,
                                       activation=activation, nan=nan))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.order == 'relu-bn':
            out = self.bn1(self.act(out))
        elif self.order == 'bn-relu':
            out = self.act(self.bn1(out))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class MyActivation(nn.Module):
    def __init__(self, dim, nan=True, act='relu6'):
        super().__init__()

        if act == 'relu6':
            self.act = nn.ReLU()
        elif act == 'gelu6':
            self.act = nn.GELU()
        self.nan = nan

    def forward(self, x):
        #x[x > 10] *= 0
        x = torch.clip(self.act(x), None, 6)
        if self.nan:
            return torch.nan_to_num(x)
        return torch.clip(self.act(x), None, 6)


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def ResNet44(num_classes=10, order='relu-bn', activation='relu', nan=False):
    return ResNet(BasicBlock, [7, 7, 7], num_classes, order, activation, nan)
