import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustActivation(nn.Module):
    def __init__(self, f, p=1.0):
        self.activation = f
        self.p = p

        self.s = None
        self.q = None
        self.N = None
    
    def forward(self, x):

        x = self.activation(x)

        if self.training:
            if self.s is None:
                self.s = torch.sum(x.detach(), dim=0)
                self.q = torch.sum(x.detach().pow(2), dim=0)
                self.N = x.size(0)
            else:
                self.s += torch.sum(x.detach(), dim=0)
                self.q += torch.sum(x.detach().pow(2), dim=0)
                self.N += x.size(0)
        
        else:
            clip_value = (self.s + self.p * torch.sqrt(self.N * self.q - self.s.pow(2))) / self.N
            x = torch.clip(x, max=clip_value)
        
        return x
