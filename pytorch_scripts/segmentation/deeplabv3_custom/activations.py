import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustActivation(nn.Module):
    def __init__(self, f, activation='max'):
        super().__init__()
        assert activation in ['max', 'mean_std', 'median_iqr']

        self.activation = f
        self.logic = activation

        if self.logic == 'max':
            self.clip_value = None
            self._forward_function = self.forward_max

        elif self.logic == 'mean_std':
            self.s = None
            self.q = None
            self.N = None
            self._forward_function = self.forward_mean_std
    
    def forward(self, x):
        return self._forward_function(x)

    def forward_max(self, x: torch.Tensor):

        x = self.activation(x)

        if self.training:
            with torch.no_grad():
                current_max = torch.max(x).item()
        
            if self.clip_value is not None:
                current_max = max(self.clip_value, current_max)
            
            self.clip_value = current_max
        
        else:
            if self.clip_value is not None:
                x = torch.clip(x, max=self.clip_value)
        
        return x
    
    def forward_mean_std(self, x):
    
        x = self.activation(x)
    
        if self.training:
            if self.s is None:
                self.s = torch.sum(x.detach())
                self.q = torch.sum(x.detach().pow(2))
                self.N = x.size(0)
            else:
                self.s += torch.sum(x.detach())
                self.q += torch.sum(x.detach().pow(2))
                self.N += x.size(0)
        
        elif self.s is not None and self.q is not None and self.N is not None:
            clip_value = (self.s + torch.sqrt(self.N * self.q - self.s.pow(2))) / self.N
            x = torch.clip(x, max=clip_value)
        
        return x

