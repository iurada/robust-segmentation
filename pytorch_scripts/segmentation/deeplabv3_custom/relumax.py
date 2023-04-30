import torch
import torch.nn as nn

class ReLUMax(nn.Module):
    '''
    Instead of clipping directly at 6, we can "learn" the working range of each activation
    and instead clip using the "learned" values.

    Hypothesis: no injections at training time!!!

    #TODO: Check speed, performance and robustness (on train and test splits) vs number of ReLUMax present in the architecture
    '''
    def __init__(self):
        super(ReLUMax, self).__init__()
        self.clip_value = None
    
    def forward(self, x):

        if self.training:
            current_max, _ = torch.max(x, dim=0)

            if self.clip_value is None:
                self.clip_value = current_max
            else:
                self.clip_value = torch.max(self.clip_value, current_max)

        x = nn.functional.relu(x, inplace=True)

        if self.clip_value is not None:
            x = torch.min(x, torch.ones_like(x) * self.clip_value)
            
        return x