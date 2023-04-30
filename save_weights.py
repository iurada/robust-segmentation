import os
import torch
from torchvision.models import resnet50, efficientnet_v2_s


models = [('resnet50', resnet50(weights="IMAGENET1K_V2")),
          ('efficientnet_v2_s', efficientnet_v2_s(weights='IMAGENET1K_V1'))]

for model_name, model in models:
    torch.save(model.state_dict(), os.path.join('pytorch_scripts/pretrained_weights', model_name))