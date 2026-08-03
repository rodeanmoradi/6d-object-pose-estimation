import numpy as np
import torchvision
import torch.nn as nn

def init_resnet():
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights) # Pre-trained ResNet18
    model.fc = nn.Identity()

    return model