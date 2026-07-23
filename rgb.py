import numpy as np
import torch
import torchvision
import torchvision.transforms.v2.functional as F
import torch.nn as nn

def init_resnet():
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights) # Pre-trained ResNet18
    model.fc = nn.Identity()

    return model