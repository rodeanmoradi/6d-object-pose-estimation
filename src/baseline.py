import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from resnet import init_resnet
from dataloader import build_dataloader

class RegressionHead(nn.Module):
    def __init__(self, input_size=512):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, 256)
        self.layer_2 = nn.Linear(256, 64)
        self.layer_3 = nn.Linear(64, 7)
    
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        
        return self.layer_3(x)

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = init_resnet()
        self.head = RegressionHead()

    def forward(self, x):
        x = self.resnet(x)
        x = self.head(x)

        return x
    
# TODO: Rectify crop issue
def train():
    baseline = Baseline()

