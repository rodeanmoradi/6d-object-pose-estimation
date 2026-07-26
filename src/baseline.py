import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from resnet import init_resnet
from dataloader import build_dataloader

class GeometryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(3, 64)

    def forward(self, x):

        return F.relu(self.layer_1(x))


class RegressionHead(nn.Module):
    def __init__(self, input_size):
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
        self.encoder = GeometryEncoder()
        self.head = RegressionHead(input_size=576) # 512 from resnet, 64 from encoder

    def forward(self, rgb, geom):
        appearance_feats = self.resnet(rgb)
        geom_feats = self.encoder(geom)
        feats = torch.cat([appearance_feats, geom_feats], dim=1)
        x = self.head(feats)

        return x

