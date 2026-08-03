import torch
import torch.nn as nn
import torch.nn.functional as F
from src.resnet import init_resnet


class GeometryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 64)

    def forward(self, x):

        return F.relu(self.layer1(x))


class RegressionHead(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        return self.layer3(x)


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = init_resnet()
        self.encoder = GeometryEncoder()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=16)
        self.head = RegressionHead(input_size=592, output_size=7) # 512 from resnet, 64 from encoder, 16 from embedding; 1 for translation, 6 for rotation

    def forward(self, rgb, geom, obj_id):
        appearance_feats = self.resnet(rgb)
        geom_feats = self.encoder(geom)
        object_embedding = self.embedding(obj_id)
        feats = torch.cat([object_embedding, appearance_feats, geom_feats], dim=1)
        out = self.head(feats)

        return out

