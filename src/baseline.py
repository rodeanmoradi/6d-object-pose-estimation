import torch
import torch.nn as nn
import torch.nn.functional as F
from src.resnet import init_resnet


class GeometryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(3, 64)

    def forward(self, x):

        return F.relu(self.layer_1(x))

class RegressionHead(nn.Module): # TODO: Two heads or one?
    def __init__(self, input_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, 256)
        self.layer_2 = nn.Linear(256, 64)
        self.layer_3 = nn.Linear(64, 7) # 1 for translation (Z), 6 for rotation
    
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        
        return self.layer_3(x)


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = init_resnet()
        self.encoder = GeometryEncoder()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=16)
        self.head = RegressionHead(input_size=592) # 512 from resnet, 64 from encoder, 16 from embedding

    def forward(self, rgb, geom, obj_id):
        appearance_feats = self.resnet(rgb)
        geom_feats = self.encoder(geom)
        object_embedding = self.embedding(obj_id)
        feats = torch.cat([object_embedding, appearance_feats, geom_feats], dim=1)
        out = self.head(feats)

        return out

