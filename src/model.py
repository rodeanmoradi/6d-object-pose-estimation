import torch
import torch.nn as nn
import torch.nn.functional as F
from src.baseline import RegressionHead
from src.resnet import init_resnet
from src.pointnet import PointNetBackbone


class CentroidEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(3, 64)

    def forward(self, centroid):

        return F.relu(self.layer_1(centroid))

class PoseEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = init_resnet()
        self.pointnet = PointNetBackbone()
        self.encoder = CentroidEncoder()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=16)
        self.head = RegressionHead(input_size=1616, output_size=9) # 512 from resnet, 64 from encoder, 16 from embedding, 1024 from pointnet; 3 for translation, 6 for rotation

    def forward(self, rgb, pointcloud, centroid, obj_id):
        appearance_feats = self.resnet(rgb)
        geometry_feats = self.pointnet(pointcloud)
        centroid_encoding = self.encoder(centroid)
        object_embedding = self.embedding(obj_id)
        feats = torch.cat([object_embedding, appearance_feats, geometry_feats, centroid_encoding], dim=1)
        out = self.head(feats)

        return out