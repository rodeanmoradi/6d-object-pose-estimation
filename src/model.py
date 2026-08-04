import torch
import torch.nn as nn
from src.baseline import RegressionHead
from src.resnet import init_resnet
from src.pointnet import PointNetBackbone


class PoseEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = init_resnet()
        self.pointnet = PointNetBackbone()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=16)
        self.head = RegressionHead(input_size=1552, output_size=9) # 512 from resnet, 16 from embedding, 1024 from pointnet; 3 for translation, 6 for rotation

    def forward(self, rgb, pointcloud, obj_id):
        appearance_feats = self.resnet(rgb)
        geometry_feats = self.pointnet(pointcloud)
        object_embedding = self.embedding(obj_id)
        feats = torch.cat([object_embedding, appearance_feats, geometry_feats], dim=1)
        out = self.head(feats)

        return out