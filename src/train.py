import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import build_dataloader

def gram_schmidt(out):
    # Head outputs a tensor of size 9 (first 3 dims translation, last 6 rotation)
    rotation_6d = out[3:]
    a = rotation_6d[:3]
    b = rotation_6d[3:]

    norm_a = torch.linalg.vector_norm(a)
    u1 = a / norm_a

    proj = torch.dot(b, u1) * u1
    u2 = (b - proj) / torch.linalg.vector_norm(b - proj)

    u3 = torch.cross(u1, u2, dim=0)

    R = torch.stack((u1, u2, u3), dim=1)

    return R

def train_model(model, ne, bs, lr):
    model = model()
    translation_criterion = nn.SmoothL1Loss(beta=0.1)
    rotation_criterion = nn.MSELoss() # Chordal loss
    optimizer = optim.Adam(lr=lr)
    train_loader, val_loader, test_loader = build_dataloader()
    for e in range(ne):
        for b in train_loader:
            optimizer.zero_grad()
            out = model(b)
            R = gram_schmidt(out).flatten()
            loss = model.loss()
            loss.backward()
            optimizer.step()