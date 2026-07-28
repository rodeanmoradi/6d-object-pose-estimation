import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import build_dataloader

def get_translation(out, geom):
    z = (torch.exp(out[:, 0])).unsqueeze(-1)
    t = torch.stack((geom[:, 0], geom[:, 1], torch.ones_like(geom[:, 0])), dim=1)
    t = z * t

    return t

def gram_schmidt(out):
    # Head outputs a tensor of size 7 (first dims translation, last 6 rotation)
    rotation_6d = out[:, 1:]
    a = rotation_6d[:, :3]
    b = rotation_6d[:, 3:]

    norm_a = torch.linalg.vector_norm(a, dim=1, keepdim=True)
    u1 = a / norm_a

    proj = (b * u1).sum(dim=1, keepdim=True) * u1
    u2 = (b - proj) / torch.linalg.vector_norm(b - proj, dim=1, keepdim=True)

    u3 = torch.cross(u1, u2, dim=1)

    rot = torch.stack((u1, u2, u3), dim=-1)

    return rot

def train_model(model, ne, bs, lr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)
    translation_criterion = nn.SmoothL1Loss(beta=0.1)
    rotation_criterion = nn.MSELoss() # Chordal loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader, _ = build_dataloader(bs)
    loss_lambda = 0.5
    
    for e in range(ne):
        for b in train_loader:
            optimizer.zero_grad()
            rgb = b["rgb"].to(device)
            geom = b["geom"].to(device)
            out = model(rgb, geom)
            translation_gt = b["translation_m2c"].to(device)
            rotation_gt = b["rotation_m2c"].to(device)
            rot = gram_schmidt(out)
            t = get_translation(out, geom)
            translation_loss = translation_criterion(t, translation_gt)
            rotation_loss = rotation_criterion(rot, rotation_gt)
            loss = translation_loss + loss_lambda * rotation_loss
            loss.backward()
            optimizer.step()
    
    return 