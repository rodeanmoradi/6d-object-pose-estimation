import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pathlib import Path
from src.dataloader import build_dataloader
from src.symmetry import SymmetryAwareChordalLoss

CHECKPOINT_DIR = Path("data/models")


def save_checkpoint(path, model, optimizer, epoch, loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "epoch": epoch
    }, path)

def get_translation(out, geom):
    log_z = torch.clamp(out[:, 0], -3.0, 2.0)
    z = (torch.exp(log_z)).unsqueeze(-1)
    t = torch.stack((geom[:, 0], geom[:, 1], torch.ones_like(geom[:, 0])), dim=1)
    t = z * t

    return t

def gram_schmidt(out, rotation_start_index):
    epsilon = 1.0e-8
    # Head outputs a tensor of size 7 or 9 (first 1 or 3 dims translation, last 6 rotation)
    rotation_6d = out[:, rotation_start_index:]
    a = rotation_6d[:, :3]
    b = rotation_6d[:, 3:]

    norm_a = torch.linalg.vector_norm(a, dim=1, keepdim=True)
    u1 = a / (norm_a + epsilon)

    proj = (b * u1).sum(dim=1, keepdim=True) * u1
    u2 = (b - proj) / (torch.linalg.vector_norm(b - proj, dim=1, keepdim=True) + epsilon)

    u3 = torch.cross(u1, u2, dim=1)

    rot = torch.stack((u1, u2, u3), dim=-1)

    return rot

# TODO: Display plot after each full loop
def train_baseline(model, ne, bs, lr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    # Beta is scale-coupled to the translation parametrisation, so it differs from
    # the RGB-D variant on purpose - see train_model.
    translation_beta = 0.1
    loss_lambda = 0.5

    run = wandb.init(
        entity="rodean-moradi-university-of-toronto",
        project="6d-pose-estimation",
        config={
            "learning_rate": lr,
            "epochs": ne,
            "batch_size": bs,
            "variant": "RGB",
            "dataset": "YCB-Video",
            "optimizer": "Adam",
            "smooth_l1_beta": translation_beta,
            "loss_lambda": loss_lambda,
            "rotation_loss": "symmetry_aware_chordal",
        }
    )

    translation_criterion = nn.SmoothL1Loss(beta=translation_beta)
    rotation_criterion = SymmetryAwareChordalLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader, val_loader, _ = build_dataloader(bs)

    best_val_loss = 1.0e9
    for e in range(ne):
        model.train()
        train_t_loss_running = 0.0
        train_rot_loss_running = 0.0
        train_loss_running = 0.0
        for b in train_loader:
            optimizer.zero_grad()

            rgb = b["rgb"].to(device)
            geom = b["geom"].to(device)
            obj_id = b["obj_id"].to(device)
            translation_gt = b["translation_m2c"].to(device)
            rotation_gt = b["rotation_m2c"].to(device)

            out = model(rgb, geom, obj_id)
            
            t = get_translation(out, geom)
            rot = gram_schmidt(out, rotation_start_index=1)

            translation_loss = translation_criterion(t, translation_gt)
            rotation_loss = rotation_criterion(rot, rotation_gt, obj_id)
            train_loss = translation_loss + loss_lambda * rotation_loss

            train_t_loss_running += translation_loss.item()
            train_rot_loss_running += rotation_loss.item()
            train_loss_running += train_loss.item()

            train_loss.backward()

            optimizer.step()

        train_t_loss_running /= len(train_loader)
        train_rot_loss_running /= len(train_loader)
        train_loss_running /= len(train_loader)

        model.eval()
        val_t_loss_running = 0.0
        val_rot_loss_running = 0.0
        val_loss_running = 0.0
        with torch.no_grad():
            for b in val_loader:
                rgb = b["rgb"].to(device)
                geom = b["geom"].to(device)
                obj_id = b["obj_id"].to(device)
                translation_gt = b["translation_m2c"].to(device)
                rotation_gt = b["rotation_m2c"].to(device)
    
                out = model(rgb, geom, obj_id)
                
                t = get_translation(out, geom)
                rot = gram_schmidt(out, rotation_start_index=1)
    
                translation_loss = translation_criterion(t, translation_gt)
                rotation_loss = rotation_criterion(rot, rotation_gt, obj_id)
                val_loss = translation_loss + loss_lambda * rotation_loss

                val_t_loss_running += translation_loss.item()
                val_rot_loss_running += rotation_loss.item()
                val_loss_running += val_loss.item()

            val_t_loss_running /= len(val_loader)
            val_rot_loss_running /= len(val_loader)
            val_loss_running /= len(val_loader)

            if val_loss_running < best_val_loss:
                best_val_loss = val_loss_running

                save_checkpoint(
                    CHECKPOINT_DIR / f"best_baseline_ne_{ne}_bs_{bs}_lr_{lr}.pt",
                    model, optimizer, e, best_val_loss
                )

        run.log({
            "epoch": e,
            "train/total_loss": train_loss_running,
            "train/rotation_loss": train_rot_loss_running,
            "train/translation_loss": train_t_loss_running,
            "val/total_loss": val_loss_running,
            "val/rotation_loss": val_rot_loss_running,
            "val/translation_loss": val_t_loss_running
        })
    run.finish()

    return

# TODO: Determine metrics/results needed
def train_model(model, ne, bs, lr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Translation here is a residual off the point-cloud centroid rather than a depth
    # regressed from scratch, so beta sits an order of magnitude below the baseline's.
    translation_beta = 0.01
    loss_lambda = 0.5
    weight_decay = 1e-4

    run = wandb.init(
        entity="rodean-moradi-university-of-toronto",
        project="6d-pose-estimation",
        config={
            "learning_rate": lr,
            "epochs": ne,
            "batch_size": bs,
            "variant": "RGB-D",
            "dataset": "YCB-Video",
            "optimizer": "AdamW",
            "weight_decay": weight_decay,
            "smooth_l1_beta": translation_beta,
            "loss_lambda": loss_lambda,
            "rotation_loss": "symmetry_aware_chordal",
        }
    )

    model.to(device)
    translation_criterion = nn.SmoothL1Loss(beta=translation_beta)
    rotation_criterion = SymmetryAwareChordalLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader, val_loader, _ = build_dataloader(bs)
    best_val_loss = 1.0e9
    for e in range(ne):
        model.train()
        train_t_loss_running = 0.0
        train_rot_loss_running = 0.0
        train_loss_running = 0.0
        for b in train_loader:
            optimizer.zero_grad()

            rgb = b["rgb"].to(device)
            pointcloud = b["pointcloud"].to(device)
            obj_id = b["obj_id"].to(device)
            centroid = b["centroid"].to(device)
            translation_gt = b["translation_m2c"].to(device)
            rotation_gt = b["rotation_m2c"].to(device)

            out = model(rgb, pointcloud, obj_id)
            
            t = out[:, :3] + centroid
            rot = gram_schmidt(out, rotation_start_index=3)

            translation_loss = translation_criterion(t, translation_gt)
            rotation_loss = rotation_criterion(rot, rotation_gt, obj_id)
            train_loss = translation_loss + loss_lambda * rotation_loss

            train_t_loss_running += translation_loss.item()
            train_rot_loss_running += rotation_loss.item()
            train_loss_running += train_loss.item()

            train_loss.backward()

            optimizer.step()

        train_t_loss_running /= len(train_loader)
        train_rot_loss_running /= len(train_loader)
        train_loss_running /= len(train_loader)

        model.eval()
        val_t_loss_running = 0.0
        val_rot_loss_running = 0.0
        val_loss_running = 0.0
        with torch.no_grad():
            for b in val_loader:
                rgb = b["rgb"].to(device)
                pointcloud = b["pointcloud"].to(device)
                obj_id = b["obj_id"].to(device)
                centroid = b["centroid"].to(device)
                translation_gt = b["translation_m2c"].to(device)
                rotation_gt = b["rotation_m2c"].to(device)
    
                out = model(rgb, pointcloud, obj_id)
                
                t = out[:, :3] + centroid
                rot = gram_schmidt(out, rotation_start_index=3)
    
                translation_loss = translation_criterion(t, translation_gt)
                rotation_loss = rotation_criterion(rot, rotation_gt, obj_id)
                val_loss = translation_loss + loss_lambda * rotation_loss

                val_t_loss_running += translation_loss.item()
                val_rot_loss_running += rotation_loss.item()
                val_loss_running += val_loss.item()

            val_t_loss_running /= len(val_loader)
            val_rot_loss_running /= len(val_loader)
            val_loss_running /= len(val_loader)

            if val_loss_running < best_val_loss:
                best_val_loss = val_loss_running

                save_checkpoint(
                    CHECKPOINT_DIR / f"best_ne_{ne}_bs_{bs}_lr_{lr}.pt",
                    model, optimizer, e, best_val_loss
                )

        run.log({
            "epoch": e,
            "train/total_loss": train_loss_running,
            "train/rotation_loss": train_rot_loss_running,
            "train/translation_loss": train_t_loss_running,
            "val/total_loss": val_loss_running,
            "val/rotation_loss": val_rot_loss_running,
            "val/translation_loss": val_t_loss_running
        })
    run.finish()

    return
