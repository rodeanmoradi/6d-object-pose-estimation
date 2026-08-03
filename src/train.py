import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import build_dataloader

def get_translation(out, geom):
    log_z = torch.clamp(out[:, 0], -3.0, 2.0)
    z = (torch.exp(log_z)).unsqueeze(-1)
    t = torch.stack((geom[:, 0], geom[:, 1], torch.ones_like(geom[:, 0])), dim=1)
    t = z * t

    return t

def gram_schmidt(out):
    epsilon = 1.0e-8
    # Head outputs a tensor of size 7 (first dims translation, last 6 rotation) TODO: Update indices and add var
    rotation_6d = out[:, 1:]
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

    translation_criterion = nn.SmoothL1Loss(beta=0.1)
    rotation_criterion = nn.MSELoss() # Chordal loss

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader, val_loader, _ = build_dataloader(bs)

    loss_lambda = 0.5
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
            rot = gram_schmidt(out)

            translation_loss = translation_criterion(t, translation_gt)
            rotation_loss = rotation_criterion(rot, rotation_gt)
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
                rot = gram_schmidt(out)
    
                translation_loss = translation_criterion(t, translation_gt)
                rotation_loss = rotation_criterion(rot, rotation_gt)
                val_loss = translation_loss + loss_lambda * rotation_loss

                val_t_loss_running += translation_loss.item()
                val_rot_loss_running += rotation_loss.item()
                val_loss_running += val_loss.item()

            val_t_loss_running /= len(val_loader)
            val_rot_loss_running /= len(val_loader)
            val_loss_running /= len(val_loader)

            if val_loss_running < best_val_loss:
                best_val_loss = val_loss_running

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "loss": best_val_loss,
                    "epoch": e
                }
                torch.save(checkpoint, f"data/models/best_baseline_ne_{ne}_bs_{bs}_lr_{lr}.pt")

        print(f"Epoch: {e}, Training Loss (Total): {train_loss_running}, Training Loss (Rotation): {train_rot_loss_running}, Training Loss (Translation): {train_t_loss_running}")
        print(f"Epoch: {e}, Validation Loss (Total): {val_loss_running}, Validation Loss (Rotation): {val_rot_loss_running}, Validation Loss (Translation): {val_t_loss_running}")

    return

# TODO: Determine metrics/results needed
def train():

    return
