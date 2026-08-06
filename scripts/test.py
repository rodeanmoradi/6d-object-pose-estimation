import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from src import YCBVDataset, Baseline, gram_schmidt, get_translation
import torch.optim as optim

if __name__ == "__main__":
    train_set = YCBVDataset("train_real")
    obj_ids = [3, 5, 7, 9, 11]
    train_indices = []
    for i in range(len(train_set)):
        if (
            int(train_set.dataset[i]["scene_id"]) in range(0, 40)
            and int(train_set.dataset[i]["obj_id"]) in obj_ids
            and train_set.dataset[i]["visib_fract"] > 0.5
        ):
            train_indices.append(i)
    train_indices = train_indices[::100]
    train_indices = train_indices[:5]
    train_ds = Subset(train_set, train_indices)
    train_loader = DataLoader(train_ds, batch_size=5, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True, drop_last=True)

    model = Baseline()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)
    translation_criterion = nn.SmoothL1Loss(beta=0.1)
    rotation_criterion = nn.MSELoss() # Chordal loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_lambda = 0.5
    for e in range(500):
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

        if e % 100 == 0:
            print(f"Epoch: {e}, Training Loss (Total): {train_loss_running}, Training Loss (Rotation): {train_rot_loss_running}, Training Loss (Translation): {train_t_loss_running}")