import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.dataloader import OBJ_IDS

DEFAULT_MODELS_DIR = "data/ycbv_models/models_eval"
# Continuous symmetries (a can, a bowl) have no finite transform list, so the
# revolution is discretised. 36 steps puts equivalent poses within 5 degrees.
DEFAULT_NUM_CONTINUOUS = 36


def load_models_info(models_dir=DEFAULT_MODELS_DIR):
    with open(Path(models_dir) / "models_info.json", "r", encoding="utf-8") as info:
        return json.load(info)


def is_symmetric(info_entry):
    # BOP flags a symmetric object with an explicit transform list.
    return "symmetries_discrete" in info_entry or "symmetries_continuous" in info_entry


def rotation_about_axis(axis, angle):
    # Rodrigues' formula.
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])

    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def load_symmetry_rotations(num_continuous=DEFAULT_NUM_CONTINUOUS, models_dir=DEFAULT_MODELS_DIR):
    # Returns (num_classes, max_syms, 3, 3) rotations and a (num_classes, max_syms)
    # validity mask, indexed by the class index the dataloader emits. Identity is always
    # first, so an asymmetric object has exactly one valid entry and the symmetry-aware
    # loss collapses to the plain chordal loss for it.
    models_info = load_models_info(models_dir)

    per_object = []
    for obj_id in OBJ_IDS:
        entry = models_info[str(obj_id)]
        rotations = [np.eye(3)]

        for transform in entry.get("symmetries_discrete", []):
            # BOP stores a 4x4 model-space transform; only its rotation block matters
            # for a rotation-only loss.
            rotations.append(np.array(transform, dtype=np.float64).reshape(4, 4)[:3, :3])

        for transform in entry.get("symmetries_continuous", []):
            angles = np.linspace(0.0, 2.0 * np.pi, num_continuous, endpoint=False)
            for angle in angles[1:]: # angle 0 is the identity already present
                rotations.append(rotation_about_axis(transform["axis"], angle))

        per_object.append(np.stack(rotations))

    max_syms = max(len(r) for r in per_object)
    symmetries = np.tile(np.eye(3), (len(OBJ_IDS), max_syms, 1, 1))
    mask = np.zeros((len(OBJ_IDS), max_syms), dtype=bool)
    for i, rotations in enumerate(per_object):
        symmetries[i, :len(rotations)] = rotations
        mask[i, :len(rotations)] = True

    return torch.tensor(symmetries, dtype=torch.float32), torch.tensor(mask)


class SymmetryAwareChordalLoss(nn.Module):
    # Chordal (squared Frobenius) distance to the *nearest* pose-equivalent ground
    # truth rotation. Without this, a symmetric object is penalised for predicting a
    # visually identical pose, which puts a floor under the rotation loss.
    def __init__(self, num_continuous=DEFAULT_NUM_CONTINUOUS, models_dir=DEFAULT_MODELS_DIR):
        super().__init__()
        symmetries, mask = load_symmetry_rotations(num_continuous, models_dir)
        # Buffers so .to(device) and checkpointing follow the module.
        self.register_buffer("symmetries", symmetries)
        self.register_buffer("mask", mask)

    def forward(self, rot_pred, rot_gt, obj_id):
        symmetries = self.symmetries[obj_id] # (B, S, 3, 3)
        mask = self.mask[obj_id] # (B, S)

        # A pose (R, t) is equivalent to (R @ S, .) for every model symmetry S.
        equivalents = rot_gt.unsqueeze(1) @ symmetries
        error = ((rot_pred.unsqueeze(1) - equivalents) ** 2).mean(dim=(-2, -1)) # (B, S)
        error = error.masked_fill(~mask, float("inf"))

        return error.min(dim=1).values.mean()
