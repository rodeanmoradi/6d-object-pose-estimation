import numpy as np
import json
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path

data_path = Path("data")
train_path = data_path / "train_real"

dataset = []
for s in train_path.iterdir():
    rgb_path = s / "rgb"
    depth_path = s / "depth"
    mask_path = s / "mask"
    mask_visib_path = s / "mask_visib"

    with open(s / "scene_camera.json", "r", encoding="utf-8") as scene_camera:
        scene_camera = json.load(scene_camera)
    with open(s / "scene_gt_info.json", "r", encoding="utf-8") as scene_gt_info:
        scene_gt_info = json.load(scene_gt_info)
    with open(s / "scene_gt.json", "r", encoding="utf-8") as scene_gt:
        scene_gt = json.load(scene_gt)

    for rgb, depth in zip(sorted(rgb_path.iterdir()), sorted(depth_path.iterdir())):
        frame_id = rgb.stem
        key = str(int(frame_id))

        cam_k = scene_camera[key]["cam_K"]
        depth_scale = scene_camera[key]["depth_scale"]

        for object_index, (gt, gt_info) in enumerate(zip(scene_gt[key], scene_gt_info[key])):
            sample = {
                "obj_id": gt["obj_id"],
                "scene_id": s.name,
                "rgb": rgb,
                "depth": depth,
                "mask": mask_path / f"{frame_id}_{object_index:06d}.png",
                "mask_visib": mask_visib_path / f"{frame_id}_{object_index:06d}.png",
                "cam_K": cam_k,
                "depth_scale": depth_scale,
                "rotation_m2c": gt["cam_R_m2c"],
                "translation_m2c": gt["cam_t_m2c"],
                "bbox_obj": gt_info["bbox_obj"],
                "bbox_visib": gt_info["bbox_visib"],
                "px_count_all": gt_info["px_count_all"],
                "px_count_valid": gt_info["px_count_valid"],
                "px_count_visib": gt_info["px_count_visib"],
                "visib_fract": gt_info["visib_fract"],
            }
            dataset.append(sample)

print(len(dataset))
print(dataset[0])