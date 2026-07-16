import numpy as np
import json
from PIL import Image
import imageio.v3 as iio
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.v2.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path

class YCBVDataset(torch.utils.data.Dataset):
    def __init__(self):
        data_path = Path("data")
        train_path = data_path / "train_real"

        self.dataset = []
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
                    if gt_info["visib_fract"] < 0.1:
                        continue

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
                    self.dataset.append(sample)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        item = {}

        rgb = Image.open(self.dataset[i]["rgb"])
        rgb = np.array(rgb)
        bbox_visib = self.dataset[i]["bbox_visib"]
        rgb = rgb[bbox_visib[1]:(bbox_visib[1] + bbox_visib[3]), bbox_visib[0]:(bbox_visib[0] + bbox_visib[2])] # Crop using the object bounding box (x, y, width, height)
        rgb = F.to_image(rgb) # Reshape to CHW 
        rgb = F.resize(rgb, size=[128, 128]) # Resize to 128 x 128
        rgb = F.to_dtype(rgb, dtype=torch.float32, scale=True)
        rgb = F.normalize(rgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        item["rgb"] = rgb # Cropped, reshaped to CHW, resized, scaled, normalized, float32

        depth = iio.imread(self.dataset[i]["depth"])
        depth = np.array(depth)
        depth = depth * self.dataset[i]["depth_scale"]
        mask_visib = iio.imread(self.dataset[i]["mask_visib"])
        mask_visib = np.array(mask_visib)
        mask = (mask_visib > 0) & (depth > 0) # Mask is true when depth is nonzero and mask_visib is not black
        y_im, x_im = np.nonzero(mask) # np.nonzero() returns tuple of two arrays (for each dim) where ith row col pair is masked coord
        # Back-projection
        cam_k = self.dataset[i]["cam_K"]
        fx = cam_k[0]
        cx = cam_k[2]
        fy = cam_k[4]
        cy = cam_k[5]
        z = depth[y_im, x_im]
        y_cam = ((y_im - cy) * z) / fy
        x_cam = ((x_im - cx) * z) / fx
        pointcloud = np.stack([x_cam, y_cam, z], axis=1) # Creates an array where each entry is a given x, y, z point
        indices = np.random.choice(range(len(pointcloud)), 1000, replace=True) # Randomly pick 1000 indices (random sampling)
        pointcloud = pointcloud[indices]
        pointcloud *= 1e-3 # mm to m
        item["pointcloud"] = torch.tensor(pointcloud, dtype=torch.float32)

        item["obj_id"] = self.dataset[i]["obj_id"]
        item["translation_m2c"] = torch.tensor(self.dataset[i]["translation_m2c"], dtype=torch.float32) * 1e-3
        rotation_m2c = torch.tensor(self.dataset[i]["rotation_m2c"], dtype=torch.float32)
        rotation_m2c = torch.reshape(rotation_m2c, (3, 3))
        item["rotation_m2c"] = rotation_m2c

        return item