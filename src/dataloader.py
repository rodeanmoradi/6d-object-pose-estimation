import numpy as np
import json
import pickle
from PIL import Image
import imageio.v3 as iio
import torch
from torchvision.transforms.v2 import InterpolationMode
import torchvision.transforms.v2.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from pathlib import Path


class YCBVDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        data_path = Path("data")
        if dataset == "train_real" or dataset == "train_pbr":
            set_path = data_path / dataset
        else:
            set_path = data_path / dataset / "test"

        pkl_path = data_path / f"{dataset}.pkl"
        if pkl_path.exists():
            with open(data_path / f"{dataset}.pkl", "rb") as file:
                self.dataset = pickle.load(file)

        else: # dataset pickle does not exist, create it
            with open(data_path / "ycbv_base" / "ycbv" / "test_targets_bop19.json", "r", encoding="utf-8") as targets:
                targets = json.load(targets)

            targets_set = []
            for t in targets:
                targets_set.append((t["im_id"], t["obj_id"], t["scene_id"]))
            targets_set = set(targets_set)
            
            self.dataset = []
            for s in sorted(set_path.iterdir()):
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
                print(f"s: {s}") # REMOVE
                for rgb, depth in zip(sorted(rgb_path.iterdir()), sorted(depth_path.iterdir())):
                    frame_id = rgb.stem
                    key = str(int(frame_id))

                    cam_k = scene_camera[key]["cam_K"]
                    depth_scale = scene_camera[key]["depth_scale"]

                    depth_arr = iio.imread(depth) if dataset != "train_pbr" else None
                    print(f"rgb: {rgb}") # REMOVE
                    for object_index, (gt, gt_info) in enumerate(zip(scene_gt[key], scene_gt_info[key])):
                        if dataset != "train_pbr":
                            mask_visib = iio.imread(mask_visib_path / f"{frame_id}_{object_index:06d}.png")
                            valid = (mask_visib > 0) & (depth_arr > 0)
                            num_valid = np.count_nonzero(valid)
                        else:
                            num_valid = gt_info["px_count_valid"]

                        if dataset == "train_real" or dataset == "train_pbr":
                            if gt_info["visib_fract"] < 0.1 or num_valid < 10:
                                continue
                        elif dataset == "ycbv_test_all":
                            if not ((int(frame_id), int(gt["obj_id"]), int(s.name)) in targets_set) or num_valid < 10:
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
                        print(f"obj: {object_index}") # REMOVE
            with open(data_path / f"{dataset}.pkl", "wb") as file:
                pickle.dump(self.dataset, file)

    def __len__(self):
        return len(self.dataset)

    # TODO: center pointcloud
    def __getitem__(self, i):
        item = {}

        rgb = Image.open(self.dataset[i]["rgb"])
        rgb = np.array(rgb)
        depth = iio.imread(self.dataset[i]["depth"])
        mask_visib = iio.imread(self.dataset[i]["mask_visib"])

        # Crop using square bbox using centre and max len of original bbox
        bbox_visib = self.dataset[i]["bbox_visib"]
        box_x_c = int(bbox_visib[0] + bbox_visib[2] / 2)
        box_y_c = int(bbox_visib[1] + bbox_visib[3] / 2)
        box_max_len = max(bbox_visib[2], bbox_visib[3])
        start_x = int(box_x_c - box_max_len / 2)
        start_y = int(box_y_c - box_max_len / 2)
        rgb = F.to_image(rgb) # Reshape to CHW
        rgb = F.crop(rgb, start_y, start_x, box_max_len, box_max_len)
        rgb = F.resize(rgb, size=[224, 224]) # Resize
        rgb = F.to_dtype(rgb, dtype=torch.float32, scale=True)
        rgb = F.normalize(rgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        mask_visib_transform = F.to_image(mask_visib)
        mask_visib_transform = F.crop(mask_visib_transform, start_y, start_x, box_max_len, box_max_len)
        mask_visib_transform = F.resize(mask_visib_transform, size=[224, 224], interpolation=InterpolationMode.NEAREST)
        mask_visib_transform = mask_visib_transform.bool().to(torch.float32)

        rgb = rgb * mask_visib_transform
        item["rgb"] = rgb # Cropped, reshaped to CHW, resized, scaled, normalized, float32

        depth = depth * self.dataset[i]["depth_scale"]
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
        rng = np.random.default_rng(i) # Randomly pick 1000 indices (random sampling)
        indices = rng.integers(low=0, high=len(pointcloud), size=1000)
        pointcloud = pointcloud[indices]
        pointcloud *= 1e-3 # mm to m
        item["pointcloud"] = torch.tensor(pointcloud, dtype=torch.float32)

        # Compute scale-invariant translation values
        ray_x = (box_x_c - cx) / fx
        ray_y = (box_y_c - cy) / fy
        scale = np.log(box_max_len) - np.log(fy)
        geom = [ray_x, ray_y, scale]
        geom = torch.tensor(geom, dtype=torch.float32)
        item["geom"] = geom

        obj_id_mapping = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8, 19: 9}
        item["obj_id"] = obj_id_mapping[self.dataset[i]["obj_id"]]

        item["translation_m2c"] = torch.tensor(self.dataset[i]["translation_m2c"], dtype=torch.float32) * 1e-3
        rotation_m2c = torch.tensor(self.dataset[i]["rotation_m2c"], dtype=torch.float32)
        rotation_m2c = torch.reshape(rotation_m2c, (3, 3))
        item["rotation_m2c"] = rotation_m2c

        return item # dict containing rgb image, pointcloud, 


def get_relevant_indices(train_set, test_set=[]):
    #Train/val/test split: 64/16/12 scenes (70%/17%/13%); Train: scenes [0-47], [60-75], Val: [76-91], Test: [48-59]
    obj_ids = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    train_indices = []
    val_indices = []
    test_indices = []
    range_end = 50 if len(test_set) == 0 else 48
    for i in range(len(train_set)):
        if (
            (int(train_set.dataset[i]["scene_id"]) in range(0, range_end)
            or int(train_set.dataset[i]["scene_id"]) in range(60, 76))
            and int(train_set.dataset[i]["obj_id"]) in obj_ids
        ):
            train_indices.append(i)
        elif (
            int(train_set.dataset[i]["scene_id"]) in range(76, 92)
            and int(train_set.dataset[i]["obj_id"]) in obj_ids
        ):
            val_indices.append(i)

    for i in range(len(test_set)):
        if int(test_set.dataset[i]["obj_id"]) in obj_ids:
            test_indices.append(i)
    
    return train_indices, val_indices, test_indices

def build_dataloader(bs):
    train_set_real = YCBVDataset("train_real")
    train_set_synt = YCBVDataset("train_pbr")
    train_set = ConcatDataset([train_set_real, train_set_synt])
    test_set = YCBVDataset("ycbv_test_all")

    train_real_indices, val_indices, test_indices = get_relevant_indices(train_set_real, test_set)
    train_synt_indices, _, _ = get_relevant_indices(train_set_synt)
    train_indices = train_real_indices + [item + len(train_set_real) for item in train_synt_indices]

    train_ds = Subset(train_set, train_indices)
    val_ds = Subset(train_set, val_indices)
    test_ds = Subset(test_set, test_indices)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, drop_last=False)

    return train_loader, val_loader, test_loader
