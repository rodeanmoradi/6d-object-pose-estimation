"""Qualitative pose-overlay report.

Runs a checkpoint over the YCB-V test targets, ranks every test sample by ADD-S,
and renders the extremes of that ranking: model points projected into the RGB
frame under the ground truth pose and under the predicted pose. Selection is
purely by ADD-S rank - that is stated on every figure so the panels can't be
mistaken for hand-picked examples.

Three figures land in the output directory:
  successes.png  - lowest-ADD-S samples
  failures.png   - highest-ADD-S samples
  symmetric.png  - best and worst sample among BOP-symmetric objects, so a
                   symmetric case is always in the report even when the two
                   figures above happen to contain none
plus report.json with the numbers behind every panel.

Run from the repo root (paths in the dataset pickles are relative to it):
    python scripts/report.py data/models/best_ne_20_bs_128_lr_0.0002.pt
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg") # No display attached; render straight to file.
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from PIL import Image
from torch.utils.data import DataLoader, Subset

from src import Baseline, PoseEstimator
from src.dataloader import YCBVDataset, get_relevant_indices
from src.evaluate import MM_TO_M, decode_pose, load_model_points, read_ply_vertices
from src.metrics import calculate_avg_distance, calculate_avg_distance_sym
from src.symmetry import DEFAULT_MODELS_DIR

# Standard YCB-V object names, for figure labels only - nothing keys off these.
OBJ_NAMES = {
    1: "master_chef_can", 3: "sugar_box", 5: "mustard_bottle", 7: "pudding_box",
    9: "potted_meat_can", 11: "pitcher_base", 13: "bowl", 15: "power_drill",
    17: "scissors", 19: "large_clamp",
}

GT_COLOR = "#00e676"
PRED_COLOR = "#ff2d95"


def build_test_loader(bs, workers):
    # Same test split the evaluation uses: build_dataloader's test branch. The loader
    # is rebuilt here rather than reused because the report needs the per-sample
    # metadata (image path, camera intrinsics, GT pose) that __getitem__ drops, and
    # with shuffle=False the position of a sample in the loader indexes test_indices.
    test_set = YCBVDataset("ycbv_test_all", deterministic=True)
    # An empty train set skips the train/val half of the filter; only the test branch runs.
    _, _, test_indices = get_relevant_indices([], test_set)
    loader = DataLoader(
        Subset(test_set, test_indices), batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False,
    )

    return test_set, test_indices, loader


@torch.no_grad()
def collect_predictions(model, loader, variant, model_points, device):
    # One pass over the test set keeping the predicted pose per sample, so the
    # renders below need no second forward pass.
    add_all, add_s_all, obj_index_all, rot_all, t_all = [], [], [], [], []
    for b in loader:
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}

        obj_index = b["obj_id"]
        geometry = b["pointcloud"] if variant == "rgbd" else b["geom"]

        out = model(b["rgb"], geometry, obj_index)
        t_pred, rot_pred = decode_pose(out, b, variant)

        points = model_points[obj_index] # (B, N, 3), per-sample so mixed batches work
        add_all.append(calculate_avg_distance(
            b["rotation_m2c"], b["translation_m2c"], rot_pred, t_pred, points).cpu())
        add_s_all.append(calculate_avg_distance_sym(
            b["rotation_m2c"], b["translation_m2c"], rot_pred, t_pred, points).cpu())
        obj_index_all.append(obj_index.cpu())
        rot_all.append(rot_pred.cpu())
        t_all.append(t_pred.cpu())

    return {
        "add": torch.cat(add_all).numpy(),
        "add_s": torch.cat(add_s_all).numpy(),
        "obj_index": torch.cat(obj_index_all).numpy(),
        "rot_pred": torch.cat(rot_all).numpy(),
        "t_pred": torch.cat(t_all).numpy(),
    }


def pick_by_rank(order, count, obj_index, unique_objects, exclude=()):
    # Walks the ADD-S ranking in order and takes the first `count` entries. With
    # unique_objects the walk skips a sample whose object class was already taken -
    # consecutive test frames of one object are near-identical, so without this the
    # top-3 is usually three views of the same can.
    chosen, seen = [], set()
    for position in order:
        if position in exclude:
            continue
        if unique_objects and obj_index[position] in seen:
            continue
        seen.add(obj_index[position])
        chosen.append(int(position))
        if len(chosen) == count:
            break

    return chosen


def get_render_points(obj_id, num_points, cache, models_dir, seed=0):
    # Denser than the metric's point set: these are only drawn, so more points make
    # the object's silhouette readable in the overlay.
    if obj_id not in cache:
        vertices = read_ply_vertices(Path(models_dir) / f"obj_{obj_id:06d}.ply") * MM_TO_M
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(vertices), size=num_points, replace=len(vertices) < num_points)
        cache[obj_id] = vertices[indices]

    return cache[obj_id]


def project(points, rot, t, cam_k):
    # p_cam = R @ p_model + t (BOP cam_R_m2c / cam_t_m2c), then the pinhole divide.
    # Points behind the camera are dropped rather than folded onto the image plane.
    camera_points = points @ np.asarray(rot).T + np.asarray(t)
    z = camera_points[:, 2]
    in_front = z > 1e-6
    camera_points, z = camera_points[in_front], z[in_front]

    fx, cx, fy, cy = cam_k[0], cam_k[2], cam_k[4], cam_k[5]
    u = fx * camera_points[:, 0] / z + cx
    v = fy * camera_points[:, 1] / z + cy

    return u, v


def view_box(gt_uv, pred_uv, bbox, shape, max_scale=3.0):
    # Square crop around the ground truth object, widened to include the prediction
    # only while it stays within max_scale of the GT box - a prediction off by a metre
    # would otherwise zoom the panel back out to the whole frame. The full-frame panel
    # is where such a prediction is meant to be seen.
    height, width = shape
    xs = [bbox[0], bbox[0] + bbox[2]]
    ys = [bbox[1], bbox[1] + bbox[3]]
    if len(gt_uv[0]):
        xs += [gt_uv[0].min(), gt_uv[0].max()]
        ys += [gt_uv[1].min(), gt_uv[1].max()]

    base = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.35
    center_x, center_y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

    if len(pred_uv[0]):
        with_pred_x = [min(xs), max(xs), pred_uv[0].min(), pred_uv[0].max()]
        with_pred_y = [min(ys), max(ys), pred_uv[1].min(), pred_uv[1].max()]
        size = max(max(with_pred_x) - min(with_pred_x), max(with_pred_y) - min(with_pred_y)) * 1.2
        if size <= max_scale * base:
            base = size
            center_x = (max(with_pred_x) + min(with_pred_x)) / 2
            center_y = (max(with_pred_y) + min(with_pred_y)) / 2

    half = max(base / 2, 20.0)
    x0, x1 = np.clip([center_x - half, center_x + half], 0, width)
    y0, y1 = np.clip([center_y - half, center_y + half], 0, height)

    return x0, x1, y0, y1


def draw_panel(ax, image, gt_uv, pred_uv, limits=None, size=1.0):
    ax.imshow(image)
    ax.scatter(gt_uv[0], gt_uv[1], s=size, c=GT_COLOR, alpha=0.45, linewidths=0)
    ax.scatter(pred_uv[0], pred_uv[1], s=size, c=PRED_COLOR, alpha=0.45, linewidths=0)
    if limits is not None:
        x0, x1, y0, y1 = limits
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0) # Image rows run top-down.
    ax.set_xticks([])
    ax.set_yticks([])


def render_figure(entries, title, subtitle, path, render_points_cache, args):
    figure, axes = plt.subplots(
        len(entries), 2, figsize=(10, 4.6 * len(entries)), squeeze=False
    )

    for row, entry in enumerate(entries):
        meta = entry["meta"]
        image = np.array(Image.open(meta["rgb"]))
        cam_k = meta["cam_K"]

        points = get_render_points(
            meta["obj_id"], args.render_points, render_points_cache, args.models_dir
        )
        rot_gt = np.asarray(meta["rotation_m2c"], dtype=np.float64).reshape(3, 3)
        t_gt = np.asarray(meta["translation_m2c"], dtype=np.float64) * MM_TO_M

        gt_uv = project(points, rot_gt, t_gt, cam_k)
        pred_uv = project(points, entry["rot_pred"], entry["t_pred"], cam_k)

        draw_panel(axes[row][0], image, gt_uv, pred_uv, size=0.6)
        draw_panel(
            axes[row][1], image, gt_uv, pred_uv, size=2.5,
            limits=view_box(gt_uv, pred_uv, meta["bbox_visib"], image.shape[:2]),
        )

        name = OBJ_NAMES.get(meta["obj_id"], "")
        symmetric = "  [symmetric]" if entry["symmetric"] else ""
        axes[row][0].set_title(
            f"obj {meta['obj_id']} {name}{symmetric}   scene {meta['scene_id']}/{Path(meta['rgb']).stem}\n"
            f"ADD-S {entry['add_s'] * 1000:.1f} mm   ADD {entry['add'] * 1000:.1f} mm"
            f"   rank {entry['rank']}/{entry['num_samples']} by ADD-S"
            # Occlusion is the usual reason a panel fails, so it belongs next to the error.
            f"   visible {meta['visib_fract'] * 100:.0f}%",
            fontsize=9, loc="left",
        )
        axes[row][1].set_title("zoomed on object", fontsize=9, loc="left")

    handles = [
        Line2D([], [], marker="o", linestyle="", color=GT_COLOR, label="ground truth pose"),
        Line2D([], [], marker="o", linestyle="", color=PRED_COLOR, label="predicted pose"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    figure.suptitle(f"{title}\n{subtitle}", fontsize=11)
    figure.tight_layout(rect=[0, 0.03, 1, 0.97])
    figure.savefig(path, dpi=args.dpi)
    plt.close(figure)


def build_entries(positions, predictions, test_set, test_indices, symmetric, ranks):
    entries = []
    for position in positions:
        entries.append({
            "position": position,
            "dataset_index": int(test_indices[position]),
            "meta": test_set.dataset[test_indices[position]],
            "rot_pred": predictions["rot_pred"][position],
            "t_pred": predictions["t_pred"][position],
            "add": float(predictions["add"][position]),
            "add_s": float(predictions["add_s"][position]),
            "symmetric": bool(symmetric[predictions["obj_index"][position]]),
            "rank": int(ranks[position]) + 1,
            "num_samples": len(predictions["add_s"]),
        })

    return entries


def summarise(entries):
    # Everything that identifies a panel, minus the arrays, for report.json.
    return [{
        "dataset_index": e["dataset_index"],
        "scene_id": e["meta"]["scene_id"],
        "frame": Path(e["meta"]["rgb"]).stem,
        "obj_id": e["meta"]["obj_id"],
        "obj_name": OBJ_NAMES.get(e["meta"]["obj_id"], ""),
        "symmetric": e["symmetric"],
        "visib_fract": e["meta"]["visib_fract"],
        "add_s_mm": e["add_s"] * 1000,
        "add_mm": e["add"] * 1000,
        "add_s_rank": e["rank"],
        "num_samples": e["num_samples"],
    } for e in entries]


def main():
    parser = argparse.ArgumentParser(description="Render pose-overlay successes and failures, ranked by ADD-S.")
    parser.add_argument("checkpoint")
    parser.add_argument("--variant", choices=["baseline", "rgbd"], default="rgbd")
    parser.add_argument("--out", default="reports", help="directory for the figures and report.json")
    parser.add_argument("--num-success", type=int, default=3)
    parser.add_argument("--num-failure", type=int, default=3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=1024, help="model points per object for the ADD-S ranking")
    parser.add_argument("--render-points", type=int, default=3000, help="model points drawn in the overlays")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--allow-duplicate-objects", action="store_true",
                        help="let one object class fill several panels instead of taking its best-ranked sample only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Baseline() if args.variant == "baseline" else PoseEstimator()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    model_points, _, symmetric = load_model_points(num_points=args.num_points, models_dir=args.models_dir)
    model_points = model_points.to(device)
    symmetric = symmetric.numpy()

    test_set, test_indices, loader = build_test_loader(args.bs, args.workers)
    print(f"scoring {len(test_indices)} test samples on {device}...")
    predictions = collect_predictions(model, loader, args.variant, model_points, device)

    add_s = predictions["add_s"]
    obj_index = predictions["obj_index"]
    order = np.argsort(add_s, kind="stable") # Ascending: best pose first, worst last.
    ranks = np.empty(len(add_s), dtype=np.int64)
    ranks[order] = np.arange(len(add_s))
    unique_objects = not args.allow_duplicate_objects

    success_positions = pick_by_rank(order, args.num_success, obj_index, unique_objects)
    failure_positions = pick_by_rank(
        order[::-1], args.num_failure, obj_index, unique_objects, exclude=set(success_positions)
    )

    # A symmetric case is guaranteed by ranking only the symmetric-object samples and
    # taking that ranking's two ends, independent of what the two figures above drew.
    symmetric_order = [p for p in order if symmetric[obj_index[p]]]
    symmetric_positions = []
    if symmetric_order:
        symmetric_positions = [int(symmetric_order[0])]
        if len(symmetric_order) > 1:
            symmetric_positions.append(int(symmetric_order[-1]))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_points_cache = {}

    selection_note = (
        f"Selected by ranking all {len(add_s)} test samples by ADD-S"
        + ("; one sample per object class." if unique_objects else ".")
    )
    figures = [
        (success_positions, "successes.png",
         f"Successes: the {len(success_positions)} lowest-ADD-S test samples", selection_note),
        (failure_positions, "failures.png",
         f"Failures: the {len(failure_positions)} highest-ADD-S test samples", selection_note),
        (symmetric_positions, "symmetric.png",
         "Symmetric object case: best and worst ADD-S among symmetric objects",
         f"Same ADD-S ranking of all {len(add_s)} test samples, restricted to objects BOP marks symmetric."),
    ]

    report = {
        "checkpoint": args.checkpoint,
        "variant": args.variant,
        "epoch": checkpoint.get("epoch"),
        "num_test_samples": int(len(add_s)),
        "selection": (
            "Every test sample was scored with ADD-S and sorted; successes are the "
            "lowest-ADD-S samples, failures the highest, and the symmetric figure is the "
            "same ranking restricted to symmetric objects."
            + (" At most one sample per object class." if unique_objects else "")
        ),
    }

    for positions, filename, title, subtitle in figures:
        if not positions:
            print(f"skipping {filename}: no samples matched")
            continue
        entries = build_entries(positions, predictions, test_set, test_indices, symmetric, ranks)
        render_figure(entries, title, subtitle, out_dir / filename, render_points_cache, args)
        report[Path(filename).stem] = summarise(entries)

        print(f"\n{title}\n  {subtitle}")
        for e in entries:
            name = OBJ_NAMES.get(e["meta"]["obj_id"], "")
            sym = " [symmetric]" if e["symmetric"] else ""
            print(
                f"  rank {e['rank']:>5}/{e['num_samples']}  obj {e['meta']['obj_id']:>2} {name:<16}"
                f" scene {e['meta']['scene_id']}/{Path(e['meta']['rgb']).stem}"
                f"  ADD-S {e['add_s'] * 1000:7.1f} mm  ADD {e['add'] * 1000:7.1f} mm{sym}"
            )
        print(f"  -> {out_dir / filename}")

    with open(out_dir / "report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"\nwrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
