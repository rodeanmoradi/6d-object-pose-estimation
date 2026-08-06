from pathlib import Path

import numpy as np
import torch

from src.dataloader import OBJ_IDS, build_dataloader
from src.metrics import calculate_avg_distance, calculate_avg_distance_sym
from src.symmetry import DEFAULT_MODELS_DIR, is_symmetric, load_models_info
from src.train import get_translation, gram_schmidt

# BOP ships models in mm; poses from the dataloader are in metres.
MM_TO_M = 1e-3
# YCB-V protocol: accuracy-threshold curve is integrated over [0, 0.1] m.
AUC_MAX_THRESHOLD = 0.1
# The other headline number from PoseCNN/DenseFusion: ADD-S under 2 cm.
ADD_S_THRESHOLD = 0.02

PLY_DTYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2",
    "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4",
    "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4",
    "double": "<f8", "float64": "<f8",
}


def read_ply_vertices(path):
    # Minimal reader for the PLY files BOP distributes: models_eval/ is binary and
    # carries only x/y/z, models/ is ascii and adds normals and colour. Both the
    # element ordering and the vertex stride come from the header rather than being
    # assumed, so either set can be read.
    with open(path, "rb") as file:
        header = b""
        while not header.endswith(b"end_header\n"):
            byte = file.read(1)
            if not byte:
                raise ValueError(f"{path}: reached EOF before end_header")
            header += byte

        fields = []
        count = None
        fmt = None
        in_vertex_element = False
        for line in header.decode("ascii").splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex_element = parts[1] == "vertex"
                if in_vertex_element:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex_element:
                if parts[1] == "list":
                    raise ValueError(f"{path}: list property in the vertex element is unsupported")
                fields.append((parts[2], PLY_DTYPES[parts[1]]))

        if count is None:
            raise ValueError(f"{path}: no vertex element in header")

        names = [name for name, _ in fields]
        if fmt == "binary_little_endian":
            dtype = np.dtype(fields)
            buffer = file.read(count * dtype.itemsize)
            if len(buffer) < count * dtype.itemsize:
                raise ValueError(f"{path}: truncated vertex data")
            vertices = np.frombuffer(buffer, dtype=dtype, count=count)
            columns = [vertices[axis] for axis in ("x", "y", "z")]
        elif fmt == "ascii":
            # One whitespace-separated row per vertex, in property order, before the faces.
            rows = []
            for line in file.read().decode("ascii").splitlines():
                if len(rows) == count:
                    break
                if line.strip():
                    rows.append(line.split()[:len(names)])
            if len(rows) < count:
                raise ValueError(f"{path}: truncated vertex data")
            values = np.array(rows, dtype=np.float64)
            columns = [values[:, names.index(axis)] for axis in ("x", "y", "z")]
        else:
            raise ValueError(f"{path}: unsupported ply format {fmt!r}")

    return np.stack(columns, axis=1).astype(np.float64)


def load_model_points(num_points=1024, seed=0, models_dir=DEFAULT_MODELS_DIR):
    # Returns a fixed number of points per object so a batch of mixed objects can be
    # stacked into a single (B, N, 3) tensor. models_eval/ is the subsampled mesh set
    # BOP intends for exactly this metric.
    models_dir = Path(models_dir)
    models_info = load_models_info(models_dir)

    rng = np.random.default_rng(seed)
    points = []
    diameters = []
    symmetric = []
    for obj_id in OBJ_IDS:
        vertices = read_ply_vertices(models_dir / f"obj_{obj_id:06d}.ply") * MM_TO_M
        indices = rng.choice(len(vertices), size=num_points, replace=len(vertices) < num_points)
        points.append(vertices[indices])

        info = models_info[str(obj_id)]
        diameters.append(info["diameter"] * MM_TO_M)
        symmetric.append(is_symmetric(info))

    points = torch.tensor(np.stack(points), dtype=torch.float32)
    diameters = torch.tensor(diameters, dtype=torch.float32)
    symmetric = torch.tensor(symmetric, dtype=torch.bool)

    return points, diameters, symmetric


def decode_pose(out, batch, variant):
    # Mirrors the decode each training loop applies to its head output.
    if variant == "baseline":
        return get_translation(out, batch["geom"]), gram_schmidt(out, rotation_start_index=1)
    if variant == "rgbd":
        return out[:, :3] + batch["centroid"], gram_schmidt(out, rotation_start_index=3)

    raise ValueError(f"unknown variant {variant!r}, expected 'baseline' or 'rgbd'")


def compute_auc(distances, max_threshold=AUC_MAX_THRESHOLD):
    # Area under the accuracy-threshold curve, normalised to a percentage.
    # acc(t) is a step function rising by 1/n at each sorted distance; samples beyond
    # max_threshold stay in the denominator but add no area.
    distances = np.sort(np.asarray(distances, dtype=np.float64))
    n = len(distances)
    if n == 0:
        return 0.0

    within = distances[distances <= max_threshold]
    if len(within) == 0:
        return 0.0

    accuracy = np.arange(1, len(within) + 1) / n
    edges = np.concatenate([within, [max_threshold]])
    area = float(np.sum(accuracy * np.diff(edges)))

    return 100.0 * area / max_threshold


@torch.no_grad()
def evaluate(model, checkpoint_path, variant, bs=64, num_points=1024, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    model_points, diameters, symmetric = load_model_points(num_points=num_points)
    model_points = model_points.to(device)
    diameters = diameters.to(device)
    symmetric = symmetric.to(device)

    _, _, test_loader = build_dataloader(bs)

    add_all = []
    add_s_all = []
    obj_index_all = []
    for b in test_loader:
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}

        obj_id = b["obj_id"]
        rgb = b["rgb"]
        geometry = b["pointcloud"] if variant == "rgbd" else b["geom"]

        out = model(rgb, geometry, obj_id)
        t_pred, rot_pred = decode_pose(out, b, variant)

        points = model_points[obj_id]  # (B, N, 3), per-sample so mixed batches work
        add = calculate_avg_distance(b["rotation_m2c"], b["translation_m2c"], rot_pred, t_pred, points)
        add_s = calculate_avg_distance_sym(b["rotation_m2c"], b["translation_m2c"], rot_pred, t_pred, points)

        add_all.append(add.cpu())
        add_s_all.append(add_s.cpu())
        obj_index_all.append(obj_id.cpu())

    add_all = torch.cat(add_all)
    add_s_all = torch.cat(add_s_all)
    obj_index_all = torch.cat(obj_index_all)

    # ADD(-S): ADD-S for objects BOP marks symmetric, plain ADD otherwise.
    is_symmetric = symmetric.cpu()[obj_index_all]
    add_sym_aware = torch.where(is_symmetric, add_s_all, add_all)
    thresholds = 0.1 * diameters.cpu()[obj_index_all]

    results = {
        "checkpoint": str(checkpoint_path),
        "variant": variant,
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("loss"),
        "num_samples": len(add_all),
        "add_s_auc": compute_auc(add_s_all.numpy()),
        "add_auc": compute_auc(add_sym_aware.numpy()),
        "add_s_below_2cm": 100.0 * (add_s_all < ADD_S_THRESHOLD).double().mean().item(),
        "add_below_0.1d": 100.0 * (add_sym_aware < thresholds).double().mean().item(),
        "per_object": {},
    }

    for index, obj_id in enumerate(OBJ_IDS):
        selected = obj_index_all == index
        if not selected.any():
            continue
        results["per_object"][obj_id] = {
            "num_samples": int(selected.sum()),
            "symmetric": bool(symmetric[index]),
            "add_s_auc": compute_auc(add_s_all[selected].numpy()),
            "add_auc": compute_auc(add_sym_aware[selected].numpy()),
            "add_s_below_2cm": 100.0 * (add_s_all[selected] < ADD_S_THRESHOLD).double().mean().item(),
            "add_below_0.1d": 100.0 * (add_sym_aware[selected] < thresholds[selected]).double().mean().item(),
        }

    return results


def print_results(results):
    print(f"\n{results['variant']}  ({results['checkpoint']}, epoch {results['epoch']}, {results['num_samples']} samples)")
    print(f"{'obj':>5} {'sym':>4} {'n':>6} {'ADD-S AUC':>10} {'ADD(-S) AUC':>12} {'ADD-S<2cm':>10} {'ADD(-S)<0.1d':>13}")
    for obj_id, r in results["per_object"].items():
        sym = "yes" if r["symmetric"] else ""
        print(
            f"{obj_id:>5} {sym:>4} {r['num_samples']:>6} {r['add_s_auc']:>10.2f} {r['add_auc']:>12.2f} "
            f"{r['add_s_below_2cm']:>10.2f} {r['add_below_0.1d']:>13.2f}"
        )
    print(
        f"{'ALL':>5} {'':>4} {results['num_samples']:>6} {results['add_s_auc']:>10.2f} {results['add_auc']:>12.2f} "
        f"{results['add_s_below_2cm']:>10.2f} {results['add_below_0.1d']:>13.2f}"
    )
