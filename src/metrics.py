import torch

# Poses are applied as p_cam = R @ p_model + t (BOP cam_R_m2c / cam_t_m2c convention).
# model_points must be in the same units as t (metres) - BOP .ply models ship in mm.
# Accepts a single sample (rot (3,3), t (3,), model_points (N,3)) or a batch
# (rot (B,3,3), t (B,3), model_points (B,N,3) for mixed objects, or (N,3) if the
# whole batch is one object). Returns a scalar or a (B,) tensor of per-sample distances.

def transform_points(rot, t, model_points):
    # (..., N, 3) @ (..., 3, 3) applies rot to each row, so transpose to get R @ p
    return model_points @ rot.transpose(-2, -1) + t.unsqueeze(-2)

def calculate_avg_distance(rot_gt, t_gt, rot_pred, t_pred, model_points):
    # ADD: mean distance between corresponding model points under the two poses
    pts_pred = transform_points(rot_pred, t_pred, model_points)
    pts_gt = transform_points(rot_gt, t_gt, model_points)

    avg_distance = torch.linalg.vector_norm(pts_pred - pts_gt, dim=-1).mean(dim=-1)

    return avg_distance

def calculate_avg_distance_sym(rot_gt, t_gt, rot_pred, t_pred, model_points, chunk_size=256):
    # ADD-S: for each predicted point, distance to the *nearest* ground truth point
    pts_pred = transform_points(rot_pred, t_pred, model_points)
    pts_gt = transform_points(rot_gt, t_gt, model_points)

    # Chunked over the query points to bound the (B, N, N) distance matrix.
    # cdist's default matmul expansion loses precision exactly where ADD-S lives
    # (near-zero distances), so use the direct computation.
    min_dists = []
    for start in range(0, pts_pred.shape[-2], chunk_size):
        chunk = pts_pred[..., start:start + chunk_size, :]
        dists = torch.cdist(chunk, pts_gt, compute_mode="donot_use_mm_for_euclid_dist")
        min_dists.append(dists.min(dim=-1).values)

    avg_distance_sym = torch.cat(min_dists, dim=-1).mean(dim=-1)

    return avg_distance_sym
