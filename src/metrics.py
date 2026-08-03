import numpy
import torch

def calculate_avg_distance(rot_gt, t_gt, rot_pred, t_pred, model_points):
    total_dist = 0
    for m in model_points:
        dist = abs((rot_pred * m + t_pred) - (rot_gt * m + t_gt))
        total_dist += dist

    avg_distance = total_dist / len(model_points)

    return avg_distance

def calculate_avg_distance_sym(rot_gt, t_gt, rot_pred, t_pred, model_points):
    total_dist = 0
    for m in model_points:
        min_dist = 1.0e9
        for m_p in model_points:
            dist = abs((rot_pred * m + t_pred) - (rot_gt * m_p + t_gt))
            if dist < min_dist:
                min_dist = dist
        total_dist += min_dist

    avg_distance_sym = total_dist / len(model_points)

    return avg_distance_sym