import numpy
import torch

def calculate_avg_distance(rot_gt, t_gt, rot_pred, t_pred, model_points, pointcloud):
    

    return

def calculate_avg_distance_sym(rot_gt, t_gt, rot_pred, t_pred, model_points):
    sum = 0
    predicted_points = rot_pred * model_points + t_pred # dont get what the starting point is and how it gets transformed or what gt is
    for p in predicted_points:
        m_min = model_points[0]
        for m, i in enumerate(model_points):
            if m - p < m_min:
                m_min = m
            sum += abs((rot_pred * m + t_pred) - (rot_gt * m_min + t_gt))

    avg_distance_sym = sum / len(model_points)

    return avg_distance_sym