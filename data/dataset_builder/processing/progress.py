import numpy as np
from scipy.interpolate import PchipInterpolator

def compute_progress(num_frames, ee_pose, finger_sensor, progress_dict=None):
    default_dict = {
        "finger_threshold": 800,
        "z_grasp": 0.10,
        "z_tolerance": 0.05
    }
    
    if progress_dict is not None:
        default_dict.update(progress_dict)
        
    finger_threshold = default_dict["finger_threshold"]
    z_grasp = default_dict["z_grasp"]
    z_tolerance = default_dict["z_tolerance"]

    z_positions = ee_pose[:, 2]
    
    idx_start = 0
    idx_end = num_frames - 1
    idx_grasp = -1
    
    for i in range(num_frames):
        is_finger_closed = finger_sensor[i] < finger_threshold
        is_at_grasp_height = abs(z_positions[i] - z_grasp) <= z_tolerance
        if is_finger_closed and is_at_grasp_height:
            idx_grasp = i
            break
            
    if idx_grasp == -1:
        return np.linspace(0.0, 1.0, num_frames)

    idx_peak = idx_grasp + np.argmax(z_positions[idx_grasp:])
    
    x_points = [idx_start, idx_grasp]
    y_points = [0.0, 0.5]
    
    if idx_grasp < idx_peak < idx_end:
        x_points.append(idx_peak)
        y_points.append(0.8)
    
    x_points.append(idx_end)
    y_points.append(1.0)
         
    # Eindeutige Stützstellen sicherstellen, falls Start/Grasp/Peak zusammenfallen
    x_points, indices = np.unique(x_points, return_index=True)
    y_points = np.array(y_points)[indices]
    
    # Monoton steigende kubische Spline-Interpolation (PCHIP)
    frames = np.arange(num_frames)
    interpolator = PchipInterpolator(x_points, y_points)
    progress_signal = interpolator(frames)
    
    return np.clip(progress_signal, 0.0, 1.0)