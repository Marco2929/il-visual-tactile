# dataset_builder/processing/post_processing.py

import numpy as np
from typing import Dict


def apply_post_processing(episode_data: Dict[str, np.ndarray],
                          config: Dict) -> Dict[str, np.ndarray]:
    """
    Apply all configured post-processing steps in order.

    Returns: Modified episode_data
    """
    if config.get("repeat_final_state", {}).get("enabled"):
        n = config["repeat_final_state"]["num_repeats"]
        episode_data = _repeat_final_state(episode_data, n)

    return episode_data


def _repeat_final_state(episode_data: Dict[str, np.ndarray],
                        num_repeats: int) -> Dict[str, np.ndarray]:
    """
    Append final frame N times to all arrays.
    Ensures the policy sees a "hold still" signal at episode end.
    """
    for key in episode_data:
        arr = episode_data[key]

        if arr is None:
            continue

        # Last frame: shape is arr[-1].shape (could be 1D, 2D, etc.)
        last_frame = arr[-1:]  # (1, ...) — keeps dims

        # Repeat along axis 0
        repeated = np.repeat(last_frame, num_repeats, axis=0)  # (N, ...)

        episode_data[key] = np.concatenate([arr, repeated], axis=0)

    episode_data = {key: val for key, val in episode_data.items() if val is not None}

    return episode_data

def compute_progress(num_frames, ee_pose, finger_sensor, progress_config):
    """
    Args:
        ee_pose: [N, 13]
        finger_sensor: [N] (Wert des Zeigefinger-Sensors)
        progress_config: Enthält 'z_grasp', 'z_lift', 'finger_threshold'
    """
    z_positions = ee_pose[:, 2]
    finger_threshold = progress_config.get("finger_threshold", 800)
    
    progress_signal = np.zeros(num_frames)
    
    # State-Tracking: Hat der Roboter den Greif-Vorgang bereits abgeschlossen?
    # Sobald der Finger einmal unter dem Schwellenwert war, 
    # sind wir in der 'Post-Grasp' Phase.
    has_grasped = False
    
    for i in range(num_frames):
        # 1. Update Greif-Status
        if finger_sensor[i] < finger_threshold:
            has_grasped = True
            
        # 2. Score-Berechnung basierend auf Status
        if not has_grasped:
            # Annäherungsphase (0.0 bis 0.5)
            # Nutze z_grasp als Anker für das Ende der Annäherung
            z_norm = (z_positions[i] - progress_config['z_grasp']) / (max(z_positions) - progress_config['z_grasp'])
            progress_signal[i] = max(0.0, 0.5 * (1.0 - z_norm))
        else:
            # Post-Grasp Phase (0.5 bis 1.0)
            # Wir sind hier, weil der Finger sicher zu ist.
            # Fortschritt skaliert jetzt mit der Höhe der Z-Bewegung nach oben
            z_lift_norm = (z_positions[i] - progress_config['z_grasp']) / (progress_config['z_lift'] - progress_config['z_grasp'])
            progress_signal[i] = 0.5 + 0.5 * np.clip(z_lift_norm, 0.0, 1.0)
            
    return np.clip(progress_signal, 0.0, 1.0)