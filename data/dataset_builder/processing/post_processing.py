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
