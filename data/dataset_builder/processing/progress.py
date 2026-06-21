# dataset_builder/processing/progress.py
"""Progress signal computation."""

import numpy as np
from typing import Dict


def compute_progress(num_frames: int, config: Dict) -> np.ndarray:
    """
    Generate progress signal over episode duration.

    Modes:
        "linear": progress[t] = t / (T - 1), range [0, 1]

    Returns: (num_frames,) float32
    """
    # TODO: Switch on config["mode"]
    pass