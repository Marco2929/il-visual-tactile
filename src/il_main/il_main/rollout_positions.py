#!/usr/bin/env python3
"""Reference start positions for robot rollout evaluation.

Defines the 5 standard positions (P1..P5) across box heights (mitte, oben, unten)
measured in the robot base coordinate system ('base_link') for tool0.

Derived from teleoperation demonstrations in data/recording/episodes/external.
"""
from typing import Dict, Tuple
import numpy as np

# Mapping of position codes to human-readable German labels
POSITIONEN: Dict[str, str] = {
    "P1": "Mitte",
    "P2": "vorne links",
    "P3": "vorne rechts",
    "P4": "hinten links",
    "P5": "hinten rechts",
}

# 7D Tool0 Pose in base_link: [x, y, z, qx, qy, qz, qw]
# Default / Anchor height is 'mitte'.
START_POSITIONS: Dict[Tuple[str, str], np.ndarray] = {
    # --- Height: mitte (Default / Anker) ---
    ("mitte", "P1"): np.array([-0.2612, -1.0014,  0.2273,  0.6944, -0.0324,  0.0542,  0.7168], dtype=np.float64),
    ("mitte", "P2"): np.array([-0.4364, -0.9164,  0.2276,  0.6944, -0.0324,  0.0542,  0.7168], dtype=np.float64),
    ("mitte", "P3"): np.array([-0.0859, -0.9337,  0.2275,  0.6940, -0.0323,  0.0542,  0.7172], dtype=np.float64),
    ("mitte", "P4"): np.array([-0.4365, -1.0690,  0.2271,  0.6948, -0.0326,  0.0543,  0.7164], dtype=np.float64),
    ("mitte", "P5"): np.array([-0.0860, -1.0863,  0.2270,  0.6948, -0.0326,  0.0543,  0.7164], dtype=np.float64),

    # --- Height: oben ---
    ("oben", "P1"):  np.array([-0.3962, -0.9722,  0.0775,  0.8089, -0.0682,  0.0547,  0.5814], dtype=np.float64),
    ("oben", "P2"):  np.array([-0.5605, -0.8772,  0.0761,  0.8146, -0.0587, -0.0206,  0.5767], dtype=np.float64),
    ("oben", "P3"):  np.array([-0.2554, -0.9215,  0.1081,  0.7849, -0.0249, -0.0097,  0.6191], dtype=np.float64),
    ("oben", "P4"):  np.array([-0.5692, -0.9826,  0.1437,  0.7758, -0.0436, -0.0007,  0.6295], dtype=np.float64),
    ("oben", "P5"):  np.array([-0.2716, -1.0767,  0.0613,  0.7852, -0.0785,  0.0190,  0.6139], dtype=np.float64),

    # --- Height: unten ---
    ("unten", "P1"): np.array([-0.3363, -1.0508, -0.0676,  0.7715, -0.0388, -0.0060,  0.6350], dtype=np.float64),
    ("unten", "P2"): np.array([-0.5522, -0.8989, -0.0549,  0.8070, -0.0153, -0.0500,  0.5882], dtype=np.float64),
    ("unten", "P3"): np.array([-0.2093, -0.9098, -0.0793,  0.7937, -0.0366, -0.0245,  0.6068], dtype=np.float64),
    ("unten", "P4"): np.array([-0.3661, -1.0869, -0.0276,  0.7474, -0.0331, -0.0178,  0.6633], dtype=np.float64),
    ("unten", "P5"): np.array([-0.1259, -1.0815, -0.0103,  0.7666, -0.0279,  0.0089,  0.6414], dtype=np.float64),
}


def get_start_pose(position: str, height: str = "mitte") -> np.ndarray:
    """Retrieve 7D pose [x, y, z, qx, qy, qz, qw] for a given position and height."""
    pos_key = position.upper().strip()
    if pos_key not in POSITIONEN:
        # Match by name or index
        matched = False
        for p_code, p_lbl in POSITIONEN.items():
            if pos_key.lower() == p_lbl.lower() or pos_key == p_code.lower():
                pos_key = p_code
                matched = True
                break
        if not matched:
            raise KeyError(
                f"Unknown position '{position}'. Available codes: {list(POSITIONEN.keys())} "
                f"({list(POSITIONEN.values())})"
            )

    h_key = height.lower().strip()
    key = (h_key, pos_key)
    if key in START_POSITIONS:
        return START_POSITIONS[key].copy()

    # Fallback to height 'mitte'
    fallback_key = ("mitte", pos_key)
    if fallback_key in START_POSITIONS:
        return START_POSITIONS[fallback_key].copy()

    raise KeyError(f"No pose found for key {key}")
