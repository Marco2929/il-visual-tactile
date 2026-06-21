# dataset_builder/processing/actions.py

import numpy as np
from typing import Dict


def compute_actions(ee_pose: np.ndarray,
                    hand_state: np.ndarray,
                    joint_states: np.ndarray,
                    config: Dict) -> np.ndarray:
    """
    Compute PRIMARY action target (robot executes this).

    Config:
        type: "next_state"
        derived_from: "ee_pose" | "joint_states"

    Returns: (N, action_dim) float32
        ee_pose    → (N, 7 + hand_dim)
        joint_states → (N, 6 + hand_dim)
    """
    derived_from = config["derived_from"]

    if derived_from == "ee_pose":
        arm_data = ee_pose                     # (N, 7)
    elif derived_from == "joint_states":
        arm_data = joint_states[:, :6]         # (N, 6)
    else:
        raise ValueError(f"Unknown derived_from: '{derived_from}'")

    if config["type"] == "next_state":
        arm_actions = _shift_next(arm_data)
        hand_actions = _shift_next(hand_state)
        return np.concatenate([arm_actions, hand_actions], axis=1).astype(np.float32)

    raise ValueError(f"Unknown action type: '{config['type']}'")


def compute_auxiliary_actions(ee_pose: np.ndarray,
                              joint_states: np.ndarray,
                              force: np.ndarray,
                              config: Dict) -> Dict[str, np.ndarray]:
    """
    Compute AUXILIARY action targets for multi-objective decoder loss.

    Supports:
      - ee_pose: geometric grounding (R3D §4.6)
      - force:   physical grounding  (TA-VLA §5, AdapTac §IV-B)
      - joint_states: if EE is primary and you want joint aux

    These are NEVER sent to the robot — only provide additional supervision.

    Returns: {} if auxiliary_actions is empty (Variante A)
             {"ee_pose": (N,7), "force": (N,6)} for Variante B
    """
    source_map = {
        "ee_pose": ee_pose,                     # (N, 7)
        "joint_states": joint_states[:, :6],    # (N, 6)
        "force": force,                         # (N, 6)
    }

    aux_actions = {}

    for entry in config:
        key = entry["key"]
        derived_from = entry["derived_from"]

        if derived_from not in source_map:
            raise ValueError(f"Unknown aux derived_from: '{derived_from}'")

        data = source_map[derived_from]

        # Skip if source is empty (e.g. force disabled → shape (N, 0))
        if data.shape[1] == 0:
            continue

        if entry["type"] == "next_state":
            aux_actions[key] = _shift_next(data).astype(np.float32)
        else:
            raise ValueError(f"Unknown aux action type: '{entry['type']}'")

    return aux_actions


def _shift_next(data: np.ndarray) -> np.ndarray:
    """Action = next state. Last frame repeats."""
    actions = np.empty_like(data)
    actions[:-1] = data[1:]
    actions[-1] = data[-1]
    return actions