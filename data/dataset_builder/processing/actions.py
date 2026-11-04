# dataset_builder/processing/actions.py

import numpy as np
from typing import Dict

from .proprioception import (
    TESOLLO_HAND_DOF, pinch_hand_indices, tesollo_joint_name,
)

# Commanded tool0 pose (xyz + quat xyzw) recorded on /teleop/target_pose.
RECORDED_ARM_DIM = 7


def reduce_recorded_action(recorded_action: np.ndarray,
                           hand_config: Dict) -> np.ndarray:
    """Apply the pinch reduction to the recorded command's hand block.

    The recorded action is always 7 + 20: the hand controller commands every
    joint, regardless of how many the policy predicts. compute_hand_state()
    reduces the OBSERVATION to the pinch DoF, so the action must be reduced
    with the SAME index map or state and action describe different hands -
    and the mismatch only shows up as a shape error much later, or worse, not
    at all.
    """
    if not hand_config.get("pinch", False):
        return recorded_action

    hand = recorded_action[:, RECORDED_ARM_DIM:]
    if hand.shape[1] != TESOLLO_HAND_DOF:
        raise ValueError(
            f"pinch: true expects a {TESOLLO_HAND_DOF}-DoF recorded hand "
            f"action (finger-major rj_dg_<1-5>_<1-4>), got {hand.shape[1]}."
        )

    keep = pinch_hand_indices()
    return np.concatenate(
        [recorded_action[:, :RECORDED_ARM_DIM], hand[:, keep]], axis=1
    ).astype(np.float32)


def compute_actions(ee_pose: np.ndarray,
                    hand_state: np.ndarray,
                    joint_states: np.ndarray,
                    config: Dict,
                    recorded_action: np.ndarray = None) -> np.ndarray:
    """
    Compute PRIMARY action target (robot executes this).

    Config:
        type: "recorded" | "next_state"
        derived_from: "ee_pose" | "joint_states"   (next_state only)

    type: "recorded" (preferred)
        Uses the command captured during teleoperation: the commanded tool0
        pose from /teleop/target_pose plus the commanded hand joints from the
        hand controller's joint_trajectory.

    type: "next_state"
        Legacy. action[t] = state[t+1], i.e. the action is inferred from where
        the robot ENDED UP. That is not the command: it contains the servo's
        scaling and singularity damping, the controller's compliance, and one
        to several ticks of tracking lag. A policy trained on it learns to
        reproduce the lag as if it were intent, and the executor then adds its
        own lag on top. Kept only so datasets recorded before arm_action.npy
        existed can still be rebuilt.

    Returns: (N, action_dim) float32
        recorded     → (N, 7 + hand_dim)
        ee_pose      → (N, 7 + hand_dim)
        joint_states → (N, 6 + hand_dim)
    """
    if config["type"] == "recorded":
        if recorded_action is None:
            raise ValueError(
                "actions.type='recorded' but this episode has no "
                "arm_action.npy/hand_action.npy. Re-extract it with "
                "record_extractor/extract_episodes.py, or fall back to "
                "type='next_state' (which bakes the controller's tracking "
                "lag into the action)."
            )

        # No _shift_next here. The recorded command at frame t IS the target
        # for frame t; shifting it would re-introduce exactly the one-frame
        # offset that next_state has to apply because its source is a
        # measurement.
        return recorded_action.astype(np.float32)

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