# dataset_builder/processing/proprioception.py

import os
import numpy as np
import kinpy as kp
from typing import Dict, List
from termcolor import cprint

import sys
sys.path.insert(0, "/ros2_ws/third_party/ManiFlow_Policy/ManiFlow")

from cond_preprocess import PalmFrameTransform

def compute_hand_palm_features(
    joint_states: np.ndarray, 
    force_raw: np.ndarray, 
    transformer: PalmFrameTransform,
    config: Dict
) -> np.ndarray:
    """
    Transformiert Gelenk- und Kraftdaten der gesamten Episode in das Palm-Frame.
    
    Args:
        joint_states: (N, num_total_joints) - Enthält Arm- und Handgelenke
        force_raw: (N, 30) - Rohe F/T Sensordaten
        transformer: Instanz der PalmFrameTransform
        config: Konfiguration (z.B. um den Index-Offset für die Handgelenke zu definieren)
        
    Returns:
        (N, 65) Array mit den Palm-Features für jeden Frame.
    """
    if not config.get("enabled", True):
        return np.zeros((force_raw.shape[0], 0), dtype=np.float32)

    N = force_raw.shape[0]
    palm_features = np.zeros((N, 65), dtype=np.float32)
    
    # Extrahiere nur die 20 Handgelenke. 
    # Annahme: Arm hat 6 Gelenke, Hand beginnt ab Index 6.
    # TODO: Could be wrong 
    hand_joints = joint_states[:, 6 : 6 + 20]

    for i in range(N):
        palm_features[i] = transformer(hand_joints[i], force_raw[i]).reshape(-1)

    return palm_features


def setup_kinematics(urdf_path: str, end_link: str):
    """Build kinematic chain from URDF."""
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found at {urdf_path}")

    with open(urdf_path, 'r') as f:
        urdf_data = f.read()

    chain = kp.build_serial_chain_from_urdf(
        urdf_data,
        end_link_name=end_link,
        root_link_name="base_link"
    )
    return chain


def compute_ee_pose(joint_names: List[str],
                    joint_states: np.ndarray,
                    chain) -> np.ndarray:
    """
    Batch FK: joint angles → end-effector pose.

    Returns: (N, 7) float32 — [x, y, z, qx, qy, qz, qw]
    """
    arm_joints = joint_states[:, :6]
    N = arm_joints.shape[0]
    ee_poses = np.zeros((N, 7), dtype=np.float32)

    for i in range(N):
        th = dict(zip(joint_names, arm_joints[i]))
        ret = chain.forward_kinematics(th)

        pos = np.array(ret.pos)
        rot = np.array(ret.rot)  # kinpy: [w, x, y, z]
        rot_xyzw = np.array([rot[1], rot[2], rot[3], rot[0]])
        ee_poses[i] = np.concatenate([pos, rot_xyzw])

    return ee_poses


# Tesollo DG-5F hand constants
TESOLLO_HAND_DOF = 20
JOINTS_PER_FINGER = 4
PINCH_FINGERS = (1, 2)  # thumb + index: rj_dg_1_*, rj_dg_2_*
PINCH_HAND_INDICES = [(f - 1) * JOINTS_PER_FINGER + j
                      for f in PINCH_FINGERS
                      for j in range(JOINTS_PER_FINGER)]


def pinch_hand_indices() -> list:
    """Return the 8 hand-local joint indices for thumb + index pinch."""
    return list(PINCH_HAND_INDICES)


def tesollo_joint_name(finger: int, joint: int) -> str:
    """Return the canonical joint name for finger (1-5) and joint (1-4)."""
    return f"rj_dg_{finger}_{joint}"


def compute_hand_state(joint_states: np.ndarray, hand_config: Dict) -> np.ndarray:
    """
    Extract hand joint positions.

    Returns: (N, num_hand_joints) float32
    """
    hand_joints = joint_states[:, 6:]
    num_hand = hand_joints.shape[1]
    hardware = hand_config.get("hardware", "tesollo")
    expected = {"inspire": 6, "tesollo": 20}

    if hardware in expected and num_hand != expected[hardware]:
        cprint(
            f"Hand mismatch: expected {expected[hardware]} for '{hardware}', got {num_hand}",
            "yellow", attrs=["bold"]
        )

    if hand_config.get("pinch", False) and hardware == "tesollo":
        hand_joints = hand_joints[:, pinch_hand_indices()]

    return hand_joints.astype(np.float32)


def compute_force(
    force_raw: np.ndarray, 
    joint_states: np.ndarray, 
    force_config: Dict,
    transformer: PalmFrameTransform = None
) -> np.ndarray:
    """
    Process force sensor data based on the mode defined in the config.
    Returns: (N, D) array where D depends on the mode.
    """
    if not force_config.get("enabled", False):
        return np.zeros((force_raw.shape[0], 0), dtype=np.float32)

    mode = force_config.get("mode", "raw")
    
    if mode == "raw":
        return force_raw.astype(np.float32)
        
    elif mode == "hand_palm_features":
        if transformer is None:
            raise ValueError("PalmFrameTransform instance is required for 'hand_palm_features' mode.")
        return compute_hand_palm_features(
            joint_states=joint_states, 
            force_raw=force_raw, 
            transformer=transformer,
            config=force_config
        )
        
    else:
        raise ValueError(f"Unknown force mode: '{mode}'")


def build_state_vector(ee_pose: np.ndarray,
                       hand_state: np.ndarray,
                       joint_states: np.ndarray,
                       config: Dict) -> np.ndarray:
    """
    Build policy observation vector from state_components list.

    Returns: (N, state_dim) float32
    """
    components = []

    for comp in config["state_components"]:
        comp_type = comp["type"]

        if comp_type == "ee_pose":
            components.append(ee_pose)

        elif comp_type == "joint_states":
            joints = comp.get("joints", "arm")
            if joints == "arm":
                components.append(joint_states[:, :6].astype(np.float32))
            elif joints == "all":
                components.append(joint_states.astype(np.float32))
            elif isinstance(joints, list):
                components.append(joint_states[:, joints].astype(np.float32))

        elif comp_type == "hand":
            components.append(hand_state)
        elif comp_type == "force":
            pass
        else:
            raise ValueError(f"Unknown state_component: '{comp_type}'")

    return np.concatenate(components, axis=1).astype(np.float32)