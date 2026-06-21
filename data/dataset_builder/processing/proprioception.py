# dataset_builder/processing/proprioception.py

import os
import numpy as np
import kinpy as kp
from typing import Dict, List
from termcolor import cprint


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


def compute_hand_state(joint_states: np.ndarray, hand_config: Dict) -> np.ndarray:
    """
    Extract hand joint positions.

    Returns: (N, num_hand_joints) float32
    """
    hand_joints = joint_states[:, 6:]
    num_hand = hand_joints.shape[1]
    hardware = hand_config["hardware"]
    expected = {"inspire": 6, "tesollo": 20}

    if hardware in expected and num_hand != expected[hardware]:
        cprint(
            f"Hand mismatch: expected {expected[hardware]} for '{hardware}', got {num_hand}",
            "yellow", attrs=["bold"]
        )

    return hand_joints.astype(np.float32)


def compute_force(force_raw: np.ndarray, force_config: Dict) -> np.ndarray:
    """
    Process force sensor data.

    Returns: (N, 6) if enabled, (N, 0) if disabled
    """
    if not force_config.get("enabled", False):
        return np.zeros((force_raw.shape[0], 0), dtype=np.float32)

    mode = force_config.get("mode", "raw")
    if mode == "raw":
        return force_raw.astype(np.float32)
    else:
        raise ValueError(f"Unknown force mode: '{mode}'")


def build_state_vector(ee_pose: np.ndarray,
                       hand_state: np.ndarray,
                       force: np.ndarray,
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
            components.append(force)

        else:
            raise ValueError(f"Unknown state_component: '{comp_type}'")

    return np.concatenate(components, axis=1).astype(np.float32)