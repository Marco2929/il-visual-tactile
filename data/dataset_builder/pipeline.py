# dataset_builder/pipeline.py
"""Main pipeline: process_episode + orchestration."""

import numpy as np
from pathlib import Path
from typing import Dict, List

from config import DatasetConfig
from cache import (
    load_front_rgb, load_wrist_rgb, load_joint_states, load_force,
    load_front_pc, load_wrist_pc, merge_point_clouds,
)
from processing.proprioception import (
    compute_ee_pose, compute_hand_state, compute_force,
    build_state_vector
)
from processing.actions import compute_actions, compute_auxiliary_actions
from processing.progress import compute_progress
from processing.post_processing import apply_post_processing
from processing.resampling import measure_fps, resample_episode

import sys
sys.path.insert(0, "/ros2_ws/third_party/ManiFlow_Policy/ManiFlow")

from cond_preprocess import PalmFrameTransform


def process_episode(episode_dir: Path, config: DatasetConfig,
                    chain, joint_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Full processing pipeline for a single episode.

    Args:
        episode_dir: Path to cached episode
        config: Dataset configuration
        chain: Kinematic chain (pre-loaded)
        joint_names: Names of joints

    Returns:
        Dict of named arrays ready for writing
    """
    episode_data = {}
    flags = config.cache_flags

    palm_transformer = PalmFrameTransform(urdf_path=config.kinematics["urdf_path"])

    # --- 1. Load raw data from cache ---
    joint_states = load_joint_states(episode_dir)
    force_raw = load_force(episode_dir)
    num_frames = len(joint_states)

    src_fps = measure_fps(episode_dir)

    # --- 1c. Smoothing + downsampling BEFORE FK ---
    rs = config.resample_config
    joint_states, force_raw, keep_idx = resample_episode(
        joint_states, force_raw,
        target_fps=rs["target_fps"],
        src_fps=src_fps,
        state_smooth=rs.get("state_smoothing", {}),
        force_smooth=rs.get("force_smoothing", {}),
        enabled=rs.get("enabled", True),
    )
    num_frames = len(joint_states)

    # --- 2. RGB (optional) ---
    if flags["front_rgb"]:
        episode_data["front_rgb"] = load_front_rgb(episode_dir)
    if flags["wrist_rgb"]:
        episode_data["wrist_rgb"] = load_wrist_rgb(episode_dir)

    # --- 3. Point Clouds ---
    front_pc = load_front_pc(episode_dir) if flags["front_pc"] else None
    wrist_pc = load_wrist_pc(episode_dir) if flags["wrist_pc"] else None
    episode_data["point_cloud"] = merge_point_clouds(front_pc=front_pc,
                                                     wrist_pc=wrist_pc,
                                                     chain=chain,
                                                     joint_names=joint_names,
                                                     joint_states=joint_states,
                                                     config=config.pc_config)

    # --- 3. Compute Representations ---
    ee_pose = compute_ee_pose(joint_names, joint_states, chain)
    hand_state = compute_hand_state(joint_states, config.hand_config)
    force = compute_force(force_raw, joint_states, config.force_config, transformer=palm_transformer)
    if force is not None and force.size > 0 and all(d > 0 for d in force.shape):
        episode_data["hand_obs"] = force

    # --- 4. State (Policy Observation) ---
    episode_data["state"] = build_state_vector(
        ee_pose=ee_pose,
        hand_state=hand_state,
        joint_states=joint_states,
        config=config.proprio_config,
    )

    # --- 5. Primary Action (Robot executes this) ---
    episode_data["action"] = compute_actions(
        ee_pose=ee_pose,
        hand_state=hand_state,
        joint_states=joint_states,
        config=config.action_config,
    )

    # --- 6. Auxiliary Actions (Only for training loss) ---
    aux_actions = compute_auxiliary_actions(
        ee_pose=ee_pose,
        joint_states=joint_states,
        force=force,
        config=config.aux_action_config,
    )
    for key, arr in aux_actions.items():
        episode_data[f"aux_action/{key}"] = arr

    # --- 7. Progress ---
    if config.progress_config.get("enabled"):
        progress_signal = compute_progress(
            num_frames=num_frames,
            ee_pose=ee_pose,
            finger_sensor=hand_state[:, 3]
        )
        
        if progress_signal is None:
            raise ValueError("compute_progress returned None.")
            
        episode_data["progress"] = progress_signal

    # --- 8. Post-Processing ---
    episode_data = apply_post_processing(episode_data=episode_data, config=config.post_processing_config)

    return episode_data