# dataset_builder/pipeline.py
"""Main pipeline: process_episode + orchestration."""

import numpy as np
from pathlib import Path
from typing import Dict, List
from termcolor import cprint

from config import DatasetConfig
from cache import (
    load_front_rgb, load_front_depth, load_wrist_rgb, load_joint_states,
    load_force, load_recorded_action, load_front_pc, load_wrist_pc,
    merge_point_clouds,
)
from processing.proprioception import (
    compute_ee_pose, compute_hand_state, compute_force,
    build_state_vector
)
from processing.actions import (
    compute_actions, compute_auxiliary_actions, reduce_recorded_action,
)
from processing.progress import compute_progress
from processing.trim import compute_trim, trim_metadata
from processing.post_processing import apply_post_processing
from processing.resampling import measure_fps, resample_episode
from processing.rgb import preprocess_rgb
from processing.depth import preprocess_depth

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
        Dict of named arrays ready for writing, or None if the episode was
        dropped by the trim (see processing/trim.py). The caller must handle
        None - see cli.py.

        With `trim.enabled: true` the dict carries a non-array entry under
        the key "_trim" with the per-episode provenance.
    """
    episode_data = {}
    flags = config.cache_flags

    palm_transformer = PalmFrameTransform(urdf_path=config.kinematics["urdf_path"])

    # --- 1. Load raw data from cache ---
    joint_states = load_joint_states(episode_dir)
    force_raw = load_force(episode_dir)
    num_frames = len(joint_states)

    # The teleoperator's actual command (arm + hand), or None on episodes
    # recorded before it was captured. Loaded at full rate and decimated with
    # the SAME keep_idx as everything else below - it is a per-frame signal
    # and must never be indexed independently.
    recorded_action = load_recorded_action(episode_dir, num_frames)
    if recorded_action is not None:
        # Reduced with the same index map compute_hand_state() uses for the
        # observation, so state and action always describe the same hand.
        recorded_action = reduce_recorded_action(
            recorded_action, config.hand_config)

    # Unwrap arm joints
    joint_states[:, :6] = np.unwrap(joint_states[:, :6], axis=0)

    # --- 1c. Optional downsampling BEFORE FK ---
    # Only active if explicitly enabled in config (resample.enabled: true)
    rs = config.resample_config
    keep_idx = None
    if rs.get("enabled", False):
        src_fps = measure_fps(episode_dir)
        joint_states, force_raw, keep_idx = resample_episode(
            joint_states, force_raw,
            target_fps=rs["target_fps"],
            src_fps=src_fps,
            state_smooth=rs.get("state_smoothing"),
            force_smooth=rs.get("force_smoothing"),
            enabled=True,
        )
        num_frames = len(joint_states)
        if recorded_action is not None:
            recorded_action = recorded_action[keep_idx]

    # --- 1d. Trim the approach away (optional) ---
    ee_pose = compute_ee_pose(joint_names, joint_states, chain)
    hand_state = compute_hand_state(joint_states, config.hand_config)
    resampled_frames = num_frames

    if config.trim_config.get("enabled"):
        trim_state = build_state_vector(
            ee_pose=ee_pose,
            hand_state=hand_state,
            joint_states=joint_states,
            config=config.proprio_config,
        )
        trim = compute_trim(trim_state, config.trim_config)
        if trim.skip:
            cprint(f"  ⊘ dropped: {trim.reason}", "yellow")
            return None

        if trim.start > 0 or trim.end < num_frames:
            sl = trim.slice
            joint_states = joint_states[sl]
            if force_raw is not None:
                force_raw = force_raw[sl]
            ee_pose = ee_pose[sl]
            hand_state = hand_state[sl]
            if recorded_action is not None:
                recorded_action = recorded_action[sl]
            if keep_idx is not None:
                keep_idx = keep_idx[sl]
            else:
                keep_idx = sl
            num_frames = len(joint_states)

    # --- 2. RGB (optional) ---
    if flags["front_rgb"]:
        front_rgb = load_front_rgb(episode_dir)
        if keep_idx is not None:
            front_rgb = front_rgb[keep_idx]
        episode_data["front_rgb"] = preprocess_rgb(
            front_rgb,
            config.rgb_config_for("front_rgb"),
        )
    if flags["front_depth"]:
        front_depth = load_front_depth(episode_dir)
        if keep_idx is not None:
            front_depth = front_depth[keep_idx]
        episode_data["front_depth"] = preprocess_depth(
            front_depth,
            config.depth_config,
        )
    if flags["wrist_rgb"]:
        wrist_rgb = load_wrist_rgb(episode_dir)
        if keep_idx is not None:
            wrist_rgb = wrist_rgb[keep_idx]
        episode_data["wrist_rgb"] = preprocess_rgb(
            wrist_rgb,
            config.rgb_config_for("wrist_rgb"),
        )

    # --- 3. Point Clouds ---
    front_pc = load_front_pc(episode_dir) if flags["front_pc"] else None
    wrist_pc = load_wrist_pc(episode_dir) if flags["wrist_pc"] else None
    if front_pc is not None and keep_idx is not None:
        front_pc = front_pc[keep_idx]
    if wrist_pc is not None and keep_idx is not None:
        wrist_pc = wrist_pc[keep_idx]

    point_cloud = merge_point_clouds(front_pc=front_pc,
                                     wrist_pc=wrist_pc,
                                     chain=chain,
                                     joint_names=joint_names,
                                     joint_states=joint_states,
                                     config=config.pc_config)
    # `mode: none` (wrist-only experiments) returns None. Omit the key
    # entirely rather than storing an empty array.
    if point_cloud is not None:
        episode_data["point_cloud"] = point_cloud

    # --- 3. Compute Representations ---
    # ee_pose and hand_state are already computed and trimmed above.
    force = compute_force(force_raw, joint_states, config.force_config, transformer=palm_transformer)
    if force is not None and force.size > 0 and all(d > 0 for d in force.shape):
        episode_data["hand_obs"] = force

    # --- 4. State (Policy Observation) ---
    # Rebuilt from the trimmed inputs rather than sliced from trim_state, so
    # that this stays correct if build_state_vector ever gains a component
    # that is not a plain per-frame concatenation.
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
        recorded_action=recorded_action,
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
            finger_sensor=hand_state,
            progress_dict=config.progress_config,
            joint_states=joint_states,
            hand_state=hand_state,
        )
        
        if progress_signal is None:
            raise ValueError("compute_progress returned None.")
            
        episode_data["progress"] = progress_signal

    # --- 8. Post-Processing ---
    episode_data = apply_post_processing(episode_data=episode_data, config=config.post_processing_config)

    # Provenance for the trim. Lets the training dataloader randomise the
    # start height within the margin that was built, instead of being stuck
    # with one fixed crop.
    # Set AFTER post-processing on purpose: _repeat_final_state iterates over
    # every value and would choke on a dict.
    if config.trim_config.get("enabled"):
        episode_data["_trim"] = trim_metadata(
            trim, orig_frames=resampled_frames)

    return episode_data