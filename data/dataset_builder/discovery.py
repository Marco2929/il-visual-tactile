# dataset_builder/discovery.py
"""Episode discovery and validation."""

import os
import numpy as np
from pathlib import Path
from typing import List
from termcolor import cprint


def discover_episodes(cache_dir: str, require_success: bool = False) -> List[Path]:
    """
    Find all valid episode directories in the cache.

    Args:
        cache_dir: Root directory containing episode folders
        require_success: If True, only include *_success episodes

    Returns:
        Sorted list of episode directory paths
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    # Find all episode_* directories
    all_dirs = sorted([
        d for d in cache_path.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    ])

    if not all_dirs:
        raise FileNotFoundError(f"No episode_* directories found in {cache_dir}")

    # Filter by success suffix if required
    if require_success:
        all_dirs = [d for d in all_dirs if d.name.endswith("_success")]

    return all_dirs


def validate_episode(episode_dir: Path, cache_flags: dict) -> bool:
    """
    Check that an episode directory has all required cached files.

    Args:
        episode_dir: Path to episode
        cache_flags: Which modalities are expected
            {"front_rgb": bool, "wrist_rgb": bool,
             "front_pc": bool, "wrist_pc": bool}

    Returns:
        True if episode is complete and usable
    """
    # State is always required
    state_path = episode_dir / "state.npy"
    if not state_path.exists():
        cprint(f"  Missing state.npy in {episode_dir.name}", "red")
        return False

    # Force is always required
    force_path = episode_dir / "force.npy"
    if not force_path.exists():
        cprint(f"  Missing force.npy in {episode_dir.name}", "red")
        return False

    # Get expected frame count from state
    state = np.load(state_path)
    n_frames = len(state)

    if n_frames == 0:
        cprint(f"  Empty state.npy in {episode_dir.name}", "red")
        return False

    # Check force frame count matches
    force = np.load(force_path)
    if len(force) != n_frames:
        cprint(f"  Frame mismatch in {episode_dir.name}: "
               f"state={n_frames}, force={len(force)}", "red")
        return False

    # Check RGB directories
    if cache_flags.get("front_rgb", False):
        rgb_dir = episode_dir / "front_rgb"
        if not rgb_dir.exists() or not rgb_dir.is_dir():
            cprint(f"  Missing front_rgb/ in {episode_dir.name}", "red")
            return False
        n_imgs = len(list(rgb_dir.glob("*.png")))
        if n_imgs != n_frames:
            cprint(f"  Frame mismatch in {episode_dir.name}: "
                   f"state={n_frames}, front_rgb={n_imgs}", "red")
            return False

    if cache_flags.get("wrist_rgb", False):
        rgb_dir = episode_dir / "wrist_rgb"
        if not rgb_dir.exists() or not rgb_dir.is_dir():
            cprint(f"  Missing wrist_rgb/ in {episode_dir.name}", "red")
            return False
        n_imgs = len(list(rgb_dir.glob("*.png")))
        if n_imgs != n_frames:
            cprint(f"  Frame mismatch in {episode_dir.name}: "
                   f"state={n_frames}, wrist_rgb={n_imgs}", "red")
            return False

    # Check point cloud directories
    if cache_flags.get("front_pc", False):
        pc_dir = episode_dir / "front_pc"
        if not pc_dir.exists() or not pc_dir.is_dir():
            cprint(f"  Missing front_pc/ directory in {episode_dir.name}", "red")
            return False
        n_pcs = len(list(pc_dir.glob("*.npy")))
        if n_pcs != n_frames:
            cprint(f"  Frame mismatch in {episode_dir.name}: "
                   f"state={n_frames}, front_pc={n_pcs}", "red")
            return False

    if cache_flags.get("wrist_pc", False):
        pc_dir = episode_dir / "wrist_pc"
        if not pc_dir.exists() or not pc_dir.is_dir():
            cprint(f"  Missing wrist_pc/ directory in {episode_dir.name}", "red")
            return False
        n_pcs = len(list(pc_dir.glob("*.npy")))
        if n_pcs != n_frames:
            cprint(f"  Frame mismatch in {episode_dir.name}: "
                   f"state={n_frames}, wrist_pc={n_pcs}", "red")
            return False

    return True


def discover_and_validate(cache_dir: str, cache_flags: dict,
                          require_success: bool = False) -> List[Path]:
    """
    Discover episodes and filter out invalid ones.

    Args:
        cache_dir: Root directory
        cache_flags: Expected modalities
        require_success: Only include *_success episodes

    Returns:
        List of validated episode paths
    """
    all_episodes = discover_episodes(cache_dir, require_success)

    valid = []
    skipped = 0

    for ep_dir in all_episodes:
        if validate_episode(ep_dir, cache_flags):
            valid.append(ep_dir)
        else:
            skipped += 1

    if skipped > 0:
        cprint(f"Skipped {skipped}/{len(all_episodes)} invalid episodes", "yellow")

    return valid