# dataset_builder/cli.py
"""Command-line entry point with subcommands."""

import argparse
import json
import os
import sys
from pathlib import Path
from termcolor import cprint

from config import load_config
from discovery import discover_and_validate
from pipeline import process_episode
from processing.proprioception import setup_kinematics
from writers.zarr_writer import ZarrWriter
from writers.lerobot_writer import LeRobotWriter
from episode_manager import EpisodeManager


HOME_DIR = "/ros2_ws/data"


def parse_episode_list(value: str):
    """Parse '0,2,5-8' → [0, 2, 5, 6, 7, 8]"""
    indices = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    return sorted(indices)


# ═══════════════════════════════════════════════════
# BUILD command
# ═══════════════════════════════════════════════════
def cmd_build(args):
    """Build dataset from scratch (asks before overwriting)."""
    config = load_config(os.path.join(HOME_DIR, "configs", args.config))

    zarr_path = os.path.join(HOME_DIR, "zarr_datasets", config.output_config["zarr_path"])
    lerobot_path = os.path.join(HOME_DIR, "lerobot_datasets", config.name)

    if os.path.exists(zarr_path) or os.path.exists(lerobot_path):
        if not args.force:
            cprint(f"\n⚠️  Dataset '{config.name}' already exists!", "yellow", attrs=["bold"])
            response = input("Overwrite? (yes/no): ").strip().lower()
            if response not in ("yes", "y"):
                cprint("Stopped.", "yellow")
                return

    _run_pipeline(config, append=False)


# ═══════════════════════════════════════════════════
# APPEND command
# ═══════════════════════════════════════════════════
def cmd_append(args):
    """Append new episodes to existing dataset."""
    config = load_config(os.path.join(HOME_DIR, "configs", args.config))
    _run_pipeline(config, append=True)


# ═══════════════════════════════════════════════════
# DELETE command
# ═══════════════════════════════════════════════════
def cmd_delete(args):
    """Delete specific episodes from dataset."""
    config = load_config(os.path.join(HOME_DIR, "configs", args.config))
    episode_indices = parse_episode_list(args.episodes)
    target = args.target

    cprint(f"\n🗑️  Delete episodes: {episode_indices}", "red", attrs=["bold"])
    cprint(f"   Datasets: {config.name}", "red")
    cprint(f"   Format:    {target}", "red")

    if not args.force:
        response = input("Continue? (yes/no): ").strip().lower()
        if response not in ("yes", "y"):
            cprint("Stopped.", "yellow")
            return

    manager = EpisodeManager(config=config, home_dir=HOME_DIR)
    manager.delete_episodes(episode_indices=episode_indices, target=target)


# ═══════════════════════════════════════════════════
# INFO command
# ═══════════════════════════════════════════════════
def cmd_info(args):
    """Show info about existing dataset."""
    config = load_config(os.path.join(HOME_DIR, "configs", args.config))
    manager = EpisodeManager(config=config, home_dir=HOME_DIR)
    manager.print_info()


# ═══════════════════════════════════════════════════
# Integrity checks
# ═══════════════════════════════════════════════════
def _check_integrity(zarr_writer, lerobot_writer, processed_zarr: set, processed_lerobot: set, manager: EpisodeManager) -> tuple:
    repaired = False

    if zarr_writer and zarr_writer.existing_frame_count > 0 and len(processed_zarr) == 0:
        cprint(f"⚠️  Integrety problem: Zarr has {zarr_writer.existing_frame_count} Frames but 0 sources!", "yellow", attrs=["bold"])

        lerobot_sources = processed_lerobot or manager._read_pending_lerobot_sources()
        if lerobot_sources:
            cprint(f"   🔧 Repair: Taking {len(lerobot_sources)} Sources of LeRobot", "cyan")
            manager._write_sources_to_zarr(sorted(lerobot_sources))
            processed_zarr = lerobot_sources.copy()
            repaired = True
        else:
            cprint(f"   ❌ Unable to reconstruct source!", "red")
            raise RuntimeError("Zarr-Sources corrupt unable to repair. Stopped to avoid duplicates.")

    if lerobot_writer:
        lerobot_info_path = os.path.join(manager.lerobot_path, "meta", "info.json")
        lerobot_has_dataset = os.path.exists(lerobot_info_path)

        if not lerobot_has_dataset:

            pending_sources = manager._read_pending_lerobot_sources()
            if pending_sources:
                cprint(f"ℹ️ LeRobot: No dataset available, {len(pending_sources)} sources waiting for rebuild", "cyan")
                cprint(f"   → Dataset will be rebuilt with these episodes", "cyan")

            processed_lerobot = set()

        elif lerobot_has_dataset and len(processed_lerobot) == 0:
            cprint(f"⚠️  Integrity problem: LeRobot has Dataset but 0 Sources!", "yellow", attrs=["bold"])
            if zarr_writer and len(processed_zarr) > 0:
                cprint(f"   🔧 Repair: Takingo {len(processed_zarr)} Sources of Zarr", "cyan")
                manager._write_sources_to_lerobot(sorted(processed_zarr))
                processed_lerobot = processed_zarr.copy()
                repaired = True
            else:
                raise RuntimeError("LeRobot-Sources korrupt und nicht reparierbar.")

    if zarr_writer and zarr_writer.existing_frame_count == 0 and len(processed_zarr) > 0:
        cprint(f"ℹ️  Zarr: No data available, but {len(processed_zarr)} Sources registered", "cyan")
        cprint(f"   → Sources will be ignored, Dataset will not be build", "cyan")
        processed_zarr = set()

    if zarr_writer and zarr_writer.existing_frame_count > 0:
        zarr_ep_count = len(zarr_writer.existing_episode_ends)
        if zarr_ep_count != len(processed_zarr):
            cprint(f"⚠️  Zarr hat {zarr_ep_count} Episoden aber {len(processed_zarr)} Sources", "yellow")
            raise RuntimeError(
                f"Zarr Episode/Source Mismatch ({zarr_ep_count} vs {len(processed_zarr)}). "
                "Stopped to avoid duplicates use 'build --fource'."
            )

    if lerobot_writer:
        lerobot_info_path = os.path.join(manager.lerobot_path, "meta", "info.json")
        if os.path.exists(lerobot_info_path):
            with open(lerobot_info_path, "r") as f:
                info = json.load(f)
            lerobot_ep_count = info.get("total_episodes", 0)
            if lerobot_ep_count > 0 and lerobot_ep_count != len(processed_lerobot):
                cprint(f"⚠️  LeRobot hat {lerobot_ep_count} Episoden aber {len(processed_lerobot)} Sources", "yellow")
                raise RuntimeError(
                    f"LeRobot Episode/Source Mismatch ({lerobot_ep_count} vs {len(processed_lerobot)}). "
                     "Stopped to avoid duplicates use 'build --fource'."
                )

    if repaired:
        cprint(f"✅ Integrity repair finished", "green")

    return processed_zarr, processed_lerobot


# ═══════════════════════════════════════════════════
# Shared pipeline logic
# ═══════════════════════════════════════════════════
def _run_pipeline(config, append: bool):
    """Core pipeline with decoupled Zarr/LeRobot state tracking and error resilience."""
    cprint(f"{'📎 Append' if append else '🔨 Build'}: {config.name}", "green", attrs=["bold"])

    # --- Setup ---
    kin_cfg = config.kinematics
    chain = setup_kinematics(urdf_path=kin_cfg["urdf_path"], end_link=kin_cfg["end_link"])
    joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                   'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    # --- Discover ---
    all_episodes = discover_and_validate(
        cache_dir=os.path.join(HOME_DIR, "episodes", config.cache["base_dir"]),
        cache_flags=config.cache_flags,
        require_success=config.episodes.get("require_success", False),
    )

    max_ep = config.episodes.get("max_episodes")
    if max_ep is not None:
        all_episodes = all_episodes[:max_ep]

    # --- Writers & Manager ---
    output_fmt = config.output_config["format"]
    manager = EpisodeManager(config=config, home_dir=HOME_DIR)

    zarr_writer = ZarrWriter(config=config, home_dir=HOME_DIR, append=append) if output_fmt in ("zarr", "both") else None
    lerobot_writer = LeRobotWriter(config=config, append=append) if output_fmt in ("lerobot", "both") else None

    # --- Decoupled Filtering ---
    processed_zarr = manager.get_processed_episode_dirs("zarr") if zarr_writer else set()
    processed_lerobot = manager.get_processed_episode_dirs("lerobot") if lerobot_writer else set()

    # --- Integrity-Check only for append mode---
    if append:
        try:
            processed_zarr, processed_lerobot = _check_integrity(
                zarr_writer=zarr_writer,
                lerobot_writer=lerobot_writer,
                processed_zarr=processed_zarr,
                processed_lerobot=processed_lerobot,
                manager=manager,
            )
        except RuntimeError as e:
            cprint(f"\n❌ {e}", "red", attrs=["bold"])
            return

    # --- Pipeline Loop ---
    episodes_to_process = [
        ep for ep in all_episodes
        if (zarr_writer and ep.name not in processed_zarr) or
           (lerobot_writer and ep.name not in processed_lerobot)
    ]

    if not episodes_to_process:
        cprint("✅ All episodes already in target format", "green")
        return

    cprint(f"🚀 Processing {len(episodes_to_process)} new episodes", "cyan")


    episode_ends = zarr_writer.existing_episode_ends.copy() if (append and zarr_writer) else []
    total_frames = zarr_writer.existing_frame_count if (append and zarr_writer) else 0

    for ep_idx, episode_dir in enumerate(episodes_to_process):
        cprint(f"[{ep_idx+1}/{len(episodes_to_process)}] {episode_dir.name}", "cyan")

        episode_data = process_episode(
            episode_dir=episode_dir,
            config=config,
            chain=chain,
            joint_names=joint_names,
        )

        n_frames = len(episode_data["state"])
        total_frames += n_frames
        episode_ends.append(total_frames)

        if zarr_writer and episode_dir.name not in processed_zarr:
            zarr_writer.append_episode(episode_data=episode_data)

        if lerobot_writer and episode_dir.name not in processed_lerobot:
            lerobot_writer.append_episode(episode_data=episode_data)

    zarr_ok = False
    lerobot_ok = False

    if zarr_writer:
        try:
            zarr_writer.finalize(episode_ends=episode_ends)
            zarr_ok = True
        except Exception as e:
            cprint(f"❌ Zarr finalize failed: {e}", "red")

    if lerobot_writer:
        try:
            lerobot_writer.finalize(push=False)
            lerobot_ok = True
        except Exception as e:
            cprint(f"❌ LeRobot finalize failed: {e}", "red")


    manager.log_processed_episodes(
        new_episodes=episodes_to_process,
        zarr_ok=zarr_ok,
        lerobot_ok=lerobot_ok,
    )


    if zarr_ok and lerobot_ok:
        cprint(f"\n✅ Done: {len(episode_ends)} episodes, {total_frames} frames", "green", attrs=["bold"])
    elif zarr_ok or lerobot_ok:
        cprint(f"\n⚠️  Partially done – one writer failed!", "yellow", attrs=["bold"])
        cprint(f"   Zarr: {'✅' if zarr_ok else '❌'}  |  LeRobot: {'✅' if lerobot_ok else '❌'}", "yellow")
        cprint(f"   Fix the error and run 'append' again – only the missing part will be retried.", "yellow")
    else:
        cprint(f"\n❌ Both writers failed!", "red", attrs=["bold"])


# ═══════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Dataset Builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- build ---
    p_build = subparsers.add_parser("build", help="Build dataset from scratch")
    p_build.add_argument("--config", required=True)
    p_build.add_argument("--force", action="store_true", help="Skip confirmation")
    p_build.set_defaults(func=cmd_build)

    # --- append ---
    p_append = subparsers.add_parser("append", help="Append new episodes")
    p_append.add_argument("--config", required=True)
    p_append.set_defaults(func=cmd_append)

    # --- delete ---
    p_delete = subparsers.add_parser("delete", help="Delete episodes from dataset")
    p_delete.add_argument("--config", required=True)
    p_delete.add_argument("--episodes", required=True, help="e.g. '0,2,5-8'")
    p_delete.add_argument("--target", default="both", choices=["zarr", "lerobot", "both"])
    p_delete.add_argument("--force", action="store_true", help="Skip confirmation")
    p_delete.set_defaults(func=cmd_delete)

    # --- info ---
    p_info = subparsers.add_parser("info", help="Show dataset info")
    p_info.add_argument("--config", required=True)
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()