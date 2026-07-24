"""Dataset builder — generates datasets from raw episodes based on config."""

import argparse
import os
from termcolor import cprint

from config import load_config
from discovery import discover_and_validate
from pipeline import process_episode
from processing.proprioception import setup_kinematics
from writer_manager import WriterManager


HOME_DIR = "/ros2_ws/data"


def run(args):
    """Discover raw episodes → process → write dataset."""
    config = load_config(os.path.join(HOME_DIR, "configs", args.config))

    cprint(f"🔨 Building: {config.name}", "green", attrs=["bold"])

    # Kinematics
    kin_cfg = config.kinematics
    chain = setup_kinematics(urdf_path=kin_cfg["urdf_path"], end_link=kin_cfg["end_link"])
    joint_names = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
    ]

    # Discover raw episodes
    all_episodes = discover_and_validate(
        cache_dir=os.path.join(HOME_DIR, "episodes/external", config.cache["base_dir"]),
        cache_flags=config.cache_flags,
        require_success=config.episodes.get("require_success", False),
    )
    max_ep = config.episodes.get("max_episodes")
    if max_ep is not None:
        all_episodes = all_episodes[:max_ep]

    if not all_episodes:
        cprint("❌ No valid episodes found.", "red")
        return

    cprint(f"📂 Found {len(all_episodes)} episodes", "cyan")

    # Writer
    writer_mgr = WriterManager(config=config, home_dir=HOME_DIR, append=False)

    # Process
    for i, episode_dir in enumerate(all_episodes):
        cprint(f"[{i+1}/{len(all_episodes)}] {episode_dir.name}", "cyan")
        episode_data = process_episode(
            episode_dir=episode_dir, config=config, chain=chain, joint_names=joint_names,
        )
        writer_mgr.append_episode(episode_dir, episode_data)

    # Finalize
    writer_mgr.finalize(all_episodes)
    cprint(f"✅ Done! {len(all_episodes)} episodes written.", "green", attrs=["bold"])


def main():
    parser = argparse.ArgumentParser(description="Generate dataset from raw episodes.")
    parser.add_argument("--config", required=True, help="Config filename in configs/")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()