# dataset_builder/writer_manager.py
"""Unified writer management – handles format routing and state tracking."""

import os
from pathlib import Path
from typing import List, Set, Optional
from termcolor import cprint

from writers.zarr_writer import ZarrWriter
from writers.lerobot_writer import LeRobotWriter
from episode_manager import EpisodeManager


class WriterManager:
    """Manages active writers, filters episodes, and finalizes output."""

    def __init__(self, config, home_dir: str, append: bool):
        self.config = config
        self.append = append
        self.output_fmt = config.output_config["format"]
        self.manager = EpisodeManager(config=config, home_dir=home_dir)

        # Instantiate only needed writers
        self.zarr_writer: Optional[ZarrWriter] = None
        self.lerobot_writer: Optional[LeRobotWriter] = None

        if self.output_fmt in ("zarr", "both"):
            self.zarr_writer = ZarrWriter(config=config, home_dir=home_dir, append=append)
        if self.output_fmt in ("lerobot", "both"):
            self.lerobot_writer = LeRobotWriter(config=config, append=append)

        # Per-writer processed sources
        self._processed = {}
        if self.zarr_writer:
            self._processed["zarr"] = self.manager.get_processed_episode_dirs("zarr")
        if self.lerobot_writer:
            self._processed["lerobot"] = self.manager.get_processed_episode_dirs("lerobot")

        # Frame tracking
        self._total_frames = self.zarr_writer.existing_frame_count if (append and self.zarr_writer) else 0
        self._episode_ends = list(self.zarr_writer.existing_episode_ends) if (append and self.zarr_writer) else []

    def filter_new_episodes(self, all_episodes: List[Path]) -> List[Path]:
        """Return episodes not yet processed by at least one active writer."""
        return [
            ep for ep in all_episodes
            if any(ep.name not in self._processed.get(fmt, set()) for fmt in self._processed)
        ]

    def append_episode(self, episode_dir: Path, episode_data: dict):
        """Append one episode to all active writers that need it."""
        n_frames = len(episode_data["state"])
        self._total_frames += n_frames
        self._episode_ends.append(self._total_frames)

        if self.zarr_writer and episode_dir.name not in self._processed.get("zarr", set()):
            self.zarr_writer.append_episode(episode_data=episode_data)

        if self.lerobot_writer and episode_dir.name not in self._processed.get("lerobot", set()):
            self.lerobot_writer.append_episode(episode_data=episode_data)

    def finalize(self, new_episodes: List[Path]):
        """Finalize writers and log sources."""
        zarr_ok = False
        lerobot_ok = False

        if self.zarr_writer:
            try:
                self.zarr_writer.finalize(episode_ends=self._episode_ends)
                zarr_ok = True
            except Exception as e:
                cprint(f"❌ Zarr finalize failed: {e}", "red")

        if self.lerobot_writer:
            try:
                self.lerobot_writer.finalize(push=False)
                lerobot_ok = True
            except Exception as e:
                cprint(f"❌ LeRobot finalize failed: {e}", "red")

        self.manager.log_processed_episodes(
            new_episodes=new_episodes,
            zarr_ok=zarr_ok,
            lerobot_ok=lerobot_ok,
        )

        self._print_summary(zarr_ok, lerobot_ok)

    def _print_summary(self, zarr_ok: bool, lerobot_ok: bool):
        results = {}
        if self.zarr_writer:
            results["Zarr"] = zarr_ok
        if self.lerobot_writer:
            results["LeRobot"] = lerobot_ok

        if all(results.values()):
            cprint(f"\n✅ Done: {len(self._episode_ends)} episodes, {self._total_frames} frames", "green", attrs=["bold"])
        elif any(results.values()):
            status = " | ".join(f"{k}: {'✅' if v else '❌'}" for k, v in results.items())
            cprint(f"\n⚠️  Partial: {status}", "yellow", attrs=["bold"])
            cprint("   Run 'append' again to retry the failed format.", "yellow")
        else:
            cprint(f"\n❌ All writers failed!", "red", attrs=["bold"])