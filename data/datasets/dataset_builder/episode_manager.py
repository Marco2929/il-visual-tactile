# dataset_builder/episode_manager.py
"""Manage episode tracking – reads/writes source metadata in zarr & lerobot."""

import json
import os
import shutil
import numpy as np
import zarr
from pathlib import Path
from typing import List, Set, Optional, Dict
from termcolor import cprint


class EpisodeManager:
    """Reads/writes episode source metadata directly in dataset stores."""

    def __init__(self, config, home_dir: str):
        self.config = config
        self.home_dir = home_dir

        self.zarr_path = os.path.join(
            home_dir, "zarr_datasets", config.output_config["zarr_path"]
        )
        self.lerobot_path = os.path.join(
            home_dir, "lerobot_datasets", config.lerobot_name
        )

    # ──────────────────────────────────────────────
    # Existence checks
    # ──────────────────────────────────────────────
    def any_dataset_exists(self) -> bool:
        return os.path.exists(self.zarr_path) or os.path.exists(self.lerobot_path)

    def get_lerobot_info(self) -> Optional[Dict]:
        """Return parsed info.json or None if dataset doesn't exist."""
        info_path = os.path.join(self.lerobot_path, "meta", "info.json")
        if not os.path.exists(info_path):
            return None
        with open(info_path, "r") as f:
            return json.load(f)

    # ──────────────────────────────────────────────
    # Source reading
    # ──────────────────────────────────────────────
    def get_processed_episode_dirs(self, target: str) -> Set[str]:
        """Get set of episode directory names already processed for a target."""
        if target == "zarr":
            return self._read_sources_from_zarr()
        elif target == "lerobot":
            return self._read_sources_from_lerobot()
        return set()

    def _read_sources_from_zarr(self) -> Set[str]:
        if not os.path.exists(self.zarr_path):
            return set()
        try:
            root = zarr.open(self.zarr_path, mode="r")
            if "meta/episode_sources" in root:
                return set(root["meta/episode_sources"][:])
        except Exception as e:
            cprint(f"⚠️  Error reading zarr sources: {e}", "yellow")
        return set()

    def _read_sources_from_lerobot(self) -> Set[str]:
        meta_file = os.path.join(self.lerobot_path, "meta", "episodes_source.json")
        if not os.path.exists(meta_file):
            return set()
        with open(meta_file, "r") as f:
            data = json.load(f)
        return set(data.get("sources", []))

    def _read_pending_lerobot_sources(self) -> Set[str]:
        """Alias – pending sources are stored in the same file."""
        return self._read_sources_from_lerobot()

    # ──────────────────────────────────────────────
    # Source writing
    # ──────────────────────────────────────────────
    def log_processed_episodes(self, new_episodes: List[Path], zarr_ok: bool, lerobot_ok: bool):
        """Update source lists for successfully written formats."""
        new_sources = {ep.name for ep in new_episodes}
        fmt = self.config.output_config["format"]

        if fmt in ("zarr", "both") and zarr_ok:
            existing = self.get_processed_episode_dirs("zarr")
            self._write_sources_to_zarr(sorted(existing | new_sources))

        if fmt in ("lerobot", "both") and lerobot_ok:
            existing = self.get_processed_episode_dirs("lerobot")
            self._write_sources_to_lerobot(sorted(existing | new_sources))

    def _write_sources_to_zarr(self, sources: list):
        if not os.path.exists(self.zarr_path):
            return
        root = zarr.open(self.zarr_path, mode="a")
        meta = root.require_group("meta")
        source_array = np.array(sources, dtype=object)
        meta.create_dataset(
            "episode_sources",
            data=source_array,
            object_codec=(
                zarr.codecs.VLenUTF8Codec()
                if hasattr(zarr.codecs, 'VLenUTF8Codec') else None
            ),
            overwrite=True,
            dtype=str,
        )

    def _write_sources_to_lerobot(self, sources: list):
        meta_dir = os.path.join(self.lerobot_path, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        meta_file = os.path.join(meta_dir, "episodes_source.json")
        with open(meta_file, "w") as f:
            json.dump({"sources": sources, "count": len(sources)}, f, indent=2)

    # ──────────────────────────────────────────────
    # INFO
    # ──────────────────────────────────────────────
    def print_info(self):
        cprint(f"\n📦 Dataset: {self.config.name}", "cyan", attrs=["bold"])
        print("─" * 50)

        # Zarr info
        if os.path.exists(self.zarr_path):
            root = zarr.open(self.zarr_path, mode="r")
            ends = root["meta/episode_ends"][:] if "meta/episode_ends" in root else []
            n_frames = int(ends[-1]) if len(ends) > 0 else 0
            cprint(f"  Zarr:    ✅ {len(ends)} episodes, {n_frames} frames", "green")
        else:
            cprint(f"  Zarr:    ❌ not available", "red")

        # LeRobot info
        info = self.get_lerobot_info()
        if info:
            cprint(f"  LeRobot: ✅ {info.get('total_episodes', '?')} episodes, "
                   f"{info.get('total_frames', '?')} frames", "green")
        else:
            cprint(f"  LeRobot: ❌ not available", "red")

        # Sources
        print()
        for fmt in ("zarr", "lerobot"):
            sources = self.get_processed_episode_dirs(fmt)
            if sources:
                print(f"  {fmt.capitalize()} sources ({len(sources)}):")
                for s in sorted(sources)[:10]:
                    print(f"    • {s}")
                if len(sources) > 10:
                    print(f"    ... and {len(sources) - 10} more")
            else:
                print(f"  {fmt.capitalize()} sources: none")

    # ──────────────────────────────────────────────
    # DELETE
    # ──────────────────────────────────────────────
    def delete_episodes(self, episode_indices: List[int], target: str = "both"):
        deleted_any = False

        if target in ("zarr", "both"):
            deleted_any |= self._delete_from_zarr(episode_indices)
        if target in ("lerobot", "both"):
            deleted_any |= self._delete_from_lerobot(episode_indices)

        if deleted_any:
            cprint(f"\n✅ Episodes {episode_indices} deleted.", "green", attrs=["bold"])
        else:
            cprint(f"\n⚠️  Nothing deleted.", "yellow")

    def _delete_from_zarr(self, episode_indices: List[int]) -> bool:
        if not os.path.exists(self.zarr_path):
            cprint(f"⚠️  Zarr not found: {self.zarr_path}", "yellow")
            return False

        root = zarr.open(self.zarr_path, mode="r")
        episode_ends = np.array(root["meta/episode_ends"][:])
        n_episodes = len(episode_ends)

        invalid = [i for i in episode_indices if i >= n_episodes]
        if invalid:
            cprint(f"⚠️  Invalid indices (max={n_episodes-1}): {invalid}", "red")
            return False

        existing_sources = (
            list(root["meta/episode_sources"][:])
            if "meta/episode_sources" in root else []
        )

        # Build keep mask
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        keep_mask = np.ones(int(episode_ends[-1]), dtype=bool)
        indices_set = set(episode_indices)

        for idx in episode_indices:
            keep_mask[int(episode_starts[idx]):int(episode_ends[idx])] = False

        cprint(f"   Removing {np.sum(~keep_mask)} frames from {int(episode_ends[-1])}", "cyan")

        # Rebuild into temp
        temp_path = self.zarr_path + ".tmp"
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        new_root = zarr.open(temp_path, mode="w")
        self._copy_filtered_arrays(root, new_root, keep_mask)

        # New episode_ends
        new_ends = []
        cumulative = 0
        for idx in range(n_episodes):
            if idx in indices_set:
                continue
            n_frames = int(episode_ends[idx]) - int(episode_starts[idx])
            cumulative += n_frames
            new_ends.append(cumulative)

        meta = new_root.require_group("meta")
        meta.create_dataset("episode_ends", data=np.array(new_ends, dtype=np.int64), overwrite=True)

        # Filter sources
        if existing_sources:
            remaining = [s for i, s in enumerate(existing_sources) if i not in indices_set]
            meta.create_dataset(
                "episode_sources",
                data=np.array(remaining, dtype=object),
                object_codec=(
                    zarr.codecs.VLenUTF8Codec()
                    if hasattr(zarr.codecs, 'VLenUTF8Codec') else None
                ),
                overwrite=True,
                dtype=str,
            )

        # Swap
        shutil.rmtree(self.zarr_path)
        shutil.move(temp_path, self.zarr_path)
        cprint(f"   ✅ Zarr: {len(new_ends)} episodes remaining", "green")
        return True

    def _copy_filtered_arrays(self, src_root, dst_root, keep_mask: np.ndarray):
        """Copy all arrays from src to dst, applying keep_mask where dimensions match."""
        for key in src_root.keys():
            if key == "meta":
                continue

            item = src_root[key]
            if isinstance(item, zarr.hierarchy.Group):
                dst_group = dst_root.require_group(key)
                for arr_name in item.keys():
                    arr = item[arr_name]
                    data = arr[:]
                    filtered = data[keep_mask] if len(data) == len(keep_mask) else data
                    dst_group.create_dataset(
                        arr_name, data=filtered, chunks=arr.chunks, dtype=arr.dtype
                    )
            elif isinstance(item, zarr.core.Array):
                data = item[:]
                filtered = data[keep_mask] if len(data) == len(keep_mask) else data
                dst_root.create_dataset(
                    key, data=filtered, chunks=item.chunks, dtype=item.dtype
                )

    def _delete_from_lerobot(self, episode_indices: List[int]) -> bool:
        info = self.get_lerobot_info()
        if info is None:
            cprint(f"⚠️  LeRobot dataset not found.", "yellow")
            return False

        total_episodes = info.get("total_episodes", 0)
        invalid = [i for i in episode_indices if i >= total_episodes]
        if invalid:
            cprint(f"⚠️  Invalid indices (max={total_episodes-1}): {invalid}", "red")
            return False

        try:
            from lerobot.datasets import LeRobotDataset
            from lerobot.datasets.dataset_tools import delete_episodes

            dataset = LeRobotDataset(self.config.name, root=self.lerobot_path)
            delete_episodes(dataset, episode_indices, self.config.name)

            # Update sources
            sources = sorted(self.get_processed_episode_dirs("lerobot"))
            remaining = [s for i, s in enumerate(sources) if i not in set(episode_indices)]
            self._write_sources_to_lerobot(remaining)

            cprint(f"   ✅ LeRobot: {len(remaining)} episodes remaining", "green")
            return True

        except ImportError:
            cprint("⚠️  LeRobot API not available.", "red")
            return False
        except Exception as e:
            cprint(f"❌ LeRobot deletion failed: {e}", "red")
            return False