# dataset_builder/episode_manager.py
"""Manage episode tracking – reads truth from the dataset itself."""

import json
import os
import shutil
import numpy as np
import zarr
from pathlib import Path
from typing import List, Set, Optional
from termcolor import cprint


class EpisodeManager:
    """Reads/writes episode metadata directly in zarr & lerobot."""

    def __init__(self, config, home_dir: str):
        self.config = config
        self.home_dir = home_dir
        self.datasets_dir = os.path.join(home_dir)

        self.zarr_path = os.path.join(
            self.datasets_dir, "zarr_datasets", config.output_config["zarr_path"]
        )
        self.lerobot_path = os.path.join(
            self.datasets_dir, "lerobot_datasets", config.name
        )


    def get_processed_episode_dirs(self, target: str) -> Set[str]:
        if target == "zarr" and os.path.exists(self.zarr_path):
            return self._read_sources_from_zarr()

        elif target == "lerobot" and os.path.exists(self.lerobot_path):
            return self._read_sources_from_lerobot()

        return set()

    def _read_sources_from_zarr(self) -> Set[str]:

        try:
            root = zarr.open(self.zarr_path, mode="r")
            if "meta/episode_sources" in root:
                sources = root["meta/episode_sources"][:]
                return set(s for s in sources)
        except Exception as e:
            cprint(f"⚠️  Error while reading zarr dataset meta data: {e}", "yellow")
        return set()

    def _read_sources_from_lerobot(self) -> Set[str]:
        info_path = os.path.join(self.lerobot_path, "meta", "info.json")
        if not os.path.exists(info_path):

            return set()

        meta_file = os.path.join(self.lerobot_path, "meta", "episodes_source.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                data = json.load(f)
            return set(data.get("sources", []))
        return set()

    def log_processed_episodes(self, new_episodes: List[Path], zarr_ok: bool = True, lerobot_ok: bool = True):

        new_sources = {ep.name for ep in new_episodes}
        output_fmt = self.config.output_config["format"]

        written_count = 0

        if output_fmt in ("zarr", "both") and zarr_ok:
            existing_zarr = self.get_processed_episode_dirs("zarr")
            all_zarr = sorted(existing_zarr | new_sources)
            self._write_sources_to_zarr(all_zarr)
            cprint(f"📋 {len(all_zarr)} Episode-Soucres updated in Zarr", "green")
            written_count = len(all_zarr)

        if output_fmt in ("lerobot", "both") and lerobot_ok:
            existing_lerobot = self.get_processed_episode_dirs("lerobot")
            all_lerobot = sorted(existing_lerobot | new_sources)
            self._write_sources_to_lerobot(all_lerobot)
            cprint(f"📋 {len(all_lerobot)} Episoden-Sources updated in lerobot", "green")
            written_count = max(written_count, len(all_lerobot))

        if written_count > 0:
            cprint(f"📋 {written_count} Episoden-Sources saved in dataset", "green")
        elif not zarr_ok and not lerobot_ok:
            cprint("⚠️  No sources registed both datasets failed", "yellow")

    def _write_sources_to_zarr(self, sources: list):
        """Write the given source to zarr dataset"""
        if not os.path.exists(self.zarr_path):
            return

        root = zarr.open(self.zarr_path, mode="a")
        meta = root.require_group("meta")

        source_array = np.array(sources, dtype=object)
        meta.create_dataset(
            "episode_sources",
            data=source_array,
            object_codec=zarr.codecs.VLenUTF8Codec() if hasattr(zarr.codecs, 'VLenUTF8Codec')
            else None,
            overwrite=True,
            dtype=str,
        )

    def _write_sources_to_lerobot(self, sources: list):
        """Write the given source to lerobot dataset"""
        if not os.path.exists(self.lerobot_path):
            os.makedirs(self.lerobot_path, exist_ok=True)

        meta_dir = os.path.join(self.lerobot_path, "meta")
        os.makedirs(meta_dir, exist_ok=True)

        meta_file = os.path.join(meta_dir, "episodes_source.json")
        data = {
            "sources": sources,
            "count": len(sources),
        }
        with open(meta_file, "w") as f:
            json.dump(data, f, indent=2)

    # ──────────────────────────────────────────────
    # INFO
    # ──────────────────────────────────────────────

    def print_info(self):
        """ Print information about the dataset. """
        cprint(f"\n📦 Datensatz: {self.config.name}", "cyan", attrs=["bold"])
        print(f"{'─' * 50}")

        # Zarr
        if os.path.exists(self.zarr_path):
            root = zarr.open(self.zarr_path, mode="r")
            episode_ends = root["meta/episode_ends"][:] if "meta/episode_ends" in root else []
            n_eps = len(episode_ends)
            n_frames = int(episode_ends[-1]) if len(episode_ends) > 0 else 0
            cprint(f"  Zarr:    ✅ {n_eps} Episoden, {n_frames} Frames", "green")
            cprint(f"           {self.zarr_path}", "white")
        else:
            cprint(f"  Zarr:    ❌ not available", "red")

        # LeRobot
        lerobot_info_path = os.path.join(self.lerobot_path, "meta", "info.json")
        if os.path.exists(lerobot_info_path):
            with open(lerobot_info_path, "r") as f:
                info = json.load(f)
            n_eps = info.get("total_episodes", "?")
            n_frames = info.get("total_frames", "?")
            cprint(f"  LeRobot: ✅ {n_eps} Episoden, {n_frames} Frames", "green")
            cprint(f"           {self.lerobot_path}", "white")
        else:
            cprint(f"  LeRobot: ❌ not available", "red")

        # Sources (pro Format)
        output_fmt = self.config.output_config["format"]
        print()

        if output_fmt in ("zarr", "both"):
            zarr_sources = self.get_processed_episode_dirs("zarr")
            if zarr_sources:
                print(f"  Zarr-Sources ({len(zarr_sources)}):")
                for s in sorted(zarr_sources)[:10]:
                    print(f"    • {s}")
                if len(zarr_sources) > 10:
                    print(f"    ... und {len(zarr_sources) - 10} weitere")
            else:
                print(f"  Zarr-Sources: not registered")

        if output_fmt in ("lerobot", "both"):
            lerobot_sources = self.get_processed_episode_dirs("lerobot")
            if lerobot_sources:
                print(f"  LeRobot-Sources ({len(lerobot_sources)}):")
                for s in sorted(lerobot_sources)[:10]:
                    print(f"    • {s}")
                if len(lerobot_sources) > 10:
                    print(f"    ... und {len(lerobot_sources) - 10} weitere")
            else:
                print(f"  LeRobot-Sources: not registered")

        # Consistency check
        if output_fmt == "both":
            zarr_sources = self.get_processed_episode_dirs("zarr")
            lerobot_sources = self.get_processed_episode_dirs("lerobot")

            zarr_exists = os.path.exists(self.zarr_path)
            lerobot_exists = os.path.exists(lerobot_info_path)

            if zarr_exists and not lerobot_exists:
                cprint(f"\n  ⚠️  INCONSISTENCY: Zarr exists, but LeRobot is missing!", "yellow", attrs=["bold"])
                cprint(f"     → 'append' will rebuild LeRobot with {len(zarr_sources)} episodes", "yellow")
            elif not zarr_exists and lerobot_exists:
                cprint(f"\n  ⚠️  INCONSISTENCY: LeRobot exists, but Zarr is missing!", "yellow", attrs=["bold"])
                cprint(f"     → 'append' will rebuild Zarr with {len(lerobot_sources)} episodes", "yellow")
            elif zarr_sources != lerobot_sources:
                diff_zarr_only = zarr_sources - lerobot_sources
                diff_lerobot_only = lerobot_sources - zarr_sources
                cprint(f"\n  ⚠️  INCONSISTENCY detected!", "yellow", attrs=["bold"])
                if diff_zarr_only:
                    cprint(f"     Only in Zarr: {sorted(diff_zarr_only)}", "yellow")
                if diff_lerobot_only:
                    cprint(f"     Only in LeRobot: {sorted(diff_lerobot_only)}", "yellow")
                cprint(f"     → 'append' will automatically backfill the missing episodes", "yellow")
            else:
                cprint(f"\n  ✅ Zarr and LeRobot are consistent", "green")

    def _read_pending_lerobot_sources(self) -> Set[str]:   # ← HIER EINFÜGEN
        """Read the pending Lerobot sources from the meta file."""
        meta_file = os.path.join(self.lerobot_path, "meta", "episodes_source.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                data = json.load(f)
            return set(data.get("sources", []))
        return set()

    # ──────────────────────────────────────────────
    # DELETE
    # ──────────────────────────────────────────────

    def delete_episodes(self, episode_indices: List[int], target: str = "both"):
        """Delete episodes by index."""
        zarr_deleted = False
        lerobot_deleted = False

        if target in ("zarr", "both"):
            zarr_deleted = self._delete_from_zarr(episode_indices)
        if target in ("lerobot", "both"):
            lerobot_deleted = self._delete_from_lerobot(episode_indices)

        if zarr_deleted or lerobot_deleted:
            cprint(f"\n✅ Episoden {episode_indices} deleted.", "green", attrs=["bold"])
        else:
            cprint(f"\n⚠️  Nothing deleted (wrong indices or index missing).", "yellow")

    def _remove_sources_by_index(self, indices: List[int], update_zarr: bool = True, update_lerobot: bool = True):
        """Remove episode data from zarr and lerobot."""
        output_fmt = self.config.output_config["format"]
        indices_set = set(indices)

        if output_fmt in ("zarr", "both") and update_zarr:
            sources = sorted(self.get_processed_episode_dirs("zarr"))
            remaining = [s for i, s in enumerate(sources) if i not in indices_set]
            self._write_sources_to_zarr(remaining)
            cprint(f"📋 Zarr-Sources updated: {len(remaining)} remaining", "green")

        if output_fmt in ("lerobot", "both") and update_lerobot:
            sources = sorted(self.get_processed_episode_dirs("lerobot"))
            remaining = [s for i, s in enumerate(sources) if i not in indices_set]
            self._write_sources_to_lerobot(remaining)
            cprint(f"📋 LeRobot-Sources updated: {len(remaining)} remaining", "green")

    def _delete_from_zarr(self, episode_indices: List[int]) -> bool:
        """Delete episodes from zarr dataset and rebuild. Returns True if successful."""
        if not os.path.exists(self.zarr_path):
            cprint(f"⚠️  Zarr-Dataset not found: {self.zarr_path}", "yellow")
            return False

        cprint(f"🔧 Delete episodes from Zarr: {self.zarr_path}", "cyan")

        root = zarr.open(self.zarr_path, mode="r")

        # Get episode boundaries
        episode_ends = np.array(root["meta/episode_ends"][:])
        n_episodes = len(episode_ends)

        # Validate indices
        invalid = [i for i in episode_indices if i >= n_episodes]
        if invalid:
            cprint(f"⚠️  Wrong Episode-Indices (max={n_episodes-1}): {invalid}", "red")
            return False

        # Read existing sources BEFORE rebuilding
        existing_sources = []
        if "meta/episode_sources" in root:
            existing_sources = list(root["meta/episode_sources"][:])

        # Calculate frame ranges to keep
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        keep_mask = np.ones(int(episode_ends[-1]), dtype=bool)
        indices_set = set(episode_indices)

        for idx in episode_indices:
            start = int(episode_starts[idx])
            end = int(episode_ends[idx])
            keep_mask[start:end] = False

        cprint(f"   Remove {np.sum(~keep_mask)} Frames from {int(episode_ends[-1])} total", "cyan")

        # Rebuild zarr
        temp_path = self.zarr_path + ".tmp"
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        new_root = zarr.open(temp_path, mode="w")

        # Copy and filter all data arrays
        for key in root.keys():
            if key == "meta":
                continue

            group = root[key]
            if isinstance(group, zarr.hierarchy.Group):
                new_group = new_root.require_group(key)
                for array_name in group.keys():
                    data = group[array_name][:]
                    if len(data) == len(keep_mask):
                        new_group.create_dataset(
                            array_name,
                            data=data[keep_mask],
                            chunks=group[array_name].chunks,
                            dtype=group[array_name].dtype,
                        )
                    else:
                        new_group.create_dataset(
                            array_name, data=data,
                            chunks=group[array_name].chunks,
                            dtype=group[array_name].dtype,
                        )
            elif isinstance(group, zarr.core.Array):
                data = group[:]
                if len(data) == len(keep_mask):
                    new_root.create_dataset(
                        key, data=data[keep_mask],
                        chunks=group.chunks, dtype=group.dtype,
                    )
                else:
                    new_root.create_dataset(
                        key, data=data,
                        chunks=group.chunks, dtype=group.dtype,
                    )

        # Recalculate episode_ends
        new_episode_ends = []
        cumulative = 0
        for idx in range(n_episodes):
            if idx in indices_set:
                continue
            start = int(episode_starts[idx])
            end = int(episode_ends[idx])
            n_frames = end - start
            cumulative += n_frames
            new_episode_ends.append(cumulative)

        # Write meta group with episode_ends AND episode_sources
        meta_group = new_root.require_group("meta")
        meta_group.create_dataset(
            "episode_ends",
            data=np.array(new_episode_ends, dtype=np.int64),
            overwrite=True,
        )

        # Preserve and filter episode_sources
        if existing_sources:
            remaining_sources = [s for i, s in enumerate(existing_sources) if i not in indices_set]
            source_array = np.array(remaining_sources, dtype=object)
            meta_group.create_dataset(
                "episode_sources",
                data=source_array,
                object_codec=zarr.codecs.VLenUTF8Codec() if hasattr(zarr.codecs, 'VLenUTF8Codec')
                else None,
                overwrite=True,
                dtype=str,
            )

        # Replace old with new
        shutil.rmtree(self.zarr_path)
        shutil.move(temp_path, self.zarr_path)

        cprint(f"   ✅ Zarr: {n_episodes - len(episode_indices)} Episoden pending", "green")
        return True

    def _delete_from_lerobot(self, episode_indices: List[int]) -> bool:
        """Delete episodes from LeRobot dataset safely using the official API (no rebuild required)."""
        if not os.path.exists(self.lerobot_path):
            cprint(f"⚠️  LeRobot-Dataset not found: {self.lerobot_path}", "yellow")
            return False

        info_path = os.path.join(self.lerobot_path, "meta", "info.json")
        if not os.path.exists(info_path):
            cprint(f"   ⚠️  No LeRobot-Dataset available", "yellow")
            return False

        cprint(f"🔧 Delete episodes from LeRobot via API: {self.lerobot_path}", "cyan")

        with open(info_path, "r") as f:
            info = json.load(f)

        total_episodes = info.get("total_episodes", 0)
        invalid = [i for i in episode_indices if i >= total_episodes]
        if invalid:
            cprint(f"⚠️  Invalid Episode-Indices (max={total_episodes-1}): {invalid}", "red")
            return False

        try:
            from lerobot.datasets import LeRobotDataset
            from lerobot.datasets.dataset_tools import delete_episodes
            
            dataset = LeRobotDataset(self.config.name, root=self.lerobot_path)
            
            cprint(f"   Cut indices {episode_indices} from dataset...", "cyan")
            # In-place update: new_repo_id = self.config.name
            delete_episodes(dataset, episode_indices, self.config.name)
            
            cprint(f"   ✅ LeRobot updated (no rebuild needed on next append).", "green")
            return True
            
        except ImportError:
            cprint("⚠️  LeRobot API could not be loaded.", "red")
            return False
        except Exception as e:
            cprint(f"❌  Error during API deletion: {e}", "red")
            return False
        