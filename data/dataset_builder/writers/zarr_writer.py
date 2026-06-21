# dataset_builder/writers/zarr_writer.py
"""Zarr dataset writer with append support."""

import os
import numpy as np
import zarr
from termcolor import cprint


class ZarrWriter:
    """Write dataset to Zarr format iteratively to avoid memory overflow."""

    def __init__(self, config, home_dir: str, append: bool = False):
        self.config = config
        self.home_dir = home_dir
        self.append = append

        self.zarr_path = os.path.join(
            home_dir, "zarr_datasets", config.output_config["zarr_path"]
        )

        self.existing_frame_count = 0
        self.existing_episode_ends = []

        if append and os.path.exists(self.zarr_path):
            self._load_existing_metadata()
            self.root = zarr.open(self.zarr_path, mode="a")
            cprint(f"📎 Zarr append mode: {self.existing_frame_count} existing frames", "cyan")
        else:
            # Create a fresh Zarr store
            os.makedirs(os.path.dirname(self.zarr_path), exist_ok=True)
            self.root = zarr.open(self.zarr_path, mode="w")

    def _load_existing_metadata(self):
        """Load frame count and episode ends from an existing dataset."""
        root = zarr.open(self.zarr_path, mode="r")
        if "meta/episode_ends" in root:
            self.existing_episode_ends = list(root["meta/episode_ends"][:])
            if self.existing_episode_ends:
                self.existing_frame_count = int(self.existing_episode_ends[-1])

    def append_episode(self, episode_data: dict):
        """Write episode data directly to Zarr arrays on disk. Ignores RGB data."""
        chunk_size = self.config.output_config.get("chunk_size", 100)

        for key, data in episode_data.items():
            # Filter out heavy RGB image arrays to save storage
            if "rgb" in key.lower() or key in ["image", "img"]:
                continue

            if key not in self.root:
                # Array does not exist yet (first episode in build mode)
                self.root.create_dataset(
                    key,
                    data=data,
                    chunks=(chunk_size,) + data.shape[1:],
                    dtype=data.dtype,
                    overwrite=True,
                )
            else:
                # Zarr's native append method writes directly to disk
                self.root[key].append(data)

    def finalize(self, episode_ends: list):
        """Write metadata to finish the dataset."""
        if not episode_ends:
            cprint("⚠️  No episodes to write to Zarr", "yellow")
            return

        # Write metadata
        meta = self.root.require_group("meta")
        meta.create_dataset(
            "episode_ends",
            data=np.array(episode_ends, dtype=np.int64),
            overwrite=True,
        )

        cprint(f"💾 Zarr saved: {self.zarr_path}", "green")
        cprint(f"   Episodes: {len(episode_ends)}, Frames: {episode_ends[-1]}", "green")