# dataset_builder/writers/lerobot_writer.py
"""LeRobot dataset writer with append support."""

import os
import json
import numpy as np
from pathlib import Path
from termcolor import cprint
from lerobot.datasets import LeRobotDataset
import torch
import shutil

from metadata import (
    build_prehensio_jsonl_row,
    build_prehensio_record,
)


class LeRobotWriter:
    """Write dataset in LeRobot format iteratively to avoid memory overflow."""

    def __init__(self, config, append: bool = False):
        self.config = config
        self.append = append

        self.repo_id = config.output_config["lerobot_repo_id"]
        self.fps = config.fps
        self.task = config.task

        self.existing_episodes = 0
        self.existing_frames = 0
        self.dataset = None

        # Rows for meta/prehensio/episodes.jsonl, collected during the build
        # and written once in finalize(). Buffered rather than appended
        # per-episode so an aborted build cannot leave the index file
        # describing episodes whose parquet/video never got written.
        self._prehensio_rows = []

        # Identity fields the Prehensio viewer shows but that are not part of
        # the recording itself. Overridable from the task YAML's meta block.
        meta_cfg = getattr(config, "meta_config", None) or {}
        self.cell = meta_cfg.get("cell", "bosch")
        self.recording_pc = meta_cfg.get("pc", "monty5-pc")

        # Determine local path and dataset name from lerobot_repo_id (or fallback to config.name)
        self.dataset_name = config.lerobot_name
        self.local_dir = os.path.join(
            "/ros2_ws/data", "lerobot_datasets", self.dataset_name
        )

        if not append and os.path.exists(self.local_dir):
            shutil.rmtree(self.local_dir)
            cprint(f"🧹 Deleted existing dataset: {self.local_dir}", "yellow")

        if append and os.path.exists(self.local_dir):
            self._load_existing_metadata()
            cprint(f"📎 LeRobot append mode: {self.existing_episodes} existing episodes", "cyan")
            self.dataset = LeRobotDataset.resume(
                repo_id=self.repo_id,
                root=self.local_dir
            )

    def _load_existing_metadata(self):
        """Load existing dataset metadata for append."""
        info_path = os.path.join(self.local_dir, "meta", "info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)
            self.existing_episodes = info.get("total_episodes", 0)
            self.existing_frames = info.get("total_frames", 0)

    def append_episode(self, episode_data: dict, episode_ref=None):
        """Write episode directly to LeRobot Dataset to save RAM.

        Args:
            episode_data: the per-frame streams for one episode.
            episode_ref: optional EpisodeRef. Carries `.metadata`
                (labels.json) and `.name`, which together make up the
                Prehensio sidecar. Optional so the writer stays usable
                standalone, in which case only the viewer's mandatory
                fields get written.
        """
        
        # Initialize the dataset on the very first call if not started in append mode
        if self.dataset is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lerobot", self.repo_id)
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                cprint(f"🧹 Deleted old cache: {cache_dir}", "yellow")

            features = self._build_features(episode_data)
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                features=features,
                robot_type="ur10e",
                root=self.local_dir
            )

        n_frames = len(episode_data["state"])

        for frame_idx in range(n_frames):
            # State-Tensor bauen: proprio + progress
            state = episode_data["state"][frame_idx].astype(np.float32)
            if "progress" in episode_data:
                progress = np.array([episode_data["progress"][frame_idx]], dtype=np.float32)
                state = np.concatenate([state, progress])

            frame = {
                "observation.state": torch.from_numpy(state),
                "action": torch.from_numpy(
                    episode_data["action"][frame_idx].astype(np.float32)
                ),
                "task": self.task,
            }

            if "front_rgb" in episode_data:
                frame["observation.images.front"] = torch.from_numpy(
                    episode_data["front_rgb"][frame_idx]
                ).permute(2, 0, 1)

            if "wrist_rgb" in episode_data:
                frame["observation.images.wrist"] = torch.from_numpy(
                    episode_data["wrist_rgb"][frame_idx]
                ).permute(2, 0, 1)

            self.dataset.add_frame(frame)

        # Trigger saving the episode to disk
        self.dataset.save_episode()
        
        # Episode index must be read BEFORE the counter is bumped: the
        # episode just written is the one at the current count.
        self._record_prehensio_meta(
            episode_ref=episode_ref,
            episode_index=self.existing_episodes,
            n_frames=n_frames,
        )

        # Update counters
        self.existing_episodes += 1
        self.existing_frames += n_frames
        
        cprint(f"   Episode {self.existing_episodes - 1}: {n_frames} frames", "cyan")

    def _record_prehensio_meta(self, episode_ref, episode_index: int,
                              n_frames: int):
        """Write one meta/prehensio/episode_XXXXXX.json and buffer its row.

        The viewer reads this directory, not LeRobot's own metadata, so the
        file is written per episode (cheap, and survives an interrupted
        build) while the index file is deferred to finalize().

        Failures here are logged and swallowed: the sidecar is for the
        viewer, and losing it must not destroy an otherwise complete
        dataset.
        """
        record = getattr(episode_ref, "metadata", None) or {}
        source = getattr(episode_ref, "name", "") or ""

        try:
            prehensio_dir = Path(self.local_dir) / "meta" / "prehensio"
            prehensio_dir.mkdir(parents=True, exist_ok=True)

            episode_json = build_prehensio_record(
                record,
                episode_index=episode_index,
                source=source,
                dataset=self.dataset_name,
                task=self.task,
                n_frames=n_frames,
                fps=self.fps,
                cell=self.cell,
                pc=self.recording_pc,
            )
            out_path = prehensio_dir / f"episode_{episode_index:06d}.json"
            with open(out_path, "w") as f:
                json.dump(episode_json, f, indent=2)

            self._prehensio_rows.append(build_prehensio_jsonl_row(
                record,
                episode_index=episode_index,
                task=self.task,
                n_frames=n_frames,
                fps=self.fps,
                cell=self.cell,
                pc=self.recording_pc,
            ))
        except OSError as e:
            cprint(f"   ⚠️  Prehensio metadata not written: {e}", "yellow")

    def _write_prehensio_index(self):
        """Write meta/prehensio/episodes.jsonl from the buffered rows.

        In append mode the existing rows are read back and kept, then only
        the indices this build produced are replaced. Rewriting the file from
        the buffer alone would drop every episode from previous builds, which
        is exactly what append mode exists to avoid.
        """
        if not self._prehensio_rows:
            return

        path = Path(self.local_dir) / "meta" / "prehensio" / "episodes.jsonl"
        rows_by_index = {}

        if self.append and path.exists():
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        rows_by_index[row.get("episode_index")] = row
            except (OSError, json.JSONDecodeError) as e:
                cprint(
                    f"   ⚠️  Could not read existing episodes.jsonl "
                    f"({e}); writing only this build's rows",
                    "yellow",
                )
                rows_by_index = {}

        for row in self._prehensio_rows:
            rows_by_index[row["episode_index"]] = row

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                for index in sorted(
                    k for k in rows_by_index if k is not None
                ):
                    f.write(json.dumps(rows_by_index[index]) + "\n")
            cprint(
                f"   Prehensio metadata: {len(rows_by_index)} episodes",
                "green",
            )
        except OSError as e:
            cprint(f"   ⚠️  episodes.jsonl not written: {e}", "yellow")

    def finalize(self, push: bool = False):
        """Finalize the LeRobot dataset (push if requested)."""
        if self.dataset is None:
            cprint("⚠️  No episodes to write to LeRobot", "yellow")
            return

        # Flush LeRobot dataset writer and write parquet footers
        if hasattr(self.dataset, "finalize"):
            self.dataset.finalize()

        self._write_prehensio_index()

        if push:
            self.dataset.push_to_hub()
            cprint(f"🚀 Pushed to HuggingFace: {self.repo_id}", "green")

        cprint(f"💾 LeRobot gespeichert: {self.local_dir}", "green")
        cprint(f"   Episodes: {self.existing_episodes}, Frames: {self.existing_frames}", "green")

    def _build_features(self, sample_episode: dict) -> dict:
        """Build feature dict dynamically based on the config."""

        dim_names = []

        proprio_cfg = self.config.proprio_config
        components = proprio_cfg.get("state_components", [])

        state_dim = sample_episode["state"].shape[-1]
        action_dim = sample_episode["action"].shape[-1]

        for comp in components:
            c_type = comp.get("type")

            if c_type == "ee_pose":
                dim_names.extend(["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"])
            elif c_type == "joint_states":
                dim_names.extend([
                    "shoulder_pan", "shoulder_lift", "elbow",
                    "wrist_1", "wrist_2", "wrist_3"
                ])
            elif c_type == "hand":
                hand_hw = proprio_cfg.get("hand", {}).get("hardware", "inspire")
                if hand_hw == "inspire":
                    dim_names.extend([
                        "thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "little_1"
                    ])
                elif hand_hw == "tesollo":
                    pinch = proprio_cfg.get("hand", {}).get("pinch", False)
                    n_joints = 8 if pinch else 20
                    dim_names.extend([f"tesollo_joint_{i}" for i in range(n_joints)])
            elif c_type == "force":
                force_enabled = proprio_cfg.get("hand", {}).get("force", {}).get("enabled", False)
                if force_enabled:
                    dim_names.extend(["thumb_1_force", "thumb_2_force", "index_1_force",
                                    "middle_1_force", "ring_1_force", "little_1_force"])

        if len(dim_names) != state_dim:
            cprint(f"⚠️ Name mismatch: state has {state_dim} dims, but {len(dim_names)} names. Using generic.", "yellow")
            dim_names = [f"dim_{i}" for i in range(state_dim)]

        # Action names (ohne progress)
        if action_dim == len(dim_names):
            action_names = dim_names
        else:
            action_names = [f"action_dim_{i}" for i in range(action_dim)]

        # State = proprio + progress als letzte Dimension
        has_progress = "progress" in sample_episode
        state_names = dim_names + (["progress"] if has_progress else [])
        state_shape = state_dim + (1 if has_progress else 0)

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_shape,),
                "names": state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": action_names,
            },
        }

        if "front_rgb" in sample_episode:
            h, w, c = sample_episode["front_rgb"].shape[1:]
            features["observation.images.front"] = {
                "dtype": "video",
                "shape": (c, h, w),
                "names": ["channels", "height", "width"],
            }

        if "wrist_rgb" in sample_episode:
            h, w, c = sample_episode["wrist_rgb"].shape[1:]
            features["observation.images.wrist"] = {
                "dtype": "video",
                "shape": (c, h, w),
                "names": ["channels", "height", "width"],
            }

        return features