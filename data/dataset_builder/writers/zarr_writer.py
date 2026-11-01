# dataset_builder/writers/zarr_writer.py
"""Zarr dataset writer with append support."""

import os
import numpy as np
import zarr
from termcolor import cprint

try:
    from numcodecs import Blosc
except ImportError:  # pragma: no cover
    Blosc = None

from metadata import build_zarr_columns


# Sentinel for episodes whose id the config did not declare. An explicit
# value beats omitting the array: a consumer can then tell "this build had
# no ids" from "this key is missing because the build is older", and the
# column stays index-aligned with episode_ends either way.
UNKNOWN_ID = -1


class ZarrWriter:
    """Write dataset to Zarr format iteratively to avoid memory overflow."""

    def __init__(self, config, home_dir: str, append: bool = False):
        self.config = config
        self.home_dir = home_dir
        self.append = append

        self.zarr_path = os.path.join(
            home_dir, "zarr_datasets", config.output_config["zarr_path"]
        )

        # Which image keys to persist. Historically ALL '*rgb*' keys were
        # silently dropped here, which meant wrist_rgb never reached the zarr
        # even though the builder config had `wrist_rgb: true`.
        # Now opt-in via `output.image_keys` (default: keep wrist_rgb).
        # NOTE: front_depth counts as an image key too (see _is_image_key), so
        # it must be listed here to be persisted.
        self.image_keys = set(
            config.output_config.get("image_keys", ["wrist_rgb"])
        )
        self.image_chunk_size = int(
            config.output_config.get("image_chunk_size", 8)
        )

        compression = config.output_config.get("compression", "zstd")
        self.compressor = None
        if compression and Blosc is not None:
            # BITSHUFFLE is a poor fit for uint8 image data; use plain SHUFFLE.
            self.compressor = Blosc(
                cname=compression, clevel=3, shuffle=Blosc.SHUFFLE
            )

        # Episode ids declared by the config. Captured once here so that
        # finalize() cannot be handed a different config than the build ran
        # with.
        self.task_id = config.task_id
        self.robot_id = config.robot_id

        self.existing_frame_count = 0
        self.existing_episode_ends = []
        # Per-episode ids already in the store, index-aligned with
        # existing_episode_ends. Must be carried over in append mode or the
        # arrays would come out shorter than episode_ends and the alignment
        # would be lost silently.
        self.existing_task_ids = []
        self.existing_robot_ids = []

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

        # Datasets built before these ids existed have no such arrays. Fill
        # with UNKNOWN_ID so the column stays index-aligned with
        # episode_ends instead of being short by the old episode count -
        # which is what would make a later per-task split quietly wrong.
        n_existing = len(self.existing_episode_ends)
        for field, attr in (("task_id", "existing_task_ids"),
                            ("robot_id", "existing_robot_ids")):
            path = f"meta/{field}"
            if path in root:
                values = [int(v) for v in root[path][:]]
                if len(values) != n_existing:
                    cprint(
                        f"⚠️  existing meta/{field} has {len(values)} "
                        f"entries for {n_existing} episodes; padding with "
                        f"{UNKNOWN_ID}",
                        "yellow",
                    )
                    values = (values + [UNKNOWN_ID] * n_existing)[:n_existing]
            else:
                values = [UNKNOWN_ID] * n_existing
            setattr(self, attr, values)

    def _is_image_key(self, key: str) -> bool:
        """Per-frame image-like streams, which need small time chunks.

        'depth' belongs here: a (192, 320, 2) float32 frame is ~0.5 MB, so the
        default chunk_size of 100 would allocate ~49 MB chunks and force the
        loader to decompress 100 frames to read one.
        """
        k = key.lower()
        return "rgb" in k or "depth" in k or key in ("image", "img")

    def append_episode(self, episode_data: dict):
        """Write episode data directly to Zarr arrays on disk.

        Image keys are only written when whitelisted in `output.image_keys`,
        and use a much smaller chunk size along time since a single frame is
        orders of magnitude larger than a state/action row.
        """
        chunk_size = self.config.output_config.get("chunk_size", 100)

        for key, data in episode_data.items():
            is_image = self._is_image_key(key)

            # Drop only the image streams we were not asked to keep.
            if is_image and key not in self.image_keys:
                continue

            key_chunk = self.image_chunk_size if is_image else chunk_size
            key_chunk = max(1, min(key_chunk, data.shape[0]))

            if key not in self.root:
                # Array does not exist yet (first episode in build mode)
                kwargs = dict(
                    data=data,
                    chunks=(key_chunk,) + data.shape[1:],
                    dtype=data.dtype,
                    overwrite=True,
                )
                try:
                    self.root.create_dataset(
                        key, compressor=self.compressor, **kwargs
                    )
                except TypeError:
                    # zarr v3 renamed `compressor` -> `compressors`; fall back
                    # to the store default rather than failing the build.
                    self.root.create_dataset(key, **kwargs)
            else:
                # Zarr's native append method writes directly to disk
                self.root[key].append(data)

    def _write_episode_ids(self, meta, n_episodes: int):
        """Write per-episode task_id / robot_id under meta/.

        Stored as one int64 value PER EPISODE rather than a single scalar
        attribute, even though one build writes one id. Two reasons:

          * append mode merges episodes from different configs into one
             store, so the id genuinely varies within a dataset and a scalar
             would describe only whichever build happened to run last;
          * it matches episode_ends and the trim_* arrays, so a dataloader
            indexes all of them the same way and needs no special case.
        """
        for field, existing, declared in (
            ("task_id", self.existing_task_ids, self.task_id),
            ("robot_id", self.existing_robot_ids, self.robot_id),
        ):
            value = UNKNOWN_ID if declared is None else int(declared)
            n_new = n_episodes - len(existing)
            if n_new < 0:
                # finalize() is given the full cumulative list, so this
                # means the store shrank - refuse rather than truncate
                # somebody's data.
                cprint(
                    f"⚠️  {n_episodes} episodes is fewer than the "
                    f"{len(existing)} already in the store; "
                    f"meta/{field} not written",
                    "yellow",
                )
                continue

            ids = list(existing) + [value] * n_new
            meta.create_dataset(
                field,
                data=np.array(ids, dtype=np.int64),
                overwrite=True,
            )

    def _log_episode_ids(self):
        """Report the ids, and say so loudly when they are missing.

        An undeclared id is not an error - the older configs do not set one
        - but it is silent in the output otherwise, and a multi-task build
        that forgot task_id produces a dataset that looks fine and cannot
        be split by task.
        """
        parts = []
        missing = []
        for label, value in (("task_id", self.task_id),
                             ("robot_id", self.robot_id)):
            if value is None:
                missing.append(label)
            else:
                parts.append(f"{label}={value}")

        if parts:
            cprint(f"   Episode ids: {', '.join(parts)}", "green")
        if missing:
            cprint(
                f"   ⚠️  {', '.join(missing)} not set in meta: written as "
                f"{UNKNOWN_ID}. Add it to the task YAML's meta block if "
                f"this dataset will be mixed with others.",
                "yellow",
            )

    def _write_episode_metadata(self, meta, episode_meta: list,
                                n_episodes: int):
        """Write labels.json-derived per-episode columns under meta/.

        Categorical fields become `<field>_id` int64 columns; the matching
        label vocabularies are stored in the meta group's attrs, so the
        dataset explains its own codes without the builder config.

        Episodes from the old collections carry no labels and simply code as
        UNKNOWN_ID, which is why this can run unconditionally.
        """
        if len(episode_meta) != n_episodes:
            # Same reasoning as the trim block: a shifted column would
            # attach one episode's scene to another, which is worse than
            # having no scene column at all.
            cprint(
                f"⚠️  episode metadata count {len(episode_meta)} != "
                f"{n_episodes} episodes - not written",
                "yellow",
            )
            return

        columns, vocabularies = build_zarr_columns(episode_meta)

        for field, values in columns.items():
            meta.create_dataset(
                field,
                data=np.array(values, dtype=np.int64),
                overwrite=True,
            )

        # Only non-empty vocabularies: an all-unknown field would otherwise
        # add an empty list that reads like a bug.
        for field, vocab in vocabularies.items():
            if vocab:
                meta.attrs[f"{field}_labels"] = list(vocab)

        n_labelled = sum(1 for m in episode_meta if m.get("has_labels"))
        if n_labelled:
            scenes = vocabularies.get("scene") or []
            cprint(
                f"   Metadata: {n_labelled}/{n_episodes} episodes labelled, "
                f"{len(scenes)} scene(s)",
                "green",
            )
        else:
            cprint(
                "   Metadata: no labels.json found - metadata columns "
                f"written as {UNKNOWN_ID}",
                "yellow",
            )

    def finalize(self, episode_ends: list, trim_meta: list = None,
                 episode_meta: list = None):
        """Write metadata to finish the dataset.

        Args:
            episode_ends: cumulative frame counts, one per episode.
            trim_meta: optional per-episode trim provenance (see
                processing/trim.trim_metadata), index-aligned with
                `episode_ends`. Stored as separate int arrays under meta/ so
                a dataloader can read them without a JSON sidecar.
            episode_meta: optional per-episode labels.json records (see
                metadata.load_episode_metadata), also index-aligned with
                `episode_ends`.
        """
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

        self._write_episode_ids(meta, len(episode_ends))

        if episode_meta:
            self._write_episode_metadata(meta, episode_meta, len(episode_ends))

        if trim_meta:
            if len(trim_meta) != len(episode_ends):
                # Misalignment would silently attach one episode's grasp
                # index to another, so refuse to write rather than guess.
                cprint(
                    f"⚠️  trim metadata count {len(trim_meta)} != "
                    f"{len(episode_ends)} episodes - not written",
                    "yellow",
                )
            else:
                for field in ("orig_frames", "trim_start", "n_frames",
                              "first_grasp", "n_attempts"):
                    meta.create_dataset(
                        f"trim_{field}",
                        data=np.array([m[field] for m in trim_meta],
                                      dtype=np.int64),
                        overwrite=True,
                    )
                kept = sum(m["n_frames"] for m in trim_meta)
                orig = sum(m["orig_frames"] for m in trim_meta)
                cprint(
                    f"✂️  trim: kept {kept}/{orig} frames "
                    f"({kept / max(orig, 1):.1%})",
                    "cyan",
                )

        cprint(f"💾 Zarr saved: {self.zarr_path}", "green")
        self._log_episode_ids()
        cprint(f"   Episodes: {len(episode_ends)}, Frames: {episode_ends[-1]}", "green")