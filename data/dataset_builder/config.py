# dataset_builder/config.py
"""Configuration loading and access."""

import yaml
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path


def _parse_fraction(value, where: str) -> float:
    """Validate a keep-fraction, with the config path in the error message.

    Bools are rejected explicitly: `fraction: true` is almost certainly a
    mistyped flag, and in Python it would otherwise pass as 1.0 and look
    like a deliberate "keep everything".
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{where} must be a number in [0, 1], got {value!r}"
        )
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{where} must be in [0, 1], got {value}")
    return float(value)


@dataclass
class DatasetConfig:
    """Typed access to the YAML configuration."""
    raw: Dict[str, Any]

    @property
    def name(self) -> str:
        return self.raw["meta"]["name"]

    @property
    def task(self) -> str:
        return self.raw["meta"]["task"]

    @property
    def task_id(self):
        """Numeric task id from `meta.task_id`, or None if not declared.

        `task` is a free-form instruction string meant for humans and
        language conditioning; this is the stable integer a multi-task
        policy embeds. They are deliberately separate: two datasets can
        share a task_id while phrasing the instruction differently, and the
        wording must be editable without silently repartitioning the data.

        None (field absent) is allowed so the older task configs keep
        building unchanged - see ZarrWriter for how that is recorded.
        """
        return self._meta_id("task_id")

    @property
    def robot_id(self):
        """Numeric robot/embodiment id from `meta.robot_id`, or None.

        Separate from task_id because the two vary independently: the same
        pinch-and-lift task recorded on a different arm or hand is the same
        task on another embodiment, and a cross-embodiment policy needs to
        tell those apart.
        """
        return self._meta_id("robot_id")

    def _meta_id(self, field: str):
        """Read and validate an optional integer id from the meta block.

        Validated at load time rather than at write time: these end up as
        int64 zarr arrays, so a string or a float here would either raise
        deep inside the writer after a full build, or - worse - be coerced
        and change meaning.
        """
        value = self.raw.get("meta", {}).get(field)
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"meta.{field} must be an integer, got "
                f"{type(value).__name__}: {value!r}")
        if value < 0:
            # -1 is the writer's "not declared" sentinel, so the valid
            # range has to exclude it.
            raise ValueError(
                f"meta.{field} must be >= 0 (-1 is reserved for "
                f"'unknown'), got {value}")
        return int(value)

    @property
    def meta_config(self) -> Dict:
        """The raw `meta:` block, for consumers that read optional fields.

        Used by LeRobotWriter for the Prehensio viewer's identity fields
        (`cell`, `pc`), which describe WHERE the data was recorded rather
        than what it contains. They have cell-wide defaults and are not
        worth a typed property each::

            meta:
              cell: bosch
              pc: monty5-pc
        """
        return self.raw.get("meta", {}) or {}

    @property
    def fps(self) -> int:
        return self.raw["meta"]["fps"]
    
    @property
    def resample_config(self) -> Dict:
        rs = self.raw.get("resample", {"enabled": False})
        # target_fps defaults to output fps to avoid a duplicated source of truth
        rs.setdefault("target_fps", self.fps)
        return rs
    
    @property
    def kinematics(self) -> Dict:
        return self.raw.get("kinematics", {})

    @property
    def cache(self) -> Dict:
        return self.raw.get("cache", {})

    @property
    def episodes(self) -> Dict:
        return self.raw.get("episodes", {})

    @property
    def subsample_config(self) -> Dict:
        """Per-collection episode fractions (see subsample.py).

        The fraction belongs to the SOURCE, so it is declared inline on the
        collection it caps::

            cache:
              base_dirs:
                - dir: foil_lifting_collection_tesollo_bottom
                  position: bottom
                  fraction: 0.35      # keep 35% of this collection
                - dir: folien_oben_a_saeule   # no fraction = keep all

            episodes:
              default_fraction: null  # for sources without one
              subsample_seed: 0

        Naming the collection a second time under `episodes:` (the earlier
        `source_fractions` mapping) is still accepted for configs that use
        it, but it is the worse form: the name has to stay in sync with
        `cache.base_dirs` by hand, which is exactly the kind of duplication
        that silently caps nothing after a directory is renamed.

        Validated here rather than in the sampler so a typo fails before a
        multi-hour build instead of after discovery.
        """
        episodes = self.raw.get("episodes", {}) or {}
        sources = self.cache_sources

        # Inline form first - it is the authority.
        parsed = {
            s["dir"]: s["fraction"]
            for s in sources
            if s.get("fraction") is not None
        }

        legacy = episodes.get("source_fractions") or {}
        if not isinstance(legacy, dict):
            raise ValueError(
                "episodes.source_fractions must be a mapping of "
                "collection dir -> fraction"
            )

        known = {s["dir"] for s in sources}
        for key, value in legacy.items():
            fraction = _parse_fraction(
                value, f"episodes.source_fractions[{key!r}]"
            )
            if key not in known:
                # A fraction for a collection that is not built is always a
                # mistake - usually a renamed directory - and silently doing
                # nothing would look like the cap was applied.
                raise ValueError(
                    f"episodes.source_fractions[{key!r}] does not match any "
                    f"cache.base_dirs entry. Known: {sorted(known)}"
                )
            if key in parsed and parsed[key] != fraction:
                # Two different answers to the same question: refuse rather
                # than pick one, because either choice would quietly build a
                # dataset the config does not describe.
                raise ValueError(
                    f"conflicting fractions for {key!r}: "
                    f"cache.base_dirs says {parsed[key]}, "
                    f"episodes.source_fractions says {fraction}. "
                    f"Keep the one on cache.base_dirs."
                )
            parsed[key] = fraction

        default = episodes.get("default_fraction")
        if default is not None:
            default = _parse_fraction(default, "episodes.default_fraction")

        return {
            "source_fractions": parsed,
            "default_fraction": default,
            "seed": int(episodes.get("subsample_seed", 0)),
        }

    @property
    def cache_dir(self) -> Path:
        return Path(self.raw["cache"]["base_dir"])

    @property
    def cache_sources(self) -> list:
        """Normalised list of raw-episode sources.

        Supports both the legacy single-directory form::

            cache:
              base_dir: "foil_lifting_collection"

        and the multi-source form, where each entry may carry a free-form
        label describing the recording condition::

            cache:
              base_dirs:
                - dir: foil_lifting_collection_tesollo_front
                  position: front
                - dir: foil_lifting_collection_tesollo_middle
                  position: middle

        A plain string entry in `base_dirs` is also accepted and yields a
        source without a position label.

        Returns:
            List of dicts with keys ``dir`` (str) and ``position``
            (str or None).
        """
        cache = self.raw.get("cache", {})
        raw_sources = cache.get("base_dirs")

        if raw_sources is None:
            base_dir = cache.get("base_dir")
            if not base_dir:
                raise ValueError(
                    "cache config must define either 'base_dir' or 'base_dirs'"
                )
            return [{"dir": base_dir, "position": None}]

        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]

        sources = []
        seen = set()
        for entry in raw_sources:
            if isinstance(entry, str):
                directory, position = entry, None
            elif isinstance(entry, dict):
                directory = entry.get("dir") or entry.get("base_dir")
                position = entry.get("position")
                if not directory:
                    raise ValueError(
                        f"cache.base_dirs entry missing 'dir': {entry!r}"
                    )
            else:
                raise ValueError(
                    f"cache.base_dirs entry must be a string or mapping, "
                    f"got {type(entry).__name__}: {entry!r}"
                )

            if directory in seen:
                raise ValueError(
                    f"duplicate cache.base_dirs entry: {directory!r}"
                )
            seen.add(directory)

            fraction = None
            if isinstance(entry, dict) and "fraction" in entry:
                fraction = _parse_fraction(
                    entry["fraction"],
                    f"cache.base_dirs[{directory!r}].fraction",
                )

            sources.append({
                "dir": directory,
                "position": position,
                "fraction": fraction,
            })

        if not sources:
            raise ValueError("cache.base_dirs is empty")

        return sources

    @property
    def cache_flags(self) -> Dict[str, bool]:
        cache = self.raw["cache"]
        return {
            "front_rgb": cache.get("front_rgb", False),
            "front_depth": cache.get("front_depth", False),
            "wrist_rgb": cache.get("wrist_rgb", False),
            "front_pc": cache.get("front_pc", False),
            "wrist_pc": cache.get("wrist_pc", False),
        }

    @property
    def pc_config(self) -> Dict:
        return self.raw["point_cloud"]

    @property
    def proprio_config(self) -> Dict:
        return self.raw["proprioception"]

    @property
    def hand_config(self) -> Dict:
        return self.proprio_config.get("hand", {})

    @property
    def action_config(self) -> Dict:
        return self.raw["actions"]

    @property
    def progress_config(self) -> Dict:
        return self.raw.get("progress", {"enabled": False})

    @property
    def trim_config(self) -> Dict:
        """Approach-trimming settings (see processing/trim.py).

        Defaults to DISABLED so every existing task config keeps producing
        byte-identical datasets. Enable per task::

            trim:
              enabled: true
              margin_m: 0.20   # cut this far above the lowest hand position
              min_frames: 24
              require_grasp: true

        margin_m is deliberately an UPPER bound: build generously and let
        the training dataloader sample the real start height inside it,
        rather than baking one crop into the dataset.
        """
        trim = self.raw.get("trim", {})
        trim.setdefault("enabled", False)
        trim.setdefault("margin_m", 0.20)
        trim.setdefault("min_frames", 24)
        trim.setdefault("require_grasp", True)
        return trim

    @property
    def post_processing_config(self) -> Dict:
        return self.raw.get("post_processing", {})

    @property
    def output_config(self) -> Dict:
        return self.raw["output"]

    @property
    def rgb_config(self) -> Dict:
        """RGB preprocessing settings (resize/crop before writing to disk).

        Override in the task YAML under a top-level `rgb:` block. Resolution
        must stay divisible by the encoder's downsample ratio (see
        processing/rgb.py for why).

        This is the WRIST default. The front camera has a different source
        aspect (16:9 vs 4:3), so it gets its own block via
        `rgb_config_for("front_rgb")` - sharing one resolution would force an
        anisotropic squash on one of the two.
        """
        rgb = self.raw.get("rgb", {})
        rgb.setdefault("enabled", True)
        # 192x256 keeps the native 4:3 FOV and is divisible by 32.
        rgb.setdefault("resolution", [192, 256])
        rgb.setdefault("center_crop", True)
        rgb.setdefault("interpolation", "area")
        rgb.setdefault("encoder_downsample_ratio", 32)
        return rgb

    def rgb_config_for(self, key: str) -> Dict:
        """Per-camera RGB config, falling back to the shared `rgb:` block.

        Different resolutions per key are legal because the encoder runs with
        share_rgb_model: false - each tower sees its own batch, so they need
        not agree on a shape.

        Override per camera in the task YAML::

            rgb:
              resolution: [192, 256]     # wrist default
              front_rgb:
                resolution: [192, 320]   # 16:9, half-HD720 source
        """
        base = dict(self.rgb_config)
        override = base.pop(key, None) or {}
        # Strip the other cameras' nested blocks so they cannot leak into the
        # flat config that preprocess_rgb() consumes.
        for other in ("front_rgb", "wrist_rgb"):
            base.pop(other, None)

        if key == "front_rgb":
            # 192x320 is 16:9 like the 360x640 source, so center_crop is a
            # no-op and nothing is distorted. Bound to the DINOv3 ViT-S/16
            # patch size: 192/16=12, 320/16=20 -> 12x20 = 240 tokens, exact.
            base["resolution"] = [192, 320]

        base.update(override)
        return base

    @property
    def depth_config(self) -> Dict:
        """Depth preprocessing settings.

        Separate from `rgb:` because the two share almost no parameters -
        notably there is no `interpolation` knob here: depth may only ever be
        resized with nearest-neighbour (see processing/depth.py).

        `depth_range` defaults to camera_info.FRONT_DEPTH_RANGE_M, resolved
        lazily in preprocess_depth() so the builder does not need the
        record_utils import path when depth is disabled.
        """
        depth = self.raw.get("depth", {})
        depth.setdefault("enabled", True)
        # Matches front_rgb so the two stay pixel-aligned after preprocessing.
        depth.setdefault("resolution", [192, 320])
        depth.setdefault("center_crop", True)
        depth.setdefault("encoder_downsample_ratio", 32)
        depth.setdefault("depth_range", None)
        return depth
    
    @property
    def force_config(self) -> Dict:
        return self.hand_config.get("force", {})

    @property
    def aux_action_config(self) -> list:
        return self.raw.get("auxiliary_actions", [])

def load_config(config_path: str) -> DatasetConfig:
    """Load YAML config from disk and return typed wrapper."""
    path = Path(config_path)
    if not path.exists() and path.with_suffix(".yaml").exists():
        path = path.with_suffix(".yaml")
    elif not path.exists() and path.with_suffix(".yml").exists():
        path = path.with_suffix(".yml")

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    cfg = DatasetConfig(raw=raw)

    # Set FOLDER_NAME and EPISODES_SUBDIR from config if not already set in the
    # environment, so camera_info and other tools align with the dataset being built.
    import os
    if "FOLDER_NAME" not in os.environ and cfg.cache_sources:
        first_folder = cfg.cache_sources[0]["dir"]
        os.environ["FOLDER_NAME"] = first_folder
        if "EPISODES_SUBDIR" not in os.environ:
            for base in ["/ros2_ws/data/episodes", str(path.resolve().parents[1] / "episodes")]:
                if os.path.exists(os.path.join(base, "external", first_folder)):
                    os.environ["EPISODES_SUBDIR"] = "external"
                    break
                elif os.path.exists(os.path.join(base, first_folder)):
                    os.environ["EPISODES_SUBDIR"] = ""
                    break

    return cfg