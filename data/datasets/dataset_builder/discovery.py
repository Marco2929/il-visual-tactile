# dataset_builder/discovery.py
"""Episode discovery and validation."""

import os
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from termcolor import cprint

from metadata import load_episode_metadata, scene_key


@dataclass(frozen=True)
class EpisodeRef:
    """A raw episode plus the collection it came from.

    `key` is what gets persisted as the episode source, so it MUST be unique
    across all collections in a build. Episode folders are numbered per
    collection (every collection has its own ``episode_001_success``), so for
    multi-source builds the key is qualified with the collection directory.
    Single-source builds keep the bare folder name to stay compatible with
    datasets that were already written before multi-source existed.
    """
    path: Path
    source_dir: str
    position: Optional[str]
    key: str
    # Per-episode labels.json record (see metadata.py). compare=False keeps
    # EpisodeRef hashable despite holding a dict - the identity of an episode
    # is its key, not its metadata.
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def name(self) -> str:
        """Unique source identifier (used for dedup + provenance logging)."""
        return self.key

    @property
    def scene(self) -> str:
        """Scene label, from labels.json when present, else the YAML position.

        Used as the stratification key for the percentage subsampling in
        cli.py: a fraction has to be taken per scene, not off the front of a
        flat list, or it would silently drop whole scenes.
        """
        return scene_key(self.metadata,
                         fallback=self.position or self.source_dir)


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
            {"front_rgb": bool, "front_depth": bool, "wrist_rgb": bool,
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

    if cache_flags.get("front_depth", False):
        depth_dir = episode_dir / "front_depth"
        if not depth_dir.exists() or not depth_dir.is_dir():
            cprint(f"  Missing front_depth/ in {episode_dir.name}", "red")
            return False
        n_imgs = len(list(depth_dir.glob("*.png")))
        if n_imgs != n_frames:
            cprint(f"  Frame mismatch in {episode_dir.name}: "
                   f"state={n_frames}, front_depth={n_imgs}", "red")
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


def discover_sources(external_dir: str, sources: List[dict], cache_flags: dict,
                     require_success: bool = False) -> List[EpisodeRef]:
    """
    Discover and validate episodes across several raw collections.

    Args:
        external_dir: Directory containing the collection folders
        sources: Normalised source list (see DatasetConfig.cache_sources)
        cache_flags: Expected modalities
        require_success: Only include *_success episodes

    Returns:
        Flat list of validated EpisodeRefs, in source order
    """
    qualify = len(sources) > 1
    refs: List[EpisodeRef] = []

    for source in sources:
        source_dir = source["dir"]
        position = source.get("position")
        cache_dir = os.path.join(external_dir, source_dir)

        label = f"{source_dir}" + (f" [{position}]" if position else "")

        try:
            episodes = discover_and_validate(
                cache_dir=cache_dir,
                cache_flags=cache_flags,
                require_success=require_success,
            )
        except FileNotFoundError as e:
            # One missing/unextracted collection must not silently shrink the
            # dataset, so be loud about it but keep going.
            cprint(f"  ⚠️  {label}: {e}", "yellow")
            continue

        if not episodes:
            cprint(f"  ⚠️  {label}: no valid episodes", "yellow")
            continue

        n_labelled = 0
        for ep_dir in episodes:
            key = f"{source_dir}/{ep_dir.name}" if qualify else ep_dir.name
            # Read here rather than in process_episode: discovery is the only
            # stage that holds the raw folder for EVERY episode, including the
            # ones a later percentage/limit filter drops - and those filters
            # need the metadata to stratify by scene.
            ep_meta = load_episode_metadata(ep_dir)
            if ep_meta.get("has_labels"):
                n_labelled += 1
            refs.append(EpisodeRef(
                path=ep_dir,
                source_dir=source_dir,
                position=position,
                key=key,
                metadata=ep_meta,
            ))

        # Not a warning when absent: the older collections legitimately have
        # no labels.json, and they must keep building unchanged.
        suffix = (f" ({n_labelled} with labels.json)" if n_labelled
                  else " (no labels.json - legacy collection)")
        cprint(f"  ✓ {label}: {len(episodes)} episodes{suffix}", "cyan")

    return refs