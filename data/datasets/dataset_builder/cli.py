"""Dataset builder — generates datasets from raw episodes based on config."""

import argparse
import os
import sys
from pathlib import Path
from termcolor import cprint

HOME_DIR = "/ros2_ws/data"

_this_dir = Path(__file__).resolve().parent
_data_dir = _this_dir.parents[1]
for _p in (_this_dir, _data_dir / "recording" / "record_utils", _data_dir / "record_utils", Path("/ros2_ws/data/recording/record_utils")):
    if str(_p) not in sys.path and _p.exists():
        sys.path.insert(0, str(_p))

# Pre-parse --config or --yaml early before importing pipeline or writers so that the config
# establishes the environment (e.g. FOLDER_NAME, EPISODES_SUBDIR) first.
def _preconfigure_environment():
    for flag in ("--config", "--yaml"):
        if flag in sys.argv:
            try:
                idx = sys.argv.index(flag)
                if idx + 1 < len(sys.argv):
                    cfg_arg = sys.argv[idx + 1]
                    cfg_path = Path(cfg_arg)
                    if not cfg_path.is_absolute():
                        for base in (
                            Path(HOME_DIR) / "datasets" / "configs",
                            Path(HOME_DIR) / "configs",
                            Path(__file__).resolve().parents[1] / "configs",
                        ):
                            for cand in (base / cfg_arg, (base / cfg_arg).with_suffix(".yaml"), (base / cfg_arg).with_suffix(".yml")):
                                if cand.exists():
                                    cfg_path = cand
                                    break
                            if cfg_path.exists():
                                break
                    if cfg_path.exists():
                        from config import load_config
                        load_config(str(cfg_path))
                        break
            except Exception:
                pass

_preconfigure_environment()

from config import load_config
from discovery import discover_sources
from pipeline import process_episode
from subsample import subsample_episodes
from processing.proprioception import setup_kinematics
from writer_manager import WriterManager


def run(args):
    """Discover raw episodes → process → write dataset."""
    cfg_candidates = [
        os.path.join(HOME_DIR, "datasets", "configs", args.config),
        str(Path(__file__).resolve().parents[1] / "configs" / args.config),
        os.path.join(HOME_DIR, "configs", args.config),
    ]
    cfg_path = next((p for p in cfg_candidates if os.path.exists(p)), cfg_candidates[0])
    config = load_config(cfg_path)

    cprint(f"🔨 Building: {config.name}", "green", attrs=["bold"])

    # Kinematics
    kin_cfg = config.kinematics
    chain = setup_kinematics(urdf_path=kin_cfg["urdf_path"], end_link=kin_cfg["end_link"])
    joint_names = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
    ]

    # Discover raw episodes across all configured collections
    sources = config.cache_sources
    cprint(f"🔎 Scanning {len(sources)} source(s)", "cyan")

    ext_candidates = [
        os.path.join(HOME_DIR, "recording", "episodes", "external"),
        os.path.join(HOME_DIR, "episodes", "external"),
    ]
    ext_dir = next((p for p in ext_candidates if os.path.exists(p)), ext_candidates[0])
    all_episodes = discover_sources(
        external_dir=ext_dir,
        sources=sources,
        cache_flags=config.cache_flags,
        require_success=config.episodes.get("require_success", False),
    )

    # Fractional subsampling runs FIRST, before any cap: the fractions are
    # defined relative to what was recorded, so applying them after a cap
    # would silently make them a fraction of a fraction.
    sub = config.subsample_config
    if sub["source_fractions"] or sub["default_fraction"] is not None:
        all_episodes = subsample_episodes(
            all_episodes,
            fractions=sub["source_fractions"],
            default_fraction=sub["default_fraction"],
            seed=sub["seed"],
        )

    # Per-source cap is applied before the global cap, so that a global cap
    # does not just truncate everything to the first collection.
    max_per_source = config.episodes.get("max_episodes_per_source")
    if max_per_source is not None:
        counts = {}
        capped = []
        for ref in all_episodes:
            n = counts.get(ref.source_dir, 0)
            if n >= max_per_source:
                continue
            counts[ref.source_dir] = n + 1
            capped.append(ref)
        all_episodes = capped

    max_ep = config.episodes.get("max_episodes")
    if max_ep is not None:
        all_episodes = all_episodes[:max_ep]

    if not all_episodes:
        cprint("❌ No valid episodes found.", "red")
        return

    # Grouped by scene, which for the new collections comes from labels.json
    # and for the old ones falls back to the YAML position - so one summary
    # describes both generations of data.
    by_scene = {}
    for ref in all_episodes:
        label = ref.scene
        by_scene[label] = by_scene.get(label, 0) + 1

    cprint(f"📂 Found {len(all_episodes)} episodes", "cyan")
    for label, count in sorted(by_scene.items()):
        cprint(f"     {label}: {count}", "cyan")

    # Writer
    writer_mgr = WriterManager(config=config, home_dir=HOME_DIR, append=False)

    # Process
    written, skipped = [], []
    for i, ref in enumerate(all_episodes):
        cprint(f"[{i+1}/{len(all_episodes)}] {ref.key}", "cyan")
        episode_data = process_episode(
            episode_dir=ref.path, config=config, chain=chain, joint_names=joint_names,
        )
        # None means the trim dropped the episode (no detectable grasp, or
        # too few frames left). Keeping it untrimmed would mix full
        # approaches into a dataset defined by not having them.
        if episode_data is None:
            skipped.append(ref)
            continue
        writer_mgr.append_episode(ref, episode_data)
        written.append(ref)

    if not written:
        cprint("❌ Every episode was skipped - nothing written.", "red")
        return

    # Finalize. Only the episodes actually written may be logged as sources,
    # otherwise the source list desynchronises from meta/episode_ends.
    writer_mgr.finalize(written)

    if skipped:
        cprint(f"⊘ Skipped {len(skipped)}/{len(all_episodes)} episodes:",
               "yellow")
        for ref in skipped[:15]:
            cprint(f"     {ref.key}", "yellow")
        if len(skipped) > 15:
            cprint(f"     ... and {len(skipped) - 15} more", "yellow")

    cprint(f"✅ Done! {len(written)} episodes written.", "green", attrs=["bold"])


def main():
    parser = argparse.ArgumentParser(description="Generate dataset from raw episodes.")
    parser.add_argument("--config", "--yaml", dest="config", required=True, help="Config filename in configs/")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()