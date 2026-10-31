#!/usr/bin/env python3
"""Delete aborted short episodes from a raw collection.

Dry run by default. Nothing is removed without --apply.

Why a script and not `rm -rf`: the frame count is re-read from state.npy at
deletion time and re-checked against the limit, so a typo in the episode list
cannot take out a full-length recording. The list and the measurement have to
agree or the episode is left alone.

The episode list is per collection and MUST be passed in: episode numbering
restarts in every collection, so a list that is right for one is wrong for
the next. `episode_024_success` is an aborted 13-frame recording in
`..._middle` but a full 195-frame one in `..._top_left`.

Usage:
    # report every episode below the limit, delete nothing
    python data/record_utils/delete_short_episodes.py <collection_dir>

    # verify a specific list, then delete it
    python data/record_utils/delete_short_episodes.py <collection_dir> \\
        --episodes 3 36
    python data/record_utils/delete_short_episodes.py <collection_dir> \\
        --episodes 3 36 --apply
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

# QA's min_episode_frames. An episode on the list must measure below this,
# otherwise the list belongs to a different collection and the script refuses.
# Raise it with --min-frames when a collection's aborted recordings sit just
# above the QA limit (top_right has one at 31 frames, with the next real
# episode at 135 - the cluster gap, not the limit, is what identifies them).
MIN_FRAMES = 30


def episode_frames(path: Path):
    state = path / "state.npy"
    if not state.exists():
        return None
    return int(len(np.load(state)))


def dir_size_mb(path: Path) -> float:
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("collection", type=Path)
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    metavar="N",
                    help="episode numbers to delete, scoped to THIS "
                         "collection. Omit to only report what is short.")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default: dry run)")
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    args = ap.parse_args()

    if args.apply and not args.episodes:
        sys.exit("--apply requires an explicit --episodes list; refusing to "
                 "delete everything below the limit automatically")

    expected = args.episodes or []

    root = args.collection
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    all_eps = sorted(p for p in root.iterdir()
                     if p.is_dir() and p.name.startswith("episode_"))
    by_index = {}
    for p in all_eps:
        try:
            by_index[int(p.name.split("_")[1])] = p
        except (IndexError, ValueError):
            continue

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"=== {mode} ===")
    print(f"collection : {root}")
    print(f"episodes   : {len(all_eps)}")
    print(f"min_frames : {args.min_frames}\n")

    to_delete, refused, missing = [], [], []
    for idx in expected:
        path = by_index.get(idx)
        if path is None:
            missing.append(idx)
            continue
        n = episode_frames(path)
        if n is None:
            refused.append((path, "no state.npy"))
        elif n >= args.min_frames:
            refused.append((path, f"{n} frames >= {args.min_frames}"))
        else:
            to_delete.append((path, n))

    # Anything short that is NOT on the list: report, never delete.
    extra = []
    for idx, path in sorted(by_index.items()):
        if idx in expected:
            continue
        n = episode_frames(path)
        if n is not None and n < args.min_frames:
            extra.append((path, n))

    # Show where the length distribution actually breaks, so --min-frames is
    # chosen from the data rather than guessed. An aborted recording is
    # separated from the real ones by a large gap, not by a round number.
    lengths = sorted((n, p.name) for p in all_eps
                     for n in [episode_frames(p)] if n is not None)
    if len(lengths) > 1:
        print("shortest episodes (gap = jump to the next one up):")
        for i, (n, nm) in enumerate(lengths[:6]):
            gap = (f"  +{lengths[i + 1][0] - n}"
                   if i + 1 < len(lengths) else "")
            mark = "  <= limit" if n < args.min_frames else ""
            print(f"  {nm:<24} {n:>5}{gap:>7}{mark}")
        print()

    print(f"{'episode':<24} {'frames':>7} {'size':>9}  action")
    print("-" * 60)
    total_mb = 0.0
    for path, n in to_delete:
        mb = dir_size_mb(path)
        total_mb += mb
        print(f"{path.name:<24} {n:>7} {mb:>8.1f}M  DELETE")
    for path, why in refused:
        print(f"{path.name:<24} {'-':>7} {'-':>9}  KEEP ({why})")
    print("-" * 60)
    print(f"to delete : {len(to_delete)} episodes, {total_mb:.1f} MB")
    print(f"remaining : {len(all_eps) - len(to_delete)} episodes")

    if missing:
        print(f"\nNOT FOUND (already gone?): {missing}")
    if refused:
        print("\nREFUSED - list disagrees with measurement, not deleting:")
        for path, why in refused:
            print(f"  {path.name}: {why}")
    if extra:
        print(f"\nNOTE: {len(extra)} further episode(s) below "
              f"{args.min_frames} frames are NOT on the list:")
        for path, n in extra:
            print(f"  {path.name}: {n} frames")

    if not args.apply:
        print("\nDry run - nothing deleted. Re-run with --apply.")
        return

    if refused:
        sys.exit("\nrefusing to delete while the list disagrees with the "
                 "measured frame counts; check --episodes against this "
                 "collection")

    print()
    for path, n in to_delete:
        shutil.rmtree(path)
        print(f"deleted {path.name} ({n} frames)")
    print(f"\ndone: {len(to_delete)} deleted, "
          f"{len(all_eps) - len(to_delete)} remaining")


if __name__ == "__main__":
    main()