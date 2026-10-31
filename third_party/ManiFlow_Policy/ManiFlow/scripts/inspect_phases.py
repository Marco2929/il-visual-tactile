"""Sanity-check the phase detector against the dataset.

Prints the phase distribution, grasp timing, and how many episodes contain
retries or a failed grasp - the cases fixed thirds get wrong.

    python scripts/inspect_phases.py --zarr /path/to/foil.zarr
"""

import argparse
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for cand in [_HERE, *_HERE.parents]:
    if (cand / 'maniflow' / 'policy').is_dir():
        sys.path.insert(0, str(cand))
        break

from maniflow.common.phase_detection import (  # noqa: E402
    PHASE_NAMES, detect_dataset_phases, grasp_signal, GRASP_THRESHOLD)
from maniflow.common.replay_buffer import ReplayBuffer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zarr', required=True)
    p.add_argument('--smooth', type=int, default=5)
    args = p.parse_args()

    rb = ReplayBuffer.copy_from_path(args.zarr, keys=['state'])
    state = np.asarray(rb['state'])
    ends = np.asarray(rb.episode_ends[:])
    starts = np.concatenate([[0], ends[:-1]])
    lengths = ends - starts

    phases, infos = detect_dataset_phases(state, ends, smooth=args.smooth)

    print(f"\n[data] {len(state)} frames, {len(ends)} episodes, "
          f"length {lengths.min()}-{lengths.max()} "
          f"(median {int(np.median(lengths))})")

    sig = grasp_signal(state)
    print(f"[grasp signal] mean(rj_dg_2_*) range "
          f"{sig.min():.1f}..{sig.max():.1f}, threshold {GRASP_THRESHOLD}")

    print("\n=== frame distribution per phase ===")
    for i, name in enumerate(PHASE_NAMES):
        n = int((phases == i).sum())
        print(f"  {i} {name:<9} {n:7d} frames  {100 * n / len(phases):5.1f}%")

    n_att = np.array([inf['n_attempts'] for inf in infos])
    grasped = np.array([inf['grasp_idx'] >= 0 for inf in infos])
    failed = np.array([inf['failed_grasp'] for inf in infos])
    held = np.array([inf['held_to_end'] for inf in infos])

    print("\n=== per episode ===")
    print(f"  grasp detected    : {grasped.sum():4d} / {len(infos)}")
    print(f"  no grasp at all   : {(~grasped).sum():4d}  "
          f"(these would be mislabelled by fixed thirds)")
    print(f"  >1 attempt        : {(n_att > 1).sum():4d}")
    print(f"  failed grasp flag : {failed.sum():4d}")
    print(f"  still closed @end : {held.sum():4d}")
    print(f"  attempts per ep   : "
          + ', '.join(f"{k}x:{int((n_att == k).sum())}"
                      for k in sorted(set(n_att.tolist()))))

    rel = np.array([inf['grasp_idx'] / max(l, 1)
                    for inf, l in zip(infos, lengths)
                    if inf['grasp_idx'] >= 0])
    if len(rel):
        print(f"\n=== grasp timing (fraction into the episode) ===")
        print(f"  min {rel.min():.2f}  p25 {np.percentile(rel, 25):.2f}  "
              f"median {np.median(rel):.2f}  "
              f"p75 {np.percentile(rel, 75):.2f}  max {rel.max():.2f}")
        print(f"  Fixed thirds assume the boundary sits at 0.33/0.67 for "
              f"every episode;")
        print(f"  the real grasp spans {rel.min():.2f}..{rel.max():.2f}, so "
              f"{100 * float(np.mean((rel < 0.2) | (rel > 0.8))):.0f}% of "
              f"episodes fall well outside that.")


if __name__ == '__main__':
    main()