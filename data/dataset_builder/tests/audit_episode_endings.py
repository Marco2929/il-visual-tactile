#!/usr/bin/env python3
"""Measure what the demonstrations do AFTER the grasp closes.

Tests two hypotheses about a policy that closes timidly and lifts too far,
both of which would be learned behaviour rather than a control fault:

  H1 "I already opened the hand at the end of my demos."
     -> closure at the LAST frame is well below the episode's peak closure.
        The policy would then be reproducing a release it was shown.

  H2 "It tends to drive up too far."
     -> the end effector keeps rising for a long time after peak closure.
        The action horizon is 64 steps, so whatever happens in the last ~3 s
        of every episode is inside the chunk the policy emits at grasp time.

Finger 2 (index) is the grasping finger, state dims 11-14; finger 1 is the
opposing thumb and barely moves. Closing INCREASES the angle. Established by
measurement in ANALYSIS_TODO.md, not assumed here.

Usage:
    python data/dataset_builder/tests/audit_episode_endings.py [zarr]
"""
import sys

import numpy as np
import zarr

DEFAULT_ZARR = "/ros2_ws/data/zarr_datasets/foil_lifting_wrist_tesollo.zarr"

EE_Z = 2
GRASP = slice(11, 15)   # rj_dg_2_1 .. rj_dg_2_4, degrees
FPS = 20.0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ZARR
    z = zarr.open(path, "r")
    ends = z["meta"]["episode_ends"][:]
    src = [str(s) for s in z["meta"]["episode_sources"][:]]
    starts = np.concatenate([[0], ends])
    state = z["state"]

    rows = []
    for i in range(len(ends)):
        a, b = int(starts[i]), int(starts[i + 1])
        x = state[a:b]
        n = b - a
        closure = x[:, GRASP].sum(axis=1)
        k = int(np.argmax(closure))

        rows.append({
            "i": i, "n": n, "src": src[i],
            "peak": float(closure[k]),
            "end": float(closure[-1]),
            # Positive = the hand opened again before the episode ended.
            "reopen": float(closure[k] - closure[-1]),
            "rel_peak": k / max(n - 1, 1),
            "frames_after": n - 1 - k,
            "z_peak": float(x[k, EE_Z]),
            "z_end": float(x[-1, EE_Z]),
            "z_rise": float(x[-1, EE_Z] - x[k, EE_Z]),
            "z_max_after": float(x[k:, EE_Z].max() - x[k, EE_Z]),
        })

    def col(key):
        return np.array([r[key] for r in rows], dtype=float)

    n_ep = len(rows)
    print(f"zarr     : {path}")
    print(f"episodes : {n_ep}")

    peak, end, reopen = col("peak"), col("end"), col("reopen")
    print("\n=== H1: does the hand open again before the episode ends? ===")
    print(f"peak closure (sum of 4 joints) : median {np.median(peak):7.2f} deg"
          f"   p10 {np.percentile(peak, 10):6.2f}"
          f"   p90 {np.percentile(peak, 90):6.2f}")
    print(f"closure at last frame          : median {np.median(end):7.2f} deg"
          f"   p10 {np.percentile(end, 10):6.2f}"
          f"   p90 {np.percentile(end, 90):6.2f}")
    print(f"reopening (peak - end)         : median {np.median(reopen):7.2f} deg"
          f"   p10 {np.percentile(reopen, 10):6.2f}"
          f"   p90 {np.percentile(reopen, 90):6.2f}")
    frac_end = end / np.maximum(peak, 1e-6)
    print(f"end/peak ratio                 : median {np.median(frac_end):7.3f}")
    for thr in (5.0, 10.0, 20.0):
        k = int((reopen > thr).sum())
        print(f"  episodes reopening > {thr:4.0f} deg : {k:3d} / {n_ep} "
              f"({100*k/n_ep:.0f}%)")

    rel, after = col("rel_peak"), col("frames_after")
    print("\n=== when does peak closure happen? ===")
    print(f"relative position in episode   : median {np.median(rel):7.3f}"
          f"   p10 {np.percentile(rel, 10):6.3f}"
          f"   p90 {np.percentile(rel, 90):6.3f}")
    print(f"frames after peak              : median {np.median(after):7.1f}"
          f"  ({np.median(after)/FPS:.1f}s)"
          f"   p90 {np.percentile(after, 90):6.1f}")

    zr, zma = col("z_rise"), col("z_max_after")
    print("\n=== H2: does the arm keep rising after the grasp? ===")
    print(f"z(end) - z(peak closure)       : median {np.median(zr):7.4f} m"
          f"   p10 {np.percentile(zr, 10):7.4f}"
          f"   p90 {np.percentile(zr, 90):7.4f}")
    print(f"max z reached after peak        : median {np.median(zma):7.4f} m"
          f"   p90 {np.percentile(zma, 90):7.4f}")
    for thr in (0.05, 0.10, 0.20):
        k = int((zma > thr).sum())
        print(f"  episodes rising > {thr*100:3.0f} cm after grasp : {k:3d} / "
              f"{n_ep} ({100*k/n_ep:.0f}%)")

    print("\n=== per collection ===")
    print(f"{'collection':<28} {'n':>4} {'peak':>7} {'end':>7} {'reopen':>7} "
          f"{'z_rise':>8} {'rel_pk':>7}")
    print("-" * 76)
    cols = {}
    for r in rows:
        cols.setdefault(r["src"].split("/")[0], []).append(r)
    for name in sorted(cols):
        g = cols[name]
        f = lambda k: np.median([r[k] for r in g])
        short = name.replace("foil_lifting_collection_tesollo_", "")
        print(f"{short:<28} {len(g):>4} {f('peak'):>7.2f} {f('end'):>7.2f} "
              f"{f('reopen'):>7.2f} {f('z_rise'):>8.4f} {f('rel_peak'):>7.3f}")

    print("\n=== 12 largest reopenings ===")
    print(f"{'ep':<5} {'len':>5} {'peak':>7} {'end':>7} {'reopen':>7} "
          f"{'after':>6}  source")
    print("-" * 88)
    for r in sorted(rows, key=lambda r: -r["reopen"])[:12]:
        print(f"{r['i']:<5} {r['n']:>5} {r['peak']:>7.2f} {r['end']:>7.2f} "
              f"{r['reopen']:>7.2f} {r['frames_after']:>6d}  {r['src']}")


if __name__ == "__main__":
    main()