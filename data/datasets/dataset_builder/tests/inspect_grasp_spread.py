# dataset_builder/tests/inspect_grasp_spread.py
"""Does the wrist camera have anything to localise?

The recordings cover four pallet positions. A policy can reach a low action
error by recognising WHICH pallet it faces and aiming at that cluster's mean
grasp point - no image understanding required. Trimming the approach away
only forces the policy onto vision if the grasp points actually VARY within
a single pallet position.

This settles that without training anything:

  within-position spread << between-position spread
      -> grasp points are effectively fixed per pallet. There is nothing in
         the wrist image to localise, so trimming cannot reduce overfitting
         the way it is meant to - it just removes frames.

  within-position spread ~ between-position spread
      -> real per-episode variation. The policy must read the image to find
         the foil, and the experiment tests what it is meant to test.

Positions come from meta/episode_sources (the collection prefix), NOT from
k-means on the start pose: the labels are exact, and clustering on the start
pose stops working on a trimmed dataset, where the start pose already sits
right above the grasp point.

Also reports the segment lengths a trim would produce and how many episodes
contain a retry.

    python dataset_builder/tests/inspect_grasp_spread.py \\
        --zarr zarr_datasets/foil_lifting_wrist_tesollo.zarr
"""

import argparse
import pathlib
import sys

import numpy as np
import zarr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from processing.grasp_detection import (  # noqa: E402
    detect_episode_grasps, iter_episodes)


def position_of(source):
    """Collection prefix -> short position label."""
    prefix = source.split("/")[0] if "/" in source else source
    marker = "_tesollo_"
    if marker in prefix:
        return prefix.split(marker, 1)[1]
    return prefix


def load(zarr_path):
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"] if "data" in root else root
    state = np.asarray(data["state"])
    ends = np.asarray(root["meta/episode_ends"][:])

    if "meta/episode_sources" in root:
        srcs = [str(s) for s in root["meta/episode_sources"][:]]
    else:
        srcs = []
    if len(srcs) != len(ends):
        raise SystemExit(
            f"meta/episode_sources has {len(srcs)} entries but there are "
            f"{len(ends)} episodes - cannot assign positions reliably."
        )
    return state, ends, srcs


def analyse(state, ends, srcs):
    """Per-episode grasp point, position label and would-be segment length."""
    rows = []
    for i, s, e, ep in iter_episodes(state, ends):
        info = detect_episode_grasps(ep)
        n = e - s
        anchor = info["anchor_idx"]
        rows.append({
            "ep": i,
            "position": position_of(srcs[i]),
            "n_frames": n,
            "has_grasp": info["has_grasp"],
            "n_attempts": info["n_attempts"],
            "first_grasp": info["first_grasp"],
            "arrived": info["arrived_idx"],
            # Grasp point = ee (x, y) where the fingers start closing, i.e.
            # where the operator committed to a spot.
            "xy": ep[info["first_grasp"], :2].copy() if info["has_grasp"] else None,
            "seg_len": (n - anchor) if anchor >= 0 else -1,
        })
    return rows


def report_detection(rows):
    n = len(rows)
    grasped = [r for r in rows if r["has_grasp"]]
    retries = [r for r in grasped if r["n_attempts"] > 1]

    print("\n" + "=" * 66)
    print(" DETECTION")
    print("=" * 66)
    print(f"  episodes              : {n}")
    print(f"  grasp detected        : {len(grasped)}")
    print(f"  NO grasp detected     : {n - len(grasped)}   <- skipped by trim")
    print(f"  with retry (>1 pinch) : {len(retries)}   <- recovery, must be KEPT")

    att = np.array([r["n_attempts"] for r in grasped])
    if len(att):
        hist = ", ".join(f"{k}x:{int((att == k).sum())}"
                         for k in sorted(set(att.tolist())))
        print(f"  attempts per episode  : {hist}")

    if n - len(grasped):
        bad = [r["ep"] for r in rows if not r["has_grasp"]]
        tail = " ..." if len(bad) > 20 else ""
        print(f"  episodes w/o grasp    : {bad[:20]}{tail}")


def report_segments(rows):
    """Segment lengths a trim at arrived_idx would produce."""
    seg = np.array([r["seg_len"] for r in rows if r["seg_len"] > 0])
    full = np.array([r["n_frames"] for r in rows])
    if not len(seg):
        return

    print("\n" + "=" * 66)
    print(" SEGMENT LENGTHS AFTER TRIM (anchor = 10 cm above lowest z)")
    print("=" * 66)
    print(f"  full episode : min {full.min():4d}  median "
          f"{int(np.median(full)):4d}  max {full.max():4d}")
    print(f"  trimmed      : min {seg.min():4d}  median "
          f"{int(np.median(seg)):4d}  max {seg.max():4d}")
    print(f"  kept         : {seg.sum() / full.sum():.1%} of all frames")
    print(f"  percentiles  : p05 {np.percentile(seg, 5):.0f}  "
          f"p25 {np.percentile(seg, 25):.0f}  "
          f"p75 {np.percentile(seg, 75):.0f}  "
          f"p95 {np.percentile(seg, 95):.0f}")


def report_spread(rows):
    """The actual question: within-position vs between-position spread."""
    grasped = [r for r in rows if r["has_grasp"]]
    if len(grasped) < 2:
        print("\n!! too few grasps to measure spread")
        return

    xy = np.stack([r["xy"] for r in grasped])
    pos = np.array([r["position"] for r in grasped])
    labels = sorted(set(pos.tolist()))

    print("\n" + "=" * 66)
    print(" GRASP POINT SPREAD PER PALLET POSITION")
    print("=" * 66)
    print(f"  overall: std_x {xy[:, 0].std() * 1000:6.1f} mm   "
          f"std_y {xy[:, 1].std() * 1000:6.1f} mm")
    print()
    print(f"  {'position':<12} {'n':>4}  {'centre x/y [m]':>18}  "
          f"{'std_x':>7} {'std_y':>7} {'rms':>7}   (mm)")
    print("  " + "-" * 64)

    within, centres = [], []
    for lab in labels:
        m = pos == lab
        g = xy[m]
        sx, sy = float(g[:, 0].std()), float(g[:, 1].std())
        rms = float(np.hypot(sx, sy))
        within.append(rms)
        centres.append(g.mean(axis=0))
        print(f"  {lab:<12} {int(m.sum()):>4}  "
              f"[{g[:, 0].mean():+.3f} {g[:, 1].mean():+.3f}]  "
              f"{sx*1000:7.1f} {sy*1000:7.1f} {rms*1000:7.1f}")

    centres = np.stack(centres)
    between = float(np.hypot(centres[:, 0].std(), centres[:, 1].std()))
    w = float(np.mean(within))
    ratio = w / max(between, 1e-9)

    print()
    print(f"  within-position  rms : {w * 1000:6.1f} mm")
    print(f"  between-position rms : {between * 1000:6.1f} mm")
    print(f"  ratio within/between : {ratio:.2f}")
    print()
    if ratio < 0.3:
        print("  => WITHIN << BETWEEN. Grasp points are essentially fixed per")
        print("     pallet. The policy can solve this by recognising the")
        print("     position and replaying one point - there is little in the")
        print("     wrist image to localise, so trimming will not force it")
        print("     onto vision.")
    elif ratio < 0.7:
        print("  => MODERATE within-position variation. Some image-driven")
        print("     localisation is required, but position identity still")
        print("     explains most of the grasp point.")
    else:
        print("  => WITHIN ~ BETWEEN. Real per-episode variation: the grasp")
        print("     point cannot be predicted from the pallet position alone,")
        print("     so the wrist image carries the information. Trimming")
        print("     tests what it is meant to test.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", required=True)
    p.add_argument("--csv", default=None, help="write per-episode rows here")
    args = p.parse_args()

    state, ends, srcs = load(args.zarr)
    print(f"[data] {len(state)} frames, {len(ends)} episodes, "
          f"state {state.shape[1]}-dim")

    rows = analyse(state, ends, srcs)
    report_detection(rows)
    report_segments(rows)
    report_spread(rows)

    if args.csv:
        import csv
        out = pathlib.Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["ep", "position", "n_frames", "has_grasp",
                         "n_attempts", "first_grasp", "arrived", "seg_len",
                         "grasp_x", "grasp_y"])
            for r in rows:
                gx, gy = (r["xy"] if r["xy"] is not None else ("", ""))
                wr.writerow([r["ep"], r["position"], r["n_frames"],
                             int(r["has_grasp"]), r["n_attempts"],
                             r["first_grasp"], r["arrived"], r["seg_len"],
                             gx, gy])
        print(f"\n[csv] wrote {out}")


if __name__ == "__main__":
    main()