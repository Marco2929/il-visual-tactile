#!/usr/bin/env python3
"""Where is the relative policy supposed to get the pallet HEIGHT from?

Open point from the relative findings (section 10.4). Established so far:

  * both models fail 0/6 once the pallet height is changed;
  * the relative model overshoots by +217 mm against nominal (z_min median
    +0.272 m at "increased"), the absolute one by +76 mm;
  * the decoder is exonerated (gate 7.1 green) and there is no systematic
    drift (section 10.2), so the overshoot is not an integration artefact.

What is left is the information itself. relative_pose.py states the
trade-off plainly: relativising removes the absolute pose from the
proprioceptive input *on purpose*, so "the spatial part has to come from
the image". Height is spatial. If the point cloud does not carry the pallet
top surface, the relative policy has no source for it at all, while the
absolute policy can still read it off its own z.

That is a statement about the OBSERVATION, not about the policy, and the
rollouts logged the clouds (`pc`, N x 1024 x 6), so it is checkable.

TESTS
-----
  1. SURFACE PRESENT   Near the eventual grasp XY, is there any cloud point
                       at a plausible pallet-top height? If the surface is
                       missing from the cloud, no policy can see it.
  2. SURFACE TRACKS    Does the cloud-derived surface height differ between
                       the height conditions the way the real pallet did?
                       This is the actual question: an observation that does
                       not move cannot explain a correct response, and a
                       response that moves 3x further than the observation
                       is the policy's own doing.
  3. COMMAND vs CLOUD  Commanded z_min against the cloud surface height,
                       per condition. Gain ~1 means the policy follows what
                       it sees; >>1 means it amplifies it.

The cloud is in the same frame as the commanded poses (base_link; see
viz/data.py), so the comparison is direct.

Usage (in the container):
    python data/rollout_logs/height_source.py
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from viz import attempts, labels as lab  # noqa: E402
from viz.data import Rollout  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
CKPTS = ["foil_lifting_wrist_e40", "foil_lifting_relative_e40"]

# Radius around the grasp XY in which cloud points count as "the thing being
# grasped". 60 mm is roughly the pinch span, so it selects the surface the
# hand is aiming at rather than the whole pallet.
XY_RADIUS_M = 0.06


def surface_height(r, xy, t_window=None):
    """-> (z_top, n_points) of the cloud near `xy`, in metres.

    z_top is the 90th percentile of z among the selected points, not the
    maximum: a single stray point from the hand or a reflection off the foil
    would otherwise define the surface.
    """
    z = r._z
    if "pc" not in z.files or len(z["pc"]) == 0:
        return None, 0
    pc = np.asarray(z["pc"], dtype=np.float64)
    pc_t = np.asarray(z["pc_t"], dtype=np.float64)

    if t_window is not None:
        lo, hi = t_window
        sel = (pc_t >= lo) & (pc_t <= hi)
        if sel.sum() == 0:
            # Nearest single frame rather than nothing: a run whose cloud
            # stream has a gap at the grasp still has a surface before it.
            sel = np.zeros(len(pc_t), bool)
            sel[int(np.argmin(np.abs(pc_t - 0.5 * (lo + hi))))] = True
        pc = pc[sel]

    pts = pc.reshape(-1, pc.shape[-1])[:, :3]
    d = np.linalg.norm(pts[:, :2] - np.asarray(xy, float), axis=1)
    near = pts[d <= XY_RADIUS_M]
    if len(near) < 20:
        return None, len(near)
    return float(np.percentile(near[:, 2], 90)), len(near)


def collect(ckpt):
    d = ROOT / ckpt
    labels = lab.load(d)
    rows = []
    for name in sorted(labels):
        r = Rollout(d / name)
        if not r.ok:
            continue
        lb = labels[name]
        if lb.layer in ("mid", "low"):
            continue                      # height axis only
        b = attempts.best(r)
        if b is None:
            continue

        # Cloud as it was shortly BEFORE the hand arrived: once the hand is
        # in the grasp the fingers occlude and pollute the surface estimate.
        t = np.asarray(r.cmd_t, float)
        k = int(np.argmin(np.asarray(r.cmd_pose[:, 2], float)))
        t_low = float(t[k])
        zs, n = surface_height(r, b.xyz[:2], (t_low - 2.0, t_low - 0.2))

        rows.append({
            "name": name,
            "ok": lb.ok,
            "height": lb.height,
            "z_min": float(np.min(r.cmd_pose[:, 2])),
            "z_close": float(b.xyz[2]),
            "surf": zs,
            "n_pts": n,
        })
    return rows


def report(ckpt, rows):
    print(f"\n{'=' * 72}")
    print(ckpt)
    print(f"{'=' * 72}")

    print(f"  {'run':>5s} {'ok':>3s} {'cond':>9s} {'z_min':>9s} "
          f"{'surface':>9s} {'clear':>8s} {'pts':>7s}")
    print("  " + "-" * 60)
    for x in rows:
        s = f"{x['surf']:+.4f}" if x["surf"] is not None else "    --"
        clear = (f"{(x['z_min'] - x['surf']) * 1000:+7.1f}"
                 if x["surf"] is not None else "     --")
        print(f"  {x['name'][-3:]:>5s} {'OK' if x['ok'] else '--':>3s} "
              f"{x['height']:>9s} {x['z_min']:>+9.4f} {s:>9s} "
              f"{clear:>8s} {x['n_pts']:>7d}")

    order = ["too_low", "nominal", "too_high"]
    stats = {}
    for cond in order:
        g = [x for x in rows if x["height"] == cond]
        if not g:
            continue
        zs = np.array([x["z_min"] for x in g])
        sv = np.array([x["surf"] for x in g if x["surf"] is not None])
        stats[cond] = (np.median(zs),
                       np.median(sv) if len(sv) else None,
                       len(g), len(sv))

    print("\n  1/2. does the OBSERVATION move with the pallet?")
    print(f"  {'cond':>9s} {'n':>3s} {'z_min':>10s} {'surface':>10s} "
          f"{'with cloud':>11s}")
    for cond in order:
        if cond not in stats:
            continue
        zm, sv, n, nc = stats[cond]
        s = f"{sv:+.4f}" if sv is not None else "      --"
        print(f"  {cond:>9s} {n:>3d} {zm:>+10.4f} {s:>10s} {nc:>8d}/{n}")

    if "too_high" in stats and "nominal" in stats:
        dz = (stats["too_high"][0] - stats["nominal"][0]) * 1000
        s_hi, s_no = stats["too_high"][1], stats["nominal"][1]
        print(f"\n  3. GAIN, too_high vs nominal")
        print(f"     commanded z_min moved {dz:+.0f} mm")
        if s_hi is not None and s_no is not None:
            ds = (s_hi - s_no) * 1000
            print(f"     cloud surface  moved {ds:+.0f} mm")
            if abs(ds) > 5:
                print(f"     gain = {dz / ds:+.2f}")
                print("     ~1 means the policy follows what it sees;")
                print("     >>1 means it amplifies a correct observation.")
            else:
                print("     the surface barely moved: the cloud does NOT")
                print("     encode the height change, so the policy has no")
                print("     visual source for it.")
        else:
            print("     surface not recoverable for one condition")

    missing = [x["name"] for x in rows if x["surf"] is None]
    if missing:
        print(f"\n  no surface near the grasp XY in {len(missing)} run(s): "
              f"{' '.join(m[-3:] for m in missing)}")
        print("  (fewer than 20 cloud points within "
              f"{XY_RADIUS_M * 1000:.0f} mm - the policy was reaching at a")
        print("  place the cloud shows nothing at)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=None)
    a = ap.parse_args()
    print("=== height: what does the observation actually contain? ===")
    print(f"cloud points within {XY_RADIUS_M * 1000:.0f} mm of the grasp XY,")
    print("sampled 2.0-0.2 s before the lowest point (before the fingers")
    print("occlude). Surface = 90th percentile of z.")
    for ckpt in (a.ckpt or CKPTS):
        rows = collect(ckpt)
        if rows:
            report(ckpt, rows)


if __name__ == "__main__":
    main()
