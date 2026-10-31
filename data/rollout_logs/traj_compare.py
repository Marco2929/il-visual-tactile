#!/usr/bin/env python3
"""T1: do the first runs look EXACTLY the same? (000-003, 005)

THE POINT, and why this is a test rather than a repeatability check: these
five runs share the layer and the setup, but the pallet was NOT placed back
on an exact mark - its position varies slightly from run to run. So the
visual observation differs every time while the proprioceptive state at the
gate does not.

  near-identical trajectories  -> the policy is NOT reading the pallet out of
                                  the image. It replays a path driven by
                                  proprioception, and the pallet could just
                                  as well not be there. This is the
                                  overfitting finding.
  trajectories that track the
  pallet                       -> the vision encoder contributes, and the
                                  spread below is a measure of how much.

The number this script exists to produce is the SPREAD, which every other
test then uses as its yardstick: the XY test (013/014/015) and the height
test only mean something relative to how much the policy varies when nothing
was deliberately changed.

CAVEAT, stated because it bounds the conclusion: the actual pallet offsets
were never recorded (see viz/groups.py). We can therefore say "the policy
varied by X mm while the pallet moved somewhat", not "it should have varied
by Y". A small spread is still conclusive - a policy that ignores the pallet
cannot be explained away by an unknown offset - whereas a LARGE spread would
need the offsets to interpret.

Usage:
    python data/rollout_logs/traj_compare.py
    python data/rollout_logs/traj_compare.py --group mid_single
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from viz import attempts, groups, pinch, traj  # noqa: E402
from viz.filters import quat_angle_deg  # noqa: E402


# Scale of the motion itself, so a spread can be called large or small
# rather than just printed. Taken from the commanded path length over the
# closure-aligned window; a 54 mm disagreement means something very
# different on a 100 mm approach than on a 700 mm one.
def path_length_mm(curves):
    lens = [float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)) * 1000.0)
            for c in curves.values()]
    return float(np.median(lens))


def verdict(sp, xy_radius_mm, path_mm):
    """State what the measured spread implies, instead of asserting it.

    The thresholds are RELATIVE to the length of the motion, because that is
    the only thing that makes "identical" meaningful. An earlier version of
    this script printed a fixed conclusion ("a few mm -> replay"); the data
    came back at 54 mm and the text would have been simply wrong. Anything
    stated here is therefore derived from the numbers above it.
    """
    med = sp["median_mm"]
    frac = med / max(path_mm, 1e-9)

    print(f"\n        motion scale: commanded path is {path_mm:.0f} mm long")
    print(f"        disagreement is {100 * frac:.1f}% of that")

    print("\n        verdict:")
    if frac < 0.02:
        print("        * paths are effectively IDENTICAL. The pallet moved")
        print("          between these runs and the policy did not: this is")
        print("          proprioceptive replay, i.e. the overfitting finding.")
    elif frac < 0.10:
        print("        * paths are CLOSE but not identical. Some of the")
        print("          variation may be pallet tracking, some noise - and")
        print("          with the real offsets unrecorded the two cannot be")
        print("          separated from this test alone.")
    else:
        print("        * paths DIFFER substantially. The runs are not a")
        print("          single replayed motion, so plain proprioceptive")
        print("          overfitting is NOT supported by this comparison.")
        print("          Whether the variation is pallet-driven cannot be")
        print("          decided here: the actual pallet offsets were never")
        print("          recorded. If they were a few cm, this spread is")
        print("          consistent with the policy tracking the pallet; if")
        print("          they were millimetres, it is run-to-run noise.")

    print(f"\n        yardstick for T3/T4: {med:.1f} mm path RMSE,")
    print(f"        {xy_radius_mm:.1f} mm XY scatter at the grasp.")
    print("        A deliberate pallet shift (T3) or height change (T4) only")
    print("        counts as a response if it exceeds these.")


def _matrix(title, names, m, unit="mm"):
    short = [n.replace("rollout_", "") for n in names]
    out = [f"  {title}", "      " + "".join(f"{s:>8s}" for s in short)]
    for i, s in enumerate(short):
        row = "".join("       ." if i == j else f"{m[i, j]:8.1f}"
                      for j in range(len(short)))
        out.append(f"  {s:>4s}" + row)
    out.append(f"        (3D RMSE of the commanded TCP path, {unit})")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="nominal_baseline",
                    help="group key from viz/groups.py")
    ap.add_argument("--ckpt", default=groups.CKPT)
    ap.add_argument("--urdf", default=None, help="hand URDF for pinch width")
    a = ap.parse_args()

    g = groups.BY_KEY.get(a.group)
    if g is None:
        raise SystemExit(f"unknown group {a.group!r}. "
                         f"Known: {', '.join(sorted(groups.BY_KEY))}")

    root = groups.root_for(a.ckpt)
    print(f"=== T1 trajectory agreement: {g.key} ===")
    print(f"{g.label}")
    print(f"rollouts: {', '.join(g.names)}\n")

    rs = groups.load_group(g, root=root)
    if len(rs) < 2:
        raise SystemExit("need at least two loadable rollouts to compare")

    pm, note = pinch.try_load(a.urdf)
    print(f"  {note}\n")

    # ---------------------------------------------------------- per run --
    print("  per rollout")
    print(f"  {'run':>5s} {'dur':>6s} {'tries':>6s} {'t_close':>8s} "
          f"{'grasp x':>9s} {'grasp y':>9s} {'grasp z':>9s} "
          f"{'z_min':>8s} {'cmd':>7s} {'exec':>7s} {'pinch':>7s}")
    print("  " + "-" * 94)

    pts, quats, rows, ps_xy = [], [], [], float("nan")
    for r in rs:
        segs = attempts.segments(r)
        b = attempts.best(r)
        if b is None:
            print(f"  {r.name[-3:]:>5s} {r.summary['duration_s']:>5.0f}s "
                  f"{'0':>6s}   no commanded closure found")
            continue
        xyz = b.xyz
        pts.append(xyz)
        quats.append((r.name, b.quat))
        pw = pm.width_mm(r.cmd_hand[b.peak]) if pm is not None else float("nan")
        rows.append((r, b))
        print(f"  {r.name[-3:]:>5s} {r.summary['duration_s']:>5.0f}s "
              f"{len(segs):>6d} {b.t_peak:>7.2f}s "
              f"{xyz[0]:>9.4f} {xyz[1]:>9.4f} {xyz[2]:>9.4f} "
              f"{b.z_min:>8.4f} {b.peak_cmd_close:>7.1f} "
              f"{b.peak_exec_close:>7.1f} {pw:>7.1f}")

    # ------------------------------------------------- grasp point scatter
    if len(pts) >= 2:
        ps = traj.point_spread(np.array(pts))
        print("\n  grasp point scatter (commanded TCP at the closure peak)")
        print(f"    centroid        "
              f"{ps['centroid'][0]:+.4f} {ps['centroid'][1]:+.4f} "
              f"{ps['centroid'][2]:+.4f} m")
        print(f"    mean radius     {ps['radius_mm']:7.1f} mm")
        print(f"    max radius      {ps['max_radius_mm']:7.1f} mm")
        print(f"    widest pair     {ps['max_pair_mm']:7.1f} mm")
        print(f"    std per axis    x {ps['std_mm'][0]:.1f}  "
              f"y {ps['std_mm'][1]:.1f}  z {ps['std_mm'][2]:.1f} mm")

        # XY only: the height was nominally the same for all of these, so a
        # tight XY with a loose Z means something quite different from the
        # reverse, and the XY test (T3) compares against the XY figure.
        xy = np.array(pts)[:, :2]
        c = xy.mean(axis=0)
        d = np.linalg.norm(xy - c, axis=1)
        ps_xy = float(d.mean() * 1000)
        print(f"    XY only: mean radius {ps_xy:.1f} mm, "
              f"max {d.max() * 1000:.1f} mm")

    if len(quats) >= 2:
        ang = [quat_angle_deg(quats[i][1], quats[j][1])
               for i in range(len(quats)) for j in range(i + 1, len(quats))]
        print(f"    orientation at closure: pairwise angle median "
              f"{np.median(ang):.2f} deg, max {max(ang):.2f} deg")

    # ------------------------------------------------------- whole paths --
    print("\n  path agreement")

    phase = {r.name: c for r in rs
             if (c := traj.resample_phase(r)) is not None}
    if len(phase) >= 2:
        names, m = traj.pairwise(phase)
        print()
        print(_matrix("phase-aligned (t normalised to the gate window)",
                      names, m))
        sp = traj.spread(phase)
        print(f"        spread: median {sp['median_mm']:.1f} mm, "
              f"p90 {sp['p90_mm']:.1f}, max {sp['max_mm']:.1f} "
              f"over {sp['n_pairs']} pairs")

    grid, ev = traj.common_event_window(rs)
    if grid is not None and len(ev) >= 2:
        names, m = traj.pairwise(ev)
        print()
        print(_matrix(f"closure-aligned (t=0 at the grasp, "
                      f"{grid[0]:+.1f}..{grid[-1]:+.1f}s)", names, m))
        sp = traj.spread(ev)
        print(f"        spread: median {sp['median_mm']:.1f} mm, "
              f"p90 {sp['p90_mm']:.1f}, max {sp['max_mm']:.1f} "
              f"over {sp['n_pairs']} pairs")

        # Per-axis, because an XY-identical path that differs only in z is
        # the signature of a height-seeking policy, and the pooled 3D RMSE
        # would hide it.
        stack = np.stack([ev[n] for n in names], axis=0)
        band = stack.max(axis=0) - stack.min(axis=0)
        print("\n        envelope across runs, per axis (mm)")
        for i, ax in enumerate("xyz"):
            print(f"          {ax}: median {np.median(band[:, i]) * 1000:6.1f}"
                  f"   max {band[:, i].max() * 1000:6.1f}")

        verdict(sp, ps_xy, path_length_mm(ev))


if __name__ == "__main__":
    main()
