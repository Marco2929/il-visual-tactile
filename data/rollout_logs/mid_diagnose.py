#!/usr/bin/env python3
"""T2: what is the problem in the mid-layer runs? (007, 008, 009 each; 006)

Per rollout, never pooled: the question is what each run did wrong, and the
whole point of 006 vs 007-009 is that they differ (006 retried, the others
were one attempt) while their meta.json notes are byte-identical.

Six signals per run, ordered so that each one rules out a cause the next
would otherwise be blamed for:

  1. DEPTH      z_at_close and z_min against the nominal-height mid runs and
                the successful nominal baseline. Did it even get down to the
                layer?
  2. TIMING     closure peak vs. the moment of lowest z. Closing well before
                the bottom means the hand shut in mid-air.
  3. APERTURE   pinch width in mm at the peak. The closure SUM cannot answer
                this - viz/pinch.py documents rollout_010, where the weaker
                sum was the grasp that held.
  4. LOSS       filter loss vs. mechanical loss, same split as
                deficit_split.py. A large mechanical loss means the finger
                met resistance; a small one with full closure means nothing
                was between the fingers at all, which points upstream to
                perception rather than to the hand.
  5. CONTACT    sustained z tracking error (commanded vs. /target_frame).
                The only contact proxy available: no force is logged.
  6. AGREEMENT  spread between overlapping chunks around the grasp. Chunks
                that disagree at the critical instant are the direct
                correlate of "the contour is too complex" - a multimodal
                prediction at the moment it matters.

Usage:
    python data/rollout_logs/mid_diagnose.py
    python data/rollout_logs/mid_diagnose.py --urdf /path/dg5f_right.urdf
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from viz import attempts, filters, groups, pinch, traj  # noqa: E402
from viz.data import GRASP_LOCAL, lww_sequence  # noqa: E402


def reference_depth(root):
    """Depth of the runs that worked, as the yardstick for 'got down'.

    The nominal baseline is used rather than a fixed number: it is the same
    policy on the same rig, so its grasp depth is what this policy does when
    it succeeds.
    """
    zc, zm = [], []
    for r in groups.load_group(groups.NOMINAL_BASELINE, root=root,
                              verbose=False):
        b = attempts.best(r)
        if b is not None:
            zc.append(b.xyz[2])
            zm.append(float(np.min(r.cmd_pose[:, 2])))
    if not zc:
        return None
    return {"z_close": float(np.median(zc)), "z_min": float(np.median(zm))}


def chunk_spread(r, t0, window=1.0):
    """Disagreement between overlapping chunk predictions near time t0.

    For each absolute step in the window, collect what every chunk that
    covered it predicted for the TCP position, and measure the spread. The
    executor keeps the newest (last-write-wins), so a wide spread means the
    policy changed its mind about where to go as it got closer.

    -> (median_mm, p95_mm, n_steps) or None.
    """
    z = r._z
    acts, offs = z["chunk_actions"], z["chunk_offsets"]
    anchors, ct = z["chunk_anchors"], z["chunk_t"]
    if len(offs) < 3:
        return None

    # Steps covered by the window, via the same step->time fit data.py uses.
    sel = np.abs(np.asarray(ct, float) - t0) <= window
    if sel.sum() < 2:
        return None

    per_step = {}
    for c in range(len(offs) - 1):
        if not sel[c] if c < len(sel) else True:
            continue
        a, b = int(offs[c]), int(offs[c + 1])
        base = int(anchors[c])
        for k in range(b - a):
            per_step.setdefault(base + k, []).append(acts[a + k, :3])

    spreads = []
    for step, preds in per_step.items():
        if len(preds) < 2:
            continue
        p = np.asarray(preds, float)
        d = np.linalg.norm(p - p.mean(axis=0), axis=1)
        spreads.append(d.max() * 1000.0)
    if len(spreads) < 3:
        return None
    return (float(np.median(spreads)), float(np.percentile(spreads, 95)),
            len(spreads))


def z_error(r):
    """Commanded vs. executor-output z over the last third, in mm."""
    if not r.has_target or len(r.tf_t) < 20:
        return None
    zc = np.interp(r.tf_t, r.cmd_t, r.cmd_pose[:, 2])
    err = (np.asarray(r.tf_pose[:, 2]) - zc) * 1000.0
    tail = err[int(0.66 * len(err)):]
    if len(tail) < 5:
        return None
    return float(np.median(tail)), float(np.percentile(np.abs(tail), 95))


def diagnose(r, ref, pm):
    print(f"\n{'=' * 72}")
    print(f"{r.name}   {r.notes}")
    print(f"{'=' * 72}")

    segs = attempts.segments(r)
    reop = attempts.reopenings(r)
    print(f"  duration {r.summary['duration_s']:.0f}s, "
          f"{len(segs)} commanded attempt(s), "
          f"{len(reop)} reopening(s) >5 deg")
    if not segs:
        print("  no commanded closure at all - nothing to diagnose")
        return
    for a, b, drop in reop:
        print(f"    hand reopened {drop:.1f} deg between attempt "
              f"{a} and {b}")

    # ---- 1. depth -------------------------------------------------------
    b = attempts.best(r)
    zmin = float(np.min(r.cmd_pose[:, 2]))
    print("\n  1. DEPTH")
    print(f"     z at closure {b.xyz[2]:+.4f} m, lowest commanded z "
          f"{zmin:+.4f} m")
    if ref:
        print(f"     successful nominal runs: z_close "
              f"{ref['z_close']:+.4f}, z_min {ref['z_min']:+.4f} m")
        print(f"     difference: z_close "
              f"{(b.xyz[2] - ref['z_close']) * 1000:+.1f} mm, "
              f"z_min {(zmin - ref['z_min']) * 1000:+.1f} mm")

    hv = traj.hover(r)
    dp = traj.descent_profile(r)
    if hv and dp:
        print(f"     reached its floor at t={dp['t_zmin']:.1f}s "
              f"({100 * dp['frac_to_zmin']:.0f}% in), "
              f"descended {dp['descent_mm']:.0f} mm")
        print(f"     longest stay within 10 mm of that floor: "
              f"{hv['longest_s']:.1f}s")

    # ---- 2. timing ------------------------------------------------------
    t = np.asarray(r.cmd_t, float)
    k = int(np.argmin(np.asarray(r.cmd_pose[:, 2], float)))
    dt = b.t_peak - float(t[k])
    print("\n  2. TIMING")
    print(f"     lowest z at t={t[k]:.2f}s, closure peak at "
          f"t={b.t_peak:.2f}s  ->  {dt:+.2f}s")
    if dt < -0.5:
        print("     the hand closed BEFORE the arm was at its lowest:")
        print("     it shut while still descending.")
    elif dt > 2.0:
        print("     the hand closed well after the bottom was reached.")
    else:
        print("     closure and lowest point essentially coincide.")

    # ---- 3. aperture ----------------------------------------------------
    print("\n  3. APERTURE")
    if pm is None:
        print("     (no URDF - pinch width unavailable)")
    else:
        for s in segs:
            w_cmd = pm.width_mm(r.cmd_hand[s.peak])
            j = int(np.argmin(np.abs(r.js_t - s.t_peak)))
            w_exe = pm.width_mm(r.exec_hand[j])
            print(f"     attempt {s.index}: commanded {w_cmd:5.1f} mm, "
                  f"executed {w_exe:5.1f} mm  "
                  f"(closure sum {s.peak_cmd_close:.0f} deg)")

    # ---- 4. loss split --------------------------------------------------
    print("\n  4. LOSS SPLIT (per attempt)")
    filt, _ = filters.run_hand_chain(r.cmd_hand)
    fclose = filt[:, GRASP_LOCAL].sum(axis=1)
    print(f"     {'#':>3s} {'cmd':>7s} {'filtered':>9s} {'exec':>7s} "
          f"{'filter':>7s} {'mech':>7s}")
    for s in segs:
        c = s.peak_cmd_close
        f = float(fclose[s.peak])
        e = s.peak_exec_close
        print(f"     {s.index:>3d} {c:>7.1f} {f:>9.1f} {e:>7.1f} "
              f"{c - f:>7.1f} {f - e:>7.1f}")
    print("     large mech -> the finger met resistance (something there);"
          "\n     small mech at full closure -> it closed on nothing.")

    # ---- 5. contact proxy ----------------------------------------------
    print("\n  5. CONTACT PROXY (no force is logged)")
    ze = z_error(r)
    if ze is None:
        print("     /target_frame not logged for this rollout")
    else:
        print(f"     z error, last third: median {ze[0]:+.1f} mm, "
              f"|p95| {ze[1]:.1f} mm")
        print("     a persistent positive error = the executor holding above"
              "\n     what was asked, i.e. the arm pressing into the layer.")

    # ---- 6. chunk agreement --------------------------------------------
    print("\n  6. CHUNK AGREEMENT around the grasp")
    cs = chunk_spread(r, b.t_peak)
    if cs is None:
        print("     too few overlapping chunks in the window")
    else:
        print(f"     overlapping predictions for the same step differ by "
              f"median {cs[0]:.1f} mm, p95 {cs[1]:.1f} mm "
              f"({cs[2]} steps)")
        print("     wide spread = the policy kept revising where to grasp,"
              "\n     which is what an ambiguous contour looks like.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=groups.CKPT)
    ap.add_argument("--urdf", default=None)
    a = ap.parse_args()

    root = groups.root_for(a.ckpt)
    print("=== T2 mid-layer diagnosis ===")
    print("006 retried; 007-009 were one attempt each.")
    print("Their meta.json notes are identical, so the split is only in")
    print("viz/groups.py.")

    pm, note = pinch.try_load(a.urdf)
    print(f"\n{note}")

    ref = reference_depth(root)

    for g in (groups.MID_RETRY, groups.MID_SINGLE):
        for r in groups.load_group(g, root=root):
            diagnose(r, ref, pm)


if __name__ == "__main__":
    main()
