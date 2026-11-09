#!/usr/bin/env python3
"""T4: how do the too-high and too-low runs compare? (17-19 | 22-25 | 27,29)

Read per rollout, never averaged: the protocol claims two DIFFERENT failure
modes, and a mean over four runs that stalled at four depths describes none
of them.

  too high (17,18,19)   claim: behaviour stays stable, no inappropriate
                        pushing down. Tested with the sustained z tracking
                        error - if the policy keeps commanding a z the arm
                        cannot reach, it IS pushing.
  too low (22-25, 27,29) claim: the policy stops and does not travel further
                        down. Tested with the stagnation detector on the
                        COMMANDED z, which is the policy's own decision
                        rather than a blocked arm.

The decisive plot is the z_min column across all conditions. If the lowest
commanded z clusters within a few mm whether the layer sits high, nominal or
low, there is no height adaptation at all - which is the direct argument for
the planned variable-height data collection.

Mid-too-low and low-too-low are kept apart: different layer, different
contour, and merging them would blur the one comparison that separates
"cannot reach this height" from "cannot grasp this contour".

LIMITATION: the real height deltas were not recorded, so a missing response
can be shown but not scaled.

Usage:
    python data/rollout_logs/height_compare.py
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz import attempts, groups, traj  # noqa: E402

# An attempt whose executed closure meets or exceeds the commanded one never
# closed on anything - the same criterion viz.data.flag_outliers uses. Those
# rows are shown but marked, because a "grasp" with no object in it carries
# no information about height.
NO_GRASP_DEFICIT = 0.5


def rollout_row(r):
    segs = attempts.segments(r)
    st = traj.hover(r)
    dp = traj.descent_profile(r)
    zc = np.asarray(r.cmd_pose[:, 2], dtype=float)

    best = attempts.best(r)
    z_at = best.xyz[2] if best is not None else float("nan")
    cmd = best.peak_cmd_close if best is not None else float("nan")
    dfc = best.deficit if best is not None else float("nan")

    ghost = sum(1 for s in segs if s.deficit < NO_GRASP_DEFICIT)

    return {
        "name": r.name,
        "dur": float(r.summary["duration_s"]),
        "n": len(segs),
        "ghost": ghost,
        "z_min": float(zc.min()),
        "z_at_close": float(z_at),
        "cmd": float(cmd),
        "deficit": float(dfc),
        "hover_s": st["longest_s"] if st else float("nan"),
        "hover_tot": st["total_s"] if st else float("nan"),
        "stall": bool(st["is_stall"]) if st else False,
        "t_zmin": dp["t_zmin"] if dp else float("nan"),
        "frac": dp["frac_to_zmin"] if dp else float("nan"),
        "after": dp["after_zmin_s"] if dp else float("nan"),
    }


def press_proxy(r):
    """Sustained z tracking error: is the arm being pushed into the layer?

    No force is logged (log_rollout.py does not subscribe to hand_ft), so a
    persistent gap between the commanded z and the executed z is the only
    available contact indicator. Executed z needs no FK here because only
    the TREND matters, and /target_frame is the executor's own output in the
    same frame as cmd_pose.

    -> (median_mm, p95_mm) over the last third of the run, or None.
    """
    if not r.has_target or len(r.tf_t) < 20:
        return None
    zc = np.interp(r.tf_t, r.cmd_t, r.cmd_pose[:, 2])
    err = (np.asarray(r.tf_pose[:, 2]) - zc) * 1000.0
    tail = err[int(0.66 * len(err)):]
    if len(tail) < 5:
        return None
    return float(np.median(tail)), float(np.percentile(np.abs(tail), 95))


def show(group, root):
    print(f"\n  --- {group.label}  [{group.height}] ---")
    rs = groups.load_group(group, root=root)
    if not rs:
        print("    no loadable rollouts")
        return []

    print(f"    {'run':>5s} {'dur':>6s} {'n':>3s} {'ghost':>6s} "
          f"{'z_min':>8s} {'z_close':>8s} {'cmd':>7s} {'defic':>7s} "
          f"{'hover':>7s} {'t_zmin':>7s} {'after':>7s}")
    print("    " + "-" * 80)

    rows = []
    for r in rs:
        d = rollout_row(r)
        rows.append(d)
        flag = " <-" if d["stall"] else ""
        print(f"    {d['name'][-3:]:>5s} {d['dur']:>5.0f}s {d['n']:>3d} "
              f"{d['ghost']:>6d} {d['z_min']:>8.4f} "
              f"{d['z_at_close']:>8.4f} {d['cmd']:>7.1f} "
              f"{d['deficit']:>7.1f} {d['hover_s']:>6.1f}s "
              f"{d['t_zmin']:>6.1f}s {d['after']:>6.1f}s{flag}")

    for r in rs:
        p = press_proxy(r)
        if p is not None:
            print(f"      {r.name[-3:]}: z tracking error in the last third "
                  f"median {p[0]:+.1f} mm, |p95| {p[1]:.1f} mm")

    print("      (ghost  = attempts closing on nothing: executed >= commanded)")
    print("      (hover  = longest unbroken time within 10 mm of this run's"
          " own z_min;")
    print("                '<-' if >=2s, i.e. it arrived at the floor and"
          " stayed)")
    print("      (t_zmin = when the lowest z was reached, after = time left"
          " afterwards)")
    return rows


def compare(all_rows, baseline_rows):
    print("\n  === z_min across conditions ===")
    print("  If these cluster, the policy does not adapt its depth.\n")
    print(f"  {'condition':>16s} {'n':>3s} {'z_min median':>13s} "
          f"{'min':>9s} {'max':>9s} {'spread':>9s}")
    print("  " + "-" * 64)

    rows = list(baseline_rows) + list(all_rows)
    pooled = []
    for label, rs in rows:
        if not rs:
            continue
        z = np.array([d["z_min"] for d in rs])
        pooled.extend(z.tolist())
        print(f"  {label:>16s} {len(z):>3d} {np.median(z):>13.4f} "
              f"{z.min():>9.4f} {z.max():>9.4f} "
              f"{(z.max() - z.min()) * 1000:>8.1f}m")

    if len(pooled) < 3:
        return
    p = np.array(pooled)
    print(f"\n  pooled over every condition: median {np.median(p):.4f} m, "
          f"full range {(p.max() - p.min()) * 1000:.1f} mm")

    print("\n  verdict")
    groups_z = [np.median([d["z_min"] for d in rs])
                for _, rs in rows if rs]
    between = (max(groups_z) - min(groups_z)) * 1000 if groups_z else 0.0
    within = np.median([
        (max(d["z_min"] for d in rs) - min(d["z_min"] for d in rs)) * 1000
        for _, rs in rows if len(rs) > 1] or [0.0])

    print(f"    spread of condition medians : {between:7.1f} mm")
    print(f"    typical spread within one   : {within:7.1f} mm")

    if between < max(within, 10.0):
        print("\n    * The lowest commanded z is the same whether the layer")
        print("      sits high, nominal or low. No height generalisation:")
        print("      the policy descends to a learnt depth, not to the")
        print("      surface it is looking at.")
    else:
        print("\n    * The commanded depth does shift between conditions.")
        print("      Compare against the real height deltas - not recorded -")
        print("      to judge whether the shift is of the right size.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=groups.CKPT)
    a = ap.parse_args()
    root = groups.root_for(a.ckpt)

    print("=== T4 height variation ===")
    print("read per rollout; height deltas were not recorded")

    out = []
    for g in groups.HEIGHT_GROUPS:
        out.append((g.key, show(g, root)))

    # Nominal-height references: the mid single-attempt runs are the right
    # comparison for the mid height tests, and the nominal baseline anchors
    # what a successful depth looks like.
    base = []
    for g in (groups.MID_SINGLE, groups.NOMINAL_BASELINE):
        base.append((g.key + " (ref)", show(g, root)))

    compare(out, base)


if __name__ == "__main__":
    main()
