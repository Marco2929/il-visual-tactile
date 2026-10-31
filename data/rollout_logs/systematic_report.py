#!/usr/bin/env python3
"""Re-run the section-6 measurements on both checkpoints, from index.csv.

The earlier findings (section 6 of the wrist findings.md) were measured on
the pre-systematic recordings, where conditions lived in free-text `notes`
and there was no success label at all - every rate had to be left as an
assertion. index.csv now carries verdict, failure_mode and a stable
condition string, so the same questions can be asked properly and asked of
BOTH models.

What is measured per checkpoint:

  SUCCESS      rate overall and per condition, from the human verdict. New:
               this was not computable before.
  PINCH        commanded vs. executed closure, and the filter/mechanical
               split. Section 6.1 claimed 8-9 mm too open with the filter
               innocent - re-checked here.
  HEIGHT       commanded depth per condition. Section 6.2 said the absolute
               model DOES adapt; the relative model's operator notes claim
               the opposite ("gar kein Verstaendnis der Hoehe").
  XY           grasp centroid per pallet position, against within-condition
               scatter.
  TIMING       closure peak vs. lowest z. Section 6.3 found 2-4 s too late.
  AGREEMENT    overlapping-chunk disagreement at the grasp.

Each block prints the metric next to the human verdict, so a metric that
disagrees with the operator is visible rather than averaged away.

Usage:
    python data/rollout_logs/systematic_report.py
    python data/rollout_logs/systematic_report.py --ckpt foil_lifting_relative_e40
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from viz import attempts, filters, labels as lab, pinch, traj  # noqa: E402
from viz.data import GRASP_LOCAL, Rollout  # noqa: E402

CKPTS = ["foil_lifting_wrist_e40", "foil_lifting_relative_e40"]
ROOT = pathlib.Path(__file__).resolve().parent


def load(ckpt):
    """-> (labels, {name: Rollout}). Unloadable rollouts are reported."""
    d = ROOT / ckpt
    labels = lab.load(d)
    rs = {}
    for name in sorted(labels):
        p = d / name
        if not p.is_dir():
            print(f"  !! {name}: in index.csv but not on disk")
            continue
        r = Rollout(p)
        if r.ok:
            rs[name] = r
        else:
            print(f"  !! {name}: {r.error}")
    return labels, rs


# --------------------------------------------------------------- success --
def success_block(labels):
    print("\n  SUCCESS (human verdict)")
    good, n, pct = lab.rate(list(labels.values()))
    print(f"    overall: {good}/{n} = {pct:.1f}%")

    bad = {}
    for lb in labels.values():
        if not lb.ok and lb.failure_mode:
            bad[lb.failure_mode] = bad.get(lb.failure_mode, 0) + 1
    if bad:
        modes = ", ".join(f"{k} {v}" for k, v in sorted(
            bad.items(), key=lambda kv: -kv[1]))
        print(f"    failure modes: {modes}")

    print("\n    per condition")
    for cond, group in lab.by_condition(labels).items():
        g, n, pct = lab.rate(group)
        names = " ".join(x.name[-3:] for x in group)
        print(f"      {cond:<34s} {g}/{n} {pct:5.0f}%   {names}")

    cc = lab.condition_conflicts(labels)
    if cc:
        print("\n    NOTE: index.csv and meta.json disagree on the condition"
              f" for {len(cc)} rollout(s):")
        for name, csv_v, notes_v in cc:
            print(f"      {name}: index.csv {csv_v!r} vs notes {notes_v!r}")
        print("    notes is used (written at record time, per run).")

    conf = lab.conflicts(labels)
    if conf:
        print("\n    NOTE: verdict and the success column disagree for "
              f"{len(conf)} rollout(s):")
        for lb in conf:
            print(f"      {lb.name}: verdict={lb.verdict} "
                  f"success={lb.success_field}")
        print("    verdict is used here, as in annotate_rollouts.py.")


# ----------------------------------------------------------------- pinch --
def pinch_block(labels, rs, pm):
    print("\n  PINCH / CLOSURE  (section 6.1: ~8-9 mm too open, filter"
          " innocent)")
    print(f"    {'run':>5s} {'ok':>3s} {'cmd':>7s} {'filt':>7s} {'exec':>7s}"
          f" {'filtr':>6s} {'mech':>6s} {'w_cmd':>7s} {'w_exe':>7s}")
    print("    " + "-" * 64)

    gaps, mechs, filts = [], [], []
    for name, r in rs.items():
        b = attempts.best(r)
        if b is None:
            continue
        filt, _ = filters.run_hand_chain(r.cmd_hand)
        f = float(filt[:, GRASP_LOCAL].sum(axis=1)[b.peak])
        c, e = b.peak_cmd_close, b.peak_exec_close
        filts.append(c - f)
        mechs.append(f - e)

        wc = we = float("nan")
        if pm is not None:
            wc = pm.width_mm(r.cmd_hand[b.peak])
            j = int(np.argmin(np.abs(r.js_t - b.t_peak)))
            we = pm.width_mm(r.exec_hand[j])
            gaps.append(we - wc)

        mark = "OK" if labels[name].ok else "-"
        print(f"    {name[-3:]:>5s} {mark:>3s} {c:>7.1f} {f:>7.1f} "
              f"{e:>7.1f} {c - f:>6.1f} {f - e:>6.1f} "
              f"{wc:>7.1f} {we:>7.1f}")

    if filts:
        print(f"\n    filter loss   median {np.median(filts):5.1f} deg")
        print(f"    mech loss     median {np.median(mechs):5.1f} deg")
    if gaps:
        print(f"    aperture gap  median {np.median(gaps):+5.1f} mm "
              f"(executed minus commanded)")
        print("    positive = the hand stayed wider open than asked.")


# ---------------------------------------------------------------- height --
def height_block(labels, rs):
    print("\n  HEIGHT  (section 6.2: does commanded depth follow the pallet?)")
    print("    Only the nominal LAYER is used: 'Zwischenhoehe' and 'untere")
    print("    Hoehe' are different layers of the stack, not the nominal")
    print("    layer raised or lowered, and pooling them would mix the two")
    print("    axes - the stack layers are listed separately below.")
    buckets, other = {}, {}
    for name, r in rs.items():
        lb = labels[name]
        z = float(np.min(r.cmd_pose[:, 2]))
        b = attempts.best(r)
        row = (name, z, b.xyz[2] if b else float("nan"), lb.ok)
        if lb.layer in ("mid", "low"):
            other.setdefault(lb.layer, []).append(row)
        else:
            buckets.setdefault(lb.height, []).append(row)

    order = ["too_low", "nominal", "too_high"]
    meds = {}
    for key in order:
        if key not in buckets:
            continue
        rows = buckets[key]
        zs = np.array([x[1] for x in rows])
        meds[key] = float(np.median(zs))
        good = sum(1 for x in rows if x[3])
        print(f"\n    {key:<9s} n={len(rows)}  success {good}/{len(rows)}"
              f"   z_min median {np.median(zs):+.4f} m"
              f"   range {(zs.max() - zs.min()) * 1000:.0f} mm")
        for nm, z, zc, ok in sorted(rows):
            print(f"      {nm[-3:]} {'OK' if ok else '  '} "
                  f"z_min {z:+.4f}  z_close {zc:+.4f}")

    if "too_high" in meds and "too_low" in meds:
        d = (meds["too_high"] - meds["too_low"]) * 1000
        print(f"\n    too_high minus too_low: {d:+.0f} mm")
        if "nominal" in meds:
            print(f"    vs nominal: too_high "
                  f"{(meds['too_high'] - meds['nominal']) * 1000:+.0f} mm, "
                  f"too_low "
                  f"{(meds['too_low'] - meds['nominal']) * 1000:+.0f} mm")
        print("    a depth that does not shift with the pallet means no"
              " height adaptation.")

    for layer in ("mid", "low"):
        if layer not in other:
            continue
        rows = other[layer]
        zs = np.array([x[1] for x in rows])
        good = sum(1 for x in rows if x[3])
        print(f"\n    [stack layer {layer}] n={len(rows)}  "
              f"success {good}/{len(rows)}   "
              f"z_min median {np.median(zs):+.4f} m")
        for nm, z, zc, ok in sorted(rows):
            print(f"      {nm[-3:]} {'OK' if ok else '  '} "
                  f"z_min {z:+.4f}  z_close {zc:+.4f}")


# -------------------------------------------------------------------- xy --
def xy_block(labels, rs):
    print("\n  XY  (grasp centroid per pallet position)")
    buckets = {}
    for name, r in rs.items():
        xy = labels[name].xy
        if xy is None:
            continue
        for s in attempts.segments(r):
            buckets.setdefault(xy, []).append(s.xyz)

    if len(buckets) < 2:
        print("    fewer than two pallet positions annotated")
        return

    cents, within = {}, []
    for key in ("links", "mitte", "rechts"):
        if key not in buckets:
            continue
        p = np.array(buckets[key])
        cents[key] = p.mean(axis=0)
        d = np.linalg.norm(p[:, :2] - p[:, :2].mean(axis=0), axis=1) * 1000
        within.append(d.mean())
        print(f"    {key:>6s} n={len(p):2d}  centroid "
              f"{cents[key][0]:+.4f} {cents[key][1]:+.4f}  "
              f"within-scatter {d.mean():.1f} mm")

    ks = list(cents)
    seps = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            s = np.linalg.norm((cents[ks[i]] - cents[ks[j]])[:2]) * 1000
            seps.append(s)
            print(f"    {ks[i]} <-> {ks[j]}: {s:.1f} mm")
    if seps and within:
        print(f"    between {np.mean(seps):.1f} mm vs within "
              f"{np.mean(within):.1f} mm  -> ratio "
              f"{np.mean(seps) / max(np.mean(within), 1e-9):.2f}")


# ---------------------------------------------------------------- timing --
def timing_block(labels, rs):
    print("\n  TIMING  (section 6.3: closure 2-4 s after the lowest point)")
    dts = []
    for name, r in rs.items():
        b = attempts.best(r)
        if b is None:
            continue
        t = np.asarray(r.cmd_t, float)
        k = int(np.argmin(np.asarray(r.cmd_pose[:, 2], float)))
        dt = b.t_peak - float(t[k])
        dts.append(dt)
        hv = traj.hover(r)
        print(f"    {name[-3:]} {'OK' if labels[name].ok else '  '} "
              f"z_min t={t[k]:6.2f}s  closure t={b.t_peak:6.2f}s  "
              f"{dt:+6.2f}s   hover {hv['longest_s'] if hv else 0:.1f}s")
    if dts:
        print(f"\n    median offset {np.median(dts):+.2f}s "
              f"(positive = hand closes AFTER the bottom)")


# ------------------------------------------------------------- agreement --
def agreement_block(labels, rs):
    print("\n  CHUNK AGREEMENT at the grasp")
    vals = []
    for name, r in rs.items():
        b = attempts.best(r)
        if b is None:
            continue
        cs = chunk_spread(r, b.t_peak)
        if cs is None:
            continue
        vals.append(cs[0])
        print(f"    {name[-3:]} {'OK' if labels[name].ok else '  '} "
              f"median {cs[0]:5.1f} mm  p95 {cs[1]:6.1f} mm")
    if vals:
        print(f"\n    median across rollouts {np.median(vals):.1f} mm")


def chunk_spread(r, t0, window=1.0):
    """Spread of overlapping chunk predictions for the same step near t0."""
    z = r._z
    acts, offs = z["chunk_actions"], np.asarray(z["chunk_offsets"], int)
    anchors, ct = np.asarray(z["chunk_anchors"], int), np.asarray(
        z["chunk_t"], float)
    if len(offs) < 3:
        return None

    per_step = {}
    for c in range(len(offs) - 1):
        if c < len(ct) and abs(ct[c] - t0) > window:
            continue
        a, b = int(offs[c]), int(offs[c + 1])
        for k in range(b - a):
            per_step.setdefault(int(anchors[c]) + k, []).append(
                acts[a + k, :3])

    sp = []
    for preds in per_step.values():
        if len(preds) < 2:
            continue
        p = np.asarray(preds, float)
        sp.append(float(np.linalg.norm(p - p.mean(axis=0), axis=1).max()
                        * 1000))
    if len(sp) < 3:
        return None
    return float(np.median(sp)), float(np.percentile(sp, 95)), len(sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=None)
    ap.add_argument("--urdf", default=None)
    a = ap.parse_args()

    pm, note = pinch.try_load(a.urdf)

    for ckpt in (a.ckpt or CKPTS):
        print(f"\n{'=' * 74}")
        print(f"{ckpt}")
        print(f"{'=' * 74}")
        labels, rs = load(ckpt)
        if not labels:
            print("  no index.csv - run annotate_rollouts.py first")
            continue
        print(f"  {len(labels)} annotated, {len(rs)} loadable. {note}")

        success_block(labels)
        pinch_block(labels, rs, pm)
        height_block(labels, rs)
        xy_block(labels, rs)
        timing_block(labels, rs)
        agreement_block(labels, rs)


if __name__ == "__main__":
    main()
