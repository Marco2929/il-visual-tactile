#!/usr/bin/env python3
"""Aggregate every qa_report.json and audit camera-to-state alignment.

The per-episode verdict answers "is this episode usable"; it does not answer
"are the modalities describing the same instant". This script asks the second
question across all collections at once, because misalignment is systematic
(a clock offset, a missing trim) and only visible in the distribution.

What is checked, and why each one matters for alignment:

  1. FILE COUNTS. Every per-frame artifact must have exactly n_frames
     entries. front_rgb with one file fewer than state means every index
     after the gap pairs an image with the wrong state row - a silent
     off-by-one for the rest of the episode.
  2. ZED->wrist gap. The wrist camera runs on its own clock in a separate
     bag. A mean gap above half a wrist period means the paired frames are
     closer to the neighbouring instant than to their own.
  3. Stale lead frames / trimming. Frames that precede the first wrist
     message carry a carried-forward image. If they were not trimmed, the
     start of the episode shows the wrist BEFORE the motion began.
  4. Byte-identical consecutive wrist images. Ground truth for stale
     duplicates - no clock involved.
  5. ZED->ROS state gap. Same question for the proprioception stream.
  6. Depth/RGB pixel alignment and resolution consistency.
  7. Frozen state rows + dropped frames, which break the time base itself.

Usage:
    python data/record_extractor/audit_alignment.py [external_dir]
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

DEFAULT_ROOT = "data/episodes/external"

# Alignment-relevant limits. Deliberately independent of qa_report.THRESHOLDS:
# this is an audit, so it should not inherit a limit that was itself relaxed.
WRIST_MEAN_MS = 20.0    # half a 30 fps wrist period
WRIST_MAX_MS = 33.0     # one full wrist period: beyond this the pairing flips
STATE_MEAN_MS = 5.0     # sub-frame at 60 fps


def load_reports(root):
    out = defaultdict(list)
    for coll in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(coll):
            continue
        for p in sorted(glob.glob(os.path.join(coll, "episode_*",
                                               "qa_report.json"))):
            try:
                with open(p) as fh:
                    out[os.path.basename(coll)].append((p, json.load(fh)))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  !! unreadable: {p} ({exc})")
    return out


def _g(d, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def fmt_stat(vals, unit=""):
    a = np.array([v for v in vals if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "n/a"
    return (f"n={a.size} mean={a.mean():.2f}{unit} "
            f"med={np.median(a):.2f}{unit} p95={np.percentile(a, 95):.2f}"
            f"{unit} max={a.max():.2f}{unit}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    reports = load_reports(root)
    if not reports:
        sys.exit(f"no qa_report.json found under {root}")

    total = sum(len(v) for v in reports.values())
    print("=" * 78)
    print(f"ALIGNMENT AUDIT  {root}")
    print("=" * 78)
    print(f"collections: {len(reports)}   episodes: {total}\n")

    verdicts = Counter()
    for coll, items in reports.items():
        c = Counter(r["status"] for _, r in items)
        verdicts.update(c)
        print(f"  {coll:<46} n={len(items):<4} "
              f"PASS {c['PASS']:<4} WARN {c['WARN']:<4} FAIL {c['FAIL']}")
    print(f"\n  {'TOTAL':<46} n={total:<4} "
          f"PASS {verdicts['PASS']:<4} WARN {verdicts['WARN']:<4} "
          f"FAIL {verdicts['FAIL']}")

    flat = [(coll, p, r) for coll, items in reports.items()
            for p, r in items]

    # ---- 1. file-count mismatches: the hard off-by-one check -------------
    print("\n" + "-" * 78)
    print("1. PER-FRAME FILE COUNTS vs state rows (off-by-one detector)")
    print("-" * 78)
    mismatches = []
    missing_dirs = Counter()
    for coll, p, r in flat:
        mm = _g(r, "metrics", "counts", "mismatches", default=[])
        for x in mm:
            mismatches.append((coll, r["episode"], x))
        for name, cnt in (_g(r, "metrics", "counts", "dirs",
                             default={}) or {}).items():
            if cnt is None:
                missing_dirs[name] += 1
    if mismatches:
        print(f"  {len(mismatches)} MISMATCH(ES) - these desynchronise "
              f"image<->state pairing:")
        for coll, ep, x in mismatches[:25]:
            print(f"    {coll}/{ep}: {x['dir']} has {x['found']} files, "
                  f"expected {x['expected']}")
    else:
        print("  OK - every stream has exactly n_frames entries in all "
              f"{total} episodes")
    if missing_dirs:
        print("  streams absent (not an error if never recorded):")
        for name, n in missing_dirs.items():
            print(f"    {name}: absent in {n}/{total} episodes")

    # ---- 2. ZED -> wrist sync -------------------------------------------
    print("\n" + "-" * 78)
    print("2. ZED -> WRIST CAMERA SYNC (cross-clock, separate bag)")
    print("-" * 78)
    w_avail = [(c, r) for c, _, r in flat
               if _g(r, "metrics", "wrist", "available")]
    print(f"  wrist available: {len(w_avail)}/{total}")
    means = [_g(r, "metrics", "wrist", "gap_stats", "mean") for _, r in w_avail]
    maxs = [_g(r, "metrics", "wrist", "gap_stats", "max") for _, r in w_avail]
    print(f"  mean gap : {fmt_stat(means, 'ms')}")
    print(f"  max gap  : {fmt_stat(maxs, 'ms')}")
    bad_mean = [(c, r["episode"], m) for (c, r), m in zip(w_avail, means)
                if m is not None and m > WRIST_MEAN_MS]
    bad_max = [(c, r["episode"], m) for (c, r), m in zip(w_avail, maxs)
               if m is not None and m > WRIST_MAX_MS]
    print(f"  over {WRIST_MEAN_MS}ms mean: {len(bad_mean)}/{len(w_avail)}")
    for c, ep, m in bad_mean[:10]:
        print(f"    {c}/{ep}: {m:.1f}ms")
    print(f"  over {WRIST_MAX_MS}ms max : {len(bad_max)}/{len(w_avail)}"
          f"   (one full wrist period - pairing may flip)")
    for c, ep, m in bad_max[:10]:
        print(f"    {c}/{ep}: {m:.1f}ms")

    # ---- 3. lead / trimming --------------------------------------------
    print("\n" + "-" * 78)
    print("3. BAG START LEAD + TRIMMING (stale frames at episode start)")
    print("-" * 78)
    leads = [_g(r, "metrics", "wrist", "lead_ms") for _, r in w_avail]
    print(f"  lead     : {fmt_stat(leads, 'ms')}  (positive = wrist began "
          f"late)")
    stale = [(c, r["episode"], _g(r, "metrics", "wrist", "n_stale_start", default=0),
              _g(r, "metrics", "wrist", "trimmed_frames", default=0))
             for c, r in w_avail]
    untrimmed = [x for x in stale if x[2] > 0]
    trimmed_n = [x[3] for x in stale]
    print(f"  trimmed  : {fmt_stat(trimmed_n, ' frames')}")
    if untrimmed:
        print(f"  !! {len(untrimmed)} episode(s) with UNTRIMMED stale lead "
              f"frames - start of episode shows a pre-motion wrist image:")
        for c, ep, n, t in untrimmed[:15]:
            print(f"    {c}/{ep}: {n} stale, {t} trimmed")
    else:
        print("  OK - no untrimmed stale lead frames")

    # ---- 4. duplicate wrist images -------------------------------------
    print("\n" + "-" * 78)
    print("4. BYTE-IDENTICAL CONSECUTIVE WRIST IMAGES (clock-free check)")
    print("-" * 78)
    dups = [(c, r["episode"],
             len(_g(r, "metrics", "wrist", "duplicate_pairs", default=[])),
             _g(r, "metrics", "wrist", "n_images", default=0),
             _g(r, "metrics", "wrist", "n_unique_images"))
            for c, _, r in flat]
    withdup = [x for x in dups if x[2] > 0]
    if withdup:
        tot = sum(x[2] for x in withdup)
        print(f"  !! {len(withdup)}/{total} episode(s), {tot} duplicate "
              f"pair(s) total - a repeated frame means the image did not "
              f"advance with the state:")
        for c, ep, n, ni, nu in sorted(withdup, key=lambda x: -x[2])[:15]:
            print(f"    {c}/{ep}: {n} pair(s)  ({nu}/{ni} unique)")
    else:
        print(f"  OK - zero duplicate pairs across all {total} episodes")

    # ---- 5. ZED -> ROS state sync --------------------------------------
    print("\n" + "-" * 78)
    print("5. ZED -> ROS STATE SYNC (proprioception vs camera clock)")
    print("-" * 78)
    s_means = [_g(r, "metrics", "state_sync", "gap_stats", "mean")
               for _, _, r in flat]
    s_maxs = [_g(r, "metrics", "state_sync", "gap_stats", "max")
              for _, _, r in flat]
    print(f"  mean gap : {fmt_stat(s_means, 'ms')}")
    print(f"  max gap  : {fmt_stat(s_maxs, 'ms')}")
    bad_s = [(c, r["episode"], m) for (c, _, r), m in zip(flat, s_means)
             if m is not None and m > STATE_MEAN_MS]
    print(f"  over {STATE_MEAN_MS}ms mean : {len(bad_s)}/{total}")
    for c, ep, m in bad_s[:10]:
        print(f"    {c}/{ep}: {m:.2f}ms")

    # ---- 6. depth vs rgb ------------------------------------------------
    print("\n" + "-" * 78)
    print("6. DEPTH <-> RGB PIXEL ALIGNMENT")
    print("-" * 78)
    d_avail = [(c, r) for c, _, r in flat
               if _g(r, "metrics", "depth", "available")]
    print(f"  depth available: {len(d_avail)}/{total}")
    shape_pairs = Counter()
    bad_align = []
    dtypes = Counter()
    for c, r in d_avail:
        d = _g(r, "metrics", "depth", default={})
        shape_pairs[(tuple(d.get("shape") or []),
                     tuple(d.get("rgb_shape") or []))] += 1
        dtypes[d.get("dtype")] += 1
        if d.get("shape_matches_rgb") is False:
            bad_align.append((c, r["episode"], d.get("shape"),
                              d.get("rgb_shape")))
    for (ds, rs), n in shape_pairs.items():
        flag = "OK" if ds == rs else "MISMATCH"
        print(f"    depth {list(ds)} vs rgb {list(rs)}: {n} episode(s) "
              f"[{flag}]")
    print(f"    dtypes: {dict(dtypes)}")
    if bad_align:
        print(f"  !! {len(bad_align)} episode(s) where depth is NOT "
              f"pixel-aligned with its RGB")
    inv_roi = [_g(r, "metrics", "depth", "invalid_pct") for _, r in d_avail]
    inv_full = [_g(r, "metrics", "depth", "invalid_pct_full")
                for _, r in d_avail]
    print(f"  invalid in roi : {fmt_stat(inv_roi, '%')}")
    print(f"  invalid full   : {fmt_stat(inv_full, '%')}")
    if all(v is None for v in inv_full):
        print("    (invalid_pct_full absent -> these reports predate the "
              "ROI change; re-run QA to refresh)")

    # ---- 7. time base --------------------------------------------------
    print("\n" + "-" * 78)
    print("7. TIME BASE: dropped frames + frozen state rows")
    print("-" * 78)
    drops = [(c, r["episode"], _g(r, "metrics", "timeline", "dropped_frames",
                                  default=0)) for c, _, r in flat]
    withdrop = [x for x in drops if x[2] > 0]
    print(f"  episodes with dropped ZED frames: {len(withdrop)}/{total}")
    for c, ep, n in sorted(withdrop, key=lambda x: -x[2])[:12]:
        print(f"    {c}/{ep}: {n} frame(s)")
    periods = [_g(r, "metrics", "timeline", "median_period_ms")
               for _, _, r in flat]
    fps = [_g(r, "metrics", "timeline", "effective_fps") for _, _, r in flat]
    print(f"  median period : {fmt_stat(periods, 'ms')}")
    print(f"  effective fps : {fmt_stat(fps)}")
    per_coll_fps = defaultdict(list)
    for c, _, r in flat:
        v = _g(r, "metrics", "timeline", "effective_fps")
        if v:
            per_coll_fps[c].append(v)
    for c, v in per_coll_fps.items():
        print(f"    {c:<46} fps med={np.median(v):.2f}")

    frozen = []
    for c, _, r in flat:
        fz = _g(r, "metrics", "state", "frozen_frames", default=0)
        nf = _g(r, "metrics", "state", "n_frames", default=0)
        if nf and fz:
            frozen.append((c, r["episode"], fz, nf, 100.0 * fz / nf))
    print(f"  episodes with frozen state rows: {len(frozen)}/{total}")
    for c, ep, fz, nf, pct in sorted(frozen, key=lambda x: -x[4])[:12]:
        print(f"    {c}/{ep}: {fz}/{nf} ({pct:.1f}%)")

    # ---- 8. state dim consistency --------------------------------------
    print("\n" + "-" * 78)
    print("8. STATE VECTOR SHAPE (must be identical across all episodes)")
    print("-" * 78)
    dims = Counter(_g(r, "metrics", "state", "dim") for _, _, r in flat)
    for dim, n in dims.items():
        print(f"    dim={dim}: {n} episode(s)")
    if len(dims) > 1:
        print("  !! MIXED state dimensions - the builder would concatenate "
              "incompatible episodes")
    nan_inf = [(c, r["episode"]) for c, _, r in flat
               if _g(r, "metrics", "state", "has_nan")
               or _g(r, "metrics", "state", "has_inf")]
    print(f"  NaN/Inf in state: {len(nan_inf)} episode(s)")
    for c, ep in nan_inf[:10]:
        print(f"    {c}/{ep}")

    # ---- 9. recurring findings -----------------------------------------
    print("\n" + "-" * 78)
    print("9. RECURRING FINDINGS (systematic if on most episodes)")
    print("-" * 78)

    def shape(text):
        out, prev = [], False
        for ch in text:
            if ch.isdigit() or (ch == "." and prev):
                if not prev:
                    out.append("N")
                prev = True
            else:
                out.append(ch)
                prev = False
        return "".join(out)

    tally = defaultdict(list)
    for c, _, r in flat:
        for sev in ("fails", "warns"):
            for it in r.get(sev) or []:
                tally[(sev.upper()[:-1], shape(it))].append(f"{c}/{r['episode']}")
    for (sev, key), eps in sorted(tally.items(), key=lambda kv: -len(kv[1])):
        print(f"  [{sev}] {len(eps)}/{total}: {key[:150]}")
    if not tally:
        print("  none")
    print()


if __name__ == "__main__":
    main()