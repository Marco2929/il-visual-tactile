#!/usr/bin/env python3
"""Compare ROI vs full-frame depth invalid fractions across episodes.

Confirms the ROI change actually resolves the 56/56 warning instead of just
moving the number around.

Usage (in the container):
    python data/record_extractor/check_depth_roi.py <collection_dir> [n_eps]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qa_report import THRESHOLDS, _check_depth  # noqa: E402


def main():
    root = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    roi = THRESHOLDS["depth_roi"]
    thr = THRESHOLDS["max_depth_invalid_pct"]

    eps = sorted(p for p in root.iterdir()
                 if p.is_dir() and p.name.startswith("episode_"))[:limit]

    print(f"roi={roi}  threshold={thr}%  episodes={len(eps)}\n")
    print(f"{'episode':<24} {'shape':>12} {'roi%':>8} {'full%':>8} "
          f"{'median_mm':>10}  verdict")
    print("-" * 78)

    n_warn_roi = n_warn_full = n_avail = 0
    for ep in eps:
        d = _check_depth(str(ep), roi=roi)
        if not d["available"]:
            print(f"{ep.name:<24} {'n/a':>12} - no depth")
            continue
        n_avail += 1
        roi_pct = d["invalid_pct"]
        full_pct = d["invalid_pct_full"]
        warn_roi = roi_pct is not None and roi_pct > thr
        warn_full = full_pct is not None and full_pct > thr
        n_warn_roi += warn_roi
        n_warn_full += warn_full
        med = d["valid_median_mm"]
        clamp = f" CLAMPED->{d['roi_clamped']}" if d["roi_clamped"] else ""
        print(f"{ep.name:<24} {str(d['shape']):>12} {roi_pct:>8.2f} "
              f"{full_pct:>8.2f} {med if med else 0:>10.0f}"
              f"  {'WARN' if warn_roi else 'ok'}{clamp}")

    print("-" * 78)
    print(f"episodes with depth : {n_avail}")
    print(f"WARN on ROI         : {n_warn_roi}/{n_avail}")
    print(f"WARN on full frame  : {n_warn_full}/{n_avail}   "
          f"(the old behaviour)")


if __name__ == "__main__":
    main()