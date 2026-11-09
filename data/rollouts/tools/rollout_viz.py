#!/usr/bin/env python3
"""Inspect logged policy rollouts.

One entrypoint with four subcommands, because the previous five scripts each
re-derived the same loading and filter logic and the copies had already drifted
apart.

    table     per-rollout numbers plus aggregates, on stdout
    dash      one self-contained HTML page (overview / per rollout / table)
    rerun     3D + timeline for a single rollout, in the Rerun viewer
    filters   replay the executor's filter chain over one rollout

Examples:
    python data/rollout_logs/rollout_viz.py table
    python data/rollout_logs/rollout_viz.py dash --open
    python data/rollout_logs/rollout_viz.py rerun rollout_010
    python data/rollout_logs/rollout_viz.py rerun 10 --video
    python data/rollout_logs/rollout_viz.py rerun --all --no-pc --save all.rrd
    python data/rollout_logs/rollout_viz.py filters rollout_010

The log root defaults to $ROLLOUT_LOG_DIR, the same variable log_rollout.py
uses, so the logger and the viewer cannot end up pointing at different
directories.
"""
import argparse
import pathlib
import sys

# Importable both as `python rollout_viz.py` and as part of a package.
_this_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))
if str(_this_dir.parent) not in sys.path:
    sys.path.insert(0, str(_this_dir.parent))

from viz.data import (DEFAULT_ROOT, flag_outliers, load_all,  # noqa: E402
                      pick_group, report_dir)
from viz.report import (hand_filter_report, pose_filter_report,  # noqa: E402
                        summary_table)


def _resolve(rollouts, name):
    """Accept 'rollout_010', '010' or '10' - the directory names are padded
    but nobody types the padding."""
    by_name = {r.name: r for r in rollouts}
    if name in by_name:
        return by_name[name]
    for r in rollouts:
        if r.name.replace("rollout_", "").lstrip("0") == name.lstrip("0"):
            return r
    raise SystemExit(f"{name!r} not found. Available: "
                     f"{', '.join(r.name for r in rollouts)}")


def cmd_table(args):
    groups = load_all(args.root, args.ckpt)
    if not groups:
        raise SystemExit(f"no rollouts under {args.root}")
    for ck, rs in groups.items():
        ok = [r for r in rs if r.ok]
        print(f"\n{'=' * 104}")
        print(f"checkpoint: {ck}   ({len(ok)} loadable of {len(rs)})")
        print(f"{'=' * 104}\n")
        for r in rs:
            if not r.ok:
                print(f"  SKIPPED {r.name}: {r.error}")
        if ok:
            print(summary_table(ok))


def cmd_dash(args):
    from viz.dashboard import build_page

    groups = load_all(args.root, args.ckpt)
    if not groups:
        raise SystemExit(f"no rollouts under {args.root}")

    written = []
    for ck, rs in groups.items():
        ok = [r for r in rs if r.ok]
        for r in rs:
            if not r.ok:
                print(f"  skipped {r.name}: {r.error}")
        if not ok:
            continue

        flagged = {} if args.include_flagged else flag_outliers(ok)
        for name, why in flagged.items():
            print(f"  flagged {name}: {why}")

        out = (pathlib.Path(args.out) if args.out
               else report_dir(args.root) / f"{ck}.html")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_page(ok, flagged, ck), encoding="utf-8")
        size = out.stat().st_size / 1e6
        print(f"{ck}: {len(ok)} rollouts -> {out} ({size:.1f} MB)")
        written.append(out)

    if args.open and written:
        import webbrowser
        webbrowser.open(written[0].resolve().as_uri())


def cmd_rerun(args):
    import rerun as rr

    from viz import pinch
    from viz.rerun_view import blueprint, log_rollout

    ck, rollouts = pick_group(args.root, args.ckpt)
    if not rollouts:
        raise SystemExit(f"no loadable rollouts for {ck}")

    if args.all:
        chosen = rollouts
    elif args.rollout:
        chosen = [_resolve(rollouts, n) for n in args.rollout]
    else:
        chosen = [rollouts[-1]]

    print(f"checkpoint: {ck}")

    pm = None
    if not args.no_pinch:
        pm, note = pinch.try_load(args.urdf)
        print(f"  {note}")

    rr.init("rollouts", spawn=args.save is None)

    if len(chosen) == 1:
        r = chosen[0]
        print(f"rollout   : {r.name}  {r.notes}")
        rr.send_blueprint(blueprint())
        log_rollout(r, with_pc=args.with_pc, with_video=args.video,
                    max_frames=args.max_frames, pinch_model=pm)
    else:
        entries = []
        for r in chosen:
            # The tab label carries the notes because that is the independent
            # variable of this experiment (pallet height, foil thickness),
            # not an afterthought.
            entries.append((r.name, f"{r.stats()['short']} {r.notes}"[:48]))
            print(f"  {r.name}  {r.notes}")
        rr.send_blueprint(blueprint(entries))
        for r in chosen:
            log_rollout(r, ns=r.name, with_pc=args.with_pc,
                        with_video=args.video, max_frames=args.max_frames,
                        pinch_model=pm)

    if args.save:
        rr.save(args.save)
        out = pathlib.Path(args.save).resolve()
        size = out.stat().st_size / 1e6
        # Absolute path, and said explicitly: the viewer is a GUI, so it
        # cannot open inside the container (no X server). Printing a
        # relative path here invited exactly that attempt.
        print(f"wrote {out} ({size:.0f} MB)")
        print(f"open it ON THE HOST with: rerun {out}")


def cmd_filters(args):
    ck, rollouts = pick_group(args.root, args.ckpt)
    r = _resolve(rollouts, args.rollout) if args.rollout else rollouts[-1]
    print(f"checkpoint : {ck}")
    print(f"rollout    : {r.name}  {r.notes}\n")
    if args.side in ("hand", "both"):
        print(hand_filter_report(r))
        print()
    if args.side in ("pose", "both"):
        print(pose_filter_report(r))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help="rollout log root (default: $ROLLOUT_LOG_DIR)")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint subdirectory (default: all / the first)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("table", help="per-rollout numbers on stdout")
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("dash", help="self-contained HTML dashboard")
    p.add_argument("--out", default=None,
                   help="output file (default: <root>/reports/<ckpt>.html)")
    p.add_argument("--open", action="store_true",
                   help="open the page in a browser when done")
    p.add_argument("--include-flagged", action="store_true",
                   help="include non-grasp rollouts in the FITS as well as "
                        "the plots (off by default; the reason is on the "
                        "page)")
    p.set_defaults(func=cmd_dash)

    p = sub.add_parser("rerun", help="3D + timeline in the Rerun viewer")
    p.add_argument("rollout", nargs="*",
                   help="rollout names, e.g. rollout_010 or just 10. "
                        "Default: the last one. Several get one tab each.")
    p.add_argument("--all", action="store_true",
                   help="every rollout as its own tab. With point clouds "
                        "this is ~6 MB per rollout, so combine with --no-pc "
                        "unless you need the geometry.")
    p.add_argument("--no-pc", dest="with_pc", action="store_false",
                   help="skip point clouds (much smaller, much faster)")
    p.add_argument("--video", action="store_true",
                   help="also load the wrist camera from the rollout's "
                        "rosbag, on the same timeline. Needs an MCAP reader "
                        "(pip install mcap mcap-ros2-support); says so and "
                        "carries on without images if none is found.")
    p.add_argument("--urdf", default=None,
                   help="hand URDF for the pinch width (default: the "
                        "installed dg5f_right.urdf, or $DG5F_URDF). The file "
                        "sits in a different place in the container than in "
                        "a source checkout, hence the override.")
    p.add_argument("--no-pinch", action="store_true",
                   help="skip the URDF forward kinematics")
    p.add_argument("--max-frames", type=int, default=200,
                   help="camera frames per rollout, uniformly spread over "
                        "the run (default 200; 0 = all, which for a 2402 "
                        "frame bag at 1280x720 is ~6.6 GB in RAM)")
    p.add_argument("--save", default=None,
                   help="write an .rrd instead of spawning the viewer - use "
                        "this when running headless, then open it with "
                        "`rerun out.rrd` on the host")
    p.set_defaults(func=cmd_rerun)

    p = sub.add_parser("filters", help="replay the executor's filter chain")
    p.add_argument("rollout", nargs="?", default=None)
    p.add_argument("--side", choices=["hand", "pose", "both"], default="both")
    p.set_defaults(func=cmd_filters)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()