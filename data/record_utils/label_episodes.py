#!/usr/bin/env python3
"""Label recorded episodes after the fact, and report plan progress.

The recorder derives every CONDITION label (Boxhoehe, Bauteil, Boxposition,
Folienlage) from the plan slot, so there is nothing to type for those. One
thing cannot be derived and is asked here:

    retry   did the operator miss the foil and grab again?

It can be answered during recording, but this tool exists so it does not have
to be - answering it with the episode in front of you beats answering it from
memory between takes.

Usage:
    # what is still open, per block
    python3 label_episodes.py status

    # mark the retry episodes (successes only)
    python3 label_episodes.py retry
    python3 label_episodes.py retry --collection folien_block1

    # repair a collection recorded with the wrong PLAN_BLOCK
    python3 label_episodes.py reslot --collection folien_block1 --block 1

Only successful episodes are asked about: a failed take was aborted rather
than retried, and its slot is still open anyway.

Environment:
    WORKDIR, EPISODES_SUBDIR   where collections live (as in record.py)
"""

import argparse
import os
import sys

# Import as a package so `from . import` inside plan/ resolves, whether this is
# run as a script or imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from record_utils.plan import schedule, store  # noqa: E402

WORKDIR = os.environ.get("WORKDIR", "/root/ros2_ws/data/")
EPISODES_SUBDIR = os.environ.get("EPISODES_SUBDIR", "external")
DEFAULT_BASE = os.path.join(WORKDIR, "episodes", EPISODES_SUBDIR)

RETRY_HELP = "\n".join([
    "  ENTER   saubere Episode, ein Zugriff",
    "  r       Retry drin: Folie verpasst und nochmal zugegriffen",
    "  s       ueberspringen      q  beenden",
])


def _records(base: str, collection):
    if collection:
        root = os.path.join(base, collection)
        if not os.path.isdir(root):
            sys.exit(f"Collection nicht gefunden: {root}")
        return store.scan_episodes(root)
    return store.scan_collections(base)


def _head(i, total, rec):
    lab = rec.data
    return (f"[{i}/{total}] {os.path.basename(rec.episode_dir)}  "
            f"{lab.get('code', '?')}  "
            f"{lab.get('boxhoehe') or '?'} / "
            f"{lab.get('bauteil_label') or '?'} / "
            f"Pos{lab.get('position') or '?'} / "
            f"{lab.get('folienlage') or '?'}")


def cmd_status(args) -> None:
    records = _records(args.base, args.collection)
    if not records:
        sys.exit(f"Keine Episoden mit labels.json unter {args.base}")
    prog = store.build_progress(records)

    print(f"\n{len(records)} gelabelte Episoden unter {args.base}\n")
    print(f"{'Blk':<5}{'Boxhoehe':<20}{'Bauteil':<14}"
          f"{'Slots':>8}{'offen':>7}{'Takes':>7}")
    print("-" * 61)

    total_done = total_takes = 0
    for b in range(1, schedule.NUM_BLOCKS + 1):
        slots = list(schedule.block_slots(b))
        takes = sum(prog.attempt_count(s) for s in slots)
        if not takes and not args.all:
            continue
        done = len([s for s in slots if prog.is_done(s)])
        blk = schedule.BLOCKS[b - 1]
        print(f"{b:<5}{schedule.BOXHOEHEN[blk['boxhoehe']]['label']:<20}"
              f"{schedule.BAUTEILE[blk['bauteil']]['label']:<14}"
              f"{done:>5}/{len(slots):<2}"
              f"{len(slots) - done:>7}{takes:>7}")
        total_done += done
        total_takes += takes

    print("-" * 61)
    print(f"{'Summe':<39}{total_done:>5}/{schedule.TOTAL_SLOTS:<3}"
          f"{schedule.TOTAL_SLOTS - total_done:>6}{total_takes:>7}")

    pending = prog.needing_retry_answer()
    if pending:
        print(f"\n!  {len(pending)} Successes ohne Retry-Angabe"
              f"\n   -> label_episodes.py retry")

    retries = [r for recs in prog.successes.values() for r in recs
               if r.retry is True]
    if retries:
        print(f"\ni  {len(retries)} Episoden mit Retry")

    if prog.unplaced:
        print(f"\n!  {len(prog.unplaced)} Episoden ohne Plan-Slot "
              f"(in_plan=false)")

    multi = [s for s in range(schedule.TOTAL_SLOTS)
             if prog.success_count(s) > 1]
    if multi:
        print(f"\ni  {len(multi)} Slots mit mehreren Successes "
              f"({sum(prog.success_count(s) - 1 for s in multi)} Zusatz-Takes)")
    print()


def cmd_retry(args) -> None:
    prog = store.build_progress(_records(args.base, args.collection))
    pending = prog.needing_retry_answer()
    if not pending:
        print("Alle Successes sind beantwortet.")
        return

    print(f"\n{len(pending)} Episoden offen.")
    print(RETRY_HELP + "\n")

    answered = 0
    for i, rec in enumerate(pending, 1):
        print(_head(i, len(pending), rec))
        while True:
            try:
                raw = input("   ENTER=sauber · r=Retry · s=skip · q=Ende: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print(f"\nAbgebrochen. {answered} Episoden gespeichert.")
                return
            if raw == "q":
                print(f"{answered} Episoden gespeichert.")
                return
            if raw == "s":
                break
            if raw == "?":
                print(RETRY_HELP)
                continue
            try:
                retry = schedule.parse_retry(raw)
            except ValueError as exc:
                print(f"   {exc} - nochmal.")
                continue
            store.set_retry(rec.episode_dir, retry)
            print(f"   -> {schedule.retry_text(retry)}")
            answered += 1
            break

    print(f"\nFertig. {answered} von {len(pending)} beantwortet.")


def cmd_reslot(args) -> None:
    """Re-derive every condition label for a collection's episodes.

    For a collection recorded with the wrong (or no) PLAN_BLOCK: the slot
    numbers are wrong, so Boxhoehe/Bauteil/Position/Folienlage are all wrong
    too. The ORDER of episodes is trusted - takes of one slot are consecutive
    - and take numbers within a slot are preserved, so the mapping is rebuilt
    rather than guessed per episode.
    """
    records = _records(args.base, args.collection)
    if not records:
        sys.exit("Keine Episoden gefunden.")

    slots = list(schedule.block_slots(args.block))
    # Group by the slot each episode currently claims, preserving order, so
    # the i-th distinct recorded slot maps onto the i-th slot of the target
    # block. Collapsing to "one episode per slot" would destroy the extra
    # takes this very design was built to allow.
    groups = []
    seen = {}
    for rec in records:
        key = rec.data.get("slot")
        if key not in seen:
            seen[key] = len(groups)
            groups.append([])
        groups[seen[key]].append(rec)

    if len(groups) > len(slots):
        sys.exit(f"{len(groups)} verschiedene Slots, aber Block {args.block} "
                 f"hat nur {len(slots)} - nichts geaendert.")

    print(f"\n{schedule.block_title(args.block)}: {len(groups)} Slots, "
          f"{len(records)} Episoden\n")
    for gi, group in enumerate(groups):
        print(f"  {schedule.slot_code(slots[gi]):<20} <- "
              f"{', '.join(os.path.basename(r.episode_dir) for r in group)}")
    if input("\nFortfahren? (yes/no): ").strip().lower() != "yes":
        print("Abgebrochen.")
        return

    for gi, group in enumerate(groups):
        slot = slots[gi]
        for attempt, rec in enumerate(group, 1):
            fresh = schedule.labels_for(slot)
            # retry is NOT touched: it is an observation, and re-deriving the
            # conditions says nothing about what happened during the take.
            fresh.update({"attempt": attempt, "plan_block": args.block})
            store.update_labels(rec.episode_dir, **fresh)
    print(f"{len(records)} Episoden neu gelabelt.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help=f"Basisverzeichnis (default: {DEFAULT_BASE})")
    ap.add_argument("--collection", default=None,
                    help="nur diese Collection, sonst alle")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="Fortschritt pro Block")
    s.add_argument("--all", action="store_true",
                   help="auch Bloecke ohne Aufnahmen zeigen")
    s.set_defaults(func=cmd_status)

    r = sub.add_parser("retry", help="Retry-Episoden markieren")
    r.set_defaults(func=cmd_retry)

    rs = sub.add_parser("reslot", help="Condition-Labels neu ableiten")
    rs.add_argument("--block", type=int, required=True,
                    choices=range(1, schedule.NUM_BLOCKS + 1))
    rs.set_defaults(func=cmd_reslot)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()