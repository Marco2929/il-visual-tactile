#!/usr/bin/env python3
"""Attach a human verdict to recorded rollouts, after the fact.

log_rollout.py records what happened; it cannot know whether it WORKED. That
judgement is manual, it arrives seconds-to-days later, and it must never be
allowed to block or crash the recording process. So it lives here, in a
separate pass over the directories that already exist on disk.

Writes, per rollout:

    <ckpt>/rollout_NNN/outcome.json     the verdict, the comment, the scores

and, per checkpoint:

    <ckpt>/results.jsonl                one object per rollout, append-free
                                        (rebuilt from the outcome.json files)
    <ckpt>/index.csv                    the same, flat, for pandas/Excel
    <ckpt>/REPORT.md                    success rate + failure-mode tally

The csv/jsonl/md are DERIVED - outcome.json is the only thing you edit, by
hand or through this script. Rebuilding is therefore always safe.

USAGE

    # walk every rollout of a checkpoint that has no verdict yet
    python annotate_rollouts.py --ckpt "epoch=0040-val_action_mse=9.478029"

    # revisit one, or a range, including already-labelled ones
    python annotate_rollouts.py --ckpt <label> --only 3,7,12-15 --force

    # non-interactive, e.g. right after a run while it is fresh
    python annotate_rollouts.py --ckpt <label> --only 3 \
        --verdict fail --failure-mode approach_height \
        --comment "went for the pallet surface, ~3cm low the whole approach" \
        --smoothness 1 --accuracy 0

    # only regenerate the tables, ask nothing
    python annotate_rollouts.py --ckpt <label> --rebuild

INTERACTIVE KEYS
    s / p / f / v   success, partial, fail, void
    Enter           skip this rollout, decide later
    e               open the existing comment in $EDITOR for a long note
    q               stop (everything already answered is saved)

`void` means the run says nothing about the policy - operator error, e-stop,
camera dropout. It is excluded from the success rate rather than counted as a
failure, because "the experiment did not happen" and "the policy failed" are
different facts and averaging them together is how an eval starts lying.
"""
import argparse
import csv
import json
import os
import pathlib
import subprocess
import tempfile
from datetime import datetime

DEFAULT_ROOT = os.environ.get("ROLLOUT_LOG_DIR", "/ros2_ws/data/rollouts")

VERDICTS = ("success", "partial", "fail", "void")
VERDICT_KEYS = {"s": "success", "p": "partial", "f": "fail", "v": "void"}

# Fixed vocabulary. The free-text comment stays the place for nuance, but
# "how often did the grasp slip across 40 rollouts" is only answerable if the
# category is a token and not a sentence retyped slightly differently.
FAILURE_MODES = (
    "none",
    "approach_height",      # reached for the wrong z
    "approach_xy",          # wrong lateral target
    "grasp_miss",           # closed on nothing
    "grasp_slip",           # grasped it, lost it
    "premature_lift",       # lifted before the hand had closed
    "no_lift",              # grasped, never lifted
    "stalled",              # stopped making progress
    "oscillation",          # chattering / jitter
    "collision",            # hit pallet, fixture or itself
    "joint_limit",          # driven into a limit or protective stop
    "buffer_starved",       # chunks arrived late, executor ran dry
    "operator_abort",       # gate closed by hand
    "other",
)

# Qualitative impression, 0 = bad, 1 = acceptable, 2 = clean. Three levels
# because a human cannot reproduce a 1-10 scale across two sessions, and a
# score you cannot reproduce is noise dressed up as data.
BEHAVIOR_KEYS = ("smoothness", "accuracy", "speed", "recovery")

SUCCESS_OF = {"success": True, "partial": False, "fail": False, "void": None}


# ----------------------------------------------------------------------
# disk layout
# ----------------------------------------------------------------------
def rollout_dirs(ckpt_dir: pathlib.Path):
    def key(p):
        try:
            return int(p.name.split("_")[1])
        except (IndexError, ValueError):
            return 1 << 30
    return sorted((p for p in ckpt_dir.glob("rollout_*") if p.is_dir()),
                  key=key)


def is_complete(d: pathlib.Path) -> bool:
    """summary.json is written by log_rollout.py at gate close, so its
    presence is the marker that a rollout is finished.

    This matters because the intended workflow is to leave the logger running
    for a whole session and label in between runs: without this check, the
    rollout you are still recording would be offered for annotation, and the
    verdict would land on a directory that is about to grow.
    """
    return (d / "summary.json").is_file()


def rollout_index(d: pathlib.Path):
    try:
        return int(d.name.split("_")[1])
    except (IndexError, ValueError):
        return None


def parse_only(spec: str):
    """'3,7,12-15' -> {3,7,12,13,14,15}. None means 'all'."""
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def read_json(path: pathlib.Path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def blank_outcome(d: pathlib.Path, ckpt: str) -> dict:
    """A verdict-less outcome, pre-filled with what the recording knows.

    Context is copied in rather than referenced so that outcome.json can be
    read, sorted and reviewed on its own - including by someone who does not
    have the 6 MB of npz next to it.
    """
    meta = read_json(d / "meta.json")
    summary = read_json(d / "summary.json")
    return {
        "rollout": d.name,
        "rollout_index": rollout_index(d),
        "checkpoint": meta.get("checkpoint", ckpt),
        # verdict is what a human wrote; success is what a plot reads.
        # None (not False) when unlabelled: "nobody looked" must never
        # silently become "it failed".
        "verdict": None,
        "success": None,
        "failure_mode": None,
        "comment": "",
        "behavior": {k: None for k in BEHAVIOR_KEYS},
        "tags": [],
        # Setup context, from --notes at record time.
        "conditions": meta.get("notes", "").strip(),
        "duration_s": summary.get("duration_s"),
        "stop_reason": summary.get("stop_reason"),
        "num_inference_steps": meta.get("num_inference_steps"),
        "started_wall": meta.get("started_wall"),
        "chunks": (summary.get("counts") or {}).get("chunks"),
        "annotator": None,
        "annotated_wall": None,
    }


# ----------------------------------------------------------------------
# interactive pass
# ----------------------------------------------------------------------
def edit_in_editor(initial: str) -> str:
    """For comments longer than one line. Falls back to input() if $EDITOR
    is unusable - a missing editor must not cost the verdict."""
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
    try:
        with tempfile.NamedTemporaryFile("w+", suffix=".md",
                                         delete=False) as fh:
            fh.write(initial)
            path = fh.name
        subprocess.run([editor, path], check=True)
        with open(path) as f:
            text = f.read().strip()
        os.unlink(path)
        return text
    except Exception as e:
        print(f"  (editor '{editor}' failed: {e}; type one line instead)")
        return input("  comment > ").strip()


def ask_score(label: str, current):
    shown = "-" if current is None else current
    while True:
        raw = input(f"  {label:<11} 0=bad 1=ok 2=clean "
                    f"[{shown}] > ").strip()
        if raw == "":
            return current
        if raw in ("0", "1", "2"):
            return int(raw)
        if raw == "-":
            return None
        print("  enter 0, 1, 2, '-' to clear, or Enter to keep.")


def ask_failure_mode(current):
    print("  failure modes:")
    for i, m in enumerate(FAILURE_MODES):
        print(f"    {i:>2}  {m}")
    while True:
        raw = input(f"  failure_mode (number or name) "
                    f"[{current or '-'}] > ").strip()
        if raw == "":
            return current
        if raw.isdigit() and int(raw) < len(FAILURE_MODES):
            return FAILURE_MODES[int(raw)]
        if raw in FAILURE_MODES:
            return raw
        # Accepted anyway, but flagged: a typo'd mode silently stops
        # aggregating, which is worse than a loud warning.
        print(f"  '{raw}' is not in the vocabulary - keeping it, but it will "
              f"show up as its own category in REPORT.md. Enter to cancel.")
        confirm = input("  use it anyway? [y/N] > ").strip().lower()
        if confirm == "y":
            return raw


def annotate_one(d: pathlib.Path, ckpt: str, args) -> str:
    """Returns 'saved', 'skipped' or 'quit'."""
    path = d / "outcome.json"
    outcome = read_json(path, blank_outcome(d, ckpt))
    # Refresh the recorded context even on re-annotation: summary.json is
    # ground truth, a stale copy in outcome.json is not.
    outcome = {**blank_outcome(d, ckpt),
               **{k: v for k, v in outcome.items() if v not in (None, "", [])}}

    print(f"\n{'=' * 64}")
    print(f"  {ckpt}  /  {d.name}")
    print(f"  {outcome.get('duration_s')} s, "
          f"{outcome.get('chunks')} chunks, "
          f"stop={outcome.get('stop_reason')}, "
          f"started {outcome.get('started_wall')}")
    if outcome.get("conditions"):
        print(f"  setup: {outcome['conditions']}")
    if outcome.get("verdict"):
        print(f"  CURRENT: {outcome['verdict']} / "
              f"{outcome.get('failure_mode')} - {outcome.get('comment')}")

    while True:
        raw = input("  verdict [s]uccess [p]artial [f]ail [v]oid, "
                    "Enter=skip, q=quit > ").strip().lower()
        if raw == "":
            return "skipped"
        if raw == "q":
            return "quit"
        verdict = VERDICT_KEYS.get(raw, raw if raw in VERDICTS else None)
        if verdict:
            break
        print("  one of s/p/f/v, or Enter to skip.")

    outcome["verdict"] = verdict
    outcome["success"] = SUCCESS_OF[verdict]

    if verdict == "success":
        outcome["failure_mode"] = "none"
    else:
        outcome["failure_mode"] = ask_failure_mode(outcome.get("failure_mode"))

    print("  comment (what the policy actually did). "
          "'e' opens $EDITOR, Enter keeps.")
    line = input(f"  [{outcome.get('comment') or '-'}] > ").strip()
    if line == "e":
        outcome["comment"] = edit_in_editor(outcome.get("comment", ""))
    elif line:
        outcome["comment"] = line

    if not args.quick:
        behavior = dict(outcome.get("behavior") or {})
        for k in BEHAVIOR_KEYS:
            behavior[k] = ask_score(k, behavior.get(k))
        outcome["behavior"] = behavior

        tags = input(f"  tags, comma separated "
                     f"[{','.join(outcome.get('tags') or []) or '-'}] > ")
        if tags.strip():
            outcome["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

    outcome["annotator"] = args.annotator
    outcome["annotated_wall"] = datetime.now().isoformat()

    with open(path, "w") as f:
        json.dump(outcome, f, indent=2)
    print(f"  -> {path}")
    return "saved"


def apply_flags(d: pathlib.Path, ckpt: str, args) -> bool:
    """Non-interactive path: set exactly the fields given on the CLI."""
    path = d / "outcome.json"
    outcome = read_json(path, blank_outcome(d, ckpt))
    outcome = {**blank_outcome(d, ckpt),
               **{k: v for k, v in outcome.items() if v not in (None, "", [])}}

    if args.verdict:
        outcome["verdict"] = args.verdict
        outcome["success"] = SUCCESS_OF[args.verdict]
        if args.verdict == "success" and not args.failure_mode:
            outcome["failure_mode"] = "none"
    if args.failure_mode:
        outcome["failure_mode"] = args.failure_mode
    if args.comment is not None:
        outcome["comment"] = args.comment
    if args.tags:
        outcome["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    behavior = dict(outcome.get("behavior") or {})
    for k in BEHAVIOR_KEYS:
        v = getattr(args, k)
        if v is not None:
            behavior[k] = v
    outcome["behavior"] = behavior
    outcome["annotator"] = args.annotator
    outcome["annotated_wall"] = datetime.now().isoformat()

    with open(path, "w") as f:
        json.dump(outcome, f, indent=2)
    print(f"{d.name}: {outcome['verdict']} / {outcome['failure_mode']}")
    return True


# ----------------------------------------------------------------------
# derived tables
# ----------------------------------------------------------------------
CSV_COLUMNS = [
    "rollout", "rollout_index", "verdict", "success", "failure_mode",
    "duration_s", "chunks", "num_inference_steps", "stop_reason",
    *(f"behavior_{k}" for k in BEHAVIOR_KEYS),
    "tags", "conditions", "comment", "annotator", "annotated_wall",
    "started_wall",
]


def flatten(outcome: dict) -> dict:
    behavior = outcome.get("behavior") or {}
    row = {c: outcome.get(c) for c in CSV_COLUMNS}
    for k in BEHAVIOR_KEYS:
        row[f"behavior_{k}"] = behavior.get(k)
    row["tags"] = ",".join(outcome.get("tags") or [])
    # Newlines in a csv cell are legal but make the file unreadable with
    # head/grep, which is how it will actually be looked at.
    for k in ("comment", "conditions"):
        if isinstance(row.get(k), str):
            row[k] = row[k].replace("\n", " ").strip()
    return row


def rebuild(ckpt_dir: pathlib.Path, ckpt: str):
    outcomes = []
    for d in rollout_dirs(ckpt_dir):
        if not is_complete(d):
            # Still recording - counting it would make the denominator jump
            # around between rebuilds.
            continue
        p = d / "outcome.json"
        # Rollouts with no outcome.json are listed as unlabelled rather than
        # omitted, so the denominator of the success rate stays honest.
        outcomes.append(read_json(p) if p.is_file()
                        else blank_outcome(d, ckpt))

    with open(ckpt_dir / "results.jsonl", "w") as f:
        for o in outcomes:
            f.write(json.dumps(o) + "\n")

    with open(ckpt_dir / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS,
                           extrasaction="ignore")
        w.writeheader()
        for o in outcomes:
            w.writerow(flatten(o))

    write_report(ckpt_dir, ckpt, outcomes)
    return outcomes


def write_report(ckpt_dir: pathlib.Path, ckpt: str, outcomes):
    total = len(outcomes)
    by_verdict = {v: 0 for v in VERDICTS}
    unlabelled = 0
    for o in outcomes:
        v = o.get("verdict")
        if v in by_verdict:
            by_verdict[v] += 1
        else:
            unlabelled += 1

    # Denominator excludes void and unlabelled: the rate is over runs that
    # actually tested the policy and that someone actually judged.
    scored = by_verdict["success"] + by_verdict["partial"] + by_verdict["fail"]
    rate = (by_verdict["success"] / scored * 100.0) if scored else float("nan")

    modes = {}
    for o in outcomes:
        if o.get("verdict") in ("partial", "fail"):
            modes[o.get("failure_mode") or "unspecified"] = \
                modes.get(o.get("failure_mode") or "unspecified", 0) + 1

    means = {}
    for k in BEHAVIOR_KEYS:
        vals = [(o.get("behavior") or {}).get(k) for o in outcomes]
        vals = [v for v in vals if isinstance(v, (int, float))]
        means[k] = sum(vals) / len(vals) if vals else None

    lines = [
        f"# Rollout report - `{ckpt}`",
        "",
        f"_generated {datetime.now().isoformat(timespec='seconds')} "
        f"by annotate_rollouts.py; do not edit, edit the outcome.json files_",
        "",
        "## Verdicts",
        "",
        f"- rollouts recorded: **{total}**",
        f"- judged (success/partial/fail): **{scored}**",
        f"- success: **{by_verdict['success']}**, "
        f"partial: {by_verdict['partial']}, fail: {by_verdict['fail']}",
        f"- void (not a valid test): {by_verdict['void']}, "
        f"unlabelled: {unlabelled}",
        "",
        f"**Success rate: {rate:.1f}%** "
        f"({by_verdict['success']}/{scored}) - "
        f"void and unlabelled excluded from the denominator.",
        "",
        "## Failure modes",
        "",
    ]
    if modes:
        lines += ["| mode | n |", "|---|---|"]
        lines += [f"| {m} | {n} |"
                  for m, n in sorted(modes.items(),
                                     key=lambda kv: -kv[1])]
    else:
        lines.append("_none recorded_")

    lines += ["", "## Behaviour (0 bad / 1 ok / 2 clean)", "",
              "| aspect | mean |", "|---|---|"]
    for k, v in means.items():
        lines.append(f"| {k} | {'-' if v is None else f'{v:.2f}'} |")

    lines += ["", "## Per rollout", "",
              "| rollout | verdict | failure mode | s | dur | comment |",
              "|---|---|---|---|---|---|"]
    for o in outcomes:
        b = o.get("behavior") or {}
        comment = (o.get("comment") or "").replace("\n", " ").replace("|", "/")
        lines.append(
            f"| {o.get('rollout')} | {o.get('verdict') or '-'} | "
            f"{o.get('failure_mode') or '-'} | "
            f"{b.get('smoothness') if b.get('smoothness') is not None else '-'} | "
            f"{o.get('duration_s') or '-'} | {comment} |")

    with open(ckpt_dir / "REPORT.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nsuccess {by_verdict['success']}/{scored} "
          f"({rate:.1f}%), void {by_verdict['void']}, "
          f"unlabelled {unlabelled}")
    print(f"wrote {ckpt_dir / 'results.jsonl'}")
    print(f"wrote {ckpt_dir / 'index.csv'}")
    print(f"wrote {ckpt_dir / 'REPORT.md'}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Attach success/comment/behaviour notes to recorded "
                    "rollouts and rebuild the campaign tables.")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint directory name under --root. Omit to "
                         "list what is available.")
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help=f"rollout log root (default {DEFAULT_ROOT})")
    ap.add_argument("--only", default=None,
                    help="restrict to indices, e.g. '3,7,12-15'")
    ap.add_argument("--force", action="store_true",
                    help="also revisit rollouts that already have a verdict")
    ap.add_argument("--rebuild", action="store_true",
                    help="only regenerate results.jsonl / index.csv / "
                         "REPORT.md, ask nothing")
    ap.add_argument("--quick", action="store_true",
                    help="verdict + comment only, skip the behaviour scores")
    ap.add_argument("--annotator", default=os.environ.get("USER", "unknown"))

    g = ap.add_argument_group("non-interactive (needs --only)")
    g.add_argument("--verdict", choices=VERDICTS)
    g.add_argument("--failure-mode", default=None)
    g.add_argument("--comment", default=None)
    g.add_argument("--tags", default=None, help="comma separated")
    for k in BEHAVIOR_KEYS:
        g.add_argument(f"--{k}", type=int, choices=(0, 1, 2), default=None)

    args = ap.parse_args()
    root = pathlib.Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"no such root: {root}")

    if not args.ckpt:
        print("checkpoints under", root)
        search_dirs = [root]
        if (root / "runs").is_dir():
            search_dirs.append(root / "runs")
        seen = set()
        skip = {"tools", "viz", "analysis", "runs", "__pycache__"}
        for s_dir in search_dirs:
            for d in sorted(s_dir.iterdir()):
                if d.name in seen or d.name in skip:
                    continue
                if d.is_dir() and any(p.is_dir() for p in d.glob("rollout_*")):
                    seen.add(d.name)
                    n = len(rollout_dirs(d))
                    labelled = sum(1 for r in rollout_dirs(d)
                                   if (r / "outcome.json").is_file())
                    print(f"  {d.name}   {labelled}/{n} labelled")
        return

    ckpt_dir = root / args.ckpt
    if not ckpt_dir.is_dir() and (root / "runs" / args.ckpt).is_dir():
        ckpt_dir = root / "runs" / args.ckpt
    if not ckpt_dir.is_dir():
        raise SystemExit(f"no such checkpoint directory: {ckpt_dir}")

    if args.rebuild:
        rebuild(ckpt_dir, args.ckpt)
        return

    only = parse_only(args.only)
    targets = [d for d in rollout_dirs(ckpt_dir)
               if only is None or rollout_index(d) in only]
    in_flight = [d for d in targets if not is_complete(d)]
    if in_flight:
        print(f"skipping {', '.join(d.name for d in in_flight)}: "
              f"no summary.json yet, i.e. the gate is still open.")
    targets = [d for d in targets if is_complete(d)]
    if not args.force:
        targets = [d for d in targets
                   if not read_json(d / "outcome.json").get("verdict")]

    if args.verdict or args.comment is not None or args.failure_mode:
        if only is None:
            raise SystemExit(
                "non-interactive mode needs --only, so a single verdict "
                "cannot be stamped onto an entire campaign by accident.")
        for d in targets:
            apply_flags(d, args.ckpt, args)
    elif not targets:
        print("nothing to annotate (use --force to revisit labelled ones)")
    else:
        print(f"{len(targets)} rollout(s) to annotate. "
              f"Enter skips, q quits - progress is saved as you go.")
        for d in targets:
            if annotate_one(d, args.ckpt, args) == "quit":
                print("stopped.")
                break

    rebuild(ckpt_dir, args.ckpt)


if __name__ == "__main__":
    main()