#!/usr/bin/env python3
"""Read the annotations that annotate_rollouts.py writes.

This REPLACES the hand-maintained group table that used to live in
groups.py. That table existed only because the recordings carried no
outcome and no machine-readable condition - the test design sat in a
chat message and had to be transcribed. index.csv now carries both,
so transcribing it again would just be a second source of truth that
can disagree with the first.

What index.csv gives us that meta.json never did:

  success / verdict   whether the run WORKED. Every rate in the reports
                      below is impossible without it.
  failure_mode        grasp_miss / approach_xy / collision / no_lift -
                      a human's attribution, independent of our metrics,
                      which makes it a cross-check rather than an input.
  conditions          the experimental condition as a stable string
                      ("Palette Nominalhoehe increased"), instead of the
                      free-text `notes` with its typos and trailing
                      newlines.

ONE CAVEAT, and it matters for any aggregate: `success` and `verdict`
can disagree. In foil_lifting_relative_e40, rollout_008 is
verdict=fail with success=True. `verdict` is the field
annotate_rollouts.py drives its own report from, so that is the one
used here, and the disagreement is surfaced rather than silently
resolved - see `conflicts()`.
"""
import csv
import json
import pathlib

# Condition strings -> (height, xy). Derived from the suffix the operator
# appended, so this stays a presentation detail: the raw string is kept on
# every record and is what gets printed.
#
# "increased" / "lowered" describe the PALLET being moved up or down, i.e.
# exactly the height test. "Zwischenhoehe" / "untere Hoehe" are different
# LAYERS of the stack, which is a separate axis - a higher layer is not the
# same experiment as the nominal layer lifted.
HEIGHT_WORDS = {
    "increased": "too_high",
    "lowered": "too_low",
}
XY_WORDS = {
    "links": "links",
    "mitte": "mitte",
    "rechts": "rechts",
}
LAYER_WORDS = {
    "nominalhoehe": "nominal",
    "zwischenhoehe": "mid",
    "untere hoehe": "low",
}


class Label:
    """One row of index.csv, with the condition string parsed."""

    def __init__(self, row):
        self.raw = row
        self.name = row["rollout"]
        self.verdict = (row.get("verdict") or "").strip()
        self.failure_mode = (row.get("failure_mode") or "").strip()
        self.conditions = (row.get("conditions") or "").strip()
        self.notes = ""
        self.condition_conflict = None
        self.comment = (row.get("comment") or "").strip()
        self.duration = _float(row.get("duration_s"))
        self.chunks = _int(row.get("chunks"))

        self.success_field = (row.get("success") or "").strip().lower()
        # verdict is authoritative: it is what annotate_rollouts.py counts.
        self.ok = self.verdict == "success"

        self.set_condition(self.conditions)

    def set_condition(self, text):
        """Re-parse the condition from `text` (used when notes override)."""
        self.conditions = text.strip()
        c = self.conditions.lower()
        self.layer = next((v for k, v in LAYER_WORDS.items() if k in c), None)
        self.height = next((v for k, v in HEIGHT_WORDS.items() if k in c),
                           "nominal")
        self.xy = next((v for k, v in XY_WORDS.items() if k in c), None)
        self.is_retry = "retry" in c

    @property
    def conflicted(self):
        """verdict and the success column disagree."""
        truthy = self.success_field in ("true", "1", "yes")
        return truthy != self.ok

    def __repr__(self):
        return f"<Label {self.name} {self.verdict} {self.conditions!r}>"


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def load(ckpt_dir, prefer="notes"):
    """-> {rollout_name: Label}. Empty dict when index.csv is absent.

    Absence is not an error: a checkpoint can be recorded before anyone has
    annotated it. The callers report the gap instead of failing, because a
    missing annotation should not make the geometric analyses unavailable.

    THE CONDITION IS CROSS-CHECKED against meta.json's `notes`, because the
    two disagree. In foil_lifting_relative_e40, rollouts 003/004/005 are
    "Palette Zwischenhoehe" in notes and "Palette Nominalhoehe" in
    index.csv. That is not cosmetic: 003 and 004 are successes, so the
    choice moves two wins between the nominal and the mid layer and inverts
    which layer looks better.

    `notes` wins by default. It is written by log_rollout.py AT RECORD TIME
    from what the operator typed for that run; the index.csv column is
    filled in during a later annotation pass and can be carried over from
    the previous row. Every disagreement is recorded on the Label and
    reported by `condition_conflicts()` rather than silently resolved.
    """
    d = pathlib.Path(ckpt_dir)
    path = d / "index.csv"
    if not path.is_file():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("rollout") or "").strip()
            if not name:
                continue
            lb = Label(row)
            lb.notes = _notes(d / name)
            if lb.notes and lb.notes != lb.conditions:
                lb.condition_conflict = (lb.conditions, lb.notes)
                if prefer == "notes":
                    lb.set_condition(lb.notes)
            out[name] = lb
    return out


def _notes(rollout_dir):
    try:
        with open(pathlib.Path(rollout_dir) / "meta.json",
                  encoding="utf-8") as f:
            return (json.load(f).get("notes") or "").strip()
    except (OSError, ValueError):
        return ""


def condition_conflicts(labels):
    """-> [(name, index_csv_value, meta_notes_value)]."""
    return [(lb.name,) + lb.condition_conflict
            for lb in labels.values() if lb.condition_conflict]


def conflicts(labels):
    return [lb for lb in labels.values() if lb.conflicted]


def by_condition(labels):
    """-> {condition string: [Label]}, insertion-ordered by first sighting."""
    out = {}
    for lb in labels.values():
        out.setdefault(lb.conditions, []).append(lb)
    return out


def group(labels, height=None, xy=None, layer=None, retry=None):
    """Select labels by parsed condition. None means 'do not filter on it'."""
    sel = []
    for lb in labels.values():
        if height is not None and lb.height != height:
            continue
        if xy is not None and lb.xy != xy:
            continue
        if layer is not None and lb.layer != layer:
            continue
        if retry is not None and lb.is_retry != retry:
            continue
        sel.append(lb)
    return sorted(sel, key=lambda x: x.name)


def rate(labels):
    """-> (n_success, n_total, percent) over judged rollouts."""
    judged = [lb for lb in labels if lb.verdict in ("success", "partial",
                                                    "fail")]
    if not judged:
        return 0, 0, float("nan")
    good = sum(1 for lb in judged if lb.ok)
    return good, len(judged), 100.0 * good / len(judged)
