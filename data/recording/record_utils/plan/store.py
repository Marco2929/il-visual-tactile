#!/usr/bin/env python3
"""labels.json per episode, plus plan progress reconstructed from disk.

There is deliberately NO progress file. Which slots are done is derived by
reading the labels.json of every episode on disk, every time it is asked for.
A cursor file would be a second source of truth that drifts the moment an
episode is deleted, renamed, moved between folders, or relabelled
success->failed by review_labels.py - and it would drift silently, because
nothing cross-checks it. Reading 40 small JSON files costs nothing at the rate
a human records episodes.

The consequence is the behaviour that matters: a slot counts as DONE only
while a successful episode for it exists. Toggling that episode to failed
afterwards reopens the slot by itself, with no bookkeeping anywhere.

A labels.json records which slot an episode belongs to; the episode DIRECTORY
name records whether it succeeded (episode_007_success). Both are needed, and
neither is inferred from the other.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from . import schedule

LABELS_FILENAME = "labels.json"

# Episode status, taken from the directory suffix written by record.py.
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_OPEN = "open"        # still recording, or interrupted before rename

# No separate "was this confirmed" flag: the retry field is tri-state, with
# None meaning unanswered (see schedule.RETRY_FIELD). A flag beside the value
# would be a second thing to keep in sync, and a desynced one would quietly
# turn unanswered episodes into clean ones.


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def status_of(episode_dir: str) -> str:
    name = os.path.basename(os.path.normpath(episode_dir))
    if name.endswith("_" + STATUS_SUCCESS):
        return STATUS_SUCCESS
    if name.endswith("_" + STATUS_FAILED):
        return STATUS_FAILED
    return STATUS_OPEN


@dataclass
class EpisodeLabels:
    """One episode's labels.json together with where it lives."""
    path: str                      # the labels.json itself
    episode_dir: str
    data: dict

    @property
    def slot(self) -> Optional[int]:
        v = self.data.get("slot")
        return int(v) if isinstance(v, int) else None

    @property
    def status(self) -> str:
        """Live status from the directory name, not the recorded field.

        The recorded `episode_status` is a snapshot from finalisation time;
        the directory name is current. review_labels.py renames directories
        without touching labels.json, so trusting the field would miss every
        later relabel.
        """
        return status_of(self.episode_dir)

    @property
    def retry(self):
        """True / False / None (unanswered)."""
        return self.data.get(schedule.RETRY_FIELD, schedule.RETRY_UNANSWERED)

    @property
    def retry_answered(self) -> bool:
        return self.retry is not schedule.RETRY_UNANSWERED

    @property
    def code(self) -> str:
        return self.data.get("code") or os.path.basename(self.episode_dir)


def write_labels(episode_dir: str, slot: int, *,
                 collection: Optional[str] = None,
                 operator: Optional[str] = None,
                 attempt: int = 1, extra: Optional[dict] = None) -> dict:
    """Create labels.json for an episode that is about to be recorded.

    Called at episode START, not at finalisation: an interrupted take still
    leaves a directory on disk, and the slot it belonged to is only known
    here. Without the file, that recording cannot be placed afterwards.

    No `hoehe` argument: the Boxhoehe is the outermost loop of the plan, so it
    follows from the slot like every other condition.
    """
    labels = (schedule.labels_for(slot)
              or schedule.out_of_plan_labels(slot))
    labels.update({
        "collection": collection,
        "episode_dir": os.path.basename(os.path.normpath(episode_dir)),
        "attempt": attempt,
        "operator": operator,
        "recorded_at": _now(),
        "episode_status": status_of(episode_dir),
        # Unanswered until the operator says otherwise. Written as an explicit
        # null so the field exists in every file and a scan never has to guess
        # whether it is missing or merely unset.
        schedule.RETRY_FIELD: schedule.RETRY_UNANSWERED,
    })
    if extra:
        labels.update(extra)
    _atomic_write(os.path.join(episode_dir, LABELS_FILENAME), labels)
    return labels


def read_labels(episode_dir: str) -> Optional[EpisodeLabels]:
    path = os.path.join(episode_dir, LABELS_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return EpisodeLabels(path=path, episode_dir=episode_dir, data=data)


def update_labels(episode_dir: str, /, **fields) -> bool:
    """Merge fields into an existing labels.json, keeping everything else.

    `episode_dir` is positional-ONLY on purpose: "episode_dir" is also a FIELD
    inside labels.json (it records the directory name, which changes on the
    success/failed rename). Without the `/` marker the call

        update_labels(final_dir, episode_dir="episode_007_success")

    binds the same parameter twice and raises TypeError. The marker makes the
    first argument the target directory and everything keyword-passed a label
    field, so the two can never collide again.
    """
    rec = read_labels(episode_dir)
    if rec is None:
        return False
    rec.data.update(fields)
    rec.data["updated_at"] = _now()
    _atomic_write(rec.path, rec.data)
    return True


def set_retry(episode_dir: str, retry: bool) -> bool:
    return update_labels(episode_dir, **{schedule.RETRY_FIELD: retry})


def _atomic_write(path: str, payload: dict) -> None:
    """Write via a temp file + rename so a crash cannot truncate labels.json.

    A half-written labels.json is worse than none: it parses as nothing and
    the episode silently drops out of every progress scan.
    """
    # encoding is pinned because the labels contain German text ("Boxhöhe",
    # "A-Säule") and ensure_ascii=False writes those characters verbatim. The
    # container's default locale is ASCII, so relying on it raised
    # UnicodeEncodeError mid-write - exactly the truncated labels.json this
    # function exists to prevent.
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# --- Scanning ---------------------------------------------------------------
def scan_episodes(root_dir: str) -> List[EpisodeLabels]:
    """All labelled episodes directly under `root_dir`, sorted by name."""
    if not os.path.isdir(root_dir):
        return []
    out = []
    for name in sorted(os.listdir(root_dir)):
        if not name.startswith("episode_"):
            continue
        ep_dir = os.path.join(root_dir, name)
        if not os.path.isdir(ep_dir):
            continue
        rec = read_labels(ep_dir)
        if rec is not None:
            out.append(rec)
    return out


def scan_collections(base_dir: str) -> List[EpisodeLabels]:
    """Same, but across every collection folder under `base_dir`.

    Needed because the plan is recorded into nine folders (one per block)
    while slot numbering is global: progress for block 2 lives in a different
    directory than block 1, and both are part of the same plan.
    """
    if not os.path.isdir(base_dir):
        return []
    out: List[EpisodeLabels] = []
    for name in sorted(os.listdir(base_dir)):
        sub = os.path.join(base_dir, name)
        if os.path.isdir(sub):
            out.extend(scan_episodes(sub))
    return out


@dataclass
class Progress:
    """Which slots are closed, and by what.

    `done` is success-gated: a slot is only in here while a *_success episode
    referencing it exists on disk. That is what keeps a failed take from
    claiming its slot, and what makes a later success->failed relabel reopen
    the slot without any bookkeeping.
    """
    attempts: Dict[int, List[EpisodeLabels]] = field(default_factory=dict)
    successes: Dict[int, List[EpisodeLabels]] = field(default_factory=dict)
    unplaced: List[EpisodeLabels] = field(default_factory=list)

    @property
    def done(self) -> set:
        return set(self.successes)

    def is_done(self, slot: int) -> bool:
        return slot in self.successes

    def attempt_count(self, slot: int) -> int:
        return len(self.attempts.get(slot, []))

    def success_count(self, slot: int) -> int:
        return len(self.successes.get(slot, []))

    def open_slots(self, slots) -> List[int]:
        return [s for s in slots if s not in self.successes]

    def next_open(self, slots) -> Optional[int]:
        for s in slots:
            if s not in self.successes:
                return s
        return None

    def needing_retry_answer(self) -> List[EpisodeLabels]:
        """Successful episodes nobody has marked as retry / no-retry yet.

        Only successes: a failed take was aborted rather than retried, so the
        question does not apply and asking would produce noise.
        """
        out = [rec for recs in self.successes.values() for rec in recs
               if not rec.retry_answered]
        out.sort(key=lambda r: r.episode_dir)
        return out


def build_progress(records: List[EpisodeLabels]) -> Progress:
    p = Progress()
    for rec in records:
        slot = rec.slot
        if slot is None or not rec.data.get("in_plan", False):
            p.unplaced.append(rec)
            continue
        p.attempts.setdefault(slot, []).append(rec)
        if rec.status == STATUS_SUCCESS:
            p.successes.setdefault(slot, []).append(rec)
    return p


def progress_for(base_dir: str) -> Progress:
    """Plan-wide progress across all collection folders under `base_dir`."""
    return build_progress(scan_collections(base_dir))