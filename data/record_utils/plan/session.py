#!/usr/bin/env python3
"""Recorder-facing half of the plan: cursor, attempts, operator prompt.

record.py touches this through a handful of calls and knows nothing else
about the plan:

    plan = attach(node, root_dir=..., collection=...)   # None if disabled
    plan.announce()                        # at SETUP, before the operator sets up
    plan.on_episode_start(ep_dir, idx)     # writes labels.json
    plan.on_episode_final(final_dir)       # after the success/failed rename
    plan.on_episode_discarded(ep_dir)      # take thrown away
    plan.shutdown()

Two deliberate behaviours, both required by the recording procedure:

1. A slot is claimed by a SUCCESS, never merely by an attempt. Failed and
   discarded takes leave the slot open, so the plan cannot quietly lose a row
   because something did not fit. Progress is recomputed from disk on every
   query (store.py), so relabelling an episode success->failed later reopens
   its slot by itself.

2. The cursor advances only when the operator says so ('n'). Several takes per
   setting are the normal case - the box is already set up, so extra takes of
   the same condition are nearly free. Auto-advancing would make "one more
   try" impossible and would silently promote a mediocre take to the single
   representative of its slot.
"""

import os
import queue
import select
import sys
import threading
from typing import List, Optional

from . import schedule, store

env = os.environ.get

# Which block (1..9) this collection records. The plan is recorded into nine
# folders, one per block, but slot numbering is global - without this the
# recorder cannot tell which 40 of the 360 rows the current folder covers.
# Unset means "whole plan in one folder", i.e. all 360 slots in order.
PLAN_BLOCK_ENV = "PLAN_BLOCK"
# Which box position inside that scene. Optional: unset means the whole scene
# in plan order. Set, it restricts the session to that ONE placement and its
# three layers - which is the unit the operator physically works in. Moving
# the box is then a restart with a new PLAN_POSITION rather than having to sit
# through the remaining positions of the scene.
PLAN_POSITION_ENV = "PLAN_POSITION"
# There is no PLAN_HEIGHT any more: the Boxhoehe is the outermost loop of the
# plan, so blocks 1-3 are "unten", 4-6 "mitte", 7-9 "oben" and the height is
# derived from the slot. Setting the block now sets the height too, which
# removes the one label that previously had to be declared by hand.

HELP_LINES = [
    "",
    "  ENTER   Episode war sauber - gleiche Lage nochmal abraeumen",
    "  r       Folie verpasst und nochmal zugegriffen (Episode bleibt gueltig)",
    "  n       diese Folienlage ist fertig -> naechste Lage/Boxposition",
    "  rn      beides: Retry drin UND Lage fertig",
    "  s       Frage spaeter beantworten (label_episodes.py retry)",
    "  p       Fortschritt dieser Szene",
    "  ?       diese Hilfe",
    "",
    "  Reihenfolge: L1 oben -> L2 Zwischenschicht -> L3 unten, dann Box",
    "  verschieben und von vorne. Der Plan rueckt nur bei 'n' weiter.",
    "",
]


class PlanSession:
    def __init__(self, node, root_dir: str, collection: str,
                 block: Optional[int] = None,
                 position: Optional[str] = None,
                 operator: Optional[str] = None, prompt: bool = True):
        self.node = node
        self.log = node.get_logger()
        self.root_dir = root_dir
        self.collection = collection
        self.block = block
        self.position = position
        self.operator = operator

        # Progress spans all collection folders, not just this one: block 2's
        # successes live in a different directory than block 1's, and both
        # belong to the same 360-slot plan.
        self.base_dir = os.path.dirname(os.path.normpath(root_dir))

        if block and position:
            self.slots: List[int] = list(
                schedule.position_slots(block, position))
        elif block:
            self.slots = list(schedule.block_slots(block))
        else:
            self.slots = list(range(schedule.TOTAL_SLOTS))
        self.cursor: Optional[int] = None
        self._attempt = 1
        self._lock = threading.Lock()

        # Episodes waiting for their retry answer. Unbounded on purpose: the
        # operator may answer late, and dropping an entry would silently leave
        # that episode unanswered.
        self._io_queue: "queue.Queue" = queue.Queue()
        # Successful episodes handed over by the recorder but not yet answered.
        # Held in the IO thread rather than left in the queue so stdin can be
        # read even when nothing is pending - that is what makes a bare 'n'
        # work between takes.
        self._pending: List[str] = []
        # Which prompt is currently on screen, or None if none is. Comparing
        # the TEXT rather than keeping a boolean "already printed" flag is
        # what makes the prompt reappear when the situation changes: with a
        # flag, the idle prompt printed at startup counted as "shown", so the
        # question for the first finished episode was never asked and the
        # episode waited for an answer nobody had been asked for.
        self._prompt_text: Optional[str] = None
        self._running = True
        self._io_thread = None

        self.prompt = bool(prompt and sys.stdin and sys.stdin.isatty())
        if prompt and not self.prompt:
            # Launched from a launch file or a service: nobody can answer. The
            # cursor must still move or every episode would record slot 1, so
            # it advances on success and the retry flag is filled in afterwards
            # with label_episodes.py.
            self.log.warn(
                "PLAN: kein Terminal an stdin - keine Abfragen. Cursor rueckt "
                "bei Success automatisch vor, Retry bleibt unbeantwortet "
                "(spaeter mit label_episodes.py retry nachtragen).")

        self._sync_cursor()
        self._log_intro()

        if self.prompt:
            self._io_thread = threading.Thread(target=self._io_loop, daemon=True)
            self._io_thread.start()

    # --- Progress ----------------------------------------------------------
    def progress(self) -> store.Progress:
        """Always re-read from disk; see store.py for why there is no cache."""
        return store.progress_for(self.base_dir)

    def _sync_cursor(self) -> None:
        """Put the cursor on the first slot of this block without a success."""
        prog = self.progress()
        nxt = prog.next_open(self.slots)
        with self._lock:
            self.cursor = nxt
            self._attempt = (prog.attempt_count(nxt) + 1) if nxt is not None else 1

    def _log_intro(self) -> None:
        prog = self.progress()
        done = len([s for s in self.slots if prog.is_done(s)])
        # Spelled out rather than "Block 2/9": two fractions side by side read
        # as "2 of 9 done", which is not what the block number means.
        scope = (schedule.block_title(self.block) if self.block
                 else "gesamter Plan")
        if self.position:
            scope += f" · Box {schedule.POSITIONEN[self.position]}"
        self.log.info(f"PLAN: {scope} von {schedule.NUM_BLOCKS} Bloecken, "
                      f"{done}/{len(self.slots)} Slots erledigt.")

        if self.block is None:
            self.log.warn(
                f"!  {PLAN_BLOCK_ENV} nicht gesetzt - diese Collection deckt "
                f"alle {schedule.TOTAL_SLOTS} Slots ab. Bei einem Ordner pro "
                f"Block {PLAN_BLOCK_ENV}=1..{schedule.NUM_BLOCKS} setzen.")
        if self.cursor is None:
            self.log.info("PLAN: alle Slots dieses Blocks haben einen Success. "
                          "Weitere Aufnahmen werden als Zusatz-Takes des "
                          "letzten Slots gelabelt.")
        if self.prompt:
            self.log.info("PLAN: nach jeder Episode ENTER (sauber) / r "
                          "(Retry) / n (naechster Slot). '?' = Hilfe.")

    # --- Recorder hooks ----------------------------------------------------
    def announce(self) -> None:
        """Tell the operator what to set up for the current slot."""
        with self._lock:
            slot, attempt = self.cursor, self._attempt
        if slot is None:
            self.log.warn("PLAN: kein offener Slot mehr in diesem Block - "
                          "Aufnahme laeuft als Zusatz-Take.")
            return
        for line in schedule.setup_lines(slot, attempt=attempt):
            self.log.info(line)

    def _effective_slot(self) -> int:
        """Slot to label the next episode with.

        When every slot of the block is closed, extra takes are attributed to
        the last slot rather than to nothing: they were recorded under that
        setting, and dropping the label would make them unusable.
        """
        with self._lock:
            if self.cursor is not None:
                return self.cursor
        return self.slots[-1]

    def on_episode_start(self, episode_dir: str, episode_idx: int) -> None:
        slot = self._effective_slot()
        with self._lock:
            attempt = self._attempt
        store.write_labels(
            episode_dir, slot,
            collection=self.collection,
            operator=self.operator, attempt=attempt,
            extra={"episode_idx": episode_idx,
                   "plan_block": self.block},
        )
        self.log.info(f"PLAN: {schedule.slot_code(slot)} · Take {attempt}")

    def on_episode_final(self, final_dir: str) -> None:
        """After the success/failed rename: refresh labels, then ask.

        The directory name changed, so the episode_dir recorded at start is
        stale and progress scans would attribute the episode to the wrong
        status.
        """
        status = store.status_of(final_dir)
        store.update_labels(
            final_dir,
            episode_dir=os.path.basename(os.path.normpath(final_dir)),
            episode_status=status,
        )

        if status != store.STATUS_SUCCESS:
            # Failed takes leave the slot open and are not asked about: the
            # take was aborted rather than retried, so the question does not
            # apply.
            with self._lock:
                self._attempt += 1
            self.log.info("PLAN: Slot bleibt offen (kein Success).")
            return

        if self.prompt:
            self._io_queue.put(final_dir)
        else:
            # Nobody can answer, so holding the cursor would repeat one slot
            # forever. Advance and leave retry unanswered.
            self._advance()

    def on_episode_discarded(self, episode_dir: str) -> None:
        """A take was thrown away; the slot must stay open."""
        self.log.info("PLAN: verworfene Aufnahme - Slot bleibt offen.")

    def _advance(self) -> None:
        self._prompt_text = None
        self._sync_cursor()
        with self._lock:
            slot = self.cursor
        prog = self.progress()
        done = len([s for s in self.slots if prog.is_done(s)])
        if slot is None:
            self.log.info(f"PLAN: Block fertig - {done}/{len(self.slots)} "
                          f"Slots mit Success.")
            return
        self.log.info(f"PLAN: {done}/{len(self.slots)} erledigt -> naechster "
                      f"Slot {schedule.slot_code(slot)}")
        self.announce()

    # --- Operator I/O ------------------------------------------------------
    def _io_loop(self) -> None:
        """Read stdin continuously in its own thread.

        Separate thread so a slow answer never stalls ROS callbacks or delays
        the next take: the operator can start recording again with the
        previous answer still outstanding.

        stdin is polled rather than read with a blocking input() per episode.
        The blocking version only listened while an episode was awaiting its
        retry answer and returned as soon as that answer arrived - so the 'n'
        that was supposed to close the slot afterwards was read by nobody. The
        cursor then stayed on one slot and every further take was recorded as
        another attempt of it. Polling means 'n' works at any time, whether or
        not an answer is outstanding.
        """
        while self._running:
            while True:
                try:
                    self._pending.append(self._io_queue.get_nowait())
                except queue.Empty:
                    break
            if None in self._pending:
                return

            self._show_prompt()
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
            except (OSError, ValueError):
                self._stdin_gone()
                return
            if not ready:
                continue
            line = sys.stdin.readline()
            if line == "":
                self._stdin_gone()
                return
            raw = line.strip().lower()
            self._handle_line(raw)

    def _prompt_for_state(self) -> str:
        """The prompt names the SETTING, not the slot code.

        'HM-AS-P1-L2_s002' is the right identifier for a filename and the
        wrong one for a question: the operator has to decode it to know what
        is in front of them, and a decoding mistake is exactly how takes end
        up on the wrong slot. The physical setting - which layer, which box
        position, which take - is what the answer depends on, so that is what
        is shown.
        """
        if self._pending:
            name = os.path.basename(os.path.normpath(self._pending[0]))
            # Episode number only: the full directory name adds a '_success'
            # suffix that the operator did not ask about.
            short = name.replace("episode_", "Episode ").replace("_success", "")
            # The backlog is named explicitly. Recording continues while an
            # answer is outstanding, so several episodes can queue up, and
            # without the count the operator cannot tell that the question on
            # screen is about an older take than the one just recorded.
            backlog = (f" ({len(self._pending)} offen)"
                       if len(self._pending) > 1 else "")
            return (f"{short} sauber?{backlog} "
                    f"[ENTER]=ja  [r]=Retry  [n]=Lage fertig  [?]: ")

        with self._lock:
            slot, attempt = self.cursor, self._attempt
        lab = schedule.labels_for(slot) if slot is not None else None
        if lab is None:
            return "Plan fertig · [n]=weiter  [p]=Fortschritt  [?]: "
        return (f"{lab['folienlage_label']} · "
                f"Box {lab['position_label']} · "
                f"Take {attempt}/{schedule.TAKES_PER_LAGE} "
                f"-> [ENTER]=noch ein Take  [n]=Lage fertig  [p]  [?]: ")

    def _show_prompt(self) -> None:
        """Print the prompt whenever the situation it describes has changed.

        Re-printed on change rather than once per poll: a prompt per 0.5s tick
        would bury the ROS log, while printing it only once leaves a finished
        episode waiting for an answer that was never requested.
        """
        text = self._prompt_for_state()
        if text == self._prompt_text:
            return
        # A newline first: the ROS logger writes whole lines, so without it the
        # prompt would be appended to whatever log line was printed last.
        sys.stdout.write("\n" + text)
        sys.stdout.flush()
        self._prompt_text = text

    def _stdin_gone(self) -> None:
        self.log.warn("PLAN: stdin weg - Abfragen aus, Retry bleibt "
                      "unbeantwortet.")
        self.prompt = False
        self._advance()

    def _handle_line(self, raw: str) -> None:
        if raw == "?":
            print("\n".join(HELP_LINES))
            self._prompt_text = None
            return
        if raw == "p":
            print(self.progress_report())
            self._prompt_text = None
            return
        if raw == "s":
            if self._pending:
                name = os.path.basename(os.path.normpath(self._pending.pop(0)))
                self.log.info(f"{name}: spaeter (label_episodes.py retry). "
                              f"Slot bleibt.")
                self._prompt_text = None
            return

        advance = "n" in raw
        answer = raw.replace("n", "")

        if not self._pending:
            if advance:
                self._prompt_text = None
                self._advance()
            elif answer:
                self.log.warn("PLAN: keine offene Episode - 'n' = naechster "
                              "Slot, '?' = Hilfe.")
            else:
                # Bare ENTER between takes: the prompt offers it as "noch ein
                # Take", and the setting for that take is already on screen -
                # so confirm it and leave the prompt alone. Reprinting it here
                # is what produced a wall of identical lines on every
                # keypress; clearing _prompt_text would do exactly that.
                self.log.info(
                    f"PLAN: bereit fuer Take {self._attempt} - Aufnahme per "
                    f"Tracker-Button starten.")
            return

        name = os.path.basename(os.path.normpath(self._pending[0]))
        try:
            retry = schedule.parse_retry(answer)
        except ValueError as exc:
            # Re-ask instead of guessing: a typo must not be recorded as a
            # clean grasp, since that is the zero-keystroke answer.
            self.log.warn(f"{name}: {exc} - nochmal.")
            return

        store.set_retry(self._pending.pop(0), retry)
        self.log.info(f"{name}: {schedule.retry_text(retry)}")
        # The pending list shrank, so the prompt must be redrawn either way:
        # it either names the next queued episode or falls back to the
        # between-takes line.
        self._prompt_text = None

        if advance:
            self._advance()
        else:
            with self._lock:
                self._attempt += 1
                slot = self.cursor
            code = schedule.slot_code(slot) if slot is not None else "-"
            self.log.info(f"PLAN: Slot {code} bleibt - 'n' fuer den naechsten.")

    # --- Reporting ---------------------------------------------------------
    def progress_report(self) -> str:
        prog = self.progress()
        lines = ["", "Fortschritt" + (f" Block {self.block}" if self.block else "")]
        open_slots = prog.open_slots(self.slots)
        done = len(self.slots) - len(open_slots)
        lines.append(f"  Slots mit Success : {done}/{len(self.slots)}")
        lines.append(f"  noch offen        : {len(open_slots)}")
        extra = sum(max(0, prog.success_count(s) - 1) for s in self.slots)
        if extra:
            lines.append(f"  Zusatz-Takes      : {extra}")
        retries = sum(1 for s in self.slots
                      for r in prog.successes.get(s, []) if r.retry is True)
        if retries:
            lines.append(f"  davon mit Retry   : {retries}")
        pending = prog.needing_retry_answer()
        if pending:
            lines.append(f"  Retry offen       : {len(pending)} "
                         f"(label_episodes.py retry)")
        if open_slots:
            preview = ", ".join(schedule.slot_code(s) for s in open_slots[:5])
            lines.append(f"  naechste offen    : {preview}"
                         + (" ..." if len(open_slots) > 5 else ""))
        lines.append("")
        return "\n".join(lines)

    def shutdown(self) -> None:
        self._running = False
        self._io_queue.put(None)
        self.log.info(self.progress_report())


def attach(node, root_dir: str, collection: str) -> Optional[PlanSession]:
    """Build a PlanSession from the environment, or None when disabled.

    Returns None rather than a no-op object so the recorder's call sites read
    `if self.plan:` and the plan stays visibly optional - record.py must work
    unchanged with PLAN_ENABLED=0.
    """
    if env("PLAN_ENABLED", "1") == "0":
        node.get_logger().info("PLAN: deaktiviert (PLAN_ENABLED=0) - es wird "
                               "kein labels.json geschrieben.")
        return None

    raw_block = env(PLAN_BLOCK_ENV)
    block = None
    if raw_block:
        try:
            block = int(raw_block)
            schedule.block_slots(block)      # validates the range
        except ValueError as exc:
            node.get_logger().error(
                f"PLAN: {PLAN_BLOCK_ENV}={raw_block!r} ungueltig ({exc}) - "
                f"erwartet 1..{schedule.NUM_BLOCKS}. Plan deaktiviert, damit "
                f"keine falsch gelabelten Episoden entstehen.")
            return None

    raw_pos = env(PLAN_POSITION_ENV)
    position = None
    if raw_pos:
        position = schedule.parse_position(raw_pos)
        if position is None:
            node.get_logger().error(
                f"PLAN: {PLAN_POSITION_ENV}={raw_pos!r} ungueltig - erwartet "
                f"{'/'.join(schedule.POSITION_ORDER)} oder 1.."
                f"{len(schedule.POSITION_ORDER)}. Plan deaktiviert, damit "
                f"keine falsch gelabelten Episoden entstehen.")
            return None
        if block is None:
            # Without a scene the position has no slots to point at: the same
            # placement exists in all nine scenes.
            node.get_logger().error(
                f"PLAN: {PLAN_POSITION_ENV} ohne {PLAN_BLOCK_ENV} ist "
                f"mehrdeutig - jede Szene hat dieselbe Boxposition. Plan "
                f"deaktiviert.")
            return None

    return PlanSession(
        node, root_dir=root_dir, collection=collection, block=block,
        position=position,
        operator=env("PLAN_OPERATOR") or None,
        prompt=env("PLAN_PROMPT", "1") != "0",
    )