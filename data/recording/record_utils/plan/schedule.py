#!/usr/bin/env python3
"""The recording plan: "Folienlagen entfernen".

The nesting is the physical workflow, innermost first - each level is what is
cheapest to change next:

    take        N Takes derselben Lage              TAKES_PER_LAGE
      lage      L1 oben -> L2 mitte -> L3 unten     3
        position  Box verschieben                   6
          bauteil   naechstes Bauteil               3
            boxhoehe  Szene eine Ebene verschieben  3

- TAKE innermost: the box is built and the layer exposed, so another take of
  the same layer costs a button press. Repeats are the NORMAL case here
  ("die Folie einige Male abraeumen"), which is why the plan counts them
  instead of leaving them to the operator.
- LAGE next: the layers are physically stacked, so they only come off
  top-down. L1 -> L2 -> L3 empties one build; any other order would mean
  rebuilding in between.
- POSITION next: moving the box is quick, but the stack has to be rebuilt.
- BAUTEIL next: swapping the part means a whole new scene.
- BOXHOEHE outermost: moving the box to another rack level is the most work.
  Doing it last also means every part gets rebuilt from scratch at each
  height, which adds exactly the variation (part placement, foil draping,
  light incidence) a policy has to tolerate.

A SLOT is one (boxhoehe, bauteil, position, lage) row. It is NOT one episode:
a slot is recorded TAKES_PER_LAGE times, and the cursor only moves on when the
operator says so, so a slot may also take more. Slot numbering is 0-based and
continuous across all heights.

Every condition label follows from the slot number alone and is DERIVED, never
typed. Exactly one label needs a human (see RETRY_FIELD).

The plan is addressed by NAME, not by number, because the heights are not
recorded in a fixed order - starting in the middle is normal:

    PLAN_SCENE=mitte/a_saeule     # one part at one height = one session
    PLAN_SCENE=mitte              # first part at that height

Naming also means selection is free: a height nobody records simply has no
episodes, and no numbering has to be renegotiated for it.

This module is pure data: no ROS, no filesystem, no I/O.

    python3 record_utils/plan/schedule.py --scenes
    python3 record_utils/plan/schedule.py --scene mitte/a_saeule
    python3 record_utils/plan/schedule.py --slot 5
"""

import argparse
from typing import Dict, List, Optional

LABEL_SCHEMA_VERSION = 4

# How many takes of the SAME layer the plan aims for. The cursor still only
# advances on 'n', so more or fewer is always possible - this is the target,
# not a limit.
TAKES_PER_LAGE = 3

# --- Vocabulary -------------------------------------------------------------
# Keys are the machine-readable values that end up in labels.json; the German
# wording is kept alongside so the recorder prints the words the operator uses.
# Do NOT rename a key once episodes exist with it - recorded metadata would
# stop matching the code.
BAUTEILE: Dict[str, Dict[str, str]] = {
    "a_saeule":    {"code": "AS", "label": "A-Säule"},
    "b_saeule":    {"code": "BS", "label": "B-Säule"},
    "heckleuchte": {"code": "HL", "label": "Heckleuchte"},
}
BAUTEIL_ORDER = ["a_saeule", "b_saeule", "heckleuchte"]

# Rack level the box sits on. "mitte" comes first because that is where
# recording starts; from there the scene moves one level up or down, and only
# later much further down. This ORDER IS NOT A SCHEDULE - scenes are picked by
# name, so any sequence is fine. It only fixes slot numbering.
BOXHOEHEN: Dict[str, Dict[str, str]] = {
    "mitte": {"code": "HM", "label": "mittlere Boxhöhe (Start)"},
    "oben":  {"code": "HO", "label": "eine Ebene höher"},
    "unten": {"code": "HU", "label": "eine Ebene tiefer"},
}
HOEHEN_ORDER = ["mitte", "oben", "unten"]

# Which foil layer is being removed. Innermost ordered axis: stacked, so
# top-down only.
FOLIENLAGEN: Dict[str, str] = {
    "L1": "oberste Folienlage (über den Teilen)",
    "L2": "mittlere Folienlage (zwischen den Teilen)",
    "L3": "unterste Folienage (Boden, zwischen Karton und Teilen)",
}
LAGEN_ORDER = ["L1", "L2", "L3"]
FOLIEN_ORDER = LAGEN_ORDER          # old name, kept so importers do not break

# Box placement in the cell: "dann werde ich die Box verschieben und das
# wiederholen" - one full pass of all layers per placement. Named by WHERE the
# box stands, not by "verschoben (quer)": the operator has to reproduce the
# placement, and a direction of movement only says where it came from.
POSITIONEN: Dict[str, str] = {
    "P1": "Mitte",
    "P2": "vorne links",
    "P3": "vorne rechts",
    "P4": "hinten links",
    "P5": "hinten rechts",
}
POSITION_ORDER = ["P1", "P2", "P3", "P4", "P5"]

# --- Nuisance variables: alternated per scene, NOT looped over -------------
# Gloves change the hand's appearance and friction, lighting changes the image.
# Both must vary, but neither is an axis here: as a loop they would double the
# episode count (405 -> 810), and as a free choice they would end up constant
# because nobody switches mid-session.
#
# Alternating by scene index gives each Bauteil both conditions, because every
# part recurs at all three heights (scenes 0/3/6 for the first part -> ohne,
# mit, ohne). The cost is honest and worth naming: gloves are CONFOUNDED with
# the scene, so their effect is only separable across scenes, never within
# one. Separating it properly would need the real axis and twice the data.
HANDSCHUH_ORDER = [False, True]
LICHT_ORDER = ["hell", "gedimmt"]

# --- Wrist remount ----------------------------------------------------------
# Not a label but a procedure: unscrewing and remounting the wrist camera
# perturbs its extrinsics on purpose, so the policy cannot rely on one exact
# mounting. Hooked to moments where the scene is being rebuilt anyway - the
# start of a scene, and its midpoint at P4 - which is 2 per scene, 18 overall,
# matching the original plan without adding a single episode.
REMOUNT_AT_POSITIONS = ("P1", "P4")
REMOUNT_TOLERANZ = "Rot ±1-3°/max 5°, Transl ±2-4 mm/max 8 mm"

# Top-Cam is only touched when the box changes rack level, i.e. at the first
# scene of a new Boxhoehe.
TOP_CAM_CHECK_AT_SCENE_START = True

# --- Loop geometry (derived, never hardcoded) ------------------------------
SLOTS_PER_POSITION = len(LAGEN_ORDER)                           # 3
SLOTS_PER_SCENE = SLOTS_PER_POSITION * len(POSITION_ORDER)      # 18
SLOTS_PER_HOEHE = SLOTS_PER_SCENE * len(BAUTEIL_ORDER)          # 54
TOTAL_SLOTS = SLOTS_PER_HOEHE * len(HOEHEN_ORDER)               # 162
TAKES_PER_SCENE = SLOTS_PER_SCENE * TAKES_PER_LAGE              # 54
TOTAL_TAKES = TOTAL_SLOTS * TAKES_PER_LAGE                      # 486

# Old names. A "block" in the previous plan is exactly a scene here, so these
# stay meaningful rather than being stubs.
SLOTS_PER_BLOCK = SLOTS_PER_SCENE


# --- Scenes -----------------------------------------------------------------
class Scene:
    """One (boxhoehe, bauteil) pair: one build-up, one session, one folder."""

    def __init__(self, boxhoehe: str, bauteil: str):
        self.boxhoehe = boxhoehe
        self.bauteil = bauteil

    @property
    def name(self) -> str:
        return f"{self.boxhoehe}/{self.bauteil}"

    @property
    def code(self) -> str:
        return (f"{BOXHOEHEN[self.boxhoehe]['code']}-"
                f"{BAUTEILE[self.bauteil]['code']}")

    @property
    def title(self) -> str:
        return (f"{BAUTEILE[self.bauteil]['label']} @ "
                f"{BOXHOEHEN[self.boxhoehe]['label']}")

    @property
    def index(self) -> int:
        """0-based position in SCENES."""
        return (HOEHEN_ORDER.index(self.boxhoehe) * len(BAUTEIL_ORDER)
                + BAUTEIL_ORDER.index(self.bauteil))

    @property
    def slots(self) -> range:
        start = self.index * SLOTS_PER_SCENE
        return range(start, start + SLOTS_PER_SCENE)

    @property
    def folder(self) -> str:
        """Suggested FOLDER_NAME, e.g. 'folien_mitte_a_saeule'."""
        return f"folien_{self.boxhoehe}_{self.bauteil}"

    def __repr__(self) -> str:
        return f"Scene({self.name})"


SCENES: List[Scene] = [Scene(h, b) for h in HOEHEN_ORDER for b in BAUTEIL_ORDER]
NUM_SCENES = len(SCENES)
NUM_BLOCKS = NUM_SCENES             # old name

# Old name: label_episodes.py reads BLOCKS[i]["boxhoehe"] / ["bauteil"].
BLOCKS: List[Dict[str, str]] = [
    {"boxhoehe": s.boxhoehe, "bauteil": s.bauteil} for s in SCENES
]


def scene_by_name(name: str) -> Optional[Scene]:
    """'mitte/a_saeule' or 'mitte' -> Scene, else None.

    A bare height yields the first part at that height, so PLAN_SCENE=mitte
    starts a session without naming the part.
    """
    key = (name or "").strip().lower().replace("-", "_").replace("\\", "/")
    if not key:
        return None
    if "/" in key:
        h, _, b = key.partition("/")
        if h in BOXHOEHEN and b in BAUTEILE:
            return Scene(h, b)
        return None
    if key in BOXHOEHEN:
        return Scene(key, BAUTEIL_ORDER[0])
    return None


def scenes_at(boxhoehe: str) -> List[Scene]:
    return [s for s in SCENES if s.boxhoehe == boxhoehe]


def scene_of_slot(slot: int) -> Optional[Scene]:
    if slot < 0 or slot >= TOTAL_SLOTS:
        return None
    return SCENES[slot // SLOTS_PER_SCENE]


# --- Old block API, now scene-backed --------------------------------------
def block_of(slot: int) -> int:
    """1-based scene number containing `slot`."""
    return slot // SLOTS_PER_SCENE + 1


def block_slots(block: int) -> range:
    """Slot range of a 1-based scene number."""
    if not 1 <= block <= NUM_SCENES:
        raise ValueError(f"Szene {block} außerhalb 1..{NUM_SCENES}")
    return SCENES[block - 1].slots


def block_title(block: int) -> str:
    return f"Szene {block}: {SCENES[block - 1].title}"


def parse_position(raw: str) -> Optional[str]:
    """'P3', 'p3', '3' or 'rechts' -> 'P3'. None if unknown.

    The label is accepted as well as the code because the operator reads the
    label off the prompt ('Box rechts'), not the code.
    """
    s = (raw or "").strip().lower()
    if not s:
        return None
    if s.isdigit():
        i = int(s)
        return POSITION_ORDER[i - 1] if 1 <= i <= len(POSITION_ORDER) else None
    up = s.upper()
    if up in POSITIONEN:
        return up
    for code, label in POSITIONEN.items():
        if label.lower() == s:
            return code
    return None


def position_slots(block: int, position: str) -> range:
    """The layer slots of ONE box position inside a scene.

    One position is one box placement, i.e. one build-up of the stack and its
    three layers L1-L3. That is the unit the operator actually works in: the
    box is moved, then emptied top-down. Selecting it from the environment
    means moving the box is a restart with a new PLAN_POSITION instead of
    sitting through the rest of the scene.
    """
    if position not in POSITIONEN:
        raise ValueError(f"Position {position!r} unbekannt "
                         f"(erwartet {'/'.join(POSITION_ORDER)})")
    start = (block_slots(block).start
             + POSITION_ORDER.index(position) * SLOTS_PER_POSITION)
    return range(start, start + SLOTS_PER_POSITION)


# --- The one label a human must supply -------------------------------------
# `retry` = the operator missed the foil and grabbed again WITHIN this episode.
# Not a failure: the episode is valid and contains a missed grasp plus the
# recovery, which is exactly why it must be marked. Such episodes are
# systematically longer, so leaving them unmarked mixes two length
# distributions and hides a second behaviour inside the clean-grasp data.
#
# Deliberately NOT derived from episode length: length also varies with Lage,
# Position and operator speed, so a threshold would mislabel both ways. The
# operator knows the instant it happens; a classifier never will.
#
# Three states, and the third is load-bearing:
#   False  confirmed single clean grasp
#   True   confirmed retry
#   None   nobody has said yet
# None rather than a parallel "confirmed" flag: one field cannot drift out of
# sync with itself, and an unanswered episode stays visibly unanswered instead
# of masquerading as a clean one.
RETRY_FIELD = "retry"
RETRY_UNANSWERED = None


def parse_retry(raw: str):
    """'' -> False, 'r' -> True. Raises ValueError on anything else.

    Raising rather than defaulting: a typo must not become a clean grasp,
    because that is the answer needing no keystroke at all.
    """
    s = raw.strip().lower()
    if s == "":
        return False
    if s == "r":
        return True
    raise ValueError(f"unbekannt: {s!r} (ENTER = ohne Retry, r = mit Retry)")


def retry_text(retry) -> str:
    if retry is RETRY_UNANSWERED:
        return "unbeantwortet"
    return "mit Retry" if retry else "ohne Retry"


# --- Slot -> labels ---------------------------------------------------------
def _decode(slot: int):
    """slot -> (scene, position, folienlage)."""
    scene = SCENES[slot // SLOTS_PER_SCENE]
    within = slot % SLOTS_PER_SCENE
    position = POSITION_ORDER[within // SLOTS_PER_POSITION]
    lage = LAGEN_ORDER[within % SLOTS_PER_POSITION]
    return scene, position, lage


def labels_for(slot: int) -> Optional[dict]:
    """Every derivable label of a slot, or None if out of range.

    Callers must handle None rather than clamping to a nearby slot: a
    wrong-but-plausible label is worse than a missing one.
    """
    if slot < 0 or slot >= TOTAL_SLOTS:
        return None

    scene, position, lage = _decode(slot)
    within = slot % SLOTS_PER_SCENE

    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "slot": slot,
        "slot_no": slot + 1,
        "in_plan": True,
        "code": slot_code(slot),

        # --- primary labels: what the dataset is meant to be split by ---
        "boxhoehe": scene.boxhoehe,
        "boxhoehe_label": BOXHOEHEN[scene.boxhoehe]["label"],
        "bauteil": scene.bauteil,
        "bauteil_label": BAUTEILE[scene.bauteil]["label"],

        # --- remaining plan axes ---
        "position": position,
        "position_label": POSITIONEN[position],
        "folienlage": lage,
        "folienlage_label": FOLIENLAGEN[lage],

        "scene": scene.name,
        "block": scene.index + 1,
        "takes_geplant": TAKES_PER_LAGE,

        # --- context / nuisance variables (derived, see above) ---
        "handschuhe": HANDSCHUH_ORDER[scene.index % len(HANDSCHUH_ORDER)],
        "licht": LICHT_ORDER[scene.index % len(LICHT_ORDER)],

        # --- procedure reminders, not labels ---
        # A fresh stack is needed whenever a position starts, i.e. once all
        # layers of the previous placement have been removed.
        "neuaufbau_before": within % SLOTS_PER_POSITION == 0,
        "verschieben_before": (within % SLOTS_PER_POSITION == 0
                               and within > 0),
        "neue_szene_before": within == 0,
        # Only on the FIRST layer of the position: a remount between layers of
        # one stack would change the camera mid-position.
        "remount_before": (position in REMOUNT_AT_POSITIONS
                           and within % SLOTS_PER_POSITION == 0),
        "top_cam_check_before": (within == 0
                                 and scene.bauteil == BAUTEIL_ORDER[0]),
    }


def out_of_plan_labels(slot: int) -> dict:
    """Minimal record for a take belonging to no slot (reserve/extra)."""
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "slot": slot,
        "slot_no": slot + 1,
        "in_plan": False,
        "code": f"RSV_s{slot + 1:03d}",
        "boxhoehe": None, "boxhoehe_label": None,
        "bauteil": None, "bauteil_label": None,
        "position": None, "position_label": None,
        "folienlage": None, "folienlage_label": None,
        "scene": None, "block": None, "takes_geplant": None,
        "handschuhe": None, "licht": None,
        "neuaufbau_before": False, "verschieben_before": False,
        "neue_szene_before": False, "remount_before": False,
        "top_cam_check_before": False,
    }


def slot_code(slot: int) -> str:
    """Sheet-style identifier, e.g. 'HM-AS-P2-L3_s006'.

    Printed by the recorder so the operator can check it against the scene in
    front of them: if the code does not match, PLAN_SCENE is wrong and every
    label after it would be wrong too.
    """
    if slot < 0 or slot >= TOTAL_SLOTS:
        return f"RSV_s{slot + 1:03d}"
    scene, position, lage = _decode(slot)
    return f"{scene.code}-{position}-{lage}_s{slot + 1:03d}"


# --- Human-readable rendering ----------------------------------------------
def setup_lines(slot: int, attempt: int = 1, done_takes: int = 0) -> List[str]:
    """What the operator must physically do for this slot."""
    lab = labels_for(slot)
    if lab is None:
        return [f"PLAN  Slot {slot + 1} liegt außerhalb des Plans "
                f"(1..{TOTAL_SLOTS}) - Labels werden in_plan=false."]

    scene = scene_of_slot(slot)
    lines = [
        f"PLAN  {lab['code']}   {scene.title}",
        f"  Boxposition    {lab['position']}  {lab['position_label']}",
        f"  Folienlage     {lab['folienlage']}  {lab['folienlage_label']}",
        f"  Take           {attempt} von {TAKES_PER_LAGE} geplant"
        + (f"   ({done_takes} sauber)" if done_takes else ""),
        f"  Hand           {'MIT Handschuh' if lab['handschuhe'] else 'ohne Handschuh'}"
        f"   ·  Licht {lab['licht']}",
    ]
    # Reminders only before the first take: rebuilding between takes of one
    # layer would change the setup mid-slot.
    if attempt == 1:
        if lab["neue_szene_before"]:
            lines.append(f"  ** Szene aufbauen: {scene.title} **")
        elif lab["verschieben_before"]:
            lines.append(f"  ** Box verschieben -> {lab['position_label']}, "
                         f"Stapel neu aufbauen **")
        elif lab["neuaufbau_before"]:
            lines.append("  ** Stapel neu aufbauen (alle Lagen frisch) **")

        # Remount AFTER the build-up reminder: the camera comes off while the
        # box is being rebuilt anyway, so the two belong to the same pause.
        if lab["remount_before"]:
            lines.append(f"  ** WRIST-REMOUNT: abschrauben und neu ansetzen "
                         f"({REMOUNT_TOLERANZ}) **")
            lines.append("     danach ArUco-Bild am Boxrand ablegen. "
                         "Ausserhalb der Toleranz nochmal ansetzen, "
                         "NICHT nachkalibrieren.")
        if lab["top_cam_check_before"]:
            lines.append("  ** Neue Boxhoehe: Top-Cam leicht antasten **")
    return lines


def one_line(slot: int) -> str:
    """Compact summary for logs and tables."""
    lab = labels_for(slot)
    if lab is None:
        return f"s{slot + 1:03d}  <außerhalb des Plans>"
    return (f"{lab['code']:<22} {lab['boxhoehe_label']:<24} "
            f"{lab['bauteil_label']:<12} {lab['position']} "
            f"{lab['folienlage']}  {lab['folienlage_label']:<38} "
            f"{'Handschuh' if lab['handschuhe'] else 'bloss    '} "
            f"{lab['licht']:<8}"
            + ("  REMOUNT" if lab["remount_before"] else ""))


def plan_summary() -> str:
    return "\n".join([
        f"  {len(HOEHEN_ORDER)} Boxhöhen x {len(BAUTEIL_ORDER)} Bauteile "
        f"= {NUM_SCENES} Szenen",
        f"  pro Szene: {len(POSITION_ORDER)} Positionen x "
        f"{len(LAGEN_ORDER)} Lagen = {SLOTS_PER_SCENE} Slots",
        f"  pro Slot : {TAKES_PER_LAGE} Takes geplant "
        f"-> {TAKES_PER_SCENE} Episoden/Szene",
        f"  gesamt   : {TOTAL_SLOTS} Slots, ~{TOTAL_TAKES} Episoden",
    ])


def _main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scenes", action="store_true", help="Szenenübersicht")
    g.add_argument("--scene", help="z.B. mitte/a_saeule")
    g.add_argument("--slot", type=int, help="1-basierte Slotnummer")
    g.add_argument("--list", action="store_true", help="alle Slots")
    a = ap.parse_args()

    if a.scenes:
        print(f"\n{plan_summary()}\n")
        for h in HOEHEN_ORDER:
            print(f"  {BOXHOEHEN[h]['label']}")
            for sc in scenes_at(h):
                r = sc.slots
                lab = labels_for(r.start)
                print(f"    {sc.name:<24} s{r.start + 1:03d}-s{r.stop:03d}"
                      f"   {TAKES_PER_SCENE} Episoden   "
                      f"{'Handschuh' if lab['handschuhe'] else 'bloss    '} "
                      f"{lab['licht']:<8} "
                      f"FOLDER_NAME={sc.folder}")
        return

    if a.scene:
        sc = scene_by_name(a.scene)
        if sc is None:
            raise SystemExit(
                f"unbekannte Szene {a.scene!r} - erwartet "
                f"<{'|'.join(HOEHEN_ORDER)}>/<{'|'.join(BAUTEIL_ORDER)}>")
        print(f"\n=== {sc.title} ===")
        print(f"  PLAN_SCENE={sc.name}   FOLDER_NAME={sc.folder}\n")
        for s in sc.slots:
            print(f"  {one_line(s)}")
        return

    if a.slot is not None:
        print("\n".join(setup_lines(a.slot - 1)))
        return

    for sc in SCENES:
        print(f"\n=== {sc.title} ===")
        for s in sc.slots:
            print(f"  {one_line(s)}")


if __name__ == "__main__":
    _main()