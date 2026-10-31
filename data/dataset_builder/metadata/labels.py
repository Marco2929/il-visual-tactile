# dataset_builder/metadata.py
"""Per-episode metadata extraction from the raw episode folders.

The newer `folien_*` collections ship a `labels.json` next to state.npy that
describes the recording condition far more precisely than the single
`position` string the older collections were labelled with by hand in the
task YAML:

    {"scene": "mitte/a_saeule", "boxhoehe": "mitte", "bauteil": "a_saeule",
     "position": "P2", "folienlage": "L2", "block": 1, "attempt": 3, ...}

Those German keys/values are normalised to a stable dataset vocabulary here
(see STRING_FIELDS) - field names in English, part names kept German but
prefixed, e.g. bauteil="a_saeule" -> part="foil_lifting_a_saeule".

That is exactly the information a later per-scene / per-foil-layer split
needs, so it must survive into the datasets instead of being re-derived from
directory names. Older collections have no labels.json at all - they get the
same dict with every field at its "unknown" sentinel, so the per-episode
columns stay index-aligned with meta/episode_ends either way. Nothing here
raises on a missing or malformed file: a broken sidecar must not be able to
fail a multi-hour build.

Two consumers:
  * ZarrWriter  - writes the categorical/integer fields as per-episode arrays
  * LeRobotWriter - mirrors the full dict into meta/prehensio/, the layout
    the Prehensio viewer reads (see /home/mm/prehensio_data/datasets).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

# Mirrors ZarrWriter.UNKNOWN_ID. Duplicated rather than imported to keep this
# module free of writer imports (discovery.py pulls it in long before any
# writer exists).
UNKNOWN_ID = -1
UNKNOWN_STR = ""

# ─── German source vocabulary -> canonical dataset vocabulary ─────────────
# labels.json is written by the German-language recording tool, so both its
# keys and its values are German. Everything is renamed ONCE here, at the
# only point that reads the raw file, so no German *field name* ever reaches
# a zarr column, a LeRobot feature or a training config. Doing it later would
# mean every consumer carries its own copy of these tables.
#
# Field names are English. VALUES that name a physical part keep their German
# term on purpose - "a_saeule" is what the part is called on the line, in the
# recording plan and in the plan_code (HM-AS-...), and inventing "a_pillar"
# would break that trace for no gain. They are prefixed so a bare value is
# self-describing in a flat metadata dict.
PART_PREFIX = "foil_lifting_"

# Which part is stacked in the box. The foil is lifted off THIS part.
PART_MAP = {
    "a_saeule": f"{PART_PREFIX}a_saeule",
    "b_saeule": f"{PART_PREFIX}b_saeule",
    "heckleuchte": f"{PART_PREFIX}heckleuchte",
}

# Height of the box/pallet the part is taken from.
BOX_HEIGHT_MAP = {
    "mitte": "middle",   # starting height
    "oben": "upper",     # one level higher
    "unten": "lower",    # one level lower
}

# Slot inside the box, as seen by the robot. P1..P5 are opaque codes in the
# source file; the German `position_label` is what gives them meaning, so the
# mapping uses that rather than the bare code.
SLOT_POSITION_MAP = {
    "P1": "center",
    "P2": "front_left",
    "P3": "front_right",
    "P4": "rear_left",
    "P5": "rear_right",
}

# Which foil layer of the stack is lifted. NOTE the ordering: L1 is the
# TOPMOST layer (above the parts) and L3 the bottom one (on the cardboard),
# so the numbering counts downwards - the opposite of what "L1 = first =
# bottom" suggests. Hence an explicit table instead of an index.
FOIL_LAYER_MAP = {
    "L1": "top",      # oberste Folienlage, above the parts
    "L2": "middle",   # between the parts
    "L3": "bottom",   # on the cardboard, below the parts
}

# String fields, as `dataset_field: (source_key, value_map)`.
# An explicit whitelist, not "everything in the file": labels.json gains
# fields over time, and an unreviewed new key silently becoming a dataset
# column is how two builds stop being comparable.
STRING_FIELDS = {
    "part": ("bauteil", PART_MAP),
    "box_height": ("boxhoehe", BOX_HEIGHT_MAP),
    "slot_position": ("position", SLOT_POSITION_MAP),
    "foil_layer": ("folienlage", FOIL_LAYER_MAP),
    # Plan code, e.g. HM-AS-P2-L2_s005. Opaque by design - it is the key back
    # into the recording plan, so it must NOT be rewritten.
    "plan_code": ("code", None),
    "collection": ("collection", None),
    "outcome": ("episode_status", None),
    "recorded_at": ("recorded_at", None),
}

# `scene` is a composite "boxhoehe/bauteil" (e.g. "mitte/a_saeule"), so it is
# rebuilt from the already-mapped components rather than string-mapped.
SCENE_FIELD = "scene"

# Integer fields, as `dataset_field: source_key`.
INT_FIELDS = {
    "block": "block",
    "slot": "slot",
    "slot_no": "slot_no",
    "attempt": "attempt",
    "episode_idx": "episode_idx",
    "labels_schema_version": "schema_version",
}


def empty_metadata() -> Dict[str, Any]:
    """The 'no labels.json' record - all sentinels, never None values.

    Returning a fully-populated dict instead of None is deliberate: callers
    build fixed-width per-episode columns from these keys, so a None here
    would turn into a ragged array or a KeyError deep inside a writer.
    """
    meta: Dict[str, Any] = {"has_labels": False}
    meta.update({f: UNKNOWN_STR for f in STRING_FIELDS})
    meta[SCENE_FIELD] = UNKNOWN_STR
    meta["qa_status"] = UNKNOWN_STR
    meta.update({f: UNKNOWN_ID for f in INT_FIELDS})
    return meta


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return UNKNOWN_ID
    return UNKNOWN_ID


def _as_str(value: Any) -> str:
    if value is None:
        return UNKNOWN_STR
    if isinstance(value, str):
        return value
    return str(value)


def _map_value(value: str, value_map: Optional[Dict[str, str]]) -> str:
    """Map one raw source value to its canonical dataset form.

    An UNRECOGNISED value is passed through unchanged rather than replaced by
    a sentinel: a new part or box height appearing in the recordings should
    show up verbatim in the dataset (and so be noticeable), not be erased.
    """
    if not value or value_map is None:
        return value
    return value_map.get(value, value)


def _map_scene(scene: str) -> str:
    """Map a raw "boxhoehe/bauteil" scene string component-wise.

    Only a fallback for records that carry `scene` but not the two fields it
    is built from.
    """
    if not scene:
        return UNKNOWN_STR
    parts = scene.split("/")
    if len(parts) != 2:
        return scene
    height, part = parts
    return f"{_map_value(height, BOX_HEIGHT_MAP)}/{_map_value(part, PART_MAP)}"


def load_episode_metadata(episode_dir: Path) -> Dict[str, Any]:
    """Read labels.json (+ the QA status) for one episode.

    Args:
        episode_dir: raw episode folder.

    Returns:
        Flat dict with a fixed key set (see empty_metadata). `has_labels`
        tells a consumer whether any of it came from disk, which is what the
        old collections rely on to stay buildable.

        `raw_labels` additionally carries the UNFILTERED file contents for
        the LeRobot sidecar, where there is no fixed-width constraint and
        dropping operator notes would be a pointless loss.
    """
    meta = empty_metadata()
    labels_path = Path(episode_dir) / "labels.json"

    if not labels_path.exists():
        meta["raw_labels"] = {}
        return meta

    try:
        with open(labels_path, "r") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Deliberately not fatal - see module docstring.
        meta["raw_labels"] = {}
        return meta

    if not isinstance(raw, dict):
        meta["raw_labels"] = {}
        return meta

    meta["has_labels"] = True
    meta["raw_labels"] = raw

    for name, (source_key, value_map) in STRING_FIELDS.items():
        if source_key in raw:
            meta[name] = _map_value(_as_str(raw[source_key]), value_map)
    for name, source_key in INT_FIELDS.items():
        if source_key in raw:
            meta[name] = _as_int(raw[source_key])

    # Rebuilt from the mapped components so the scene label agrees with the
    # `box_height` / `part` columns exactly. Re-deriving instead of mapping
    # the raw "mitte/a_saeule" means a new box height or part propagates here
    # automatically, with no second table to forget to update.
    if meta["box_height"] and meta["part"]:
        meta[SCENE_FIELD] = f"{meta['box_height']}/{meta['part']}"
    else:
        meta[SCENE_FIELD] = _map_scene(_as_str(raw.get(SCENE_FIELD, "")))

    # QA status is a separate file but belongs to the same provenance record:
    # it is the difference between "this episode is clean" and "this episode
    # had an 86ms sync gap", which you want when a trained policy behaves
    # oddly on a subset.
    meta["qa_status"] = _read_qa_status(episode_dir)

    return meta


def _read_qa_status(episode_dir: Path) -> str:
    """PASS / WARN / FAIL from qa_report.json, or "" when absent."""
    qa_path = Path(episode_dir) / "qa_report.json"
    if not qa_path.exists():
        return UNKNOWN_STR
    try:
        with open(qa_path, "r") as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError):
        return UNKNOWN_STR
    if isinstance(report, dict):
        return _as_str(report.get("status", UNKNOWN_STR))
    return UNKNOWN_STR


def scene_key(meta: Dict[str, Any], fallback: Optional[str] = None) -> str:
    """Best available scene identifier for grouping/stratifying.

    Prefers labels.json's `scene`, falls back to the YAML `position` label,
    and finally to the collection directory. Having one function own this
    precedence keeps the zarr column, the LeRobot sidecar and the CLI summary
    from disagreeing about which episodes form a scene.
    """
    scene = meta.get(SCENE_FIELD) or UNKNOWN_STR
    if scene:
        return scene
    return fallback or UNKNOWN_STR