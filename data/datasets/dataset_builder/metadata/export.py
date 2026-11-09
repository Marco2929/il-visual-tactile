# dataset_builder/metadata/export.py
"""Turn per-episode metadata records into what each writer wants.

`metadata.load_episode_metadata` produces one flat dict per episode. The two
writers need very different shapes of the same information:

  * Zarr wants fixed-width numeric COLUMNS, one entry per episode, aligned
    with meta/episode_ends. Strings cannot go in a plain int array, so each
    categorical field becomes an int code column plus a vocabulary list in
    the group attrs. That keeps a dataloader's fast path integer-only while
    still being self-describing - you can read the labels back without the
    builder config.

  * LeRobot's Prehensio viewer wants one JSON per episode plus an append-only
    episodes.jsonl, with ITS key names (`part_id`, `outcome`, `source`, ...).
    That schema is fixed by the viewer, not by us - see
    /home/mm/prehensio_data/datasets/foil_lifting_wrist/meta/prehensio/.

Keeping both mappings here means neither writer knows about labels.json, and
the two can never disagree about what an episode's scene was.
"""

from typing import Any, Dict, List, Tuple

from metadata.labels import (
    INT_FIELDS,
    SCENE_FIELD,
    UNKNOWN_ID,
    UNKNOWN_STR,
)

# Categorical fields that become int code columns in zarr. Deliberately NOT
# every string field: `plan_code` and `recorded_at` are near-unique per
# episode, so coding them would produce a vocabulary as long as the dataset.
# Those stay in the LeRobot sidecar only.
CATEGORICAL_FIELDS = (
    SCENE_FIELD,
    "part",
    "box_height",
    "slot_position",
    "foil_layer",
    "outcome",
    "qa_status",
)


def build_zarr_columns(
    records: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[int]], Dict[str, List[str]]]:
    """Build integer columns + vocabularies from per-episode records.

    Args:
        records: one metadata dict per episode, in episode order. Entries may
            be empty dicts (old collections) - they code as UNKNOWN_ID.

    Returns:
        (columns, vocabularies)
          columns: field -> list of ints, one per episode. Categorical
            fields are suffixed `_id`; integer fields keep their name.
          vocabularies: field -> list of label strings, where index == code.
            Code UNKNOWN_ID (-1) is NOT in the list; it means "unknown", so
            a reader must treat -1 specially rather than indexing with it.

    The vocabulary is built in FIRST-SEEN order rather than sorted. Sorting
    would look tidier but makes the codes depend on which episodes happen to
    be in the build, so appending a new scene could renumber the existing
    ones and silently invalidate anything that stored a raw code.
    """
    vocabularies: Dict[str, List[str]] = {f: [] for f in CATEGORICAL_FIELDS}
    columns: Dict[str, List[int]] = {}

    for field in CATEGORICAL_FIELDS:
        codes = []
        vocab = vocabularies[field]
        for record in records:
            value = (record or {}).get(field) or UNKNOWN_STR
            if not value:
                codes.append(UNKNOWN_ID)
                continue
            if value not in vocab:
                vocab.append(value)
            codes.append(vocab.index(value))
        columns[f"{field}_id"] = codes

    for field in INT_FIELDS:
        columns[field] = [
            int((record or {}).get(field, UNKNOWN_ID)) for record in records
        ]

    # Lets a consumer split old from new data without reverse-engineering it
    # from sentinel values in the other columns.
    columns["has_labels"] = [
        1 if (record or {}).get("has_labels") else 0 for record in records
    ]

    return columns, vocabularies


# ─── Prehensio viewer sidecar ─────────────────────────────────────────────
# Key names below are the VIEWER's, taken from an existing dataset. Do not
# "tidy" them to match our internal names or the viewer stops reading them.

def build_prehensio_record(
    record: Dict[str, Any],
    *,
    episode_index: int,
    source: str,
    dataset: str,
    task: str,
    n_frames: int,
    fps: int,
    cell: str = "bosch",
    pc: str = "",
) -> Dict[str, Any]:
    """Build the per-episode JSON the Prehensio viewer reads.

    The viewer's own fields come first and keep their exact names; everything
    labels.json adds that the viewer does not know about goes under a nested
    `prehensio_labels` key. Nesting rather than flattening is what keeps the
    viewer forward-compatible: it ignores the extra block, while a later
    analysis script still has the full record.
    """
    record = record or {}

    out: Dict[str, Any] = {
        "source": source,
        "cell": cell,
        "dataset": dataset,
        "task": task,
        "outcome": record.get("outcome") or "success",
        "part_id": record.get("part") or UNKNOWN_STR,
        "notes": "",
        "frames": int(n_frames),
        "fps": int(fps),
        "pc": pc,
    }

    # Only emit the extras when there is something to say, so datasets built
    # purely from the old collections stay byte-identical to what the viewer
    # has always received.
    extras = {
        key: record[key]
        for key in (
            SCENE_FIELD, "box_height", "slot_position", "foil_layer",
            "plan_code", "collection", "recorded_at", "qa_status",
        )
        if record.get(key)
    }
    extras.update({
        key: record[key]
        for key in INT_FIELDS
        if record.get(key, UNKNOWN_ID) != UNKNOWN_ID
    })
    if extras:
        out["prehensio_labels"] = extras

    return out


def build_prehensio_jsonl_row(
    record: Dict[str, Any],
    *,
    episode_index: int,
    task: str,
    n_frames: int,
    fps: int,
    cell: str = "bosch",
    pc: str = "",
) -> Dict[str, Any]:
    """One row of meta/prehensio/episodes.jsonl (the viewer's index file)."""
    record = record or {}

    row: Dict[str, Any] = {
        "episode_index": int(episode_index),
        "cell": cell,
        "pc": pc,
        "recording": "",
        "frames": int(n_frames),
        "fps": int(fps),
        "task_raw": task,
        "outcome": record.get("outcome") or "success",
        "part_id": record.get("part") or UNKNOWN_STR,
        "notes": "",
        "bundle": "",
    }

    # The scene is the one extra the index file really needs: it is what you
    # filter on in the viewer's episode list, and digging into 350 individual
    # JSONs to find a scene is not a workflow.
    scene = record.get(SCENE_FIELD)
    if scene:
        row["scene"] = scene
    if record.get("foil_layer"):
        row["foil_layer"] = record["foil_layer"]

    return row