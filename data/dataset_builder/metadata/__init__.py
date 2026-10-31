# dataset_builder/metadata/__init__.py
"""Per-episode metadata: reading it, and handing it to the writers.

Two halves, deliberately split:

  * `labels` - reads the raw `labels.json` sidecar an episode folder may
    carry and normalises its German keys/values into one flat record with a
    fixed key set. This is the ONLY place that knows the raw file's schema.

  * `export`  - reshapes those records into what each writer needs: fixed
    width int columns + vocabularies for zarr, the viewer's own JSON schema
    for LeRobot. Knows nothing about labels.json.

Re-exported here so callers write `from metadata import load_episode_metadata`
and never depend on which of the two files something lives in.
"""

from metadata.labels import (
    BOX_HEIGHT_MAP,
    FOIL_LAYER_MAP,
    INT_FIELDS,
    PART_MAP,
    PART_PREFIX,
    SCENE_FIELD,
    SLOT_POSITION_MAP,
    STRING_FIELDS,
    UNKNOWN_ID,
    UNKNOWN_STR,
    empty_metadata,
    load_episode_metadata,
    scene_key,
)
from metadata.export import (
    CATEGORICAL_FIELDS,
    build_prehensio_jsonl_row,
    build_prehensio_record,
    build_zarr_columns,
)

__all__ = [
    "BOX_HEIGHT_MAP",
    "CATEGORICAL_FIELDS",
    "FOIL_LAYER_MAP",
    "INT_FIELDS",
    "PART_MAP",
    "PART_PREFIX",
    "SCENE_FIELD",
    "SLOT_POSITION_MAP",
    "STRING_FIELDS",
    "UNKNOWN_ID",
    "UNKNOWN_STR",
    "build_prehensio_jsonl_row",
    "build_prehensio_record",
    "build_zarr_columns",
    "empty_metadata",
    "load_episode_metadata",
    "scene_key",
]