"""Smoke-check the metadata + subsample wiring against the real data.

Run inside the ros2_dev container:
    python3 /ros2_ws/data/dataset_builder/tests/check_metadata_wiring.py

Checks the parts that a dry build would not reach: that both configs parse,
that labels.json is read and mapped for the new collections, that the old
collections still produce a usable (all-sentinel) record, and that the 35%
cap lands where the comment claims it does.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/ros2_ws/data/dataset_builder")

from config import load_config
from metadata import (
    build_prehensio_record,
    build_zarr_columns,
    load_episode_metadata,
)
from subsample import subsample_episodes

EXTERNAL = Path("/ros2_ws/data/episodes/external")
CONFIGS = ("foil_lifting_wrist", "foil_lifting_wrist_recorded_action")


class FakeRef:
    """Stand-in for EpisodeRef, so the sampler can be tested without IO."""

    def __init__(self, name, source_dir, scene):
        self.name = name
        self.source_dir = source_dir
        self.scene = scene


def check_configs():
    print("== configs")
    seeds = set()
    for name in CONFIGS:
        cfg = load_config(f"/ros2_ws/data/configs/{name}.yaml")
        sub = cfg.subsample_config
        dirs = [s["dir"] for s in cfg.cache_sources]
        folien = [d for d in dirs if d.startswith("folien_")]
        print(f"  {name}: {len(dirs)} sources ({len(folien)} folien_*), "
              f"{len(sub['source_fractions'])} capped, seed={sub['seed']}, "
              f"default={sub['default_fraction']}")
        assert len(folien) == 9, f"expected 9 folien_* dirs, got {len(folien)}"
        assert len(sub["source_fractions"]) == 4
        # The cap must come from the inline form now.
        inline = {s["dir"]: s["fraction"] for s in cfg.cache_sources
                  if s.get("fraction") is not None}
        assert inline == sub["source_fractions"], \
            "inline fractions did not reach subsample_config"
        # folien_* must carry no cap at all, not 1.0 by accident.
        for source in cfg.cache_sources:
            if source["dir"].startswith("folien_"):
                assert source["fraction"] is None, source["dir"]
        seeds.add(sub["seed"])

    # Same seed in both configs, or the A/B compares different subsets.
    assert len(seeds) == 1, f"configs disagree on subsample_seed: {seeds}"
    print(f"  both configs share subsample_seed={seeds.pop()}: ok")
    return load_config(f"/ros2_ws/data/configs/{CONFIGS[0]}.yaml")


def check_config_validation():
    """The inline/legacy forms must agree, and bad values must fail early."""
    print("== config validation")
    from config import DatasetConfig

    base = {
        "meta": {"name": "t", "task": "t", "fps": 20},
        "cache": {"base_dirs": [{"dir": "a", "fraction": 0.35},
                                {"dir": "b"}]},
        "episodes": {},
    }
    assert DatasetConfig(raw=base).subsample_config["source_fractions"] \
        == {"a": 0.35}
    print("  inline form: ok")

    # Legacy mapping still works for configs that use it.
    legacy = {
        "meta": base["meta"],
        "cache": {"base_dirs": [{"dir": "a"}, {"dir": "b"}]},
        "episodes": {"source_fractions": {"a": 0.35}},
    }
    assert DatasetConfig(raw=legacy).subsample_config["source_fractions"] \
        == {"a": 0.35}
    print("  legacy form still accepted: ok")

    def expect_error(raw, label):
        try:
            DatasetConfig(raw=raw).subsample_config
        except ValueError as e:
            print(f"  {label}: rejected ({str(e)[:60]}...)")
            return
        raise AssertionError(f"{label} was NOT rejected")

    # Two different answers for the same collection.
    expect_error({
        "meta": base["meta"],
        "cache": {"base_dirs": [{"dir": "a", "fraction": 0.35}]},
        "episodes": {"source_fractions": {"a": 0.5}},
    }, "conflicting fractions")

    # Cap naming a collection that is not built - usually a rename.
    expect_error({
        "meta": base["meta"],
        "cache": {"base_dirs": [{"dir": "a"}]},
        "episodes": {"source_fractions": {"typo": 0.35}},
    }, "unknown collection")

    # Out of range, and a mistyped flag.
    expect_error({
        "meta": base["meta"],
        "cache": {"base_dirs": [{"dir": "a", "fraction": 1.5}]},
        "episodes": {},
    }, "fraction > 1")
    expect_error({
        "meta": base["meta"],
        "cache": {"base_dirs": [{"dir": "a", "fraction": True}]},
        "episodes": {},
    }, "fraction: true")

    # Identical values in both forms are harmless, so they must NOT raise.
    same = {
        "meta": base["meta"],
        "cache": {"base_dirs": [{"dir": "a", "fraction": 0.35}]},
        "episodes": {"source_fractions": {"a": 0.35}},
    }
    assert DatasetConfig(raw=same).subsample_config["source_fractions"] \
        == {"a": 0.35}
    print("  agreeing duplicate: accepted")


def check_metadata():
    print("== metadata (new collection)")
    ep = EXTERNAL / "folien_mitte_a_saeule" / "episode_014_success"
    meta = load_episode_metadata(ep)
    for key in ("scene", "part", "box_height", "slot_position",
                "foil_layer", "plan_code", "block", "attempt"):
        print(f"  {key}: {meta[key]!r}")
    assert meta["has_labels"]
    assert meta["part"] == "foil_lifting_a_saeule"
    assert meta["box_height"] == "middle"
    assert meta["scene"] == "middle/foil_lifting_a_saeule"

    print("== metadata (old collection, no labels.json)")
    old_root = EXTERNAL / "foil_lifting_collection_tesollo_bottom"
    old_ep = sorted(p for p in old_root.iterdir() if p.is_dir())[0]
    old = load_episode_metadata(old_ep)
    print(f"  {old_ep.name}: has_labels={old['has_labels']}, "
          f"scene={old['scene']!r}, part={old['part']!r}, "
          f"block={old['block']}")
    assert old["has_labels"] is False
    assert old["scene"] == ""
    assert old["block"] == -1
    return meta, old


def check_export(new_meta, old_meta):
    print("== zarr columns (mixed old + new)")
    records = [new_meta, old_meta, new_meta]
    columns, vocab = build_zarr_columns(records)
    for key in ("scene_id", "part_id", "foil_layer_id", "has_labels", "block"):
        print(f"  {key}: {columns[key]}")
    print(f"  scene labels: {vocab['scene']}")
    for key, values in columns.items():
        assert len(values) == len(records), f"{key} is not aligned"
    # The unlabelled episode must code as -1, not as a real scene.
    assert columns["scene_id"][1] == -1

    print("== prehensio sidecar")
    rec = build_prehensio_record(
        new_meta, episode_index=0,
        source="folien_mitte_a_saeule/episode_014_success",
        dataset="foil_lifting_wrist", task="Pinch the foil and lift it",
        n_frames=294, fps=20, cell="bosch", pc="monty5-pc",
    )
    print(f"  part_id: {rec['part_id']!r}  outcome: {rec['outcome']!r}")
    print(f"  prehensio_labels: {sorted(rec.get('prehensio_labels', {}))}")
    # Viewer-mandatory keys must be present and flat.
    for key in ("source", "cell", "dataset", "task", "outcome", "part_id",
                "notes", "frames", "fps", "pc"):
        assert key in rec, f"viewer key {key} missing"

    old_rec = build_prehensio_record(
        old_meta, episode_index=1, source="old/episode_000_success",
        dataset="foil_lifting_wrist", task="Pinch the foil and lift it",
        n_frames=100, fps=20,
    )
    # No labels -> no extras block, so old datasets stay as the viewer
    # has always seen them.
    assert "prehensio_labels" not in old_rec
    print("  old episode: no prehensio_labels block (as intended)")


def check_subsample(cfg):
    print("== subsample (real episode counts)")
    refs = []
    for source in cfg.cache_sources:
        root = EXTERNAL / source["dir"]
        if not root.exists():
            print(f"  !! missing on disk: {source['dir']}")
            continue
        for ep in sorted(p for p in root.iterdir() if p.is_dir()):
            meta = load_episode_metadata(ep)
            scene = meta["scene"] or source["dir"]
            refs.append(FakeRef(f"{source['dir']}/{ep.name}",
                                source["dir"], scene))

    sub = cfg.subsample_config
    kept = subsample_episodes(
        refs,
        fractions=sub["source_fractions"],
        default_fraction=sub["default_fraction"],
        seed=sub["seed"],
    )

    def split(items):
        old = sum(1 for r in items if r.source_dir.startswith("foil_lifting"))
        return old, len(items) - old

    old_before, new_before = split(refs)
    old_after, new_after = split(kept)
    print(f"  before: {old_before} old + {new_before} new = {len(refs)}")
    print(f"  after:  {old_after} old + {new_after} new = {len(kept)}")
    if kept:
        print(f"  mixture: {old_after / len(kept):.0%} old / "
              f"{new_after / len(kept):.0%} new")

    # New collections are uncapped; old ones must actually shrink.
    assert new_after == new_before, "folien_* episodes must not be dropped"
    assert old_after < old_before, "old collections were not capped"

    # Determinism: same inputs -> same selection.
    again = subsample_episodes(
        refs, fractions=sub["source_fractions"],
        default_fraction=sub["default_fraction"], seed=sub["seed"],
    )
    assert [r.name for r in again] == [r.name for r in kept], \
        "subsample is not deterministic"
    print("  deterministic across two runs: ok")

    # Order must match discovery order, or zarr episode_ends stop being
    # comparable to an unsampled build.
    order = [r.name for r in refs if r in set(kept)]
    assert [r.name for r in kept] == order, "discovery order not preserved"
    print("  discovery order preserved: ok")


def main():
    cfg = check_configs()
    check_config_validation()
    new_meta, old_meta = check_metadata()
    check_export(new_meta, old_meta)
    check_subsample(cfg)
    print("\nall checks passed")


if __name__ == "__main__":
    main()