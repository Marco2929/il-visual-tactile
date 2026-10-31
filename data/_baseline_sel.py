"""Throwaway: capture the pre-change episode selection for the old collections."""
import json

from config import load_config
from discovery import discover_sources
from subsample import subsample_episodes

OLD = (
    "foil_lifting_collection_tesollo_bottom",
    "foil_lifting_collection_tesollo_middle",
    "foil_lifting_collection_tesollo_top_left",
    "foil_lifting_collection_tesollo_top_right",
)

cfg = load_config("foil_lifting_wrist_recorded_action.yaml")
eps = discover_sources(cfg)

pre = {}
for e in eps:
    if e.source_dir in OLD:
        pre.setdefault(e.source_dir, []).append((e.scene, str(e.episode_dir)))

kept = subsample_episodes(eps, cfg)
post = {}
for e in kept:
    if e.source_dir in OLD:
        post.setdefault(e.source_dir, []).append(str(e.episode_dir))

out = {
    "scenes": {k: sorted({s for s, _ in v}) for k, v in pre.items()},
    "discovered_counts": {k: len(v) for k, v in pre.items()},
    "kept_counts": {k: len(v) for k, v in post.items()},
    "kept": {k: sorted(v) for k, v in post.items()},
    "total_discovered_all": len(eps),
    "total_kept_all": len(kept),
}
print(json.dumps({k: v for k, v in out.items() if k != "kept"}, indent=1))
with open("/ros2_ws/data/_baseline_sel.json", "w") as fh:
    json.dump(out, fh, indent=1)
