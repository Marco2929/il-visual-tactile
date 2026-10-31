# dataset_builder/subsample.py
"""Per-source episode subsampling.

Motivation: the older `foil_lifting_collection_tesollo_*` collections show a
DIFFERENT scene than the new `folien_*` ones while carrying the same task. At
full weight they would dominate the mixture and teach the policy that scene's
layout rather than the task, so they are capped to a fraction (35%) instead
of being dropped - they still contribute pose/grasp variety, just not a
majority of the gradient.

Design choices worth stating, because each of them is a trap otherwise:

  * The fraction is taken PER SCENE, not off a flat list. `episodes[:n]`
    would keep whole collections and delete others outright, since discovery
    returns episodes grouped by source.
  * Selection is DETERMINISTIC given (seed, fraction, episode keys), and the
    key - not the list index - is what gets hashed/sorted. Adding or
    re-extracting one episode therefore does not reshuffle which of the
    others were kept, so two builds stay comparable.
  * Rounding is round-half-up with a floor of 1 kept episode per group: a
    small scene must not silently vanish at a low fraction.
"""

import random
from typing import Any, Callable, Dict, List, Optional

from termcolor import cprint


def _group_key(ref) -> str:
    """Stratification group: the scene when known, else the collection.

    Scene rather than collection is the right granularity for the new data:
    one `folien_*` collection is one scene, but that is a convention of how
    they happened to be recorded, and labels.json is the authority.
    """
    scene = getattr(ref, "scene", "") or ""
    return scene or ref.source_dir


def _n_keep(n_total: int, fraction: float) -> int:
    """How many of `n_total` to keep, rounding half up, never below 1."""
    if n_total <= 0:
        return 0
    if fraction >= 1.0:
        return n_total
    if fraction <= 0.0:
        return 0
    # +0.5 = round half up. int() alone truncates, which at 35% of 3
    # episodes would keep 1 instead of 1 - harmless - but at 35% of 23 would
    # keep 8 rather than 8; the floor below is the part that actually matters.
    n = int(n_total * fraction + 0.5)
    return max(1, min(n_total, n))


def subsample_episodes(
    episodes: List,
    fractions: Dict[str, float],
    default_fraction: Optional[float] = None,
    seed: int = 0,
    group_fn: Callable[[Any], str] = _group_key,
) -> List:
    """Keep a deterministic fraction of episodes, stratified by scene.

    Args:
        episodes: EpisodeRefs from discovery, in discovery order.
        fractions: per-collection fraction in [0, 1], keyed by the collection
            directory (``cache.base_dirs[].dir``). Takes precedence over
            `default_fraction`, so the old collections can be capped while
            everything else stays whole.
        default_fraction: fraction applied to collections absent from
            `fractions`. None means "keep all", which is what makes this
            feature opt-in and the existing configs byte-identical.
        seed: selection seed. Part of the dataset's identity - record it.
        group_fn: stratification key. Overridable for tests.

    Returns:
        The kept episodes, in the ORIGINAL discovery order. Order is restored
        on purpose: writers assign episode indices in the order they are
        appended, and a shuffled dataset would make zarr episode_ends
        non-comparable to an unsampled build of the same sources.
    """
    if not episodes:
        return []

    # Nothing configured -> no-op, and no log line pretending otherwise.
    if not fractions and default_fraction is None:
        return list(episodes)

    # Group by (collection, scene): the fraction is configured per
    # collection, but applied within each of its scenes.
    groups: Dict[tuple, List] = {}
    for ref in episodes:
        groups.setdefault((ref.source_dir, group_fn(ref)), []).append(ref)

    keep_keys = set()
    report: Dict[str, List[int]] = {}

    for (source_dir, group), refs in groups.items():
        fraction = fractions.get(source_dir, default_fraction)
        if fraction is None:
            fraction = 1.0
        fraction = float(fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"episodes.source_fractions[{source_dir!r}] must be in "
                f"[0, 1], got {fraction}"
            )

        n_keep = _n_keep(len(refs), fraction)

        if n_keep == len(refs):
            chosen = refs
        else:
            # Sort by key first so the input order cannot affect the draw,
            # then sample with a per-group seed. Seeding per group (rather
            # than once globally) means changing one collection's fraction
            # does not re-draw the others.
            ordered = sorted(refs, key=lambda r: r.name)
            rng = random.Random(f"{seed}|{source_dir}|{group}")
            chosen = rng.sample(ordered, n_keep)

        keep_keys.update(r.name for r in chosen)

        totals = report.setdefault(source_dir, [0, 0])
        totals[0] += len(chosen)
        totals[1] += len(refs)

    kept = [ref for ref in episodes if ref.name in keep_keys]

    for source_dir, (n_kept, n_total) in sorted(report.items()):
        if n_kept == n_total:
            continue
        cprint(
            f"     ✂️  {source_dir}: {n_kept}/{n_total} episodes "
            f"({n_kept / max(n_total, 1):.0%}, seed={seed})",
            "yellow",
        )

    return kept