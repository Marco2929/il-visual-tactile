#!/usr/bin/env python3
"""Which rollout belongs to which test, stated explicitly.

NOT parsed from meta.json's `notes`. Three reasons, all of them learned from
the files themselves:

  * notes carry neither the OUTCOME nor the NUMBER OF ATTEMPTS, and both are
    part of the experiment design. rollout_006 and rollout_007 have the
    byte-identical note "Palette zwischenhohe dunne folie", but 006 retried
    and 007 was a single attempt - the whole point of comparing them.
  * the free text is inconsistent: trailing "\n" on 000/005, a typo'd
    "zwischenhohe", and rollout_027 reads "Palette dunne folie zu niedrig"
    with no layer word at all although it is a LOW-layer run.
  * 016, 020, 021, 026 and 028 no longer exist on disk. A parser over the
    directory listing silently renumbers nothing, but a hand-written range
    like 22..25 would quietly pick up a missing 26 if one reappeared.

So the mapping is data, reviewed once, and every analysis reads it from here.

PALLET OFFSETS ARE NOT RECORDED. The real left/right displacement in mm and
the real height deltas were not logged, so `offset_mm` and `height_mm` stay
None and every generalisation statement from them is qualitative: we can say
"the policy moved X mm" but not "it should have moved Y". Noted here rather
than in each script so the limitation cannot be read as an oversight.
"""
import pathlib

from .data import Rollout

CKPT = "epoch=0040-val_action_mse=9.478029"


class Group:
    """One test condition: the rollouts in it plus how to read them.

    `separate=True` means the members are to be reported ROW BY ROW and not
    averaged. The height and mid-layer tests are diagnostic - the question is
    what each run did, and a mean over four runs that stalled at different
    depths describes none of them.
    """

    def __init__(self, key, ids, label, layer, height="nominal", xy=None,
                 tries="single", outcome=None, separate=False,
                 offset_mm=None, height_mm=None, role="context"):
        self.key = key
        self.ids = list(ids)
        self.label = label
        self.layer = layer          # nominal | mid | low
        self.height = height        # nominal | too_high | too_low
        self.xy = xy                # mitte | links | rechts | None
        self.tries = tries          # single | multi
        self.outcome = outcome      # top | partial | fail | None
        self.separate = separate
        self.offset_mm = offset_mm  # real pallet displacement: NOT LOGGED
        self.height_mm = height_mm  # real height delta: NOT LOGGED
        self.role = role            # baseline | xy | height | mid | context

    @property
    def names(self):
        return [f"rollout_{i:03d}" for i in self.ids]

    def load(self, root):
        """-> [Rollout]. Missing or unloadable members are reported by the
        caller, not swallowed: a group quietly short one member changes every
        spread figure computed from it."""
        out = []
        for name in self.names:
            d = pathlib.Path(root) / name
            if not d.is_dir():
                out.append((name, None, "directory does not exist"))
                continue
            r = Rollout(d)
            out.append((name, r, None if r.ok else r.error))
        return out

    def __repr__(self):
        return f"<Group {self.key}: {len(self.ids)} rollouts>"


# ---------------------------------------------------------------------------
# T1 - proprioception overfitting baseline
#
# Same layer, same setup, SLIGHTLY DIFFERENT PALLET POSITION each time (the
# pallet was not re-placed to an exact mark). That difference is what makes
# this a test rather than a repeatability check: if the commanded
# trajectories come out near-identical anyway, the policy is not reading the
# pallet out of the image, it is replaying a path conditioned on
# proprioception.
#
# 005 had thin foil. Kept in the group on purpose - if it falls inside the
# spread of 000..003 then foil thickness is not a behavioural variable
# either, which is the same question asked about a different nuisance factor.
# ---------------------------------------------------------------------------
NOMINAL_BASELINE = Group(
    "nominal_baseline", [0, 1, 2, 3, 5],
    "nominal, one attempt, pallet position varies slightly",
    layer="nominal", tries="single", outcome="top", role="baseline")

# ---------------------------------------------------------------------------
# T2 - the mid-layer problem. Reported per rollout.
#
# 006 retried (rarely), 007..009 were one attempt each. Same note text, so
# the split exists only here.
# ---------------------------------------------------------------------------
MID_SINGLE = Group(
    "mid_single", [7, 8, 9], "mid layer, one attempt",
    layer="mid", tries="single", separate=True, role="mid")

MID_RETRY = Group(
    "mid_retry", [6], "mid layer, retried but not often",
    layer="mid", tries="multi", separate=True, role="mid")

# ---------------------------------------------------------------------------
# T3 - XY. Several attempts per run, so these are evaluated PER ATTEMPT.
# 013/014/015 are the reference for the XY question (mid layer, pallet moved
# centre / left / right).
# ---------------------------------------------------------------------------
XY_MITTE = Group("xy_mitte", [13], "mid layer, pallet centre", layer="mid",
                 xy="mitte", tries="multi", role="xy")
XY_LINKS = Group("xy_links", [14], "mid layer, pallet left", layer="mid",
                 xy="links", tries="multi", role="xy")
XY_RECHTS = Group("xy_rechts", [15], "mid layer, pallet right", layer="mid",
                  xy="rechts", tries="multi", role="xy")

# ---------------------------------------------------------------------------
# T4 - height. Each run read on its own row.
# ---------------------------------------------------------------------------
MID_TOO_HIGH = Group(
    "mid_too_high", [17, 18, 19], "mid layer, pallet too HIGH",
    layer="mid", height="too_high", separate=True, role="height")

MID_TOO_LOW = Group(
    "mid_too_low", [22, 23, 24, 25], "mid layer, pallet too LOW",
    layer="mid", height="too_low", separate=True, role="height")

LOW_TOO_LOW = Group(
    "low_too_low", [27, 29], "low layer, pallet too LOW",
    layer="low", height="too_low", separate=True, role="height")

# ---------------------------------------------------------------------------
# Context: not a test on their own, but useful as a comparison for the
# retry/reactivity question and for what a long successful run looks like.
# ---------------------------------------------------------------------------
NOMINAL_RETRY = Group(
    "nominal_retry", [4], "nominal, very long, several retries, succeeded",
    layer="nominal", tries="multi", outcome="top", separate=True)

LOW_RETRY = Group(
    "low_retry", [10], "low layer, long, retried",
    layer="low", tries="multi", separate=True)

SINGLE_11_12 = Group(
    "single_11_12", [11, 12], "one attempt each",
    layer="unknown", tries="single", separate=True)


ALL = [NOMINAL_BASELINE, MID_RETRY, MID_SINGLE, XY_MITTE, XY_LINKS,
       XY_RECHTS, MID_TOO_HIGH, MID_TOO_LOW, LOW_TOO_LOW, NOMINAL_RETRY,
       LOW_RETRY, SINGLE_11_12]

BY_KEY = {g.key: g for g in ALL}

XY_GROUPS = [XY_MITTE, XY_LINKS, XY_RECHTS]
HEIGHT_GROUPS = [MID_TOO_HIGH, MID_TOO_LOW, LOW_TOO_LOW]
MID_GROUPS = [MID_RETRY, MID_SINGLE]


def root_for(ckpt=CKPT, base=None):
    """Checkpoint directory. Defaults to the one next to this checkout, so
    the scripts work without arguments the way the existing ones do."""
    base = pathlib.Path(base) if base else pathlib.Path(__file__).parents[1]
    return base / ckpt


def load_group(group, root=None, ckpt=CKPT, verbose=True):
    """-> [Rollout] that loaded, printing the ones that did not.

    Failures are printed rather than raised: one unreadable member should not
    cost the other four, but it must not vanish either, because every spread
    figure below is computed over however many rollouts came back.
    """
    root = root or root_for(ckpt)
    ok, bad = [], []
    for name, r, err in group.load(root):
        if err:
            bad.append((name, err))
        else:
            ok.append(r)
    if verbose and bad:
        for name, err in bad:
            print(f"  !! {name}: {err}")
    return ok


def load_many(groups, root=None, ckpt=CKPT, verbose=True):
    """-> [(group, [Rollout])] in the given order."""
    root = root or root_for(ckpt)
    return [(g, load_group(g, root=root, verbose=verbose)) for g in groups]