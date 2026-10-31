"""Inference-time modality ablation for ManiFlowMultimodalPolicy.

Answers "what does the policy actually USE?", which gradient magnitudes only
answer indirectly. One checkpoint, held-out episodes, no training.

Each variant reuses the SAME encoded conditioning and only manipulates one
tensor before sampling, so the difference is attributable to that modality
alone (same batch, same noise seed, same ODE steps).

    python scripts/ablate_modalities.py \
        --ckpt data/outputs/<run>/checkpoints/epoch=0010-val_loss=0.059693.ckpt \
        --n-batches 10

IMPORTANT - how to read the output:
  * The metric is action MSE in NORMALIZED action space on held-out episodes.
    It is NOT task success: whether the foil is actually removed can only be
    measured by a rollout (cfg.task.env_runner is null in this project).
  * zero_* vs shuffle_*: zeroing state_cond is IN-distribution (the policy was
    trained with proprio_dropout_prob, so it has seen zeros), while zeroing
    wrist/pc is OUT-of-distribution and inflates the error for reasons that
    have nothing to do with importance. Prefer the shuffle_* rows when
    comparing modalities - they keep the feature statistics intact and destroy
    only the observation->action correspondence.
"""

import argparse
import pathlib
import sys

import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Locate the ManiFlow package root (the dir containing `maniflow/`).
# Resolved by search rather than a fixed number of parent hops because only
# third_party/ is bind-mounted into the ros2_dev container, so this file is
# reachable at more than one depth relative to the repo root.
def _find_maniflow_root(start):
    for cand in [start, *start.parents]:
        # Checked as a DIRECTORY, not via __init__.py: `maniflow` is an
        # implicit namespace package and has no __init__.py.
        if (cand / 'maniflow' / 'policy').is_dir():
            return cand
    raise RuntimeError(
        f"could not locate the 'maniflow' package above {start}")


_MANIFLOW_ROOT = _find_maniflow_root(pathlib.Path(__file__).resolve().parent)
if str(_MANIFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(_MANIFLOW_ROOT))

from maniflow.common.phase_detection import (  # noqa: E402
    PHASE_NAMES, PHASE_SEARCH, phase_of_windows)
from maniflow.common.pytorch_util import dict_apply  # noqa: E402

# The task config uses ${eval:'${n_obs_steps}-1'} for pad_before/pad_after.
# `eval` is a CUSTOM resolver that maniflow.workspace.train registers at
# module level; instantiating cfg.task.dataset without it raises
# UnsupportedInterpolationType. Registered identically here so the dataset is
# built with exactly the pad values of the original run.
OmegaConf.register_new_resolver("eval", eval, replace=True)


def _zero(t):
    return None if t is None else torch.zeros_like(t)


def _shuffle(t, generator):
    """Permute along the BATCH axis, keeping each sample's tokens intact.

    This is the key comparison: feature statistics (norm, distribution) stay
    exactly as the model expects, only the pairing with the ground-truth
    action is broken. If the error does not rise, the policy is not reading
    the content of this modality.
    """
    if t is None:
        return None
    idx = torch.randperm(t.shape[0], generator=generator, device=t.device)
    # Guard against the identity permutation on tiny batches, which would
    # silently report "shuffling changes nothing".
    if t.shape[0] > 1 and bool((idx == torch.arange(
            t.shape[0], device=t.device)).all()):
        idx = torch.roll(idx, 1)
    return t[idx]


VARIANTS = [
    # (name, transform(vis, state, wrist, gen) -> (vis, state, wrist), note)
    ('full',
     lambda v, s, w, g: (v, s, w),
     'reference'),
    ('shuffle_wrist',
     lambda v, s, w, g: (v, s, _shuffle(w, g)),
     'wrist RGB content broken, stats intact  <- main RGB test'),
    ('shuffle_pc',
     lambda v, s, w, g: (_shuffle(v, g), s, w),
     'point cloud content broken, stats intact'),
    ('shuffle_state',
     lambda v, s, w, g: (v, _shuffle(s, g), w),
     'proprio content broken, stats intact'),
    ('zero_state',
     lambda v, s, w, g: (v, _zero(s), w),
     'in-distribution (proprio_dropout saw this)'),
    ('zero_wrist',
     lambda v, s, w, g: (v, s, _zero(w)),
     'OOD - contrast only'),
    ('zero_pc',
     lambda v, s, w, g: (_zero(v), s, w),
     'OOD - contrast only'),
    ('vision_only',
     lambda v, s, w, g: (v, _zero(s), w),
     'same as zero_state; kept for readability'),
    ('state_only',
     lambda v, s, w, g: (_zero(v), s, _zero(w)),
     'OOD - upper bound on proprio-only performance'),
]


def episode_phase_of_windows(val_dataset, n_phases=4):
    """Semantic phase label (approach / search / hold) per sampled window.

    Replaces the earlier fixed-thirds proxy. Thirds put the boundary at 0.67
    of every episode, but the measured grasp happens at 0.35..0.77 (median
    0.59), so the 'search' bucket was contaminated with post-grasp frames -
    exactly the frames where proprio dominates - and the RGB contribution in
    the deciding phase came out diluted.

    n_phases is accepted only for interface compatibility; the detector
    always returns the three task phases.
    """
    if n_phases != len(PHASE_NAMES):
        print(f"[phase] note: detector yields {len(PHASE_NAMES)} phases, "
              f"--n-phases {n_phases} ignored")
    phases, infos = phase_of_windows(
        val_dataset.replay_buffer, val_dataset.sampler.indices)
    n_retry = sum(1 for i in infos if i['n_attempts'] > 1)
    print(f"[phase] {len(infos)} val episodes, "
          f"{sum(1 for i in infos if i['grasp_idx'] >= 0)} with a grasp, "
          f"{n_retry} with retries")
    return phases


def set_pc_points(policy, n_points):
    """Re-point the encoder to a lower FPS point count, no retraining.

    Valid without touching weights because (a) PointNetEncoderXYZ is a
    pointwise MLP, so its parameters do not depend on the point count, and
    (b) DiTX slices vis_cond_pos_embed to vis_cond.shape[1] at forward time.
    Only num_points is changed - visual_cond_len must NOT be touched, since
    that would reshape vis_cond_pos_embed and invalidate the checkpoint.

    NOTE: with a 1024-point zarr and num_points=1024 the encoder's
    `points.shape[1] > self.num_points` guard is False, so FPS never ran in
    the original run. Lowering this activates that path for the first time.
    """
    enc = policy.obs_encoder
    if not getattr(enc, 'downsample_points', False):
        print("[pc] WARNING downsample_points=False; FPS will not run")
    enc.num_points = n_points
    return policy


def load_policy(ckpt_path, device, use_ema=True):
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill,
                         map_location='cpu', weights_only=False)
    cfg = payload['cfg']

    policy = hydra.utils.instantiate(cfg.policy)
    key = 'ema_model' if (use_ema and 'ema_model' in payload['state_dicts']
                          ) else 'model'
    # strict=False because the frozen DINOv3 trunks are not all saved/loaded
    # identically across versions; report anything unexpected rather than
    # failing silently on a genuinely wrong checkpoint.
    missing, unexpected = policy.load_state_dict(
        payload['state_dicts'][key], strict=False)
    real_missing = [k for k in missing if 'key_model_map' not in k]
    print(f"[load] weights from '{key}'")
    if real_missing:
        print(f"[load] WARNING missing (non-trunk): {real_missing[:8]}")
    if unexpected:
        print(f"[load] WARNING unexpected: {list(unexpected)[:8]}")

    policy.to(device)
    policy.eval()
    return policy, cfg


def build_val_dataset(cfg):
    """The held-out split, built exactly as training builds it.

    Shared with measure_sample_std.py so both scripts evaluate the same
    windows - val_ratio and seed come from the checkpoint's own cfg, so a
    different run's split cannot leak in.
    """
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset.get_validation_dataset()
    print(f"[data] {len(val_dataset)} validation windows "
          f"(val_ratio={cfg.task.dataset.val_ratio}, "
          f"seed={cfg.task.dataset.seed})")
    return val_dataset


def build_val_loader(cfg, batch_size, num_workers):
    val_dataset = build_val_dataset(cfg)
    return torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=False)


@torch.no_grad()
def evaluate(policy, loader, args, phases=None, n_phases=4,
             variants=VARIANTS, collect_per_dim=False):
    """Run every variant over n_batches; return (results, per_phase, per_dim).

    All variants share one encoding pass and one noise seed per batch, so the
    only difference between rows is the manipulated tensor.
    """
    device = next(policy.parameters()).device
    results = {name: [] for name, _, _ in variants}
    per_phase = {name: [[] for _ in range(n_phases)] for name, _, _ in variants}
    per_dim = {name: [] for name, _, _ in variants}
    n_done = 0
    cursor = 0

    for batch in loader:
        if n_done >= args.n_batches:
            break
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

        nobs = policy.normalizer.normalize(batch['obs'])
        nactions = policy.normalizer['action'].normalize(batch['action'])
        if not policy.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        B = nactions.shape[0]
        # Loader is shuffle=False, so windows arrive in sampler order and the
        # phase labels line up with this slice.
        batch_phase = (phases[cursor:cursor + B] if phases is not None
                       else None)
        cursor += B

        vis, state, pc_pe = policy._encode_obs(nobs, B)
        wrist = (policy._encode_wrist(nobs, B)
                 if policy.wrist_conditioned else None)

        for name, transform, _ in variants:
            gen = torch.Generator(device=device).manual_seed(args.seed + n_done)
            v, s, w = transform(vis, state, wrist, gen)

            torch.manual_seed(args.seed + n_done)
            cond_data = torch.zeros(
                (B, policy.horizon, policy.action_dim),
                device=device, dtype=nactions.dtype)
            pred = policy.conditional_sample(
                cond_data, vis_cond=v, state_cond=s, wrist_cond=w,
                lang_cond=None, pc_pe=pc_pe)[..., :policy.action_dim]

            sq = (pred - nactions) ** 2
            results[name].append(sq.mean().item())

            if batch_phase is not None:
                # (B,) per-window error, so each window lands in its own phase
                # bucket instead of being averaged across the batch.
                per_win = sq.mean(dim=(1, 2)).cpu().numpy()
                for ph in range(n_phases):
                    m = batch_phase == ph
                    if m.any():
                        per_phase[name][ph].append(float(per_win[m].mean()))
            if collect_per_dim:
                per_dim[name].append(sq.mean(dim=(0, 1)).cpu().numpy())
                if batch_phase is not None:
                    # xy error restricted to the decision window: the single
                    # number that reflects visual grasp-point selection.
                    sel = batch_phase == PHASE_SEARCH
                    if sel.any():
                        xy = sq[torch.as_tensor(sel, device=sq.device)][
                            ..., [0, 1]].mean().item()
                        per_dim.setdefault(f'__xy_search__{name}',
                                           []).append(xy)

        n_done += 1
        print(f"  batch {n_done}/{args.n_batches}", end='\r', flush=True)

    print(f"\n[done] {n_done} batches x {args.batch_size} windows")
    return results, per_phase, per_dim


def print_table(results, variants=VARIANTS, title=None):
    if title:
        print(f"\n=== {title} ===")
    base = float(np.mean(results['full']))
    width = max(len(n) for n, _, _ in variants)
    print(f"{'variant':<{width}}  {'MSE':>9}  {'xfull':>7}   note")
    print('-' * (width + 34))
    for name, _, note in variants:
        m = float(np.mean(results[name]))
        ratio = m / base if base > 0 else float('nan')
        print(f"{name:<{width}}  {m:9.5f}  {ratio:7.2f}   {note}")
    return base


def print_xy_search(per_dim, variants=VARIANTS):
    """Lateral (xy) error inside the decision window only."""
    rows = [(n, per_dim.get(f'__xy_search__{n}', []))
            for n, _, _ in variants]
    if not rows or not rows[0][1]:
        return
    base = float(np.mean(rows[0][1]))
    print("\n=== xy error in the DECISION window (search phase only) ===")
    print("The one number that isolates visual grasp-point selection:")
    print("lateral error while the hand is still open, a chunk before")
    print("closing. z and gripper are excluded, so the sign reversal at")
    print("the grasp cannot leak in.")
    width = max(len(n) for n, _ in rows)
    print(f"{'variant':<{width}}  {'xy MSE':>9}  {'xfull':>7}")
    print('-' * (width + 20))
    for name, vals in rows:
        if not vals:
            continue
        m = float(np.mean(vals))
        print(f"{name:<{width}}  {m:9.5f}  {m / base:7.2f}")


def print_phase_table(per_phase, n_phases, variants=VARIANTS):
    print(f"\n=== per task phase (x full, within the same phase) ===")
    print("approach   = descending to the pallet (point cloud: pallet pose)")
    print("search     = DECISION window [t_close-32, t_close-2): hand open,")
    print("             one action chunk before closing - where the grasp")
    print("             point is actually chosen")
    print("transition = +-few frames around closing; z reverses sign here,")
    print("             so its MSE is a timing artefact, not spatial error")
    print("hold       = pinch closed (proprio; vision only to catch a slip)")
    labels = list(PHASE_NAMES[:n_phases])
    width = max(len(n) for n, _, _ in variants)
    base = [float(np.mean(per_phase['full'][p])) if per_phase['full'][p]
            else float('nan') for p in range(n_phases)]
    print(f"{'variant':<{width}}  " + '  '.join(f"{l:>7}" for l in labels))
    print('-' * (width + 9 * n_phases))
    print(f"{'full (abs MSE)':<{width}}  "
          + '  '.join(f"{b:7.5f}" for b in base))
    for name, _, _ in variants:
        if name == 'full':
            continue
        cells = []
        for p in range(n_phases):
            vals = per_phase[name][p]
            cells.append(f"{np.mean(vals) / base[p]:7.2f}"
                         if vals and base[p] > 0 else f"{'-':>7}")
        print(f"{name:<{width}}  " + '  '.join(cells))


# Action layout: 0,1 = ee x,y | 2 = ee z | 3..6 = quat | 7..14 = hand.
_AXIS_GROUPS = (('xy', (0, 1)), ('z', (2,)), ('rot', (3, 4, 5, 6)),
                ('hand', tuple(range(7, 15))))


def print_per_dim(per_dim, variants=VARIANTS):
    """Error split by axis group.

    Finer than ee_pose-vs-hand, because those mix the three questions that
    matter here: xy is the lateral choice of WHERE to grasp (the visual
    decision), z is descent height (geometry, and the axis that reverses sign
    at the grasp), and hand is the grasp itself. Only xy inside the search
    window says anything about visual grasp-point selection.

    Values stay in normalized action space, so groups are comparable to each
    other only as ratios between variants, not as absolute magnitudes.
    """
    print("\n=== error by axis group (xy=lateral, z=height, rot=quat) ===")
    width = max(len(n) for n, _, _ in variants)
    hdr = f"{'variant':<{width}}  " + '  '.join(
        f"{n:>9}" for n, _ in _AXIS_GROUPS)
    print(hdr)
    print('-' * len(hdr))
    for name, _, _ in variants:
        d = np.mean(np.stack(per_dim[name]), axis=0)
        cells = [f"{d[list(ix)].mean():9.5f}" for _, ix in _AXIS_GROUPS]
        print(f"{name:<{width}}  " + '  '.join(cells))


def run(args):
    device = torch.device(args.device)
    policy, cfg = load_policy(args.ckpt, device, use_ema=not args.no_ema)
    if args.pc_points is not None:
        set_pc_points(policy, args.pc_points)
        print(f"[pc] num_points -> {args.pc_points} (weights untouched)")

    loader = build_val_loader(cfg, args.batch_size, args.num_workers)
    phases = episode_phase_of_windows(loader.dataset, args.n_phases)

    results, per_phase, per_dim = evaluate(
        policy, loader, args, phases=phases, n_phases=args.n_phases,
        collect_per_dim=True)

    print_table(results, title=f"aggregate  ({pathlib.Path(args.ckpt).name})")
    print_phase_table(per_phase, args.n_phases)
    print_per_dim(per_dim)
    print_xy_search(per_dim)

    print("\nReading the result:")
    print("  shuffle_wrist ~ 1.0x  -> policy ignores wrist RGB content")
    print("  shuffle_state >> 1.0x -> policy rides on proprio (the shortcut)")
    print("  a modality whose ratio PEAKS in one phase is used specifically")
    print("  there - the aggregate row understates it.")
    print("\nNOTE: normalized action MSE on held-out episodes, NOT task "
          "success. A rollout is still required to tell whether the foil "
          "is removed.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--n-batches', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-ema', action='store_true',
                   help='use raw model weights instead of EMA')
    p.add_argument('--n-phases', type=int, default=4,
                   help='episode segments for the per-phase breakdown')
    p.add_argument('--pc-points', type=int, default=None,
                   help='re-run with this many FPS point-cloud points '
                        '(no retraining; tests whether 1024 is redundant)')
    run(p.parse_args())


if __name__ == '__main__':
    main()