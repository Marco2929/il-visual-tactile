"""Multimodality vs mode collapse: spread of K samples per observation.

WHY
---
The decision-window ablation showed shuffle_wrist 1.12x, shuffle_pc 1.01x and
shuffle_state 0.96x on lateral (xy) error, i.e. destroying ANY single
modality's content barely moves the prediction, while zeroing any of them
costs 4x. That is the signature of a policy that needs the conditioning to be
PRESENT but not its CONTENT - so the single-sample MSE of 0.0073 has two
mutually exclusive explanations:

  1. Healthy multimodality. The model commits sharply to valid grasp point A
     or B depending on the initial noise x0, and the recorded demo happened to
     pick B. Only the metric is wrong, the model is fine.
  2. Mode collapse. The model ignores both the fine image detail and the noise
     prior and emits the same averaged trajectory for every seed.

minMSE@K alone cannot separate these (both can lower it), so this measures the
SPREAD of the predictions as well:

    sample_std  = std over the K samples, averaged over horizon/dims
    mse_1       = error of a single sample      (what training logs)
    mse_minK    = best of K samples
    gain        = mse_1 / mse_minK

  gain >> 1 AND sample_std high  -> case 1, multimodal; fix the metric
  gain ~ 1  AND sample_std ~ 0   -> case 2, collapsed; the model is the problem

Reported per task phase, because the decision window is the only place where
multiple valid answers are expected - 'hold' SHOULD be nearly deterministic,
and a low spread there is correct rather than a warning sign.

    python scripts/measure_sample_std.py --ckpt <ckpt> -k 16
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ablate_modalities import build_val_dataset, load_policy  # noqa: E402
from maniflow.common.phase_detection import (  # noqa: E402
    PHASE_NAMES, phase_of_windows)
from maniflow.common.pytorch_util import dict_apply  # noqa: E402

# xy only for the headline number: it isolates WHERE to grasp from descent
# height and gripper timing.
XY = [0, 1]

# Action layout: 0,1 = ee x,y | 2 = ee z | 3..6 = quat | 7..14 = hand.
AXIS_GROUPS = (('xy', [0, 1]), ('z', [2]), ('rot', [3, 4, 5, 6]),
               ('hand', list(range(7, 15))))


def stratified_batches(dataset, phases, batch_size, n_per_phase, seed=0):
    """Yield (indices, batch) with every phase equally represented.

    A sequential DataLoader walks the buffer in order, so the first few
    batches come from the first episodes and land almost entirely in
    'approach' - the earlier run saw 64 windows, all of them approach, and
    reported nothing for the decision window. Plain shuffle=True would fix
    coverage but break the cursor->phase mapping, since the loader would no
    longer hand back windows in sampler order.

    Drawing explicit indices solves both: the phase of every window is known
    by construction, and each phase gets the same number of samples
    regardless of how rare it is (the decision window is only ~7% of frames).
    """
    rng = np.random.default_rng(seed)
    picks = []
    for p in range(len(PHASE_NAMES)):
        pool = np.flatnonzero(phases == p)
        if len(pool) == 0:
            print(f"[warn] no windows in phase {PHASE_NAMES[p]}")
            continue
        take = min(n_per_phase, len(pool))
        picks.append(rng.choice(pool, size=take, replace=False))
        print(f"[sample] {PHASE_NAMES[p]:<11} {take:4d} of {len(pool)} "
              f"available windows")
    picks = np.concatenate(picks)

    for s in range(0, len(picks), batch_size):
        idx = picks[s:s + batch_size]
        items = [dataset[int(i)] for i in idx]
        batch = {
            'action': torch.stack([it['action'] for it in items]),
            'obs': {k: torch.stack([it['obs'][k] for it in items])
                    for k in items[0]['obs']},
        }
        yield idx, batch


@torch.no_grad()
def run(args):
    device = torch.device(args.device)
    policy, cfg = load_policy(args.ckpt, device, use_ema=not args.no_ema)
    dataset = build_val_dataset(cfg)
    phases, _ = phase_of_windows(dataset.replay_buffer,
                                 dataset.sampler.indices)

    # Everything below is in NORMALIZED action space. predict_action()
    # un-normalizes its output, so comparing it against batch['action']
    # directly mixes metres, radians and joint degrees in one average - that
    # is what produced mse_1=5.44 instead of ~0.009 and made the numbers
    # incomparable with sweep_checkpoints.py.
    norm = policy.normalizer['action']

    rows = {'mse_1': [], 'mse_min': [], 'std': [],
            'mse_1_xy': [], 'mse_min_xy': [], 'std_xy': [], 'phase': []}

    gt_scale = []
    n_batches = 0
    for idx, batch in stratified_batches(dataset, phases, args.batch_size,
                                         args.n_per_phase, args.seed):
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        gt = norm.normalize(batch['action'])
        B, T, Da = gt.shape

        preds = []
        for k in range(args.k):
            # predict_action takes no generator, so the seed is set globally.
            # Distinct per (batch, k) so every sample draws a different x0 -
            # with one shared seed the K samples would be identical and the
            # spread would be zero by construction.
            torch.manual_seed(args.seed + 1000 * n_batches + k)
            out = policy.predict_action(batch['obs'])['action_pred']
            preds.append(norm.normalize(out[:, :T, :Da]))
        preds = torch.stack(preds)                      # (K, B, T, Da)

        err = ((preds - gt.unsqueeze(0)) ** 2).mean(dim=(2, 3))   # (K, B)
        err_xy = ((preds[..., XY] - gt[..., XY].unsqueeze(0)) ** 2
                  ).mean(dim=(2, 3))
        # Spread across samples of the SAME observation: this is the quantity
        # that separates the two explanations.
        std = preds.std(dim=0).mean(dim=(1, 2))                   # (B,)
        std_xy = preds[..., XY].std(dim=0).mean(dim=(1, 2))

        for gname, gidx in AXIS_GROUPS:
            ge = ((preds[..., gidx] - gt[..., gidx].unsqueeze(0)) ** 2
                  )[0].mean(dim=(1, 2))
            # Variance of the TARGET within the window, i.e. how much this
            # group actually has to move - the denominator for a fair
            # comparison between groups.
            gv = gt[..., gidx].var(dim=1).mean(dim=1)
            gs = preds[..., gidx].std(dim=0).mean(dim=(1, 2))
            rows.setdefault(f'g_mse_{gname}', []).append(ge.cpu().numpy())
            rows.setdefault(f'g_var_{gname}', []).append(gv.cpu().numpy())
            rows.setdefault(f'g_std_{gname}', []).append(gs.cpu().numpy())
        # Reference scale: how much the TARGET actions vary within a window.
        # A sample spread only means something relative to this.
        gt_scale.append(gt[..., XY].std(dim=1).mean(dim=1).cpu().numpy())

        rows['mse_1'].append(err[0].cpu().numpy())
        rows['mse_min'].append(err.min(dim=0).values.cpu().numpy())
        rows['std'].append(std.cpu().numpy())
        rows['mse_1_xy'].append(err_xy[0].cpu().numpy())
        rows['mse_min_xy'].append(err_xy.min(dim=0).values.cpu().numpy())
        rows['std_xy'].append(std_xy.cpu().numpy())
        rows['phase'].append(phases[idx])
        n_batches += 1
        print(f"  batch {n_batches}", end='\r', flush=True)

    # Group entries stay as lists of arrays; they are reduced in the table
    # below and must not be flattened into the per-window arrays.
    r = {k: (v if k.startswith('g_') else np.concatenate(v))
         for k, v in rows.items()}
    n = len(r['mse_1'])
    print(f"\n[done] {n} windows x K={args.k} samples "
          f"(normalized action space)\n")

    gt_xy_std = float(np.concatenate(gt_scale).mean())
    print(f"[scale] within-window std of the TARGET xy actions: "
          f"{gt_xy_std:.5f}")
    print("        sample_std is only meaningful against this.")

    # Is the larger rot/hand error real, or just a bigger target range?
    # Normalization maps each dim onto its training range, so a group whose
    # targets move further within a window also carries a larger MSE at equal
    # relative accuracy. Dividing by the target spread removes that, and only
    # a group that stays high AFTER dividing is worth re-weighting in the
    # loss.
    print(f"\n=== error relative to target spread, per axis group ===")
    print(f"{'group':<6} {'mse_1':>9} {'gt_var':>9} {'mse/gt_var':>11} "
          f"{'std':>9}")
    print('-' * 50)
    for gname, gidx in AXIS_GROUPS:
        e = float(np.concatenate(r[f'g_mse_{gname}']).mean())
        v = float(np.concatenate(r[f'g_var_{gname}']).mean())
        s = float(np.concatenate(r[f'g_std_{gname}']).mean())
        print(f"{gname:<6} {e:9.5f} {v:9.5f} {e / max(v, 1e-12):11.2f} "
              f"{s:9.5f}")
    print("  mse/gt_var equal across groups -> the raw differences are a")
    print("  normalization effect, not a weighting problem.")

    def block(title, m1, mmin, sd):
        gain = m1.mean() / max(mmin.mean(), 1e-12)
        print(f"\n=== {title} ===")
        print(f"  mse_1       {m1.mean():.5f}")
        print(f"  mse_min{args.k:<3}   {mmin.mean():.5f}")
        print(f"  gain        {gain:.2f}x")
        print(f"  sample_std  {sd.mean():.5f}")
        return gain

    block('all dims, all phases', r['mse_1'], r['mse_min'], r['std'])
    block('xy only, all phases', r['mse_1_xy'], r['mse_min_xy'],
          r['std_xy'])

    print(f"\n=== xy per phase ===")
    print(f"{'phase':<11} {'n':>5} {'mse_1':>9} {f'min{args.k}':>9} "
          f"{'gain':>6} {'std':>9} {'std/gt':>7}")
    print('-' * 62)
    for p, name in enumerate(PHASE_NAMES):
        m = r['phase'] == p
        if not m.any():
            continue
        g = r['mse_1_xy'][m].mean() / max(r['mse_min_xy'][m].mean(), 1e-12)
        sd = r['std_xy'][m].mean()
        print(f"{name:<11} {int(m.sum()):5d} {r['mse_1_xy'][m].mean():9.5f} "
              f"{r['mse_min_xy'][m].mean():9.5f} {g:6.2f} {sd:9.5f} "
              f"{sd / max(gt_xy_std, 1e-12):7.2f}")

    print("\nreading it:")
    print("  search: gain >> 1 and std high -> multimodal, metric is at "
          "fault")
    print("          gain ~ 1 and std ~ 0   -> collapsed onto the mean")
    print("  hold:   low std is EXPECTED there, not a problem")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('-k', type=int, default=16)
    p.add_argument('--n-per-phase', type=int, default=48,
                   help='windows sampled per phase (equal across phases)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-ema', action='store_true')
    run(p.parse_args())
