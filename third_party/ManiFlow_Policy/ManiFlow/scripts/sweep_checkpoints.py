"""Action MSE on held-out episodes for every saved checkpoint.

WHY THIS EXISTS
---------------
The run ranks checkpoints by `val_loss`, which is the FLOW loss: velocity MSE
at a random t, averaged over the whole flow path. What the policy is actually
judged on is the endpoint of the full 10-step ODE rollout. The two can move in
opposite directions, and in this project they do:

    epoch 10   val_loss 0.0597   action MSE 0.01493
    epoch 249  val_loss 0.2356   action MSE 0.01208   <- better, not worse

So `checkpoint.topk.monitor_key: val_loss` may well be selecting the wrong
checkpoint. This script computes the metric that is missing (there is no
val_action_mse in the logs, only train_action_mse_error) for every epoch that
was saved, so the choice can be made on evidence.

    python scripts/sweep_checkpoints.py \
        --ckpt-dir /ros2_ws/data/outputs/<run>/checkpoints \
        --n-batches 5

Still NOT task success - see the note in ablate_modalities.py.
"""

import argparse
import pathlib
import re
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ablate_modalities import (  # noqa: E402
    VARIANTS, build_val_loader, evaluate, load_policy, set_pc_points)

_EPOCH_RE = re.compile(r'epoch=(\d+)')

# Only the rows needed for a trend; the full 9-variant table per checkpoint
# would be ~25x slower for no extra insight.
_SWEEP_VARIANTS = [v for v in VARIANTS
                   if v[0] in ('full', 'shuffle_wrist', 'shuffle_pc',
                               'shuffle_state')]


def discover(ckpt_dir):
    """Sorted (epoch, path). 'latest.ckpt' carries no epoch in its name, so it
    is placed last via a sentinel rather than being skipped."""
    out = []
    for p in sorted(pathlib.Path(ckpt_dir).glob('*.ckpt')):
        m = _EPOCH_RE.search(p.name)
        out.append((int(m.group(1)) if m else 10 ** 9, p))
    return sorted(out)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt-dir', required=True)
    p.add_argument('--n-batches', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-ema', action='store_true')
    p.add_argument('--pc-points', type=int, default=None)
    args = p.parse_args()

    ckpts = discover(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(f"no .ckpt found in {args.ckpt_dir}")
    print(f"[sweep] {len(ckpts)} checkpoints, {args.n_batches} batches each")

    device = torch.device(args.device)
    loader = None
    rows = []

    for epoch, path in ckpts:
        policy, cfg = load_policy(path, device, use_ema=not args.no_ema)
        if args.pc_points is not None:
            set_pc_points(policy, args.pc_points)
        # The dataset is identical for every checkpoint (same cfg, same seed,
        # same val_ratio), so it is built once - rebuilding it per checkpoint
        # would dominate the runtime and risk a different split.
        if loader is None:
            loader = build_val_loader(cfg, args.batch_size, args.num_workers)

        res, _, _ = evaluate(policy, loader, args, phases=None,
                             variants=_SWEEP_VARIANTS)
        row = {n: float(np.mean(res[n])) for n, _, _ in _SWEEP_VARIANTS}
        row['epoch'] = epoch
        row['name'] = path.name
        rows.append(row)

        del policy
        torch.cuda.empty_cache()
        print(f"[{path.name}] action_mse={row['full']:.5f}")

    names = [n for n, _, _ in _SWEEP_VARIANTS]
    print("\n=== action MSE per checkpoint (held-out episodes) ===")
    hdr = f"{'epoch':>6}  " + '  '.join(f"{n:>14}" for n in names)
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        ep = 'latest' if r['epoch'] == 10 ** 9 else str(r['epoch'])
        print(f"{ep:>6}  " + '  '.join(f"{r[n]:14.5f}" for n in names))

    best = min(rows, key=lambda r: r['full'])
    ep = 'latest' if best['epoch'] == 10 ** 9 else best['epoch']
    print(f"\nbest action MSE: epoch {ep} ({best['name']}) = "
          f"{best['full']:.5f}")
    print("Compare against the val_loss ranking in the checkpoint filenames:")
    print("if they disagree, checkpoint.topk.monitor_key is picking the")
    print("wrong checkpoint for this task.")


if __name__ == '__main__':
    main()