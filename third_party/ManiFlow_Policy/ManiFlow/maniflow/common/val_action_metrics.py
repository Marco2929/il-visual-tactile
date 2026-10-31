"""Validation action metrics: per-phase error and multimodality-aware MSE.

WHY
---
1. `val_loss` is flow-velocity MSE at a random t, averaged over the whole
   flow path. The policy is judged on the ENDPOINT of the ODE rollout, and the
   two diverge: on run bmq17ag4 the val_loss optimum was epoch 10 while the
   action-MSE optimum was epoch 50, i.e. `checkpoint.topk.monitor_key:
   val_loss` selected a checkpoint ~53% worse than the best available.
   `val_action_mse` is that missing metric. Note the repo only logs
   `train_action_mse_error`, which is measured on TRAINING data and therefore
   reports memorisation.

2. The task is ambiguous: several foil edges are valid grasp points, so a
   policy that picks a different-but-valid one is penalised by MSE against the
   single recorded demo. `min-over-k` samples the policy k times and keeps the
   best match, which asks "was the demo inside the predicted distribution?"
   instead of "did the mean coincide with the demo?". Flow matching can
   represent that multimodality; mean MSE just hides it.
     * mse_k1 >> mse_mink  -> healthy multimodality
     * mse_k1 ~= mse_mink  -> genuine error, not ambiguity

3. Phase breakdown, because the aggregate average hides where the error sits.
   The phases are detected from proprioception (see phase_detection.py), not
   from fixed fractions of the episode.

All values are in NORMALIZED action space, so they are comparable across
epochs but are not metres/radians. None of this is task success - that needs
a rollout.
"""

import numpy as np
import torch
import torch.nn.functional as F

from maniflow.common.phase_detection import PHASE_NAMES, phase_of_windows


class ValActionEvaluator:
    """Computes val action MSE (optionally min-over-k) split by task phase.

    Phase labels are derived once from the validation replay buffer and cached;
    they depend only on the dataset, not on the model.
    """

    def __init__(self, val_dataset, n_samples=1, max_batches=None,
                 verbose=True):
        self.n_samples = n_samples
        self.max_batches = max_batches
        self.phases = None

        try:
            self.phases, infos = phase_of_windows(
                val_dataset.replay_buffer, val_dataset.sampler.indices)
            if verbose:
                n_grasp = sum(1 for i in infos if i['grasp_idx'] >= 0)
                n_retry = sum(1 for i in infos if i['n_attempts'] > 1)
                counts = np.bincount(self.phases, minlength=len(PHASE_NAMES))
                print(f"[val-metrics] {len(infos)} val episodes, "
                      f"{n_grasp} with a grasp, {n_retry} with retries; "
                      f"windows per phase: "
                      + ', '.join(f"{PHASE_NAMES[k]} {counts[k]}"
                                  for k in range(len(PHASE_NAMES))))
        except Exception as exc:
            # Phase detection depends on the proprio layout (hand joint dims),
            # so a different task would fail here. Degrade to the aggregate
            # metric instead of taking down the training loop.
            print(f"[val-metrics] phase detection unavailable "
                  f"({type(exc).__name__}: {exc}); aggregate only")

    @torch.no_grad()
    def evaluate(self, policy, val_dataloader, device):
        """Returns a flat dict of scalars ready for step_log."""
        was_training = policy.training
        policy.eval()

        sq_first, sq_best, phase_ids = [], [], []
        cursor = 0

        for batch_idx, batch in enumerate(val_dataloader):
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break
            batch = {k: (v.to(device, non_blocking=True)
                         if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            obs_dict = batch['obs']
            # NORMALIZED space. predict_action() un-normalizes its output, so
            # comparing it straight against batch['action'] averages metres,
            # radians and joint degrees together - that is what made this
            # metric report ~66 instead of ~0.01 and left it incomparable
            # with scripts/sweep_checkpoints.py. Ranking was unaffected (the
            # factor is constant), absolute values were meaningless.
            norm = policy.normalizer['action']
            gt = norm.normalize(batch['action'])
            B = gt.shape[0]

            # (k, B, T, Da): k independent samples of the same conditioning.
            preds = []
            for _ in range(max(self.n_samples, 1)):
                out = policy.predict_action(obs_dict)['action_pred']
                preds.append(norm.normalize(
                    out[:, :gt.shape[1], :gt.shape[2]]))
            preds = torch.stack(preds, dim=0)

            # Per-sample error, reduced over horizon and action dims so the
            # min below picks the best WHOLE trajectory rather than mixing
            # the best dimensions from different samples.
            err = ((preds - gt.unsqueeze(0)) ** 2).mean(dim=(2, 3))  # (k, B)
            sq_first.append(err[0].cpu().numpy())
            sq_best.append(err.min(dim=0).values.cpu().numpy())

            if self.phases is not None:
                phase_ids.append(self.phases[cursor:cursor + B])
            cursor += B

        if was_training:
            policy.train()

        if not sq_first:
            return {}

        first = np.concatenate(sq_first)
        best = np.concatenate(sq_best)
        log = {'val_action_mse': float(first.mean())}
        if self.n_samples > 1:
            log[f'val_action_mse_min{self.n_samples}'] = float(best.mean())
            # >1 means several samples are valid-but-different, i.e. the
            # policy is covering multiple modes rather than being wrong.
            log['val_action_mse_multimodality'] = float(
                first.mean() / max(best.mean(), 1e-12))

        if phase_ids:
            ph = np.concatenate(phase_ids)[:len(first)]
            for k, name in enumerate(PHASE_NAMES):
                m = ph == k
                if m.any():
                    log[f'val_action_mse/{name}'] = float(first[m].mean())
                    if self.n_samples > 1:
                        log[f'val_action_mse_min/{name}'] = float(
                            best[m].mean())
        return log
