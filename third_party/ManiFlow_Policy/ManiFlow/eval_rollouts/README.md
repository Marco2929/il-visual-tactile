# Offline Progress Head Rollout Evaluation

Evaluate a trained `ProgressHeadST` checkpoint on real-world test rollouts recorded by `log_rollout.py` without requiring physical robot hardware.

## How it works
1. Loads real wrist camera frames from `wrist_bag/wrist_bag_0.mcap`.
2. Extracts DINOv3 patch tokens via `DINOv3ObsEncoder`.
3. Runs the spatio-temporal `ProgressHeadST` on the 4-frame window `(t, t-s, t-2s, t-3s)`.
4. Correlates the predicted progress curve with the recorded outcome (success vs failure modes like `grasp_miss`, `slip`, etc.) and saves evaluation plots.

## Usage

### Evaluate a single rollout:
```bash
python eval_rollouts/eval_progress_on_rollout.py \
    --rollout /ros2_ws/data/rollout_logs/foil_lifting_wrist_e40/rollout_000 \
    --checkpoint /ros2_ws/checkpoints/progress_head/best_progress_head.pt \
    --output_dir eval_rollouts/plots
```

### Evaluate an entire rollout suite:
```bash
python eval_rollouts/eval_progress_on_rollout.py \
    --rollout_dir /ros2_ws/data/rollout_logs/foil_lifting_wrist_e40 \
    --checkpoint /ros2_ws/checkpoints/progress_head/best_progress_head.pt \
    --output_dir eval_rollouts/plots
```
Generates individual progress curve plots for each rollout and an `evaluation_summary.json` with success detection and false positive rates.

