# Create a test script: test_model_sanity.py
import torch
import numpy as np
import hydra

from model_serve import init_model

cfg, policy = init_model(
    task_name="test_block_grasping",
    alg_name="maniflow_pointcloud_policy",
    addition_info=1211,
    run_dir="/ros2_ws/data/outputs/test_block_grasping-maniflow_pointcloud_policy-0112_seed0"
)

# Load actual training data
import zarr
z = zarr.open('/ros2_ws/data/zarr_datasets/test_collection.zarr', 'r')

# Take first 2 frames as obs (n_obs_steps=2)
pc = z['data/point_cloud'][:2]          # (2, 1024, 3)
state = z['data/state'][:2]             # (2, 13)
gt_action = z['data/action'][1:33]      # (32, 13) ground truth

# Format as model expects
obs = {
    'point_cloud': torch.from_numpy(pc).unsqueeze(0).float().cuda(),
    'agent_pos': torch.from_numpy(state).unsqueeze(0).float().cuda()
}

with torch.no_grad():
    result = policy.predict_action(obs)
    pred_action = result['action'][0].cpu().numpy()

print(f"Ground truth action[0]: {np.round(gt_action[0], 3)}")
print(f"Predicted   action[0]: {np.round(pred_action[0], 3)}")
print(f"\nGT hand range: [{gt_action[:,7:13].min():.3f}, {gt_action[:,7:13].max():.3f}]")
print(f"Pred hand range: [{pred_action[:,7:13].min():.3f}, {pred_action[:,7:13].max():.3f}]")
print(f"\nGT pose x range: [{gt_action[:,0].min():.3f}, {gt_action[:,0].max():.3f}]")
print(f"Pred pose x range: [{pred_action[:,0].min():.3f}, {pred_action[:,0].max():.3f}]")

policy.num_inference_steps = 100
mses = []
for seed in range(20):
    torch.manual_seed(seed)
    with torch.no_grad():
        result = policy.predict_action(obs)
        pred = result['action'][0].cpu().numpy()
    mses.append(np.mean((pred - gt_action[:32])**2))
print(f"MSE: mean={np.mean(mses):.4f}, std={np.std(mses):.4f}, min={np.min(mses):.4f}, max={np.max(mses):.4f}")
