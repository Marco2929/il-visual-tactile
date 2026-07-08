import dill
import hydra
from omegaconf import OmegaConf
import pathlib
import torch
import os
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from termcolor import cprint

import sys
sys.path.insert(0, "/ros2_ws/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy")

from diffusion_policy_3d.policy.simple_dp3 import SimpleDP3

class DeployDP3Workspace:
    """Model loading and management"""
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.model: SimpleDP3 = hydra.utils.instantiate(cfg.policy)
        self.ema_model: SimpleDP3 = None
        if cfg.training.use_ema:
            self.ema_model = hydra.utils.instantiate(cfg.policy)

    def load_checkpoint(self, path):
        payload = torch.load(path, pickle_module=dill, map_location='cpu')
        if 'state_dicts' in payload:
            if 'model' in payload['state_dicts']:
                self.model.load_state_dict(payload['state_dicts']['model'])
            if self.ema_model is not None and 'ema_model' in payload['state_dicts']:
                self.ema_model.load_state_dict(payload['state_dicts']['ema_model'])
        return True


if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')


def init_model(task_name, alg_name, addition_info, run_dir):
    config_dir_path = pathlib.Path('/ros2_ws/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config')
    config_dir_str = str(config_dir_path.absolute())

    exp_name = f"{task_name}-{alg_name}-{addition_info}"

    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir_str):
        cfg = hydra.compose(
            config_name=alg_name,
            overrides=[
                f"task={task_name}",
                f"+run_dir={run_dir}",
                "training.debug=False",
                "training.seed=42",
                "training.device=cuda:0",
                f"exp_name={exp_name}",
                "logging.mode=offline",
                "checkpoint.save_ckpt=False"
            ]
        )

    ckpt_path = os.path.join(run_dir, 'checkpoints/latest.ckpt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    workspace = DeployDP3Workspace(cfg)
    workspace.load_checkpoint(ckpt_path)

    policy = workspace.ema_model or workspace.model
    policy.eval().to(cfg.training.device)

    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        cprint("[INFO] Compiling model with torch.compile (mode=reduce-overhead)...", "cyan")
        cprint("[INFO] Note: First inference chunk will take 1-2 minutes!", "yellow")
        try:
            policy = torch.compile(policy, mode="reduce-overhead")
        except Exception as e:
            cprint(f"[WARN] torch.compile failed, falling back to standard inference. Error: {e}", "yellow")

    return cfg, policy


def normalize(result):
    raw_tensor = result['action'][0]
    all_actions = raw_tensor.cpu().numpy()

    quats = all_actions[:, 3:7]
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    all_actions[:, 3:7] = quats / norms

    return all_actions