import dill
import hydra
from omegaconf import OmegaConf
import pathlib
import torch
import os
import numpy as np
from hydra.core.global_hydra import GlobalHydra

from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy

class DeployManiWorkspace:
    """Model loading and management"""
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.model: ManiFlowTransformerPointcloudPolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: ManiFlowTransformerPointcloudPolicy = None
        if cfg.training.use_ema:
            self.ema_model = hydra.utils.instantiate(cfg.policy)
            
    def load_checkpoint(self, path):
        payload = torch.load(path, pickle_module=dill, map_location='cpu')
        if 'state_dicts' in payload:
            if 'model' in payload['state_dicts']:
                self.model.load_state_dict(payload['state_dicts']['model'])
            if self.ema_model is not None and 'ema_model' in payload['state_dicts']:
                self.ema_model.load_state_dict(payload['state_dicts']['ema_model'])
            print("====== CHECKPOINT DEBUG ======")
            print("Keys in state_dicts:", payload['state_dicts'].keys())
            if 'normalizer' in payload['state_dicts']:
                print("Normalizer keys:", payload['state_dicts']['normalizer'].keys())
            print("==============================")
        return True


if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
# -----------------------------------------------------------

def init_model(task_name, alg_name, addition_info, run_dir):
    config_dir_path = pathlib.Path(__file__).parent / 'maniflow' / 'config'
    config_dir_str = str(config_dir_path.absolute())

    exp_name = f"{task_name}-{alg_name}-{addition_info}"
    print(config_dir_str)
    
    # Clean Hydra
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

    workspace = DeployManiWorkspace(cfg)
    ckpt_path = os.path.join(cfg.run_dir, 'checkpoints/latest.ckpt')
    workspace.load_checkpoint(ckpt_path)

    policy = workspace.ema_model or workspace.model
    policy.eval().to(cfg.training.device) 
        
    # --- OPTIMIERUNG 2: PyTorch 2.0 Kompilierung (CUDA Graphen) ---
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        print("[INFO] Kompiliere PyTorch Modell für extrem schnelle Inferenz...")
        print("[INFO] Hinweis: Der erste Durchlauf (Chunk) wird dadurch 1-2 Minuten dauern!")
        
        # 'reduce-overhead' ist ideal, um den Python-Dispatch-Overhead bei vielen
        # kleinen Iterationen (wie den 10 Denoising-Schritten) zu minimieren.
        try:
            policy = torch.compile(policy, mode="reduce-overhead")
        except Exception as e:
            print(f"[WARN] torch.compile fehlgeschlagen. Falle auf Standard-Inferenz zurück. Fehler: {e}")
    # -------------------------------------------------------------

    print("=== Flash Attention Diagnose ===")
    print(f"PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Native PyTorch Flash Attention (SDPA) bereit: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"Native PyTorch Math Attention (Langsam) bereit: {torch.backends.cuda.math_sdp_enabled()}")
        
        try:
            import flash_attn
            print(f"Externes 'flash-attn' Paket installiert: JA (Version {flash_attn.__version__})")
        except ImportError:
            print("Externes 'flash-attn' Paket installiert: NEIN")
    else:
        print("CUDA ist nicht verfügbar!")

    return cfg, policy

def normalize(result):
    raw_tensor = result['action'][0]
    all_actions = raw_tensor.cpu().numpy()

    quats = all_actions[:, 3:7]
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    all_actions[:, 3:7] = quats / norms

    return all_actions