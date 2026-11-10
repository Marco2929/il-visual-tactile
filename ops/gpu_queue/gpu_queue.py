#!/usr/bin/env python3
"""
Multi-GPU Training Queue & Time-Window Dispatcher (`gpu_queue.py`).

Manages training jobs across local Docker GPUs (24/7) and remote colleague
SSH GPUs (e.g. 18:00-05:00 night window + weekends), checking `nvidia-smi`
VRAM/utilization before dispatching and automatically syncing code, Zarr
datasets, logs, and checkpoints via `rsync`.
"""

import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "gpu_queue.yaml"


def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict[str, Any], path: Path = DEFAULT_CONFIG_PATH) -> None:
    tmp_path = path.with_suffix(".yaml.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    tmp_path.replace(path)


def is_in_schedule(schedule_str: str, weekend_always: bool = True, now: Optional[dt.datetime] = None) -> bool:
    """Check whether current local time falls inside the allowed start window."""
    if now is None:
        now = dt.datetime.now()
    s = str(schedule_str).strip().lower()
    if s in ("always", "24/7", "*"):
        return True
    if weekend_always and now.weekday() in (5, 6):
        return True

    # Parse "HH:MM-HH:MM"
    m = re.match(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$", s)
    if not m:
        raise ValueError(f"Invalid schedule format: {schedule_str!r}. Expected 'always' or 'HH:MM-HH:MM'.")

    sh, sm, eh, em = map(int, m.groups())
    start_t = dt.time(sh, sm)
    end_t = dt.time(eh, em)
    cur_t = now.time()

    if start_t <= end_t:
        return start_t <= cur_t < end_t
    else:
        # Overnight window, e.g. 18:00-05:00
        return cur_t >= start_t or cur_t < end_t


def query_worker_gpu(w_name: str, w_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Query nvidia-smi on a local docker or remote ssh worker."""
    gpu_id = w_cfg.get("gpu_id", 0)
    smi_cmd = (
        f"nvidia-smi --id={gpu_id} "
        f"--query-gpu=name,memory.free,memory.total,utilization.gpu "
        f"--format=csv,noheader,nounits"
    )
    w_type = w_cfg.get("type", "docker")
    target = w_cfg["target"]

    if w_type == "docker":
        cmd = ["bash", "-c", smi_cmd]
    elif w_type == "ssh":
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", target, smi_cmd]
    else:
        return {"online": False, "error": f"Unknown type {w_type}"}

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        if res.returncode != 0 or not res.stdout.strip():
            return {"online": False, "error": res.stderr.strip() or "nvidia-smi failed"}
        line = res.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        gpu_name = parts[0]
        free_mb = int(float(parts[1]))
        total_mb = int(float(parts[2]))
        util_pct = int(float(parts[3]))
        min_free = int(w_cfg.get("min_free_vram_mb", 10000))
        max_util = int(w_cfg.get("max_gpu_util_pct", 25))
        is_free = (free_mb >= min_free) and (util_pct <= max_util)
        return {
            "online": True,
            "gpu_name": gpu_name,
            "free_mb": free_mb,
            "total_mb": total_mb,
            "util_pct": util_pct,
            "is_free": is_free,
        }
    except Exception as e:
        return {"online": False, "error": str(e)}


def parse_log_progress(log_path: Optional[Path]) -> str:
    """Extract latest epoch and val error/loss from a training log file."""
    if not log_path or not log_path.exists():
        return "-"
    try:
        # Read last 32 KB (excluding any 'rm:' lines)
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 32768), os.SEEK_SET)
            raw_tail = f.read().decode("utf-8", errors="replace")
        tail = "\n".join(line for line in raw_tail.splitlines() if not line.startswith("rm:"))

        prg_tqdm = re.findall(r"Epoch\s+(\d+)/(\d+):\s+(\d+)%", tail)
        prg_eval = re.findall(r"Epoch\s+(\d+):\s+Train Loss=([0-9.]+).*?Val Loss=([0-9.]+).*?FPR.*?=([0-9.]+)%", tail)
        epochs = re.findall(r"Epoch\s+(\d+)/(\d+)", tail)
        tqdm_ep = re.findall(r"Training epoch\s+(\d+):\s+(\d+%)", tail)
        ckpts = re.findall(r"epoch=(\d+)-val_action_mse=([0-9.]+)\.ckpt", tail)
        val_mses = re.findall(r"val_action_mse[:=\s]+([0-9.]+)", tail)

        info = []
        if prg_tqdm:
            ep, total, pct = prg_tqdm[-1]
            info.append(f"Ep {ep}/{total} ({pct}%)")
        elif tqdm_ep:
            ep, pct = tqdm_ep[-1]
            info.append(f"Ep {ep}/80 ({pct})")
        elif epochs:
            ep, total = epochs[-1]
            info.append(f"Ep {ep}/{total}")
        elif ckpts:
            info.append(f"Ep {int(ckpts[-1][0])}")

        if prg_eval:
            info.append(f"val_loss={float(prg_eval[-1][2]):.4f}")
            info.append(f"fpr={float(prg_eval[-1][3]):.1f}%")
        elif ckpts:
            info.append(f"best_ckpt={float(ckpts[-1][1]):.5f}")
        elif val_mses:
            info.append(f"val_mse={float(val_mses[-1]):.5f}")

        return ", ".join(info) if info else "starting..."
    except Exception:
        return "-"


def parse_log_detailed(log_path: Optional[Path], started_at_str: Optional[str] = None) -> Dict[str, Any]:
    """Return structured progress info (epoch, pct, speed, eta_str, best_mse, loss) for dashboard."""
    res: Dict[str, Any] = {
        "ep": 0,
        "total_ep": 80,
        "ep_pct": 0,
        "overall_pct": 0.0,
        "eta_str": "-",
        "best_mse": None,
        "loss": None,
        "val_loss": None,
        "fpr_09": None,
        "it_s": None,
        "summary": "-",
    }
    if not log_path or not log_path.exists():
        return res
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 32768), os.SEEK_SET)
            raw_tail = f.read().decode("utf-8", errors="replace")
        tail = "\n".join(line for line in raw_tail.splitlines() if not line.startswith("rm:"))

        prg_tqdm = re.findall(r"Epoch\s+(\d+)/(\d+):\s+(\d+)%", tail)
        prg_eval = re.findall(r"Epoch\s+(\d+):\s+Train Loss=([0-9.]+).*?Val Loss=([0-9.]+).*?FPR.*?=([0-9.]+)%", tail)
        tqdm_ep = re.findall(r"Training epoch\s+(\d+):\s+(\d+)%", tail)
        ckpts = re.findall(r"epoch=(\d+)-val_action_mse=([0-9.]+)\.ckpt", tail)
        val_mses = re.findall(r"val_action_mse[:=\s]+([0-9.]+)", tail)
        losses = re.findall(r"loss=([0-9.]+)", tail)
        speeds = re.findall(r"([0-9.]+)it/s", tail)

        if prg_tqdm:
            ep_str, total_str, pct_str = prg_tqdm[-1]
            ep = int(ep_str)
            total_ep = int(total_str)
            ep_pct = int(pct_str)
            res["ep"] = ep
            res["total_ep"] = total_ep
            res["ep_pct"] = ep_pct
            res["overall_pct"] = min(100.0, ((ep - 1 + ep_pct / 100.0) / max(1, total_ep)) * 100.0)
        elif prg_eval:
            ep = int(prg_eval[-1][0])
            res["ep"] = ep
            res["overall_pct"] = min(100.0, (ep / max(1, res["total_ep"])) * 100.0)
        elif tqdm_ep:
            ep_str, pct_str = tqdm_ep[-1]
            ep = int(ep_str)
            ep_pct = int(pct_str)
            res["ep"] = ep
            res["ep_pct"] = ep_pct
            res["overall_pct"] = min(100.0, ((ep + ep_pct / 100.0) / 80.0) * 100.0)
        elif ckpts:
            ep = int(ckpts[-1][0])
            res["ep"] = ep
            res["overall_pct"] = min(100.0, (ep / 80.0) * 100.0)

        if prg_eval:
            res["val_loss"] = float(prg_eval[-1][2])
            res["best_mse"] = min(float(e[2]) for e in prg_eval)
            res["fpr_09"] = float(prg_eval[-1][3])
        elif ckpts:
            res["best_mse"] = min(float(c[1]) for c in ckpts)
        elif val_mses:
            res["best_mse"] = float(val_mses[-1])

        if losses:
            res["loss"] = float(losses[-1])
        if speeds:
            res["it_s"] = float(speeds[-1])

        if started_at_str and res["overall_pct"] > 1.0 and res["overall_pct"] < 99.9:
            try:
                t_start = dt.datetime.strptime(started_at_str, "%Y-%m-%d %H:%M:%S")
                elapsed_s = (dt.datetime.now() - t_start).total_seconds()
                if elapsed_s > 15:
                    rem_s = elapsed_s * (100.0 - res["overall_pct"]) / res["overall_pct"]
                    m, s = divmod(int(rem_s), 60)
                    h, m = divmod(m, 60)
                    res["eta_str"] = f"{h}h{m:02d}m" if h > 0 else f"{m}m{s:02d}s"
            except Exception:
                pass

        res["summary"] = parse_log_progress(log_path)
        return res
    except Exception:
        return res


def resolve_local_run_dir(local_outputs_dir: Path, rel_dir: str) -> Path:
    """Find run_dir directly or inside subdirectories (baselines, ablation, progress_heads)."""
    direct = local_outputs_dir / rel_dir
    if direct.exists():
        return direct
    for sub in ("baselines", "ablation", "progress_heads"):
        cand = local_outputs_dir / sub / rel_dir
        if cand.exists():
            return cand
    return direct


def load_eval_summary(job: Dict[str, Any], settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load `<run_dir>/eval_summary.json` if present."""
    _, rel_dir = job_exp_and_rel_dir(job)
    local_run_dir = resolve_local_run_dir(Path(settings["local_outputs_dir"]), rel_dir)
    summary_path = local_run_dir / "eval_summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def run_postflight_eval(job: Dict[str, Any], settings: Dict[str, Any], log_f: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """Run `eval_run.py` inside local `ros2_dev` container on the synced run directory."""
    if job.get("type") == "progress_head":
        return load_eval_summary(job, settings)

    _, rel_dir = job_exp_and_rel_dir(job)
    local_base = Path(settings["local_outputs_dir"])
    local_run_dir = resolve_local_run_dir(local_base, rel_dir)
    if not (local_run_dir / "checkpoints").is_dir():
        return None

    out_dir = settings.get("output_dir", "/ros2_ws/data/training/models")
    try:
        rel_to_base = local_run_dir.relative_to(local_base)
        container_run_dir = f"{out_dir}/{rel_to_base}"
    except ValueError:
        container_run_dir = f"{out_dir}/{rel_dir}"

    msg = f"\n[{now_str()}] [EVAL] Running post-training evaluation for {job['id']} ({rel_dir})...\n"
    if log_f:
        log_f.write(msg)
        log_f.flush()
    else:
        print(msg.strip())

    eval_shell = (
        "source /ros2_ws/venv/bin/activate && "
        "export PYTHONPATH=/ros2_ws/third_party/ManiFlow_Policy/ManiFlow:${PYTHONPATH:-} && "
        f"python3 /ros2_ws/third_party/ManiFlow_Policy/ManiFlow/scripts/eval_run.py "
        f"--run-dir '{container_run_dir}' --n-batches 6 --ablate-batches 8"
    )
    cmd = ["docker", "exec", "ros2_dev", "bash", "-c", eval_shell]
    if log_f:
        subprocess.run(cmd, stdout=log_f, stderr=log_f)
    else:
        subprocess.run(cmd)
    return load_eval_summary(job, settings)


def job_exp_and_rel_dir(job: Dict[str, Any]) -> Tuple[str, str]:
    """Return (exp_name, relative_run_dir) matching train_policy.sh or progress_head convention."""
    if job.get("type") == "progress_head":
        exp_name = job["id"]
        rel_dir = job.get("rel_dir", f"progress_heads/{job['id']}")
        return exp_name, rel_dir

    alg = job["alg"]
    task = job["task"]
    addition_info = job["id"]
    seed = job.get("seed", 0)
    exp_name = f"{task}-{alg}-{addition_info}"
    rel_dir = f"{exp_name}_seed{seed}"
    return exp_name, rel_dir


def check_worker_dataset_present(w_cfg: Dict[str, Any], job: Dict[str, Any], settings: Dict[str, Any]) -> bool:
    """Return True if all dataset files needed for `job` already exist on worker."""
    w_type = w_cfg.get("type", "docker")
    if w_type == "docker":
        return True  # Local PC always has datasets locally mounted

    target = w_cfg.get("target")
    remote_zarr_dir = w_cfg.get("zarr_dir")
    if not target or not remote_zarr_dir:
        return False

    is_progress_head = (job.get("type") == "progress_head")
    if is_progress_head:
        split_name = job.get("split_json", "progress_split_70_30.json")
        local_split = Path(settings["local_zarr_dir"]) / split_name
        if not local_split.exists():
            return False
        try:
            with open(local_split, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            stores_cfg = manifest.get("metadata", {}).get("stores", {})
            if not stores_cfg:
                return False
            checks = []
            for s_name, s_path in stores_cfg.items():
                bname = os.path.basename(s_path.rstrip("/"))
                chk_str = f"(test -f '{remote_zarr_dir}/{bname}/.sync_complete' || (test -d '{remote_zarr_dir}/{bname}/state' && test -f '{remote_zarr_dir}/{bname}/wrist_dinov3_tokens/.zarray'))"
                checks.append(chk_str)
            cmd = " && ".join(checks)
            res = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", target, cmd],
                capture_output=True,
            )
            return (res.returncode == 0)
        except Exception:
            return False
    else:
        dataset = job.get("dataset", "")
        if not dataset:
            return True
        cmd = f"(test -f '{remote_zarr_dir}/{dataset}/.sync_complete' || test -d '{remote_zarr_dir}/{dataset}/state')"
        res = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", target, cmd],
            capture_output=True,
        )
        return (res.returncode == 0)


def get_worker_priority_tier(w_name: str, w_cfg: Dict[str, Any], job: Dict[str, Any], settings: Dict[str, Any]) -> Tuple[int, str]:
    """
    Compute worker priority tier (1=Highest, 4=Lowest) for a given job:
      Tier 1: Local PC (type == 'docker' or connection == 'local')
      Tier 2: Remote PC with dataset already present (cache hit)
      Tier 3: Remote PC without dataset via fast SSH / Ethernet
      Tier 4: Remote PC without dataset via WLAN / wifi
    """
    w_type = w_cfg.get("type", "docker")
    conn = w_cfg.get("connection", "local" if w_type == "docker" else "ethernet").lower()

    if w_type == "docker" or conn == "local":
        return 1, "P1:Local"

    has_dataset = check_worker_dataset_present(w_cfg, job, settings)
    if has_dataset:
        return 2, "P2:Cached"

    if conn in ("wifi", "wlan"):
        return 4, "P4:WifiNoData"
    else:
        return 3, "P3:LanNoData"


def preflight_sync_ssh(w_name: str, w_cfg: Dict[str, Any], job: Dict[str, Any], settings: Dict[str, Any], log_f) -> bool:
    """Sync ManiFlow code and ensure dataset exists on remote SSH worker."""
    target = w_cfg["target"]
    local_code = Path(settings["local_code_dir"])
    remote_code = f"{w_cfg['code_dir']}/ManiFlow"
    remote_zarr_dir = w_cfg["zarr_dir"]

    log_f.write(f"[{now_str()}] [PREFLIGHT:{w_name}] Syncing ManiFlow code to {target}:{remote_code}...\n")
    log_f.flush()

    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", target, f"mkdir -p '{remote_code}/maniflow' '{remote_code}/scripts' '{w_cfg['code_dir']}/scripts' '{remote_zarr_dir}' '{w_cfg['output_dir']}'"],
        stdout=log_f,
        stderr=log_f,
    )
    subprocess.run(
        [
            "rsync", "-az",
            f"{local_code.parent}/scripts/",
            f"{target}:{w_cfg['code_dir']}/scripts/",
        ],
        stdout=log_f,
        stderr=log_f,
    )
    subprocess.run(
        [
            "rsync", "-az",
            f"{local_code}/scripts/",
            f"{target}:{remote_code}/scripts/",
        ],
        stdout=log_f,
        stderr=log_f,
    )
    res = subprocess.run(
        [
            "rsync", "-az",
            f"{local_code}/maniflow/",
            f"{target}:{remote_code}/maniflow/",
        ],
        stdout=log_f,
        stderr=log_f,
    )
    if res.returncode != 0:
        log_f.write(f"[{now_str()}] [PREFLIGHT:{w_name}] ERROR: Code rsync failed!\n")
        log_f.flush()
        return False

    is_progress_head = (job.get("type") == "progress_head")
    if is_progress_head:
        split_name = job.get("split_json", "progress_split_70_30.json")
        local_split = Path(settings["local_zarr_dir"]) / split_name
        if not local_split.exists():
            log_f.write(f"[{now_str()}] [PREFLIGHT:{w_name}] ERROR: Local split JSON {local_split} not found!\n")
            log_f.flush()
            return False
        subprocess.run(
            ["rsync", "-az", f"{local_split}", f"{target}:{remote_zarr_dir}/"],
            stdout=log_f,
            stderr=log_f,
        )
        try:
            with open(local_split, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            stores_cfg = manifest.get("metadata", {}).get("stores", {})
            for s_name, s_path in stores_cfg.items():
                bname = os.path.basename(s_path.rstrip("/"))
                local_s_zarr = Path(settings["local_zarr_dir"]) / bname
                remote_s_zarr = f"{remote_zarr_dir}/{bname}"
                chk = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", target, f"test -d '{remote_s_zarr}'"],
                    capture_output=True,
                )
                if chk.returncode != 0:
                    if not local_s_zarr.exists():
                        log_f.write(f"[{now_str()}] [PREFLIGHT:{w_name}] ERROR: Local dataset {local_s_zarr} not found!\n")
                        log_f.flush()
                        return False
                    log_f.write(f"[{now_str()}] [PREFLIGHT:{w_name}] Progress store '{s_name}' ({bname}) missing on {target}. Syncing via rsync...\n")
                    log_f.flush()
                    res_z = subprocess.run(
                        [
                            "rsync", "-az",
                            "--exclude=front_rgb",
                            "--exclude=wrist_rgb",
                            "--exclude=point_cloud",
                            f"{local_s_zarr}",
                            f"{target}:{remote_zarr_dir}/",
                        ],
                        stdout=log_f,
                        stderr=log_f,
                    )
                    if res_z.returncode != 0:
                        log_f.write(f"[{now_str()}] [PREFLIGHT:{w_name}] ERROR: Dataset {bname} rsync failed!\n")
                        log_f.flush()
                        return False
                    subprocess.run(
                        ["ssh", "-o", "BatchMode=yes", target, f"touch '{remote_s_zarr}/.sync_complete'"],
                        stdout=log_f,
                        stderr=log_f,
                    )
        except Exception as e:
            log_f.write(f"[{now_str()}] [PREFLIGHT:{w_name}] ERROR inspecting split JSON: {e}\n")
            log_f.flush()
            return False
    else:
        local_zarr = Path(settings["local_zarr_dir"]) / job["dataset"]
        remote_zarr = f"{remote_zarr_dir}/{job['dataset']}"
        chk = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", target, f"test -d '{remote_zarr}/state'"],
            capture_output=True,
        )
        if chk.returncode != 0:
            if not local_zarr.exists():
                log_f.write(f"[{now_str()}] [PREFLIGHT:{w_name}] ERROR: Local dataset {local_zarr} not found!\n")
                log_f.flush()
                return False
            log_f.write(f"[{now_str()}] [PREFLIGHT:{w_name}] Dataset {job['dataset']} missing on {target}. Syncing via rsync...\n")
            log_f.flush()
            subprocess.run(["ssh", "-o", "BatchMode=yes", target, f"mkdir -p '{remote_zarr_dir}'"], stdout=log_f, stderr=log_f)
            res_z = subprocess.run(
                ["rsync", "-az", f"{local_zarr}", f"{target}:{remote_zarr_dir}/"],
                stdout=log_f,
                stderr=log_f,
            )
            if res_z.returncode != 0:
                log_f.write(f"[{now_str()}] [PREFLIGHT:{w_name}] ERROR: Dataset rsync failed!\n")
                log_f.flush()
                return False
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", target, f"touch '{remote_zarr}/.sync_complete'"],
                stdout=log_f,
                stderr=log_f,
            )

    return True


def postflight_sync_ssh(w_name: str, w_cfg: Dict[str, Any], job: Dict[str, Any], settings: Dict[str, Any], log_f) -> bool:
    """Sync completed run directory from remote SSH worker back to local PC."""
    target = w_cfg["target"]
    _, rel_dir = job_exp_and_rel_dir(job)
    remote_run_dir = f"{w_cfg['output_dir']}/{rel_dir}"
    local_run_dir = resolve_local_run_dir(Path(settings["local_outputs_dir"]), rel_dir)
    local_run_dir.mkdir(parents=True, exist_ok=True)

    log_f.write(f"\n[{now_str()}] [POSTFLIGHT:{w_name}] Syncing checkpoints from {target}:{remote_run_dir} -> {local_run_dir}...\n")
    log_f.flush()

    res = subprocess.run(
        ["rsync", "-az", f"{target}:{remote_run_dir}/", f"{local_run_dir}/"],
        stdout=log_f,
        stderr=log_f,
    )
    if res.returncode != 0:
        log_f.write(f"[{now_str()}] [POSTFLIGHT:{w_name}] ERROR: Checkpoint rsync failed!\n")
        log_f.flush()
        return False

    if w_cfg.get("cleanup_remote_after_sync", False):
        log_f.write(f"[{now_str()}] [POSTFLIGHT:{w_name}] Cleaning up remote .ckpt files in {remote_run_dir}/checkpoints/...\n")
        log_f.flush()
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", target, f"rm -f '{remote_run_dir}/checkpoints/'*.ckpt"],
            stdout=log_f,
            stderr=log_f,
        )
    return True


def is_ssh_disconnect_error(ret: int, log_path: Optional[Path], w_cfg: Dict[str, Any]) -> bool:
    """Check if process exit was due to network disconnection or SSH failure."""
    if w_cfg.get("type") != "ssh":
        return False
    if ret in (255, -signal.SIGPIPE, 124, 137):
        return True

    if log_path and log_path.exists():
        try:
            with open(log_path, "rb") as f:
                f.seek(max(0, f.seek(0, os.SEEK_END) - 4096), os.SEEK_SET)
                tail_txt = f.read().decode("utf-8", errors="ignore")
            signals = [
                "Timeout, server",
                "Connection timed out",
                "Connection closed by remote host",
                "Broken pipe",
                "No route to host",
                "Network is unreachable",
                "Host key verification failed",
                "ssh: connect to host",
            ]
            if any(s.lower() in tail_txt.lower() for s in signals):
                return True
        except Exception:
            pass
    return False


def inspect_remote_job_state(w_cfg: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect remote SSH worker to check if training process is alive and what checkpoints exist."""
    target = w_cfg.get("target")
    if not target:
        return {"online": False, "process_alive": False, "checkpoints": []}

    _, rel_dir = job_exp_and_rel_dir(job)
    remote_run_dir = f"{w_cfg['output_dir']}/{rel_dir}"
    job_id = job["id"]

    probe_script = f"""
    alive=0
    if pgrep -f '{job_id}' >/dev/null 2>&1; then
        alive=1
    fi
    echo "PROCESS_ALIVE=$alive"
    echo "---CHECKPOINTS---"
    if [ -d '{remote_run_dir}/checkpoints' ]; then
        ls -1 '{remote_run_dir}/checkpoints/'*.ckpt 2>/dev/null || true
    fi
    if [ -f '{remote_run_dir}/best_progress_head.pt' ]; then
        echo '{remote_run_dir}/best_progress_head.pt'
    fi
    if [ -f '{remote_run_dir}/latest.pt' ]; then
        echo '{remote_run_dir}/latest.pt'
    fi
    """
    try:
        res = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", target, probe_script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode != 0:
            return {"online": False, "process_alive": False, "checkpoints": []}

        lines = res.stdout.strip().splitlines()
        alive = False
        ckpts = []
        in_ckpts = False
        for l in lines:
            l = l.strip()
            if l.startswith("PROCESS_ALIVE="):
                alive = (l.split("=")[1].strip() == "1")
            elif l == "---CHECKPOINTS---":
                in_ckpts = True
            elif in_ckpts and l:
                ckpts.append(l)

        return {
            "online": True,
            "process_alive": alive,
            "checkpoints": ckpts,
            "remote_run_dir": remote_run_dir,
        }
    except Exception:
        return {"online": False, "process_alive": False, "checkpoints": []}


def handle_worker_reconnected(
    w_name: str,
    w_cfg: Dict[str, Any],
    disc_info: Dict[str, Any],
    cfg: Dict[str, Any],
    settings: Dict[str, Any],
) -> None:
    """
    Triggered as soon as a previously disconnected PC comes back online.
    Inspects what happened to the job that was running mid-process when it disconnected,
    reconciles/syncs checkpoints, and restores worker availability.
    """
    target = w_cfg.get("target")
    interrupted_job_id = disc_info.get("job_id")
    print(f"\n[{now_str()}] [RECONNECTED] Worker '{w_name}' ({target}) is BACK ONLINE after disconnecting mid-process!")
    sys.stdout.flush()

    jobs = cfg.get("jobs", [])
    job_obj = next((j for j in jobs if j["id"] == interrupted_job_id), None)

    if job_obj is not None:
        remote_state = inspect_remote_job_state(w_cfg, job_obj)
        ckpts = remote_state.get("checkpoints", [])
        is_alive = remote_state.get("process_alive", False)

        print(
            f"[{now_str()}] [RECONNECT-INSPECT] Worker '{w_name}' mid-process inspection: "
            f"process_alive={is_alive}, remote_checkpoints={len(ckpts)}"
        )
        sys.stdout.flush()

        current_status = job_obj.get("status")
        current_worker = job_obj.get("worker")

        _, rel_dir = job_exp_and_rel_dir(job_obj)
        remote_dir = f"{w_cfg['output_dir']}/{rel_dir}"

        if (current_status == "completed" or current_status == "running") and current_worker != w_name:
            # Job was already taken over and completed or actively running on another worker -> redundant!
            print(
                f"[{now_str()}] [RECONNECT-DISCARD] Job '{interrupted_job_id}' is already {current_status} on '{current_worker}'. "
                f"Skipping redundant artifact sync (not saving anywhere)."
            )
            if is_alive:
                print(f"[{now_str()}] [RECONNECT-STOP] Terminating redundant orphaned process on '{w_name}'...")
                subprocess.run(["ssh", "-o", "BatchMode=yes", target, f"pkill -f '{interrupted_job_id}'"], capture_output=True)
            # Clean up remote redundant directory
            subprocess.run(["ssh", "-o", "BatchMode=yes", target, f"rm -rf '{remote_dir}'"], capture_output=True)

        else:
            if ckpts:
                local_run_dir = resolve_local_run_dir(Path(settings["local_outputs_dir"]), rel_dir)
                local_run_dir.mkdir(parents=True, exist_ok=True)
                print(f"[{now_str()}] [RECONNECT-SYNC] Syncing remote checkpoints from '{w_name}' to {local_run_dir}...")
                subprocess.run(
                    ["rsync", "-az", f"{target}:{remote_dir}/", f"{local_run_dir}/"],
                    capture_output=True,
                )
                if settings.get("auto_eval_after_training", True):
                    run_postflight_eval(job_obj, settings)
                print(f"[{now_str()}] [RECONNECT-RECOVER] Restored {len(ckpts)} checkpoint(s) from '{w_name}'.")

    cfg.setdefault("disconnected_workers", {}).pop(w_name, None)
    print(f"[{now_str()}] [RECONNECT-READY] Worker '{w_name}' cache cleared. Worker is now READY for new tasks.\n")
    sys.stdout.flush()


def launch_job_process(
    w_name: str,
    w_cfg: Dict[str, Any],
    job: Dict[str, Any],
    settings: Dict[str, Any],
) -> Tuple[Optional[subprocess.Popen], Optional[Any], Path]:
    """Launch training job on worker and return (Popen, log_file_handle, log_path)."""
    log_dir = Path(settings["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{ts}_{job['id']}_{w_name}.log"
    log_f = open(log_path, "w", encoding="utf-8")

    exp_name, rel_dir = job_exp_and_rel_dir(job)
    seed = job.get("seed", 0)
    gpu_id = w_cfg.get("gpu_id", 0)
    extra_args = job.get("extra_args", "")
    w_type = w_cfg.get("type", "docker")

    if w_type == "ssh":
        # Remove stale local output directory before remote run
        local_base = Path(settings["local_outputs_dir"])
        local_run_dir = resolve_local_run_dir(local_base, rel_dir)
        if local_run_dir.exists():
            out_base = settings.get("output_dir", "/ros2_ws/data/training/models")
            try:
                rel_to_base = local_run_dir.relative_to(local_base)
                cont_p = f"{out_base}/{rel_to_base}"
            except ValueError:
                cont_p = f"{out_base}/{rel_dir}"
            subprocess.run(["docker", "exec", "ros2_dev", "rm", "-rf", cont_p], stdout=log_f, stderr=log_f)
        if not preflight_sync_ssh(w_name, w_cfg, job, settings, log_f):
            log_f.close()
            return None, None, log_path

    code_dir = w_cfg["code_dir"]
    venv_act = w_cfg["venv_activate"]
    run_dir = f"{w_cfg['output_dir']}/{rel_dir}"

    is_progress_head = (job.get("type") == "progress_head")
    if is_progress_head:
        split_name = job.get("split_json", "progress_split_70_30.json")
        split_path = f"{w_cfg['zarr_dir']}/{split_name}"
        epochs = job.get("epochs", 30)
        batch_size = job.get("batch_size", 128)
        lr = job.get("lr", 1e-3)
        train_shell = (
            f"set -e; "
            f"rm -rf '{run_dir}'; "
            f"mkdir -p '{run_dir}'; "
            f"source '{venv_act}'; "
            f"export PYTHONPATH='{code_dir}/ManiFlow':${{PYTHONPATH:-}}; "
            f"export CUDA_VISIBLE_DEVICES={gpu_id}; "
            f"cd '{code_dir}/ManiFlow'; "
            f"python3 -u scripts/train_progress_head.py "
            f"--split_json '{split_path}' "
            f"--save_dir '{run_dir}' "
            f"--epochs {epochs} "
            f"--batch_size {batch_size} "
            f"--lr {lr} "
            f"--seed {seed} "
            f"{extra_args}"
        )
    else:
        zarr_path = f"{w_cfg['zarr_dir']}/{job['dataset']}"
        train_shell = (
            f"set -e; "
            f"rm -rf '{run_dir}'; "
            f"source '{venv_act}'; "
            f"export PYTHONPATH='{code_dir}/ManiFlow':${{PYTHONPATH:-}}; "
            f"export VK_ICD_FILENAMES='{code_dir}/scripts/nvidia_icd.json'; "
            f"export TOKENIZERS_PARALLELISM=false; "
            f"export HYDRA_FULL_ERROR=1; "
            f"export CUDA_VISIBLE_DEVICES={gpu_id}; "
            f"cd '{code_dir}/ManiFlow/maniflow/workspace'; "
            f"python3 -u train.py "
            f"--config-name={job['alg']}.yaml "
            f"task={job['task']} "
            f"task.dataset.zarr_path='{zarr_path}' "
            f"hydra.run.dir='{run_dir}' "
            f"training.debug=False "
            f"training.seed={seed} "
            f"training.device='cuda:0' "
            f"exp_name='{exp_name}' "
            f"logging.mode=online "
            f"checkpoint.save_ckpt=True "
            f"{extra_args}"
        )

    if w_type == "docker":
        cmd = ["docker", "exec", w_cfg["target"], "bash", "-c", train_shell]
    elif w_type == "ssh":
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", w_cfg["target"], train_shell]
    else:
        raise ValueError(f"Unsupported worker type: {w_type}")

    log_f.write(f"[{now_str()}] Launching {job['id']} on {w_name} ({w_type}:{w_cfg['target']})...\n")
    log_f.write(f"[{now_str()}] Command: {' '.join(cmd)}\n\n")
    log_f.flush()

    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, log_f, log_path


def _make_bar(pct: float, width: int = 12) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round((pct / 100.0) * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def render_dashboard_str(cfg: Dict[str, Any]) -> str:
    now = dt.datetime.now()
    settings = cfg["settings"]
    lines = [
        "=" * 118,
        f" Multi-GPU Training Queue & Evaluation Dashboard  ({now.strftime('%Y-%m-%d %H:%M:%S')})",
        "=" * 118,
    ]

    daemon_pid_file = Path(settings["log_dir"]) / "daemon.pid"
    daemon_state = "STOPPED"
    if daemon_pid_file.exists():
        try:
            pid = int(daemon_pid_file.read_text().strip())
            os.kill(pid, 0)
            daemon_state = f"RUNNING (PID {pid})"
        except Exception:
            daemon_state = "STOPPED (stale pidfile)"
    lines.append(f" Dispatcher Daemon: {daemon_state}\n")

    lines.append(" [GPU WORKERS]")
    lines.append(
        f" {'Worker':<18} {'Target':<18} {'Schedule':<12} {'Win':<6} {'VRAM Usage':<26} {'GPU Util':<18} {'State'}"
    )
    lines.append(" " + "-" * 116)
    worker_disp_map = {k: v.get("display_name", k) for k, v in cfg.get("workers", {}).items()}
    for w_name, w_cfg in cfg.get("workers", {}).items():
        w_disp = w_cfg.get("display_name", w_name)
        if not w_cfg.get("enabled", True):
            lines.append(f" {w_disp:<18} {'disabled':<18} {'-':<12} {'-':<6} {'-':<26} {'-':<18} DISABLED")
            continue
        sched = w_cfg.get("schedule", "always")
        in_win = is_in_schedule(sched, w_cfg.get("weekend_always", True), now)
        win_str = "OPEN" if in_win else "CLOSED"
        info = query_worker_gpu(w_name, w_cfg)
        tt = f"{w_cfg['type']}:{w_cfg['target']}"
        if not info["online"]:
            vram_str = f"OFFLINE ({info.get('error', '')[:16]})"
            util_str = "-"
            avail = "UNAVAILABLE"
        else:
            used_mb = info["total_mb"] - info["free_mb"]
            vram_pct = (used_mb / max(1, info["total_mb"])) * 100.0
            vram_str = f"{used_mb:5d}/{info['total_mb']:5d} MB ({vram_pct:2.0f}%)"
            util_str = f"{_make_bar(info['util_pct'], 8)} {info['util_pct']:3d}%"
            if info["is_free"] and in_win:
                avail = "READY"
            elif not in_win:
                avail = f"WAIT ({sched})"
            else:
                avail = "TRAINING / BUSY"
        lines.append(f" {w_disp:<18} {tt:<18} {sched:<12} {win_str:<6} {vram_str:<26} {util_str:<18} {avail}")

    lines.append("\n [TRAINING QUEUE]")
    lines.append(
        f" {'#':<3} {'Job ID':<34} {'Status':<10} {'Worker':<18} {'Progress':<26} {'ETA':<9} {'Best Val MSE / Eval'}"
    )
    lines.append(" " + "-" * 116)

    eval_summaries: List[Tuple[str, Dict[str, Any]]] = []
    for idx, job in enumerate(cfg.get("jobs", []), 1):
        st = job.get("status", "pending")
        w = job.get("worker", "-")
        w_disp = worker_disp_map.get(w, w)
        log_p = Path(job["log_file"]) if job.get("log_file") else None
        det = parse_log_detailed(log_p, job.get("started_at")) if st in ("running", "completed", "failed") else None
        ev = load_eval_summary(job, settings)
        if ev is not None:
            eval_summaries.append((job["id"], ev))

        if st in ("pending", "queued"):
            prog_col = f"{_make_bar(0, 10)}   0%"
            eta_col = "-"
            mse_col = "-"
        elif st == "running" and det:
            total_ep = det.get("total_ep", 80)
            prog_col = f"{_make_bar(det['overall_pct'], 10)} Ep {det['ep']:02d}/{total_ep} ({det['ep_pct']:2d}%)"
            eta_col = det["eta_str"]
            parts = []
            if det["loss"] is not None:
                parts.append(f"loss={det['loss']:.4f}")
            if det["it_s"] is not None:
                parts.append(f"{det['it_s']:.1f}it/s")
            if det.get("val_loss") is not None:
                parts.append(f"val_loss={det['val_loss']:.4f}")
            if det.get("fpr_09") is not None:
                parts.append(f"fpr={det['fpr_09']:.1f}%")
            elif det["best_mse"] is not None:
                parts.append(f"val_mse={det['best_mse']:.5f}")
            mse_col = ", ".join(parts) if parts else "warming up..."
        elif st == "completed":
            total_ep = det.get("total_ep", 80) if det else 80
            prog_col = f"{_make_bar(100, 10)} Ep {total_ep}/{total_ep}"
            eta_col = "done"
            if ev is not None:
                if ev.get("job_type") == "progress_head":
                    mse_col = (
                        f"Ep {ev.get('best_epoch', 0):02d}: loss={ev.get('best_val_loss', 0):.4f}, "
                        f"mae={ev.get('best_val_mae', 0):.4f}, fpr={ev.get('best_fpr_09', 0)*100:.1f}%"
                    )
                else:
                    mse_col = (
                        f"Ep {ev['best_epoch']:02d}: mse={ev['best_val_action_mse']:.5f}, "
                        f"xy_search={ev['best_xy_search_mse']:.5f}"
                    )
            elif det and det["best_mse"] is not None:
                mse_col = f"ckpt_loss={det['best_mse']:.5f}"
            else:
                mse_col = "completed"
        else:
            prog_col = det["summary"] if det else st
            eta_col = "-"
            mse_col = "-"

        lines.append(f" {idx:<3} {job['id']:<34} {st.upper():<10} {w_disp:<18} {prog_col:<26} {eta_col:<9} {mse_col}")

    policy_evals = [e for e in eval_summaries if e[1].get("job_type") != "progress_head"]
    prg_evals = [e for e in eval_summaries if e[1].get("job_type") == "progress_head"]

    if policy_evals:
        lines.append("\n [POST-TRAINING POLICY EVALUATION (Validation Set + Modality Sensitivity)]")
        lines.append(
            f" {'Job ID':<34} {'Best Ep':<8} {'Val MSE':>10} {'XY Search':>11} {'shuf_wrist':>11} {'shuf_pc':>9} {'shuf_state':>11}"
        )
        lines.append(" " + "-" * 114)
        for j_id, ev in policy_evals:
            ab = ev.get("modality_ablation_on_best", {})
            sw = ab.get("shuffle_wrist", {}).get("ratio_vs_full", float("nan"))
            sp = ab.get("shuffle_pc", {}).get("ratio_vs_full", float("nan"))
            ss = ab.get("shuffle_state", {}).get("ratio_vs_full", float("nan"))
            sw_s = f"{sw:.2f}x" if not (sw != sw) else "-"
            sp_s = f"{sp:.2f}x" if not (sp != sp) else "-"
            ss_s = f"{ss:.2f}x" if not (ss != ss) else "-"
            lines.append(
                f" {j_id:<34} Ep {ev['best_epoch']:03d}   {ev['best_val_action_mse']:10.5f} "
                f"{ev['best_xy_search_mse']:11.5f} {sw_s:>11} {sp_s:>9} {ss_s:>11}"
            )

    if prg_evals:
        lines.append("\n [PROGRESS HEAD EVALUATION SUMMARY (Temporal Continuity & Precision)]")
        lines.append(
            f" {'Job ID':<34} {'Best Ep':<8} {'Val Loss':>10} {'Val MAE':>10} {'FPR (>=0.9)':>14} {'Smooth W':>10} {'Delta Max':>10}"
        )
        lines.append(" " + "-" * 114)
        for j_id, ev in prg_evals:
            ep = ev.get("best_epoch", 0)
            vl = ev.get("best_val_loss", 0.0)
            vm = ev.get("best_val_mae", 0.0)
            fpr = ev.get("best_fpr_09", 0.0) * 100.0
            sw = ev.get("smooth_weight", 0.0)
            dm = ev.get("delta_max", 0.03)
            lines.append(
                f" {j_id:<34} Ep {ep:03d}   {vl:10.5f} {vm:10.5f} {fpr:13.2f}% {sw:10.3f} {dm:10.3f}"
            )

    lines.append("=" * 118)
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    print(render_dashboard_str(cfg))


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Interactive live-refreshing terminal dashboard (`watch` / `dashboard`)."""
    interval = getattr(args, "interval", 3.0)
    try:
        while True:
            cfg = load_config(args.config)
            out = render_dashboard_str(cfg)
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(out + f"\n (Refreshing every {interval:.0f}s — Press Ctrl+C to exit)\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nExited live dashboard.")


def cmd_eval(args: argparse.Namespace) -> None:
    """Manually trigger post-training evaluation for a completed job (or 'all')."""
    cfg = load_config(args.config)
    settings = cfg["settings"]
    target = args.job_id
    for job in cfg.get("jobs", []):
        if target == "all" and job.get("status") == "completed":
            run_postflight_eval(job, settings)
        elif job["id"] == target:
            run_postflight_eval(job, settings)
            return


def cmd_run(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    poll_sec = int(cfg["settings"].get("poll_interval_sec", 30))
    log_dir = Path(cfg["settings"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    pid_file = log_dir / "daemon.pid"
    pid_file.write_text(str(os.getpid()))

    # Map worker_name -> (job_id, Popen, log_file_handle, log_path)
    active: Dict[str, Tuple[str, subprocess.Popen, Any, Path]] = {}

    # Recover any orphaned 'running' jobs from managed workers back to 'pending'
    recovered = 0
    managed_workers = set(cfg.get("workers", {}).keys())
    for j in cfg.get("jobs", []):
        if j.get("status") == "running" and j.get("worker") in managed_workers:
            w_name = j.get("worker")
            w_cfg = cfg.get("workers", {}).get(w_name, {})
            # If docker worker, check if training process for this job is still actively running in container!
            if w_cfg.get("type") == "docker":
                res = subprocess.run(["docker", "exec", w_cfg["target"], "pgrep", "-f", f"python.*{j['id']}"], capture_output=True)
                if res.returncode == 0:
                    print(f"[{now_str()}] Job {j['id']} is actively running in docker container {w_cfg['target']}. Monitoring...")
                    # Re-adopt into active tracking so completion is caught!
                    log_p = Path(j.get("log_file", log_dir / f"{j['id']}.log"))
                    log_f = open(log_p, "a", encoding="utf-8")
                    attach_proc = subprocess.Popen(["docker", "exec", w_cfg["target"], "bash", "-c", f"while pgrep -f 'python.*{j['id']}' >/dev/null 2>&1; do sleep 3; done"], stdout=log_f, stderr=log_f)
                    active[w_name] = (j["id"], attach_proc, log_f, log_p)
                    continue
            j["status"] = "pending"
            recovered += 1
    if recovered:
        save_config(cfg, cfg_path)
        print(f"[{now_str()}] Recovered {recovered} interrupted job(s) back to 'pending'.")

    idle_logged = False

    print(f"[{now_str()}] Starting Always-On Multi-GPU Queue Dispatcher (PID={os.getpid()}, poll={poll_sec}s)...")
    sys.stdout.flush()

    try:
        while True:
            cfg = load_config(cfg_path)
            settings = cfg["settings"]
            jobs = cfg.get("jobs", [])
            workers = cfg.get("workers", {})
            changed = False

            # 1. Check active jobs for completion or worker disconnection
            for w_name, (job_id, proc, log_f, log_path) in list(active.items()):
                ret = proc.poll()
                if ret is not None:
                    w_cfg = workers.get(w_name, {})
                    job_obj = next((j for j in jobs if j["id"] == job_id), None)
                    ok = (ret == 0)

                    # Check if failure was due to worker disconnection / SSH drop
                    disconnected = False
                    if not ok and is_ssh_disconnect_error(ret, log_path, w_cfg):
                        disconnected = True

                    if disconnected:
                        disc_entry = {
                            "worker": w_name,
                            "job_id": job_id,
                            "disconnected_at": now_str(),
                            "exit_code": ret,
                            "reason": f"ssh_disconnect (exit {ret})",
                        }
                        cfg.setdefault("disconnected_workers", {})[w_name] = disc_entry

                        log_f.write(f"\n[{now_str()}] [FAILOVER] Worker '{w_name}' disconnected mid-process (exit code {ret})!\n")
                        log_f.write(f"[{now_str()}] [FAILOVER] Cached '{w_name}' in disconnected registry. Re-queueing '{job_id}' for automatic failover to another healthy worker.\n")
                        log_f.close()
                        del active[w_name]

                        if job_obj is not None:
                            job_obj["status"] = "pending"
                            job_obj["failover_from"] = w_name
                            job_obj["failover_at"] = now_str()
                            job_obj.pop("worker", None)
                            job_obj.pop("finished_at", None)
                            changed = True

                        print(f"[{now_str()}] [FAILOVER] Worker '{w_name}' disconnected mid-process on job '{job_id}' (code {ret})!")
                        print(f"[{now_str()}] [FAILOVER] Cached '{w_name}' in disconnected registry. Re-queueing '{job_id}' for auto-failover to another machine.")
                        sys.stdout.flush()
                        continue

                    if ok and w_cfg.get("type") == "ssh" and job_obj is not None:
                        ok = postflight_sync_ssh(w_name, w_cfg, job_obj, settings, log_f)
                    if ok and job_obj is not None and settings.get("auto_eval_after_training", True):
                        run_postflight_eval(job_obj, settings, log_f=log_f)

                    log_f.write(f"\n[{now_str()}] Job {job_id} on {w_name} finished with code {ret} (success={ok}).\n")
                    log_f.close()
                    del active[w_name]

                    if job_obj is not None:
                        job_obj["status"] = "completed" if ok else "failed"
                        job_obj["finished_at"] = now_str()
                        changed = True
                    print(f"[{now_str()}] [DONE] {job_id} on {w_name} -> {'COMPLETED' if ok else 'FAILED'}")
                    sys.stdout.flush()

            # 2. Check if previously disconnected workers have come back online
            disc_workers = cfg.get("disconnected_workers", {})
            for disc_name, disc_meta in list(disc_workers.items()):
                w_cfg = workers.get(disc_name)
                if not w_cfg or not w_cfg.get("enabled", True):
                    continue
                gpu_chk = query_worker_gpu(disc_name, w_cfg)
                if gpu_chk.get("online"):
                    handle_worker_reconnected(disc_name, w_cfg, disc_meta, cfg, settings)
                    changed = True

            if changed:
                save_config(cfg, cfg_path)
                changed = False

            # 3. Check if queue is currently idle (stay alive as standard background service)
            pending_jobs = [j for j in jobs if j.get("status", "pending") in ("pending", "queued")]
            if not pending_jobs and not active:
                if not idle_logged:
                    print(f"[{now_str()}] Queue is currently empty/idle. Watching {cfg_path.name} for new pending jobs...")
                    sys.stdout.flush()
                    idle_logged = True
                time.sleep(poll_sec)
                continue
            idle_logged = False

            # 4. Try to dispatch pending jobs to available workers with 4-tier priority:
            # Tier 1: Local PC (type == 'docker' or connection == 'local')
            # Tier 2: Remote PC with dataset already present (cache hit)
            # Tier 3: Remote PC without dataset via fast SSH / Ethernet
            # Tier 4: Remote PC without dataset via WLAN / wifi
            now = dt.datetime.now()
            available_workers = []
            for w_name, w_cfg in workers.items():
                if not w_cfg.get("enabled", True):
                    continue
                if w_name in active:
                    continue
                # Exclude disconnected workers currently cached
                if w_name in cfg.get("disconnected_workers", {}):
                    continue
                sched = w_cfg.get("schedule", "always")
                if not is_in_schedule(sched, w_cfg.get("weekend_always", True), now):
                    continue
                gpu_info = query_worker_gpu(w_name, w_cfg)
                if not gpu_info.get("online") or not gpu_info.get("is_free"):
                    continue
                available_workers.append((w_name, w_cfg, gpu_info))

            while pending_jobs and available_workers:
                next_job = pending_jobs[0]

                # Rank available workers for this specific job:
                ranked = []
                for w_name, w_cfg, gpu_info in available_workers:
                    tier, tier_name = get_worker_priority_tier(w_name, w_cfg, next_job, settings)
                    ranked.append((tier, -gpu_info["free_mb"], w_name, w_cfg, gpu_info, tier_name))

                ranked.sort(key=lambda x: (x[0], x[1]))
                best_tier, _, best_w_name, best_w_cfg, best_gpu_info, tier_name = ranked[0]

                pending_jobs.pop(0)
                print(
                    f"[{now_str()}] [DISPATCH] Starting {next_job['id']} on {best_w_name} "
                    f"[{tier_name}] ({best_gpu_info['free_mb']} MB free, util {best_gpu_info['util_pct']}%)..."
                )
                sys.stdout.flush()

                proc, log_f, log_path = launch_job_process(best_w_name, best_w_cfg, next_job, settings)
                if proc is None:
                    if best_w_cfg.get("type") == "ssh" and not query_worker_gpu(best_w_name, best_w_cfg).get("online"):
                        disc_entry = {
                            "worker": best_w_name,
                            "job_id": next_job["id"],
                            "disconnected_at": now_str(),
                            "reason": "preflight_ssh_unreachable",
                        }
                        cfg.setdefault("disconnected_workers", {})[best_w_name] = disc_entry
                        print(f"[{now_str()}] [FAILOVER] Worker '{best_w_name}' unreachable during preflight. Caching as disconnected and re-dispatching {next_job['id']}...")
                        next_job["status"] = "pending"
                        next_job.pop("worker", None)
                    else:
                        next_job["status"] = "failed"
                        next_job["worker"] = best_w_name
                        next_job["log_file"] = str(log_path)
                    save_config(cfg, cfg_path)
                    print(f"[{now_str()}] [ERROR] Preflight failed for {next_job['id']} on {best_w_name}")
                    available_workers = [w for w in available_workers if w[0] != best_w_name]
                    continue

                next_job["status"] = "running"
                next_job["worker"] = best_w_name
                next_job["started_at"] = now_str()
                next_job["log_file"] = str(log_path)
                save_config(cfg, cfg_path)
                active[best_w_name] = (next_job["id"], proc, log_f, log_path)

                available_workers = [w for w in available_workers if w[0] != best_w_name]

            time.sleep(poll_sec)

    except KeyboardInterrupt:
        print(f"\n[{now_str()}] Dispatcher interrupted by user.")


def cmd_start_daemon(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    log_dir = Path(cfg["settings"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    pid_file = log_dir / "daemon.pid"
    daemon_log = log_dir / "dispatcher.log"

    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            os.kill(old_pid, 0)
            print(f"Daemon is already running (PID {old_pid}).")
            return
        except Exception:
            pass

    f_out = open(daemon_log, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(args.config), "run"],
        stdout=f_out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_file.write_text(str(proc.pid))
    print(f"Started queue daemon (PID {proc.pid}).")
    print(f"Master log: {daemon_log}")


def cmd_stop_daemon(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    pid_file = Path(cfg["settings"]["log_dir"]) / "daemon.pid"
    if not pid_file.exists():
        print("No daemon pidfile found.")
        return
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped queue daemon (PID {pid}).")
    except ProcessLookupError:
        print("Daemon process was already stopped.")
    finally:
        pid_file.unlink(missing_ok=True)


def cmd_reset(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    target_id = args.job_id
    count = 0
    for job in cfg.get("jobs", []):
        if target_id == "all" or job["id"] == target_id:
            job["status"] = "pending"
            job.pop("worker", None)
            job.pop("started_at", None)
            job.pop("finished_at", None)
            count += 1
    save_config(cfg, cfg_path)
    print(f"Reset {count} job(s) to 'pending'.")


def cmd_add(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    job_type = getattr(args, "type", "policy")
    if job_type == "progress_head":
        new_job = {
            "id": args.id,
            "type": "progress_head",
            "split_json": getattr(args, "split_json", "progress_split_70_30.json"),
            "epochs": getattr(args, "epochs", 30),
            "batch_size": getattr(args, "batch_size", 128),
            "lr": getattr(args, "lr", 1e-3),
            "extra_args": args.extra_args,
            "seed": args.seed if args.seed != 0 else 42,
            "status": "pending",
        }
    else:
        if not args.alg or not args.task:
            raise ValueError("--alg and --task are required for policy training jobs")
        new_job = {
            "id": args.id,
            "alg": args.alg,
            "task": args.task,
            "dataset": args.dataset,
            "extra_args": args.extra_args,
            "seed": args.seed,
            "status": "pending",
        }
    cfg.setdefault("jobs", []).append(new_job)
    save_config(cfg, cfg_path)
    print(f"Added job {args.id!r} ({job_type}) to queue.")


def cmd_claim_next(args: argparse.Namespace) -> None:
    """Atomically claim the next pending job for a pull-worker (no sshd needed on worker)."""
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    log_dir = Path(cfg["settings"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    for job in cfg.get("jobs", []):
        if job.get("status", "pending") in ("pending", "queued"):
            w_name = args.worker
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"{ts}_{job['id']}_{w_name}.log"
            job["status"] = "running"
            job["worker"] = w_name
            job["started_at"] = now_str()
            job["log_file"] = str(log_path)
            save_config(cfg, cfg_path)
            exp_name, rel_dir = job_exp_and_rel_dir(job)
            print(f"CLAIMED|{job['id']}|{job['alg']}|{job['task']}|{job['dataset']}|{job.get('extra_args','')}|{job.get('seed',0)}|{exp_name}|{rel_dir}|{log_path}")
            return
    print("NONE")


def cmd_finish_job(args: argparse.Namespace) -> None:
    """Mark a pull-worker job as completed or failed."""
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    for job in cfg.get("jobs", []):
        if job["id"] == args.id:
            job["status"] = args.status
            job["finished_at"] = now_str()
            save_config(cfg, cfg_path)
            print(f"Updated {args.id} -> {args.status}")
            return



def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-GPU Training Queue & Scheduler")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to gpu_queue.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Show live status of workers, job queue, and eval summaries")
    p_watch = sub.add_parser("watch", help="Live-refreshing terminal dashboard")
    p_watch.add_argument("--interval", type=float, default=3.0, help="Refresh interval in seconds")
    p_dash = sub.add_parser("dashboard", help="Alias for watch (live-refreshing terminal dashboard)")
    p_dash.add_argument("--interval", type=float, default=3.0, help="Refresh interval in seconds")
    p_eval = sub.add_parser("eval", help="Run post-training evaluation on a completed job (or 'all')")
    p_eval.add_argument("job_id", nargs="?", default="all", help="Job ID or 'all'")
    sub.add_parser("run", help="Run dispatcher loop in foreground")
    sub.add_parser("start-daemon", help="Start dispatcher loop in background")
    sub.add_parser("stop-daemon", help="Stop background dispatcher loop")

    p_reset = sub.add_parser("reset", help="Reset a job (or 'all') back to pending")
    p_reset.add_argument("job_id", help="Job ID or 'all'")

    p_add = sub.add_parser("add", help="Add a new training job to the queue")
    p_add.add_argument("--id", required=True, help="Unique run ID (e.g. 0031_...)")
    p_add.add_argument("--type", default="policy", choices=["policy", "progress_head"], help="Job type ('policy' or 'progress_head')")
    p_add.add_argument("--alg", default=None, help="Hydra config name (e.g. ablation/abl_rel_wrist_height_delta_to4)")
    p_add.add_argument("--task", default=None, help="Hydra task name")
    p_add.add_argument("--dataset", default="foil_lifting_box_recorded_action.zarr", help="Zarr dataset filename")
    p_add.add_argument("--split-json", default="progress_split_70_30.json", help="Manifest JSON for progress_head")
    p_add.add_argument("--epochs", type=int, default=30, help="Epochs for progress_head")
    p_add.add_argument("--batch-size", type=int, default=128, help="Batch size for progress_head")
    p_add.add_argument("--lr", type=float, default=1e-3, help="Learning rate for progress_head")
    p_add.add_argument("--extra-args", default="", help="Extra Hydra overrides or script args")
    p_add.add_argument("--seed", type=int, default=0, help="Random seed")

    p_claim = sub.add_parser("claim-next", help="Claim next pending job for a pull-worker")
    p_claim.add_argument("--worker", required=True, help="Worker name")

    p_fin = sub.add_parser("finish-job", help="Mark a claimed job as completed or failed")
    p_fin.add_argument("--id", required=True, help="Job ID")
    p_fin.add_argument("--status", choices=["completed", "failed", "pending"], required=True)

    args = parser.parse_args()
    if args.cmd == "status":
        cmd_status(args)
    elif args.cmd in ("watch", "dashboard"):
        cmd_dashboard(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "start-daemon":
        cmd_start_daemon(args)
    elif args.cmd == "stop-daemon":
        cmd_stop_daemon(args)
    elif args.cmd == "reset":
        cmd_reset(args)
    elif args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "claim-next":
        cmd_claim_next(args)
    elif args.cmd == "finish-job":
        cmd_finish_job(args)


if __name__ == "__main__":
    main()
