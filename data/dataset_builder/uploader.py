"""Dataset uploader — pushes built LeRobot datasets to the Prehensio data server."""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from termcolor import cprint


def _find_env_file() -> Optional[Path]:
    """Locate dataset_server.env across container and host paths."""
    candidates = [
        Path("/ros2_ws/data/configs/dataset_server.env"),
        Path(__file__).resolve().parents[1] / "configs" / "dataset_server.env",
        Path.cwd() / "configs" / "dataset_server.env",
        Path.cwd() / "data" / "configs" / "dataset_server.env",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def load_server_env(env_path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """Parse dataset_server.env with environment variable overrides."""
    env_vars: Dict[str, str] = {}
    path = Path(env_path) if env_path else _find_env_file()

    if path and path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        env_vars[k] = v
        except OSError as e:
            cprint(f"⚠️  Could not read {path}: {e}", "yellow")

    # System environment variables take precedence
    for key in (
        "DATASET_SERVER_ENABLED",
        "DATASET_SERVER",
        "DATASET_SERVER_USER",
        "DATASET_SERVER_ROOT",
        "DATASET_SERVER_KEY",
        "DATASET_SERVER_REPLACE",
        "DATASET_SERVER_VIEWER_PORT",
    ):
        if key in os.environ:
            env_vars[key] = os.environ[key]

    return env_vars


def resolve_push_config(raw_cfg: Union[bool, Dict[str, Any], None], env_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Merge YAML push_to_server setting with defaults and dataset_server.env."""
    env = load_server_env(env_path)
    default_enabled = env.get("DATASET_SERVER_ENABLED", "false").lower() in ("true", "1", "yes")

    # Base defaults
    config: Dict[str, Any] = {
        "enabled": False,
        "server": env.get("DATASET_SERVER", "192.168.50.44"),
        "user": env.get("DATASET_SERVER_USER", "dataset_server"),
        "remote_root": env.get("DATASET_SERVER_ROOT", "/data/prehensio_data"),
        "ssh_key": env.get("DATASET_SERVER_KEY", "~/.ssh/prehensio_upload"),
        "replace": env.get("DATASET_SERVER_REPLACE", "true").lower() in ("true", "1", "yes"),
        "viewer_port": int(env.get("DATASET_SERVER_VIEWER_PORT", 8080)),
    }

    if raw_cfg is True:
        config["enabled"] = True
    elif isinstance(raw_cfg, dict):
        config["enabled"] = raw_cfg.get("enabled", default_enabled)
        if "server" in raw_cfg:
            config["server"] = raw_cfg["server"]
        if "user" in raw_cfg:
            config["user"] = raw_cfg["user"]
        if "remote_root" in raw_cfg:
            config["remote_root"] = raw_cfg["remote_root"]
        if "ssh_key" in raw_cfg:
            config["ssh_key"] = raw_cfg["ssh_key"]
        if "replace" in raw_cfg:
            config["replace"] = bool(raw_cfg["replace"])
        if "viewer_port" in raw_cfg:
            config["viewer_port"] = int(raw_cfg["viewer_port"])
    elif raw_cfg is False or raw_cfg is None:
        config["enabled"] = False

    return config


def _resolve_local_dataset_dir(path_or_name: Union[str, Path]) -> Optional[Path]:
    """Find valid local LeRobot dataset directory."""
    cand = Path(path_or_name).expanduser().resolve()
    if cand.is_dir() and (cand / "meta" / "info.json").is_file():
        return cand

    # If passed name rather than full path, look inside lerobot_datasets
    name = Path(path_or_name).name
    bases = [
        Path("/ros2_ws/data/lerobot_datasets") / name,
        Path(__file__).resolve().parents[1] / "lerobot_datasets" / name,
        Path.cwd() / "lerobot_datasets" / name,
        Path.cwd() / "data" / "lerobot_datasets" / name,
    ]
    for b in bases:
        if b.is_dir() and (b / "meta" / "info.json").is_file():
            return b.resolve()

    return None


def upload_dataset(
    local_dataset_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
) -> bool:
    """Upload a LeRobot dataset to the remote Prehensio visualization server.

    Steps:
      1. Verify local LeRobot structure (meta/info.json).
      2. Verify SSH connection to remote server.
      3. Handle remote collision: if datasets/<name> exists, replace or abort.
      4. Rsync to remote incoming_builds/<name>.
      5. Checksum verification (rsync -anc).
      6. Atomic remote move into datasets/<name>.
    """
    if config is None:
        config = resolve_push_config(True)

    if not config.get("enabled", True):
        cprint("ℹ️  Dataset server upload is disabled in config.", "cyan")
        return False

    resolved_dir = _resolve_local_dataset_dir(local_dataset_dir)
    if not resolved_dir:
        cprint(f"❌ Invalid LeRobot dataset: '{local_dataset_dir}' (meta/info.json missing)", "red")
        return False

    # Ensure files are readable by rsync and the web server
    try:
        subprocess.run(["chmod", "-R", "a+rX", str(resolved_dir)], check=False)
    except Exception:
        pass

    name = dataset_name or resolved_dir.name
    server = config["server"]
    user = config["user"]
    remote_root = config["remote_root"]
    replace = config.get("replace", True)
    viewer_port = config.get("viewer_port", 8080)

    # Resolve SSH key
    ssh_key_opt = []
    raw_key = config.get("ssh_key")
    if raw_key:
        key_path = Path(raw_key).expanduser()
        if key_path.is_file():
            ssh_key_opt = ["-i", str(key_path)]
        else:
            # Fallback check: default keys
            for default_key in ("~/.ssh/prehensio_upload", "~/.ssh/id_ed25519", "~/.ssh/id_rsa"):
                cand = Path(default_key).expanduser()
                if cand.is_file():
                    ssh_key_opt = ["-i", str(cand)]
                    break

    ssh_base = ["ssh", *ssh_key_opt, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    remote_target = f"{user}@{server}"

    cprint(f"\n🚀 Pushing dataset '{name}' to {server} ...", "cyan", attrs=["bold"])

    # 1. Test SSH connectivity and root directory
    test_cmd = [*ssh_base, remote_target, f"test -d '{remote_root}'"]
    res = subprocess.run(test_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        cprint(f"❌ SSH connection to {remote_target} failed (exit {res.returncode}):", "red")
        if res.stderr.strip():
            cprint(f"   {res.stderr.strip()}", "yellow")
        cprint(f"   Ensure SSH key is authorized and {remote_root} exists.", "yellow")
        return False

    remote_dataset_dir = f"{remote_root}/datasets/{name}"
    remote_incoming_dir = f"{remote_root}/incoming_builds/{name}"
    remote_replaced_dir = f"{remote_root}/datasets_replaced"

    # 2. Collision check
    check_exists_cmd = [*ssh_base, remote_target, f"test -e '{remote_dataset_dir}'"]
    exists_res = subprocess.run(check_exists_cmd)
    if exists_res.returncode == 0:
        if not replace:
            cprint(
                f"❌ '{remote_dataset_dir}' already exists on server.\n"
                f"   Set replace: true (or DATASET_SERVER_REPLACE=true) to replace.",
                "red",
            )
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{remote_replaced_dir}/{name}.{timestamp}"
        cprint(f"📦 Existing dataset found on server; moving to {backup_dir} ...", "yellow")
        backup_cmd = [*ssh_base, remote_target, f"mkdir -p '{remote_replaced_dir}' && mv '{remote_dataset_dir}' '{backup_dir}'"]
        bk_res = subprocess.run(backup_cmd, capture_output=True, text=True)
        if bk_res.returncode != 0:
            cprint(f"❌ Failed to backup existing dataset: {bk_res.stderr.strip()}", "red")
            return False

    # 3. Clean and prepare incoming staging
    clean_cmd = [*ssh_base, remote_target, f"rm -rf '{remote_incoming_dir}'"]
    subprocess.run(clean_cmd, check=False)

    # 4. Rsync transfer
    ssh_cmd_str = " ".join(ssh_base)
    rsync_cmd = [
        "rsync",
        "-avz",
        "--partial",
        "--mkpath",
        "-e",
        ssh_cmd_str,
        f"{resolved_dir}/",
        f"{remote_target}:{remote_incoming_dir}/",
    ]
    cprint(f"⏳ Syncing files to {server}:{remote_incoming_dir} ...", "cyan")
    sync_res = subprocess.run(rsync_cmd, text=True)
    if sync_res.returncode != 0:
        cprint(f"❌ rsync transfer failed (exit {sync_res.returncode})", "red")
        return False

    # 5. Checksum verification
    verify_cmd = [
        "rsync",
        "-anc",
        "--itemize-changes",
        "-e",
        ssh_cmd_str,
        f"{resolved_dir}/",
        f"{remote_target}:{remote_incoming_dir}/",
    ]
    v_res = subprocess.run(verify_cmd, capture_output=True, text=True)
    diff = v_res.stdout.strip()
    if diff:
        cprint(f"❌ Checksum verification failed; differences detected:\n{diff}", "red")
        return False

    # 6. Atomic move into datasets/
    cprint(f"🔄 Activating dataset on server ...", "cyan")
    activate_cmd = [
        *ssh_base,
        remote_target,
        f"mkdir -p '{remote_root}/datasets' && mv '{remote_incoming_dir}' '{remote_dataset_dir}'",
    ]
    act_res = subprocess.run(activate_cmd, capture_output=True, text=True)
    if act_res.returncode != 0:
        cprint(f"❌ Failed to activate dataset on server: {act_res.stderr.strip()}", "red")
        return False

    cprint(f"✅ Successfully uploaded: {remote_dataset_dir}", "green", attrs=["bold"])
    cprint(
        f"🌐 Prehensio Viewer: http://{server}:{viewer_port}/prehensio/{name}/episode_0\n",
        "green",
    )
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upload LeRobot dataset to the Prehensio data server.")
    parser.add_argument("dataset", help="Dataset name in lerobot_datasets/ or absolute path")
    parser.add_argument("--server", help="Server IP or hostname")
    parser.add_argument("--key", help="SSH key path")
    parser.add_argument("--replace", action="store_true", default=None, help="Backup & replace existing dataset")
    parser.add_argument("--no-replace", action="store_false", dest="replace", help="Abort if dataset exists")
    args = parser.parse_args()

    cfg = resolve_push_config(True)
    if args.server:
        cfg["server"] = args.server
    if args.key:
        cfg["ssh_key"] = args.key
    if args.replace is not None:
        cfg["replace"] = args.replace

    success = upload_dataset(args.dataset, config=cfg)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
