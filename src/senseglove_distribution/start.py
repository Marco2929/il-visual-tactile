"""
SenseGlove Docker launcher.

Pulls a SenseGlove product image from Docker Hub and runs it via docker compose.

All images live in a single repo: senseglove/senseglove_robotics
Tag format:
  latest              - both products (default)
  r1-latest           - r1 only
  tesollo-latest      - tesollo only
  r1-1.2.0 / 1.2.0   - versioned variants

Usage:
    python start.py r1 [options]
    python start.py tesollo [options]

Examples:
    uv run start.py r1
    uv run start.py r1 --num-gloves 2 --simulation
    uv run start.py tesollo --hand right --glove-id 0
    uv run start.py tesollo --launch dg5f_robot_hand_mapper_both --hand left --glove-id 1
    uv run start.py r1 --tag r1-1.2.0 --pull
"""

import argparse
import contextlib
import os
import subprocess
import sys

REPO = "zippeth/senseglove_robotics"

PRODUCTS: dict[str, dict] = {
    "r1": {
        "compose": "senseglove_r1/docker-compose.yml",
        "service": "senseglove_r1",
        "launch_package": "r1_bringup",
        "launch_file": "r1.launch.py",
    },
    "tesollo": {
        "compose": "senseglove_tesollo/docker-compose.yml",
        "service": "senseglove_tesollo",
        "launch_package": "dg5f_interaction",
        "launch_file": "dg5f_robot_hand_mapper.launch.py",
        "launch_files": {
            "dg5f_robot_hand_mapper": "dg5f_robot_hand_mapper.launch.py",
            "dg5f_robot_hand_mapper_both": "dg5f_robot_hand_mapper_both.launch.py",
            "dg5f_robot_hand_mapper_gazebo": "dg5f_robot_hand_mapper_gazebo.launch.py",
        },
    },
}


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, **kwargs)


def image_exists_locally(image_with_tag: str) -> bool:
    result = subprocess.run(["docker", "image", "inspect", image_with_tag], capture_output=True)
    return result.returncode == 0

def build_launch_command(product_key: str, product: dict, args: argparse.Namespace) -> str:
    pkg = product["launch_package"]

    if product_key == "tesollo" and hasattr(args, "launch") and args.launch:
        launch_files = product.get("launch_files", {})
        lf = launch_files.get(args.launch)
        if lf is None:
            choices = ", ".join(launch_files.keys())
            print(f"ERROR: unknown tesollo launch '{args.launch}'. Choose from: {choices}")
            sys.exit(1)
    else:
        lf = product["launch_file"]

    cmd = f"ros2 launch {pkg} {lf}"
    extra: list[str] = []

    if product_key == "r1":
        if hasattr(args, "num_gloves") and args.num_gloves is not None:
            extra.append(f"num_gloves:={args.num_gloves}")
        if hasattr(args, "simulation") and args.simulation:
            extra.append("use_simulation:=true")
        if hasattr(args, "no_gui") and args.no_gui:
            extra.append("display_rhm_pinch_gui:=false")

    if product_key == "tesollo":
        if hasattr(args, "hand") and args.hand is not None:
            extra.append(f"hand:={args.hand}")
        if hasattr(args, "glove_id") and args.glove_id is not None:
            extra.append(f"glove_id:={args.glove_id}")
        if hasattr(args, "delto_ip") and args.delto_ip is not None:
            extra.append(f"delto_ip:={args.delto_ip}")
            # Also set the hand-specific IP arg used by the mapper launch
            hand_val = getattr(args, "hand", None)
            if hand_val == "right":
                extra.append(f"delto_ip_rh:={args.delto_ip}")
            elif hand_val == "left":
                extra.append(f"delto_ip_lh:={args.delto_ip}")
        if hasattr(args, "delto_port") and args.delto_port is not None:
            extra.append(f"delto_port:={args.delto_port}")

    cmd = cmd + (" " + " ".join(extra) if extra else "")

    # MoveIt's planning scene monitor only subscribes to /joint_states, but the
    # hand publishes on /dg5f_<hand>/joint_states. ur_tesollo_moveit.launch.py no
    # longer starts tesollo_control (it would contend for Modbus 502 with this
    # container), so without this relay move_group and servo_node warn forever:
    #   "The complete state of the robot is not yet known. Missing rj_dg_1_1..."
    # The monitor merges publishers by joint name, so the UR's 6 joints and the
    # hand's 20 coexist. Needs ros-jazzy-topic-tools (in Dockerfile.cyclone).
    if product_key == "tesollo" and not getattr(args, "no_joint_state_relay", False):
        hand_val = getattr(args, "hand", None) or "right"
        src_topic = f"/dg5f_{hand_val}/joint_states"
        relay = f"ros2 run topic_tools relay {src_topic} /joint_states"
        # Backgrounded; the launch is exec'd so it stays PID 1 and keeps owning
        # shutdown. The relay is reaped with the container.
        cmd = f"{relay} & exec {cmd}"

    return cmd


def allow_display_access() -> None:
    subprocess.run(["xhost", "+local:docker"], capture_output=True)


def run_compose(product: dict, launch_cmd: str, image_with_tag: str, interactive: bool = False) -> int:
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required — install with: pip install pyyaml")
        sys.exit(1)

    compose_file = product["compose"]
    if not os.path.exists(compose_file):
        print(f"ERROR: compose file not found: {compose_file}")
        sys.exit(1)

    with open(compose_file) as f:
        compose = yaml.safe_load(f)

    service_name = product["service"]
    service = compose["services"][service_name]
    service["image"] = image_with_tag
    service["command"] = ["bash"] if interactive else ["bash", "-c", launch_cmd]
    if interactive:
        service["stdin_open"] = True

    runtime_compose = "compose_runtime.yml"
    with open(runtime_compose, "w") as f:
        yaml.safe_dump(compose, f, default_flow_style=False, sort_keys=False)

    allow_display_access()

    print(f"Starting {service_name}")
    print(f"  image  : {image_with_tag}")
    print(f"  command: {launch_cmd}")

    result = subprocess.CompletedProcess([], 0)
    with contextlib.suppress(KeyboardInterrupt):
        result = _run(["docker", "compose", "-f", runtime_compose, "up", "--abort-on-container-exit"])

    _run(["docker", "compose", "-f", runtime_compose, "down", "-t", "1"])

    if os.path.exists(runtime_compose):
        os.remove(runtime_compose)

    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SenseGlove launcher - pulls and runs a SenseGlove product Docker image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "product",
        choices=list(PRODUCTS.keys()) + ["bash"],
        help="Product to launch, or 'bash' to open an interactive shell.",
    )
    parser.add_argument(
        "--tag",
        default="latest",
        metavar="TAG",
        help=(
            "Image tag to pull and run (default: latest). "
            "Use product-prefixed tags for single-product images: r1-latest, tesollo-1.2.0, etc."
        ),
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Print the launch command without starting a container.",
    )

    r1_group = parser.add_argument_group("r1 options")
    r1_group.add_argument(
        "--num-gloves",
        type=int,
        default=1,
        metavar="N",
        help="Number of SenseGlove gloves (default: 1).",
    )
    r1_group.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode (no physical glove needed).",
    )
    r1_group.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable the pinch GUI.",
    )

    tesollo_group = parser.add_argument_group("tesollo options")
    tesollo_group.add_argument(
        "--launch",
        default=None,
        metavar="LAUNCH_NAME",
        help=(
            "Tesollo launch file to use. Choices: "
            "dg5f_robot_hand_mapper (default), "
            "dg5f_robot_hand_mapper_both, "
            "dg5f_robot_hand_mapper_gazebo."
        ),
    )
    tesollo_group.add_argument(
        "--hand",
        default=None,
        metavar="HAND",
        help="Which hand to use (e.g. left, right).",
    )
    tesollo_group.add_argument(
        "--glove-id",
        dest="glove_id",
        default=None,
        metavar="ID",
        help="Glove device ID.",
    )
    tesollo_group.add_argument(
        "--delto-ip",
        dest="delto_ip",
        default=None,
        metavar="IP",
        help="IP address of the Tesollo hand (default inside image: 169.254.186.72).",
    )
    tesollo_group.add_argument(
        "--delto-port",
        dest="delto_port",
        default=None,
        metavar="PORT",
        help="Modbus TCP port of the Tesollo hand (default: 502).",
    )
    tesollo_group.add_argument(
        "--no-joint-state-relay",
        dest="no_joint_state_relay",
        action="store_true",
        help=(
            "Do not relay /dg5f_<hand>/joint_states onto /joint_states. "
            "The relay is what lets MoveIt see the rj_dg_* joints; disable it "
            "only if something else already publishes them (e.g. you re-enabled "
            "tesollo_control in ur_tesollo_moveit.launch.py)."
        ),
    )

    args = parser.parse_args()
    image_with_tag = f"{REPO}:{args.tag}"

    if not image_exists_locally(image_with_tag):
        print(f"{image_with_tag} not found locally. Run: uv run image_manager.py --pull")
        sys.exit(1)

    if args.product == "bash":
        allow_display_access()
        ret = _run([
            "docker", "run", "--rm", "-it",
            "--network", "host",
            "--privileged",
            "-e", f"DISPLAY={os.environ.get('DISPLAY', '')}",
            "-v", "/dev:/dev",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
            image_with_tag,
            "bash",
        ]).returncode
        sys.exit(ret)

    product = PRODUCTS[args.product]

    launch_cmd = build_launch_command(args.product, product, args)

    if args.no_run:
        print(f"Image  : {image_with_tag}")
        print(f"Command: {launch_cmd}")
        sys.exit(0)

    ret = run_compose(product, launch_cmd, image_with_tag)
    sys.exit(ret)
