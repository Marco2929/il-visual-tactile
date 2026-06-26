"""
SenseGlove image manager — pull and inspect Docker images.

Usage:
    uv run image_manager.py --pull
    uv run image_manager.py --pull --tag 1.2.0
    uv run image_manager.py -c
"""

import argparse
import subprocess
import sys

DEFAULT_IMAGE = "zippeth/senseglove_robotics"


def pull(image: str, tag: str) -> None:
    full = f"{image}:{tag}"
    print(f"Pulling {full} ...")
    result = subprocess.run(["docker", "pull", full], check=False)
    if result.returncode != 0:
        print("ERROR: pull failed. Make sure Docker is running and you are logged in:")
        print("  docker login -u zippeth")
        sys.exit(1)
    print(f"Done. {full} is ready.")


def check(image: str, tag: str) -> None:
    full = f"{image}:{tag}"
    result = subprocess.run(
        ["docker", "image", "inspect", full, "--format",
         "ID: {{.Id}}\nCreated: {{.Created}}\nSize: {{.Size}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"{full} — not found locally. Run: uv run image_manager.py --pull")
        sys.exit(1)

    # Parse size into MB
    lines = result.stdout.strip().splitlines()
    info = {}
    for line in lines:
        k, _, v = line.partition(": ")
        info[k] = v.strip()

    size_mb = int(info.get("Size", 0)) / 1_000_000
    print(f"Image : {full}")
    print(f"ID    : {info.get('Id', 'unknown')[:19]}")
    print(f"Created: {info.get('Created', 'unknown')[:19]}")
    print(f"Size  : {size_mb:.0f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manage SenseGlove Docker images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Image name (default: {DEFAULT_IMAGE}).",
    )
    parser.add_argument(
        "--tag",
        default="latest",
        help="Image tag (default: latest).",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pull", "-p",
        action="store_true",
        help="Pull the image from Docker Hub.",
    )
    group.add_argument(
        "-c", "--check",
        action="store_true",
        help="Check if the image is available locally and show info.",
    )

    args = parser.parse_args()

    if args.pull:
        pull(args.image, args.tag)
    elif args.check:
        check(args.image, args.tag)
