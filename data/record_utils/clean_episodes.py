#!/usr/bin/env python3
"""
Remove all files/folders from episode directories except realsense_bag/ and recording.svo2.

Usage:
    # Dry run (default) — only prints what would be deleted
    python clean_episodes.py

    # Actually delete
    python clean_episodes.py --execute

Environment:
    FOLDER_NAME  — dataset folder name (default: test_collection)
    WORKDIR      — workspace root (default: /root/ros2_ws/data/)
"""

import os
import sys
import shutil
import argparse

FOLDER_NAME = os.environ.get("FOLDER_NAME", "foil_lifting_collection_60")
WORKDIR = os.environ.get("WORKDIR", "/ros2_ws/data/")

KEEP = {"realsense_bag", "recording.svo2"}


def collect_deletions(root_dir: str) -> list[str]:
    """Walk episode directories and collect paths to delete."""
    deletions: list[str] = []

    if not os.path.isdir(root_dir):
        print(f"ERROR: root directory does not exist: {root_dir}", file=sys.stderr)
        sys.exit(1)

    for entry in sorted(os.listdir(root_dir)):
        episode_path = os.path.join(root_dir, entry)

        # Only process episode_* directories
        if not os.path.isdir(episode_path) or not entry.startswith("episode_"):
            continue

        for item in sorted(os.listdir(episode_path)):
            if item in KEEP:
                continue
            deletions.append(os.path.join(episode_path, item))

    return deletions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform deletions. Without this flag, only a dry-run is shown.",
    )
    args = parser.parse_args()

    root_dir = os.path.join(WORKDIR, f"episodes/{FOLDER_NAME}")
    print(f"Dataset root : {root_dir}")
    print(f"Keeping      : {KEEP}")
    print(f"Mode         : {'🔴 EXECUTE' if args.execute else '🟡 DRY RUN'}")
    print("-" * 60)

    deletions = collect_deletions(root_dir)

    if not deletions:
        print("Nothing to delete — all episodes are already clean.")
        return

    total_size = 0
    for path in deletions:
        if os.path.isdir(path):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fnames in os.walk(path)
                for f in fnames
            )
        else:
            size = os.path.getsize(path)
        total_size += size
        tag = "DIR " if os.path.isdir(path) else "FILE"
        print(f"  [{tag}] {path}  ({size / 1024 / 1024:.2f} MB)")

    print("-" * 60)
    print(f"Total items to delete : {len(deletions)}")
    print(f"Total size to free    : {total_size / 1024 / 1024:.1f} MB")
    print("-" * 60)

    if not args.execute:
        print("\n⚠️  Dry run complete. Re-run with --execute to actually delete.")
        return

    # Final confirmation gate
    answer = input("\n⚠️  Type 'yes' to confirm deletion: ").strip().lower()
    if answer != "yes":
        print("Aborted.")
        return

    for path in deletions:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print(f"  Deleted: {path}")

    print(f"\n✅ Done. Removed {len(deletions)} items.")


if __name__ == "__main__":
    main()