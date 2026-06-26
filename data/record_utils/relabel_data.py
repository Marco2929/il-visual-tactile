# # Umbenennen (success -> failed), Standard
# python relabel_episodes.py 1 30 55 59 66

# # Löschen statt umbenennen
# python relabel_episodes.py 53, 54, 59, 66 --action delete

# # Anderes Datenverzeichnis
# python relabel_episodes.py 1 30 55 59 66 --data-dir /path/to/episodes

import os
import sys
import argparse
import shutil
from pathlib import Path
from natsort import natsorted


FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")

DATA_DIR = os.path.join("data/episodes", FOLDER_NAME)


def find_success_episodes(data_dir: Path) -> list[Path]:
    """Find all success episode folders, sorted naturally (matching lerobot ordering)."""
    return natsorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and "success" in d.name.lower()
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Relabel or delete success episodes by their LeRobot index."
    )
    parser.add_argument(
        "indices",
        type=int,
        nargs="+",
        help="LeRobot indices (0-based, only counting success episodes) to process."
    )
    parser.add_argument(
        "--action",
        choices=["rename", "delete"],
        default="rename",
        help="'rename' marks them as failed, 'delete' removes them permanently. (default: rename)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Path to episodes directory. (default: {DATA_DIR})"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    indices_to_remove = sorted(set(args.indices))
    action = args.action

    if not data_dir.exists():
        print(f"ERROR: Directory '{data_dir}' does not exist.")
        sys.exit(1)

    # Find all success episodes in order
    success_episodes = find_success_episodes(data_dir)

    print(f"Found {len(success_episodes)} success episodes total.\n")

    # Validate indices
    invalid_indices = [i for i in indices_to_remove if i < 0 or i >= len(success_episodes)]
    if invalid_indices:
        print(f"ERROR: Indices {invalid_indices} are out of range (valid: 0–{len(success_episodes) - 1})")
        sys.exit(1)

    # Map indices to episode paths
    episodes_to_process = [(idx, success_episodes[idx]) for idx in indices_to_remove]

    # === PREVIEW ===
    action_label = "RENAMED to failed" if action == "rename" else "PERMANENTLY DELETED"
    print("=" * 70)
    print(f"The following {len(episodes_to_process)} episodes will be {action_label}:")
    print("=" * 70)
    print(f"{'LeRobot Idx':<13} {'Path'}")
    print("-" * 70)
    for idx, path in episodes_to_process:
        if action == "rename":
            new_name = path.name.replace("success", "failed")
            print(f"{idx:<13} {path.relative_to(data_dir)}  ->  {new_name}")
        else:
            print(f"{idx:<13} {path.relative_to(data_dir)}")
    print("=" * 70)

    # === CONFIRMATION ===
    confirmation = input("\nProceed? (yes/no): ").strip().lower()

    if confirmation != "yes":
        print("Aborted. No changes were made.")
        sys.exit(0)

    # === EXECUTE ===
    for idx, path in episodes_to_process:
        if action == "rename":
            new_name = path.parent / path.name.replace("success", "failed")
            path.rename(new_name)
            print(f"  Renamed: {path.name} -> {new_name.name}")
        elif action == "delete":
            shutil.rmtree(path)
            print(f"  Deleted: {path}")

    print(f"\nDone! {len(episodes_to_process)} episodes processed.")

    remaining = find_success_episodes(data_dir)
    print(f"Remaining success episodes: {len(remaining)}")


if __name__ == "__main__":
    main()