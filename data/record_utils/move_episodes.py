#!/usr/bin/env python3
import os
import re
import shutil

BASE_PATH = "/root/ros2_ws/data/episodes/external"

SOURCE = os.path.join(BASE_PATH, "test_collection")
DESTINATION = os.path.join(BASE_PATH, "foil_lifting_collection")


def parse_episode(dirname):
    """Parse 'episode_003_success' → (3, 'success')"""
    match = re.match(r'episode_(\d+)_(.*)', dirname)
    if match:
        return int(match.group(1)), match.group(2)
    # Handle case like 'episode_002' with no suffix
    match = re.match(r'episode_(\d+)$', dirname)
    if match:
        return int(match.group(1)), ""
    return None, None


def make_episode_name(number, suffix, pad=3):
    """Build 'episode_007_success' from (7, 'success')"""
    if suffix:
        return f"episode_{str(number).zfill(pad)}_{suffix}"
    return f"episode_{str(number).zfill(pad)}"


def get_existing_numbers(directory):
    """Get set of episode numbers already in directory."""
    numbers = set()
    for name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, name)):
            num, _ = parse_episode(name)
            if num is not None:
                numbers.add(num)
    return numbers


def find_gaps(existing_numbers):
    """Find missing numbers in sequence."""
    if not existing_numbers:
        return []
    return [i for i in range(max(existing_numbers)) if i not in existing_numbers]


def move_episodes(source_dir, destination_dir, dry_run=True):
    """Move episode folders from source to destination, avoiding overwrites."""
    if not os.path.exists(source_dir):
        print(f"❌ Source not found: {source_dir}")
        return
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    dest_numbers = get_existing_numbers(destination_dir)
    gaps = find_gaps(dest_numbers)
    next_append = max(dest_numbers) + 1 if dest_numbers else 0

    source_dirs = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d)) and d.startswith("episode_")
    ])

    print(f"📂 Source:      {source_dir} ({len(source_dirs)} episodes)")
    print(f"📂 Destination: {destination_dir} ({len(dest_numbers)} existing)")
    print(f"🕳️  Gaps: {gaps}")
    print(f"➡️  Next append: {next_append}")
    print("-" * 60)

    moved = 0
    renamed = 0

    for dirname in source_dirs:
        src_path = os.path.join(source_dir, dirname)
        ep_num, suffix = parse_episode(dirname)

        if ep_num is None:
            print(f"⚠️  SKIPPED (can't parse): {dirname}")
            continue

        # Check if this number already exists in destination
        if ep_num not in dest_numbers:
            # No conflict — move as-is
            dest_path = os.path.join(destination_dir, dirname)
            if dry_run:
                print(f"🔍 [DRY RUN] Move: {dirname}")
            else:
                shutil.move(src_path, dest_path)
                print(f"✅ Moved: {dirname}")
            dest_numbers.add(ep_num)
            moved += 1
        else:
            # Conflict — assign new number (fill gap or append)
            if gaps:
                new_num = gaps.pop(0)
            else:
                new_num = next_append
                next_append += 1

            # Safety: make sure new_num isn't taken
            while new_num in dest_numbers:
                new_num = next_append
                next_append += 1

            new_name = make_episode_name(new_num, suffix)
            dest_path = os.path.join(destination_dir, new_name)

            if dry_run:
                print(f"🔍 [DRY RUN] Rename & move: {dirname} → {new_name}")
            else:
                shutil.move(src_path, dest_path)
                print(f"🔄 Renamed & moved: {dirname} → {new_name}")

            dest_numbers.add(new_num)
            renamed += 1

    print("\n--- Summary ---")
    print(f"Moved (as-is): {moved}")
    print(f"Renamed:       {renamed}")
    print(f"Total:         {moved + renamed}")


if __name__ == "__main__":
    # Preview first
    move_episodes(SOURCE, DESTINATION, dry_run=True)

    # Uncomment to execute:
    # move_episodes(SOURCE, DESTINATION, dry_run=False)