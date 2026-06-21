import os
import shutil
from pathlib import Path


def cleanup_episodes(root_dir):
    # Keep these specific files or folder names
    keep_extensions = {'.svo2', '.bag'}
    keep_folders = {'realsense_bag'}

    root_path = Path(root_dir)

    for episode_folder in root_path.iterdir():
        if episode_folder.is_dir():
            print(f"Processing: {episode_folder.name}")

            for item in episode_folder.iterdir():
                # Check if we should keep this item
                if item.is_file() and item.suffix in keep_extensions:
                    continue
                if item.is_dir() and item.name in keep_folders:
                    continue

                # Delete the item
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    print(f"Error deleting {item}: {e}")


if __name__ == "__main__":
    # Ensure this points to your parent directory
    target_dir = "/home/mm/dev/il-visual-tactile/data/episodes/foil_lifting_collection"
    cleanup_episodes(target_dir)