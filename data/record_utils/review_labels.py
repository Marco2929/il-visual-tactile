#!/usr/bin/env python3
"""
Episode Label Reviewer (matplotlib version)
- Left/Right arrows: navigate between episodes
- Space: toggle success/failed label (renames folder)
- Q/Esc: quit
"""

import os
import re
import glob
import shutil
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")

BASE_DIR = os.path.join("data/episodes/external", FOLDER_NAME)


def find_episodes(base_dir):
    """Find all episode folders recursively."""
    episodes = []
    for root, dirs, files in os.walk(base_dir):
        for d in sorted(dirs):
            if re.match(r"episode_\d+_(success|failed)", d):
                episodes.append(os.path.join(root, d))
    episodes.sort()
    return episodes


def get_first_last_frames(episode_path):
    """Get first and last frame from front_rgb subfolder."""
    rgb_dir = os.path.join(episode_path, "front_rgb")
    if not os.path.exists(rgb_dir):
        for sub in sorted(os.listdir(episode_path)):
            candidate = os.path.join(episode_path, sub)
            if os.path.isdir(candidate):
                pngs = sorted(glob.glob(os.path.join(candidate, "*.png")))
                if pngs:
                    rgb_dir = candidate
                    break

    frames = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    if not frames:
        return None, None
    return frames[0], frames[-1]


def get_label(episode_path):
    name = os.path.basename(episode_path)
    if "success" in name:
        return "success"
    return "failed"


def _merge_directories(src, dst):
    """Recursively move all contents from src into dst without overwriting.
    Returns counts of (moved, skipped) files."""
    moved = 0
    skipped = 0
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)

        if os.path.isdir(src_item):
            if os.path.exists(dst_item):
                m, s = _merge_directories(src_item, dst_item)
                moved += m
                skipped += s
            else:
                shutil.move(src_item, dst_item)
                moved += 1
        else:
            if not os.path.exists(dst_item):
                shutil.move(src_item, dst_item)
                moved += 1
            else:
                skipped += 1
    return moved, skipped


def toggle_label(episode_path):
    """Toggle episode label. Merges into target if it already exists."""
    name = os.path.basename(episode_path)
    parent = os.path.dirname(episode_path)

    if "success" in name:
        new_name = name.replace("success", "failed")
    else:
        new_name = name.replace("failed", "success")

    new_path = os.path.join(parent, new_name)

    if os.path.exists(new_path):
        moved, skipped = _merge_directories(episode_path, new_path)
        shutil.rmtree(episode_path)
        print(f"  ⚠️  Merged into existing '{new_name}' ({moved} moved, {skipped} skipped)")
    else:
        os.rename(episode_path, new_path)

    return new_path


class EpisodeReviewer:
    def __init__(self, episodes):
        self.episodes = episodes
        self.idx = 0
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_display()
        plt.tight_layout()
        plt.show()

    def update_display(self):
        self.ax1.clear()
        self.ax2.clear()

        episode_path = self.episodes[self.idx]
        label = get_label(episode_path)
        first_path, last_path = get_first_last_frames(episode_path)

        if first_path and last_path:
            first_img = mpimg.imread(first_path)
            last_img = mpimg.imread(last_path)
            self.ax1.imshow(first_img)
            self.ax2.imshow(last_img)
        else:
            self.ax1.text(0.5, 0.5, "No frames found", ha='center', va='center')
            self.ax2.text(0.5, 0.5, "No frames found", ha='center', va='center')

        self.ax1.set_title("FIRST FRAME")
        self.ax2.set_title("LAST FRAME")
        self.ax1.axis('off')
        self.ax2.axis('off')

        color = "green" if label == "success" else "red"
        episode_name = os.path.basename(episode_path)
        collection = os.path.basename(os.path.dirname(episode_path))
        self.fig.suptitle(
            f"[{self.idx+1}/{len(self.episodes)}] {collection}/{episode_name}\n"
            f"Label: {label.upper()}    |    Space: toggle  |  ←→: navigate  |  Q: quit",
            fontsize=12, color=color, fontweight='bold'
        )
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.idx = min(len(self.episodes) - 1, self.idx + 1)
            self.update_display()
        elif event.key == 'left':
            self.idx = max(0, self.idx - 1)
            self.update_display()
        elif event.key == ' ':
            new_path = toggle_label(self.episodes[self.idx])
            self.episodes[self.idx] = new_path
            # Deduplicate in case of merge into existing entry
            self.episodes = sorted(set(self.episodes))
            self.idx = min(self.idx, len(self.episodes) - 1)
            print(f"Toggled -> {os.path.basename(new_path)}")
            self.update_display()
        elif event.key in ('q', 'escape'):
            plt.close()


def main():
    episodes = find_episodes(BASE_DIR)

    print(f"Base dir: {BASE_DIR}")

    if not episodes:
        print(f"No episodes found in {BASE_DIR}")
        return

    print(f"Found {len(episodes)} episodes")
    print("Controls:")
    print("  ← →    : navigate episodes")
    print("  Space   : toggle success/failed")
    print("  Q / Esc : quit")

    EpisodeReviewer(episodes)
    print("Done!")


if __name__ == "__main__":
    main()