#!/usr/bin/env python3
import os
import numpy as np
import cv2
import torch
from pathlib import Path
from termcolor import cprint
import kinpy as kp
import argparse
from PIL import Image
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FOLDER_NAME = os.environ.get("FOLDER_NAME", "box_opening_collection")
URDF_PATH = os.environ.get("URDF_PATH", "/ros2_ws/src/moveit_ur_hand_config/urdf/inspire_strong_hand_ur10e_flange_workcell.urdf")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=f'/ros2_ws/data/episodes/{FOLDER_NAME}')
    parser.add_argument('--repo_id', type=str,
                        default=f'your-hf-user/{FOLDER_NAME}',
                        help='HuggingFace repo id (user/dataset_name)')
    parser.add_argument('--task', type=str,
                        default='Pick up the object',
                        help='Task description for the dataset')
    parser.add_argument('--fps', type=int, default=30,
                        help='Recording FPS')
    parser.add_argument('--push_to_hub', action='store_true',
                        help='Push to HuggingFace Hub after creation')
    parser.add_argument('--local_dir', type=str, default=None,
                        help='Local root dir for dataset storage (default: ~/.cache/huggingface/lerobot)')
    return parser.parse_args()


def setup_kinematics():
    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF not found at {URDF_PATH}")
    with open(URDF_PATH, 'r') as f:
        urdf_data = f.read()
    chain = kp.build_serial_chain_from_urdf(
        urdf_data,
        end_link_name=os.environ.get("ROBOT_TIP_FRAME", "right_inspire_base_link"),
        root_link_name="base_link"
    )
    return chain


def compute_fk(chain, full_state):
    has_success_flag = (len(full_state) == 13)
    if has_success_flag:
        joints = full_state[:-1]
        success_flag = full_state[-1]
    else:
        joints = full_state
        success_flag = 0.0

    ur_joints = joints[:6]
    hand_joints = joints[6:]

    joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                   'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    th = dict(zip(joint_names, ur_joints))
    ret = chain.forward_kinematics(th)

    pos = ret.pos
    rot = ret.rot
    rot_xyzw = np.array([rot[1], rot[2], rot[3], rot[0]])

    return np.concatenate([pos, rot_xyzw, hand_joints])


def main():
    args = parse_args()
    chain = setup_kinematics()

    episodes = sorted([d for d in os.listdir(args.data_dir) if d.startswith('episode_')])
    if not episodes:
        cprint(f"No episodes found in {args.data_dir}", 'red')
        return

    # --- Determine image shape from first frame of first episode ---
    first_ep_dir = os.path.join(args.data_dir, episodes[0])
    first_frame_files = sorted([f for f in os.listdir(os.path.join(first_ep_dir, 'img'))
                                 if f.startswith('frame_') and f.endswith('.png')])
    sample_img = cv2.imread(os.path.join(first_ep_dir, 'img', first_frame_files[0]))
    img_h, img_w, _ = sample_img.shape

    # --- Determine state/action dims ---
    raw_states = np.load(os.path.join(first_ep_dir, 'state.npy')).astype(np.float32)
    sample_cart = compute_fk(chain, raw_states[0])
    state_dim = len(sample_cart)  # 13 (3 pos + 4 rot + 6 hand)

    cprint(f"Image size: {img_w}x{img_h}", 'green')
    cprint(f"State dim: {state_dim}", 'green')
    cprint(f"Found {len(episodes)} episodes", 'green')

    # --- Define features for LeRobot ---
    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": (3, img_h, img_w),  # C, H, W
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["action"],
        },
    }

    default_root = Path.home() / ".cache" / "huggingface" / "lerobot"
    dataset_dir = Path(args.local_dir) if args.local_dir else default_root / args.repo_id

    if dataset_dir.exists():
        cprint(f"Dataset already exists at {dataset_dir}", 'red')
        user_input = input("Do you want to overwrite? (y/n): ")
        if user_input.lower() == 'y':
            shutil.rmtree(dataset_dir)
            cprint(f"Deleted {dataset_dir}", 'red')
        else:
            cprint("Exiting.", 'red')
            return

    # --- Create LeRobot dataset ---
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.local_dir,  # None = default cache dir
    )

    # --- Process each episode ---
    for ep_idx, episode in enumerate(episodes):
        episode_dir = os.path.join(args.data_dir, episode)
        cprint(f'Processing {episode} ({ep_idx + 1}/{len(episodes)})', 'green')

        # Load and compute states
        raw_states = np.load(os.path.join(episode_dir, 'state.npy')).astype(np.float32)
        cartesian_states = np.array([compute_fk(chain, s) for s in raw_states], dtype=np.float32)

        # Compute actions (next state as action)
        actions = np.zeros_like(cartesian_states)
        actions[:-1] = cartesian_states[1:]
        actions[-1] = cartesian_states[-1]

        # Get frame list
        frames = sorted([f[6:-4] for f in os.listdir(os.path.join(episode_dir, 'img'))
                         if f.startswith('frame_') and f.endswith('.png')])

        for i, frame_id in enumerate(frames):
            # Read image (BGR -> RGB) and convert to PIL
            img_bgr = cv2.imread(os.path.join(episode_dir, 'img', f'frame_{frame_id}.png'))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Build frame dict
            frame_data = {
                "observation.images.front": img_pil,
                "observation.state": torch.from_numpy(cartesian_states[i]),
                "action": torch.from_numpy(actions[i]),
                "task": args.task,
            }

            dataset.add_frame(frame_data)

        # End of episode
        dataset.save_episode()
        cprint(f"  Saved episode {ep_idx} ({len(frames)} frames)", 'cyan')

    # --- Finalize (REQUIRED before push) ---
    dataset.finalize()
    cprint(f"Dataset finalized: {args.repo_id}", 'green')

    if args.push_to_hub:
        dataset.push_to_hub()
        cprint(f"Pushed to Hub: {args.repo_id}", 'green')


if __name__ == '__main__':
    main()