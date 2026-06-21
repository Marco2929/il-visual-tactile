#!/usr/bin/env python3
import os
import zarr
import numpy as np
import cv2
from termcolor import cprint
import kinpy as kp
import argparse

'''
This script merges multiple episodes into a single zarr file.
Stores raw hand data (joints + F/T) for offline transformation during training.
'''

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")
URDF_PATH = os.environ.get("URDF_PATH", "/ros2_ws/src/moveit_ur_hand_config/urdf/ur10e_workcell.urdf")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=f'/ros2_ws/data/episodes/{FOLDER_NAME}',
                        help='Raw data directory')
    parser.add_argument('--save_path', type=str,
                        default=f'/ros2_ws/data/zarr_datasets/{FOLDER_NAME}.zarr',
                        help='Path to save zarr file')
    args = parser.parse_args()

    args.data_dir = os.path.expanduser(args.data_dir)
    args.save_path = os.path.expanduser(args.save_path)
    return args


def setup_kinematics():
    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF not found at {URDF_PATH}")

    with open(URDF_PATH, 'r') as f:
        urdf_data = f.read()

    chain = kp.build_serial_chain_from_urdf(
        urdf_data,
        end_link_name="tool0",  # UR10e flange (no Inspire)
        root_link_name="base_link"
    )
    return chain


def compute_fk(chain, ur_joints):
    """
    Input:  (6,) UR joint angles in radians
    Output: (7,) [x, y, z, qx, qy, qz, qw] — TCP pose
    """
    joint_names = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]
    th = dict(zip(joint_names, ur_joints))
    ret = chain.forward_kinematics(th)

    pos = ret.pos                                           # (3,)
    rot = ret.rot                                           # [w, x, y, z]
    rot_xyzw = np.array([rot[1], rot[2], rot[3], rot[0]])  # → [x, y, z, w]

    return np.concatenate([pos, rot_xyzw])  # (7,)


def main():
    args = parse_args()
    chain = setup_kinematics()

    if os.path.exists(args.save_path):
        cprint(f'Data already exists at {args.save_path}', 'red')
        user_input = input("Do you want to overwrite? (y/n): ")
        if user_input.lower() == 'y':
            cprint(f'Overwriting {args.save_path}', 'red')
            os.system(f'rm -rf {args.save_path}')
        else:
            cprint('Exiting program', 'red')
            exit()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Initialize Zarr
    zarr_root = zarr.group(args.save_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    total_count = 0
    episode_ends = []
    datasets_initialized = False

    episodes = sorted([d for d in os.listdir(args.data_dir) if d.startswith('episode_')])

    for episode in episodes:
        episode_dir = os.path.join(args.data_dir, episode)
        cprint(f'Processing {episode}', 'green')

        # --- Load raw data ---
        arm_states = np.load(os.path.join(episode_dir, 'arm_state.npy')).astype(np.float32)      # (N, 6)
        hand_joints = np.load(os.path.join(episode_dir, 'hand_joints.npy')).astype(np.float32)   # (N, 20)
        hand_ft = np.load(os.path.join(episode_dir, 'hand_ft.npy')).astype(np.float32)           # (N, 30)

        # --- Compute FK for arm: joint space → Cartesian ---
        cartesian_states = np.array(
            [compute_fk(chain, s) for s in arm_states], dtype=np.float32
        )  # (N, 7)

        # --- Compute actions (next-state as action target) ---
        # Action = [next_tcp_pose(7), next_hand_joints(20)]
        # Hand action is raw degrees (policy outputs what Tesollo expects)
        arm_actions = np.zeros_like(cartesian_states)       # (N, 7)
        arm_actions[:-1] = cartesian_states[1:]
        arm_actions[-1] = cartesian_states[-1]

        hand_actions = np.zeros_like(hand_joints)           # (N, 20)
        hand_actions[:-1] = hand_joints[1:]
        hand_actions[-1] = hand_joints[-1]

        actions = np.concatenate([arm_actions, hand_actions], axis=1)  # (N, 27)

        # --- State for policy observation ---
        # state = [tcp_pose(7), hand_joints_deg(20)]
        states = np.concatenate([cartesian_states, hand_joints], axis=1)  # (N, 27)

        # --- Load images, depth, point clouds ---
        ep_imgs, ep_depths, ep_pcs = [], [], []

        frames = sorted([f[6:-4] for f in os.listdir(os.path.join(episode_dir, 'img'))
                         if f.startswith('frame_') and f.endswith('.png')])

        for frame in frames:
            img = cv2.imread(os.path.join(episode_dir, 'img', f'frame_{frame}.png'))
            depth = np.load(os.path.join(episode_dir, 'depth', f'frame_{frame}.npy')).astype(np.float32)
            point_cloud = np.load(os.path.join(episode_dir, 'point_cloud', f'frame_{frame}.npy')).astype(np.float32)

            ep_imgs.append(img)
            ep_depths.append(depth)
            ep_pcs.append(point_cloud)

        ep_imgs = np.stack(ep_imgs, axis=0)
        ep_depths = np.stack(ep_depths, axis=0)
        ep_pcs = np.stack(ep_pcs, axis=0)

        n_frames = len(frames)
        assert n_frames == len(arm_states), \
            f"Frame count mismatch: {n_frames} images vs {len(arm_states)} states"

        total_count += n_frames
        episode_ends.append(total_count)

        # --- Initialize Zarr datasets on first episode ---
        if not datasets_initialized:
            zarr_data.create_dataset('img', shape=(0, *ep_imgs.shape[1:]),
                                     chunks=(100, *ep_imgs.shape[1:]), dtype='uint8', compressor=compressor)
            zarr_data.create_dataset('point_cloud', shape=(0, *ep_pcs.shape[1:]),
                                     chunks=(100, *ep_pcs.shape[1:]), dtype='float32', compressor=compressor)
            zarr_data.create_dataset('depth', shape=(0, *ep_depths.shape[1:]),
                                     chunks=(100, *ep_depths.shape[1:]), dtype='float32', compressor=compressor)
            zarr_data.create_dataset('state', shape=(0, states.shape[1]),
                                     chunks=(100, states.shape[1]), dtype='float32', compressor=compressor)
            zarr_data.create_dataset('action', shape=(0, actions.shape[1]),
                                     chunks=(100, actions.shape[1]), dtype='float32', compressor=compressor)
            # Raw hand data for offline palm_transform during training
            zarr_data.create_dataset('hand_joints', shape=(0, 20),
                                     chunks=(100, 20), dtype='float32', compressor=compressor)
            zarr_data.create_dataset('hand_ft', shape=(0, 30),
                                     chunks=(100, 30), dtype='float32', compressor=compressor)
            datasets_initialized = True

        # --- Append to Zarr ---
        zarr_data['img'].append(ep_imgs)
        zarr_data['point_cloud'].append(ep_pcs)
        zarr_data['depth'].append(ep_depths)
        zarr_data['state'].append(states)
        zarr_data['action'].append(actions)
        zarr_data['hand_joints'].append(hand_joints)
        zarr_data['hand_ft'].append(hand_ft)

    # --- Save metadata ---
    episode_ends_arr = np.array(episode_ends, dtype=np.int64)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arr, chunks=(100,),
                             dtype='int64', overwrite=True, compressor=compressor)

    # --- Summary ---
    cprint(f"\n{'='*50}", 'cyan')
    cprint(f"Final Image shape:       {zarr_data['img'].shape}", 'green')
    cprint(f"Final Point cloud shape: {zarr_data['point_cloud'].shape}", 'green')
    cprint(f"Final Depth map shape:   {zarr_data['depth'].shape}", 'green')
    cprint(f"Final State shape:       {zarr_data['state'].shape} (tcp7 + hand20)", 'green')
    cprint(f"Final Action shape:      {zarr_data['action'].shape} (tcp7 + hand20)", 'green')
    cprint(f"Final Hand joints shape: {zarr_data['hand_joints'].shape} (raw degrees)", 'green')
    cprint(f"Final Hand F/T shape:    {zarr_data['hand_ft'].shape} (raw tip-local)", 'green')
    cprint(f"Episode ends:            {episode_ends_arr}", 'green')
    cprint(f"Total frames:            {total_count}", 'green')
    cprint(f"Saved to: {args.save_path}", 'green', attrs=['bold'])


if __name__ == '__main__':
    main()