#!/usr/bin/env python3
import os
import zarr
import numpy as np
import cv2
from termcolor import cprint
import kinpy as kp
import argparse
from scipy.spatial.transform import Rotation as R

'''
This script merges multiple episodes into a single zarr file.
- State: Absolute End-Effector Pose (Pos + 6D Rot) + Hand Joints
- Action: Relative End-Effector Control (dPos + 6D Relative Rot) + Relative Hand Joints
- Rotation Format: Continuous 6D representation (first two columns of rotation matrix)
''' 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                       default='/ros2_ws/src/imitation_grasping/episodes',
                       help='Raw data directory')
    parser.add_argument('--save_path', type=str, 
                       default='/ros2_ws/src/imitation_grasping/imitation_grasping/third_party/3D-Diffusion-Policy/data/cube_grasping_rel_rot.zarr',
                       help='Path to save zarr file')
    parser.add_argument('--urdf_path', type=str, 
                    default='/ros2_ws/src/moveit_ur10e_inspire_config/urdf/combined.urdf',
                    help='Path to save urdf file')
    args = parser.parse_args()
    
    args.data_dir = os.path.expanduser(args.data_dir)
    args.save_path = os.path.expanduser(args.save_path)
    return args

def setup_kinematics(urdf_path):
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found at {urdf_path}")
    
    with open(urdf_path, 'r') as f:
        urdf_data = f.read()
    
    chain = kp.build_serial_chain_from_urdf(
        urdf_data, 
        end_link_name="inspire_base_link",
        root_link_name="base_link" 
    )
    
    return chain

def matrix_to_6d(matrix):
    """
    Converts a 3x3 rotation matrix to 6D representation.
    Takes the first two columns (r1, r2).
    Input: [3, 3]
    Output: [6]
    """
    return matrix[:, :2].flatten(order='F') 

def compute_kinematics_data(chain, full_state):
    """
    Computes FK and returns the raw components needed for State and Action calculation.
    Returns:
        pos: [3]
        rot_mat: [3, 3]
        hand_joints: [6]
        success_flag: float
    """
    # 1. Parse Input
    has_success_flag = (len(full_state) == 13)
    if has_success_flag:
        joints = full_state[:-1]
        success_flag = full_state[-1]
    else:
        joints = full_state
        success_flag = 0.0

    ur_joints = joints[:6]
    hand_joints = joints[6:]

    # 2. Map names for Kinpy
    joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                   'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    th = dict(zip(joint_names, ur_joints))

    # 3. Compute FK
    ret = chain.forward_kinematics(th)
    pos = ret.pos # [x, y, z]
    
    # 4. Convert Rotation
    # Kinpy output is quaternion [w, x, y, z]
    # Scipy expects [x, y, z, w]
    k_quat = ret.rot
    s_quat = [k_quat[1], k_quat[2], k_quat[3], k_quat[0]]
    rot_mat = R.from_quat(s_quat).as_matrix()

    return pos, rot_mat, hand_joints, success_flag

def process_episode_kinematics(chain, raw_states):
    """
    Process an entire episode to generate States and Actions in one pass.
    """
    num_steps = len(raw_states)
    
    # Buffers
    positions = []
    rot_matrices = []
    hands = []
    flags = []

    # 1. Compute FK for all steps
    for s in raw_states:
        p, r, h, f = compute_kinematics_data(chain, s)
        positions.append(p)
        rot_matrices.append(r)
        hands.append(h)
        flags.append(f)
    
    positions = np.array(positions)      # (T, 3)
    rot_matrices = np.array(rot_matrices) # (T, 3, 3)
    hands = np.array(hands)              # (T, 6)
    flags = np.array(flags)              # (T,)

    # --- BUILD STATES (Absolute) ---
    # State: [Pos(3), Rot6D(6), Hand(6), Flag(1)] = 16 dims
    states = []
    for i in range(num_steps):
        rot_6d = matrix_to_6d(rot_matrices[i])
        s_vec = np.concatenate([positions[i], rot_6d, hands[i], [flags[i]]])
        states.append(s_vec)
    states = np.array(states, dtype=np.float32)

    # --- BUILD ACTIONS (Relative) ---
    # Action: [dPos(3), dRot6D(6), dHand(6), Flag(1)] = 16 dims
    # dRot6D is the 6D representation of the Relative Rotation Matrix
    actions = np.zeros_like(states)
    
    for i in range(num_steps - 1):
        # Position Diff
        d_pos = positions[i+1] - positions[i]
        
        # Hand Diff
        d_hand = hands[i+1] - hands[i]
        
        # Rotation Diff: R_next = R_diff * R_curr  ->  R_diff = R_next * R_curr^T
        # This gives the relative rotation in the Base Frame
        r_diff = rot_matrices[i+1] @ rot_matrices[i].T
        d_rot_6d = matrix_to_6d(r_diff)
        
        # Flag (Use next step's flag)
        flag = flags[i+1]
        
        actions[i] = np.concatenate([d_pos, d_rot_6d, d_hand, [flag]])

    # Handle last action (Zero motion)
    # Identity matrix 6D rep is [1,0,0, 0,1,0]
    identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    actions[-1] = np.concatenate([np.zeros(3), identity_6d, np.zeros(6), [flags[-1]]])

    return states, actions

def main():
    args = parse_args()

    chain = setup_kinematics(args.urdf_path)
    
    if os.path.exists(args.save_path):
        cprint(f'Data already exists at {args.save_path}', 'red')
        cprint("Please delete the existing directory first if you want to overwrite.", "red")
        user_input = input("Do you want to overwrite? (y/n): ")
        if user_input.lower() == 'y':
            cprint(f'Overwriting {args.save_path}', 'red')
            os.system(f'rm -rf {args.save_path}')
        else:
            cprint('Exiting program', 'red')
            exit()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Storage arrays
    total_count = 0
    img_arrays = []
    depth_arrays = []
    point_cloud_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends = []

    # Process each episode
    episodes = sorted([d for d in os.listdir(args.data_dir) if d.startswith('episode_')])
    for episode in episodes:
        episode_dir = os.path.join(args.data_dir, episode)
        cprint(f'Processing {episode}', 'green')

        # Load raw joint states
        raw_states = np.load(os.path.join(episode_dir, 'state.npy')).astype(np.float32)
        
        # Compute States and Actions
        # States: Absolute 6D
        # Actions: Relative 6D
        cartesian_states, actions = process_episode_kinematics(chain, raw_states)

        # Load Frames
        frames = sorted([f[6:-4] for f in os.listdir(os.path.join(episode_dir, 'img')) 
                       if f.startswith('frame_') and f.endswith('.png')])
        
        for frame in frames:
            img = cv2.imread(os.path.join(episode_dir, 'img', f'frame_{frame}.png'))
            depth = np.load(os.path.join(episode_dir, 'depth', f'frame_{frame}.npy')).astype(np.float64)
            point_cloud = np.load(os.path.join(episode_dir, 'point_cloud', f'frame_{frame}.npy')).astype(np.float64)
            
            img_arrays.append(img)
            depth_arrays.append(depth)
            point_cloud_arrays.append(point_cloud)
            total_count += 1
            
        state_arrays.append(cartesian_states)
        action_arrays.append(actions)
        episode_ends.append(total_count)

    # Convert to numpy arrays
    img_arrays = np.stack(img_arrays, axis=0)
    depth_arrays = np.stack(depth_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    state_arrays = np.concatenate(state_arrays, axis=0)
    action_arrays = np.concatenate(action_arrays, axis=0)
    episode_ends = np.array(episode_ends, dtype=np.int64)

    # Create zarr file
    zarr_root = zarr.group(args.save_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    # Action/State chunks are (100, 16)
    data_chunk_size = (100, state_arrays.shape[1]) 

    zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, 
                            dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, 
                            dtype='float64', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, 
                            dtype='float64', overwrite=True, compressor=compressor)
    
    # 
    zarr_data.create_dataset('action', data=action_arrays, chunks=data_chunk_size, 
                            dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=data_chunk_size, 
                            dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends, chunks=(100,), 
                            dtype='int64', overwrite=True, compressor=compressor)

    cprint(f'Image shape: {img_arrays.shape}', 'green')
    cprint(f'State shape: {state_arrays.shape} (Pos=3, Rot6D=6, Hand=6, Flag=1)', 'green')
    cprint(f'Action shape: {action_arrays.shape} (dPos=3, dRot6D=6, dHand=6, Flag=1)', 'green')
    cprint(f'Total frames: {total_count}', 'green')
    cprint(f'Saved zarr file to {args.save_path}', 'green')

if __name__ == '__main__':
    main()