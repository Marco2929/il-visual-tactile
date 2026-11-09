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
This script is used to merge the data of multiple episodes into a single zarr file.
Modified for Relative End-Effector Control (Arm) and Relative Joint Control (Hand).
''' 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                       default='/ros2_ws/src/imitation_grasping/episodes',
                       help='Raw data directory')
    parser.add_argument('--save_path', type=str, 
                       default='/ros2_ws/src/imitation_grasping/imitation_grasping/third_party/3D-Diffusion-Policy/data/cube_grasping_rel.zarr',
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

def compute_fk(chain, full_state):
    """
    Input:  [13] (6 UR joints + 6 Hand joints + 1 Success Flag)
            OR [12] (Backward compatibility if flag is missing)
    Output: [14] (3 Pos + 4 Rot + 6 Hand joints + 1 Success Flag)
    """
    # 1. Separate Joints and Success Flag
    has_success_flag = (len(full_state) == 13)
    
    if has_success_flag:
        joints = full_state[:-1]      # First 12 are joints
        success_flag = full_state[-1] # Last 1 is success
    else:
        joints = full_state
        success_flag = 0.0 

    # 2. Split UR and Hand
    ur_joints = joints[:6]
    hand_joints = joints[6:]

    # 3. Map names 
    joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                   'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    
    th = dict(zip(joint_names, ur_joints))

    # 4. Compute FK
    ret = chain.forward_kinematics(th)
    
    # 5. Extract
    pos = ret.pos # [x, y, z]
    rot = ret.rot # [w, x, y, z] (Kinpy outputs w first)
    
    # 6. Reorder Quaternion for DP3 / Scipy
    # DP3 usually expects [x, y, z, w]. Kinpy gives [w, x, y, z].
    rot_xyzw = np.array([rot[1], rot[2], rot[3], rot[0]])

    # 7. Concatenate [Pos(3), Rot(4), Hand(6), Success(1)]
    return np.concatenate([pos, rot_xyzw, hand_joints, [success_flag]])

def compute_relative_actions(states):
    """
    Computes relative actions from a sequence of absolute states.
    State Format:  [Pos(3), Rot(4, xyzw), Hand(6), Flag(1)]
    Action Format: [dPos(3), dRot(3, axis-angle), dHand(6), Flag(1)]
    """
    # Initialize actions array with shape (T, 13)
    # Dimension 13: 3 (d_pos) + 3 (d_rot) + 6 (d_hand) + 1 (flag)
    num_steps = states.shape[0]
    actions = np.zeros((num_steps, 13), dtype=np.float32)
    
    # Process all steps except the last one
    for i in range(num_steps - 1):
        curr_state = states[i]
        next_state = states[i+1]
        
        # 1. Relative Position (Base Frame): Next - Curr
        d_pos = next_state[:3] - curr_state[:3]
        
        # 2. Relative Rotation (Base Frame): R_diff = R_next * R_curr_inv
        # Input quaternions are [x, y, z, w]
        r_curr = R.from_quat(curr_state[3:7])
        r_next = R.from_quat(next_state[3:7])
        
        # Calculate diff such that: r_next = r_diff * r_curr
        # This gives the rotation vector to apply in the Base Frame
        r_diff = r_next * r_curr.inv()
        d_rot = r_diff.as_rotvec() # Returns [rx, ry, rz] in radians
        
        # 3. Relative Hand Joints: Next - Curr
        d_hand = next_state[7:13] - curr_state[7:13]
        
        # 4. Flag (Keep next step's flag)
        flag = next_state[13]
        
        actions[i] = np.concatenate([d_pos, d_rot, d_hand, [flag]])
        
    # Handle last step: Set action to zero (stop) or copy previous
    # Usually standard is zero velocity at end of episode
    actions[-1] = np.zeros(13, dtype=np.float32)
    actions[-1, -1] = states[-1, -1] # Keep success flag
    
    return actions

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

        # Load data
        raw_states = np.load(os.path.join(episode_dir, 'state.npy')).astype(np.float32)
        
        cartesian_states = []
        for s in raw_states:
            cartesian_states.append(compute_fk(chain, s))
        cartesian_states = np.array(cartesian_states, dtype=np.float32)

        # --- PROCESS ACTIONS (Relative Control) ---
        actions = compute_relative_actions(cartesian_states)
        
        # Get all frames
        frames = sorted([f[6:-4] for f in os.listdir(os.path.join(episode_dir, 'img')) 
                       if f.startswith('frame_') and f.endswith('.png')])
        
        for frame in frames:
            # Read and ensure correct data types
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

    # Set compression
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    # Set chunk sizes
    img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])

    # Create datasets with specified data types
    zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, 
                            dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, 
                            dtype='float64', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, 
                            dtype='float64', overwrite=True, compressor=compressor)
    
    # 
    # Note: Action shape is now different from State shape due to 6D rotation representation
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, 
                            dtype='float32', overwrite=True, compressor=compressor)
    
    zarr_data.create_dataset('state', data=state_arrays, chunks=(100, state_arrays.shape[1]), 
                            dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends, chunks=(100,), 
                            dtype='int64', overwrite=True, compressor=compressor)

    # Print shapes and ranges
    cprint(f'Image shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
    cprint(f'Point cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'Depth map shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'Action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'State shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'Episode ends shape: {episode_ends.shape}, range: [{np.min(episode_ends)}, {np.max(episode_ends)}]', 'green')
    cprint(f'Total frames: {total_count}', 'green')
    cprint(f'Saved zarr file to {args.save_path}', 'green')

if __name__ == '__main__':
    main()