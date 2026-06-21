#!/usr/bin/env python3
import os
import zarr
import numpy as np
import cv2
from termcolor import cprint
import kinpy as kp
import argparse
'''
This script is used to merge the data of multiple episodes into a single zarr file.
''' 

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")
URDF_PATH = os.environ.get("URDF_PATH", "/ros2_ws/src/moveit_ur_hand_config/urdf/inspire_strong_hand_ur10e_flange_workcell.urdf")  

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
        end_link_name=os.environ.get("ROBOT_TIP_FRAME", "right_inspire_base_link")   ,
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
        success_flag = 0.0 # Default to 0 if missing

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
    
    # 6. Reorder Quaternion for DP3
    # DP3 usually expects [x, y, z, w]. Kinpy gives [w, x, y, z].
    rot_xyzw = np.array([rot[1], rot[2], rot[3], rot[0]])

    # 7. Concatenate [Pos(3), Rot(4), Hand(6), Success(1)]
    # Result dim: 3 + 4 + 6 + 1 = 14
    return np.concatenate([pos, rot_xyzw, hand_joints])

def main():
    args = parse_args()

    chain = setup_kinematics()
    
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

    # Initialize Zarr groups
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

        # Load states
        raw_states = np.load(os.path.join(episode_dir, 'state.npy')).astype(np.float32)
        
        cartesian_states = []
        for s in raw_states:
            cartesian_states.append(compute_fk(chain, s))
        cartesian_states = np.array(cartesian_states, dtype=np.float32)

        # Process actions
        actions = np.zeros_like(cartesian_states)
        actions[:-1] = cartesian_states[1:]
        actions[-1] = cartesian_states[-1]

        # Episode-level storage
        ep_imgs, ep_depths, ep_pcs = [], [], []

        frames = sorted([f[6:-4] for f in os.listdir(os.path.join(episode_dir, 'img')) 
                       if f.startswith('frame_') and f.endswith('.png')])
        
        for frame in frames:
            img = cv2.imread(os.path.join(episode_dir, 'img', f'frame_{frame}.png'))
            # Downgraded from float64 to float32 to save 50% RAM
            depth = np.load(os.path.join(episode_dir, 'depth', f'frame_{frame}.npy')).astype(np.float32)
            point_cloud = np.load(os.path.join(episode_dir, 'point_cloud', f'frame_{frame}.npy')).astype(np.float32)
            
            ep_imgs.append(img)
            ep_depths.append(depth)
            ep_pcs.append(point_cloud)
            
        # Stack episode data
        ep_imgs = np.stack(ep_imgs, axis=0)
        ep_depths = np.stack(ep_depths, axis=0)
        ep_pcs = np.stack(ep_pcs, axis=0)
        
        total_count += len(frames)
        episode_ends.append(total_count)

        # Initialize Zarr datasets dynamically on the first episode
        if not datasets_initialized:
            zarr_data.create_dataset('img', shape=(0, *ep_imgs.shape[1:]), 
                                    chunks=(100, *ep_imgs.shape[1:]), dtype='uint8', compressor=compressor)
            zarr_data.create_dataset('point_cloud', shape=(0, *ep_pcs.shape[1:]), 
                                    chunks=(100, *ep_pcs.shape[1:]), dtype='float32', compressor=compressor)
            zarr_data.create_dataset('depth', shape=(0, *ep_depths.shape[1:]), 
                                    chunks=(100, *ep_depths.shape[1:]), dtype='float32', compressor=compressor)
            zarr_data.create_dataset('joint_state', shape=(0, raw_states.shape[1]), 
                                    chunks=(100, raw_states.shape[1]), dtype='float32', compressor=compressor)
            zarr_data.create_dataset('action', shape=(0, actions.shape[1]), 
                                    chunks=(100, actions.shape[1]), dtype='float32', compressor=compressor)
            zarr_data.create_dataset('state', shape=(0, cartesian_states.shape[1]), 
                                    chunks=(100, cartesian_states.shape[1]), dtype='float32', compressor=compressor)
            datasets_initialized = True

        # Append directly to disk
        zarr_data['img'].append(ep_imgs)
        zarr_data['point_cloud'].append(ep_pcs)
        zarr_data['depth'].append(ep_depths)
        zarr_data['joint_state'].append(raw_states)
        zarr_data['action'].append(actions)
        zarr_data['state'].append(cartesian_states)

    # Save meta data at the end
    episode_ends_arr = np.array(episode_ends, dtype=np.int64)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arr, chunks=(100,), 
                            dtype='int64', overwrite=True, compressor=compressor)

    # Print final validation stats mapping directly to the Zarr disk arrays
    cprint(f"Final Image shape: {zarr_data['img'].shape}", 'green')
    cprint(f"Final Point cloud shape: {zarr_data['point_cloud'].shape}", 'green')
    cprint(f"Final Depth map shape: {zarr_data['depth'].shape}", 'green')
    cprint(f"Final Action shape: {zarr_data['action'].shape}", 'green')
    cprint(f"Final State shape: {zarr_data['state'].shape}", 'green')
    cprint(f"Final Episode ends shape: {episode_ends_arr.shape}", 'green')
    cprint(f'Total frames: {total_count}', 'green')
    cprint(f'Successfully saved chunked zarr file to {args.save_path}', 'green')

if __name__ == '__main__':
    main() 