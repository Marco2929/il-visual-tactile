#!/usr/bin/env python3
"""Compute joint states from a target EE pose using inverse kinematics (Kinpy + scipy)."""

import sys
import numpy as np
import kinpy as kp
from scipy.optimize import minimize

URDF_PATH = "/ros2_ws/src/moveit_ur_hand_config/urdf/inspire_strong_hand_ur10e_flange_workcell.urdf"
END_LINK = "right_inspire_base_link"

UR_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]


def _build_chain():
    with open(URDF_PATH, 'r') as f:
        chain = kp.build_serial_chain_from_urdf(
            f.read(), end_link_name=END_LINK, root_link_name='base_link')
    return chain


def _fk(chain, joints):
    """FK using kinpy. Returns (pos[3], quat_xyzw[4])."""
    th = dict(zip(UR_JOINT_NAMES, joints))
    ret = chain.forward_kinematics(th)
    pos = np.array(ret.pos)
    rot_wxyz = np.array(ret.rot)  # kinpy: [w, x, y, z]
    rot_xyzw = np.array([rot_wxyz[1], rot_wxyz[2], rot_wxyz[3], rot_wxyz[0]])
    return pos, rot_xyzw


def _quat_distance(q1, q2):
    """Angular distance between two quaternions (xyzw format)."""
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def solve_ik(target_pose_7d, q_init=None):
    """
    Solve IK for target EE pose using scipy optimization + kinpy FK.
    
    Args:
        target_pose_7d: [x, y, z, qx, qy, qz, qw]
        q_init: initial joint guess (6,). If None, uses zeros.
    
    Returns:
        joint_positions (6,) or None if failed
    """
    chain = _build_chain()
    target_pos = np.array(target_pose_7d[:3])
    target_quat = np.array(target_pose_7d[3:7])  # xyzw
    
    if q_init is None:
        q_init = np.zeros(6)
    
    def cost(q):
        pos, quat = _fk(chain, q)
        pos_err = np.sum((pos - target_pos) ** 2)
        rot_err = _quat_distance(quat, target_quat) ** 2
        return pos_err + 0.1 * rot_err
    
    result = minimize(
        cost, q_init, method='L-BFGS-B',
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    joints = result.x
    pos, quat = _fk(chain, joints)
    pos_err = np.linalg.norm(pos - target_pos)
    rot_err = _quat_distance(quat, target_quat)
    
    if pos_err < 0.001 and rot_err < 0.01:  # < 1mm and < 0.5deg
        print(f"IK converged (pos_err={pos_err*1000:.2f}mm, rot_err={np.degrees(rot_err):.2f}deg)")
    else:
        print(f"IK WARNING: pos_err={pos_err*1000:.2f}mm, rot_err={np.degrees(rot_err):.2f}deg")
    
    return joints


def verify_fk(joints):
    """Forward kinematics to verify the IK solution."""
    chain = _build_chain()
    pos, quat = _fk(chain, joints)
    return np.concatenate([pos, quat])


def main():
    if len(sys.argv) < 4:
        print("Usage: python get_joint_states_from_ee_pose.py X Y Z [QX QY QZ QW]")
        print("       python get_joint_states_from_ee_pose.py X Y Z QX QY QZ QW [J0 J1 J2 J3 J4 J5]")
        print()
        print("Examples:")
        print("  python get_joint_states_from_ee_pose.py -0.394 0.273 0.307 -0.5796 -0.4769 0.5278 0.3975")
        print("  python get_joint_states_from_ee_pose.py -0.394 0.273 0.307 -0.5796 -0.4769 0.5278 0.3975 -0.5 -1.5 -2.0 3.5 -2.0 3.0")
        sys.exit(1)
    
    x, y, z = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    
    if len(sys.argv) >= 8:
        qx, qy, qz, qw = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
    else:
        # Default orientation from training data
        qx, qy, qz, qw = -0.5796, -0.4769, 0.5278, 0.3975
        print(f"Using default orientation: qx={qx}, qy={qy}, qz={qz}, qw={qw}")
    
    # Optional: initial joint guess
    q_init = None
    if len(sys.argv) >= 14:
        q_init = np.array([float(sys.argv[i]) for i in range(8, 14)])
        print(f"Using initial joint guess: {q_init.round(4)}")
    
    target = [x, y, z, qx, qy, qz, qw]
    
    print(f"\nTarget EE pose: pos={np.array(target[:3]).round(4)}, quat={np.array(target[3:]).round(4)}")
    print("-" * 50)
    
    joints = solve_ik(target, q_init)
    
    if joints is not None:
        print(f"\nJoint states (rad): {joints.round(4)}")
        print(f"Joint states (deg): {np.degrees(joints).round(2)}")
        print()
        
        # Verify with FK
        ee_verify = verify_fk(joints)
        print(f"Verification FK: pos={ee_verify[:3].round(4)}, quat={ee_verify[3:].round(4)}")
        
        pos_err = np.linalg.norm(ee_verify[:3] - np.array(target[:3]))
        print(f"Position error: {pos_err*1000:.2f} mm")


if __name__ == '__main__':
    main()