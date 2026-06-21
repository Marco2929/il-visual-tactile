# palm_transform.py — wird von DataLoader UND Inference genutzt

import numpy as np
import kinpy as kp


class PalmFrameTransform:
    """Reusable: works in DataLoader AND live inference."""

    def __init__(self, urdf_path):
        with open(urdf_path, 'r') as f:
            urdf_data = f.read()
        self.chain = kp.build_chain_from_urdf(urdf_data)
        self.joint_names = self.chain.get_joint_parameter_names()
        self.tip_links = [f'rl_dg_{i}_4' for i in range(1, 6)]
        self.palm_link = 'tool0' # was set to rl_dg_palm should also work with tool0

    def __call__(self, joints_deg: np.ndarray, force_torque: np.ndarray) -> np.ndarray:
        """
        Args:
            joints_deg: (20,) joint angles in degrees
            force_torque: (30,) raw F/T values (5×6, tip-local)

        Returns:
            (5, 13) palm-frame observation
        """
        joints_rad = np.deg2rad(joints_deg)
        joint_dict = {name: joints_rad[i] for i, name in enumerate(self.joint_names)}

        fk_dict = self.chain.forward_kinematics(joint_dict)

        T_palm = fk_dict[self.palm_link]
        R_palm = self._quat_to_matrix(T_palm.rot)
        p_palm = np.array(T_palm.pos)

        ft = force_torque.reshape(5, 6)
        fingers = []

        for i, tip_link in enumerate(self.tip_links):
            T_tip = fk_dict[tip_link]
            R_tip = self._quat_to_matrix(T_tip.rot)
            p_tip = np.array(T_tip.pos)

            tip_in_palm = R_palm.T @ (p_tip - p_palm)
            f_palm = R_palm.T @ (R_tip @ ft[i, 0:3])
            tau_palm = R_palm.T @ (R_tip @ ft[i, 3:6]) + np.cross(tip_in_palm, f_palm)
            finger_joints = joints_rad[i*4:(i+1)*4]

            fingers.append(np.concatenate([tip_in_palm, f_palm, tau_palm, finger_joints]))

        return np.stack(fingers)  # (5, 13)

    def _quat_to_matrix(self, q):
        q = np.array(q)
        w, x, y, z = q[0], q[1], q[2], q[3]
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
        ])