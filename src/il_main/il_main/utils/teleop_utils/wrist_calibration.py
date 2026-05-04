#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import sys, termios, tty, select
import time

class WristCalibrator(Node):
    def __init__(self):
        super().__init__('wrist_calibrator')

        # --- CONFIGURATION ---
        self.target_frame = 'LHB-12898A07'  # World / Base
        self.source_frame = 'LHR-9F6576C3'  # Tracker
        self.virtual_frame = 'virtual_palm_link'
        self.z_thickness_offset = -0.04     # Estimated distance from tracker down to wrist bone (meters)
        # ---------------------

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.samples = []
        self.is_recording = False
        self.wrist_center_world = None 

        print("\n" + "=" * 60)
        print(" FULL WRIST CALIBRATION (TRANSLATION + ROTATION)")
        print("=" * 60)
        print("STEP 1: FIND PIVOT CENTER")
        print("1. Rest forearm/wrist flat on the table.")
        print("2. Press [SPACE] to start recording.")
        print("3. Perform a 'windshield wiper' motion with your hand.")
        print("4. Press [SPACE] to stop.")
        print("5. Press [c] to calculate the center.")
        print("-" * 60)
        print("STEP 2: FIND ROTATION ALIGNMENT")
        print("6. Hold the tracker vertically/normally in your 'Zero' pose.")
        print("   (Hand straight, fingers pointing forward)")
        print("7. Press [m] to measure and generate the static transform.")
        print("-" * 60)
        print("Press [q] to quit.")

    def get_transform(self):
        try:
            if not self.tf_buffer.can_transform(self.target_frame, self.source_frame, Time()):
                return None
            return self.tf_buffer.lookup_transform(self.target_frame, self.source_frame, Time())
        except Exception:
            return None

    def run(self):
        settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())

            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.01)

                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    
                    if key == ' ':
                        self.is_recording = not self.is_recording
                        print(f"\nRecording: {'ON' if self.is_recording else 'OFF'}")
                    
                    elif key == 'c':
                        self.calculate_circle()
                        self.samples = []
                    
                    elif key == 'm':
                        self.calculate_alignment()

                    elif key == 'q':
                        break

                if self.is_recording:
                    t = self.get_transform()
                    if t:
                        self.samples.append([
                            t.transform.translation.x, 
                            t.transform.translation.y,
                            t.transform.translation.z
                        ])
                        print(f"\rPoints: {len(self.samples)}", end="")
                    time.sleep(0.02)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    def calculate_circle(self):
        if len(self.samples) < 10:
            print("\nNot enough points recorded.")
            return

        points = np.array(self.samples)
        
        # 2D Circle Fit for X/Y
        def calc_R(c): return np.sqrt((points[:,0]-c[0])**2 + (points[:,1]-c[1])**2)
        def f_2(c): return calc_R(c) - calc_R(c).mean()
        
        center_est = np.mean(points[:, :2], axis=0)
        center_2d = least_squares(f_2, center_est).x
        
        # Take the average Z of the tracker, and apply the physical thickness offset
        mean_z = np.mean(points[:, 2]) + self.z_thickness_offset
        
        self.wrist_center_world = np.array([center_2d[0], center_2d[1], mean_z])
        radius = calc_R(center_2d).mean()

        print(f"\n\n[STEP 1 COMPLETE]")
        print(f"Wrist Pivot (World): X={center_2d[0]:.4f}, Y={center_2d[1]:.4f}, Z={mean_z:.4f}")
        print(f"Pivot Radius: {radius:.4f} m")
        print("-> Now hold hand straight/vertically and press [m]!")

    def calculate_alignment(self):
        if self.wrist_center_world is None:
            print("\nPlease perform Step 1 (Circle fit) first!")
            return

        t = self.get_transform()
        if t is None:
            print("Tracker not found!")
            return

        # 1. Get current Tracker Pose in World
        tx = t.transform.translation.x
        ty = t.transform.translation.y
        tz = t.transform.translation.z
        
        tracker_quat = [
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w
        ]
        
        tracker_rot = R.from_quat(tracker_quat)

        # 2. Vector from Tracker to Wrist (in World Frame)
        v_world = self.wrist_center_world - np.array([tx, ty, tz])

        # 3. Rotate vector into Tracker's Local Frame
        # By multiplying by the inverse of the tracker's rotation, 
        # we see exactly where the wrist is relative to the tracker itself.
        v_local = tracker_rot.inv().apply(v_world)
        local_x, local_y, local_z = v_local

        # 4. Calculate Yaw offset
        # We align the new virtual frame's X-axis to point directly from the tracker to the wrist pivot.
        yaw_offset = np.arctan2(local_y, local_x)
        
        print("\n" + "#" * 50)
        print(" READY STATIC TRANSFORM COMMAND")
        print("#" * 50)
        print(f"Local Translation Offset: X={local_x:.4f}, Y={local_y:.4f}, Z={local_z:.4f}")
        print(f"Local Yaw Offset: {np.degrees(yaw_offset):.1f} degrees")
        print("-" * 50)
        print("Run this in a new terminal, or add it to your launch file:\n")
        
        # ROS 2 static_transform_publisher uses: x y z yaw pitch roll parent child
        print(f"ros2 run tf2_ros static_transform_publisher "
              f"{local_x:.5f} {local_y:.5f} {local_z:.5f} "
              f"{yaw_offset:.5f} 0.0 0.0 "
              f"{self.source_frame} {self.virtual_frame}")
        print("-" * 50)

def main():
    rclpy.init()
    node = WristCalibrator()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()