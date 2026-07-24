#!/usr/bin/env python3
"""Dry-run monitor: prints current EE vs. model target side-by-side."""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import Buffer, TransformListener


class DryRunMonitor(Node):
    def __init__(self):
        super().__init__('dry_run_monitor')

        # TF listener for current robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to model targets
        self.last_target = None
        self.create_subscription(
            PoseStamped, '/target_frame', self._target_cb, 10)

        # Print at 2Hz
        self.create_timer(0.5, self._print_comparison)

        self.get_logger().info("Dry-run monitor started. Comparing current EE vs model target.")

    def _target_cb(self, msg: PoseStamped):
        self.last_target = msg

    def _print_comparison(self):
        # Get current EE pose from TF (try tool0 first, fallback to inspire_base)
        cur_pos = None
        cur_quat = None
        cur_frame = None
        for frame in ['tool0', 'right_inspire_base_link']:
            try:
                t = self.tf_buffer.lookup_transform(
                    'base_link', frame, rclpy.time.Time())
                cur_pos = np.array([
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z])
                cur_quat = np.array([
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w])
                cur_frame = frame
                break
            except Exception:
                continue

        # Get model target (note: this is tool0, but we can compare)
        if self.last_target is None:
            tgt_pos = None
            tgt_quat = None
        else:
            p = self.last_target.pose
            tgt_pos = np.array([p.position.x, p.position.y, p.position.z])
            tgt_quat = np.array([
                p.orientation.x, p.orientation.y,
                p.orientation.z, p.orientation.w])

        # Print
        print("\n" + "=" * 70)
        if cur_pos is not None:
            print(f"  CURRENT ({cur_frame}): xyz={np.round(cur_pos, 4).tolist()}  "
                  f"quat={np.round(cur_quat, 3).tolist()}")
        else:
            print("  CURRENT: TF not available (tried tool0, right_inspire_base_link)")

        if tgt_pos is not None:
            print(f"  TARGET  (tool0):        xyz={np.round(tgt_pos, 4).tolist()}  "
                  f"quat={np.round(tgt_quat, 3).tolist()}")
        else:
            print("  TARGET: no chunk received yet")

        if cur_pos is not None and tgt_pos is not None:
            dist = np.linalg.norm(cur_pos - tgt_pos)
            note = "" if cur_frame == "tool0" else " (includes ~18.6mm frame offset)"
            print(f"  DISTANCE: {dist*1000:.1f} mm{note}")
            delta = tgt_pos - cur_pos
            print(f"  DELTA:    dx={delta[0]*1000:.1f}mm  "
                  f"dy={delta[1]*1000:.1f}mm  dz={delta[2]*1000:.1f}mm")
        print("=" * 70)


def main(args=None):
    rclpy.init(args=args)
    node = DryRunMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()