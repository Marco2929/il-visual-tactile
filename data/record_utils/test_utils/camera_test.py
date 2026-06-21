import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
import message_filters

class MultiRealSenseSyncNode(Node):
    def __init__(self):
        super().__init__("multicam_sync_node")

        self.JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        self.latest_joint_state = {name: None for name in self.JOINT_NAMES}
        self.sync_count = 0
        self.last_log_time = self.get_clock().now()

        # Buffer für Depth-Bilder (da wir sie nicht in den harten Sync zwingen)
        self.latest_l_depth = None
        self.latest_r_depth = None

        # 1. Depth-Subscriber (Manuelle Speicherung statt harter Sync)
        self.create_subscription(Image, "/cam_left/cam_left/aligned_depth_to_color/image_raw", 
                                 self._l_depth_cb, qos_profile_sensor_data)
        self.create_subscription(Image, "/cam_right/cam_right/aligned_depth_to_color/image_raw", 
                                 self._r_depth_cb, qos_profile_sensor_data)

        # 2. Message Filters für den Sync (Joints + Color Left + Color Right)
        self.joint_filt = message_filters.SimpleFilter()
        self.l_color_sub = message_filters.Subscriber(self, Image, "/cam_left/cam_left/color/image_raw", qos_profile=qos_profile_sensor_data)
        self.r_color_sub = message_filters.Subscriber(self, Image, "/cam_right/cam_right/color/image_raw", qos_profile=qos_profile_sensor_data)

        # 3. Aggregated Joint Subscription
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

        # 4. Der Synchronizer (Reduziert auf 3 Quellen für höhere Trefferrate)
        # Slop auf 0.03 (30ms), da 40ms bei 30fps manchmal falsche Nachbarn greift
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.joint_filt, self.l_color_sub, self.r_color_sub],
            queue_size=30,
            slop=0.03 
        )
        self.ts.registerCallback(self._sync_callback)

        self.get_logger().info("🚀 Synchronizer läuft auf 3 Topics (Joints + 2x Color)")

    def _l_depth_cb(self, msg): self.latest_l_depth = msg
    def _r_depth_cb(self, msg): self.latest_r_depth = msg

    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self.latest_joint_state:
                self.latest_joint_state[name] = pos
        
        if 'shoulder_pan_joint' in msg.name:
            full_msg = JointState()
            full_msg.header = msg.header
            full_msg.position = [self.latest_joint_state[n] or 0.0 for n in self.JOINT_NAMES]
            self.joint_filt.signalMessage(full_msg)

    def _sync_callback(self, joint_msg, l_rgb, r_rgb):
        # Wir prüfen hier, ob wir passende Depth-Bilder haben
        if self.latest_l_depth is None or self.latest_r_depth is None:
            return

        self.sync_count += 1
        
        # Performance Logging
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > 2e9:
            l_ts = l_rgb.header.stamp.sec + l_rgb.header.stamp.nanosec * 1e-9
            r_ts = r_rgb.header.stamp.sec + r_rgb.header.stamp.nanosec * 1e-9
            cam_diff = abs(l_ts - r_ts) * 1000
            
            self.get_logger().info(f"SYNC [OK] | FPS: {self.sync_count/2:.1f} | Delta: {cam_diff:.2f}ms")
            self.sync_count = 0
            self.last_log_time = now

        # Hier würdest du jetzt dein Zarr-Dataset oder Training-Buffer befüllen:
        # data = { 'joints': joint_msg, 'left_rgb': l_rgb, 'left_depth': self.latest_l_depth, ... }

def main():
    rclpy.init()
    node = MultiRealSenseSyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()