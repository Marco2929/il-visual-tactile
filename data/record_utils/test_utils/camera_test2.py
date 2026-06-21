import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters

class CameraSyncTest(Node):
    def __init__(self):
        super().__init__('camera_sync_test')
        
        # DEBUG: Logge die Topics, auf die wir warten
        # Basierend auf deinen Logs sind das wahrscheinlich diese:
        topic_l = "/cam_left/cam_left/aligned_depth_to_color/image_raw"
        topic_r = "/cam_right/cam_right/aligned_depth_to_color/image_raw"
        
        self.get_logger().info(f"Warte auf: {topic_l} und {topic_r}")

        self.l_sub = message_filters.Subscriber(self, Image, topic_l)
        self.r_sub = message_filters.Subscriber(self, Image, topic_r)
        
        # slop=0.05 (50ms) ist großzügig für den ersten Test
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.l_sub, self.r_sub], 
            queue_size=10, 
            slop=0.04
        )
        self.ts.registerCallback(self.callback)

    def callback(self, msg_l, msg_r):
        stamp_l = msg_l.header.stamp.sec + msg_l.header.stamp.nanosec * 1e-9
        stamp_r = msg_r.header.stamp.sec + msg_r.header.stamp.nanosec * 1e-9
        delta = abs(stamp_l - stamp_r) * 1000
        self.get_logger().info(f"SYNC TREFFER! Delta: {delta:.2f} ms")

def main():
    rclpy.init()
    node = CameraSyncTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()