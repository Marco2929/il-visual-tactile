#!/usr/bin/env python3
"""Publishes UR10e TCP pose as PoseStamped from tf2."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf2_ros import TransformException


class TCPPosePublisher(Node):
    def __init__(self):
        super().__init__('tcp_pose_publisher')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tcp_frame', 'tool0')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('pose_topic', '/robot/pose')

        self._base_frame = self.get_parameter('base_frame').value
        self._tcp_frame = self.get_parameter('tcp_frame').value
        rate = self.get_parameter('publish_rate').value
        topic = self.get_parameter('pose_topic').value

        # TF2
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Publisher
        self._pub = self.create_publisher(PoseStamped, topic, 10)

        # Timer
        self._timer = self.create_timer(1.0 / rate, self._timer_callback)

        self.get_logger().info(
            f'TCP Pose Publisher: {self._base_frame} → {self._tcp_frame} on "{topic}"')

    def _timer_callback(self):
        try:
            t = self._tf_buffer.lookup_transform(
                self._base_frame, self._tcp_frame, rclpy.time.Time())

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self._base_frame
            msg.pose.position.x = t.transform.translation.x
            msg.pose.position.y = t.transform.translation.y
            msg.pose.position.z = t.transform.translation.z
            msg.pose.orientation = t.transform.rotation

            self._pub.publish(msg)

        except TransformException:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = TCPPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()