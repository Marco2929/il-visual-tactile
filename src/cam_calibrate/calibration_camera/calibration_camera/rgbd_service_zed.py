#!/usr/bin/env python3
"""RGBD Service for ZED 2i — subscribes to ROS topics from zed_wrapper."""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

from interfaces.srv import GetRGBD
from interfaces.msg import RGBD

import threading


class ZedRGBDService(Node):
    def __init__(self):
        super().__init__('zed_rgbd_service')

        self._bridge = CvBridge()
        self._cb_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()

        # Parameters (match zed_wrapper topics)
        self.declare_parameter('color_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/color/rect/camera_info')

        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value

        # State
        self._latest_color = None
        self._latest_depth = None
        self._latest_info = None

        # Subscribers
        self.create_subscription(Image, color_topic, self._color_cb, 10)
        self.create_subscription(Image, depth_topic, self._depth_cb, 10)
        self.create_subscription(CameraInfo, info_topic, self._info_cb, 10)

        # Service
        self.create_service(
            GetRGBD, '/zed/get_rgbd',
            self._handle_get_rgbd,
            callback_group=self._cb_group)

        self.get_logger().info(
            f'ZED RGBD Service (topic-based)\n'
            f'  Color: {color_topic}\n'
            f'  Depth: {depth_topic}\n'
            f'  Info:  {info_topic}\n'
            f'  Service: /zed/get_rgbd')

    def _color_cb(self, msg: Image):
        with self._lock:
            self._latest_color = msg

    def _depth_cb(self, msg: Image):
        with self._lock:
            self._latest_depth = msg

    def _info_cb(self, msg: CameraInfo):
        with self._lock:
            self._latest_info = msg

    def _handle_get_rgbd(self, request, response):
        with self._lock:
            color = self._latest_color
            depth = self._latest_depth
            info = self._latest_info

        if color is None:
            response.success = False
            response.message = 'No color image received yet — is zed_wrapper running?'
            return response

        if depth is None:
            response.success = False
            response.message = 'No depth image received yet'
            return response

        if info is None:
            response.success = False
            response.message = 'No camera info received yet'
            return response

        # Build RGBD message
        rgbd = RGBD()
        rgbd.rgb = color
        rgbd.depth = depth
        rgbd.rgb_camera_info = info

        response.success = True
        response.message = f'OK ({color.width}x{color.height})'
        response.rgbd = rgbd

        return response


def main(args=None):
    rclpy.init(args=args)
    node = ZedRGBDService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()