#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
import pyzed.sl as sl
import json
import sys
import time

class SVOPlayer(Node):
    def __init__(self, svo_filepath):
        super().__init__('svo_player')
        
        # We publish to a slightly different topic to avoid colliding with real hardware
        self.joint_pub = self.create_publisher(JointState, '/replay/joint_states', 10)
        
        self.cam = sl.Camera()
        self._init_camera(svo_filepath)
        
        # Run playback loop at roughly the camera's framerate
        fps = self.cam.get_camera_information().camera_configuration.fps
        timer_period = 1.0 / fps if fps > 0 else 1.0 / 60.0
        
        self.get_logger().info(f"Playing back at {fps} FPS...")
        self.timer = self.create_timer(timer_period, self._playback_step)

    def _init_camera(self, filepath):
        init = sl.InitParameters()
        init.set_from_svo_file(filepath)
        init.svo_real_time_mode = True  # Set to True to play at recorded speed
        init.depth_mode = sl.DEPTH_MODE.NONE # Change if you need depth playback
        
        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open SVO: {status}")
            sys.exit(1)
            
        self.get_logger().info("SVO File opened successfully.")

    def _playback_step(self):
        runtime = sl.RuntimeParameters()
        err = self.cam.grab(runtime)
        
        if err == sl.ERROR_CODE.SUCCESS:
            # 1. Get the current frame's timestamp
            current_ts = self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
            
            # 2. Retrieve custom data for this exact timestamp
            custom_data = {}
            res = self.cam.retrieve_svo_data("joint_states", custom_data, current_ts, current_ts)
            
            if res == sl.ERROR_CODE.SUCCESS and current_ts in custom_data:
                # 3. Extract and parse the JSON string
                svo_content = custom_data[current_ts]
                
                # Handle varying PyZED API versions
                if hasattr(svo_content, 'get_content_as_string'):
                    json_string = svo_content.get_content_as_string()
                else:
                    json_string = svo_content.get_string_content()
                    
                joint_states_array = json.loads(json_string)
                
                # 4. Publish all ROS messages that were queued during this frame
                for state in joint_states_array:
                    msg = JointState()
                    
                    # Reconstruct the original ROS timestamp
                    sec = int(state['timestamp_ns'] // 1e9)
                    nanosec = int(state['timestamp_ns'] % 1e9)
                    msg.header.stamp = Time(sec=sec, nanosec=nanosec)
                    msg.header.frame_id = "replay_base_link"
                    
                    msg.name = state['names']
                    msg.position = state['positions']
                    
                    self.joint_pub.publish(msg)
                    
        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            self.get_logger().info("End of SVO file reached. Shutting down...")
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    
    # You can change this to accept a command line argument later
    filepath = '/root/ros2_ws/data/episodes/test_collection/episode_000/recording.svo2'
    
    node = SVOPlayer(filepath)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cam.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()