#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController, ListControllers
from builtin_interfaces.msg import Duration

from service_interfaces.msg import SetAngle1

TARGET_CONTROLLER = "scaled_joint_trajectory_controller"
CONFLICTING_CONTROLLER = "forward_position_controller"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class TrajectoryControllerClient(Node):
    def __init__(self):
        super().__init__('trajectory_controller_client')

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            f'/{TARGET_CONTROLLER}/follow_joint_trajectory',
        )
        self._switch_client = self.create_client(
            SwitchController, '/controller_manager/switch_controller'
        )
        self._list_client = self.create_client(
            ListControllers, '/controller_manager/list_controllers'
        )
        self._angle_publisher = self.create_publisher(SetAngle1, '/set_angle_data', 10)

    def ensure_controller_active(self):
        if not self._list_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('Controller Manager Service nicht verfügbar.')
            return False

        req = ListControllers.Request()
        future = self._list_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        for c in future.result().controller:
            if c.name == TARGET_CONTROLLER and c.state == "active":
                self.get_logger().info(f'{TARGET_CONTROLLER} ist bereits aktiv.')
                return True

        self.get_logger().info(f'Wechsle zu {TARGET_CONTROLLER}...')
        if not self._switch_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('Switch Controller Service nicht verfügbar.')
            return False

        switch_req = SwitchController.Request()
        switch_req.activate_controllers = [TARGET_CONTROLLER]
        switch_req.deactivate_controllers = [CONFLICTING_CONTROLLER]
        switch_req.strictness = SwitchController.Request.BEST_EFFORT

        switch_future = self._switch_client.call_async(switch_req)
        rclpy.spin_until_future_complete(self, switch_future)

        if switch_future.result().ok:
            self.get_logger().info('Controller-Wechsel erfolgreich.')
            return True
        else:
            self.get_logger().error('Controller-Wechsel fehlgeschlagen.')
            return False

    def switch_to_forward_position(self):
        self.get_logger().info(f'Schalte um auf {CONFLICTING_CONTROLLER}...')

        if not self._switch_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('Switch Controller Service nicht verfügbar.')
            return

        switch_req = SwitchController.Request()
        switch_req.activate_controllers = [CONFLICTING_CONTROLLER]
        switch_req.deactivate_controllers = [TARGET_CONTROLLER]
        switch_req.strictness = SwitchController.Request.BEST_EFFORT

        switch_future = self._switch_client.call_async(switch_req)
        rclpy.spin_until_future_complete(self, switch_future)

        if switch_future.result().ok:
            self.get_logger().info('Rückwechsel erfolgreich.')
        else:
            self.get_logger().error('Rückwechsel fehlgeschlagen.')

    def send_hand_command(self):
        msg = SetAngle1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]
        msg.angles = [1000, 1000, 1000, 1000, 1000, 0]
        self.get_logger().info('Publiziere Hand-Daten auf /set_angle_data...')
        self._angle_publisher.publish(msg)

    def send_joint_goal(self, joint_positions, duration_sec=5.0):
        """
        Sendet ein Joint-Ziel direkt an den Trajectory Controller.
        joint_positions: dict {joint_name: position} oder list [pos1, ..., pos6]
        duration_sec: Zeit für die Bewegung
        """
        if not self.ensure_controller_active():
            return False

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('FollowJointTrajectory Action Server nicht verfügbar.')
            return False

        # Build trajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()

        if isinstance(joint_positions, dict):
            point.positions = [joint_positions[name] for name in JOINT_NAMES]
        elif isinstance(joint_positions, (list, tuple)):
            point.positions = list(joint_positions)
        else:
            self.get_logger().error('joint_positions muss dict oder list sein.')
            return False

        point.velocities = [0.0] * 6
        point.time_from_start = Duration(
            sec=int(duration_sec),
            nanosec=int((duration_sec % 1) * 1e9),
        )
        trajectory.points.append(point)

        # Send goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory

        self.get_logger().info(f'Sende Trajectory-Ziel...')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self._goal_response_callback)
        return True

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Ziel abgelehnt.')
            raise SystemExit
            return

        self.get_logger().info('Ziel akzeptiert, führe aus...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        result = future.result().result
        if result.error_code == 0:
            self.get_logger().info('Bewegung erfolgreich.')
        else:
            self.get_logger().error(f'Fehlercode: {result.error_code}')

        raise SystemExit


def main(args=None):
    rclpy.init(args=args)

    start_joints = {
        "shoulder_pan_joint": -0.5362704435931605,
        "shoulder_lift_joint": -1.418149085049965,
        "elbow_joint": -2.6668741703033447,
        "wrist_1_joint": 4.063593550319336,
        "wrist_2_joint": -2.154945675526754,
        "wrist_3_joint": 3.0416924953460693,
    }

    client = TrajectoryControllerClient()

    try:
        success = client.send_joint_goal(start_joints, duration_sec=5.0)
        if success:
            rclpy.spin(client)
    except SystemExit:
        client.send_hand_command()

        import time
        time.sleep(0.5)

        client.switch_to_forward_position()
    except Exception as e:
        print(f"Laufzeitfehler: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()