#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from controller_manager_msgs.srv import SwitchController, ListControllers

from service_interfaces.msg import SetAngle1

GROUP_NAME = "ur_workcell"
TARGET_CONTROLLER = "scaled_joint_trajectory_controller"
CONFLICTING_CONTROLLER = "forward_position_controller"

class MoveItControllerClient(Node):
    def __init__(self):
        super().__init__('moveit_controller_client')
        
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self._switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self._list_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        
        self._angle_publisher = self.create_publisher(SetAngle1, '/set_angle_data', 10)

    def ensure_controller_active(self):
        if not self._list_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('Controller Manager Service nicht verfügbar.')
            return False

        req = ListControllers.Request()
        future = self._list_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        active_controllers = [c.name for c in future.result().controller]
        
        if TARGET_CONTROLLER in active_controllers:
            state = next((c.state for c in future.result().controller if c.name == TARGET_CONTROLLER), "")
            if state == "active":
                self.get_logger().info(f'Controller {TARGET_CONTROLLER} ist bereits aktiv.')
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
        """
        Deaktiviert den Trajectory Controller und aktiviert den Forward Position Controller.
        """
        self.get_logger().info(f'Ende der Ausführung: Schalte um auf {CONFLICTING_CONTROLLER}...')
        
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
            self.get_logger().info('Rückwechsel auf Forward Position Controller erfolgreich.')
        else:
            self.get_logger().error('Rückwechsel fehlgeschlagen.')

    def send_hand_command(self):
        msg = SetAngle1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]

        msg.angles = [1000, 1000, 1000, 1000, 1000, 0]
        
        self.get_logger().info('Publiziere Hand-Daten auf /set_angle_data...')
        # self._angle_publisher.publish(msg)

    def send_joint_goal(self, joint_goals):
        if not self.ensure_controller_active():
            return False

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup Action Server nicht verfügbar.')
            return False

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = GROUP_NAME
        
        constraints = Constraints()
        constraints.name = "joint_goal"
        
        for joint_name, position in joint_goals.items():
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = position
            joint_constraint.tolerance_above = 0.01
            joint_constraint.tolerance_below = 0.01
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)

        goal_msg.request.goal_constraints.append(constraints)
        
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.max_acceleration_scaling_factor = 0.1
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.replan = True

        self.get_logger().info(f'Sende Ziel an {GROUP_NAME}...')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Ziel abgelehnt.')
            return

        self.get_logger().info('Ziel akzeptiert, führe aus...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == 1:
            self.get_logger().info('Bewegung erfolgreich.')
        else:
            self.get_logger().error(f'Fehlercode: {result.error_code.val}')
            
        raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    
    target_joints = {
        "elbow_joint": -2.291002035140991,
        "shoulder_lift_joint": -1.621289392510885,
        "shoulder_pan_joint": -2.270664993916647,
        "wrist_1_joint": 0.4653627115437011,
        "wrist_2_joint": 2.271588087081909,
        "wrist_3_joint": 2.998962640762329
    }

    start_joints = {
    "elbow_joint": -2.1890,
    "shoulder_lift_joint": -1.0298,
    "shoulder_pan_joint": -1.1536,
    "wrist_1_joint": 3.2357,
    "wrist_2_joint": -1.1918,
    "wrist_3_joint": -3.1573
}

    client = MoveItControllerClient()
    
    try:
        success = client.send_joint_goal(start_joints)
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