#!/usr/bin/env python3

"""
                        RH8D Hand Service Controller Node

This ROS 2 Python node provides a high-level service interface to control the RH8D robotic hand. 

The node runs in parallel using ROS timers, subscriptions, and services, 
so different parts (receiving joint states, handling commands, moving joints) happen at the same time. 
This makes the flow non-linear, because the hand’s movement is controlled continuously in the background while 
waiting for new commands or sensor updates.

Key features include:
- Real-time subscription to joint state and fingertip force feedback.
- Gradual, rate-limited actuator control with per-joint speeds.
- Object presence detection using a palm range sensor.
- Stable grasp detection and contact-based grasp locking.
- Multi-threaded executor support with ReentrantCallbackGroups to handle service requests, timers, and subscriptions concurrently.

Workflow:
1. HandCommand service receives high-level commands from other nodes.
2. Node ramps joints toward target positions while monitoring contact sensors.
3. Grasp commands latch finger positions once stable contact is detected.
4. Command status (success/failure) is returned synchronously via the service.


Note:
A commented version of the file is avaialble here: /v4_rh8d_ws/src/rh8d_mujoco_control/rh8d_mujoco_control/rh8dL_object_pick_services.py
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup

from control_msgs.msg import DynamicJointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from rh8d_mujoco_interfaces.srv import HandCommand


class RH8DHandServiceController(Node):
    def __init__(self):
        super().__init__('rh8d_hand_service_controller')

        self.state_lock = threading.Lock()

        self.service_cb_group = ReentrantCallbackGroup()
        self.timer_cb_group = ReentrantCallbackGroup()
        self.sub_cb_group = ReentrantCallbackGroup()

        self.have_state = False
        self.latest_dynamic_state = None

        self.create_subscription(
            DynamicJointState,
            '/dynamic_joint_states',
            self.dynamic_state_callback,
            10,
            callback_group=self.sub_cb_group
        )

        self.command_pub = self.create_publisher(
            JointTrajectory,
            '/hand_trajectory_controller/joint_trajectory',
            10
        )

        self.srv = self.create_service(
            HandCommand,
            '/rh8d/hand_command',
            self.handle_hand_command,
            callback_group=self.service_cb_group
        )

        self.command_joint_names = [
            'forearm:1--base:1',
            'palm_axis:1--forearm:1',
            'palm_assembly:1--palm_axis:1',
            'Thumb_axis--palm_assembly:1',
            'Thumb_Methacarpal--Thumb_axis',
            'Index_Proximal--palm_assembly:1',
            'Middle_Proximal--palm_assembly:1',
            'Ring_Proximal--palm_assembly:1'
        ]

        self.contact_counter = {
            'thumb': 0,
            'index': 0,
            'middle': 0,
            'ring': 0,
            'small': 0,
        }
        self.required_contact_cycles = 1

        self.range_threshold = 0.25
        self.tip_force_threshold = 0.1 # force applied on the objects.

        self.rate_forearm = 0.5
        self.rate_palm_axis = 0.5
        self.rate_palm_l = 0.5
        self.rate_thumb_axis = 0.5

        self.rate_thumb = 0.1
        self.rate_index = 0.1
        self.rate_middle = 0.1
        self.rate_ring_small = 0.1

        # [forearm, palm_axis, palm_l, thumb_axis, thumb_meta, index, middle, ring]
        self.current_command = [0.0] * 8

        self.thumb_axis_open = 1.57

        self.pose_fully_open = [0.0, 0.0, 0.0, self.thumb_axis_open, 0.0, 0.0, 0.0, 0.0]
        self.pose_fully_closed = [0.0, 0.0, 0.0, self.thumb_axis_open, 1.57, 1.57, 1.57, 1.57]

        self.gesture_map = {
            'fully_open': {
                'positions': [0.0, 0.0, 0.0, self.thumb_axis_open, 0.0, 0.0, 0.0, 0.0]
            },
            'fully_closed': {
                'positions': [0.0, 0.0, 0.0, self.thumb_axis_open, 1.57, 1.57, 1.57, 1.57]
            },
            'index_point': {
                'positions': [0.0, 0.0, 0.0, self.thumb_axis_open, 1.57, 0.0, 1.57, 1.57]
            },
            'middle_point': {
                'positions': [0.0, 0.0, 0.0, self.thumb_axis_open, 1.57, 1.57, 0.0, 1.57]
            },
            'victory_sign': {
                'positions': [0.0, 0.0, 0.0, self.thumb_axis_open, 1.57, 0.0, 0.0, 1.57]
            }
        }

        self.active_mode = None
        self.active_command = None
        self.command_done = False
        self.command_success = False
        self.command_message = ""
        self.busy = False

        self.last_object_seen_time = None

        self.grasp_locked = False
        self.latched_positions = {
            'thumb_meta': 0.0,
            'index': 0.0,
            'middle': 0.0,
            'ring': 0.0
        }

        self.timer_period = 0.05
        self.timer = self.create_timer(
            self.timer_period,
            self.step,
            callback_group=self.timer_cb_group
        )

        self.get_logger().info('RH8D hand service controller started')

    def dynamic_state_callback(self, msg: DynamicJointState):
        with self.state_lock:
            self.latest_dynamic_state = msg
            self.have_state = True

    def get_interface_value(self, name: str, interface_name: str):
        if self.latest_dynamic_state is None:
            return None

        msg = self.latest_dynamic_state
        for i, joint_name in enumerate(msg.joint_names):
            if joint_name != name:
                continue

            if i >= len(msg.interface_values):
                return None

            block = msg.interface_values[i]
            for j, if_name in enumerate(block.interface_names):
                if if_name == interface_name and j < len(block.values):
                    return block.values[j]

        return None

    def get_joint_value(self, joint_name: str) -> float:
        value = self.get_interface_value(joint_name, 'position')
        return 0.0 if value is None else value

    def get_tip_force_mag(self, sensor_name: str) -> float:
        x = self.get_interface_value(sensor_name, 'x')
        y = self.get_interface_value(sensor_name, 'y')
        z = self.get_interface_value(sensor_name, 'z')
        if x is None or y is None or z is None:
            return 0.0
        return math.sqrt(x * x + y * y + z * z)

    def object_present(self) -> bool:
        rf = self.get_interface_value('palm_range', 'value')
        if rf is None:
            return False
        return (rf > 0.0) and (rf < self.range_threshold)

    def publish_command(self, cmd_values):
        msg = JointTrajectory()
        msg.joint_names = list(self.command_joint_names)

        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in cmd_values]
        pt.time_from_start = Duration(
            sec=0,
            nanosec=int(self.timer_period * 1e9)
        )

        msg.points.append(pt)
        self.command_pub.publish(msg)

    def move_towards(self, current: float, target: float, rate: float, dt: float) -> float:
        if current < target:
            return min(current + rate * dt, target)
        if current > target:
            return max(current - rate * dt, target)
        return current

    def ramp_command_towards(self, target_positions, dt, freeze_mask=None):
        if freeze_mask is None:
            freeze_mask = [False] * 8

        new_cmd = list(self.current_command)

        def maybe_move(index, target, rate):
            if freeze_mask[index]:
                return self.current_command[index]
            return self.move_towards(self.current_command[index], target, rate, dt)

        new_cmd[0] = maybe_move(0, target_positions[0], self.rate_forearm)
        new_cmd[1] = maybe_move(1, target_positions[1], self.rate_palm_axis)
        new_cmd[2] = maybe_move(2, target_positions[2], self.rate_palm_l)
        new_cmd[3] = maybe_move(3, target_positions[3], self.rate_thumb_axis)

        new_cmd[4] = maybe_move(4, target_positions[4], self.rate_thumb)
        new_cmd[5] = maybe_move(5, target_positions[5], self.rate_index)
        new_cmd[6] = maybe_move(6, target_positions[6], self.rate_middle)
        new_cmd[7] = maybe_move(7, target_positions[7], self.rate_ring_small)

        self.current_command = new_cmd
        self.publish_command(new_cmd)

        reached = all(
            freeze_mask[i] or abs(new_cmd[i] - target_positions[i]) < 0.02
            for i in range(8)
        )
        return reached

    def is_grasp_contact_stable(self, finger_name: str) -> bool:
        tip_map = {
            'thumb': 'thumb_tip_force',
            'index': 'index_tip_force',
            'middle': 'middle_tip_force',
            'ring': 'ring_tip_force',
            'small': 'small_tip_force',
        }

        sensor_name = tip_map.get(finger_name)
        if sensor_name is None:
            return False

        tip_mag = self.get_tip_force_mag(sensor_name)
        contact_now = tip_mag > self.tip_force_threshold

        if contact_now:
            self.contact_counter[finger_name] += 1
        else:
            self.contact_counter[finger_name] = 0

        return self.contact_counter[finger_name] >= self.required_contact_cycles

    def reset_contact_counters(self):
        self.contact_counter = {
            'thumb': 0,
            'index': 0,
            'middle': 0,
            'ring': 0,
            'small': 0,
        }

    def latch_current_grasp(self):
        self.latched_positions['thumb_meta'] = self.current_command[4]
        self.latched_positions['index'] = self.current_command[5]
        self.latched_positions['middle'] = self.current_command[6]
        self.latched_positions['ring'] = self.current_command[7]
        self.grasp_locked = True

        self.get_logger().info(
            f"Latched grasp: thumb_meta={self.latched_positions['thumb_meta']:.3f}, "
            f"index={self.latched_positions['index']:.3f}, "
            f"middle={self.latched_positions['middle']:.3f}, "
            f"ring={self.latched_positions['ring']:.3f}"
        )

    def finish_command(self, success: bool, message: str):
        self.command_done = True
        self.command_success = success
        self.command_message = message
        self.active_mode = None
        self.active_command = None
        self.busy = False

    def start_new_command(self, mode: str, gesture_name: str):
        self.active_mode = mode
        self.active_command = gesture_name
        self.command_done = False
        self.command_success = False
        self.command_message = ""
        self.busy = True

        if mode == 'object' and gesture_name == 'grasp':
            self.grasp_locked = False
            self.reset_contact_counters()

        if mode == 'object' and gesture_name == 'open':
            self.grasp_locked = False

    def handle_hand_command(self, request, response):
        mode = request.mode.strip().lower()
        gesture_name = request.gesture_name.strip().lower()

        with self.state_lock:
            if self.busy:
                response.success = False
                response.message = 'Hand is busy'
                return response

            if mode == 'object':
                if gesture_name not in ['open', 'grasp']:
                    response.success = False
                    response.message = f'Unsupported object command: {gesture_name}'
                    return response

            elif mode == 'gesture':
                if gesture_name not in self.gesture_map:
                    response.success = False
                    response.message = f'Unknown gesture: {gesture_name}'
                    return response

            else:
                response.success = False
                response.message = f'Unsupported mode: {mode}'
                return response

            self.start_new_command(mode, gesture_name)

        start = time.time()
        timeout_sec = 30.0

        while rclpy.ok():
            with self.state_lock:
                if self.command_done:
                    response.success = self.command_success
                    response.message = self.command_message
                    return response

            if time.time() - start > timeout_sec:
                with self.state_lock:
                    self.finish_command(False, 'Command timed out')
                    response.success = self.command_success
                    response.message = self.command_message
                return response

            time.sleep(0.05)

        response.success = False
        response.message = 'ROS shutdown while waiting for command'
        return response

    def run_object_open(self):
        reached = self.ramp_command_towards(self.pose_fully_open, self.timer_period)
        if reached:
            time.sleep(3.0)
            self.finish_command(True, 'Hand opened')

    def run_named_gesture(self, gesture_name: str):
        target = self.gesture_map[gesture_name]['positions']
        reached = self.ramp_command_towards(target, self.timer_period)
        if reached:
            self.finish_command(True, f'Gesture {gesture_name} completed')

    def run_object_grasp(self):

        target = self.pose_fully_closed

        thumb_contact = self.is_grasp_contact_stable('thumb')
        index_contact = self.is_grasp_contact_stable('index')
        middle_contact = self.is_grasp_contact_stable('middle')
        ring_contact = self.is_grasp_contact_stable('ring')
        small_contact = self.is_grasp_contact_stable('small')

        freeze_mask = [
            False,                 # forearm
            False,                 # palm_axis
            False,                 # palm_l
            False,                 # thumb_axis
            thumb_contact,         # thumb metacarpal
            index_contact,         # index
            middle_contact,        # middle
            (ring_contact or small_contact)  # ring/small combined
        ]

        thumb_tip_mag = self.get_tip_force_mag('thumb_tip_force')
        index_tip_mag = self.get_tip_force_mag('index_tip_force')
        middle_tip_mag = self.get_tip_force_mag('middle_tip_force')
        ring_tip_mag = self.get_tip_force_mag('ring_tip_force')
        small_tip_mag = self.get_tip_force_mag('small_tip_force')

        palm_range = self.get_interface_value('palm_range', 'value')
        palm_range = -1.0 if palm_range is None else palm_range

        self.get_logger().info(
            f"Grasp status | "
            f"thumb={thumb_contact}, index={index_contact}, middle={middle_contact}, "
            f"ring={ring_contact}, small={small_contact}, palm_range={palm_range:.4f} | "
            f"forces thumb={thumb_tip_mag:.3f}, index={index_tip_mag:.3f}, "
            f"middle={middle_tip_mag:.3f}, ring={ring_tip_mag:.3f}, small={small_tip_mag:.3f}"
        )

        enough_contact = (
            thumb_contact and
            index_contact and
            middle_contact and
            (ring_contact or small_contact)
        )

        if enough_contact and not self.grasp_locked:
            self.latch_current_grasp()

        if self.grasp_locked:
            latched_target = [
                0.0,
                0.0,
                0.0,
                self.thumb_axis_open,
                self.latched_positions['thumb_meta'],
                self.latched_positions['index'],
                self.latched_positions['middle'],
                self.latched_positions['ring']
            ]
            freeze_mask = [False, False, False, False, True, True, True, True]
            self.ramp_command_towards(latched_target, self.timer_period, freeze_mask=freeze_mask)
            self.finish_command(True, 'Object grasp completed with contact lock')
            return

        self.ramp_command_towards(target, self.timer_period, freeze_mask=freeze_mask)

    def step(self):
        with self.state_lock:
            if not self.have_state:
                return

            if not self.busy:
                return

            active_mode = self.active_mode
            active_command = self.active_command

        if active_mode == 'object':
            if active_command == 'open':
                with self.state_lock:
                    self.run_object_open()
            elif active_command == 'grasp':
                with self.state_lock:
                    if self.object_present():
                        self.get_logger().info('Object detected in grasp range')
                        self.last_object_seen_time = self.get_clock().now()
                        self.run_object_grasp()
                    # else:
                    #     # If we've never seen the object, or it's been gone for > 1 second
                    #     now = self.get_clock().now()
                    #     if self.last_object_seen_time is None:
                    #         # Optional: Allow a small window at start to find the object
                    #         self.finish_command(False, 'No object detected at start')
                    #     elif (now - self.last_object_seen_time).nanoseconds > 1e9: # 1 second
                    #         self.finish_command(False, 'Object lost during grasp')
                    else:
                        # Object flickered out, but keep trying for a moment
                        self.run_object_grasp()

        elif active_mode == 'gesture':
                with self.state_lock:
                    self.run_named_gesture(active_command)


def main(args=None):
    rclpy.init(args=args)
    node = RH8DHandServiceController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            executor.remove_node(node)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()