#!/usr/bin/env python3

import math
import os

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


HARDWARE_ORDER = [
    "forearm:1--base:1",
    "palm_axis:1--forearm:1",
    "palm_assembly:1--palm_axis:1",
    "thumb_adduction_motor",
    "thumb_flexion_motor",
    "index_flexion_motor",
    "middle_flexion_motor",
    "ring_small_flexion_motor",
]

SIM_JOINT_ORDER = [
    "forearm:1--base:1",
    "palm_axis:1--forearm:1",
    "palm_assembly:1--palm_axis:1",
    "Thumb_axis--palm_assembly:1",

    "Thumb_Methacarpal--Thumb_axis",
    "Thumb_Proximal--Thumb_Methacarpal",
    "Thumb_Distal--Thumb_Proximal",

    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",

    "Middle_Proximal--palm_assembly:1",
    "Middle_Middle--Middle_Proximal",
    "Middle_Distal--Middle_Middle",

    "Ring_Proximal--palm_assembly:1",
    "Ring_Middle--Ring_Proximal",
    "Ring_Distal--Ring_Middle",

    "Small_Proximal--palm_assembly:1",
    "Small_Middle--Small_Proximal",
    "Small_Distal--Small_Middle",
]

DEFAULT_OPEN_HW = [
    2052.0,
    2010.0,
    2191.0,
    3971.0,
    100.0,
    100.0,
    100.0,
    0.0,
]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def move_towards(current, target, max_step):
    if current < target:
        return min(current + max_step, target)
    if current > target:
        return max(current - max_step, target)
    return current


def clamp_mapping(value, in_min, in_max, out_min, out_max):
    if value < in_min:
        return out_min
    if value > in_max:
        return out_max
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def wrist_rotation_ticks_to_rad(tick):
    return clamp_mapping(tick, 0.0, 4095.0, -math.pi / 2.0, math.pi / 2.0)


def wrist_adduction_ticks_to_rad(tick):
    return clamp_mapping(tick, 0.0, 4095.0, -math.pi / 4.0, math.pi / 4.0)


def wrist_flexion_ticks_to_rad(tick):
    return clamp_mapping(tick, 0.0, 4095.0, -math.pi / 4.0, math.pi / 4.0)


def thumb_adduction_ticks_to_rad(tick):
    return clamp_mapping(tick, 0.0, 4095.0, 0.0, math.pi / 2.0)


def normal_finger_stages(tick):
    proximal = clamp_mapping(tick, 0.0, 2000.0, 0.0, math.pi / 2.0)
    middle = clamp_mapping(tick, 2000.0, 3800.0, 0.0, math.pi / 2.0)
    distal = clamp_mapping(tick, 3800.0, 4095.0, 0.0, math.pi / 4.0)
    return proximal, middle, distal


def thumb_finger_stages(tick):
    # Matches the hardware joint-state broadcaster representation.
    # Thumb PIP closes first, thumb MCP second, thumb distal third.
    thumb_mcp = clamp_mapping(tick, 2000.0, 3800.0, 0.0, math.pi / 2.0)
    thumb_pip = clamp_mapping(tick, 0.0, 2000.0, 0.0, math.pi / 2.0)
    thumb_dip = clamp_mapping(tick, 3800.0, 4095.0, 0.0, math.pi / 4.0)
    return thumb_mcp, thumb_pip, thumb_dip


def hardware_ticks_to_sim_positions(hw_values):
    hw = dict(zip(HARDWARE_ORDER, hw_values))

    thumb_mcp, thumb_pip, thumb_dip = thumb_finger_stages(hw["thumb_flexion_motor"])

    index_prox, index_mid, index_dist = normal_finger_stages(hw["index_flexion_motor"])
    middle_prox, middle_mid, middle_dist = normal_finger_stages(hw["middle_flexion_motor"])
    ring_prox, ring_mid, ring_dist = normal_finger_stages(hw["ring_small_flexion_motor"])

    # Real RH8D has one ring/small flexion motor.
    small_prox, small_mid, small_dist = ring_prox, ring_mid, ring_dist

    return [
        wrist_rotation_ticks_to_rad(hw["forearm:1--base:1"]),
        wrist_adduction_ticks_to_rad(hw["palm_axis:1--forearm:1"]),
        wrist_flexion_ticks_to_rad(hw["palm_assembly:1--palm_axis:1"]),
        thumb_adduction_ticks_to_rad(hw["thumb_adduction_motor"]),

        thumb_mcp,
        thumb_pip,
        thumb_dip,

        index_prox,
        index_mid,
        index_dist,

        middle_prox,
        middle_mid,
        middle_dist,

        ring_prox,
        ring_mid,
        ring_dist,

        small_prox,
        small_mid,
        small_dist,
    ]


class RH8DSimRLMotorAdapter(Node):
    def __init__(self):
        super().__init__("rh8d_sim_rl_motor_adapter")

        self.control_hz = float(os.environ.get("RH8D_RL_ADAPTER_HZ", "20.0"))
        self.dt = 1.0 / self.control_hz

        # Motor speed limit. This is allowed because real motors also cannot jump instantly.
        self.max_speed_ticks_per_sec = float(
            os.environ.get("RH8D_RL_MAX_SPEED_TICKS_PER_SEC", "550.0")
        )

        self.desired_hw = list(DEFAULT_OPEN_HW)
        self.current_hw = list(DEFAULT_OPEN_HW)

        self.sub = self.create_subscription(
            Float64MultiArray,
            "/rh8dr/hand_position_controller/commands",
            self.command_callback,
            10,
        )

        self.pub = self.create_publisher(
            JointTrajectory,
            "/hand_trajectory_controller/joint_trajectory",
            10,
        )

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.loop_count = 0

        self.get_logger().info("RH8D clean RL motor adapter started.")
        self.get_logger().info("No force latch, no hidden hold, no sim-only grasp intelligence.")
        self.get_logger().info("Input:  /rh8dr/hand_position_controller/commands")
        self.get_logger().info("Output: /hand_trajectory_controller/joint_trajectory")
        self.get_logger().info(
            f"Motor speed limit: {self.max_speed_ticks_per_sec:.1f} ticks/sec"
        )

    def command_callback(self, msg):
        if len(msg.data) < 8:
            self.get_logger().warn(f"Expected 8 values, got {len(msg.data)}. Ignoring.")
            return

        raw = [float(v) for v in msg.data[:8]]

        self.desired_hw = [
            clamp(raw[0], 0.0, 4095.0),
            clamp(raw[1], 0.0, 4095.0),
            clamp(raw[2], 0.0, 4095.0),
            clamp(raw[3], 0.0, 4095.0),
            clamp(raw[4], 0.0, 4000.0),
            clamp(raw[5], 0.0, 4000.0),
            clamp(raw[6], 0.0, 4000.0),
            clamp(raw[7], 0.0, 4000.0),
        ]

        self.get_logger().info(
            "New RL motor command: "
            f"T={self.desired_hw[4]:.0f}, "
            f"I={self.desired_hw[5]:.0f}, "
            f"M={self.desired_hw[6]:.0f}, "
            f"R/S={self.desired_hw[7]:.0f}"
        )

    def update_motor_ramp(self):
        max_step = self.max_speed_ticks_per_sec * self.dt

        for i in range(8):
            self.current_hw[i] = move_towards(
                self.current_hw[i],
                self.desired_hw[i],
                max_step,
            )

    def publish_sim_command(self):
        sim_positions = hardware_ticks_to_sim_positions(self.current_hw)

        traj = JointTrajectory()
        traj.joint_names = list(SIM_JOINT_ORDER)

        point = JointTrajectoryPoint()
        point.positions = sim_positions
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 120_000_000

        traj.points.append(point)
        self.pub.publish(traj)

    def control_loop(self):
        self.update_motor_ramp()
        self.publish_sim_command()

        self.loop_count += 1

        if self.loop_count % int(self.control_hz) == 0:
            self.get_logger().info(
                "current ticks "
                f"T={self.current_hw[4]:.0f}, "
                f"I={self.current_hw[5]:.0f}, "
                f"M={self.current_hw[6]:.0f}, "
                f"R/S={self.current_hw[7]:.0f}"
            )


def main():
    rclpy.init()
    node = RH8DSimRLMotorAdapter()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
