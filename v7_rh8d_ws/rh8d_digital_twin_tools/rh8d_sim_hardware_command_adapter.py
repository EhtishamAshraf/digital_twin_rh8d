#!/usr/bin/env python3

import math
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


def clamp_mapping(value, in_min, in_max, out_min, out_max):
    if value < in_min:
        return out_min
    if value > in_max:
        return out_max

    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def wrist_rotation_ticks_to_rad(tick):
    return clamp_mapping(tick, 0.0, 4095.0, -math.pi / 2.0, math.pi / 2.0)


def wrist_adduction_ticks_to_rad(tick):
    # Hardware broadcaster maps wrist adduction to -pi/4 to +pi/4.
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
    # Exact hardware broadcaster behavior:
    # pip_joint_thumb is stage 1, mcp_joint_thumb is stage 2, dip is stage 3.
    thumb_methacarpal_mcp = clamp_mapping(tick, 2000.0, 3800.0, 0.0, math.pi / 2.0)
    thumb_proximal_pip = clamp_mapping(tick, 0.0, 2000.0, 0.0, math.pi / 2.0)
    thumb_distal_dip = clamp_mapping(tick, 3800.0, 4095.0, 0.0, math.pi / 4.0)
    return thumb_methacarpal_mcp, thumb_proximal_pip, thumb_distal_dip


class RH8DSimHardwareCommandAdapter(Node):
    def __init__(self):
        super().__init__("rh8d_sim_hardware_command_adapter")

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

        self.get_logger().info("RH8D staged sim hardware command adapter started.")
        self.get_logger().info("Subscribing: /rh8dr/hand_position_controller/commands")
        self.get_logger().info("Publishing:  /hand_trajectory_controller/joint_trajectory")

    def command_callback(self, msg):
        if len(msg.data) < 8:
            self.get_logger().warn(
                f"Expected 8 hardware command values, got {len(msg.data)}. Ignoring."
            )
            return

        hw = dict(zip(HARDWARE_ORDER, [float(v) for v in msg.data[:8]]))

        thumb_mcp, thumb_pip, thumb_dip = thumb_finger_stages(hw["thumb_flexion_motor"])
        index_prox, index_mid, index_dist = normal_finger_stages(hw["index_flexion_motor"])
        middle_prox, middle_mid, middle_dist = normal_finger_stages(hw["middle_flexion_motor"])
        ring_prox, ring_mid, ring_dist = normal_finger_stages(hw["ring_small_flexion_motor"])

        # Ring and small are driven by the same real motor.
        small_prox, small_mid, small_dist = ring_prox, ring_mid, ring_dist

        sim_positions = [
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

        traj = JointTrajectory()
        traj.joint_names = list(SIM_JOINT_ORDER)

        point = JointTrajectoryPoint()
        point.positions = sim_positions
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0

        traj.points.append(point)
        self.pub.publish(traj)

        self.get_logger().info(
            "Published staged sim command from hardware ticks. "
            f"index=({index_prox:.3f},{index_mid:.3f},{index_dist:.3f}), "
            f"middle=({middle_prox:.3f},{middle_mid:.3f},{middle_dist:.3f}), "
            f"ring/small=({ring_prox:.3f},{ring_mid:.3f},{ring_dist:.3f}), "
            f"thumb=({thumb_mcp:.3f},{thumb_pip:.3f},{thumb_dip:.3f})"
        )


def main():
    rclpy.init()
    node = RH8DSimHardwareCommandAdapter()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
