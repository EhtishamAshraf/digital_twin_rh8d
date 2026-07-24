#!/usr/bin/env python3

from pathlib import Path
import shutil
import re


WS = Path.home() / "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/digital_twin/v7_rh8d_ws"

XACRO = WS / "src/rh8d_mujoco_control/description/rh8dR_mujoco_ros2_control.xacro"
MJCF = WS / "src/ur5e_mujoco_description/mjcf/ur5e_rh8d.xml"
ADAPTER = WS / "rh8d_digital_twin_tools/rh8d_sim_hardware_command_adapter.py"

BACKUP_DIR = WS / "rh8d_digital_twin_backups_step2"
BACKUP_DIR.mkdir(exist_ok=True)


def backup(path: Path):
    dst = BACKUP_DIR / (path.name + ".bak")
    shutil.copy2(path, dst)
    print(f"backup: {path} -> {dst}")


def replace_between(text, start_marker, end_marker, replacement):
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + replacement + text[end:]


# ---------------------------------------------------------------------
# 1. Update ros2_control xacro hand command interfaces
# ---------------------------------------------------------------------

backup(XACRO)
xacro_text = XACRO.read_text()

xacro_joint_block = """      <joint name="forearm:1--base:1">
        <param name="mujoco_actuator">pos_forearm</param>
        <command_interface name="position" min="-1.5708" max="1.5708"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="palm_axis:1--forearm:1">
        <param name="mujoco_actuator">pos_palm_axis</param>
        <command_interface name="position" min="-1.5708" max="1.5708"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="palm_assembly:1--palm_axis:1">
        <param name="mujoco_actuator">pos_palm_assembly</param>
        <command_interface name="position" min="-0.785398" max="0.785398"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Thumb_axis--palm_assembly:1">
        <param name="mujoco_actuator">pos_thumb_axis</param>
        <command_interface name="position" min="0.0" max="1.5708"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Thumb_Methacarpal--Thumb_axis">
        <param name="mujoco_actuator">pos_thumb_tendon</param>
        <command_interface name="position" min="0.0" max="1.5708"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Thumb_Proximal--Thumb_Methacarpal">
        <param name="mujoco_actuator">pos_thumb_proximal</param>
        <command_interface name="position" min="0.0" max="1.5708"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Thumb_Distal--Thumb_Proximal">
        <param name="mujoco_actuator">pos_thumb_distal</param>
        <command_interface name="position" min="0.0" max="0.785398"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Index_Proximal--palm_assembly:1">
        <param name="mujoco_actuator">pos_index_tendon</param>
        <command_interface name="position" min="0.0" max="1.5708"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Index_Middle--Index_Proximal">
        <param name="mujoco_actuator">pos_index_middle</param>
        <command_interface name="position" min="0.0" max="1.5708"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Index_Distal--Index_Middle">
        <param name="mujoco_actuator">pos_index_distal</param>
        <command_interface name="position" min="0.0" max="0.785398"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Middle_Proximal--palm_assembly:1">
        <param name="mujoco_actuator">pos_middle_tendon</param>
        <command_interface name="position" min="0.0" max="1.5708"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Middle_Middle--Middle_Proximal">
        <param name="mujoco_actuator">pos_middle_middle</param>
        <command_interface name="position" min="0.0" max="1.5708"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Middle_Distal--Middle_Middle">
        <param name="mujoco_actuator">pos_middle_distal</param>
        <command_interface name="position" min="0.0" max="0.785398"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Ring_Proximal--palm_assembly:1">
        <param name="mujoco_actuator">pos_ring_small_tendon</param>
        <command_interface name="position" min="0.0" max="1.5708"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Ring_Middle--Ring_Proximal">
        <param name="mujoco_actuator">pos_ring_middle</param>
        <command_interface name="position" min="0.0" max="1.5708"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Ring_Distal--Ring_Middle">
        <param name="mujoco_actuator">pos_ring_distal</param>
        <command_interface name="position" min="0.0" max="0.785398"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Small_Proximal--palm_assembly:1">
        <param name="mujoco_actuator">pos_small_proximal</param>
        <command_interface name="position" min="0.0" max="1.5708"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Small_Middle--Small_Proximal">
        <param name="mujoco_actuator">pos_small_middle</param>
        <command_interface name="position" min="0.0" max="1.5708"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

      <joint name="Small_Distal--Small_Middle">
        <param name="mujoco_actuator">pos_small_distal</param>
        <command_interface name="position" min="0.0" max="0.785398"/>
        <state_interface name="position"/>
        <state_interface name="effort"/>
      </joint>

"""

xacro_text = replace_between(
    xacro_text,
    '      <joint name="forearm:1--base:1">',
    '      <sensor name="palm_range">',
    xacro_joint_block,
)
XACRO.write_text(xacro_text)
print(f"updated: {XACRO}")


# ---------------------------------------------------------------------
# 2. Replace old tendon finger actuators with direct joint actuators
# ---------------------------------------------------------------------

backup(MJCF)
mjcf_text = MJCF.read_text()

direct_finger_actuators = """
    <!-- RH8D staged digital-twin direct joint actuators.
         The old tendon actuator names are kept for proximal joints so existing
         ros2_control mappings and actuator force sensors remain valid. -->

    <position name="pos_index_tendon" joint="Index_Proximal--palm_assembly:1"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 1.5708"/>
    <position name="pos_index_middle" joint="Index_Middle--Index_Proximal"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 1.5708"/>
    <position name="pos_index_distal" joint="Index_Distal--Index_Middle"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 0.785398"/>

    <position name="pos_middle_tendon" joint="Middle_Proximal--palm_assembly:1"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 1.5708"/>
    <position name="pos_middle_middle" joint="Middle_Middle--Middle_Proximal"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 1.5708"/>
    <position name="pos_middle_distal" joint="Middle_Distal--Middle_Middle"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 0.785398"/>

    <position name="pos_ring_small_tendon" joint="Ring_Proximal--palm_assembly:1"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 1.5708"/>
    <position name="pos_ring_middle" joint="Ring_Middle--Ring_Proximal"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 1.5708"/>
    <position name="pos_ring_distal" joint="Ring_Distal--Ring_Middle"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 0.785398"/>

    <position name="pos_small_proximal" joint="Small_Proximal--palm_assembly:1"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 1.5708"/>
    <position name="pos_small_middle" joint="Small_Middle--Small_Proximal"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 1.5708"/>
    <position name="pos_small_distal" joint="Small_Distal--Small_Middle"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 0.785398"/>

    <position name="pos_thumb_tendon" joint="Thumb_Methacarpal--Thumb_axis"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 1.5708"/>
    <position name="pos_thumb_proximal" joint="Thumb_Proximal--Thumb_Methacarpal"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 1.5708"/>
    <position name="pos_thumb_distal" joint="Thumb_Distal--Thumb_Proximal"
              kp="5" kv="1" forcelimited="true" forcerange="-5 5"
              ctrllimited="true" ctrlrange="0 0.785398"/>
"""

pattern = re.compile(
    r'\n\s*<position name="pos_index_tendon" tendon="idx_flex_spatial"[\s\S]*?'
    r'\n\s*<position name="pos_thumb_tendon" tendon="th_flex_spatial"[\s\S]*?/>',
    re.MULTILINE,
)

mjcf_text_new, count = pattern.subn("\n" + direct_finger_actuators, mjcf_text, count=1)

if count != 1:
    raise RuntimeError("Could not replace old tendon actuator block. No file written.")

MJCF.write_text(mjcf_text_new)
print(f"updated: {MJCF}")


# ---------------------------------------------------------------------
# 3. Replace adapter with 19-joint staged adapter
# ---------------------------------------------------------------------

backup(ADAPTER)

adapter_text = r'''#!/usr/bin/env python3

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
'''
ADAPTER.write_text(adapter_text)
ADAPTER.chmod(0o755)
print(f"updated: {ADAPTER}")

print("\nStep 2 staged sim edits applied.")
