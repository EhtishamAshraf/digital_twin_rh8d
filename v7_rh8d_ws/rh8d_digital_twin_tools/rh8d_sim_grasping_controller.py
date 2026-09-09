#!/usr/bin/env python3

import os
from enum import Enum

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Float64MultiArray, String
from hand_sensors_interfaces.msg import FTS3Sensors, SingleCommand


MOTOR_ORDER = [
    "forearm:1--base:1",
    "palm_axis:1--forearm:1",
    "palm_assembly:1--palm_axis:1",
    "thumb_adduction_motor",
    "thumb_flexion_motor",
    "index_flexion_motor",
    "middle_flexion_motor",
    "ring_small_flexion_motor",
]

DEFAULT_OPEN_COMMAND = [
    2052.0,
    2010.0,
    2191.0,
    3971.0,
    100.0,
    100.0,
    100.0,
    0.0,
]

MIN_COMMAND = [
    0.0,
    0.0,
    0.0,
    0.0,
    100.0,   # thumb flexion open/home lower bound
    100.0,   # index flexion open/home lower bound
    100.0,   # middle flexion open/home lower bound
    0.0,     # ring/small open lower bound
]

MAX_COMMAND = [
    4095.0,
    4095.0,
    4095.0,
    4095.0,
    4000.0,
    4000.0,
    4000.0,
    4000.0,
]

CONTROL_FINGERS = ["thumb", "index", "middle", "ring"]

FINGER_TO_MOTOR_INDEX = {
    "thumb": 4,
    "index": 5,
    "middle": 6,
    "ring": 7,
}

FINGER_TO_MOTOR_NAME = {
    "thumb": "thumb_flexion_motor",
    "index": "index_flexion_motor",
    "middle": "middle_flexion_motor",
    "ring": "ring_small_flexion_motor",
}

F_DES_TOPICS = {
    "thumb": "/rh8dr/thumb_force_sensor/f_des",
    "index": "/rh8dr/index_force_sensor/f_des",
    "middle": "/rh8dr/middle_force_sensor/f_des",
    "ring": "/rh8dr/ring_force_sensor/f_des",
}

ERROR_TOPICS = {
    "thumb": "/rh8dr/thumb_force_sensor/error",
    "index": "/rh8dr/index_force_sensor/error",
    "middle": "/rh8dr/middle_force_sensor/error",
    "ring": "/rh8dr/ring_force_sensor/error",
}

INTEGRATOR_INPUT_TOPICS = {
    "thumb": "/rh8dr/thumb_flexion_motor/integrator_input",
    "index": "/rh8dr/index_flexion_motor/integrator_input",
    "middle": "/rh8dr/middle_flexion_motor/integrator_input",
    "ring": "/rh8dr/ring_small_flexion_motor/integrator_input",
}


class State(Enum):
    IDLE = "IDLE"
    CLOSING = "CLOSING"
    FORCE_CONTROL = "FORCE_CONTROL"


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class RH8DSimGraspingController(Node):
    def __init__(self):
        super().__init__("rh8d_sim_grasping_controller")

        self.control_hz = float(os.environ.get("RH8D_SIM_GRASP_HZ", "50.0"))
        self.dt = 1.0 / self.control_hz

        # Hardware-matched values.
        self.kp = float(os.environ.get("RH8D_SIM_GRASP_KP", "600.0"))
        self.constant_velocity = float(
            os.environ.get("RH8D_SIM_GRASP_CONSTANT_VELOCITY", "400.0")
        )

        self.default_desired_force = float(
            os.environ.get("RH8D_SIM_GRASP_DESIRED_FORCE", "3.0")
        )

        # Contact/state-machine parameters.
        self.contact_threshold = float(
            os.environ.get("RH8D_SIM_GRASP_CONTACT_THRESHOLD", "1.0")
        )

        self.min_closing_time = float(
            os.environ.get("RH8D_SIM_GRASP_MIN_CLOSING_TIME", "0.7")
        )

        self.two_finger_timeout = float(
            os.environ.get("RH8D_SIM_GRASP_TWO_FINGER_TIMEOUT", "2.0")
        )

        self.no_contact_timeout = float(
            os.environ.get("RH8D_SIM_GRASP_NO_CONTACT_TIMEOUT", "8.0")
        )

        self.state = State.IDLE
        self.state_start_time = self.get_clock().now()

        self.motor_command = list(DEFAULT_OPEN_COMMAND)

        self.force = {
            "thumb": 0.0,
            "index": 0.0,
            "middle": 0.0,
            "ring": 0.0,
            "small": 0.0,
        }

        self.filtered_force = dict(self.force)

        self.desired_force = {
            "thumb": self.default_desired_force,
            "index": self.default_desired_force,
            "middle": self.default_desired_force,
            "ring": self.default_desired_force,
        }

        self.active_fingers = []

        self.command_pub = self.create_publisher(
            Float64MultiArray,
            "/rh8dr/hand_position_controller/commands",
            10,
        )

        self.state_pub = self.create_publisher(
            String,
            "/rh8dr/sim_grasping_controller/state",
            10,
        )

        self.error_pubs = {
            finger: self.create_publisher(SingleCommand, topic, 10)
            for finger, topic in ERROR_TOPICS.items()
        }

        self.integrator_input_pubs = {
            finger: self.create_publisher(SingleCommand, topic, 10)
            for finger, topic in INTEGRATOR_INPUT_TOPICS.items()
        }

        self.create_subscription(
            FTS3Sensors,
            "/rh8dr/fts_broadcaster/fingertip_force_sensors",
            self.force_callback,
            10,
        )

        self.create_subscription(
            Bool,
            "/rh8dr/sim_grasping_controller/enable",
            self.enable_callback,
            10,
        )

        self.create_subscription(
            Bool,
            "/rh8dr/sim_grasping_controller/open",
            self.open_callback,
            10,
        )

        for finger, topic in F_DES_TOPICS.items():
            self.create_subscription(
                SingleCommand,
                topic,
                lambda msg, f=finger: self.f_des_callback(f, msg),
                10,
            )

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.loop_count = 0

        self.get_logger().info("RH8D sim grasping controller v2 started.")
        self.get_logger().info("State machine: IDLE -> CLOSING -> FORCE_CONTROL")
        self.get_logger().info(f"Kp = {self.kp}")
        self.get_logger().info(f"constant_velocity = {self.constant_velocity}")
        self.get_logger().info(f"default desired force = {self.default_desired_force} N")
        self.get_logger().info(f"contact_threshold = {self.contact_threshold} N")

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def state_age(self):
        return self.now_sec() - self.state_start_time.nanoseconds * 1e-9

    def set_state(self, new_state):
        if self.state == new_state:
            return

        old = self.state
        self.state = new_state
        self.state_start_time = self.get_clock().now()

        msg = String()
        msg.data = self.state.value
        self.state_pub.publish(msg)

        self.get_logger().info(f"STATE {old.value} -> {new_state.value}")

    def enable_callback(self, msg):
        if bool(msg.data):
            self.active_fingers = []
            self.set_state(State.CLOSING)
            self.get_logger().info("Grasping enabled: entering CLOSING.")
        else:
            self.active_fingers = []
            self.set_state(State.IDLE)
            self.get_logger().info("Grasping disabled.")

    def open_callback(self, msg):
        if bool(msg.data):
            self.active_fingers = []
            self.motor_command = list(DEFAULT_OPEN_COMMAND)
            self.set_state(State.IDLE)
            self.publish_motor_command()
            self.get_logger().info("Opening hand and returning to IDLE.")

    def f_des_callback(self, finger, msg):
        self.desired_force[finger] = float(msg.value)
        self.get_logger().info(
            f"desired force updated: {finger} = {self.desired_force[finger]:.2f} N"
        )

    def force_callback(self, msg):
        raw = {}
        for name, norm in zip(msg.motor_names, msg.norm_forces):
            raw[name] = float(norm)

        ring_small_force = max(raw.get("ring", 0.0), raw.get("small", 0.0))

        raw["ring"] = ring_small_force
        raw["small"] = ring_small_force

        alpha = 0.12
        for finger in self.filtered_force:
            self.force[finger] = raw.get(finger, 0.0)
            self.filtered_force[finger] = (
                (1.0 - alpha) * self.filtered_force[finger]
                + alpha * self.force[finger]
            )

    def contacts(self):
        return [
            finger for finger in CONTROL_FINGERS
            if self.filtered_force[finger] >= self.contact_threshold
        ]

    def enough_contacts_for_force_control(self):
        contacts = self.contacts()
        age = self.state_age()

        thumb_in_contact = "thumb" in contacts
        contact_count = len(contacts)

        # Main hardware-like condition:
        # accept force control when thumb is involved and at least 3 total fingers are in contact.
        if age >= self.min_closing_time and thumb_in_contact and contact_count >= 3:
            self.active_fingers = contacts
            return True

        # Fallback behavior:
        # after a timeout, allow force control with at least 2 contacted fingers.
        if age >= self.two_finger_timeout and contact_count >= 2:
            self.active_fingers = contacts
            return True

        # Safety fallback:
        # if no contact happens for too long, stop closing and regulate all fingers.
        if age >= self.no_contact_timeout:
            self.active_fingers = list(CONTROL_FINGERS)
            return True

        return False

    def publish_single_command(self, pub, dof_name, value):
        msg = SingleCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.dof_name = dof_name
        msg.value = float(value)
        pub.publish(msg)

    def publish_motor_command(self):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in self.motor_command]
        self.command_pub.publish(msg)

    def apply_integrator_command(self, finger, command):
        motor_idx = FINGER_TO_MOTOR_INDEX[finger]

        self.motor_command[motor_idx] += command * self.dt
        self.motor_command[motor_idx] = clamp(
            self.motor_command[motor_idx],
            MIN_COMMAND[motor_idx],
            MAX_COMMAND[motor_idx],
        )

    def publish_integrator_debug(self, finger, command, force_error):
        self.publish_single_command(
            self.error_pubs[finger],
            f"{finger}_force_sensor",
            force_error,
        )

        self.publish_single_command(
            self.integrator_input_pubs[finger],
            FINGER_TO_MOTOR_NAME[finger],
            command,
        )

    def closing_step(self):
        command = self.constant_velocity

        for finger in CONTROL_FINGERS:
            self.apply_integrator_command(finger, command)
            self.publish_integrator_debug(finger, command, 0.0)

        self.publish_motor_command()

        if self.enough_contacts_for_force_control():
            self.get_logger().info(
                "Entering FORCE_CONTROL with active fingers: "
                + ", ".join(self.active_fingers)
            )
            self.set_state(State.FORCE_CONTROL)

    def force_control_step(self):
        for finger in self.active_fingers:
            measured_force = self.filtered_force[finger]
            desired_force = self.desired_force[finger]

            force_error = desired_force - measured_force

            command = self.kp * force_error
            command = clamp(
                command,
                -self.constant_velocity,
                self.constant_velocity,
            )

            self.apply_integrator_command(finger, command)
            self.publish_integrator_debug(finger, command, force_error)

        self.publish_motor_command()

    def control_loop(self):
        if self.state == State.CLOSING:
            self.closing_step()

        elif self.state == State.FORCE_CONTROL:
            self.force_control_step()

        self.loop_count += 1

        if self.loop_count % int(self.control_hz) == 0:
            self.get_logger().info(
                f"state={self.state.value}; "
                "force "
                f"T={self.filtered_force['thumb']:.2f}, "
                f"I={self.filtered_force['index']:.2f}, "
                f"M={self.filtered_force['middle']:.2f}, "
                f"R/S={self.filtered_force['ring']:.2f}; "
                "contacts="
                f"{','.join(self.contacts()) if self.contacts() else 'none'}; "
                "active="
                f"{','.join(self.active_fingers) if self.active_fingers else 'none'}; "
                "ticks "
                f"T={self.motor_command[4]:.0f}, "
                f"I={self.motor_command[5]:.0f}, "
                f"M={self.motor_command[6]:.0f}, "
                f"R/S={self.motor_command[7]:.0f}"
            )


def main():
    rclpy.init()
    node = RH8DSimGraspingController()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
