#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import Vector3

from hand_sensors_interfaces.msg import FTS3Sensors, PalmSensor


FINGERS = [
    ("thumb", "thumb_tip_force"),
    ("index", "index_tip_force"),
    ("middle", "middle_tip_force"),
    ("ring", "ring_tip_force"),
    ("small", "small_tip_force"),
]


class RH8DSimSensorBridge(Node):
    def __init__(self):
        super().__init__("rh8d_sim_sensor_bridge")

        # Hardware FTS topic was reliable/transient_local, so we mimic that.
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.sub = self.create_subscription(
            DynamicJointState,
            "/dynamic_joint_states",
            self.dynamic_joint_states_callback,
            10,
        )

        self.fts_pub = self.create_publisher(
            FTS3Sensors,
            "/rh8dr/fts_broadcaster/fingertip_force_sensors",
            sensor_qos,
        )

        self.palm_pub = self.create_publisher(
            PalmSensor,
            "/rh8dr/tof_broadcaster/distance_measured",
            sensor_qos,
        )

        self.get_logger().info("RH8D sim sensor bridge started.")
        self.get_logger().info("Subscribing: /dynamic_joint_states")
        self.get_logger().info("Publishing:  /rh8dr/fts_broadcaster/fingertip_force_sensors")
        self.get_logger().info("Publishing:  /rh8dr/tof_broadcaster/distance_measured")

    def dynamic_joint_states_callback(self, msg):
        data = {}

        for name, interface_value in zip(msg.joint_names, msg.interface_values):
            data[name] = dict(zip(interface_value.interface_names, interface_value.values))

        self.publish_fts(msg, data)
        self.publish_palm(msg, data)

    def publish_fts(self, dyn_msg, data):
        fts_msg = FTS3Sensors()
        fts_msg.header = dyn_msg.header

        forces = []
        norms = []
        motor_names = []
        motor_ids = []

        for idx, (finger_name, sim_sensor_name) in enumerate(FINGERS):
            interfaces = data.get(sim_sensor_name, {})

            fx = float(interfaces.get("x", 0.0))
            fy = float(interfaces.get("y", 0.0))
            fz = float(interfaces.get("z", 0.0))

            vector = Vector3()
            vector.x = fx
            vector.y = fy
            vector.z = fz

            forces.append(vector)
            norms.append(math.sqrt(fx * fx + fy * fy + fz * fz))

            # Dashboard/RL should use these stable finger names.
            motor_names.append(finger_name)

            # Placeholder IDs for sim. Names are more important for compatibility.
            motor_ids.append(idx)

        fts_msg.forces = forces
        fts_msg.norm_forces = norms
        fts_msg.motor_names = motor_names
        fts_msg.motor_ids = motor_ids

        self.fts_pub.publish(fts_msg)

    def publish_palm(self, dyn_msg, data):
        palm_msg = PalmSensor()
        palm_msg.header = dyn_msg.header

        palm_range_interfaces = data.get("palm_range", {})
        palm_value = float(palm_range_interfaces.get("value", -1.0))

        # MuJoCo range sensors usually report meters and -1 when no object is detected.
        # Hardware palm sensor values are distance-like numeric values; your hardware sample
        # showed about 254.0 for open/no-object condition.
        if palm_value < 0.0:
            palm_msg.distance = 254.0
        else:
            palm_msg.distance = palm_value * 1000.0  # meters -> mm

        self.palm_pub.publish(palm_msg)


def main():
    rclpy.init()
    node = RH8DSimSensorBridge()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
