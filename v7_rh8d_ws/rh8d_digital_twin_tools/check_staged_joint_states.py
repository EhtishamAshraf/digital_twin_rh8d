#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


WATCH = [
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


class JointStateChecker(Node):
    def __init__(self):
        super().__init__("check_staged_joint_states")
        self.sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.cb,
            10,
        )

    def cb(self, msg):
        data = dict(zip(msg.name, msg.position))

        print("\n===== staged hand joint states [rad] =====")

        for name in WATCH:
            value = data.get(name, None)
            if value is None:
                print(f"{name:45s} MISSING")
            else:
                print(f"{name:45s} {value: .4f}")

        rclpy.shutdown()


def main():
    rclpy.init()
    node = JointStateChecker()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
