#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import csv
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


BAG = Path.home() / "rh8d_system_id/index_contact_20260911_164310"
OUT = Path.home() / "rh8d_results/v35g20b_hardware_state_audit"

JOINT_TOPIC = "/joint_states"
DYNAMIC_TOPIC = "/dynamic_joint_states"

OUT.mkdir(parents=True, exist_ok=True)


reader = rosbag2_py.SequentialReader()

reader.open(
    rosbag2_py.StorageOptions(
        uri=str(BAG),
        storage_id="sqlite3",
    ),
    rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    ),
)

topic_types = {
    x.name: x.type
    for x in reader.get_all_topics_and_types()
}

print("=" * 120)
print("G20B — HARDWARE JOINT / MOTOR INTERFACE AUDIT")
print("=" * 120)

print("\nTopic types:")
for topic in [JOINT_TOPIC, DYNAMIC_TOPIC]:
    print(f"  {topic}: {topic_types.get(topic, 'NOT FOUND')}")

msg_classes = {
    topic: get_message(msg_type)
    for topic, msg_type in topic_types.items()
    if topic in (JOINT_TOPIC, DYNAMIC_TOPIC)
}

joint_stats = defaultdict(list)
dynamic_stats = defaultdict(list)

long_rows = []

joint_names_seen = set()
dynamic_names_seen = set()

first_joint_printed = False
first_dynamic_printed = False

t0_ns = None


while reader.has_next():

    topic, raw, timestamp = reader.read_next()

    if t0_ns is None:
        t0_ns = timestamp

    if topic not in msg_classes:
        continue

    t = (timestamp - t0_ns) * 1e-9

    msg = deserialize_message(
        raw,
        msg_classes[topic],
    )

    # ============================================================
    # sensor_msgs/JointState
    # ============================================================

    if topic == JOINT_TOPIC:

        if not first_joint_printed:

            print("\n" + "=" * 120)
            print("FIRST /joint_states MESSAGE")
            print("=" * 120)

            print(f"time = {t:.6f} s")
            print(f"names = {list(msg.name)}")
            print(f"position = {list(msg.position)}")
            print(f"velocity = {list(msg.velocity)}")
            print(f"effort   = {list(msg.effort)}")

            first_joint_printed = True

        n = len(msg.name)

        for i, name in enumerate(msg.name):

            joint_names_seen.add(name)

            channels = []

            if i < len(msg.position):
                channels.append(("position", float(msg.position[i])))

            if i < len(msg.velocity):
                channels.append(("velocity", float(msg.velocity[i])))

            if i < len(msg.effort):
                channels.append(("effort", float(msg.effort[i])))

            for interface, value in channels:

                joint_stats[(name, interface)].append(value)

                long_rows.append(
                    [
                        t,
                        JOINT_TOPIC,
                        name,
                        interface,
                        value,
                    ]
                )

    # ============================================================
    # control_msgs/DynamicJointState
    # ============================================================

    elif topic == DYNAMIC_TOPIC:

        if not first_dynamic_printed:

            print("\n" + "=" * 120)
            print("FIRST /dynamic_joint_states MESSAGE")
            print("=" * 120)

            print(f"time = {t:.6f} s")

            for name, iv in zip(
                msg.joint_names,
                msg.interface_values,
            ):

                print(f"\nJOINT: {name}")

                for interface, value in zip(
                    iv.interface_names,
                    iv.values,
                ):
                    print(
                        f"  {interface:<30s} = {float(value):+.9f}"
                    )

            first_dynamic_printed = True

        for name, iv in zip(
            msg.joint_names,
            msg.interface_values,
        ):

            dynamic_names_seen.add(name)

            for interface, value in zip(
                iv.interface_names,
                iv.values,
            ):

                value = float(value)

                dynamic_stats[
                    (
                        name,
                        interface,
                    )
                ].append(value)

                long_rows.append(
                    [
                        t,
                        DYNAMIC_TOPIC,
                        name,
                        interface,
                        value,
                    ]
                )


def print_stats(title, stats):

    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    for (joint, interface), values in sorted(stats.items()):

        x = np.asarray(values, dtype=float)

        finite = x[np.isfinite(x)]

        if not len(finite):
            continue

        print(
            f"{joint:<45s} "
            f"{interface:<28s} "
            f"N={len(finite):5d}  "
            f"min={np.min(finite):+.8f}  "
            f"median={np.median(finite):+.8f}  "
            f"max={np.max(finite):+.8f}"
        )


print("\n" + "=" * 120)
print("ALL JOINT NAMES IN /joint_states")
print("=" * 120)

for name in sorted(joint_names_seen):
    print(f"  {name}")


print("\n" + "=" * 120)
print("ALL JOINT NAMES IN /dynamic_joint_states")
print("=" * 120)

for name in sorted(dynamic_names_seen):
    print(f"  {name}")


print_stats(
    "/joint_states NUMERIC SUMMARY",
    joint_stats,
)

print_stats(
    "/dynamic_joint_states NUMERIC SUMMARY",
    dynamic_stats,
)


# ================================================================
# Specifically print anything likely relevant to index finger/motor.
# ================================================================

print("\n" + "=" * 120)
print("INDEX-RELATED DYNAMIC INTERFACES")
print("=" * 120)

found_index = False

for (joint, interface), values in sorted(dynamic_stats.items()):

    text = (joint + " " + interface).lower()

    if "index" not in text:
        continue

    found_index = True

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if not len(x):
        continue

    print(
        f"{joint:<45s} "
        f"{interface:<30s} "
        f"min={np.min(x):+.9f} "
        f"median={np.median(x):+.9f} "
        f"max={np.max(x):+.9f}"
    )

if not found_index:
    print("No interface containing the text 'index' was found.")


# ================================================================
# Save full long-form time series.
# ================================================================

csv_path = OUT / "hardware_joint_interfaces_long.csv"

with csv_path.open("w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "bag_time_s",
            "topic",
            "joint",
            "interface",
            "value",
        ]
    )

    writer.writerows(long_rows)


print("\n" + "=" * 120)
print("G20B COMPLETE")
print("=" * 120)

print(f"\nSaved:\n  {csv_path}")

print(
    "\nDO NOT interpret /joint_states finger angles as physical "
    "MCP/PIP/DIP measurements until their names/source are verified."
)
