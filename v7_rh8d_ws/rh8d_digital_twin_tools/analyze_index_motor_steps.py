#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


COMMAND_TOPIC = "/rh8dr/hand_position_controller/commands"
STATE_TOPIC = "/dynamic_joint_states"

MOTOR_NAME = "index_flexion_motor"
COMMAND_INDEX = 5


def read_bag(path):

    reader = rosbag2_py.SequentialReader()

    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(path),
            storage_id="sqlite3",
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    types = {
        x.name: x.type
        for x in reader.get_all_topics_and_types()
    }

    commands = []
    states = []

    t0 = None

    while reader.has_next():

        topic, raw, stamp = reader.read_next()

        if t0 is None:
            t0 = stamp

        t = (stamp - t0) * 1e-9

        if topic == COMMAND_TOPIC:

            msg = deserialize_message(
                raw,
                get_message(types[topic]),
            )

            if len(msg.data) > COMMAND_INDEX:
                commands.append(
                    (
                        t,
                        float(msg.data[COMMAND_INDEX]),
                    )
                )

        elif topic == STATE_TOPIC:

            msg = deserialize_message(
                raw,
                get_message(types[topic]),
            )

            names = list(msg.joint_names)

            if MOTOR_NAME not in names:
                continue

            i = names.index(MOTOR_NAME)

            interfaces = msg.interface_values[i]

            interface_names = list(
                interfaces.interface_names
            )

            if "position" not in interface_names:
                continue

            j = interface_names.index(
                "position"
            )

            states.append(
                (
                    t,
                    float(interfaces.values[j]),
                )
            )

    return (
        np.asarray(commands, dtype=float),
        np.asarray(states, dtype=float),
    )


def compress_commands(commands):

    out = []

    previous = None

    for t, value in commands:

        if (
            previous is None
            or abs(value - previous) > 1e-9
        ):
            out.append(
                (t, value)
            )

            previous = value

    return np.asarray(
        out,
        dtype=float,
    )


def interpolate_crossing(
    t,
    y,
    threshold,
    direction,
):

    if direction > 0:

        indexes = np.where(
            y >= threshold
        )[0]

    else:

        indexes = np.where(
            y <= threshold
        )[0]

    if len(indexes) == 0:
        return None

    i = indexes[0]

    if i == 0:
        return t[0]

    t1, t2 = t[i - 1], t[i]
    y1, y2 = y[i - 1], y[i]

    if abs(y2 - y1) < 1e-12:
        return t2

    alpha = (
        threshold - y1
    ) / (
        y2 - y1
    )

    return (
        t1
        + alpha * (t2 - t1)
    )


def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python3 analyze_index_motor_steps.py "
            "/path/to/bag_directory"
        )
        sys.exit(1)

    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()

    commands, states = read_bag(
        bag
    )

    commands = compress_commands(
        commands
    )

    t_state = states[:, 0]
    p_state = states[:, 1]

    print()
    print("=" * 100)
    print("RH8D INDEX MOTOR — INDIVIDUAL STEP ANALYSIS")
    print("=" * 100)

    print()
    print(
        f"Command transitions: {len(commands)}"
    )

    print(
        f"Encoder samples:     {len(states)}"
    )

    print()
    print(
        "NOTE: delay is measured from ROS command timestamp "
        "to first clear encoder motion."
    )

    print()
    print(
        f"{'STEP':>13s} "
        f"{'START':>8s} "
        f"{'END':>8s} "
        f"{'DELAY':>9s} "
        f"{'T10-90':>9s} "
        f"{'V10-90':>11s}"
    )

    print("-" * 100)

    delays = []
    closing_velocities = []
    opening_velocities = []

    for k in range(
        1,
        len(commands),
    ):

        t_cmd = commands[k, 0]

        u_prev = commands[k - 1, 1]
        u_new = commands[k, 1]

        delta = (
            u_new - u_prev
        )

        if abs(delta) < 50:
            continue

        direction = (
            1 if delta > 0 else -1
        )

        # Approximately 250 ms before command:
        pre_mask = (
            (t_state >= t_cmd - 0.25)
            & (t_state < t_cmd)
        )

        if np.count_nonzero(pre_mask) < 3:
            continue

        start_position = float(
            np.median(
                p_state[pre_mask]
            )
        )

        # Analyze until just before the next command.
        if k + 1 < len(commands):
            t_end = commands[k + 1, 0] - 0.02
        else:
            t_end = t_cmd + 2.0

        mask = (
            (t_state >= t_cmd)
            & (t_state <= t_end)
        )

        tt = t_state[mask]
        yy = p_state[mask]

        if len(tt) < 5:
            continue

        # Use the actual achieved endpoint rather than blindly
        # assuming the motor reaches exactly the command.
        tail_n = max(
            3,
            int(0.15 * len(yy)),
        )

        end_position = float(
            np.median(
                yy[-tail_n:]
            )
        )

        achieved_delta = (
            end_position - start_position
        )

        if abs(achieved_delta) < 50:
            continue

        # First clear motion threshold.
        #
        # 10 encoder counts is large enough to reject small
        # encoder jitter but much smaller than a commanded step.
        motion_threshold = (
            start_position
            + direction * 10.0
        )

        t_motion = interpolate_crossing(
            tt,
            yy,
            motion_threshold,
            direction,
        )

        delay = (
            None
            if t_motion is None
            else t_motion - t_cmd
        )

        p10 = (
            start_position
            + 0.10 * achieved_delta
        )

        p90 = (
            start_position
            + 0.90 * achieved_delta
        )

        t10 = interpolate_crossing(
            tt,
            yy,
            p10,
            direction,
        )

        t90 = interpolate_crossing(
            tt,
            yy,
            p90,
            direction,
        )

        rise = None
        velocity = None

        if (
            t10 is not None
            and t90 is not None
            and t90 > t10
        ):

            rise = t90 - t10

            velocity = abs(
                0.8 * achieved_delta
                / rise
            )

        if delay is not None:
            delays.append(
                delay
            )

        if velocity is not None:

            if direction > 0:
                closing_velocities.append(
                    velocity
                )
            else:
                opening_velocities.append(
                    velocity
                )

        delay_text = (
            "n/a"
            if delay is None
            else f"{1000*delay:7.1f}ms"
        )

        rise_text = (
            "n/a"
            if rise is None
            else f"{1000*rise:7.1f}ms"
        )

        velocity_text = (
            "n/a"
            if velocity is None
            else f"{velocity:9.1f}"
        )

        print(
            f"{u_prev:5.0f}->{u_new:<5.0f} "
            f"{start_position:8.1f} "
            f"{end_position:8.1f} "
            f"{delay_text:>9s} "
            f"{rise_text:>9s} "
            f"{velocity_text:>11s}"
        )

    print()
    print("=" * 100)
    print("ROBUST SUMMARY")
    print("=" * 100)

    if delays:

        d = np.asarray(
            delays
        ) * 1000.0

        print(
            f"First-motion delay median = "
            f"{np.median(d):.1f} ms"
        )

        print(
            f"First-motion delay range  = "
            f"{np.min(d):.1f} .. {np.max(d):.1f} ms"
        )

    if closing_velocities:

        x = np.asarray(
            closing_velocities
        )

        print()
        print(
            f"Closing velocity median   = "
            f"{np.median(x):.1f} ticks/s"
        )

        print(
            f"Closing velocity range    = "
            f"{np.min(x):.1f} .. {np.max(x):.1f} ticks/s"
        )

    if opening_velocities:

        x = np.asarray(
            opening_velocities
        )

        print()
        print(
            f"Opening velocity median   = "
            f"{np.median(x):.1f} ticks/s"
        )

        print(
            f"Opening velocity range    = "
            f"{np.min(x):.1f} .. {np.max(x):.1f} ticks/s"
        )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
