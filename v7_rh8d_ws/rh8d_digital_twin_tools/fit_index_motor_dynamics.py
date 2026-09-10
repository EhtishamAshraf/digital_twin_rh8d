#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


COMMAND_TOPIC = "/rh8dr/hand_position_controller/commands"
STATE_TOPIC = "/dynamic_joint_states"

MOTOR_NAME = "index_flexion_motor"

# hardware command-array index
INDEX_COMMAND_SLOT = 5


def read_bag(bag_dir):

    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir),
        storage_id="sqlite3",
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(
        storage_options,
        converter_options,
    )

    topic_types = {
        x.name: x.type
        for x in reader.get_all_topics_and_types()
    }

    commands = []
    states = []

    t0 = None

    while reader.has_next():

        topic, raw, timestamp = reader.read_next()

        if t0 is None:
            t0 = timestamp

        t = (timestamp - t0) * 1e-9


        if topic == COMMAND_TOPIC:

            msg_type = get_message(
                topic_types[topic]
            )

            msg = deserialize_message(
                raw,
                msg_type,
            )

            if len(msg.data) > INDEX_COMMAND_SLOT:

                commands.append(
                    (
                        t,
                        float(
                            msg.data[
                                INDEX_COMMAND_SLOT
                            ]
                        ),
                    )
                )


        elif topic == STATE_TOPIC:

            msg_type = get_message(
                topic_types[topic]
            )

            msg = deserialize_message(
                raw,
                msg_type,
            )

            try:
                i = list(
                    msg.joint_names
                ).index(
                    MOTOR_NAME
                )

            except ValueError:
                continue

            interfaces = msg.interface_values[i]

            names = list(
                interfaces.interface_names
            )

            values = list(
                interfaces.values
            )

            if "position" not in names:
                continue

            p = values[
                names.index("position")
            ]

            states.append(
                (
                    t,
                    float(p),
                )
            )


    return (
        np.asarray(commands),
        np.asarray(states),
    )


def compress_commands(commands):

    result = []

    last = None

    for t, value in commands:

        if (
            last is None
            or value != last
        ):

            result.append(
                (t, value)
            )

            last = value

    return np.asarray(result)


def command_at_time(
    t,
    command_steps,
    delay,
):

    query = t - delay

    idx = np.searchsorted(
        command_steps[:, 0],
        query,
        side="right",
    ) - 1

    if idx < 0:
        return command_steps[0, 1]

    return command_steps[idx, 1]


def simulate(
    state_times,
    measured,
    commands,
    delay,
    gain,
    close_velocity,
    open_velocity,
):

    pred = np.zeros_like(
        measured
    )

    pred[0] = measured[0]


    for i in range(
        1,
        len(state_times),
    ):

        dt = (
            state_times[i]
            - state_times[i - 1]
        )

        # Ignore abnormally large telemetry gaps.
        # We do not want missing state samples to be
        # identified as motor dynamics.
        if (
            dt <= 0
            or dt > 0.05
        ):
            pred[i] = pred[i - 1]
            continue


        u = command_at_time(
            state_times[i - 1],
            commands,
            delay,
        )

        error = (
            u - pred[i - 1]
        )

        velocity = (
            gain * error
        )


        if velocity >= 0:

            velocity = min(
                velocity,
                close_velocity,
            )

        else:

            velocity = max(
                velocity,
                -open_velocity,
            )


        pred[i] = (
            pred[i - 1]
            + velocity * dt
        )


    return pred


def main():

    if len(sys.argv) != 2:

        print()
        print(
            "Usage:"
        )

        print(
            "  python3 fit_index_motor_dynamics.py "
            "/path/to/rosbag_directory"
        )

        sys.exit(1)


    bag_dir = Path(
        sys.argv[1]
    ).expanduser().resolve()


    print()
    print("=" * 72)
    print("RH8D INDEX MOTOR DYNAMICS IDENTIFICATION")
    print("=" * 72)

    print()
    print("Bag:")
    print(bag_dir)


    commands, states = read_bag(
        bag_dir
    )

    commands = compress_commands(
        commands
    )


    if len(commands) < 2:

        raise RuntimeError(
            "Could not extract command sequence."
        )


    if len(states) < 100:

        raise RuntimeError(
            "Could not extract index motor encoder states."
        )


    t = states[:, 0]
    y = states[:, 1]


    print()
    print(
        f"Command transitions: {len(commands)}"
    )

    print(
        f"Encoder samples:     {len(states)}"
    )


    # Only score the controlled experiment interval.
    start = (
        commands[0, 0] - 0.2
    )

    end = (
        commands[-1, 0] + 2.0
    )

    mask = (
        (t >= start)
        & (t <= end)
    )

    tt = t[mask]
    yy = y[mask]


    # --------------------------------------------------------
    # COARSE SEARCH
    # --------------------------------------------------------

    delays = np.arange(
        0.00,
        0.121,
        0.01,
    )

    gains = [
        4.0,
        6.0,
        8.0,
        10.0,
        15.0,
        20.0,
    ]

    close_velocities = np.arange(
        900.0,
        1801.0,
        100.0,
    )

    open_velocities = np.arange(
        1100.0,
        2001.0,
        100.0,
    )


    results = []


    total = (
        len(delays)
        * len(gains)
        * len(close_velocities)
        * len(open_velocities)
    )

    print()
    print(
        f"Testing {total} motor parameter combinations..."
    )


    count = 0


    for delay in delays:

        for gain in gains:

            for vc in close_velocities:

                for vo in open_velocities:

                    pred = simulate(
                        tt,
                        yy,
                        commands,
                        delay,
                        gain,
                        vc,
                        vo,
                    )

                    # Ignore samples immediately after
                    # unusually large telemetry gaps.
                    dt = np.diff(
                        tt,
                        prepend=tt[0],
                    )

                    valid = (
                        dt < 0.05
                    )

                    error = (
                        pred[valid]
                        - yy[valid]
                    )

                    rmse = math.sqrt(
                        np.mean(
                            error ** 2
                        )
                    )

                    results.append(
                        (
                            rmse,
                            delay,
                            gain,
                            vc,
                            vo,
                        )
                    )

                    count += 1


    results.sort(
        key=lambda x: x[0]
    )


    print()
    print("=" * 72)
    print("TOP 10 MOTOR MODELS")
    print("=" * 72)


    for rank, result in enumerate(
        results[:10],
        start=1,
    ):

        (
            error,
            delay,
            gain,
            vc,
            vo,
        ) = result


        print(
            f"#{rank:2d} "
            f"RMSE={error:7.2f} ticks   "
            f"delay={delay*1000:5.0f} ms   "
            f"K={gain:5.1f}  "
            f"close={vc:6.0f} ticks/s   "
            f"open={vo:6.0f} ticks/s"
        )


    best = results[0]


    print()
    print("=" * 72)
    print("BEST IDENTIFIED MOTOR MODEL")
    print("=" * 72)

    print(
        f"Encoder RMSE       = "
        f"{best[0]:.2f} ticks"
    )

    print(
        f"Command delay      = "
        f"{best[1]*1000:.1f} ms"
    )

    print(
        f"Servo gain         = "
        f"{best[2]:.2f} 1/s"
    )

    print(
        f"Closing speed cap  = "
        f"{best[3]:.1f} ticks/s"
    )

    print(
        f"Opening speed cap  = "
        f"{best[4]:.1f} ticks/s"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
