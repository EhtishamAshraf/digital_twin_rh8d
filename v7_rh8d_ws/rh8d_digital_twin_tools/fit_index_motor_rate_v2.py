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
COMMAND_SLOT = 5


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

    topic_types = {
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
                get_message(topic_types[topic]),
            )

            if len(msg.data) > COMMAND_SLOT:

                commands.append(
                    (
                        t,
                        float(msg.data[COMMAND_SLOT]),
                    )
                )

        elif topic == STATE_TOPIC:

            msg = deserialize_message(
                raw,
                get_message(topic_types[topic]),
            )

            names = list(msg.joint_names)

            if MOTOR_NAME not in names:
                continue

            i = names.index(MOTOR_NAME)

            interfaces = msg.interface_values[i]

            inames = list(
                interfaces.interface_names
            )

            if "position" not in inames:
                continue

            j = inames.index("position")

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


def advance_motor(
    position,
    target,
    dt,
    close_speed,
    open_speed,
):

    error = target - position

    if abs(error) < 1e-12:
        return position

    if error > 0:

        movement = min(
            error,
            close_speed * dt,
        )

        return position + movement

    movement = min(
        -error,
        open_speed * dt,
    )

    return position - movement


def simulate(
    times,
    initial_position,
    commands,
    delay,
    close_speed,
    open_speed,
):

    prediction = np.empty(
        len(times),
        dtype=float,
    )

    prediction[0] = initial_position

    # First compressed command represents the initial target.
    target = commands[0, 1]

    # Remaining commands become delayed events.
    events = [
        (
            commands[i, 0] + delay,
            commands[i, 1],
        )
        for i in range(1, len(commands))
    ]

    event_index = 0

    current_time = times[0]
    current_position = initial_position

    for i in range(
        1,
        len(times),
    ):

        observation_time = times[i]

        # Process any command events occurring between
        # the previous and current encoder observation.
        while (
            event_index < len(events)
            and events[event_index][0] <= observation_time
        ):

            event_time, new_target = (
                events[event_index]
            )

            if event_time > current_time:

                current_position = advance_motor(
                    current_position,
                    target,
                    event_time - current_time,
                    close_speed,
                    open_speed,
                )

                current_time = event_time

            target = new_target
            event_index += 1

        if observation_time > current_time:

            current_position = advance_motor(
                current_position,
                target,
                observation_time - current_time,
                close_speed,
                open_speed,
            )

            current_time = observation_time

        prediction[i] = current_position

    return prediction


def score_candidate(
    state_times,
    measured,
    commands,
    delay,
    close_speed,
    open_speed,
):

    predicted = simulate(
        state_times,
        measured[0],
        commands,
        delay,
        close_speed,
        open_speed,
    )

    error = predicted - measured

    return (
        math.sqrt(
            np.mean(error ** 2)
        ),
        predicted,
    )


def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python3 fit_index_motor_rate_v2.py "
            "/path/to/rosbag_directory"
        )

        sys.exit(1)

    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()

    print()
    print("=" * 76)
    print(
        "RH8D INDEX MOTOR SYSTEM IDENTIFICATION "
        "— RATE-LIMITED MODEL"
    )
    print("=" * 76)

    print()
    print("Bag:")
    print(bag)

    commands, states = read_bag(
        bag
    )

    commands = compress_commands(
        commands
    )

    print()
    print(
        f"Command levels:  {len(commands)}"
    )

    print(
        f"Encoder samples: {len(states)}"
    )

    # Score only the actual controlled sequence.
    start = commands[0, 0] - 0.10
    end = commands[-1, 0] + 1.50

    mask = (
        (states[:, 0] >= start)
        & (states[:, 0] <= end)
    )

    times = states[mask, 0]
    measured = states[mask, 1]

    # ========================================================
    # COARSE SEARCH
    # ========================================================

    coarse_delays = np.arange(
        0.020,
        0.081,
        0.010,
    )

    coarse_close = np.arange(
        1000.0,
        1551.0,
        50.0,
    )

    coarse_open = np.arange(
        1250.0,
        1751.0,
        50.0,
    )

    results = []

    print()
    print(
        "Coarse search..."
    )

    for delay in coarse_delays:

        for close_speed in coarse_close:

            for open_speed in coarse_open:

                error, _ = score_candidate(
                    times,
                    measured,
                    commands,
                    delay,
                    close_speed,
                    open_speed,
                )

                results.append(
                    (
                        error,
                        delay,
                        close_speed,
                        open_speed,
                    )
                )

    results.sort(
        key=lambda x: x[0]
    )

    coarse_best = results[0]

    # ========================================================
    # FINE SEARCH AROUND COARSE BEST
    # ========================================================

    fine_delays = np.arange(
        max(0.0, coarse_best[1] - 0.012),
        coarse_best[1] + 0.0121,
        0.002,
    )

    fine_close = np.arange(
        coarse_best[2] - 125.0,
        coarse_best[2] + 125.1,
        25.0,
    )

    fine_open = np.arange(
        coarse_best[3] - 125.0,
        coarse_best[3] + 125.1,
        25.0,
    )

    fine_results = []

    print(
        "Fine search..."
    )

    for delay in fine_delays:

        for close_speed in fine_close:

            for open_speed in fine_open:

                error, _ = score_candidate(
                    times,
                    measured,
                    commands,
                    delay,
                    close_speed,
                    open_speed,
                )

                fine_results.append(
                    (
                        error,
                        delay,
                        close_speed,
                        open_speed,
                    )
                )

    fine_results.sort(
        key=lambda x: x[0]
    )

    print()
    print("=" * 76)
    print("TOP 15 RATE-LIMITED MOTOR MODELS")
    print("=" * 76)

    for rank, r in enumerate(
        fine_results[:15],
        start=1,
    ):

        print(
            f"#{rank:2d}  "
            f"RMSE={r[0]:7.2f} ticks   "
            f"delay={r[1]*1000:6.1f} ms   "
            f"close={r[2]:7.1f} ticks/s   "
            f"open={r[3]:7.1f} ticks/s"
        )

    best = fine_results[0]

    rmse_value, prediction = score_candidate(
        times,
        measured,
        commands,
        best[1],
        best[2],
        best[3],
    )

    abs_error = np.abs(
        prediction - measured
    )

    print()
    print("=" * 76)
    print("SELECTED MOTOR MODEL")
    print("=" * 76)

    print()
    print(
        f"Encoder RMSE      = "
        f"{rmse_value:.2f} ticks"
    )

    print(
        f"Median abs error  = "
        f"{np.median(abs_error):.2f} ticks"
    )

    print(
        f"95% abs error     = "
        f"{np.percentile(abs_error,95):.2f} ticks"
    )

    print()
    print(
        f"Command delay     = "
        f"{best[1]*1000:.1f} ms"
    )

    print(
        f"Closing speed     = "
        f"{best[2]:.1f} ticks/s"
    )

    print(
        f"Opening speed     = "
        f"{best[3]:.1f} ticks/s"
    )

    print()
    print(
        "Model:"
    )

    print(
        "  wait Td after a command change"
    )

    print(
        "  then move toward target at a fixed "
        "direction-dependent maximum rate"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()
