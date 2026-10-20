#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rosbag2_py

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


STATE_TOPIC = (
    "/rh8dr/index_contact_sysid/state"
)

TARGET_TOPIC = (
    "/rh8dr/index_contact_sysid/target_ticks"
)

FORCE_TOPIC = (
    "/rh8dr/index_contact_sysid/net_force"
)

DESIRED_TOPIC = (
    "/rh8dr/index_contact_sysid/desired_force"
)

DYNAMIC_TOPIC = (
    "/dynamic_joint_states"
)

INDEX_MOTOR = (
    "index_flexion_motor"
)


def numeric_interp(
    data,
    t,
):

    if len(data) == 0:
        return np.nan

    a = np.asarray(
        data,
        dtype=float,
    )

    order = np.argsort(
        a[:, 0]
    )

    a = a[order]

    return float(
        np.interp(
            t,
            a[:, 0],
            a[:, 1],
            left=np.nan,
            right=np.nan,
        )
    )


def state_at(
    states,
    t,
):

    if not states:
        return ""

    times = np.asarray(
        [
            x[0]
            for x in states
        ],
        dtype=float,
    )

    i = (
        np.searchsorted(
            times,
            t,
            side="right",
        )
        -
        1
    )

    if i < 0:
        return ""

    return str(
        states[
            min(
                i,
                len(states) - 1,
            )
        ][1]
    )


def compress_states(
    states,
):

    out = []

    old = None

    for t, s in states:

        if s != old:
            out.append(
                (
                    float(t),
                    str(s),
                )
            )

            old = s

    return out


def first_state_time(
    transitions,
    wanted,
):

    for t, state in transitions:

        if state == wanted:
            return float(t)

    return np.nan


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "bag",
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g19a_force_controller_audit"
        ),
    )

    args = parser.parse_args()

    bag = (
        args.bag
        .expanduser()
        .resolve()
    )

    out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reader = (
        rosbag2_py.SequentialReader()
    )

    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(bag),
            storage_id="sqlite3",
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topic_types = {
        x.name: x.type
        for x
        in reader.get_all_topics_and_types()
    }

    msg_types = {
        topic:
            get_message(
                type_name
            )
        for topic, type_name
        in topic_types.items()
    }

    states = []
    targets = []
    forces = []
    desired = []
    encoders = []

    t0 = None

    while reader.has_next():

        topic, raw, stamp = (
            reader.read_next()
        )

        if t0 is None:
            t0 = stamp

        t = (
            stamp
            -
            t0
        ) * 1e-9

        if topic not in msg_types:
            continue

        if topic == STATE_TOPIC:

            msg = deserialize_message(
                raw,
                msg_types[topic],
            )

            states.append(
                (
                    t,
                    str(msg.data),
                )
            )

        elif topic == TARGET_TOPIC:

            msg = deserialize_message(
                raw,
                msg_types[topic],
            )

            targets.append(
                (
                    t,
                    float(msg.data),
                )
            )

        elif topic == FORCE_TOPIC:

            msg = deserialize_message(
                raw,
                msg_types[topic],
            )

            forces.append(
                (
                    t,
                    float(msg.data),
                )
            )

        elif topic == DESIRED_TOPIC:

            msg = deserialize_message(
                raw,
                msg_types[topic],
            )

            desired.append(
                (
                    t,
                    float(msg.data),
                )
            )

        elif topic == DYNAMIC_TOPIC:

            msg = deserialize_message(
                raw,
                msg_types[topic],
            )

            names = list(
                msg.joint_names
            )

            if INDEX_MOTOR not in names:
                continue

            i = names.index(
                INDEX_MOTOR
            )

            if (
                i
                >=
                len(
                    msg.interface_values
                )
            ):
                continue

            iv = (
                msg.interface_values[i]
            )

            interface_names = list(
                iv.interface_names
            )

            if (
                "position"
                not in interface_names
            ):
                continue

            j = interface_names.index(
                "position"
            )

            encoders.append(
                (
                    t,
                    float(
                        iv.values[j]
                    ),
                )
            )

    transitions = (
        compress_states(
            states
        )
    )

    approach_start = (
        first_state_time(
            transitions,
            "APPROACH",
        )
    )

    regulation_start = (
        first_state_time(
            transitions,
            "FORCE_REGULATION",
        )
    )

    hold_start = (
        first_state_time(
            transitions,
            "FORCE_HOLD",
        )
    )

    print()
    print("=" * 120)
    print(
        "G19A — H1 FORCE-CONTROLLER AUDIT"
    )
    print("=" * 120)

    print()
    print(
        "STATE TRANSITIONS + MOTOR/FORCE VALUES"
    )
    print("=" * 120)

    print(
        "time       state                 "
        "encoder    target     error      "
        "net_force   desired_force"
    )

    print("-" * 120)

    for t, state in transitions:

        enc = numeric_interp(
            encoders,
            t,
        )

        target = numeric_interp(
            targets,
            t,
        )

        force = numeric_interp(
            forces,
            t,
        )

        desired_force = (
            numeric_interp(
                desired,
                t,
            )
        )

        error = (
            target
            -
            enc
        )

        print(
            f"{t:9.4f}  "
            f"{state:<20s} "
            f"{enc:9.2f} "
            f"{target:9.2f} "
            f"{error:9.2f} "
            f"{force:10.5f} "
            f"{desired_force:12.5f}"
        )

    print()
    print("=" * 120)
    print(
        "FIRST FORCE THRESHOLD CROSSINGS DURING APPROACH"
    )
    print("=" * 120)

    force_array = np.asarray(
        forces,
        dtype=float,
    )

    for threshold in [
        0.03,
        0.05,
        0.08,
        0.10,
        0.12,
        0.14,
    ]:

        mask = (
            (
                force_array[:, 0]
                >=
                approach_start
            )
            &
            (
                force_array[:, 1]
                >=
                threshold
            )
        )

        ids = np.flatnonzero(
            mask
        )

        if not len(ids):
            print(
                f"{threshold:.2f} N: "
                "never crossed"
            )
            continue

        i = int(
            ids[0]
        )

        t = float(
            force_array[
                i,
                0,
            ]
        )

        f = float(
            force_array[
                i,
                1,
            ]
        )

        enc = numeric_interp(
            encoders,
            t,
        )

        target = numeric_interp(
            targets,
            t,
        )

        desired_force = (
            numeric_interp(
                desired,
                t,
            )
        )

        state = state_at(
            states,
            t,
        )

        print(
            f"{threshold:.2f} N: "
            f"t={t:.6f} "
            f"measured={f:.5f} N "
            f"state={state:<18s} "
            f"encoder={enc:.2f} "
            f"target={target:.2f} "
            f"target-encoder={target-enc:.2f} "
            f"desired={desired_force:.5f}"
        )

    print()
    print("=" * 120)
    print(
        "PHASE SUMMARY"
    )
    print("=" * 120)

    phases = [
        (
            "APPROACH",
            approach_start,
            regulation_start,
        ),
        (
            "FORCE_REGULATION",
            regulation_start,
            hold_start,
        ),
        (
            "FIRST 0.5s HOLD",
            hold_start,
            hold_start + 0.5,
        ),
    ]

    def summarize_numeric(
        data,
        ta,
        tb,
    ):

        a = np.asarray(
            data,
            dtype=float,
        )

        mask = (
            (a[:, 0] >= ta)
            &
            (a[:, 0] < tb)
        )

        x = (
            a[
                mask,
                1,
            ]
        )

        if len(x) == 0:

            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

        return (
            float(x[0]),
            float(x[-1]),
            float(np.min(x)),
            float(np.max(x)),
        )

    for (
        name,
        ta,
        tb,
    ) in phases:

        print()
        print(name)
        print("-" * 70)

        e0, e1, emin, emax = (
            summarize_numeric(
                encoders,
                ta,
                tb,
            )
        )

        t0v, t1v, tmin, tmax = (
            summarize_numeric(
                targets,
                ta,
                tb,
            )
        )

        f0, f1, fmin, fmax = (
            summarize_numeric(
                forces,
                ta,
                tb,
            )
        )

        d0, d1, dmin, dmax = (
            summarize_numeric(
                desired,
                ta,
                tb,
            )
        )

        print(
            f"encoder: "
            f"{e0:.2f} -> {e1:.2f} "
            f"(min/max {emin:.2f}/{emax:.2f})"
        )

        print(
            f"target:  "
            f"{t0v:.2f} -> {t1v:.2f} "
            f"(min/max {tmin:.2f}/{tmax:.2f})"
        )

        print(
            f"force:   "
            f"{f0:.5f} -> {f1:.5f} N "
            f"(min/max {fmin:.5f}/{fmax:.5f})"
        )

        print(
            f"desired: "
            f"{d0:.5f} -> {d1:.5f} N "
            f"(min/max {dmin:.5f}/{dmax:.5f})"
        )

    # ------------------------------------------------------------
    # Save synchronized timeline on encoder timestamps.
    # ------------------------------------------------------------

    enc_array = np.asarray(
        encoders,
        dtype=float,
    )

    mask = (
        (
            enc_array[:, 0]
            >=
            approach_start - 0.5
        )
        &
        (
            enc_array[:, 0]
            <=
            hold_start + 1.0
        )
    )

    timeline = (
        enc_array[
            mask
        ]
    )

    csv_path = (
        out_dir
        /
        "g19a_force_controller_timeline.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "time_s",
            "state",
            "encoder_ticks",
            "target_ticks",
            "target_minus_encoder_ticks",
            "net_force_N",
            "desired_force_N",
        ])

        for t, enc in timeline:

            target = numeric_interp(
                targets,
                t,
            )

            force = numeric_interp(
                forces,
                t,
            )

            desired_force = (
                numeric_interp(
                    desired,
                    t,
                )
            )

            writer.writerow([
                t,
                state_at(
                    states,
                    t,
                ),
                enc,
                target,
                target - enc,
                force,
                desired_force,
            ])

    print()
    print(
        f"Timeline saved:\n  "
        f"{csv_path}"
    )


if __name__ == "__main__":
    main()
