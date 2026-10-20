#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import rosbag2_py

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import fit_index_contact_v1 as bagbase


JOINT_TOPIC = "/joint_states"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

HW_TARGET = np.asarray(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


def interp(
    query_t,
    data,
):
    if (
        data is None
        or
        len(data) == 0
    ):
        return np.full(
            len(query_t),
            np.nan,
        )

    data = np.asarray(
        data,
        dtype=float,
    )

    return np.interp(
        query_t,
        data[:, 0],
        data[:, 1],
    )


def fit_slope(
    x,
    y,
):

    good = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    if np.count_nonzero(good) < 5:
        return np.nan, np.nan

    c = np.polyfit(
        x[good],
        y[good],
        1,
    )

    pred = np.polyval(
        c,
        x[good],
    )

    ss_res = np.sum(
        (
            y[good]
            -
            pred
        ) ** 2
    )

    ss_tot = np.sum(
        (
            y[good]
            -
            np.mean(
                y[good]
            )
        ) ** 2
    )

    r2 = (
        1.0
        -
        ss_res
        /
        max(
            ss_tot,
            1e-12,
        )
    )

    return (
        float(c[0]),
        float(r2),
    )


def state_at_times(
    times,
    states,
):

    if len(states) == 0:
        return [
            ""
            for _ in times
        ]

    state_t = np.asarray(
        [
            s[0]
            for s in states
        ],
        dtype=float,
    )

    state_v = [
        str(s[1])
        for s in states
    ]

    out = []

    for t in times:

        i = int(
            np.searchsorted(
                state_t,
                t,
                side="right",
            )
            -
            1
        )

        if i < 0:
            out.append("")
        else:
            out.append(
                state_v[
                    min(
                        i,
                        len(state_v) - 1,
                    )
                ]
            )

    return out


def summarize_segment(
    name,
    t,
    encoder,
    force,
    mcp,
    pip,
    dip,
    mask,
):

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    n = int(
        np.count_nonzero(
            mask
        )
    )

    if n < 2:
        print(
            f"Too few samples: {n}"
        )
        return None

    tt = t[mask]
    ee = encoder[mask]
    ff = force[mask]

    q = np.column_stack(
        [
            mcp[mask],
            pip[mask],
            dip[mask],
        ]
    )

    q0 = q[0]
    q1 = q[-1]

    dq = (
        q1
        -
        q0
    )

    print(
        f"samples       = {n}"
    )

    print(
        f"time          = "
        f"{tt[0]:.6f} -> "
        f"{tt[-1]:.6f} s"
    )

    print(
        f"encoder       = "
        f"{ee[0]:.3f} -> "
        f"{ee[-1]:.3f}"
    )

    print(
        f"force median  = "
        f"{np.nanmedian(ff):.6f} N"
    )

    print(
        f"force max     = "
        f"{np.nanmax(ff):.6f} N"
    )

    print()
    print(
        "START MCP/PIP/DIP = "
        f"{q0[0]:.3f} / "
        f"{q0[1]:.3f} / "
        f"{q0[2]:.3f} deg"
    )

    print(
        "END   MCP/PIP/DIP = "
        f"{q1[0]:.3f} / "
        f"{q1[1]:.3f} / "
        f"{q1[2]:.3f} deg"
    )

    print(
        "DELTA MCP/PIP/DIP = "
        f"{dq[0]:+.3f} / "
        f"{dq[1]:+.3f} / "
        f"{dq[2]:+.3f} deg"
    )

    de = (
        ee
        -
        ee[0]
    )

    print()
    print(
        "JOINT ANGLE CHANGE / ENCODER"
    )

    for label, y in [
        (
            "MCP",
            q[:, 0]
            -
            q[0, 0],
        ),
        (
            "PIP",
            q[:, 1]
            -
            q[0, 1],
        ),
        (
            "DIP",
            q[:, 2]
            -
            q[0, 2],
        ),
    ]:

        slope, r2 = fit_slope(
            de,
            y,
        )

        print(
            f"{label}: "
            f"{slope:+.8f} deg/tick "
            f"(R2={r2:.4f})"
        )

    return {
        "name": name,
        "samples": n,
        "time_start": float(
            tt[0]
        ),
        "time_end": float(
            tt[-1]
        ),
        "encoder_start": float(
            ee[0]
        ),
        "encoder_end": float(
            ee[-1]
        ),
        "mcp_start": float(
            q0[0]
        ),
        "pip_start": float(
            q0[1]
        ),
        "dip_start": float(
            q0[2]
        ),
        "mcp_end": float(
            q1[0]
        ),
        "pip_end": float(
            q1[1]
        ),
        "dip_end": float(
            q1[2]
        ),
    }


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
            "v35g15a2_hardware_jointstates"
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

    # ========================================================
    # Use EXACT existing H1 reader for state / force / encoder
    # ========================================================

    (
        states,
        force_data,
        encoder_data,
    ) = bagbase.read_bag(
        bag
    )

    transitions = (
        bagbase.compress_states(
            states
        )
    )

    (
        approach_start,
        regulation_start,
        hold_start,
        hold_end,
    ) = bagbase.find_intervals(
        transitions
    )

    encoder_data = np.asarray(
        encoder_data,
        dtype=float,
    )

    encoder_data = (
        encoder_data.copy()
    )

    encoder_data[:, 1] = (
        bagbase.median_filter_5(
            encoder_data[:, 1]
        )
    )

    print()
    print("=" * 100)
    print(
        "G15A2 — HARDWARE JOINT-STATE AUDIT"
    )
    print("=" * 100)

    print()
    print(
        "Existing H1 intervals:"
    )

    print(
        f"  approach start   = "
        f"{approach_start:.6f} s"
    )

    print(
        f"  regulation start = "
        f"{regulation_start:.6f} s"
    )

    print(
        f"  hold start       = "
        f"{hold_start:.6f} s"
    )

    print(
        f"  hold end         = "
        f"{hold_end:.6f} s"
    )

    print()
    print(
        "State transitions:"
    )

    for x in transitions:
        print(
            " ",
            x,
        )

    # ========================================================
    # Read actual /joint_states
    # ========================================================

    reader = (
        rosbag2_py.SequentialReader()
    )

    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(
                bag
            ),
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

    if JOINT_TOPIC not in topic_types:
        raise RuntimeError(
            f"{JOINT_TOPIC} is not in bag."
        )

    joint_type = get_message(
        topic_types[
            JOINT_TOPIC
        ]
    )

    samples = []

    t0 = None

    missing = set()

    while reader.has_next():

        topic, raw, stamp = (
            reader.read_next()
        )

        if t0 is None:
            t0 = stamp

        if topic != JOINT_TOPIC:
            continue

        t = (
            stamp
            -
            t0
        ) * 1e-9

        msg = deserialize_message(
            raw,
            joint_type,
        )

        names = list(
            msg.name
        )

        values = {
            name:
                float(
                    msg.position[i]
                )
            for i, name
            in enumerate(names)
            if i < len(
                msg.position
            )
        }

        required = [
            MCP_NAME,
            PIP_NAME,
            DIP_NAME,
        ]

        if any(
            name not in values
            for name in required
        ):
            for name in required:
                if name not in values:
                    missing.add(
                        name
                    )
            continue

        samples.append(
            (
                float(t),
                values[MCP_NAME],
                values[PIP_NAME],
                values[DIP_NAME],
            )
        )

    if missing:
        print()
        print(
            "Missing names in some messages:",
            missing,
        )

    if len(samples) < 10:
        raise RuntimeError(
            "Too few index joint-state samples."
        )

    arr = np.asarray(
        samples,
        dtype=float,
    )

    t = arr[:, 0]

    # ROS joint states are radians.
    q_deg = np.rad2deg(
        arr[:, 1:4]
    )

    mcp = q_deg[:, 0]
    pip = q_deg[:, 1]
    dip = q_deg[:, 2]

    encoder = interp(
        t,
        encoder_data,
    )

    force = interp(
        t,
        force_data,
    )

    state = state_at_times(
        t,
        states,
    )

    # ========================================================
    # Basic sanity / information test
    # ========================================================

    print()
    print("=" * 100)
    print(
        "JOINT-STATE INFORMATION CHECK"
    )
    print("=" * 100)

    for label, x in [
        (
            "MCP",
            mcp,
        ),
        (
            "PIP",
            pip,
        ),
        (
            "DIP",
            dip,
        ),
    ]:

        rounded_unique = len(
            np.unique(
                np.round(
                    x,
                    6,
                )
            )
        )

        print()
        print(label)

        print(
            f"  min       = "
            f"{np.min(x):.6f} deg"
        )

        print(
            f"  max       = "
            f"{np.max(x):.6f} deg"
        )

        print(
            f"  range     = "
            f"{np.ptp(x):.6f} deg"
        )

        print(
            f"  std       = "
            f"{np.std(x):.6f} deg"
        )

        print(
            f"  unique≈   = "
            f"{rounded_unique}"
        )

    # ========================================================
    # First useful force thresholds
    # ========================================================

    first_force_03 = np.nan
    first_force_10 = np.nan

    valid_03 = np.flatnonzero(
        force >= 0.03
    )

    valid_10 = np.flatnonzero(
        force >= 0.10
    )

    if len(valid_03):
        first_force_03 = float(
            t[
                valid_03[0]
            ]
        )

    if len(valid_10):
        first_force_10 = float(
            t[
                valid_10[0]
            ]
        )

    print()
    print(
        f"first force >= 0.03 N: "
        f"{first_force_03}"
    )

    print(
        f"first force >= 0.10 N: "
        f"{first_force_10}"
    )

    # ========================================================
    # Segment summaries
    # ========================================================

    summaries = []

    summaries.append(
        summarize_segment(
            "APPROACH",
            t,
            encoder,
            force,
            mcp,
            pip,
            dip,
            (
                (t >= approach_start)
                &
                (t < regulation_start)
            ),
        )
    )

    # Stricter free-motion candidate:
    # use approach start -> first 0.03 N crossing.
    strict_end = (
        first_force_03
        if np.isfinite(
            first_force_03
        )
        else regulation_start
    )

    if strict_end <= approach_start:
        strict_end = (
            regulation_start
        )

    summaries.append(
        summarize_segment(
            "STRICT PRE-CONTACT CANDIDATE",
            t,
            encoder,
            force,
            mcp,
            pip,
            dip,
            (
                (t >= approach_start)
                &
                (t < strict_end)
            ),
        )
    )

    summaries.append(
        summarize_segment(
            "REGULATION",
            t,
            encoder,
            force,
            mcp,
            pip,
            dip,
            (
                (t >= regulation_start)
                &
                (t < hold_start)
            ),
        )
    )

    summaries.append(
        summarize_segment(
            "HOLD",
            t,
            encoder,
            force,
            mcp,
            pip,
            dip,
            (
                (t >= hold_start)
                &
                (t <= hold_end)
            ),
        )
    )

    # ========================================================
    # Crucial validity test:
    # compare /joint_states HOLD against independent image target.
    # ========================================================

    hold_mask = (
        (t >= hold_start)
        &
        (t <= hold_end)
    )

    hold_q = np.column_stack(
        [
            mcp[hold_mask],
            pip[hold_mask],
            dip[hold_mask],
        ]
    )

    hold_median = np.nanmedian(
        hold_q,
        axis=0,
    )

    hold_mean = np.nanmean(
        hold_q,
        axis=0,
    )

    hold_error = (
        hold_median
        -
        HW_TARGET
    )

    hold_rmse = float(
        np.sqrt(
            np.mean(
                hold_error ** 2
            )
        )
    )

    print()
    print("=" * 100)
    print(
        "CRITICAL VALIDITY CHECK"
    )
    print("=" * 100)

    print()
    print(
        "Independent image-measured H1 pose:"
    )

    print(
        f"  MCP/PIP/DIP = "
        f"{HW_TARGET[0]:.4f} / "
        f"{HW_TARGET[1]:.4f} / "
        f"{HW_TARGET[2]:.4f} deg"
    )

    print()
    print(
        "/joint_states HOLD median:"
    )

    print(
        f"  MCP/PIP/DIP = "
        f"{hold_median[0]:.4f} / "
        f"{hold_median[1]:.4f} / "
        f"{hold_median[2]:.4f} deg"
    )

    print()
    print(
        "difference joint_states - image:"
    )

    print(
        f"  MCP/PIP/DIP = "
        f"{hold_error[0]:+.4f} / "
        f"{hold_error[1]:+.4f} / "
        f"{hold_error[2]:+.4f} deg"
    )

    print(
        f"\n3-joint RMSE = "
        f"{hold_rmse:.4f} deg"
    )

    # ========================================================
    # Save full synchronized trajectory
    # ========================================================

    csv_path = (
        out_dir
        /
        "g15a2_hardware_trajectory.csv"
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
            "encoder_tick",
            "net_force_N",
            "mcp_deg_jointstate",
            "pip_deg_jointstate",
            "dip_deg_jointstate",
        ])

        for i in range(
            len(t)
        ):

            writer.writerow([
                t[i],
                state[i],
                encoder[i],
                force[i],
                mcp[i],
                pip[i],
                dip[i],
            ])

    # ========================================================
    # Save concise summary
    # ========================================================

    summary_path = (
        out_dir
        /
        "summary.txt"
    )

    with summary_path.open(
        "w"
    ) as f:

        f.write(
            "V35G15A2 HARDWARE JOINT-STATE AUDIT\n"
        )

        f.write(
            "===================================\n\n"
        )

        f.write(
            "Independent image target:\n"
        )

        f.write(
            f"  {HW_TARGET.tolist()}\n\n"
        )

        f.write(
            "joint_states hold median:\n"
        )

        f.write(
            f"  {hold_median.tolist()}\n\n"
        )

        f.write(
            "joint_states - image error:\n"
        )

        f.write(
            f"  {hold_error.tolist()}\n\n"
        )

        f.write(
            f"hold pose RMSE: "
            f"{hold_rmse:.6f} deg\n\n"
        )

        f.write(
            f"first force >=0.03 N: "
            f"{first_force_03}\n"
        )

        f.write(
            f"first force >=0.10 N: "
            f"{first_force_10}\n"
        )

    # ========================================================
    # Plot — useful for inspecting whether these are real,
    # changing joint signals or merely synthetic/static values.
    # ========================================================

    try:

        import matplotlib.pyplot as plt

        fig = plt.figure(
            figsize=(
                11,
                7,
            )
        )

        ax = fig.add_subplot(
            111
        )

        ax.plot(
            t,
            mcp,
            label="MCP",
        )

        ax.plot(
            t,
            pip,
            label="PIP",
        )

        ax.plot(
            t,
            dip,
            label="DIP",
        )

        ax.axvline(
            approach_start,
            linestyle="--",
            label="approach",
        )

        ax.axvline(
            regulation_start,
            linestyle="--",
            label="regulation",
        )

        ax.axvline(
            hold_start,
            linestyle="--",
            label="hold start",
        )

        ax.axvline(
            hold_end,
            linestyle="--",
            label="hold end",
        )

        ax.set_xlabel(
            "Bag time [s]"
        )

        ax.set_ylabel(
            "Joint angle [deg]"
        )

        ax.set_title(
            "H1 hardware /joint_states — index finger"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            out_dir
            /
            "g15a2_joint_trajectory.png",
            dpi=180,
        )

        plt.close(
            fig
        )

    except Exception as e:

        print()
        print(
            "Plot skipped:",
            e,
        )

    print()
    print("=" * 100)
    print(
        "G15A2 COMPLETE"
    )
    print("=" * 100)

    print()
    print(
        f"All outputs:\n  "
        f"{out_dir}"
    )


if __name__ == "__main__":
    main()
