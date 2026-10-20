#!/usr/bin/env python3

import sys
import csv
import math
from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# ======================================================================
# PATHS
# ======================================================================

WS = Path.home() / (
    "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
)

MODEL_PATH = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_identified.xml"
)

RESULTS = WS / "rh8d_digital_twin_tools/results"
RESULTS.mkdir(parents=True, exist_ok=True)

DYNAMIC_CSV = RESULTS / "index_dynamic_replay_v1.csv"
REFERENCE_CSV = RESULTS / "index_dynamic_reference_v1.csv"
MOTOR_PLOT = RESULTS / "index_dynamic_motor_v1.png"
JOINT_PLOT = RESULTS / "index_dynamic_joints_v1.png"


# ======================================================================
# HARDWARE INTERFACE
# ======================================================================

COMMAND_TOPIC = "/rh8dr/hand_position_controller/commands"
STATE_TOPIC = "/dynamic_joint_states"

MOTOR_NAME = "index_flexion_motor"
COMMAND_SLOT = 5


# ======================================================================
# IDENTIFIED INDEX MOTOR MODEL
#
# From controlled hardware experiment 2026-09-10.
#
# These are nominal values for the current system-ID dataset.
# ======================================================================

MOTOR_DELAY = 0.058            # seconds
CLOSING_RATE = 1100.0          # ticks/s
OPENING_RATE = 1575.0          # ticks/s

OPEN_ENCODER = 100.0

# Identified effective encoder -> spool transmission.
METERS_PER_TICK = 6.0e-6


# ======================================================================
# VIDEO-DERIVED PHYSICAL JOINT REFERENCE
#
# NOT ROS /joint_states.
#
# These MCP/PIP values were reconstructed from the colored-marker video.
# DIP is intentionally excluded for now.
#
# The selected video frames occur consistently about 1.55 s after the
# first command publication at each target.
# ======================================================================

REFERENCE_OFFSET = 1.550

REFERENCE = [
    # target, MCP deg, PIP deg

    (100,   0.00,  0.00),
    (500,  13.44,  5.55),
    (1000, 30.85, 10.52),
    (1500, 46.46, 15.50),
    (2000, 58.88, 26.67),
    (2500, 59.85, 41.75),
    (3000, 60.97, 62.23),
    (3500, 64.35, 75.27),

    (3000, 52.24, 69.47),
    (2500, 46.77, 52.50),
    (2000, 37.93, 49.78),
    (1500, 37.92, 31.35),
    (1000, 34.04, 12.98),
    (500,  14.58,  7.93),
    (100,   0.00,  0.00),
]


MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

SPOOL_NAME = "index_motor_spool"
TENDON_NAME = "idx_flex_spatial"
ACTUATOR_NAME = "pos_index_tendon"


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
    motor_states = []

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

            interface_names = list(
                interfaces.interface_names
            )

            if "position" not in interface_names:
                continue

            j = interface_names.index("position")

            motor_states.append(
                (
                    t,
                    float(interfaces.values[j]),
                )
            )

    return (
        np.asarray(commands, dtype=float),
        np.asarray(motor_states, dtype=float),
    )


def compress_commands(commands):

    result = []

    previous = None

    for t, value in commands:

        if (
            previous is None
            or abs(value - previous) > 1e-9
        ):

            result.append(
                (t, value)
            )

            previous = value

    return np.asarray(
        result,
        dtype=float,
    )


def joint_info(model, name):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    return (
        model.jnt_qposadr[jid],
        model.jnt_dofadr[jid],
    )


def rate_step(
    position,
    target,
    dt,
):

    error = target - position

    if error > 0.0:

        movement = min(
            error,
            CLOSING_RATE * dt,
        )

        return (
            position + movement,
            movement / dt,
        )

    if error < 0.0:

        movement = min(
            -error,
            OPENING_RATE * dt,
        )

        return (
            position - movement,
            -movement / dt,
        )

    return position, 0.0


def rms(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    return math.sqrt(
        np.mean(values ** 2)
    )


def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python3 replay_index_dynamic_v1.py "
            "/path/to/rosbag_directory"
        )

        sys.exit(1)


    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    if not MODEL_PATH.exists():

        raise RuntimeError(
            f"Identified model does not exist:\n{MODEL_PATH}"
        )


    print()
    print("=" * 82)
    print(
        "RH8D INDEX DIGITAL TWIN — "
        "FULL DYNAMIC HARDWARE REPLAY V1"
    )
    print("=" * 82)

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Hardware bag:")
    print(bag)


    # ==================================================================
    # READ REAL HARDWARE COMMAND + ENCODER DATA
    # ==================================================================

    commands, hardware_motor = read_bag(
        bag
    )

    commands = compress_commands(
        commands
    )


    expected = np.asarray(
        [x[0] for x in REFERENCE],
        dtype=float,
    )

    if len(commands) != len(REFERENCE):

        raise RuntimeError(
            f"Expected {len(REFERENCE)} command levels, "
            f"found {len(commands)}"
        )


    if not np.allclose(
        commands[:, 1],
        expected,
    ):

        print()
        print("Commands found:")

        print(
            commands[:, 1]
        )

        raise RuntimeError(
            "Hardware command sequence does not match "
            "the reference experiment."
        )


    print()
    print(
        f"Command levels  = {len(commands)}"
    )

    print(
        f"Encoder samples = {len(hardware_motor)}"
    )


    # First command defines t = 0 of the replay.
    bag_t0 = commands[0, 0]

    command_times = (
        commands[:, 0] - bag_t0
    )

    command_targets = (
        commands[:, 1]
    )


    # ==================================================================
    # LOAD IDENTIFIED MUJOCO PLANT
    # ==================================================================

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(
        model
    )

    model.opt.gravity[:] = 0.0


    mcp_q, mcp_dof = joint_info(
        model,
        MCP_NAME,
    )

    pip_q, pip_dof = joint_info(
        model,
        PIP_NAME,
    )

    dip_q, dip_dof = joint_info(
        model,
        DIP_NAME,
    )

    spool_q, spool_dof = joint_info(
        model,
        SPOOL_NAME,
    )


    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAME,
    )

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )


    if aid < 0 or tid < 0:

        raise RuntimeError(
            "Index actuator/tendon not found."
        )


    print()
    print("Identified nominal parameters:")
    print(
        f"  motor delay        = "
        f"{MOTOR_DELAY*1000:.1f} ms"
    )
    print(
        f"  closing rate       = "
        f"{CLOSING_RATE:.1f} ticks/s"
    )
    print(
        f"  opening rate       = "
        f"{OPENING_RATE:.1f} ticks/s"
    )
    print(
        f"  encoder/spool      = "
        f"{METERS_PER_TICK*1e6:.3f} um/tick"
    )
    print(
        f"  tendon stiffness   = "
        f"{model.tendon_stiffness[tid]:.1f} N/m"
    )
    print(
        f"  tendon springrange = "
        f"{model.tendon_lengthspring[tid]}"
    )


    # ==================================================================
    # SETTLE FINGER OPEN
    # ==================================================================

    mujoco.mj_resetData(
        model,
        data,
    )

    data.ctrl[:] = 0.0


    for _ in range(
        int(2.0 / model.opt.timestep)
    ):

        data.qpos[spool_q] = 0.0
        data.qvel[spool_dof] = 0.0

        data.ctrl[aid] = 0.0

        mujoco.mj_step(
            model,
            data,
        )


    print()
    print("Open state after settling:")

    print(
        f"  MCP = "
        f"{math.degrees(data.qpos[mcp_q]):.3f} deg"
    )

    print(
        f"  PIP = "
        f"{math.degrees(data.qpos[pip_q]):.3f} deg"
    )

    print(
        f"  DIP = "
        f"{math.degrees(data.qpos[dip_q]):.3f} deg"
    )


    # ==================================================================
    # INITIAL REAL MOTOR POSITION
    # ==================================================================

    pre_mask = (
        (hardware_motor[:, 0] >= bag_t0 - 0.20)
        & (hardware_motor[:, 0] <= bag_t0 + 0.02)
    )

    if np.any(pre_mask):

        motor_position = float(
            np.median(
                hardware_motor[
                    pre_mask,
                    1,
                ]
            )
        )

    else:

        motor_position = OPEN_ENCODER


    current_target = command_targets[0]

    event_index = 1


    # Delayed command events.
    delayed_events = [
        (
            command_times[i] + MOTOR_DELAY,
            command_targets[i],
        )
        for i in range(
            1,
            len(commands),
        )
    ]


    sim_end = (
        command_times[-1]
        + REFERENCE_OFFSET
        + 0.30
    )

    dt = model.opt.timestep

    nsteps = int(
        math.ceil(
            sim_end / dt
        )
    )


    print()
    print(
        f"Replay duration = {sim_end:.3f} s"
    )

    print(
        f"MuJoCo steps    = {nsteps}"
    )

    print()
    print(
        "Running full command -> motor -> "
        "spool -> tendon -> finger replay..."
    )


    # ==================================================================
    # RECORD ARRAYS
    # ==================================================================

    times = [0.0]

    motor_prediction = [
        motor_position
    ]

    spool_history = [0.0]

    mcp_history = [
        math.degrees(
            data.qpos[mcp_q]
        )
    ]

    pip_history = [
        math.degrees(
            data.qpos[pip_q]
        )
    ]

    dip_history = [
        math.degrees(
            data.qpos[dip_q]
        )
    ]

    tendon_history = [
        data.ten_length[tid]
        * 1000.0
    ]


    # ==================================================================
    # DYNAMIC REPLAY
    # ==================================================================

    for step in range(
        1,
        nsteps + 1,
    ):

        t = step * dt


        while (
            event_index - 1
            < len(delayed_events)
            and delayed_events[
                event_index - 1
            ][0] <= t
        ):

            current_target = (
                delayed_events[
                    event_index - 1
                ][1]
            )

            event_index += 1


        previous_motor = motor_position

        motor_position, motor_velocity = (
            rate_step(
                motor_position,
                current_target,
                dt,
            )
        )


        # --------------------------------------------------------------
        # MEASURED HARDWARE ENCODER -> EFFECTIVE SPOOL
        # FOR FINGER-MECHANICS IDENTIFICATION
        # --------------------------------------------------------------
        #
        # IMPORTANT:
        #
        # `motor_position` above is STILL the prediction from the
        # identified 58 ms + directional-rate motor model.
        #
        # It is retained in motor_prediction and is still compared
        # against the real encoder for MOTOR-LAYER validation.
        #
        # But the finger mechanics are now driven from the actual
        # measured hardware encoder.
        #
        # hardware_motor[:, 0] is absolute bag time.
        # Simulation t is relative to bag_t0.
        #
        # Therefore:
        #
        #     hardware time = bag_t0 + t
        #
        # This separates:
        #
        #     command -> motor model
        #
        # from:
        #
        #     measured motor coordinate -> finger mechanics
        # --------------------------------------------------------------

        mechanics_motor_position = float(
            np.interp(
                bag_t0 + t,
                hardware_motor[:, 0],
                hardware_motor[:, 1],
            )
        )


        mechanics_prev_position = float(
            np.interp(
                bag_t0 + max(
                    0.0,
                    t - dt,
                ),
                hardware_motor[:, 0],
                hardware_motor[:, 1],
            )
        )


        mechanics_motor_velocity = (
            mechanics_motor_position
            -
            mechanics_prev_position
        ) / dt


        spool_position = max(
            0.0,
            (
                mechanics_motor_position
                -
                OPEN_ENCODER
            )
            *
            METERS_PER_TICK,
        )


        if spool_position <= 0.0:

            spool_velocity = 0.0

        else:

            spool_velocity = (
                mechanics_motor_velocity
                *
                METERS_PER_TICK
            )


        # The measured hardware encoder is the observed mechanical
        # boundary condition for this identification replay.
        #
        # We still overwrite qpos/qvel before every MuJoCo step,
        # exactly as the original free-space replay does.
        #
        # The position actuator only holds this coordinate during
        # the individual 1-ms integration step.

        data.qpos[spool_q] = (
            spool_position
        )

        data.qvel[spool_dof] = (
            spool_velocity
        )

        data.ctrl[aid] = (
            spool_position
        )


        mujoco.mj_step(
            model,
            data,
        )


        times.append(t)

        motor_prediction.append(
            motor_position
        )

        spool_history.append(
            spool_position * 1000.0
        )

        mcp_history.append(
            math.degrees(
                data.qpos[mcp_q]
            )
        )

        pip_history.append(
            math.degrees(
                data.qpos[pip_q]
            )
        )

        dip_history.append(
            math.degrees(
                data.qpos[dip_q]
            )
        )

        tendon_history.append(
            data.ten_length[tid]
            * 1000.0
        )


    times = np.asarray(times)

    motor_prediction = np.asarray(
        motor_prediction
    )

    spool_history = np.asarray(
        spool_history
    )

    mcp_history = np.asarray(
        mcp_history
    )

    pip_history = np.asarray(
        pip_history
    )

    dip_history = np.asarray(
        dip_history
    )

    tendon_history = np.asarray(
        tendon_history
    )


    # ==================================================================
    # MOTOR VALIDATION AGAINST /dynamic_joint_states
    # ==================================================================

    hw_t = (
        hardware_motor[:, 0]
        - bag_t0
    )

    hw_motor = (
        hardware_motor[:, 1]
    )

    motor_mask = (
        (hw_t >= 0.0)
        & (hw_t <= sim_end)
    )

    hw_t_score = hw_t[
        motor_mask
    ]

    hw_motor_score = hw_motor[
        motor_mask
    ]

    predicted_at_hw = np.interp(
        hw_t_score,
        times,
        motor_prediction,
    )

    motor_errors = (
        predicted_at_hw
        - hw_motor_score
    )


    motor_rmse = rms(
        motor_errors
    )

    motor_mae = float(
        np.median(
            np.abs(
                motor_errors
            )
        )
    )

    motor_p95 = float(
        np.percentile(
            np.abs(
                motor_errors
            ),
            95,
        )
    )


    # ==================================================================
    # JOINT VALIDATION AGAINST SYNCHRONIZED VIDEO
    # ==================================================================

    reference_rows = []

    mcp_errors = []
    pip_errors = []


    print()
    print("=" * 82)
    print(
        "DYNAMIC FINGER VALIDATION "
        "AT SYNCHRONIZED VIDEO FRAMES"
    )
    print("=" * 82)

    print()

    print(
        f"{'PHASE':7s} "
        f"{'TARGET':>6s} "
        f"{'MOTOR':>8s} "
        f"{'HW MCP':>8s} "
        f"{'SIM MCP':>8s} "
        f"{'HW PIP':>8s} "
        f"{'SIM PIP':>8s}"
    )

    print("-" * 82)


    for i, (
        target,
        hw_mcp,
        hw_pip,
    ) in enumerate(
        REFERENCE
    ):

        sample_time = (
            command_times[i]
            + REFERENCE_OFFSET
        )

        sim_motor = float(
            np.interp(
                sample_time,
                times,
                motor_prediction,
            )
        )

        sim_mcp = float(
            np.interp(
                sample_time,
                times,
                mcp_history,
            )
        )

        sim_pip = float(
            np.interp(
                sample_time,
                times,
                pip_history,
            )
        )

        sim_dip = float(
            np.interp(
                sample_time,
                times,
                dip_history,
            )
        )


        actual_encoder = float(
            np.interp(
                sample_time,
                hw_t,
                hw_motor,
            )
        )


        em = sim_mcp - hw_mcp
        ep = sim_pip - hw_pip

        mcp_errors.append(em)
        pip_errors.append(ep)


        phase = (
            "closing"
            if i <= 7
            else "opening"
        )


        print(
            f"{phase:7s} "
            f"{target:6.0f} "
            f"{sim_motor:8.1f} "
            f"{hw_mcp:8.2f} "
            f"{sim_mcp:8.2f} "
            f"{hw_pip:8.2f} "
            f"{sim_pip:8.2f}"
        )


        reference_rows.append({

            "phase": phase,

            "target_ticks":
                target,

            "sample_time_s":
                sample_time,

            "hardware_encoder_ticks":
                actual_encoder,

            "sim_motor_ticks":
                sim_motor,

            "hardware_mcp_deg":
                hw_mcp,

            "sim_mcp_deg":
                sim_mcp,

            "mcp_error_deg":
                em,

            "hardware_pip_deg":
                hw_pip,

            "sim_pip_deg":
                sim_pip,

            "pip_error_deg":
                ep,

            "sim_dip_deg":
                sim_dip,
        })


    combined_errors = (
        mcp_errors
        + pip_errors
    )


    # ==================================================================
    # SUMMARY
    # ==================================================================

    print()
    print("=" * 82)
    print("FULL DYNAMIC REPLAY RESULTS")
    print("=" * 82)

    print()
    print("MOTOR LAYER")
    print(
        f"  encoder RMSE       = "
        f"{motor_rmse:.2f} ticks"
    )

    print(
        f"  median abs error   = "
        f"{motor_mae:.2f} ticks"
    )

    print(
        f"  95% abs error      = "
        f"{motor_p95:.2f} ticks"
    )


    print()
    print("FINGER LAYER")

    print(
        f"  MCP RMSE           = "
        f"{rms(mcp_errors):.3f} deg"
    )

    print(
        f"  PIP RMSE           = "
        f"{rms(pip_errors):.3f} deg"
    )

    print(
        f"  combined MCP/PIP   = "
        f"{rms(combined_errors):.3f} deg"
    )

    print()
    print(
        "  DIP is simulated but remains excluded "
        "from quantitative validation."
    )


    # ==================================================================
    # SAVE FULL DYNAMIC TRAJECTORY
    # ==================================================================

    with open(
        DYNAMIC_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "time_s",
            "sim_motor_ticks",
            "spool_mm",
            "sim_mcp_deg",
            "sim_pip_deg",
            "sim_dip_deg",
            "tendon_coordinate_mm",
        ])

        for row in zip(
            times,
            motor_prediction,
            spool_history,
            mcp_history,
            pip_history,
            dip_history,
            tendon_history,
        ):

            writer.writerow(row)


    with open(
        REFERENCE_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                reference_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            reference_rows
        )


    # ==================================================================
    # MOTOR PLOT
    # ==================================================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        hw_t_score,
        hw_motor_score,
        label="Hardware encoder",
    )

    plt.plot(
        times,
        motor_prediction,
        label="Identified motor model",
    )

    for command_time, target in zip(
        command_times,
        command_targets,
    ):

        plt.scatter(
            command_time,
            target,
            s=15,
        )

    plt.xlabel(
        "Time since first command [s]"
    )

    plt.ylabel(
        "Index motor position [ticks]"
    )

    plt.title(
        "RH8D index motor: hardware vs identified model"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        MOTOR_PLOT,
        dpi=180,
    )

    plt.close()


    # ==================================================================
    # FINGER PLOT
    # ==================================================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        times,
        mcp_history,
        label="MuJoCo MCP",
    )

    plt.plot(
        times,
        pip_history,
        label="MuJoCo PIP",
    )


    sample_times = np.asarray([
        x["sample_time_s"]
        for x in reference_rows
    ])


    plt.scatter(
        sample_times,
        [
            x["hardware_mcp_deg"]
            for x in reference_rows
        ],
        marker="o",
        label="Hardware MCP video",
    )

    plt.scatter(
        sample_times,
        [
            x["hardware_pip_deg"]
            for x in reference_rows
        ],
        marker="x",
        label="Hardware PIP video",
    )


    plt.xlabel(
        "Time since first command [s]"
    )

    plt.ylabel(
        "Joint angle [deg]"
    )

    plt.title(
        "RH8D index finger: dynamic digital-twin replay"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        JOINT_PLOT,
        dpi=180,
    )

    plt.close()


    print()
    print("Saved:")
    print(DYNAMIC_CSV)
    print(REFERENCE_CSV)
    print(MOTOR_PLOT)
    print(JOINT_PLOT)

    print()
    print("=" * 82)
    print("DYNAMIC REPLAY FINISHED")
    print("=" * 82)


if __name__ == "__main__":
    main()
