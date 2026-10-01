#!/usr/bin/env python3

import sys
import math
import itertools
import csv
from pathlib import Path

import numpy as np
import mujoco

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# ============================================================
# PATHS
# ============================================================

WS = Path.home() / (
    "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
)

MODEL_PATH = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_v1.xml"
)

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GRID_CSV = (
    RESULT_DIR
    / "index_contact_v1_grid.csv"
)

TRACE_CSV = (
    RESULT_DIR
    / "index_contact_v1_best_trace.csv"
)


# ============================================================
# ROS TOPICS
# ============================================================

STATE_TOPIC = (
    "/rh8dr/index_contact_sysid/state"
)

FORCE_TOPIC = (
    "/rh8dr/index_contact_sysid/net_force"
)

DYNAMIC_TOPIC = (
    "/dynamic_joint_states"
)

INDEX_MOTOR = "index_flexion_motor"


# ============================================================
# IDENTIFIED FREE-SPACE PARAMETERS
# ============================================================

OPEN_ENCODER = 100.0

METERS_PER_TICK = 6.0e-6


# ============================================================
# MUJOCO OBJECTS
# ============================================================

SPOOL_JOINT = "index_motor_spool"

ACTUATOR = "pos_index_tendon"

FIXTURE_GEOM = "index_contact_fixture"

TIP_GEOM = "collision_idx_tip"


# ============================================================
# CONTACT PARAMETER SEARCH
#
# MuJoCo direct-format solref:
#
#     solref = [-stiffness, -damping]
#
# These are EFFECTIVE simulator contact coefficients.
# They are not claimed material Young's modulus values.
# ============================================================

STIFFNESS_VALUES = [
    100.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
    16000.0,
]

DAMPING_VALUES = [
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
]


# Include a little pre-contact data.
PRECONTACT_SCORE_TIME = 0.50


def read_bag(
    bag_path,
):

    reader = (
        rosbag2_py.SequentialReader()
    )

    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(bag_path),
            storage_id="sqlite3",
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )


    topic_types = {
        t.name: t.type
        for t
        in reader.get_all_topics_and_types()
    }


    states = []

    forces = []

    encoders = []


    t0 = None


    while reader.has_next():

        topic, raw, stamp = (
            reader.read_next()
        )


        if t0 is None:

            t0 = stamp


        t = (
            stamp - t0
        ) * 1e-9


        # ----------------------------------------------------
        # Experiment phase
        # ----------------------------------------------------

        if topic == STATE_TOPIC:

            msg = deserialize_message(
                raw,
                get_message(
                    topic_types[topic]
                ),
            )

            states.append(
                (
                    t,
                    msg.data,
                )
            )


        # ----------------------------------------------------
        # Baseline-subtracted measured force
        # ----------------------------------------------------

        elif topic == FORCE_TOPIC:

            msg = deserialize_message(
                raw,
                get_message(
                    topic_types[topic]
                ),
            )

            forces.append(
                (
                    t,
                    float(msg.data),
                )
            )


        # ----------------------------------------------------
        # Real motor encoder
        # ----------------------------------------------------

        elif topic == DYNAMIC_TOPIC:

            msg = deserialize_message(
                raw,
                get_message(
                    topic_types[topic]
                ),
            )

            names = list(
                msg.joint_names
            )

            if INDEX_MOTOR not in names:
                continue


            i = names.index(
                INDEX_MOTOR
            )

            interfaces = (
                msg.interface_values[i]
            )

            interface_names = list(
                interfaces.interface_names
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
                        interfaces.values[j]
                    ),
                )
            )


    return (
        states,
        np.asarray(
            forces,
            dtype=float,
        ),
        np.asarray(
            encoders,
            dtype=float,
        ),
    )


def compress_states(
    states,
):

    result = []

    previous = None


    for t, state in states:

        if state != previous:

            result.append(
                (
                    t,
                    state,
                )
            )

            previous = state


    return result


def find_intervals(
    transitions,
):

    approach_start = None

    regulation_start = None

    hold_start = None

    hold_end = None


    for i, (
        t,
        state,
    ) in enumerate(
        transitions
    ):

        if (
            approach_start is None
            and state == "APPROACH"
        ):

            approach_start = t


        elif (
            approach_start
            is not None
            and regulation_start is None
            and state
            == "FORCE_REGULATION"
        ):

            regulation_start = t


        elif (
            regulation_start
            is not None
            and hold_start is None
            and state == "FORCE_HOLD"
        ):

            hold_start = t


        elif (
            hold_start
            is not None
            and t > hold_start
        ):

            hold_end = t

            break


    if (
        approach_start is None
        or regulation_start is None
        or hold_start is None
        or hold_end is None
    ):

        raise RuntimeError(
            "Could not identify first clean "
            "contact experiment interval."
        )


    return (
        approach_start,
        regulation_start,
        hold_start,
        hold_end,
    )


def obj_id(
    model,
    objtype,
    name,
):

    idx = mujoco.mj_name2id(
        model,
        objtype,
        name,
    )

    if idx < 0:

        raise RuntimeError(
            f"MuJoCo object not found: "
            f"{name}"
        )

    return idx


def median_filter_5(
    x,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    result = np.empty_like(
        x
    )


    for i in range(
        len(x)
    ):

        lo = max(
            0,
            i - 2,
        )

        hi = min(
            len(x),
            i + 3,
        )

        result[i] = np.median(
            x[lo:hi]
        )


    return result


def contact_force(
    model,
    data,
    fixture_gid,
    tip_gid,
):

    total_normal = 0.0

    other_contacts = []


    for i in range(
        data.ncon
    ):

        contact = data.contact[i]


        fixture_contact = (
            contact.geom1
            == fixture_gid
            or
            contact.geom2
            == fixture_gid
        )


        if not fixture_contact:
            continue


        other_gid = (
            contact.geom2
            if contact.geom1
            == fixture_gid
            else contact.geom1
        )


        if other_gid != tip_gid:

            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                other_gid,
            )

            other_contacts.append(
                name
            )

            continue


        wrench = np.zeros(
            6,
            dtype=float,
        )


        mujoco.mj_contactForce(
            model,
            data,
            i,
            wrench,
        )


        # Component 0 is normal contact force
        # in the MuJoCo contact frame.
        total_normal += max(
            0.0,
            float(wrench[0]),
        )


    return (
        total_normal,
        other_contacts,
    )


def simulate_candidate(
    model,
    data,

    stiffness,
    damping,

    approach_start,
    hold_end,

    encoder_time,
    encoder_position,

    fixture_gid,
    tip_gid,

    spool_qadr,
    spool_dof,
    actuator_id,
):

    # --------------------------------------------------------
    # Candidate contact law
    # --------------------------------------------------------

    model.geom_solref[
        fixture_gid,
        0,
    ] = -float(
        stiffness
    )

    model.geom_solref[
        fixture_gid,
        1,
    ] = -float(
        damping
    )


    model.opt.gravity[:] = 0.0


    mujoco.mj_resetData(
        model,
        data,
    )

    data.ctrl[:] = 0.0


    dt = model.opt.timestep


    # --------------------------------------------------------
    # Settle open before hardware approach
    # --------------------------------------------------------

    for _ in range(
        int(
            1.5 / dt
        )
    ):

        data.qpos[
            spool_qadr
        ] = 0.0

        data.qvel[
            spool_dof
        ] = 0.0

        data.ctrl[
            actuator_id
        ] = 0.0


        mujoco.mj_step(
            model,
            data,
        )


    duration = (
        hold_end
        - approach_start
    )


    steps = int(
        math.ceil(
            duration / dt
        )
    )


    times = np.empty(
        steps + 1,
        dtype=float,
    )

    forces = np.empty(
        steps + 1,
        dtype=float,
    )

    encoders = np.empty(
        steps + 1,
        dtype=float,
    )


    unexpected_contacts = set()


    previous_spool = 0.0


    for step in range(
        steps + 1
    ):

        local_time = (
            step * dt
        )

        bag_time = (
            approach_start
            + local_time
        )


        encoder = float(
            np.interp(
                bag_time,
                encoder_time,
                encoder_position,
            )
        )


        spool = max(
            0.0,
            (
                encoder
                - OPEN_ENCODER
            )
            * METERS_PER_TICK,
        )


        if step == 0:

            spool_velocity = 0.0

        else:

            spool_velocity = (
                spool
                - previous_spool
            ) / dt


        previous_spool = spool


        # ----------------------------------------------------
        # Measured hardware motor position is imposed.
        #
        # This isolates contact mechanics from motor-model
        # identification.
        # ----------------------------------------------------

        data.qpos[
            spool_qadr
        ] = spool

        data.qvel[
            spool_dof
        ] = spool_velocity

        data.ctrl[
            actuator_id
        ] = spool


        mujoco.mj_step(
            model,
            data,
        )


        force, others = contact_force(
            model,
            data,
            fixture_gid,
            tip_gid,
        )


        for name in others:

            unexpected_contacts.add(
                str(name)
            )


        times[
            step
        ] = bag_time

        forces[
            step
        ] = force

        encoders[
            step
        ] = encoder


    return (
        times,
        forces,
        encoders,
        unexpected_contacts,
    )


def rmse(
    x,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    return math.sqrt(
        np.mean(
            x * x
        )
    )


def main():

    if len(
        sys.argv
    ) != 2:

        print(
            "Usage:"
        )

        print(
            "  python3 "
            "fit_index_contact_v1.py "
            "/path/to/contact_rosbag_directory"
        )

        sys.exit(1)


    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    print()
    print("=" * 80)
    print(
        "RH8D INDEX CONTACT SYSTEM "
        "IDENTIFICATION V1"
    )
    print("=" * 80)

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Hardware bag:")
    print(bag)


    states, force_data, encoder_data = (
        read_bag(
            bag
        )
    )


    transitions = compress_states(
        states
    )


    (
        approach_start,
        regulation_start,
        hold_start,
        hold_end,
    ) = find_intervals(
        transitions
    )


    print()
    print("Clean experiment interval:")

    print(
        f"  APPROACH start       = "
        f"{approach_start:.6f} s"
    )

    print(
        f"  first regulation     = "
        f"{regulation_start:.6f} s"
    )

    print(
        f"  first FORCE_HOLD     = "
        f"{hold_start:.6f} s"
    )

    print(
        f"  first hold end       = "
        f"{hold_end:.6f} s"
    )


    # --------------------------------------------------------
    # Reject everything after the first clean hold.
    # --------------------------------------------------------

    score_start = (
        regulation_start
        - PRECONTACT_SCORE_TIME
    )


    score_mask = (
        (force_data[:, 0]
         >= score_start)
        &
        (force_data[:, 0]
         <= hold_end)
    )


    force_times = (
        force_data[
            score_mask,
            0,
        ]
    )

    hardware_force = (
        force_data[
            score_mask,
            1,
        ]
    )


    hold_mask_hw = (
        (force_data[:, 0]
         >= hold_start)
        &
        (force_data[:, 0]
         <= hold_end)
    )


    hardware_hold_force = (
        force_data[
            hold_mask_hw,
            1,
        ]
    )


    print()
    print(
        f"Hardware first-hold median force = "
        f"{np.median(hardware_hold_force):.4f} N"
    )

    print(
        f"Hardware first-hold mean force   = "
        f"{np.mean(hardware_hold_force):.4f} N"
    )

    print(
        f"Hardware first-hold std          = "
        f"{np.std(hardware_hold_force):.4f} N"
    )


    # --------------------------------------------------------
    # Encoder filtering
    # --------------------------------------------------------

    encoder_time = (
        encoder_data[
            :,
            0
        ]
    )

    encoder_position = median_filter_5(
        encoder_data[
            :,
            1
        ]
    )


    # --------------------------------------------------------
    # MuJoCo setup
    # --------------------------------------------------------

    model = (
        mujoco.MjModel.from_xml_path(
            str(MODEL_PATH)
        )
    )

    data = mujoco.MjData(
        model
    )


    fixture_gid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FIXTURE_GEOM,
    )


    tip_gid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        TIP_GEOM,
    )


    spool_jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_JOINT,
    )


    spool_qadr = (
        model.jnt_qposadr[
            spool_jid
        ]
    )

    spool_dof = (
        model.jnt_dofadr[
            spool_jid
        ]
    )


    actuator_id = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR,
    )


    candidates = list(
        itertools.product(
            STIFFNESS_VALUES,
            DAMPING_VALUES,
        )
    )


    print()
    print(
        f"Testing {len(candidates)} "
        "contact parameter combinations..."
    )


    results = []


    for i, (
        stiffness,
        damping,
    ) in enumerate(
        candidates,
        start=1,
    ):

        (
            sim_times,
            sim_force,
            sim_encoder,
            unexpected,
        ) = simulate_candidate(

            model,
            data,

            stiffness,
            damping,

            approach_start,
            hold_end,

            encoder_time,
            encoder_position,

            fixture_gid,
            tip_gid,

            spool_qadr,
            spool_dof,
            actuator_id,
        )


        sim_at_hw = np.interp(
            force_times,
            sim_times,
            sim_force,
        )


        errors = (
            sim_at_hw
            - hardware_force
        )


        hold_mask_sim = (
            (sim_times >= hold_start)
            &
            (sim_times <= hold_end)
        )


        sim_hold = (
            sim_force[
                hold_mask_sim
            ]
        )


        # ----------------------------------------------------
        # Simulated crossing of same 0.10 N threshold
        # --------------------------------------------------------

        threshold_idx = np.where(
            sim_force
            >= 0.10
        )[0]


        if len(
            threshold_idx
        ) > 0:

            j = threshold_idx[0]

            threshold_time = float(
                sim_times[j]
            )

            threshold_ticks = float(
                sim_encoder[j]
            )

        else:

            threshold_time = math.nan

            threshold_ticks = math.nan


        result = {

            "force_rmse_N":
                rmse(
                    errors
                ),

            "force_mae_N":
                float(
                    np.mean(
                        np.abs(
                            errors
                        )
                    )
                ),

            "hardware_hold_median_N":
                float(
                    np.median(
                        hardware_hold_force
                    )
                ),

            "sim_hold_median_N":
                float(
                    np.median(
                        sim_hold
                    )
                ),

            "sim_threshold_ticks":
                threshold_ticks,

            "sim_threshold_time_s":
                threshold_time,

            "stiffness":
                stiffness,

            "damping":
                damping,

            "unexpected_contacts":
                ",".join(
                    sorted(
                        unexpected
                    )
                ),
        }


        results.append(
            result
        )


        if (
            i % 8 == 0
            or i
            == len(candidates)
        ):

            best = min(
                results,
                key=lambda r:
                    r["force_rmse_N"],
            )

            print(
                f"[{i:2d}/"
                f"{len(candidates)}] "
                f"best RMSE="
                f"{best['force_rmse_N']:.5f} N"
            )


    results.sort(
        key=lambda r:
            r["force_rmse_N"]
    )


    print()
    print("=" * 80)
    print(
        "TOP 15 CONTACT MODELS"
    )
    print("=" * 80)


    for rank, r in enumerate(
        results[:15],
        start=1,
    ):

        print()

        print(
            f"#{rank:2d}  "
            f"RMSE="
            f"{r['force_rmse_N']:.5f} N   "
            f"MAE="
            f"{r['force_mae_N']:.5f} N"
        )

        print(
            f"     stiffness="
            f"{r['stiffness']:8.1f}   "
            f"damping="
            f"{r['damping']:7.1f}"
        )

        print(
            f"     hold force: "
            f"HW="
            f"{r['hardware_hold_median_N']:.4f} N   "
            f"SIM="
            f"{r['sim_hold_median_N']:.4f} N"
        )

        print(
            f"     simulated 0.10-N crossing = "
            f"{r['sim_threshold_ticks']:.1f} ticks"
        )

        if r[
            "unexpected_contacts"
        ]:

            print(
                "     WARNING other fixture contacts: "
                + r[
                    "unexpected_contacts"
                ]
            )


    # --------------------------------------------------------
    # Save grid
    # --------------------------------------------------------

    with open(
        GRID_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # --------------------------------------------------------
    # Re-run best candidate and save trace
    # --------------------------------------------------------

    best = results[0]


    (
        best_times,
        best_force,
        best_encoder,
        unexpected,
    ) = simulate_candidate(

        model,
        data,

        best["stiffness"],
        best["damping"],

        approach_start,
        hold_end,

        encoder_time,
        encoder_position,

        fixture_gid,
        tip_gid,

        spool_qadr,
        spool_dof,
        actuator_id,
    )


    with open(
        TRACE_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "bag_time_s",
            "encoder_ticks",
            "sim_contact_force_N",
            "hardware_net_force_N",
        ])


        for t, enc, sf in zip(
            best_times,
            best_encoder,
            best_force,
        ):

            if (
                t
                < score_start
            ):
                hw = math.nan

            else:

                hw = float(
                    np.interp(
                        t,
                        force_data[:, 0],
                        force_data[:, 1],
                    )
                )


            writer.writerow([
                t,
                enc,
                sf,
                hw,
            ])


    print()
    print("=" * 80)
    print(
        "BEST CONTACT MODEL"
    )
    print("=" * 80)

    print()

    print(
        f"Force RMSE            = "
        f"{best['force_rmse_N']:.5f} N"
    )

    print(
        f"Force MAE             = "
        f"{best['force_mae_N']:.5f} N"
    )

    print(
        f"Direct stiffness      = "
        f"{best['stiffness']:.1f}"
    )

    print(
        f"Direct damping        = "
        f"{best['damping']:.1f}"
    )

    print(
        f"Hardware hold median  = "
        f"{best['hardware_hold_median_N']:.4f} N"
    )

    print(
        f"Sim hold median       = "
        f"{best['sim_hold_median_N']:.4f} N"
    )

    print(
        f"Sim 0.10-N crossing   = "
        f"{best['sim_threshold_ticks']:.1f} ticks"
    )

    print()

    print("Saved:")
    print(GRID_CSV)
    print(TRACE_CSV)

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
