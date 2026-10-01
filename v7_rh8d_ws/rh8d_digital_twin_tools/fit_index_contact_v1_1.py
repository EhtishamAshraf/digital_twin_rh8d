#!/usr/bin/env python3

import sys
import csv
import math
from pathlib import Path

import numpy as np
import mujoco
import rosbag2_py

# Reuse all of the already-tested bag parsing,
# phase detection and MuJoCo replay code.
import fit_index_contact_v1 as base


FTS_TOPIC = (
    "/rh8dr/fts_broadcaster/"
    "fingertip_force_sensors"
)

RESULT_DIR = (
    base.WS
    / "rh8d_digital_twin_tools/results"
)

GRID_CSV = (
    RESULT_DIR
    / "index_contact_v1_1_grid.csv"
)

TRACE_CSV = (
    RESULT_DIR
    / "index_contact_v1_1_best_trace.csv"
)


# ============================================================
# CORRECTION 1:
#
# Hardware net_force is derived from the magnitude of the
# 3-axis fingertip force vector.
#
# Therefore compare against resultant MuJoCo contact force:
#
# sqrt(Fnormal^2 + Ftangent1^2 + Ftangent2^2)
#
# rather than only wrench[0].
# ============================================================

def resultant_contact_force(
    model,
    data,
    fixture_gid,
    tip_gid,
):

    total_resultant = 0.0

    unexpected_contacts = []


    for i in range(
        data.ncon
    ):

        contact = data.contact[i]


        fixture_contact = (
            contact.geom1 == fixture_gid
            or
            contact.geom2 == fixture_gid
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

            unexpected_contacts.append(
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


        # Contact-frame force:
        #
        # [0] normal
        # [1] tangent direction 1
        # [2] tangent direction 2
        #
        # The Euclidean norm is invariant under rotation,
        # so no coordinate-frame conversion is needed for
        # one resultant-force magnitude.

        force_magnitude = float(
            np.linalg.norm(
                wrench[:3]
            )
        )


        total_resultant += (
            force_magnitude
        )


    return (
        total_resultant,
        unexpected_contacts,
    )


# Monkey-patch the existing simulation routine so it now
# uses the corrected force quantity.
base.contact_force = (
    resultant_contact_force
)


# ============================================================
# Read ONLY timestamps of the raw FTS messages.
#
# We use them to reproduce the same causal 5-sample median
# window that the hardware contact script used.
# ============================================================

def read_fts_times(
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


    times = []

    t0 = None


    while reader.has_next():

        topic, raw, stamp = (
            reader.read_next()
        )


        if t0 is None:

            t0 = stamp


        if topic == FTS_TOPIC:

            times.append(
                (
                    stamp - t0
                ) * 1e-9
            )


    return np.asarray(
        times,
        dtype=float,
    )


# ============================================================
# Exact causal median-of-last-5 operation.
#
# This mirrors:
#
# deque(maxlen=5)
# median(force_history)
#
# from the hardware contact experiment.
# ============================================================

def causal_median_5(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    result = np.empty_like(
        values
    )


    for i in range(
        len(values)
    ):

        lo = max(
            0,
            i - 4,
        )

        result[i] = np.median(
            values[
                lo:i + 1
            ]
        )


    return result


def rmse(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    return math.sqrt(
        np.mean(
            values ** 2
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
            "fit_index_contact_v1_1.py "
            "/path/to/contact_rosbag_directory"
        )

        sys.exit(1)


    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    print()
    print("=" * 82)
    print(
        "RH8D INDEX CONTACT SYSTEM "
        "IDENTIFICATION V1.1"
    )
    print("=" * 82)

    print()
    print(
        "CORRECTIONS FROM V1:"
    )

    print(
        "  1. MuJoCo resultant |F| "
        "instead of normal force only"
    )

    print(
        "  2. causal median-of-5 "
        "sensor-equivalent filtering"
    )


    # --------------------------------------------------------
    # Read hardware data using existing validated parser.
    # --------------------------------------------------------

    (
        states,
        force_data,
        encoder_data,
    ) = base.read_bag(
        bag
    )


    transitions = (
        base.compress_states(
            states
        )
    )


    (
        approach_start,
        regulation_start,
        hold_start,
        hold_end,
    ) = base.find_intervals(
        transitions
    )


    print()
    print(
        "Clean hardware interval:"
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


    score_start = (
        regulation_start
        - base.PRECONTACT_SCORE_TIME
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
            0
        ]
    )

    hardware_force = (
        force_data[
            score_mask,
            1
        ]
    )


    hardware_hold_mask = (
        (force_times >= hold_start)
        &
        (force_times <= hold_end)
    )


    hardware_hold = (
        hardware_force[
            hardware_hold_mask
        ]
    )


    print()
    print(
        f"Hardware hold median = "
        f"{np.median(hardware_hold):.5f} N"
    )

    print(
        f"Hardware hold mean   = "
        f"{np.mean(hardware_hold):.5f} N"
    )


    # --------------------------------------------------------
    # Actual raw FTS sampling times.
    #
    # Include history before scoring starts so the initial
    # five-sample filter window is well-defined.
    # --------------------------------------------------------

    all_fts_times = read_fts_times(
        bag
    )


    fts_mask = (
        (all_fts_times
         >= approach_start)
        &
        (all_fts_times
         <= hold_end)
    )


    fts_times = (
        all_fts_times[
            fts_mask
        ]
    )


    if len(
        fts_times
    ) < 20:

        raise RuntimeError(
            "Not enough FTS timestamps "
            "in clean interval."
        )


    print()
    print(
        f"FTS samples in clean interval = "
        f"{len(fts_times)}"
    )

    print(
        f"Median FTS period             = "
        f"{np.median(np.diff(fts_times))*1000:.2f} ms"
    )


    # --------------------------------------------------------
    # Real motor encoder trajectory.
    # --------------------------------------------------------

    encoder_time = (
        encoder_data[
            :,
            0
        ]
    )

    encoder_position = (
        base.median_filter_5(
            encoder_data[
                :,
                1
            ]
        )
    )


    # --------------------------------------------------------
    # MuJoCo setup.
    # --------------------------------------------------------

    model = (
        mujoco.MjModel.from_xml_path(
            str(
                base.MODEL_PATH
            )
        )
    )

    data = mujoco.MjData(
        model
    )


    fixture_gid = base.obj_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        base.FIXTURE_GEOM,
    )


    tip_gid = base.obj_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        base.TIP_GEOM,
    )


    spool_jid = base.obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        base.SPOOL_JOINT,
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


    actuator_id = base.obj_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        base.ACTUATOR,
    )


    candidates = [
        (k, d)
        for k in base.STIFFNESS_VALUES
        for d in base.DAMPING_VALUES
    ]


    print()
    print(
        f"Testing {len(candidates)} "
        "contact models..."
    )


    results = []


    for number, (
        stiffness,
        damping,
    ) in enumerate(
        candidates,
        start=1,
    ):


        (
            sim_times,
            raw_sim_force,
            sim_encoder,
            unexpected,
        ) = base.simulate_candidate(

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


        # ====================================================
        # Reproduce hardware force-measurement pipeline.
        #
        # 1. sample simulation at real FTS timestamps
        # 2. causal median of most recent 5 samples
        # 3. interpolate that filtered signal to the
        #    experiment net_force publication timestamps
        # ====================================================

        sim_at_fts_raw = np.interp(
            fts_times,
            sim_times,
            raw_sim_force,
        )


        sim_at_fts_filtered = (
            causal_median_5(
                sim_at_fts_raw
            )
        )


        sim_at_hw_times = np.interp(
            force_times,
            fts_times,
            sim_at_fts_filtered,
        )


        errors = (
            sim_at_hw_times
            - hardware_force
        )


        sim_hold = (
            sim_at_hw_times[
                hardware_hold_mask
            ]
        )


        # ----------------------------------------------------
        # Sensor-equivalent 0.10-N crossing.
        # ----------------------------------------------------

        crossing = np.where(
            sim_at_fts_filtered
            >= 0.10
        )[0]


        if len(
            crossing
        ):

            j = crossing[0]

            threshold_time = float(
                fts_times[j]
            )


            threshold_ticks = float(
                np.interp(
                    threshold_time,
                    sim_times,
                    sim_encoder,
                )
            )

        else:

            threshold_time = math.nan

            threshold_ticks = math.nan


        # ----------------------------------------------------
        # Raw-contact duty cycle during hold.
        #
        # This helps distinguish a stable contact from
        # repeated contact/liftoff chatter.
        # ----------------------------------------------------

        raw_hold_mask = (
            (sim_times >= hold_start)
            &
            (sim_times <= hold_end)
        )


        raw_hold = (
            raw_sim_force[
                raw_hold_mask
            ]
        )


        duty = float(
            np.mean(
                raw_hold > 0.001
            )
        )


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
                        hardware_hold
                    )
                ),

            "sim_filtered_hold_median_N":
                float(
                    np.median(
                        sim_hold
                    )
                ),

            "sim_filtered_hold_mean_N":
                float(
                    np.mean(
                        sim_hold
                    )
                ),

            "raw_hold_max_N":
                float(
                    np.max(
                        raw_hold
                    )
                ),

            "raw_contact_duty":
                duty,

            "sim_010_threshold_ticks":
                threshold_ticks,

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
            number % 8 == 0
            or number
            == len(candidates)
        ):

            best = min(
                results,
                key=lambda r:
                    r["force_rmse_N"],
            )

            print(
                f"[{number:2d}/"
                f"{len(candidates)}] "
                f"best RMSE="
                f"{best['force_rmse_N']:.5f} N  "
                f"hold="
                f"{best['sim_filtered_hold_median_N']:.4f} N"
            )


    results.sort(
        key=lambda r:
            r["force_rmse_N"]
    )


    print()
    print("=" * 82)
    print(
        "TOP 15 CORRECTED CONTACT MODELS"
    )
    print("=" * 82)


    for rank, r in enumerate(
        results[:15],
        start=1,
    ):

        print()

        print(
            f"#{rank:2d} "
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
            f"     filtered hold: "
            f"HW="
            f"{r['hardware_hold_median_N']:.4f} N   "
            f"SIM="
            f"{r['sim_filtered_hold_median_N']:.4f} N"
        )

        print(
            f"     raw max="
            f"{r['raw_hold_max_N']:.4f} N   "
            f"contact duty="
            f"{100*r['raw_contact_duty']:.1f}%"
        )

        print(
            f"     filtered 0.10-N crossing="
            f"{r['sim_010_threshold_ticks']:.1f} ticks"
        )


        if r[
            "unexpected_contacts"
        ]:

            print(
                "     WARNING: "
                + r[
                    "unexpected_contacts"
                ]
            )


    # --------------------------------------------------------
    # Save complete corrected grid.
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
    # Re-run best candidate and save raw + filtered trace.
    # --------------------------------------------------------

    best = results[0]


    (
        sim_times,
        raw_sim_force,
        sim_encoder,
        unexpected,
    ) = base.simulate_candidate(

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


    sim_at_fts_raw = np.interp(
        fts_times,
        sim_times,
        raw_sim_force,
    )

    sim_at_fts_filtered = (
        causal_median_5(
            sim_at_fts_raw
        )
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
            "fts_time_s",
            "encoder_ticks",
            "sim_raw_resultant_force_N",
            "sim_filtered_resultant_force_N",
            "hardware_net_force_N",
        ])


        for (
            t,
            raw_force,
            filtered_force,
        ) in zip(
            fts_times,
            sim_at_fts_raw,
            sim_at_fts_filtered,
        ):

            enc = float(
                np.interp(
                    t,
                    sim_times,
                    sim_encoder,
                )
            )


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
                raw_force,
                filtered_force,
                hw,
            ])


    print()
    print("=" * 82)
    print(
        "BEST CORRECTED CONTACT MODEL"
    )
    print("=" * 82)

    print()

    print(
        f"Force RMSE              = "
        f"{best['force_rmse_N']:.5f} N"
    )

    print(
        f"Force MAE               = "
        f"{best['force_mae_N']:.5f} N"
    )

    print(
        f"Direct stiffness        = "
        f"{best['stiffness']:.1f}"
    )

    print(
        f"Direct damping          = "
        f"{best['damping']:.1f}"
    )

    print(
        f"Hardware hold median    = "
        f"{best['hardware_hold_median_N']:.4f} N"
    )

    print(
        f"Filtered sim hold median= "
        f"{best['sim_filtered_hold_median_N']:.4f} N"
    )

    print(
        f"Raw contact duty        = "
        f"{100*best['raw_contact_duty']:.1f}%"
    )

    print(
        f"Filtered 0.10-N crossing= "
        f"{best['sim_010_threshold_ticks']:.1f} ticks"
    )

    print()
    print("Saved:")
    print(GRID_CSV)
    print(TRACE_CSV)

    print("=" * 82)


if __name__ == "__main__":
    main()
