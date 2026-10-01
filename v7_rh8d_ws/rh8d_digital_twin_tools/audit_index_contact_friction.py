#!/usr/bin/env python3

import sys
import csv
import math
from pathlib import Path

import numpy as np
import mujoco

import fit_index_contact_v1 as base
import fit_index_contact_v1_1 as corrected


# ============================================================
# CONTACT-FRICTION AUDIT
#
# PURPOSE:
#
# Keep the validated free-space hand and measured bottle
# geometry completely frozen.
#
# Vary ONLY:
#
#   - bottle/fingertip sliding-friction coefficient
#   - a few previously promising contact solref pairs
#
# This tests whether the observed contact/liftoff behavior
# is primarily a stick/slip problem.
# ============================================================


MODEL_PATH = (
    base.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_v1.xml"
)


RESULT_PATH = (
    base.WS
    / "rh8d_digital_twin_tools/results/"
      "index_contact_friction_audit.csv"
)


# ============================================================
# These three contact laws came from the previous cylinder
# experiment.
#
# We deliberately do NOT perform another huge stiffness grid
# yet.
# ============================================================

CONTACT_LAWS = [

    # best force-RMSE candidate
    (24000.0, 2.5),

    # softer / less spiky candidate
    (12800.0, 20.0),

    # more damped candidate
    (32000.0, 40.0),
]


# ============================================================
# Sliding friction coefficient.
#
# 0.8 was previously assumed.
#
# We explore below and above it because rubber/plastic-like
# interfaces may have substantially different effective
# friction depending on surface material.
# ============================================================

FRICTION_VALUES = [

    0.20,
    0.40,
    0.60,
    0.80,

    1.00,
    1.20,
    1.50,

    2.00,
    2.50,
    3.00,
]


HW_HOLD_FORCE = 0.14563
HW_ONSET_TICKS = 1020.0


def oid(
    model,
    typ,
    name,
):

    idx = mujoco.mj_name2id(
        model,
        typ,
        name,
    )

    if idx < 0:

        raise RuntimeError(
            f"Could not find: {name}"
        )

    return idx


def rmse(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    return math.sqrt(
        float(
            np.mean(
                values ** 2
            )
        )
    )


def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python3 "
            "audit_index_contact_friction.py "
            "/path/to/contact_rosbag_directory"
        )

        sys.exit(1)


    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    print()
    print("=" * 100)
    print(
        "RH8D INDEX — CYLINDER CONTACT "
        "FRICTION AUDIT"
    )
    print("=" * 100)

    print()
    print("Model:")
    print(MODEL_PATH)


    # ========================================================
    # HARDWARE DATA
    # ========================================================

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


    score_start = (
        regulation_start
        - base.PRECONTACT_SCORE_TIME
    )


    force_mask = (
        (force_data[:, 0] >= score_start)
        &
        (force_data[:, 0] <= hold_end)
    )


    force_times = (
        force_data[
            force_mask,
            0
        ]
    )


    hardware_force = (
        force_data[
            force_mask,
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


    # ========================================================
    # REAL FTS SAMPLE TIMES
    # ========================================================

    all_fts_times = (
        corrected.read_fts_times(
            bag
        )
    )


    fts_mask = (
        (all_fts_times >= approach_start)
        &
        (all_fts_times <= hold_end)
    )


    fts_times = (
        all_fts_times[
            fts_mask
        ]
    )


    # ========================================================
    # REAL MOTOR TRAJECTORY
    # ========================================================

    encoder_time = (
        encoder_data[:, 0]
    )


    encoder_position = (
        base.median_filter_5(
            encoder_data[:, 1]
        )
    )


    # ========================================================
    # MUJOCO
    # ========================================================

    model = (
        mujoco.MjModel.from_xml_path(
            str(MODEL_PATH)
        )
    )


    data = mujoco.MjData(
        model
    )


    fixture_gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        base.FIXTURE_GEOM,
    )


    tip_gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        base.TIP_GEOM,
    )


    spool_jid = oid(
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


    actuator_id = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        base.ACTUATOR,
    )


    print()
    print(
        "Frozen tendon stiffness = "
        f"{model.tendon_stiffness[0]:.1f} N/m"
    )


    print()
    print(
        f"Testing "
        f"{len(CONTACT_LAWS) * len(FRICTION_VALUES)} "
        "combinations..."
    )


    # ========================================================
    # IMPORTANT:
    #
    # importing fit_index_contact_v1_1 already replaces
    # base.contact_force with the corrected resultant-force
    # calculation.
    # ========================================================


    results = []


    for (
        contact_stiffness,
        contact_damping,
    ) in CONTACT_LAWS:


        for mu in FRICTION_VALUES:


            # ------------------------------------------------
            # Fixture priority is higher than fingertip geom,
            # so this sliding-friction value controls the pair.
            # ------------------------------------------------

            model.geom_friction[
                fixture_gid,
                0
            ] = mu


            (
                sim_times,
                raw_sim_force,
                sim_encoder,
                unexpected,
            ) = base.simulate_candidate(

                model,
                data,

                contact_stiffness,
                contact_damping,

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


            # =================================================
            # SAME SENSOR PIPELINE AS HARDWARE
            # =================================================

            sim_fts_raw = np.interp(
                fts_times,
                sim_times,
                raw_sim_force,
            )


            sim_fts_filtered = (
                corrected.causal_median_5(
                    sim_fts_raw
                )
            )


            sim_force = np.interp(
                force_times,
                fts_times,
                sim_fts_filtered,
            )


            errors = (
                sim_force
                - hardware_force
            )


            sim_hold = (
                sim_force[
                    hardware_hold_mask
                ]
            )


            hold_median = float(
                np.median(
                    sim_hold
                )
            )


            hold_mean = float(
                np.mean(
                    sim_hold
                )
            )


            hold_std = float(
                np.std(
                    sim_hold
                )
            )


            # =================================================
            # RAW CONTACT DUTY
            # =================================================

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


            contact_duty = float(
                np.mean(
                    raw_hold > 0.001
                )
            )


            raw_max = float(
                np.max(
                    raw_hold
                )
            )


            # =================================================
            # FILTERED 0.10-N THRESHOLD
            # =================================================

            crossings = np.where(
                sim_fts_filtered
                >= 0.10
            )[0]


            if len(crossings):

                j = crossings[0]


                threshold_time = (
                    fts_times[j]
                )


                threshold_ticks = float(
                    np.interp(
                        threshold_time,
                        sim_times,
                        sim_encoder,
                    )
                )


                onset_error = abs(
                    threshold_ticks
                    - HW_ONSET_TICKS
                )


            else:

                threshold_ticks = math.nan
                onset_error = math.inf


            hold_error = abs(
                hold_median
                - HW_HOLD_FORCE
            )


            force_rmse = rmse(
                errors
            )


            physically_acceptable = (

                hold_error <= 0.020

                and contact_duty >= 0.90

                and onset_error <= 25.0

                and raw_max <= 0.50
            )


            results.append({

                "friction_mu":
                    mu,

                "contact_stiffness":
                    contact_stiffness,

                "contact_damping":
                    contact_damping,

                "force_rmse_N":
                    force_rmse,

                "sim_hold_median_N":
                    hold_median,

                "sim_hold_mean_N":
                    hold_mean,

                "sim_hold_std_N":
                    hold_std,

                "hold_error_N":
                    hold_error,

                "raw_contact_duty":
                    contact_duty,

                "raw_hold_max_N":
                    raw_max,

                "sim_threshold_ticks":
                    threshold_ticks,

                "onset_error_ticks":
                    onset_error,

                "acceptable":
                    physically_acceptable,
            })


    # ========================================================
    # SAVE
    # ========================================================

    with open(
        RESULT_PATH,
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


    # ========================================================
    # SORT BY PHYSICAL RELEVANCE
    # ========================================================

    results.sort(
        key=lambda r:
            (
                r["hold_error_N"],
                -r["raw_contact_duty"],
                r["force_rmse_N"],
            )
    )


    print()
    print("=" * 112)
    print(
        "FRICTION SENSITIVITY RESULTS"
    )
    print("=" * 112)


    for rank, r in enumerate(
        results,
        start=1,
    ):

        print()

        print(
            f"#{rank:2d} "
            f"mu={r['friction_mu']:.2f}   "
            f"k={r['contact_stiffness']:.0f}   "
            f"d={r['contact_damping']:.1f}"
        )


        print(
            f"     hold="
            f"{r['sim_hold_median_N']:.4f} N   "
            f"error="
            f"{r['hold_error_N']:.4f} N   "
            f"std="
            f"{r['sim_hold_std_N']:.4f} N"
        )


        print(
            f"     duty="
            f"{100*r['raw_contact_duty']:.1f}%   "
            f"raw max="
            f"{r['raw_hold_max_N']:.3f} N   "
            f"onset="
            f"{r['sim_threshold_ticks']:.1f} ticks"
        )


        print(
            f"     trajectory RMSE="
            f"{r['force_rmse_N']:.5f} N   "
            f"acceptable="
            f"{r['acceptable']}"
        )


    good = [
        r
        for r in results
        if r["acceptable"]
    ]


    print()
    print("=" * 112)
    print("PHYSICALLY ACCEPTABLE")
    print("=" * 112)


    if not good:

        print(
            "NONE"
        )

    else:

        for r in good:

            print(
                f"mu={r['friction_mu']:.2f}   "
                f"k={r['contact_stiffness']:.0f}   "
                f"d={r['contact_damping']:.1f}   "
                f"hold={r['sim_hold_median_N']:.4f} N   "
                f"duty={100*r['raw_contact_duty']:.1f}%   "
                f"onset={r['sim_threshold_ticks']:.1f}"
            )


    print()
    print("Saved:")
    print(RESULT_PATH)

    print("=" * 112)


if __name__ == "__main__":
    main()
