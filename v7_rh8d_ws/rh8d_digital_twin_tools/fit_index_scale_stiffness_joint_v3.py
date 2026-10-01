#!/usr/bin/env python3

import sys
import csv
import math
from pathlib import Path

import numpy as np
import mujoco

import fit_index_contact_v1 as base
import fit_index_contact_v1_1 as corrected


# =====================================================================
# JOINT FREE-SPACE + CONTACT IDENTIFICATION
#
# Identify:
#
#   encoder -> effective spool scale
#   tendon stiffness
#
# simultaneously.
#
# Motivation:
# free-space data alone does not strongly separate these parameters.
# Contact data adds load information and can resolve that ambiguity.
# =====================================================================


FREE_MODEL = (
    base.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_identified.xml"
)

CONTACT_MODEL = (
    base.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_v1.xml"
)

OUT = (
    base.WS
    / "rh8d_digital_twin_tools/results/"
      "index_scale_stiffness_joint_v3.csv"
)


OPEN_ENCODER = 100.0

TENDON_NAME = "idx_flex_spatial"

SPOOL_NAME = "index_motor_spool"
ACTUATOR_NAME = "pos_index_tendon"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"


# =====================================================================
# HARDWARE VIDEO REFERENCE
#
# encoder, MCP deg, PIP deg
#
# Same physical sequence used previously.
# =====================================================================

FREE_REFERENCE = [

    (100.0,   0.00,  0.00),

    (500.0,  13.44,  5.55),
    (1000.0, 30.85, 10.52),
    (1500.0, 46.46, 15.50),
    (2000.0, 58.88, 26.67),

    (2466.0, 59.85, 41.75),
    (2961.0, 60.97, 62.23),
    (3466.0, 64.35, 75.27),

    (3000.0, 52.24, 69.47),
    (2500.0, 46.77, 52.50),
    (2000.0, 37.93, 49.78),
    (1500.0, 37.92, 31.35),
    (1000.0, 34.04, 12.98),
    (500.0,  14.58,  7.93),

    (100.0,   0.00,  0.00),
]


# =====================================================================
# SEARCH
# =====================================================================

SCALE_UM_PER_TICK = [

    4.0,
    4.4,
    4.8,
    5.2,
    5.6,
    6.0,
    6.4,
]


TENDON_STIFFNESS = [

    1200.0,
    1600.0,
    1800.0,
    2000.0,
    2400.0,
    2800.0,
    3200.0,
]


# We already investigated the contact law extensively.
# Use three representative laws rather than another huge grid.

CONTACT_LAWS = [

    (12800.0, 20.0),

    (24000.0, 2.5),

    (32000.0, 40.0),
]


# Friction audit plateaued beyond roughly 1.2.
CONTACT_FRICTION = 1.2


HW_HOLD = 0.14563
HW_ONSET = 1020.0


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
            f"Missing MuJoCo object: {name}"
        )

    return idx


def rms(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    return math.sqrt(
        float(
            np.mean(
                x * x
            )
        )
    )


# =====================================================================
# FREE SPACE
# =====================================================================

def free_space_error(
    scale_m_per_tick,
    tendon_stiffness,
):

    model = mujoco.MjModel.from_xml_path(
        str(FREE_MODEL)
    )

    data = mujoco.MjData(
        model
    )

    model.opt.gravity[:] = 0.0


    tendon_id = oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    model.tendon_stiffness[
        tendon_id
    ] = tendon_stiffness


    spool_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_NAME,
    )

    spool_q = (
        model.jnt_qposadr[
            spool_jid
        ]
    )

    spool_dof = (
        model.jnt_dofadr[
            spool_jid
        ]
    )


    actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAME,
    )


    mcp_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        MCP_NAME,
    )

    pip_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        PIP_NAME,
    )


    mcp_q = (
        model.jnt_qposadr[
            mcp_jid
        ]
    )

    pip_q = (
        model.jnt_qposadr[
            pip_jid
        ]
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    # -------------------------------------------------------------
    # Initial open settling
    # -------------------------------------------------------------

    for _ in range(
        int(
            1.5
            / model.opt.timestep
        )
    ):

        data.qpos[
            spool_q
        ] = 0.0

        data.qvel[
            spool_dof
        ] = 0.0

        data.ctrl[
            actuator
        ] = 0.0

        mujoco.mj_step(
            model,
            data,
        )


    errors = []


    # Same sequential closing/opening experiment.
    for (
        encoder,
        hw_mcp,
        hw_pip,
    ) in FREE_REFERENCE:


        spool = max(
            0.0,
            (
                encoder
                - OPEN_ENCODER
            )
            * scale_m_per_tick,
        )


        samples_mcp = []
        samples_pip = []


        steps = int(
            2.0
            / model.opt.timestep
        )


        average_from = int(
            1.8
            / model.opt.timestep
        )


        for step in range(
            steps
        ):

            data.qpos[
                spool_q
            ] = spool

            data.qvel[
                spool_dof
            ] = 0.0

            data.ctrl[
                actuator
            ] = spool


            mujoco.mj_step(
                model,
                data,
            )


            if step >= average_from:

                samples_mcp.append(
                    math.degrees(
                        data.qpos[
                            mcp_q
                        ]
                    )
                )

                samples_pip.append(
                    math.degrees(
                        data.qpos[
                            pip_q
                        ]
                    )
                )


        sim_mcp = float(
            np.mean(
                samples_mcp
            )
        )

        sim_pip = float(
            np.mean(
                samples_pip
            )
        )


        errors.extend(
            [
                sim_mcp - hw_mcp,
                sim_pip - hw_pip,
            ]
        )


    return rms(
        errors
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python3 "
            "fit_index_scale_stiffness_joint_v3.py "
            "/path/to/contact_rosbag"
        )

        sys.exit(1)


    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    print()
    print("=" * 106)

    print(
        "RH8D INDEX — JOINT SPOOL-SCALE + "
        "TENDON-STIFFNESS IDENTIFICATION V3"
    )

    print("=" * 106)


    # =================================================================
    # READ HARDWARE CONTACT DATA
    # =================================================================

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


    hw_hold_mask = (
        (force_times >= hold_start)
        &
        (force_times <= hold_end)
    )


    # =================================================================
    # FTS SAMPLE TIMES
    # =================================================================

    all_fts = (
        corrected.read_fts_times(
            bag
        )
    )


    fts_mask = (
        (all_fts >= approach_start)
        &
        (all_fts <= hold_end)
    )


    fts_times = all_fts[
        fts_mask
    ]


    # =================================================================
    # MOTOR ENCODER
    # =================================================================

    encoder_time = (
        encoder_data[:, 0]
    )


    encoder_position = (
        base.median_filter_5(
            encoder_data[:, 1]
        )
    )


    # =================================================================
    # FREE-SPACE CACHE
    # =================================================================

    free_cache = {}


    print()
    print("=" * 106)
    print("FREE-SPACE PARAMETER MAP")
    print("=" * 106)


    for scale_um in (
        SCALE_UM_PER_TICK
    ):


        for k_tendon in (
            TENDON_STIFFNESS
        ):


            scale = (
                scale_um
                * 1.0e-6
            )


            error = (
                free_space_error(
                    scale,
                    k_tendon,
                )
            )


            free_cache[
                (
                    scale_um,
                    k_tendon,
                )
            ] = error


            print(
                f"scale={scale_um:4.1f} um/tick   "
                f"k={k_tendon:7.1f} N/m   "
                f"free RMSE={error:6.3f} deg"
            )


    # =================================================================
    # CONTACT MODEL
    # =================================================================

    model = mujoco.MjModel.from_xml_path(
        str(CONTACT_MODEL)
    )

    data = mujoco.MjData(
        model
    )


    tendon_id = oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
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
        SPOOL_NAME,
    )


    spool_q = (
        model.jnt_qposadr[
            spool_jid
        ]
    )


    spool_dof = (
        model.jnt_dofadr[
            spool_jid
        ]
    )


    actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAME,
    )


    model.geom_friction[
        fixture_gid,
        0
    ] = CONTACT_FRICTION


    # =================================================================
    # GRID
    # =================================================================

    results = []


    total = (
        len(SCALE_UM_PER_TICK)
        *
        len(TENDON_STIFFNESS)
        *
        len(CONTACT_LAWS)
    )


    count = 0


    print()
    print(
        f"Running {total} combined "
        "free/contact candidates..."
    )


    for scale_um in (
        SCALE_UM_PER_TICK
    ):


        scale = (
            scale_um
            * 1e-6
        )


        # Change the encoder->spool mapping used by
        # base.simulate_candidate().
        base.METERS_PER_TICK = scale


        for k_tendon in (
            TENDON_STIFFNESS
        ):


            model.tendon_stiffness[
                tendon_id
            ] = k_tendon


            free_error = (
                free_cache[
                    (
                        scale_um,
                        k_tendon,
                    )
                ]
            )


            for (
                contact_k,
                contact_d,
            ) in CONTACT_LAWS:


                count += 1


                (
                    sim_times,
                    raw_force,
                    sim_encoder,
                    unexpected,
                ) = base.simulate_candidate(

                    model,
                    data,

                    contact_k,
                    contact_d,

                    approach_start,
                    hold_end,

                    encoder_time,
                    encoder_position,

                    fixture_gid,
                    tip_gid,

                    spool_q,
                    spool_dof,
                    actuator,
                )


                # -----------------------------------------------------
                # Hardware-equivalent FTS pipeline
                # -----------------------------------------------------

                sampled = np.interp(
                    fts_times,
                    sim_times,
                    raw_force,
                )


                filtered = (
                    corrected.causal_median_5(
                        sampled
                    )
                )


                sim_force = np.interp(
                    force_times,
                    fts_times,
                    filtered,
                )


                force_error = (
                    sim_force
                    - hardware_force
                )


                trajectory_rmse = rms(
                    force_error
                )


                sim_hold = (
                    sim_force[
                        hw_hold_mask
                    ]
                )


                hold_median = float(
                    np.median(
                        sim_hold
                    )
                )


                hold_std = float(
                    np.std(
                        sim_hold
                    )
                )


                hold_error = abs(
                    hold_median
                    - HW_HOLD
                )


                # -----------------------------------------------------
                # Threshold onset
                # -----------------------------------------------------

                crossing = np.where(
                    filtered >= 0.10
                )[0]


                if len(
                    crossing
                ):

                    j = crossing[0]

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
                        - HW_ONSET
                    )

                else:

                    threshold_ticks = (
                        math.nan
                    )

                    onset_error = (
                        math.inf
                    )


                # -----------------------------------------------------
                # Raw hold behaviour
                # -----------------------------------------------------

                raw_mask = (
                    (sim_times >= hold_start)
                    &
                    (sim_times <= hold_end)
                )


                raw_hold = (
                    raw_force[
                        raw_mask
                    ]
                )


                duty = float(
                    np.mean(
                        raw_hold > 0.001
                    )
                )


                raw_max = float(
                    np.max(
                        raw_hold
                    )
                )


                # -----------------------------------------------------
                # Physical acceptance
                # -----------------------------------------------------

                acceptable = (

                    free_error <= 6.0

                    and hold_error <= 0.020

                    and onset_error <= 25.0

                    and duty >= 0.90

                    and raw_max <= 0.50

                    and hold_std <= 0.030
                )


                results.append({

                    "scale_um_per_tick":
                        scale_um,

                    "tendon_stiffness_N_per_m":
                        k_tendon,

                    "contact_stiffness":
                        contact_k,

                    "contact_damping":
                        contact_d,

                    "free_rmse_deg":
                        free_error,

                    "contact_trajectory_rmse_N":
                        trajectory_rmse,

                    "hold_median_N":
                        hold_median,

                    "hold_error_N":
                        hold_error,

                    "hold_std_N":
                        hold_std,

                    "contact_duty":
                        duty,

                    "raw_max_N":
                        raw_max,

                    "threshold_ticks":
                        threshold_ticks,

                    "onset_error_ticks":
                        onset_error,

                    "acceptable":
                        acceptable,
                })


                if (
                    count % 15 == 0
                    or count == total
                ):

                    best = min(
                        results,
                        key=lambda r:
                            (
                                r["hold_error_N"],
                                r["free_rmse_deg"],
                            )
                    )


                    print(
                        f"[{count:3d}/{total}] "
                        f"best hold error="
                        f"{best['hold_error_N']:.4f} N   "
                        f"scale="
                        f"{best['scale_um_per_tick']:.1f}   "
                        f"k="
                        f"{best['tendon_stiffness_N_per_m']:.0f}"
                    )


    # =================================================================
    # SAVE
    # =================================================================

    with open(
        OUT,
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


    # =================================================================
    # REPORT
    # =================================================================

    results.sort(
        key=lambda r:
            (
                r["hold_error_N"],
                r["free_rmse_deg"],
                r["onset_error_ticks"],
                r["contact_trajectory_rmse_N"],
            )
    )


    print()
    print("=" * 120)
    print("TOP 25 JOINT IDENTIFICATION CANDIDATES")
    print("=" * 120)


    for rank, r in enumerate(
        results[:25],
        start=1,
    ):

        print()

        print(
            f"#{rank:2d}  "
            f"scale={r['scale_um_per_tick']:.1f} um/tick   "
            f"k_tendon={r['tendon_stiffness_N_per_m']:.0f} N/m   "
            f"contact=({r['contact_stiffness']:.0f},"
            f"{r['contact_damping']:.1f})"
        )


        print(
            f"     free RMSE="
            f"{r['free_rmse_deg']:.3f} deg   "
            f"contact trajectory RMSE="
            f"{r['contact_trajectory_rmse_N']:.4f} N"
        )


        print(
            f"     hold="
            f"{r['hold_median_N']:.4f} N   "
            f"error="
            f"{r['hold_error_N']:.4f} N   "
            f"std="
            f"{r['hold_std_N']:.4f} N"
        )


        print(
            f"     onset="
            f"{r['threshold_ticks']:.1f} ticks   "
            f"duty="
            f"{100*r['contact_duty']:.1f}%   "
            f"raw max="
            f"{r['raw_max_N']:.3f} N"
        )


        print(
            f"     acceptable="
            f"{r['acceptable']}"
        )


    good = [
        r
        for r in results
        if r["acceptable"]
    ]


    print()
    print("=" * 120)
    print("PHYSICALLY ACCEPTABLE JOINT MODELS")
    print("=" * 120)


    if not good:

        print("NONE")

    else:

        for r in sorted(
            good,
            key=lambda r:
                (
                    r[
                        "contact_trajectory_rmse_N"
                    ],
                    r[
                        "free_rmse_deg"
                    ],
                )
        ):

            print(
                f"scale="
                f"{r['scale_um_per_tick']:.1f} um/tick   "
                f"k="
                f"{r['tendon_stiffness_N_per_m']:.0f} N/m   "
                f"free="
                f"{r['free_rmse_deg']:.3f} deg   "
                f"hold="
                f"{r['hold_median_N']:.4f} N   "
                f"onset="
                f"{r['threshold_ticks']:.1f}   "
                f"duty="
                f"{100*r['contact_duty']:.1f}%"
            )


    print()
    print("Saved:")
    print(OUT)

    print("=" * 120)


if __name__ == "__main__":
    main()
