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
# PURPOSE
#
# Contact V1.2 demonstrated:
#
#   contact geometry        approximately correct
#   contact onset           approximately correct
#   contact law alone       insufficient
#
# We now identify tendon stiffness UNDER LOAD while requiring
# the same model to remain compatible with the original
# free-space MCP/PIP experiment.
#
# No staged joint mapping is used.
# ============================================================


FREE_MODEL_PATH = (
    base.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_identified.xml"
)

CONTACT_MODEL_PATH = (
    base.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_v1.xml"
)

RESULT_DIR = (
    base.WS
    / "rh8d_digital_twin_tools/results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    RESULT_DIR
    / "index_loaded_tendon_v2_grid.csv"
)


TENDON_NAME = "idx_flex_spatial"

SPOOL_NAME = "index_motor_spool"

ACTUATOR_NAME = "pos_index_tendon"

MCP_NAME = (
    "Index_Proximal--palm_assembly:1"
)

PIP_NAME = (
    "Index_Middle--Index_Proximal"
)


OPEN_ENCODER = 100.0

METERS_PER_TICK = 6.0e-6


# ============================================================
# ORIGINAL FREE-SPACE PHYSICAL VIDEO DATA
#
# measured encoder value, MCP deg, PIP deg
#
# These are VIDEO-derived physical angles.
# They are NOT ROS /joint_states.
# ============================================================

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


# ============================================================
# SEARCH SPACE
#
# Current model = 1200 N/m.
#
# Contact result suggests roughly more load-carrying stiffness
# may be required, but we test it instead of assuming it.
# ============================================================

TENDON_STIFFNESS_VALUES = [

    1200.0,

    1600.0,

    2000.0,

    2400.0,

    2800.0,

    3200.0,

    4000.0,

    4800.0,
]


# Contact stiffness no longer needs an enormous broad search.
#
# V1.2 showed the sustained-force result was almost insensitive
# to raising this from tens of thousands upward.

CONTACT_STIFFNESS_VALUES = [

    32000.0,

    48000.0,

    64000.0,
]


# Search higher damping because V1.2 still showed raw
# contact/liftoff chatter.

CONTACT_DAMPING_VALUES = [

    160.0,

    320.0,

    640.0,

    1280.0,
]


# ============================================================
# ACCEPTANCE TARGETS
# ============================================================

HW_HOLD_FORCE = 0.14563

HW_ONSET_TICKS = 1020.0


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
            f"Could not find: {name}"
        )

    return idx


def rmse(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    return math.sqrt(
        np.mean(
            values * values
        )
    )


# ============================================================
# FREE-SPACE VALIDATION
# ============================================================

def free_space_rmse(
    tendon_stiffness,
):

    model = (
        mujoco.MjModel.from_xml_path(
            str(
                FREE_MODEL_PATH
            )
        )
    )

    data = mujoco.MjData(
        model
    )

    model.opt.gravity[:] = 0.0


    tendon_id = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )


    model.tendon_stiffness[
        tendon_id
    ] = tendon_stiffness


    spool_jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_NAME,
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
        ACTUATOR_NAME,
    )


    mcp_jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        MCP_NAME,
    )

    pip_jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        PIP_NAME,
    )


    mcp_qadr = (
        model.jnt_qposadr[
            mcp_jid
        ]
    )

    pip_qadr = (
        model.jnt_qposadr[
            pip_jid
        ]
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    # --------------------------------------------------------
    # Settle fully open.
    # --------------------------------------------------------

    for _ in range(
        int(
            1.5
            / model.opt.timestep
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


    hold_steps = int(
        2.0
        / model.opt.timestep
    )

    average_steps = int(
        0.20
        / model.opt.timestep
    )


    errors = []


    for (
        encoder,
        hardware_mcp,
        hardware_pip,
    ) in FREE_REFERENCE:


        spool = max(
            0.0,
            (
                encoder
                - OPEN_ENCODER
            )
            * METERS_PER_TICK,
        )


        samples = []


        for step in range(
            hold_steps
        ):

            data.qpos[
                spool_qadr
            ] = spool

            data.qvel[
                spool_dof
            ] = 0.0

            data.ctrl[
                actuator_id
            ] = spool


            mujoco.mj_step(
                model,
                data,
            )


            if (
                step
                >= hold_steps
                - average_steps
            ):

                samples.append(
                    (
                        data.qpos[
                            mcp_qadr
                        ],
                        data.qpos[
                            pip_qadr
                        ],
                    )
                )


        mean_q = np.mean(
            np.asarray(
                samples
            ),
            axis=0,
        )


        sim_mcp = math.degrees(
            mean_q[0]
        )

        sim_pip = math.degrees(
            mean_q[1]
        )


        errors.extend(
            [
                sim_mcp
                - hardware_mcp,

                sim_pip
                - hardware_pip,
            ]
        )


    return rmse(
        errors
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(
        sys.argv
    ) != 2:

        print()

        print(
            "Usage:"
        )

        print(
            "  python3 "
            "fit_index_loaded_tendon_v2.py "
            "/path/to/contact_rosbag_directory"
        )

        sys.exit(1)


    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    print()
    print("=" * 88)

    print(
        "RH8D INDEX — "
        "LOADED TENDON IDENTIFICATION V2"
    )

    print("=" * 88)


    # ========================================================
    # READ CONTACT HARDWARE DATA
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


    print()
    print(
        "Clean contact interval:"
    )

    print(
        f"  approach start   = "
        f"{approach_start:.6f}"
    )

    print(
        f"  regulation start = "
        f"{regulation_start:.6f}"
    )

    print(
        f"  hold start       = "
        f"{hold_start:.6f}"
    )

    print(
        f"  hold end         = "
        f"{hold_end:.6f}"
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


    # ========================================================
    # REAL FTS SAMPLE TIMES
    # ========================================================

    all_fts_times = (
        corrected.read_fts_times(
            bag
        )
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


    # ========================================================
    # REAL MOTOR ENCODER
    # ========================================================

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


    # ========================================================
    # FREE-SPACE ERROR FOR EACH TENDON STIFFNESS
    # ========================================================

    print()
    print("=" * 88)

    print(
        "FREE-SPACE REVALIDATION"
    )

    print("=" * 88)


    free_cache = {}


    for k_tendon in (
        TENDON_STIFFNESS_VALUES
    ):

        error = free_space_rmse(
            k_tendon
        )

        free_cache[
            k_tendon
        ] = error


        print(
            f"tendon stiffness "
            f"{k_tendon:7.1f} N/m"
            f"  ->  "
            f"MCP/PIP RMSE "
            f"{error:.3f} deg"
        )


    # ========================================================
    # CONTACT MODEL
    # ========================================================

    model = (
        mujoco.MjModel.from_xml_path(
            str(
                CONTACT_MODEL_PATH
            )
        )
    )

    data = mujoco.MjData(
        model
    )


    tendon_id = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )


    fixture_gid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        base.FIXTURE_GEOM,
    )

    tip_gid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        base.TIP_GEOM,
    )


    spool_jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_NAME,
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
        ACTUATOR_NAME,
    )


    total = (
        len(
            TENDON_STIFFNESS_VALUES
        )
        *
        len(
            CONTACT_STIFFNESS_VALUES
        )
        *
        len(
            CONTACT_DAMPING_VALUES
        )
    )


    print()
    print(
        f"Testing {total} "
        "loaded-contact models..."
    )


    results = []

    count = 0


    # ========================================================
    # GRID SEARCH
    # ========================================================

    for k_tendon in (
        TENDON_STIFFNESS_VALUES
    ):

        model.tendon_stiffness[
            tendon_id
        ] = k_tendon


        for k_contact in (
            CONTACT_STIFFNESS_VALUES
        ):


            for d_contact in (
                CONTACT_DAMPING_VALUES
            ):


                count += 1


                (
                    sim_times,
                    raw_force,
                    sim_encoder,
                    unexpected,
                ) = base.simulate_candidate(

                    model,
                    data,

                    k_contact,
                    d_contact,

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


                # ============================================
                # HARDWARE-EQUIVALENT FTS PIPELINE
                # ============================================

                sim_at_fts_raw = np.interp(
                    fts_times,
                    sim_times,
                    raw_force,
                )


                sim_at_fts_filtered = (
                    corrected.causal_median_5(
                        sim_at_fts_raw
                    )
                )


                sim_force = np.interp(
                    force_times,
                    fts_times,
                    sim_at_fts_filtered,
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

                hold_std = float(
                    np.std(
                        sim_hold
                    )
                )


                # ============================================
                # 0.10-N SENSOR THRESHOLD
                # ============================================

                crossing = np.where(
                    sim_at_fts_filtered
                    >= 0.10
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

                else:

                    threshold_ticks = (
                        math.nan
                    )


                # ============================================
                # RAW CONTACT DUTY
                # ============================================

                raw_hold_mask = (
                    (sim_times
                     >= hold_start)
                    &
                    (sim_times
                     <= hold_end)
                )


                raw_hold = (
                    raw_force[
                        raw_hold_mask
                    ]
                )


                duty = float(
                    np.mean(
                        raw_hold
                        > 0.001
                    )
                )


                contact_rmse = rmse(
                    errors
                )


                hold_error = abs(
                    hold_median
                    - HW_HOLD_FORCE
                )


                onset_error = (
                    abs(
                        threshold_ticks
                        - HW_ONSET_TICKS
                    )
                    if math.isfinite(
                        threshold_ticks
                    )
                    else math.inf
                )


                free_error = (
                    free_cache[
                        k_tendon
                    ]
                )


                # ============================================
                # TWO LEVELS OF ACCEPTANCE
                #
                # CALIBRATION:
                # good force + free-space consistency
                #
                # STABLE:
                # additionally near-continuous raw contact
                # ============================================

                calibration_candidate = (

                    hold_error
                    <= 0.025

                    and onset_error
                    <= 30.0

                    and free_error
                    <= 6.5

                    and hold_std
                    <= 0.030
                )


                stable_candidate = (

                    calibration_candidate

                    and duty
                    >= 0.90
                )


                results.append({

                    "tendon_stiffness_N_per_m":
                        k_tendon,

                    "contact_stiffness":
                        k_contact,

                    "contact_damping":
                        d_contact,

                    "free_space_rmse_deg":
                        free_error,

                    "contact_force_rmse_N":
                        contact_rmse,

                    "sim_hold_median_N":
                        hold_median,

                    "hold_error_N":
                        hold_error,

                    "sim_hold_std_N":
                        hold_std,

                    "raw_contact_duty":
                        duty,

                    "sim_threshold_ticks":
                        threshold_ticks,

                    "onset_error_ticks":
                        onset_error,

                    "raw_max_N":
                        float(
                            np.max(
                                raw_hold
                            )
                        ),

                    "calibration_candidate":
                        calibration_candidate,

                    "stable_candidate":
                        stable_candidate,
                })


                if (
                    count % 12 == 0
                    or count == total
                ):

                    best = min(
                        results,
                        key=lambda r:
                            (
                                r[
                                    "hold_error_N"
                                ],
                                r[
                                    "contact_force_rmse_N"
                                ],
                            ),
                    )


                    print(
                        f"[{count:3d}/{total}] "
                        f"best hold error="
                        f"{best['hold_error_N']:.4f} N   "
                        f"k_tendon="
                        f"{best['tendon_stiffness_N_per_m']:.0f}"
                    )


    # ========================================================
    # SAVE
    # ========================================================

    df_fields = list(
        results[0].keys()
    )


    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=df_fields,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    results.sort(
        key=lambda r:
            (
                r[
                    "hold_error_N"
                ],
                r[
                    "free_space_rmse_deg"
                ],
                r[
                    "contact_force_rmse_N"
                ],
            )
    )


    print()
    print("=" * 120)

    print(
        "TOP 20 LOADED-TENDON CANDIDATES"
    )

    print("=" * 120)


    for rank, r in enumerate(
        results[:20],
        start=1,
    ):

        print()

        print(
            f"#{rank:2d}  "
            f"k_tendon="
            f"{r['tendon_stiffness_N_per_m']:7.1f} N/m   "
            f"k_contact="
            f"{r['contact_stiffness']:8.0f}   "
            f"d_contact="
            f"{r['contact_damping']:7.0f}"
        )


        print(
            f"     FREE RMSE="
            f"{r['free_space_rmse_deg']:6.3f} deg   "
            f"CONTACT RMSE="
            f"{r['contact_force_rmse_N']:.5f} N"
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
            f"     onset="
            f"{r['sim_threshold_ticks']:.1f} ticks   "
            f"duty="
            f"{100*r['raw_contact_duty']:.1f}%   "
            f"raw max="
            f"{r['raw_max_N']:.3f} N"
        )


        print(
            f"     calibration="
            f"{r['calibration_candidate']}   "
            f"stable="
            f"{r['stable_candidate']}"
        )


    calibration = [
        r
        for r in results
        if r[
            "calibration_candidate"
        ]
    ]


    stable = [
        r
        for r in results
        if r[
            "stable_candidate"
        ]
    ]


    print()
    print("=" * 120)

    print(
        "CALIBRATION CANDIDATES"
    )

    print("=" * 120)


    if not calibration:

        print(
            "NONE"
        )

    else:

        for r in calibration:

            print(
                f"k_tendon="
                f"{r['tendon_stiffness_N_per_m']:.0f}   "
                f"k_contact="
                f"{r['contact_stiffness']:.0f}   "
                f"d_contact="
                f"{r['contact_damping']:.0f}   "
                f"free="
                f"{r['free_space_rmse_deg']:.3f} deg   "
                f"hold="
                f"{r['sim_hold_median_N']:.4f} N   "
                f"duty="
                f"{100*r['raw_contact_duty']:.1f}%"
            )


    print()
    print("=" * 120)

    print(
        "STABLE CANDIDATES"
    )

    print("=" * 120)


    if not stable:

        print(
            "NONE"
        )

    else:

        for r in stable:

            print(
                f"k_tendon="
                f"{r['tendon_stiffness_N_per_m']:.0f}   "
                f"k_contact="
                f"{r['contact_stiffness']:.0f}   "
                f"d_contact="
                f"{r['contact_damping']:.0f}   "
                f"free="
                f"{r['free_space_rmse_deg']:.3f} deg   "
                f"hold="
                f"{r['sim_hold_median_N']:.4f} N   "
                f"duty="
                f"{100*r['raw_contact_duty']:.1f}%"
            )


    print()
    print("=" * 120)

    print(
        "Saved:"
    )

    print(
        OUTPUT_CSV
    )

    print("=" * 120)


if __name__ == "__main__":
    main()
