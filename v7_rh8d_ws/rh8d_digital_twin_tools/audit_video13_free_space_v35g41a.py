#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import mujoco
import numpy as np

import scan_index_contact_friction_v35g7 as g7
import audit_index_tendon_transmission_v35g12 as g12
import test_direct_spatial_tendon_drive_v35g32a as g32

from rh8d_tabletop_stage_v1 import (
    apply_tabletop_stage,
)


# =====================================================================
# G41A
#
# FULL FREE-SPACE HARDWARE BASELINE
#
# Uses:
#   - G39 spatial tendon geometry
#   - direct spatial-tendon position drive
#   - ORIGINAL joint passives
#   - REAL free-space encoder ROS bag
#   - VIDEO-13 measured joint poses
#
# Identifies ONLY:
#   encoder -> tendon-shortening scale
#
# from free-space data.
#
# It does NOT tune:
#   joint stiffness
#   damping
#   friction
#   springs
#   contact
# =====================================================================


MAIN_TENDON = "idx_flex_spatial"

METERS_PER_TICK = 5.70e-6

ENCODER_ZERO = 100.0


# tick, MCP, PIP, DIP
#
# Negative near-zero marker values are measurement noise.
#
VIDEO13 = [
    (100.0,   0.000,  0.000,  0.000),
    (500.0,  16.952,  0.968, -0.428),
    (1000.0, 39.472, -0.907, -0.343),
    (1500.0, 57.291,  3.654, -4.019),
    (2000.0, 73.632,  4.667,  2.012),
    (2500.0, 73.632, 27.119,  3.190),
    (3000.0, 77.347, 48.581,  7.810),
    (3500.0, 80.074, 70.872, -7.199),
]


# Use these points for quantitative trajectory error.
#
# 3500 DIP is excluded because its marker orientation is unreliable.
RMSE_TICKS = {
    500.0,
    1000.0,
    1500.0,
    2000.0,
    2500.0,
    3000.0,
}


def refs(model):

    return g12.get_joint_refs(
        model
    )


def nearest_index(values, target):

    values = np.asarray(
        values,
        dtype=float,
    )

    return int(
        np.argmin(
            np.abs(
                values
                -
                target
            )
        )
    )


def set_hardware_pose(
    model,
    data,
    joint_refs,
    mcp,
    pip,
    dip,
):

    apply_tabletop_stage(
        model,
        data,
        reset_to_qpos0=True,
    )

    # Negative marker angles near zero are image-plane noise.
    pose = {
        "mcp":
            max(
                0.0,
                float(mcp),
            ),

        "pip":
            max(
                0.0,
                float(pip),
            ),

        "dip":
            max(
                0.0,
                float(dip),
            ),
    }

    for key, value in pose.items():

        data.qpos[
            joint_refs[
                key
            ][
                "qadr"
            ]
        ] = math.radians(
            value
        )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    return pose


def tendon_length_at_pose(
    model,
    data,
    joint_refs,
    mcp,
    pip,
    dip,
):

    pose = set_hardware_pose(
        model,
        data,
        joint_refs,
        mcp,
        pip,
        dip,
    )

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        MAIN_TENDON,
    )

    if tid < 0:

        raise RuntimeError(
            "idx_flex_spatial missing."
        )

    return (
        float(
            data.ten_length[
                tid
            ]
        ),
        pose,
    )


def fit_free_space_ratio(
    direct_xml,
):

    model = mujoco.MjModel.from_xml_path(
        str(
            direct_xml
        )
    )

    data = mujoco.MjData(
        model
    )

    joint_refs = refs(
        model
    )

    # -------------------------------------------------------------
    # Reference/open length at 100 ticks.
    # -------------------------------------------------------------

    open_length, _ = tendon_length_at_pose(
        model,
        data,
        joint_refs,
        0.0,
        0.0,
        0.0,
    )

    x = []
    y = []

    rows = []

    print()
    print("=" * 150)
    print(
        "G41A STEP 1 — VIDEO-13 FREE-SPACE "
        "ENCODER -> TENDON SHORTENING"
    )
    print("=" * 150)

    print()
    print(
        "tick      MCP      PIP      DIP      "
        "spool take-up    tendon shortening    ratio"
    )

    print("-" * 130)

    for (
        tick,
        mcp,
        pip,
        dip,
    ) in VIDEO13:

        length, pose = tendon_length_at_pose(
            model,
            data,
            joint_refs,
            mcp,
            pip,
            dip,
        )

        spool = (
            (
                tick
                -
                ENCODER_ZERO
            )
            *
            METERS_PER_TICK
        )

        shortening = (
            open_length
            -
            length
        )

        if spool > 1e-9:

            local_ratio = (
                shortening
                /
                spool
            )

        else:

            local_ratio = np.nan

        print(
            f"{tick:6.0f}   "
            f"{pose['mcp']:7.3f} "
            f"{pose['pip']:7.3f} "
            f"{pose['dip']:7.3f}      "
            f"{1000*spool:8.4f} mm       "
            f"{1000*shortening:8.4f} mm       "
            f"{local_ratio:7.4f}"
        )

        rows.append({
            "tick":
                tick,

            "mcp_deg":
                pose[
                    "mcp"
                ],

            "pip_deg":
                pose[
                    "pip"
                ],

            "dip_deg":
                pose[
                    "dip"
                ],

            "spool_takeup_mm":
                1000.0
                *
                spool,

            "tendon_shortening_mm":
                1000.0
                *
                shortening,

            "local_ratio":
                local_ratio,
        })

        # ---------------------------------------------------------
        # Fit on reliable 500...3000 points.
        #
        # 3500 DIP marker measurement is not trusted.
        # ---------------------------------------------------------

        if tick in RMSE_TICKS:

            x.append(
                spool
            )

            y.append(
                shortening
            )

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    # Through-origin least squares:
    #
    # tendon shortening = ratio * spool take-up
    ratio = float(
        np.dot(
            x,
            y,
        )
        /
        np.dot(
            x,
            x,
        )
    )

    residual = (
        y
        -
        ratio
        *
        x
    )

    mapping_rmse_mm = float(
        1000.0
        *
        np.sqrt(
            np.mean(
                residual
                **
                2
            )
        )
    )

    print()
    print(
        f"FREE-SPACE identified shortening ratio = "
        f"{ratio:.6f}"
    )

    print(
        f"FREE-SPACE mapping residual RMSE       = "
        f"{mapping_rmse_mm:.4f} mm"
    )

    print(
        f"direct-model open tendon length        = "
        f"{1000*open_length:.4f} mm"
    )

    return (
        open_length,
        ratio,
        mapping_rmse_mm,
        rows,
    )


def closing_branch(
    rows,
):

    encoder = np.asarray(
        [
            float(
                row[
                    "encoder_tick"
                ]
            )
            for row in rows
        ],
        dtype=float,
    )

    # Full free-space experiment closes to maximum encoder,
    # then opens again.
    peak = int(
        np.argmax(
            encoder
        )
    )

    return (
        rows[
            :
            peak
            +
            1
        ],
        peak,
    )


def evaluate_trajectory(
    rows,
):

    close_rows, peak = closing_branch(
        rows
    )

    encoder = np.asarray(
        [
            float(
                row[
                    "encoder_tick"
                ]
            )
            for row in close_rows
        ],
        dtype=float,
    )

    all_errors = []

    per_joint_errors = {
        "mcp": [],
        "pip": [],
        "dip": [],
    }

    table = []

    print()
    print("=" * 150)
    print(
        "G41A STEP 3 — HARDWARE vs SIMULATION "
        "ON REAL FREE-SPACE CLOSING SWEEP"
    )
    print("=" * 150)

    print()
    print(
        "tick       HW MCP/PIP/DIP              "
        "SIM MCP/PIP/DIP             error"
    )

    print("-" * 150)

    for (
        tick,
        hw_mcp,
        hw_pip,
        hw_dip,
    ) in VIDEO13:

        i = nearest_index(
            encoder,
            tick,
        )

        row = close_rows[
            i
        ]

        sim = np.array(
            [
                float(
                    row[
                        "mcp_deg"
                    ]
                ),

                float(
                    row[
                        "pip_deg"
                    ]
                ),

                float(
                    row[
                        "dip_deg"
                    ]
                ),
            ],
            dtype=float,
        )

        # Negative near-zero marker observations are treated
        # as zero for mechanical comparison.
        hw = np.array(
            [
                max(
                    0.0,
                    hw_mcp,
                ),

                max(
                    0.0,
                    hw_pip,
                ),

                max(
                    0.0,
                    hw_dip,
                ),
            ],
            dtype=float,
        )

        error = (
            sim
            -
            hw
        )

        print(
            f"{tick:6.0f}     "
            f"{hw[0]:7.3f}/{hw[1]:7.3f}/{hw[2]:7.3f}     "
            f"{sim[0]:7.3f}/{sim[1]:7.3f}/{sim[2]:7.3f}     "
            f"{error[0]:+7.3f}/"
            f"{error[1]:+7.3f}/"
            f"{error[2]:+7.3f}"
        )

        table.append({
            "tick":
                tick,

            "hw_mcp_deg":
                hw[
                    0
                ],

            "hw_pip_deg":
                hw[
                    1
                ],

            "hw_dip_deg":
                hw[
                    2
                ],

            "sim_mcp_deg":
                sim[
                    0
                ],

            "sim_pip_deg":
                sim[
                    1
                ],

            "sim_dip_deg":
                sim[
                    2
                ],

            "err_mcp_deg":
                error[
                    0
                ],

            "err_pip_deg":
                error[
                    1
                ],

            "err_dip_deg":
                error[
                    2
                ],
        })

        if tick in RMSE_TICKS:

            all_errors.extend(
                error.tolist()
            )

            per_joint_errors[
                "mcp"
            ].append(
                error[
                    0
                ]
            )

            per_joint_errors[
                "pip"
            ].append(
                error[
                    1
                ]
            )

            per_joint_errors[
                "dip"
            ].append(
                error[
                    2
                ]
            )

    rmse = float(
        np.sqrt(
            np.mean(
                np.asarray(
                    all_errors,
                    dtype=float,
                )
                **
                2
            )
        )
    )

    joint_rmse = {}

    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        e = np.asarray(
            per_joint_errors[
                key
            ],
            dtype=float,
        )

        joint_rmse[
            key
        ] = float(
            np.sqrt(
                np.mean(
                    e
                    **
                    2
                )
            )
        )

    print()
    print(
        "FREE-SPACE RMSE — reliable 500...3000 points"
    )

    print(
        f"  MCP = "
        f"{joint_rmse['mcp']:.3f} deg"
    )

    print(
        f"  PIP = "
        f"{joint_rmse['pip']:.3f} deg"
    )

    print(
        f"  DIP = "
        f"{joint_rmse['dip']:.3f} deg"
    )

    print(
        f"  ALL = "
        f"{rmse:.3f} deg"
    )

    return (
        table,
        joint_rmse,
        rmse,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "free_space_bag",
        type=Path,
    )

    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
        help="G39A spatial-tendon model",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g41a_video13_free_space"
        ),
    )

    args = parser.parse_args()

    bag = (
        args.free_space_bag
        .expanduser()
        .resolve()
    )

    source = (
        args.xml
        .expanduser()
        .resolve()
    )

    out = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 150)
    print(
        "G41A — FULL VIDEO-13 FREE-SPACE BASELINE"
    )
    print("=" * 150)

    print()
    print(
        f"free-space bag:\n  {bag}"
    )

    print(
        f"G39A source:\n  {source}"
    )

    # =============================================================
    # Build direct-spatial-tendon architecture from G39A.
    # =============================================================

    direct_xml = (
        out
        /
        "g41a_g39_direct_free_space.xml"
    )

    g32.build_direct_model(
        source,
        direct_xml,
    )

    # Remove bottle for true free-space replay.
    free_xml = (
        out
        /
        "g41a_g39_direct_free_space_no_fixture.xml"
    )

    g32.disable_fixture(
        direct_xml,
        free_xml,
    )

    # =============================================================
    # Fit ONLY encoder -> tendon map from VIDEO-13.
    # =============================================================

    (
        open_length,
        ratio,
        mapping_rmse_mm,
        mapping_rows,
    ) = fit_free_space_ratio(
        free_xml
    )

    # =============================================================
    # Replay REAL free-space bag.
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G41A STEP 2 — REPLAY REAL FREE-SPACE ENCODER BAG"
    )
    print("=" * 150)

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    g7.b2.v35.METERS_PER_TICK = (
        METERS_PER_TICK
    )

    try:

        (
            (
                rows,
                qpos,
                times,
                summary,
            ),
            drive_history,
        ) = g32.run_direct(
            free_xml,
            bag,
            replay,
            open_length,
            ratio,
        )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

    if not rows:

        raise RuntimeError(
            "Free-space replay returned no samples."
        )

    encoder_values = np.asarray(
        [
            float(
                row[
                    "encoder_tick"
                ]
            )
            for row in rows
        ],
        dtype=float,
    )

    print()
    print(
        f"replay samples   = "
        f"{len(rows)}"
    )

    print(
        f"encoder min/max  = "
        f"{np.min(encoder_values):.1f} / "
        f"{np.max(encoder_values):.1f} ticks"
    )

    # Hard gate: this must be the full free-space bag.
    if np.max(
        encoder_values
    ) < 3000.0:

        raise RuntimeError(
            "STOP: this does not look like the 100->3500 "
            "free-space bag. Encoder never reached 3000 ticks."
        )

    (
        comparison_rows,
        joint_rmse,
        total_rmse,
    ) = evaluate_trajectory(
        rows
    )

    # =============================================================
    # Direct-tendon tracking.
    # =============================================================

    tracking = np.asarray(
        [
            abs(
                float(
                    row[
                        "tracking_error_mm"
                    ]
                )
            )
            for row in drive_history
        ],
        dtype=float,
    )

    print()
    print(
        "DIRECT TENDON TRACKING"
    )

    print(
        f"  mean = "
        f"{np.mean(tracking):.4f} mm"
    )

    print(
        f"  max  = "
        f"{np.max(tracking):.4f} mm"
    )

    # =============================================================
    # Save.
    # =============================================================

    mapping_csv = (
        out
        /
        "g41a_free_space_input_mapping.csv"
    )

    with mapping_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                mapping_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            mapping_rows
        )

    comparison_csv = (
        out
        /
        "g41a_video13_sim_comparison.csv"
    )

    with comparison_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                comparison_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            comparison_rows
        )

    np.savez_compressed(
        out
        /
        "g41a_free_space_dynamic_trajectory.npz",
        times=np.asarray(
            times
        ),
        qpos=np.asarray(
            qpos
        ),
        encoder=encoder_values,
    )

    print()
    print("=" * 150)
    print(
        "G41A DECISION"
    )
    print("=" * 150)

    print()
    print(
        f"free-space mapping ratio = "
        f"{ratio:.6f}"
    )

    print(
        f"mapping RMSE             = "
        f"{mapping_rmse_mm:.4f} mm"
    )

    print()

    print(
        f"MCP RMSE = "
        f"{joint_rmse['mcp']:.3f} deg"
    )

    print(
        f"PIP RMSE = "
        f"{joint_rmse['pip']:.3f} deg"
    )

    print(
        f"DIP RMSE = "
        f"{joint_rmse['dip']:.3f} deg"
    )

    print(
        f"ALL RMSE = "
        f"{total_rmse:.3f} deg"
    )

    print()
    print(
        "NO passive parameters were changed."
    )

    print(
        "This result becomes the baseline for the next "
        "free-space-only passive identification."
    )

    print()
    print(
        f"Outputs:\n  {out}"
    )


if __name__ == "__main__":
    main()
