#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

import scan_index_contact_friction_v35g7 as g7
import audit_index_tendon_transmission_v35g12 as g12
import test_direct_spatial_tendon_drive_v35g32a as g32
import test_spatial_prescribed_spool_v35g29b as g29b


# =====================================================================
# G35A
#
# Clean architectural ablation:
#
# G25B original RH8D passive mechanics
#       +
# Aero-style DIRECT position control of idx_flex_spatial
#
# NO G31/G33 added return tendons.
# NO new springs.
# NO parameter scan.
# NO contact change.
# NO PIP/DIP equality.
#
# Question:
#
# Did our added transverse return-spring model cause the post-contact
# PIP blockage seen in G32/G33?
# =====================================================================


MAIN_TENDON = "idx_flex_spatial"
DIRECT_ACTUATOR = "g32_index_tendon_position"

RETURN_SPRINGS = [
    "idx_mcp_return_spring",
    "idx_pip_return_spring",
    "idx_dip_return_spring",
]


SCALE = 5.70e-6


ID_POINTS = [
    # encoder, MCP, PIP, DIP
    (416.1,   10.620, 3.994, 1.911),
    (611.7,   18.094, 4.526, 3.724),
    (673.573, 18.543, 6.161, 5.276),
]


HW_CONTACT_ENCODER = 673.573

HW_CONTACT_POSE = np.array(
    [
        18.543,
        6.161,
        5.276,
    ],
    dtype=float,
)


TRUSTED_FINAL = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


TARGET_TIME = 38.970486


def nearest(values, target):

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


def verify_clean_model(xml):

    model = mujoco.MjModel.from_xml_path(
        str(xml)
    )

    main_tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        MAIN_TENDON,
    )

    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        DIRECT_ACTUATOR,
    )

    if main_tid < 0:

        raise RuntimeError(
            "idx_flex_spatial missing."
        )

    if aid < 0:

        raise RuntimeError(
            "direct spatial-tendon actuator missing."
        )

    for name in RETURN_SPRINGS:

        tid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            name,
        )

        if tid >= 0:

            raise RuntimeError(
                f"STOP: G31/G33 return tendon still exists: {name}"
            )

    print()
    print(
        "G35A ARCHITECTURE CHECK: PASS"
    )

    print(
        "  idx_flex_spatial = present"
    )

    print(
        "  direct tendon position actuator = present"
    )

    print(
        "  G31/G33 added return springs = ABSENT"
    )

    print(
        "  original G25B joint passives = preserved"
    )


def episodes(rows, key):

    flags = np.asarray(
        [
            bool(
                row[
                    key
                ]
            )
            for row in rows
        ],
        dtype=bool,
    )

    return (
        g12.contact_episode_starts(
            flags
        ),
        g12.contact_episode_ends(
            flags
        ),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
        help="G25B baseline spatial-tendon XML",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g35a_direct_tendon_original_passives"
        ),
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
    )

    args = parser.parse_args()

    bag = (
        args.contact_bag
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

    direct_xml = (
        out
        /
        "g35a_direct_tendon_original_passives.xml"
    )

    free_xml = (
        out
        /
        "g35a_direct_tendon_original_passives_free.xml"
    )

    print()
    print("=" * 150)
    print(
        "G35A — DIRECT SPATIAL TENDON + ORIGINAL RH8D PASSIVES"
    )
    print("=" * 150)

    # --------------------------------------------------------------
    # Apply ONLY the G32 direct-actuation transformation to G25B.
    # --------------------------------------------------------------

    g32.build_direct_model(
        source,
        direct_xml,
    )

    verify_clean_model(
        direct_xml
    )

    (
        open_length,
        ratio,
        mapping_rmse,
    ) = (
        g32.fit_encoder_to_tendon_shortening(
            direct_xml
        )
    )

    print()
    print(
        f"identified encoder/tendon ratio = "
        f"{ratio:.6f}"
    )

    print(
        f"mapping RMSE                    = "
        f"{mapping_rmse:.4f} mm"
    )

    g32.disable_fixture(
        direct_xml,
        free_xml,
    )

    # --------------------------------------------------------------
    # Replay.
    # --------------------------------------------------------------

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    g7.b2.v35.METERS_PER_TICK = SCALE

    try:

        # =========================================================
        # FREE / APPROACH
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G35A STEP 1 — H1 APPROACH"
        )
        print("=" * 150)

        (
            (
                free_rows,
                free_qpos,
                free_times,
                free_summary,
            ),
            free_history,
        ) = g32.run_direct(
            free_xml,
            bag,
            replay,
            open_length,
            ratio,
        )

        enc = np.asarray(
            [
                row[
                    "encoder_tick"
                ]
                for row in free_rows
            ],
            dtype=float,
        )

        errors = []

        print()
        print(
            "encoder       HW MCP/PIP/DIP          "
            "SIM MCP/PIP/DIP          error"
        )

        print("-" * 140)

        for (
            tick,
            hm,
            hp,
            hd,
        ) in ID_POINTS:

            i = nearest(
                enc,
                tick,
            )

            row = free_rows[
                i
            ]

            sim = np.array(
                [
                    row[
                        "mcp_deg"
                    ],
                    row[
                        "pip_deg"
                    ],
                    row[
                        "dip_deg"
                    ],
                ],
                dtype=float,
            )

            hw = np.array(
                [
                    hm,
                    hp,
                    hd,
                ],
                dtype=float,
            )

            error = (
                sim
                -
                hw
            )

            errors.extend(
                error.tolist()
            )

            print(
                f"{tick:8.3f}     "
                f"{hm:7.3f}/{hp:7.3f}/{hd:7.3f}     "
                f"{sim[0]:7.3f}/{sim[1]:7.3f}/{sim[2]:7.3f}     "
                f"{error[0]:+7.3f}/"
                f"{error[1]:+7.3f}/"
                f"{error[2]:+7.3f}"
            )

        approach_rmse = float(
            np.sqrt(
                np.mean(
                    np.asarray(
                        errors
                    )
                    **
                    2
                )
            )
        )

        print()
        print(
            f"APPROACH RMSE = "
            f"{approach_rmse:.3f} deg"
        )

        # =========================================================
        # CONTACT
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G35A STEP 2 — ONE H1 CONTACT RUN"
        )
        print("=" * 150)

        (
            (
                rows,
                qpos,
                times,
                summary,
            ),
            contact_history,
        ) = g32.run_direct(
            direct_xml,
            bag,
            replay,
            open_length,
            ratio,
        )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

    mid_starts, mid_ends = episodes(
        rows,
        "mid_contact",
    )

    tip_starts, tip_ends = episodes(
        rows,
        "tip_contact",
    )

    if not mid_starts:

        raise RuntimeError(
            "No middle contact."
        )

    first = int(
        mid_starts[
            0
        ]
    )

    first_end = int(
        mid_ends[
            0
        ]
    )

    first_row = rows[
        first
    ]

    first_pose = np.array(
        [
            first_row[
                "mcp_deg"
            ],
            first_row[
                "pip_deg"
            ],
            first_row[
                "dip_deg"
            ],
        ],
        dtype=float,
    )

    first_error = (
        first_pose
        -
        HW_CONTACT_POSE
    )

    first_rmse = float(
        np.sqrt(
            np.mean(
                first_error
                **
                2
            )
        )
    )

    if len(
        mid_starts
    ) >= 2:

        gap = (
            rows[
                mid_starts[
                    1
                ]
            ][
                "time_s"
            ]
            -
            rows[
                first_end
            ][
                "time_s"
            ]
        )

    else:

        gap = 0.0

    print()
    print(
        "FIRST MIDDLE CONTACT"
    )

    print(
        f"  encoder = "
        f"{first_row['encoder_tick']:.3f}"
    )

    print(
        f"  hardware= "
        f"{HW_CONTACT_ENCODER:.3f}"
    )

    print(
        f"  error   = "
        f"{first_row['encoder_tick'] - HW_CONTACT_ENCODER:+.3f}"
    )

    print()

    print(
        "  simulation = "
        f"{first_pose[0]:.3f} / "
        f"{first_pose[1]:.3f} / "
        f"{first_pose[2]:.3f}"
    )

    print(
        "  hardware   = "
        f"{HW_CONTACT_POSE[0]:.3f} / "
        f"{HW_CONTACT_POSE[1]:.3f} / "
        f"{HW_CONTACT_POSE[2]:.3f}"
    )

    print(
        "  error      = "
        f"{first_error[0]:+.3f} / "
        f"{first_error[1]:+.3f} / "
        f"{first_error[2]:+.3f}"
    )

    print(
        f"  pose RMSE  = "
        f"{first_rmse:.3f} deg"
    )

    print()
    print(
        "CONTACT CONTINUITY"
    )

    print(
        f"  middle episodes = "
        f"{len(mid_starts)}"
    )

    print(
        f"  tip episodes    = "
        f"{len(tip_starts)}"
    )

    print(
        f"  detached gap    = "
        f"{gap:.3f} s"
    )

    # =============================================================
    # Final trusted H1
    # =============================================================

    row_times = np.asarray(
        [
            row[
                "time_s"
            ]
            for row in rows
        ],
        dtype=float,
    )

    ti = nearest(
        row_times,
        TARGET_TIME,
    )

    final = np.array(
        [
            rows[
                ti
            ][
                "mcp_deg"
            ],
            rows[
                ti
            ][
                "pip_deg"
            ],
            rows[
                ti
            ][
                "dip_deg"
            ],
        ],
        dtype=float,
    )

    final_error = (
        final
        -
        TRUSTED_FINAL
    )

    final_rmse = float(
        np.sqrt(
            np.mean(
                final_error
                **
                2
            )
        )
    )

    print()
    print(
        "TRUSTED FINAL H1"
    )

    print(
        "  simulation = "
        f"{final[0]:.3f} / "
        f"{final[1]:.3f} / "
        f"{final[2]:.3f}"
    )

    print(
        "  hardware   = "
        f"{TRUSTED_FINAL[0]:.3f} / "
        f"{TRUSTED_FINAL[1]:.3f} / "
        f"{TRUSTED_FINAL[2]:.3f}"
    )

    print(
        "  error      = "
        f"{final_error[0]:+.3f} / "
        f"{final_error[1]:+.3f} / "
        f"{final_error[2]:+.3f}"
    )

    print(
        f"  RMSE       = "
        f"{final_rmse:.3f} deg"
    )

    # --------------------------------------------------------------
    # Direct tendon tracking.
    # --------------------------------------------------------------

    track_error = np.asarray(
        [
            abs(
                row[
                    "tracking_error_mm"
                ]
            )
            for row in contact_history
        ],
        dtype=float,
    )

    print()
    print(
        "DIRECT TENDON TRACKING"
    )

    print(
        f"  mean error = "
        f"{np.mean(track_error):.4f} mm"
    )

    print(
        f"  max error  = "
        f"{np.max(track_error):.4f} mm"
    )

    # =============================================================
    # Decision
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G35A DECISION"
    )
    print("=" * 150)

    print(
        f"approach RMSE       = "
        f"{approach_rmse:.3f} deg"
    )

    print(
        f"first-contact RMSE  = "
        f"{first_rmse:.3f} deg"
    )

    print(
        f"middle episodes     = "
        f"{len(mid_starts)}"
    )

    print(
        f"middle gap          = "
        f"{gap:.3f} s"
    )

    print(
        f"final PIP           = "
        f"{final[1]:.3f} deg "
        f"(HW {TRUSTED_FINAL[1]:.3f})"
    )

    print(
        f"final RMSE          = "
        f"{final_rmse:.3f} deg"
    )

    print()

    if (
        len(mid_starts) == 1
        and
        final[
            1
        ] >= 15.0
    ):

        print(
            "CRITICAL RESULT:"
        )

        print(
            "Removing G31/G33 added return springs restores "
            "post-contact PIP recruitment."
        )

        print(
            "Discard G31/G33 spring identification."
        )

        print(
            "Keep direct spatial-tendon actuation."
        )

        print(
            "Next: identify the REAL transverse passive mechanism "
            "from free-space staging, not H1 static equilibrium."
        )

    elif (
        len(mid_starts) == 1
        and
        final[
            1
        ] < 15.0
    ):

        print(
            "Direct actuation still retains contact, but original "
            "passives also suppress PIP recruitment."
        )

        print(
            "Next: isolate original joint passive terms one-by-one."
        )

    else:

        print(
            "Removing added springs brought back contact separation."
        )

        print(
            "Therefore G31/G33 springs contributed to retention, "
            "but were over-resistant."
        )

        print(
            "Next: identify transverse springs from FREE-SPACE "
            "sequential recruitment instead of H1 equilibrium."
        )

    # =============================================================
    # Save
    # =============================================================

    summary_csv = (
        out
        /
        "g35a_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shortening_ratio",
                "approach_rmse_deg",
                "first_contact_encoder",
                "first_contact_pose_rmse_deg",
                "middle_episodes",
                "tip_episodes",
                "middle_gap_s",
                "final_mcp_deg",
                "final_pip_deg",
                "final_dip_deg",
                "final_rmse_deg",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "shortening_ratio":
                ratio,

            "approach_rmse_deg":
                approach_rmse,

            "first_contact_encoder":
                first_row[
                    "encoder_tick"
                ],

            "first_contact_pose_rmse_deg":
                first_rmse,

            "middle_episodes":
                len(mid_starts),

            "tip_episodes":
                len(tip_starts),

            "middle_gap_s":
                gap,

            "final_mcp_deg":
                final[0],

            "final_pip_deg":
                final[1],

            "final_dip_deg":
                final[2],

            "final_rmse_deg":
                final_rmse,
        })

    np.savez_compressed(
        out
        /
        "g35a_contact_trajectory.npz",
        times=np.asarray(times),
        qpos=np.asarray(qpos),
    )

    print()
    print(
        f"Candidate XML:\n  {direct_xml}"
    )

    print()
    print(
        f"Summary:\n  {summary_csv}"
    )

    print()
    print(
        f"All outputs:\n  {out}"
    )

    if not args.no_gui:

        view_xml = (
            out
            /
            "g35a_view.xml"
        )

        g29b.make_view_xml(
            direct_xml,
            view_xml,
        )

        g7.exact_gui_replay(
            view_xml,
            qpos,
            times,
            "G35A direct tendon + original RH8D passives",
        )


if __name__ == "__main__":
    main()
