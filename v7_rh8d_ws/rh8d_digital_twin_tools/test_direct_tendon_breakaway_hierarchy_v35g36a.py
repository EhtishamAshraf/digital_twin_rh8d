#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import scan_index_contact_friction_v35g7 as g7
import audit_index_tendon_transmission_v35g12 as g12
import test_direct_spatial_tendon_drive_v35g32a as g32
import test_spatial_prescribed_spool_v35g29b as g29b


# ================================================================
# G36A
#
# Combine two ALREADY independently supported mechanisms:
#
# A) G20C:
#       dry breakaway hierarchy produces
#       MCP -> PIP -> DIP recruitment.
#
# B) G32/G35:
#       direct spatial-tendon position drive eliminates
#       middle-contact rebound.
#
# NO parameter search.
# ================================================================


JOINTS = {
    "mcp":
        "Index_Proximal--palm_assembly:1",

    "pip":
        "Index_Middle--Index_Proximal",

    "dip":
        "Index_Distal--Index_Middle",
}


FRICTION = {
    "mcp": 0.00035,
    "pip": 0.00300,
    "dip": 0.00600,
}


RETURN_SPRINGS = [
    "idx_mcp_return_spring",
    "idx_pip_return_spring",
    "idx_dip_return_spring",
]


SCALE = 5.70e-6


H1_POINTS = [
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


def find_joint(root, name):

    for joint in root.iter(
        "joint"
    ):

        if joint.get(
            "name"
        ) == name:

            return joint

    raise RuntimeError(
        f"Joint not found: {name}"
    )


def apply_breakaway_hierarchy(
    source,
    destination,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    for key, name in JOINTS.items():

        joint = find_joint(
            root,
            name,
        )

        joint.set(
            "frictionloss",
            f"{FRICTION[key]:.12g}",
        )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )

    model = mujoco.MjModel.from_xml_path(
        str(
            destination
        )
    )

    # ------------------------------------------------------------
    # Absolute topology gates.
    # ------------------------------------------------------------

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        "idx_flex_spatial",
    )

    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "g32_index_tendon_position",
    )

    if tid < 0:
        raise RuntimeError(
            "idx_flex_spatial missing."
        )

    if aid < 0:
        raise RuntimeError(
            "direct tendon actuator missing."
        )

    for name in RETURN_SPRINGS:

        rid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            name,
        )

        if rid >= 0:

            raise RuntimeError(
                f"STOP: unwanted G31/G33 return spring exists: "
                f"{name}"
            )

    print()
    print(
        "G36A ARCHITECTURE CHECK: PASS"
    )

    print(
        "  idx_flex_spatial = preserved"
    )

    print(
        "  direct spatial-tendon actuator = preserved"
    )

    print(
        "  G31/G33 extra return springs = absent"
    )

    print()

    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            JOINTS[
                key
            ],
        )

        dadr = int(
            model.jnt_dofadr[
                jid
            ]
        )

        print(
            f"  {key.upper()} frictionloss = "
            f"{model.dof_frictionloss[dadr]:.6f} Nm"
        )


def episodes(
    rows,
    key,
):

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
        help="G25B baseline model",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g36a_direct_breakaway_hierarchy"
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

    baseline = (
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

    shutil.copy2(
        Path(__file__).resolve(),
        out
        /
        Path(__file__).name,
    )

    direct_xml = (
        out
        /
        "g36a_direct_prepatch.xml"
    )

    candidate_xml = (
        out
        /
        "g36a_direct_breakaway_hierarchy.xml"
    )

    free_xml = (
        out
        /
        "g36a_free.xml"
    )

    # ============================================================
    # STEP 0 — build direct spatial tendon from CLEAN G25B
    # ============================================================

    print()
    print("=" * 150)
    print(
        "G36A — DIRECT SPATIAL TENDON + BREAKAWAY HIERARCHY"
    )
    print("=" * 150)

    g32.build_direct_model(
        baseline,
        direct_xml,
    )

    apply_breakaway_hierarchy(
        direct_xml,
        candidate_xml,
    )

    # ============================================================
    # Encoder -> tendon shortening mapping
    # ============================================================

    (
        open_length,
        ratio,
        mapping_rmse,
    ) = (
        g32.fit_encoder_to_tendon_shortening(
            candidate_xml
        )
    )

    print()
    print(
        f"shortening ratio = "
        f"{ratio:.6f}"
    )

    print(
        f"mapping RMSE     = "
        f"{mapping_rmse:.4f} mm"
    )

    g32.disable_fixture(
        candidate_xml,
        free_xml,
    )

    # ============================================================
    # Replay setup
    # ============================================================

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    g7.b2.v35.METERS_PER_TICK = SCALE

    try:

        # ========================================================
        # STEP 1 — approach
        # ========================================================

        print()
        print("=" * 150)
        print(
            "G36A STEP 1 — H1 APPROACH"
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
        ) in H1_POINTS:

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

        # ========================================================
        # STEP 2 — contact
        # ========================================================

        print()
        print("=" * 150)
        print(
            "G36A STEP 2 — ONE H1 CONTACT RUN"
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
            candidate_xml,
            bag,
            replay,
            open_length,
            ratio,
        )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

    # ============================================================
    # Contact events
    # ============================================================

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

    # ============================================================
    # Final H1
    # ============================================================

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

    # ============================================================
    # Post-contact recruitment
    # ============================================================

    delta_after_contact = (
        final
        -
        first_pose
    )

    print()
    print(
        "POST-CONTACT RECRUITMENT"
    )

    print(
        f"  dMCP = "
        f"{delta_after_contact[0]:+.3f} deg"
    )

    print(
        f"  dPIP = "
        f"{delta_after_contact[1]:+.3f} deg"
    )

    print(
        f"  dDIP = "
        f"{delta_after_contact[2]:+.3f} deg"
    )

    # ============================================================
    # Decision
    # ============================================================

    print()
    print("=" * 150)
    print(
        "G36A DECISION"
    )
    print("=" * 150)

    print(
        f"approach RMSE         = "
        f"{approach_rmse:.3f} deg"
    )

    print(
        f"first-contact RMSE    = "
        f"{first_rmse:.3f} deg"
    )

    print(
        f"first-contact enc err = "
        f"{first_row['encoder_tick'] - HW_CONTACT_ENCODER:+.1f} ticks"
    )

    print(
        f"middle episodes       = "
        f"{len(mid_starts)}"
    )

    print(
        f"middle gap            = "
        f"{gap:.3f} s"
    )

    print(
        f"final RMSE            = "
        f"{final_rmse:.3f} deg"
    )

    print(
        f"post-contact dPIP     = "
        f"{delta_after_contact[1]:+.3f} deg"
    )

    print(
        f"post-contact dDIP     = "
        f"{delta_after_contact[2]:+.3f} deg"
    )

    print()

    if (
        approach_rmse <= 3.5
        and
        first_rmse <= 4.0
        and
        abs(
            first_row[
                "encoder_tick"
            ]
            -
            HW_CONTACT_ENCODER
        )
        <= 100.0
        and
        len(
            mid_starts
        ) == 1
        and
        final_rmse <= 2.5
    ):

        print(
            "PASS: existing breakaway hierarchy + direct spatial "
            "tendon reproduces H1 mechanics."
        )

        print(
            "FREEZE this H1 architecture."
        )

        print(
            "Next: quantitative free-space validation, then H2 "
            "without retuning."
        )

    elif (
        len(
            mid_starts
        ) == 1
        and
        final_rmse <= 3.0
    ):

        print(
            "STRONG PARTIAL PASS:"
        )

        print(
            "contact and final wrapping are mechanically correct."
        )

        print(
            "Do NOT change friction values."
        )

        print(
            "Next: calibrate the encoder/tendon input map from "
            "independent free-space data."
        )

    elif (
        len(
            mid_starts
        ) == 1
        and
        final[
            1
        ]
        >=
        15.0
    ):

        print(
            "PARTIAL:"
        )

        print(
            "contact retention and PIP recruitment survive,"
        )

        print(
            "but staging is still quantitatively wrong."
        )

        print(
            "Next structural candidate = measured MCP/PIP joint "
            "saturation from free-space video 13."
        )

    else:

        print(
            "REJECT this combined mechanism."
        )

        print(
            "No friction scan."
        )

        print(
            "Next = joint-limit / sequential-recruitment model "
            "derived directly from free-space video 13."
        )

    # ============================================================
    # Save
    # ============================================================

    summary_csv = (
        out
        /
        "g36a_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "approach_rmse_deg",
                "first_contact_encoder",
                "first_contact_encoder_error",
                "first_contact_pose_rmse_deg",
                "middle_episodes",
                "tip_episodes",
                "middle_gap_s",
                "final_mcp_deg",
                "final_pip_deg",
                "final_dip_deg",
                "final_rmse_deg",
                "post_contact_dmcp_deg",
                "post_contact_dpip_deg",
                "post_contact_ddip_deg",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "approach_rmse_deg":
                approach_rmse,

            "first_contact_encoder":
                first_row[
                    "encoder_tick"
                ],

            "first_contact_encoder_error":
                first_row[
                    "encoder_tick"
                ]
                -
                HW_CONTACT_ENCODER,

            "first_contact_pose_rmse_deg":
                first_rmse,

            "middle_episodes":
                len(
                    mid_starts
                ),

            "tip_episodes":
                len(
                    tip_starts
                ),

            "middle_gap_s":
                gap,

            "final_mcp_deg":
                final[
                    0
                ],

            "final_pip_deg":
                final[
                    1
                ],

            "final_dip_deg":
                final[
                    2
                ],

            "final_rmse_deg":
                final_rmse,

            "post_contact_dmcp_deg":
                delta_after_contact[
                    0
                ],

            "post_contact_dpip_deg":
                delta_after_contact[
                    1
                ],

            "post_contact_ddip_deg":
                delta_after_contact[
                    2
                ],
        })

    np.savez_compressed(
        out
        /
        "g36a_contact_trajectory.npz",
        times=np.asarray(
            times
        ),
        qpos=np.asarray(
            qpos
        ),
    )

    print()
    print(
        f"Candidate XML:\n  "
        f"{candidate_xml}"
    )

    print()

    print(
        f"Summary:\n  "
        f"{summary_csv}"
    )

    print()

    print(
        f"All outputs:\n  "
        f"{out}"
    )

    # ============================================================
    # GUI
    # ============================================================

    if not args.no_gui:

        view_xml = (
            out
            /
            "g36a_view.xml"
        )

        g29b.make_view_xml(
            candidate_xml,
            view_xml,
        )

        print()
        print("=" * 150)
        print(
            "G36A GUI"
        )
        print("=" * 150)

        print(
            "Watch the sequence:"
        )

        print(
            "  MCP dominates"
        )

        print(
            "  -> middle contact"
        )

        print(
            "  -> PIP wraps while middle remains constrained"
        )

        print(
            "  -> distal/DIP recruited later"
        )

        g7.exact_gui_replay(
            view_xml,
            qpos,
            times,
            "G36A direct spatial tendon + breakaway hierarchy",
        )


if __name__ == "__main__":
    main()
