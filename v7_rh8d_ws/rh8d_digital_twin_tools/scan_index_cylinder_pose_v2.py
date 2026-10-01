#!/usr/bin/env python3

import sys
import csv
import math
from pathlib import Path

import mujoco
import numpy as np

import fit_index_contact_v1 as base
import fit_index_contact_v1_1 as corrected


# ============================================================
# CYLINDER-POSE DIAGNOSTIC
#
# Everything about Candidate A is frozen.
#
# ONLY the measured 69.2-mm bottle center is translated.
#
# Purpose:
# determine whether the wrong loaded MCP/PIP redistribution
# is primarily caused by obstacle placement.
# ============================================================


MODEL_PATH = (
    base.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_v1.xml"
)


RESULT_CSV = (
    base.WS
    / "rh8d_digital_twin_tools/results/"
      "index_cylinder_pose_scan_v2.csv"
)


# ============================================================
# Candidate A -- frozen
# ============================================================

SCALE = 5.1e-6

TENDON_STIFFNESS = 2100.0

CONTACT_STIFFNESS = 28000.0
CONTACT_DAMPING = 1.0
CONTACT_FRICTION = 1.2

OPEN_ENCODER = 100.0


# ============================================================
# Hardware reference
# ============================================================

HW_MCP = 14.5355
HW_PIP = 23.8110
HW_DIP = 9.3270

HW_HOLD_FORCE = 0.14563
HW_ONSET_TICKS = 1020.0


VIDEO_TIME = 50.50
VIDEO_BAG_OFFSET = 11.53

TARGET_BAG_TIME = (
    VIDEO_TIME
    - VIDEO_BAG_OFFSET
)

TARGET_HALF_WINDOW = 0.10


# ============================================================
# Current cylinder basis
#
# Obtained from our measured-cylinder builder.
#
# NORMAL:
# cylinder center -> finger
#
# Therefore +NORMAL means moving bottle CLOSER.
#
# TANGENT:
# approximately along the distal finger direction.
#
# Therefore -TANGENT moves bottle toward the proximal/palm side.
# ============================================================

NORMAL = np.array([
    0.62455741,
    0.69653784,
    0.35321817,
])

TANGENT = np.array([
     0.55735692,
    -0.71434571,
     0.42315893,
])

NORMAL /= np.linalg.norm(NORMAL)
TANGENT /= np.linalg.norm(TANGENT)


# ============================================================
# Search range
#
# Normal shifts are deliberately small because fingertip
# motion near contact is sensitive to radial placement.
#
# Tangential shifts are wider because we specifically want to
# test whether moving the bottle toward the proximal side causes
# earlier distal-link obstruction and more PIP flexion.
# ============================================================

NORMAL_OFFSETS_MM = [
    -0.5,
     0.0,
     0.5,
     1.0,
     1.5,
     2.0,
]

TANGENT_OFFSETS_MM = [
    -8.0,
    -6.0,
    -4.0,
    -2.0,
     0.0,
     2.0,
     4.0,
]


TENDON_NAME = "idx_flex_spatial"

SPOOL_NAME = "index_motor_spool"
ACTUATOR_NAME = "pos_index_tendon"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

FIXTURE_NAME = "index_contact_fixture"
FTS_NAME = "index_tip_force"


def oid(model, typ, name):

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


def qpos_address(
    model,
    joint_name,
):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    return model.jnt_qposadr[
        jid
    ]


def set_spool(
    data,
    spool_qadr,
    spool_dofadr,
    actuator,
    encoder,
):

    q = max(
        0.0,
        (
            encoder
            - OPEN_ENCODER
        ) * SCALE,
    )

    data.qpos[
        spool_qadr
    ] = q

    data.qvel[
        spool_dofadr
    ] = 0.0

    data.ctrl[
        actuator
    ] = q


def fixture_contacts(
    model,
    data,
    fixture_gid,
):

    names = []

    for i in range(
        data.ncon
    ):

        c = data.contact[i]

        if (
            c.geom1 != fixture_gid
            and
            c.geom2 != fixture_gid
        ):
            continue


        other = (
            c.geom2
            if c.geom1 == fixture_gid
            else c.geom1
        )


        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            other,
        )


        if name is None:
            name = f"geom_{other}"


        names.append(
            name
        )


    return sorted(
        set(names)
    )


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python3 scan_index_cylinder_pose_v2.py "
            "/path/to/contact_rosbag"
        )

        sys.exit(1)


    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    # ========================================================
    # Hardware data
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


    encoder_time = (
        encoder_data[:, 0]
    )


    encoder_position = (
        base.median_filter_5(
            encoder_data[:, 1]
        )
    )


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
    # MuJoCo model
    # ========================================================

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
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
    ] = TENDON_STIFFNESS


    fixture_gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FIXTURE_NAME,
    )


    original_center = (
        model.geom_pos[
            fixture_gid
        ].copy()
    )


    model.geom_friction[
        fixture_gid,
        0
    ] = CONTACT_FRICTION


    model.geom_solref[
        fixture_gid,
        0
    ] = -CONTACT_STIFFNESS


    model.geom_solref[
        fixture_gid,
        1
    ] = -CONTACT_DAMPING


    spool_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_NAME,
    )


    spool_qadr = (
        model.jnt_qposadr[
            spool_jid
        ]
    )


    spool_dofadr = (
        model.jnt_dofadr[
            spool_jid
        ]
    )


    actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAME,
    )


    mcp_q = qpos_address(
        model,
        MCP_NAME,
    )

    pip_q = qpos_address(
        model,
        PIP_NAME,
    )

    dip_q = qpos_address(
        model,
        DIP_NAME,
    )


    sensor_id = oid(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        FTS_NAME,
    )


    sensor_adr = (
        model.sensor_adr[
            sensor_id
        ]
    )


    # ========================================================
    # Search
    # ========================================================

    results = []


    total = (
        len(NORMAL_OFFSETS_MM)
        *
        len(TANGENT_OFFSETS_MM)
    )


    count = 0


    print()
    print("=" * 118)
    print(
        "RH8D INDEX — CYLINDER POSITION SCAN V2"
    )
    print("=" * 118)

    print()
    print(
        "Candidate A is frozen:"
    )

    print(
        f"  scale            = "
        f"{SCALE*1e6:.3f} um/tick"
    )

    print(
        f"  tendon stiffness = "
        f"{TENDON_STIFFNESS:.1f} N/m"
    )

    print(
        f"  bottle diameter  = 69.2 mm"
    )

    print()
    print(
        "Hardware loaded pose:"
    )

    print(
        f"  MCP = {HW_MCP:.3f} deg"
    )

    print(
        f"  PIP = {HW_PIP:.3f} deg"
    )

    print(
        f"  DIP = {HW_DIP:.3f} deg"
    )

    print()
    print(
        f"Testing {total} bottle positions..."
    )


    for normal_mm in (
        NORMAL_OFFSETS_MM
    ):


        for tangent_mm in (
            TANGENT_OFFSETS_MM
        ):


            count += 1


            shift = (
                NORMAL
                * (
                    normal_mm
                    / 1000.0
                )
                +
                TANGENT
                * (
                    tangent_mm
                    / 1000.0
                )
            )


            model.geom_pos[
                fixture_gid
            ] = (
                original_center
                + shift
            )


            # =================================================
            # Reset and establish the candidate open reference
            # =================================================

            mujoco.mj_resetData(
                model,
                data,
            )


            for _ in range(
                int(
                    2.0
                    / model.opt.timestep
                )
            ):

                set_spool(
                    data,
                    spool_qadr,
                    spool_dofadr,
                    actuator,
                    OPEN_ENCODER,
                )

                mujoco.mj_step(
                    model,
                    data,
                )


            open_contacts = (
                fixture_contacts(
                    model,
                    data,
                    fixture_gid,
                )
            )


            open_mcp = math.degrees(
                data.qpos[
                    mcp_q
                ]
            )

            open_pip = math.degrees(
                data.qpos[
                    pip_q
                ]
            )

            open_dip = math.degrees(
                data.qpos[
                    dip_q
                ]
            )


            # =================================================
            # Replay measured hardware encoder
            # =================================================

            sim_times = []
            sim_encoder = []
            sim_fts = []

            target_mcp = []
            target_pip = []
            target_dip = []
            target_encoder = []

            raw_hold_contact = []

            nearest_target_dt = float(
                "inf"
            )

            target_topology = []


            elapsed = 0.0

            dt = model.opt.timestep

            duration = (
                hold_end
                - approach_start
            )


            while elapsed <= duration:

                bag_time = (
                    approach_start
                    + elapsed
                )


                encoder = float(
                    np.interp(
                        bag_time,
                        encoder_time,
                        encoder_position,
                    )
                )


                set_spool(
                    data,
                    spool_qadr,
                    spool_dofadr,
                    actuator,
                    encoder,
                )


                mujoco.mj_step(
                    model,
                    data,
                )


                xyz = (
                    data.sensordata[
                        sensor_adr:
                        sensor_adr + 3
                    ]
                )


                force = float(
                    np.linalg.norm(
                        xyz
                    )
                )


                mcp = (
                    math.degrees(
                        data.qpos[
                            mcp_q
                        ]
                    )
                    - open_mcp
                )

                pip = (
                    math.degrees(
                        data.qpos[
                            pip_q
                        ]
                    )
                    - open_pip
                )

                dip = (
                    math.degrees(
                        data.qpos[
                            dip_q
                        ]
                    )
                    - open_dip
                )


                sim_times.append(
                    bag_time
                )

                sim_encoder.append(
                    encoder
                )

                sim_fts.append(
                    force
                )


                if (
                    abs(
                        bag_time
                        - TARGET_BAG_TIME
                    )
                    <= TARGET_HALF_WINDOW
                ):

                    target_mcp.append(
                        mcp
                    )

                    target_pip.append(
                        pip
                    )

                    target_dip.append(
                        dip
                    )

                    target_encoder.append(
                        encoder
                    )


                if (
                    hold_start
                    <= bag_time
                    <= hold_end
                ):

                    raw_hold_contact.append(
                        force > 0.001
                    )


                difference = abs(
                    bag_time
                    - TARGET_BAG_TIME
                )


                if difference < nearest_target_dt:

                    nearest_target_dt = (
                        difference
                    )

                    target_topology = (
                        fixture_contacts(
                            model,
                            data,
                            fixture_gid,
                        )
                    )


                elapsed += dt


            sim_times = np.asarray(
                sim_times
            )

            sim_encoder = np.asarray(
                sim_encoder
            )

            sim_fts = np.asarray(
                sim_fts
            )


            # =================================================
            # Hardware-equivalent FTS filtering
            # =================================================

            sampled_fts = np.interp(
                fts_times,
                sim_times,
                sim_fts,
            )


            filtered_fts = (
                corrected.causal_median_5(
                    sampled_fts
                )
            )


            hold_mask = (
                (fts_times >= hold_start)
                &
                (fts_times <= hold_end)
            )


            filtered_hold = (
                filtered_fts[
                    hold_mask
                ]
            )


            hold_median = float(
                np.median(
                    filtered_hold
                )
            )


            hold_std = float(
                np.std(
                    filtered_hold
                )
            )


            hold_force_error = abs(
                hold_median
                - HW_HOLD_FORCE
            )


            # =================================================
            # 0.10-N threshold
            # =================================================

            crossing = np.where(
                filtered_fts
                >= 0.10
            )[0]


            if len(crossing):

                j = crossing[0]


                threshold_ticks = float(
                    np.interp(
                        fts_times[j],
                        sim_times,
                        sim_encoder,
                    )
                )


                onset_error = abs(
                    threshold_ticks
                    - HW_ONSET_TICKS
                )


            else:

                threshold_ticks = (
                    math.nan
                )

                onset_error = (
                    math.inf
                )


            # =================================================
            # Video-aligned pose
            # =================================================

            if not target_mcp:

                raise RuntimeError(
                    "No samples in target window."
                )


            sim_mcp = float(
                np.median(
                    target_mcp
                )
            )

            sim_pip = float(
                np.median(
                    target_pip
                )
            )

            sim_dip = float(
                np.median(
                    target_dip
                )
            )


            median_target_encoder = float(
                np.median(
                    target_encoder
                )
            )


            mcp_error = (
                sim_mcp
                - HW_MCP
            )

            pip_error = (
                sim_pip
                - HW_PIP
            )

            dip_error = (
                sim_dip
                - HW_DIP
            )


            pose_rmse = math.sqrt(
                (
                    mcp_error**2
                    +
                    pip_error**2
                )
                / 2.0
            )


            duty = float(
                np.mean(
                    raw_hold_contact
                )
            )


            # Balanced diagnostic score.
            #
            # POSE is intentionally dominant.
            #
            # This score is only for ranking this diagnostic
            # sweep; it is NOT a final identification objective.
            score = (

                pose_rmse

                +
                40.0
                * hold_force_error

                +
                (
                    0.03
                    * onset_error
                    if np.isfinite(
                        onset_error
                    )
                    else 100.0
                )

                +
                4.0
                * max(
                    0.0,
                    0.80 - duty,
                )

                +
                (
                    50.0
                    if open_contacts
                    else 0.0
                )
            )


            geometry_feasible = (

                not open_contacts

                and pose_rmse <= 8.0

                and hold_force_error <= 0.040

                and onset_error <= 30.0

                and duty >= 0.80
            )


            center = (
                model.geom_pos[
                    fixture_gid
                ].copy()
            )


            results.append({

                "normal_offset_mm":
                    normal_mm,

                "tangent_offset_mm":
                    tangent_mm,

                "center_x":
                    center[0],

                "center_y":
                    center[1],

                "center_z":
                    center[2],

                "target_encoder_ticks":
                    median_target_encoder,

                "sim_mcp_deg":
                    sim_mcp,

                "sim_pip_deg":
                    sim_pip,

                "sim_dip_deg":
                    sim_dip,

                "mcp_error_deg":
                    mcp_error,

                "pip_error_deg":
                    pip_error,

                "dip_error_deg":
                    dip_error,

                "mcp_pip_pose_rmse_deg":
                    pose_rmse,

                "hold_median_N":
                    hold_median,

                "hold_std_N":
                    hold_std,

                "hold_force_error_N":
                    hold_force_error,

                "contact_duty":
                    duty,

                "threshold_ticks":
                    threshold_ticks,

                "onset_error_ticks":
                    onset_error,

                "open_contacts":
                    "|".join(
                        open_contacts
                    ),

                "target_contacts":
                    "|".join(
                        target_topology
                    ),

                "diagnostic_score":
                    score,

                "geometry_feasible":
                    geometry_feasible,
            })


            if (
                count % 7 == 0
                or count == total
            ):

                best = min(
                    results,
                    key=lambda r:
                        r[
                            "diagnostic_score"
                        ]
                )


                print(
                    f"[{count:2d}/{total}] "
                    f"best score="
                    f"{best['diagnostic_score']:.3f}   "
                    f"normal="
                    f"{best['normal_offset_mm']:+.1f} mm   "
                    f"tangent="
                    f"{best['tangent_offset_mm']:+.1f} mm   "
                    f"pose RMSE="
                    f"{best['mcp_pip_pose_rmse_deg']:.2f} deg"
                )


    # Restore original position.
    model.geom_pos[
        fixture_gid
    ] = original_center


    # ========================================================
    # Save CSV
    # ========================================================

    RESULT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        RESULT_CSV,
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
    # Reports
    # ========================================================

    by_score = sorted(
        results,
        key=lambda r:
            r["diagnostic_score"],
    )


    by_pose = sorted(
        results,
        key=lambda r:
            r[
                "mcp_pip_pose_rmse_deg"
            ],
    )


    print()
    print("=" * 118)
    print(
        "TOP 15 — LOADED POSE MATCH"
    )
    print("=" * 118)


    for rank, r in enumerate(
        by_pose[:15],
        start=1,
    ):

        print()

        print(
            f"#{rank:2d} "
            f"normal={r['normal_offset_mm']:+.1f} mm   "
            f"tangent={r['tangent_offset_mm']:+.1f} mm"
        )

        print(
            f"     SIM pose: "
            f"MCP={r['sim_mcp_deg']:.2f}   "
            f"PIP={r['sim_pip_deg']:.2f}   "
            f"DIP={r['sim_dip_deg']:.2f} deg"
        )

        print(
            f"     pose RMSE="
            f"{r['mcp_pip_pose_rmse_deg']:.2f} deg   "
            f"hold={r['hold_median_N']:.4f} N   "
            f"duty={100*r['contact_duty']:.1f}%"
        )

        print(
            f"     onset="
            f"{r['threshold_ticks']:.1f} ticks   "
            f"contacts="
            f"{r['target_contacts']}"
        )


    print()
    print("=" * 118)
    print(
        "TOP 15 — BALANCED GEOMETRY SCORE"
    )
    print("=" * 118)


    for rank, r in enumerate(
        by_score[:15],
        start=1,
    ):

        print()

        print(
            f"#{rank:2d} "
            f"score={r['diagnostic_score']:.3f}   "
            f"normal={r['normal_offset_mm']:+.1f} mm   "
            f"tangent={r['tangent_offset_mm']:+.1f} mm"
        )

        print(
            f"     pose="
            f"({r['sim_mcp_deg']:.2f}, "
            f"{r['sim_pip_deg']:.2f}, "
            f"{r['sim_dip_deg']:.2f}) deg   "
            f"RMSE="
            f"{r['mcp_pip_pose_rmse_deg']:.2f} deg"
        )

        print(
            f"     hold="
            f"{r['hold_median_N']:.4f} N   "
            f"force error="
            f"{r['hold_force_error_N']:.4f} N"
        )

        print(
            f"     onset="
            f"{r['threshold_ticks']:.1f} ticks   "
            f"duty="
            f"{100*r['contact_duty']:.1f}%"
        )

        print(
            f"     contacts="
            f"{r['target_contacts']}"
        )


    feasible = [
        r
        for r in results
        if r[
            "geometry_feasible"
        ]
    ]


    print()
    print("=" * 118)
    print(
        "GEOMETRY-FEASIBLE CANDIDATES"
    )
    print("=" * 118)


    if not feasible:

        print(
            "NONE"
        )

    else:

        for r in sorted(
            feasible,
            key=lambda r:
                r[
                    "diagnostic_score"
                ],
        ):

            print(
                f"normal="
                f"{r['normal_offset_mm']:+.1f} mm   "
                f"tangent="
                f"{r['tangent_offset_mm']:+.1f} mm   "
                f"poseRMSE="
                f"{r['mcp_pip_pose_rmse_deg']:.2f} deg   "
                f"hold="
                f"{r['hold_median_N']:.4f} N   "
                f"onset="
                f"{r['threshold_ticks']:.1f}   "
                f"duty="
                f"{100*r['contact_duty']:.1f}%   "
                f"contacts="
                f"{r['target_contacts']}"
            )


    print()
    print("Original cylinder centre:")

    print(
        "  "
        f"{original_center[0]:.9f} "
        f"{original_center[1]:.9f} "
        f"{original_center[2]:.9f}"
    )


    print()
    print("Saved:")
    print(
        RESULT_CSV
    )

    print()
    print("=" * 118)


if __name__ == "__main__":
    main()
