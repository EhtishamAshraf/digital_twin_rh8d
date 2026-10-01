#!/usr/bin/env python3

import sys
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import fit_index_contact_v1 as base


# ============================================================
# PURPOSE
# ============================================================
#
# Identify MCP/PIP effective tendon moment-arm distribution
# using BOTH:
#
#   1. original hardware free-space trajectory
#   2. hardware loaded bottle-contact pose
#
# Bottle geometry is now fixed from hardware measurement.
#
# We deliberately DO NOT retune:
#
#   spool scale
#   tendon stiffness
#   DIP coefficient
#   bottle position
#   contact law
#
# during this diagnostic.
# ============================================================


BASE_XML = (
    base.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_hw_v1.xml"
)

TEMP_XML = (
    BASE_XML.parent
    / "_tmp_index_moment_arm_scan.xml"
)

RESULT_CSV = (
    base.WS
    / "rh8d_digital_twin_tools/results/"
      "index_moment_arm_scan_v1.csv"
)


# ============================================================
# Candidate A quantities kept frozen
# ============================================================

SCALE = 5.1e-6
TENDON_STIFFNESS = 2100.0

OPEN_ENCODER = 100.0

CONTACT_STIFFNESS = 28000.0
CONTACT_DAMPING = 1.0
CONTACT_FRICTION = 1.2


# ============================================================
# Current/baseline effective coefficients
#
# coefficient sign is negative for flexion in current model.
# ============================================================

BASELINE_MCP_MM = 5.0
BASELINE_PIP_MM = 4.0
DIP_MM = 1.0


# ============================================================
# Search space
#
# We want to test substantially more PIP leverage and somewhat
# less MCP leverage, but include current Candidate A exactly.
# ============================================================

MCP_ARMS_MM = [
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
]

PIP_ARMS_MM = [
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
]


# ============================================================
# Names
# ============================================================

TENDON_NAME = "idx_flex_spatial"

SPOOL_NAME = "index_motor_spool"
ACTUATOR_NAME = "pos_index_tendon"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

FIXTURE_NAME = "index_contact_fixture"

FTS_NAME = "index_tip_force"


# ============================================================
# Hardware FREE-SPACE measurements
#
# Same marker-derived dataset used throughout index ID.
# ============================================================

CLOSING = [
    (100,   0.00,  0.00),
    (500,  13.44,  5.55),
    (1000, 30.85, 10.52),
    (1500, 46.46, 15.50),
    (2000, 58.88, 26.67),
    (2500, 59.85, 41.75),
    (3000, 60.97, 62.23),
    (3500, 64.35, 75.27),
]

OPENING = [
    (3000, 52.24, 69.47),
    (2500, 46.77, 52.50),
    (2000, 37.93, 49.78),
    (1500, 37.92, 31.35),
    (1000, 34.04, 12.98),
    (500,  14.58,  7.93),
    (100,   0.00,  0.00),
]


# ============================================================
# Hardware loaded pose
# ============================================================

HW_LOADED_MCP = 14.5355
HW_LOADED_PIP = 23.8110
HW_LOADED_DIP = 9.3270

VIDEO_TIME = 50.50
VIDEO_BAG_OFFSET = 11.53

TARGET_BAG_TIME = (
    VIDEO_TIME
    - VIDEO_BAG_OFFSET
)

TARGET_HALF_WINDOW = 0.10


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

    displacement = max(
        0.0,
        (
            encoder
            - OPEN_ENCODER
        )
        * SCALE,
    )

    # Kinematic motor-side encoder position.
    data.qpos[
        spool_qadr
    ] = displacement

    data.qvel[
        spool_dofadr
    ] = 0.0

    data.ctrl[
        actuator
    ] = displacement


def set_tendon_coefficients_in_xml(
    mcp_mm,
    pip_mm,
):

    tree = ET.parse(
        BASE_XML
    )

    root = tree.getroot()


    fixed = None

    for element in root.iter(
        "fixed"
    ):

        if element.get(
            "name"
        ) == TENDON_NAME:

            fixed = element
            break


    if fixed is None:

        raise RuntimeError(
            f'Could not find fixed tendon "{TENDON_NAME}"'
        )


    desired = {
        MCP_NAME:
            -mcp_mm / 1000.0,

        PIP_NAME:
            -pip_mm / 1000.0,

        DIP_NAME:
            -DIP_MM / 1000.0,
    }


    found = set()


    for child in fixed:

        if child.tag != "joint":
            continue

        name = child.get(
            "joint"
        )

        if name not in desired:
            continue

        child.set(
            "coef",
            f"{desired[name]:.9f}",
        )

        found.add(
            name
        )


    missing = (
        set(
            desired.keys()
        )
        - found
    )


    if missing:

        raise RuntimeError(
            "Missing tendon joint entries:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )


    tree.write(
        TEMP_XML,
        encoding="unicode",
    )


def settle_at_encoder(
    model,
    data,
    spool_qadr,
    spool_dofadr,
    actuator,
    encoder,
    seconds=2.0,
):

    n = int(
        seconds
        / model.opt.timestep
    )


    for _ in range(n):

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


def get_angles(
    data,
    mcp_q,
    pip_q,
    dip_q,
):

    return (
        math.degrees(
            data.qpos[
                mcp_q
            ]
        ),
        math.degrees(
            data.qpos[
                pip_q
            ]
        ),
        math.degrees(
            data.qpos[
                dip_q
            ]
        ),
    )


def configure_model(
    model,
):

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


    return fixture_gid


def common_ids(
    model,
):

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


    return (
        spool_qadr,
        spool_dofadr,
        actuator,
        mcp_q,
        pip_q,
        dip_q,
        sensor_adr,
    )


def free_space_test(
    model,
    fixture_gid,
    ids,
):

    (
        spool_qadr,
        spool_dofadr,
        actuator,
        mcp_q,
        pip_q,
        dip_q,
        sensor_adr,
    ) = ids


    # --------------------------------------------------------
    # Disable bottle during the original free-space experiment.
    # --------------------------------------------------------

    saved_contype = int(
        model.geom_contype[
            fixture_gid
        ]
    )

    saved_conaffinity = int(
        model.geom_conaffinity[
            fixture_gid
        ]
    )


    model.geom_contype[
        fixture_gid
    ] = 0

    model.geom_conaffinity[
        fixture_gid
    ] = 0


    mujoco.mj_resetData(
        model,
        model_data := mujoco.MjData(
            model
        ),
    )


    # Establish open reference.
    settle_at_encoder(
        model,
        model_data,
        spool_qadr,
        spool_dofadr,
        actuator,
        OPEN_ENCODER,
        seconds=2.0,
    )


    open_mcp, open_pip, open_dip = (
        get_angles(
            model_data,
            mcp_q,
            pip_q,
            dip_q,
        )
    )


    errors_mcp = []
    errors_pip = []

    simulated_rows = []


    # --------------------------------------------------------
    # Closing
    # --------------------------------------------------------

    for (
        encoder,
        hw_mcp,
        hw_pip,
    ) in CLOSING:

        settle_at_encoder(
            model,
            model_data,
            spool_qadr,
            spool_dofadr,
            actuator,
            encoder,
            seconds=2.0,
        )


        mcp, pip, dip = get_angles(
            model_data,
            mcp_q,
            pip_q,
            dip_q,
        )


        mcp -= open_mcp
        pip -= open_pip
        dip -= open_dip


        errors_mcp.append(
            mcp - hw_mcp
        )

        errors_pip.append(
            pip - hw_pip
        )


        simulated_rows.append(
            (
                "closing",
                encoder,
                mcp,
                pip,
                dip,
            )
        )


    # --------------------------------------------------------
    # Opening
    # --------------------------------------------------------

    for (
        encoder,
        hw_mcp,
        hw_pip,
    ) in OPENING:

        settle_at_encoder(
            model,
            model_data,
            spool_qadr,
            spool_dofadr,
            actuator,
            encoder,
            seconds=2.0,
        )


        mcp, pip, dip = get_angles(
            model_data,
            mcp_q,
            pip_q,
            dip_q,
        )


        mcp -= open_mcp
        pip -= open_pip
        dip -= open_dip


        errors_mcp.append(
            mcp - hw_mcp
        )

        errors_pip.append(
            pip - hw_pip
        )


        simulated_rows.append(
            (
                "opening",
                encoder,
                mcp,
                pip,
                dip,
            )
        )


    errors_mcp = np.asarray(
        errors_mcp,
        dtype=float,
    )

    errors_pip = np.asarray(
        errors_pip,
        dtype=float,
    )


    mcp_rmse = float(
        np.sqrt(
            np.mean(
                errors_mcp**2
            )
        )
    )


    pip_rmse = float(
        np.sqrt(
            np.mean(
                errors_pip**2
            )
        )
    )


    combined_rmse = float(
        np.sqrt(
            np.mean(
                np.concatenate(
                    [
                        errors_mcp,
                        errors_pip,
                    ]
                )**2
            )
        )
    )


    # Final return-to-open residual.
    final_row = simulated_rows[
        -1
    ]


    final_open_mcp = abs(
        final_row[2]
    )

    final_open_pip = abs(
        final_row[3]
    )


    # Restore bottle collision properties.
    model.geom_contype[
        fixture_gid
    ] = saved_contype

    model.geom_conaffinity[
        fixture_gid
    ] = saved_conaffinity


    return {
        "free_mcp_rmse":
            mcp_rmse,

        "free_pip_rmse":
            pip_rmse,

        "free_combined_rmse":
            combined_rmse,

        "final_open_mcp":
            final_open_mcp,

        "final_open_pip":
            final_open_pip,
    }


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

            name = (
                f"geom_{other}"
            )


        names.append(
            name
        )


    return "|".join(
        sorted(
            set(
                names
            )
        )
    )


def loaded_contact_test(
    model,
    fixture_gid,
    ids,
    approach_start,
    encoder_time,
    encoder_position,
):

    (
        spool_qadr,
        spool_dofadr,
        actuator,
        mcp_q,
        pip_q,
        dip_q,
        sensor_adr,
    ) = ids


    data = mujoco.MjData(
        model
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    # --------------------------------------------------------
    # Open equilibrium WITH corrected bottle geometry.
    # It should remain collision-free.
    # --------------------------------------------------------

    settle_at_encoder(
        model,
        data,
        spool_qadr,
        spool_dofadr,
        actuator,
        OPEN_ENCODER,
        seconds=2.0,
    )


    open_mcp, open_pip, open_dip = (
        get_angles(
            data,
            mcp_q,
            pip_q,
            dip_q,
        )
    )


    open_contacts = fixture_contacts(
        model,
        data,
        fixture_gid,
    )


    target_mcp = []
    target_pip = []
    target_dip = []
    target_force = []
    target_encoder = []


    nearest_dt = float(
        "inf"
    )

    target_topology = ""


    elapsed = 0.0

    stop_time = (
        TARGET_BAG_TIME
        + TARGET_HALF_WINDOW
    )

    duration = (
        stop_time
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


        mcp, pip, dip = get_angles(
            data,
            mcp_q,
            pip_q,
            dip_q,
        )


        mcp -= open_mcp
        pip -= open_pip
        dip -= open_dip


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

            target_force.append(
                force
            )

            target_encoder.append(
                encoder
            )


        difference = abs(
            bag_time
            - TARGET_BAG_TIME
        )


        if difference < nearest_dt:

            nearest_dt = difference

            target_topology = (
                fixture_contacts(
                    model,
                    data,
                    fixture_gid,
                )
            )


        elapsed += (
            model.opt.timestep
        )


    if not target_mcp:

        raise RuntimeError(
            "No samples found in loaded-pose target window."
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


    force = float(
        np.median(
            target_force
        )
    )


    encoder = float(
        np.median(
            target_encoder
        )
    )


    mcp_error = (
        sim_mcp
        - HW_LOADED_MCP
    )

    pip_error = (
        sim_pip
        - HW_LOADED_PIP
    )

    dip_error = (
        sim_dip
        - HW_LOADED_DIP
    )


    pose_rmse = math.sqrt(
        (
            mcp_error**2
            +
            pip_error**2
        )
        / 2.0
    )


    return {
        "loaded_mcp":
            sim_mcp,

        "loaded_pip":
            sim_pip,

        "loaded_dip":
            sim_dip,

        "loaded_mcp_error":
            mcp_error,

        "loaded_pip_error":
            pip_error,

        "loaded_dip_error":
            dip_error,

        "loaded_pose_rmse":
            pose_rmse,

        "target_force":
            force,

        "target_encoder":
            encoder,

        "target_contacts":
            target_topology,

        "open_contacts":
            open_contacts,
    }


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python3 scan_index_moment_arm_ratio_v1.py "
            "/path/to/contact_rosbag"
        )

        sys.exit(1)


    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    # ========================================================
    # Read real contact encoder trajectory once.
    # ========================================================

    (
        states,
        force_data,
        encoder_data,
    ) = base.read_bag(
        bag
    )


    transitions = base.compress_states(
        states
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


    results = []


    total = (
        len(
            MCP_ARMS_MM
        )
        *
        len(
            PIP_ARMS_MM
        )
    )


    print()
    print("=" * 118)
    print(
        "INDEX EFFECTIVE MOMENT-ARM SCAN V1"
    )
    print("=" * 118)

    print()
    print(
        "Fixed:"
    )

    print(
        f"  spool scale       = "
        f"{SCALE*1e6:.3f} um/tick"
    )

    print(
        f"  tendon stiffness  = "
        f"{TENDON_STIFFNESS:.1f} N/m"
    )

    print(
        f"  DIP arm           = "
        f"{DIP_MM:.1f} mm/rad"
    )

    print(
        "  bottle geometry   = HARDWARE-DERIVED"
    )

    print()
    print(
        f"Testing {total} MCP/PIP coefficient pairs..."
    )


    count = 0


    try:

        for mcp_mm in (
            MCP_ARMS_MM
        ):


            for pip_mm in (
                PIP_ARMS_MM
            ):


                count += 1


                set_tendon_coefficients_in_xml(
                    mcp_mm,
                    pip_mm,
                )


                model = mujoco.MjModel.from_xml_path(
                    str(
                        TEMP_XML
                    )
                )


                fixture_gid = (
                    configure_model(
                        model
                    )
                )


                ids = common_ids(
                    model
                )


                free = free_space_test(
                    model,
                    fixture_gid,
                    ids,
                )


                loaded = loaded_contact_test(
                    model,
                    fixture_gid,
                    ids,
                    approach_start,
                    encoder_time,
                    encoder_position,
                )


                # ---------------------------------------------
                # Diagnostic ranking only.
                #
                # We prioritize loaded pose but penalize
                # degradation beyond approximately the current
                # free-space quality.
                # ---------------------------------------------

                free_penalty = max(
                    0.0,
                    free[
                        "free_combined_rmse"
                    ]
                    - 6.0,
                )


                open_penalty = (
                    free[
                        "final_open_mcp"
                    ]
                    +
                    free[
                        "final_open_pip"
                    ]
                )


                score = (
                    loaded[
                        "loaded_pose_rmse"
                    ]

                    +
                    1.5
                    * free_penalty

                    +
                    0.05
                    * open_penalty

                    +
                    (
                        50.0
                        if loaded[
                            "open_contacts"
                        ]
                        else 0.0
                    )
                )


                row = {
                    "mcp_arm_mm":
                        mcp_mm,

                    "pip_arm_mm":
                        pip_mm,

                    "pip_to_mcp_ratio":
                        pip_mm / mcp_mm,

                    **free,

                    **loaded,

                    "diagnostic_score":
                        score,
                }


                results.append(
                    row
                )


                print(
                    f"[{count:2d}/{total}] "
                    f"MCP={mcp_mm:.1f} "
                    f"PIP={pip_mm:.1f} mm   "
                    f"free={free['free_combined_rmse']:.2f}°   "
                    f"loaded={loaded['loaded_pose_rmse']:.2f}°   "
                    f"pose=({loaded['loaded_mcp']:.1f},"
                    f"{loaded['loaded_pip']:.1f},"
                    f"{loaded['loaded_dip']:.1f})"
                )


    finally:

        if TEMP_XML.exists():

            TEMP_XML.unlink()


    # ========================================================
    # Save
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
    # Baseline
    # ========================================================

    baseline = min(
        results,
        key=lambda r:
            abs(
                r["mcp_arm_mm"]
                - BASELINE_MCP_MM
            )
            +
            abs(
                r["pip_arm_mm"]
                - BASELINE_PIP_MM
            ),
    )


    print()
    print("=" * 118)
    print(
        "BASELINE — CURRENT 5 / 4 / 1 mm EFFECTIVE TRANSMISSION"
    )
    print("=" * 118)

    print(
        f"free RMSE       = "
        f"{baseline['free_combined_rmse']:.3f} deg"
    )

    print(
        f"loaded pose     = "
        f"({baseline['loaded_mcp']:.3f}, "
        f"{baseline['loaded_pip']:.3f}, "
        f"{baseline['loaded_dip']:.3f}) deg"
    )

    print(
        f"loaded RMSE     = "
        f"{baseline['loaded_pose_rmse']:.3f} deg"
    )

    print(
        f"target FTS      = "
        f"{baseline['target_force']:.4f} N"
    )

    print(
        f"contacts        = "
        f"{baseline['target_contacts']}"
    )


    # ========================================================
    # Best loaded pose regardless of free-space degradation
    # ========================================================

    by_loaded = sorted(
        results,
        key=lambda r:
            r[
                "loaded_pose_rmse"
            ],
    )


    print()
    print("=" * 118)
    print(
        "TOP 12 — LOADED POSE ONLY"
    )
    print("=" * 118)


    for rank, r in enumerate(
        by_loaded[:12],
        start=1,
    ):

        print()

        print(
            f"#{rank:2d} "
            f"MCP arm={r['mcp_arm_mm']:.1f} mm   "
            f"PIP arm={r['pip_arm_mm']:.1f} mm   "
            f"ratio={r['pip_to_mcp_ratio']:.3f}"
        )

        print(
            f"     loaded pose = "
            f"({r['loaded_mcp']:.2f}, "
            f"{r['loaded_pip']:.2f}, "
            f"{r['loaded_dip']:.2f}) deg"
        )

        print(
            f"     loaded RMSE = "
            f"{r['loaded_pose_rmse']:.2f} deg"
        )

        print(
            f"     FREE RMSE   = "
            f"{r['free_combined_rmse']:.2f} deg"
        )

        print(
            f"     free MCP/PIP RMSE = "
            f"{r['free_mcp_rmse']:.2f} / "
            f"{r['free_pip_rmse']:.2f} deg"
        )

        print(
            f"     FTS = "
            f"{r['target_force']:.4f} N   "
            f"contacts={r['target_contacts']}"
        )


    # ========================================================
    # Candidates that preserve free-space quality reasonably
    # ========================================================

    free_preserving = [
        r
        for r in results
        if (
            r[
                "free_combined_rmse"
            ] <= 6.5
            and not r[
                "open_contacts"
            ]
        )
    ]


    free_preserving = sorted(
        free_preserving,
        key=lambda r:
            r[
                "loaded_pose_rmse"
            ],
    )


    print()
    print("=" * 118)
    print(
        "TOP CANDIDATES WITH FREE-SPACE RMSE <= 6.5 DEG"
    )
    print("=" * 118)


    if not free_preserving:

        print(
            "NONE"
        )

    else:

        for rank, r in enumerate(
            free_preserving[:15],
            start=1,
        ):

            print()

            print(
                f"#{rank:2d} "
                f"MCP={r['mcp_arm_mm']:.1f} mm   "
                f"PIP={r['pip_arm_mm']:.1f} mm   "
                f"ratio={r['pip_to_mcp_ratio']:.3f}"
            )

            print(
                f"     free RMSE   = "
                f"{r['free_combined_rmse']:.2f} deg"
            )

            print(
                f"     loaded pose = "
                f"({r['loaded_mcp']:.2f}, "
                f"{r['loaded_pip']:.2f}, "
                f"{r['loaded_dip']:.2f}) deg"
            )

            print(
                f"     loaded RMSE = "
                f"{r['loaded_pose_rmse']:.2f} deg"
            )

            print(
                f"     final open  = "
                f"MCP {r['final_open_mcp']:.2f}, "
                f"PIP {r['final_open_pip']:.2f} deg"
            )

            print(
                f"     FTS         = "
                f"{r['target_force']:.4f} N"
            )


    # ========================================================
    # Strong provisional candidates
    # ========================================================

    strong = [
        r
        for r in results
        if (
            r[
                "free_combined_rmse"
            ] <= 6.5

            and

            r[
                "loaded_pose_rmse"
            ] <= 8.0

            and

            not r[
                "open_contacts"
            ]
        )
    ]


    print()
    print("=" * 118)
    print(
        "STRONG JOINT FREE/CONTACT CANDIDATES"
    )
    print("=" * 118)


    if not strong:

        print(
            "NONE"
        )

    else:

        for r in sorted(
            strong,
            key=lambda r:
                r[
                    "loaded_pose_rmse"
                ],
        ):

            print(
                f"MCP={r['mcp_arm_mm']:.1f} mm   "
                f"PIP={r['pip_arm_mm']:.1f} mm   "
                f"free={r['free_combined_rmse']:.2f} deg   "
                f"loaded={r['loaded_pose_rmse']:.2f} deg   "
                f"pose=("
                f"{r['loaded_mcp']:.2f}, "
                f"{r['loaded_pip']:.2f}, "
                f"{r['loaded_dip']:.2f})"
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
