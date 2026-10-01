#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import mujoco
import numpy as np

import fit_index_contact_v1 as base


# ============================================================
# CANDIDATE A
# ============================================================

MODEL_PATH = (
    base.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_hw_v1.xml"
)

SCALE = 5.1e-6
TENDON_STIFFNESS = 2100.0

CONTACT_STIFFNESS = 28000.0
CONTACT_DAMPING = 1.0
CONTACT_FRICTION = 1.2

OPEN_ENCODER = 100.0


# ============================================================
# VIDEO <-> BAG ALIGNMENT
#
# Previously identified:
#
# video_time ~= bag_time + 11.53 s
# ============================================================

VIDEO_TIME = 50.50
VIDEO_BAG_OFFSET = 11.53

TARGET_BAG_TIME = (
    VIDEO_TIME
    - VIDEO_BAG_OFFSET
)

TARGET_HALF_WINDOW = 0.10


# ============================================================
# HARDWARE MARKER RESULT
#
# Mean of the two manual measurements.
# ============================================================

HW_MCP = 14.5355
HW_PIP = 23.8110
HW_DIP = 9.3270

HW_MCP_UNCERT = 0.9655
HW_PIP_UNCERT = 0.8520
HW_DIP_UNCERT = 1.2670


TENDON_NAME = "idx_flex_spatial"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

SPOOL_NAME = "index_motor_spool"
ACTUATOR_NAME = "pos_index_tendon"

FIXTURE_NAME = "index_contact_fixture"


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


def joint_qpos_address(
    model,
    name,
):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return model.jnt_qposadr[
        jid
    ]


def set_spool(
    data,
    qadr,
    dofadr,
    actuator,
    encoder,
):

    spool = max(
        0.0,
        (
            encoder
            - OPEN_ENCODER
        ) * SCALE,
    )

    data.qpos[
        qadr
    ] = spool

    data.qvel[
        dofadr
    ] = 0.0

    data.ctrl[
        actuator
    ] = spool


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python3 compare_index_loaded_pose_candidate_a.py "
            "/path/to/contact_rosbag"
        )

        sys.exit(1)


    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    print()
    print("=" * 96)
    print(
        "RH8D INDEX — LOADED POSE CHECK — CANDIDATE A"
    )
    print("=" * 96)


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


    print()
    print(
        f"Approach start = {approach_start:.6f} s"
    )

    print(
        f"Hold start     = {hold_start:.6f} s"
    )

    print(
        f"Hold end       = {hold_end:.6f} s"
    )

    print(
        f"50.50-s video corresponds to "
        f"bag time ≈ {TARGET_BAG_TIME:.6f} s"
    )


    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(
        model
    )


    # Same isolated system-ID condition.
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


    spool_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_NAME,
    )

    spool_qadr = model.jnt_qposadr[
        spool_jid
    ]

    spool_dofadr = model.jnt_dofadr[
        spool_jid
    ]


    actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAME,
    )


    mcp_q = joint_qpos_address(
        model,
        MCP_NAME,
    )

    pip_q = joint_qpos_address(
        model,
        PIP_NAME,
    )

    dip_q = joint_qpos_address(
        model,
        DIP_NAME,
    )


    # ========================================================
    # OPEN REFERENCE
    # ========================================================

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


    print()
    print("=" * 96)
    print("SIMULATED OPEN REFERENCE")
    print("=" * 96)

    print(
        f"MCP = {open_mcp:.4f} deg"
    )

    print(
        f"PIP = {open_pip:.4f} deg"
    )

    print(
        f"DIP = {open_dip:.4f} deg"
    )


    # ========================================================
    # REPLAY REAL CONTACT ENCODER TRAJECTORY
    # ========================================================

    times = []
    encoders = []

    mcp_change = []
    pip_change = []
    dip_change = []


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


        mcp = math.degrees(
            data.qpos[
                mcp_q
            ]
        )

        pip = math.degrees(
            data.qpos[
                pip_q
            ]
        )

        dip = math.degrees(
            data.qpos[
                dip_q
            ]
        )


        times.append(
            bag_time
        )

        encoders.append(
            encoder
        )

        mcp_change.append(
            mcp - open_mcp
        )

        pip_change.append(
            pip - open_pip
        )

        dip_change.append(
            dip - open_dip
        )


        elapsed += dt


    times = np.asarray(
        times
    )

    encoders = np.asarray(
        encoders
    )

    mcp_change = np.asarray(
        mcp_change
    )

    pip_change = np.asarray(
        pip_change
    )

    dip_change = np.asarray(
        dip_change
    )


    # ========================================================
    # FRAME-MATCHED WINDOW
    # ========================================================

    frame_mask = (
        (times >= (
            TARGET_BAG_TIME
            - TARGET_HALF_WINDOW
        ))
        &
        (times <= (
            TARGET_BAG_TIME
            + TARGET_HALF_WINDOW
        ))
    )


    if not np.any(
        frame_mask
    ):

        raise RuntimeError(
            "No simulation samples around "
            "the aligned video frame."
        )


    sim_mcp = float(
        np.median(
            mcp_change[
                frame_mask
            ]
        )
    )

    sim_pip = float(
        np.median(
            pip_change[
                frame_mask
            ]
        )
    )

    sim_dip = float(
        np.median(
            dip_change[
                frame_mask
            ]
        )
    )


    frame_encoder = float(
        np.median(
            encoders[
                frame_mask
            ]
        )
    )


    # ========================================================
    # FULL HOLD MEDIAN
    # ========================================================

    hold_mask = (
        (times >= hold_start)
        &
        (times <= hold_end)
    )


    hold_mcp = float(
        np.median(
            mcp_change[
                hold_mask
            ]
        )
    )

    hold_pip = float(
        np.median(
            pip_change[
                hold_mask
            ]
        )
    )

    hold_dip = float(
        np.median(
            dip_change[
                hold_mask
            ]
        )
    )


    # ========================================================
    # ERROR
    # ========================================================

    error_mcp = (
        sim_mcp
        - HW_MCP
    )

    error_pip = (
        sim_pip
        - HW_PIP
    )

    error_dip = (
        sim_dip
        - HW_DIP
    )


    mcp_pip_rmse = math.sqrt(
        (
            error_mcp**2
            +
            error_pip**2
        )
        / 2.0
    )


    print()
    print("=" * 96)
    print("HARDWARE MARKER REFERENCE")
    print("=" * 96)

    print(
        f"MCP = {HW_MCP:.3f} "
        f"+/- {HW_MCP_UNCERT:.3f} deg"
    )

    print(
        f"PIP = {HW_PIP:.3f} "
        f"+/- {HW_PIP_UNCERT:.3f} deg"
    )

    print(
        f"DIP = {HW_DIP:.3f} "
        f"+/- {HW_DIP_UNCERT:.3f} deg"
    )


    print()
    print("=" * 96)
    print(
        "SIMULATION AT VIDEO-MATCHED CONTACT POSE"
    )
    print("=" * 96)

    print(
        f"Aligned bag time   = "
        f"{TARGET_BAG_TIME:.3f} s"
    )

    print(
        f"Median encoder     = "
        f"{frame_encoder:.1f} ticks"
    )

    print()
    print(
        f"MCP change = {sim_mcp:.3f} deg"
    )

    print(
        f"PIP change = {sim_pip:.3f} deg"
    )

    print(
        f"DIP change = {sim_dip:.3f} deg"
    )


    print()
    print("=" * 96)
    print("POSE ERROR: SIM - HARDWARE")
    print("=" * 96)

    print(
        f"MCP error = {error_mcp:+.3f} deg"
    )

    print(
        f"PIP error = {error_pip:+.3f} deg"
    )

    print(
        f"DIP error = {error_dip:+.3f} deg"
    )

    print(
        f"MCP/PIP pose RMSE = "
        f"{mcp_pip_rmse:.3f} deg"
    )


    print()
    print("=" * 96)
    print("SIMULATION FULL-HOLD MEDIAN")
    print("=" * 96)

    print(
        f"MCP = {hold_mcp:.3f} deg"
    )

    print(
        f"PIP = {hold_pip:.3f} deg"
    )

    print(
        f"DIP = {hold_dip:.3f} deg"
    )


    print()
    print("=" * 96)

    print(
        "NOTE: DIP remains diagnostic only; "
        "MCP/PIP are the primary quantitative check."
    )

    print("=" * 96)


if __name__ == "__main__":
    main()
