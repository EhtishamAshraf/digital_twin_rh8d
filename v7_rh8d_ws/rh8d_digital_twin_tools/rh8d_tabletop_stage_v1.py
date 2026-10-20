#!/usr/bin/env python3

import mujoco


# =====================================================================
# FROZEN RH8D TABLETOP EXPERIMENT STAGING
#
# DO NOT MODIFY THESE VALUES IN LATER EXPERIMENTS.
#
# This is the known-good UR5e configuration used for the RH8D
# tabletop bottle experiments.
# =====================================================================

TABLETOP_UR5 = {
    "shoulder_pan_joint":   1.4927,
    "shoulder_lift_joint": -0.8678,
    "elbow_joint":         -2.4300,
    "wrist_1_joint":       -0.6596,
    "wrist_2_joint":       -0.0347,
    "wrist_3_joint":        0.1260,
}


# Actuator names corresponding to the UR5 joints.
UR5_ACTUATORS = {
    "shoulder_pan_joint":  "shoulder_pan",
    "shoulder_lift_joint": "shoulder_lift",
    "elbow_joint":         "elbow",
    "wrist_1_joint":       "wrist_1",
    "wrist_2_joint":       "wrist_2",
    "wrist_3_joint":       "wrist_3",
}


# RH8D mounting/configuration used for tabletop experiments.
HAND_STAGE = {
    "forearm:1--base:1":             0.0,
    "palm_axis:1--forearm:1":        0.0,
    "palm_assembly:1--palm_axis:1":  0.0,
    "Thumb_axis--palm_assembly:1":   1.5708,
}


HAND_ACTUATORS = {
    "forearm:1--base:1":            "pos_forearm",
    "palm_axis:1--forearm:1":       "pos_palm_axis",
    "palm_assembly:1--palm_axis:1": "pos_palm_assembly",
    "Thumb_axis--palm_assembly:1":  "pos_thumb_axis",
}


def set_joint_qpos(
    model,
    data,
    joint_name,
    value,
):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if jid < 0:
        raise RuntimeError(
            f"Required staging joint missing: {joint_name}"
        )

    qadr = int(
        model.jnt_qposadr[jid]
    )

    data.qpos[qadr] = float(value)


def mirror_actuator(
    model,
    data,
    actuator_name,
    value,
):
    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )

    if aid >= 0:
        data.ctrl[aid] = float(value)


def apply_tabletop_stage(
    model,
    data,
    *,
    reset_to_qpos0=True,
):
    """
    Put UR5e + RH8D mounting joints in the FROZEN tabletop
    experimental configuration.

    IMPORTANT:
    This function does NOT set MCP/PIP/DIP. The calling experiment
    owns the index-finger configuration.
    """

    if reset_to_qpos0:
        data.qpos[:] = model.qpos0

    data.qvel[:] = 0.0

    # -------------------------------------------------------------
    # UR5e
    # -------------------------------------------------------------
    for joint_name, value in TABLETOP_UR5.items():

        set_joint_qpos(
            model,
            data,
            joint_name,
            value,
        )

        mirror_actuator(
            model,
            data,
            UR5_ACTUATORS[joint_name],
            value,
        )

    # -------------------------------------------------------------
    # RH8D mounting/orientation
    # -------------------------------------------------------------
    for joint_name, value in HAND_STAGE.items():

        set_joint_qpos(
            model,
            data,
            joint_name,
            value,
        )

        mirror_actuator(
            model,
            data,
            HAND_ACTUATORS[joint_name],
            value,
        )

    mujoco.mj_forward(
        model,
        data,
    )


def verify_tabletop_scene(
    model,
    data,
):
    """
    Hard gate: experiment is not allowed to continue if the
    tabletop/bottle objects disappeared.
    """

    fixture_gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "index_contact_fixture",
    )

    table_bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "table",
    )

    if fixture_gid < 0:
        raise RuntimeError(
            "STOP: index_contact_fixture / bottle is missing."
        )

    if table_bid < 0:
        raise RuntimeError(
            "STOP: tabletop body is missing."
        )

    mujoco.mj_forward(
        model,
        data,
    )

    bottle_pos = data.geom_xpos[
        fixture_gid
    ]

    print()
    print("=" * 100)
    print("FROZEN TABLETOP STAGING: PASS")
    print("=" * 100)

    print(
        "UR5e:"
    )

    for name, value in TABLETOP_UR5.items():
        print(
            f"  {name:28s} = {value:+.4f} rad"
        )

    print()
    print(
        "table                  = PRESENT"
    )

    print(
        "index bottle fixture   = PRESENT"
    )

    print(
        "bottle world position  = "
        f"{bottle_pos[0]:+.6f} "
        f"{bottle_pos[1]:+.6f} "
        f"{bottle_pos[2]:+.6f}"
    )

    print("=" * 100)
