#!/usr/bin/env python3

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


WS = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_tabletop_sysid_v1.xml"
)


TENDON_STIFFNESS = 2100.0

TENDON_NAME = "idx_flex_spatial"

FIXTURE_NAME = "index_contact_fixture"


ARM_POSE = {

    "shoulder_pan_joint":
         1.4927,

    "shoulder_lift_joint":
        -0.8678,

    "elbow_joint":
        -2.43,

    "wrist_1_joint":
        -0.6596,

    "wrist_2_joint":
        -0.0347,

    "wrist_3_joint":
         0.126,
}


ARM_ACTUATORS = {

    "shoulder_pan":
        "shoulder_pan_joint",

    "shoulder_lift":
        "shoulder_lift_joint",

    "elbow":
        "elbow_joint",

    "wrist_1":
        "wrist_1_joint",

    "wrist_2":
        "wrist_2_joint",

    "wrist_3":
        "wrist_3_joint",
}


HAND_QPOS = {

    "forearm:1--base:1":
        0.0,

    "palm_axis:1--forearm:1":
        0.0,

    "palm_assembly:1--palm_axis:1":
        0.0,

    "Thumb_axis--palm_assembly:1":
        1.57,

    "Thumb_Methacarpal--Thumb_axis":
        0.0,

    "Index_Proximal--palm_assembly:1":
        0.0,

    "Index_Middle--Index_Proximal":
        0.0,

    "Index_Distal--Index_Middle":
        0.0,

    "Middle_Proximal--palm_assembly:1":
        0.0,

    "Ring_Proximal--palm_assembly:1":
        0.0,
}


def object_id(
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
            f"Missing: {name}"
        )

    return idx


def set_joint(
    model,
    data,
    name,
    value,
):

    jid = object_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    adr = model.jnt_qposadr[
        jid
    ]

    data.qpos[
        adr
    ] = float(
        value
    )


def set_ctrl(
    model,
    data,
    name,
    value,
):

    aid = object_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )

    data.ctrl[
        aid
    ] = float(
        value
    )


def bottle_contacts(
    model,
    data,
):

    fixture = object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FIXTURE_NAME,
    )

    names = []


    for i in range(
        data.ncon
    ):

        c = data.contact[
            i
        ]


        if (
            c.geom1 != fixture
            and
            c.geom2 != fixture
        ):
            continue


        other = (
            c.geom2
            if c.geom1 == fixture
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
        set(
            names
        )
    )


def main():

    model = mujoco.MjModel.from_xml_path(
        str(
            MODEL_PATH
        )
    )

    data = mujoco.MjData(
        model
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    # ========================================================
    # Candidate A effective tendon
    # ========================================================

    tendon_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )


    model.tendon_stiffness[
        tendon_id
    ] = TENDON_STIFFNESS


    # ========================================================
    # Arm pose
    # ========================================================

    for name, value in (
        ARM_POSE.items()
    ):

        set_joint(
            model,
            data,
            name,
            value,
        )


    for actuator, joint in (
        ARM_ACTUATORS.items()
    ):

        set_ctrl(
            model,
            data,
            actuator,
            ARM_POSE[
                joint
            ],
        )


    # ========================================================
    # Hand pose
    # ========================================================

    for name, value in (
        HAND_QPOS.items()
    ):

        set_joint(
            model,
            data,
            name,
            value,
        )


    # Non-index actuator targets.
    optional_controls = {

        "pos_forearm":
            0.0,

        "pos_palm_axis":
            0.0,

        "pos_palm_assembly":
            0.0,

        "pos_thumb_axis":
            1.57,

        "pos_thumb_tendon":
            0.0,

        "pos_middle_tendon":
            0.0,

        "pos_ring_small_tendon":
            0.0,
    }


    for name, value in (
        optional_controls.items()
    ):

        aid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

        if aid >= 0:

            data.ctrl[
                aid
            ] = value


    # Index motor starts OPEN:
    #
    # 100 hardware ticks -> 0 spool displacement.
    set_joint(
        model,
        data,
        "index_motor_spool",
        0.0,
    )

    set_ctrl(
        model,
        data,
        "pos_index_tendon",
        0.0,
    )


    data.qvel[:] = 0.0


    mujoco.mj_forward(
        model,
        data,
    )


    print()
    print("=" * 100)
    print(
        "INDEX TABLETOP SYSID V1"
    )
    print("=" * 100)


    print()
    print(
        f"Tendon stiffness = "
        f"{TENDON_STIFFNESS:.1f} N/m"
    )


    print()
    print(
        "Bottle contacts at OPEN:"
    )


    contacts = bottle_contacts(
        model,
        data,
    )


    if contacts:

        for name in contacts:
            print(
                f"  {name}"
            )

    else:

        print(
            "  NONE"
        )


    print()
    print(
        "Index motor GUI slider:"
    )

    print(
        "  open 100 ticks  -> "
        "0.000000 m"
    )

    print(
        "  ~1020 ticks     -> "
        "0.004692 m"
    )

    print(
        "  1500 ticks      -> "
        "0.007140 m"
    )

    print(
        "  2000 ticks      -> "
        "0.009690 m"
    )

    print()
    print(
        "Use the 'pos_index_tendon' actuator slider."
    )

    print(
        "Move it SLOWLY first and observe which "
        "phalanx touches the bottle."
    )

    print()
    print(
        "Corrected proximal/middle collision capsules "
        "are visible with low transparency."
    )

    print()
    print("=" * 100)


    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        while viewer.is_running():

            mujoco.mj_step(
                model,
                data,
            )

            viewer.sync()

            time.sleep(
                model.opt.timestep
            )


if __name__ == "__main__":
    main()
