#!/usr/bin/env python3

from pathlib import Path
import mujoco


WS = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_tabletop_sysid_v1.xml"
)


JOINTS = [
    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",
    "index_motor_spool",
]


def oid(model, typ, name):

    idx = mujoco.mj_name2id(
        model,
        typ,
        name,
    )

    if idx < 0:
        raise RuntimeError(
            f"Missing object: {name}"
        )

    return idx


def main():

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )


    print()
    print("=" * 110)
    print("CURRENT COMPILED INDEX MECHANICS")
    print("=" * 110)

    print()
    print(
        "Model:",
        MODEL_PATH,
    )


    for name in JOINTS:

        jid = oid(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        qadr = model.jnt_qposadr[
            jid
        ]

        dofadr = model.jnt_dofadr[
            jid
        ]


        print()
        print("-" * 110)
        print(name)
        print("-" * 110)

        print(
            f"joint id        = {jid}"
        )

        print(
            f"qpos address    = {qadr}"
        )

        print(
            f"dof address     = {dofadr}"
        )

        print(
            f"type            = "
            f"{model.jnt_type[jid]}"
        )

        print(
            f"range           = "
            f"{model.jnt_range[jid]}"
        )

        print(
            f"axis            = "
            f"{model.jnt_axis[jid]}"
        )

        print(
            f"damping         = "
            f"{model.dof_damping[dofadr]:.9f}"
        )

        print(
            f"frictionloss    = "
            f"{model.dof_frictionloss[dofadr]:.9f}"
        )

        print(
            f"armature        = "
            f"{model.dof_armature[dofadr]:.9f}"
        )

        print(
            f"joint stiffness = "
            f"{model.jnt_stiffness[jid]:.9f}"
        )

        print(
            f"spring reference= "
            f"{model.qpos_spring[qadr]:.9f}"
        )


    tendon_id = oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        "idx_flex_spatial",
    )


    print()
    print("=" * 110)
    print("INDEX FIXED TENDON")
    print("=" * 110)

    print(
        f"stiffness    = "
        f"{model.tendon_stiffness[tendon_id]:.9f}"
    )

    print(
        f"damping      = "
        f"{model.tendon_damping[tendon_id]:.9f}"
    )

    print(
        f"frictionloss = "
        f"{model.tendon_frictionloss[tendon_id]:.9f}"
    )

    print(
        f"springlength = "
        f"{model.tendon_lengthspring[tendon_id]}"
    )


    actuator_id = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "pos_index_tendon",
    )


    print()
    print("=" * 110)
    print("INDEX MOTOR ACTUATOR")
    print("=" * 110)

    print(
        f"ctrlrange = "
        f"{model.actuator_ctrlrange[actuator_id]}"
    )

    print(
        f"forcerange = "
        f"{model.actuator_forcerange[actuator_id]}"
    )

    print(
        f"gainprm = "
        f"{model.actuator_gainprm[actuator_id]}"
    )

    print(
        f"biasprm = "
        f"{model.actuator_biasprm[actuator_id]}"
    )


    print()
    print("=" * 110)


if __name__ == "__main__":
    main()
