#!/usr/bin/env python3

from pathlib import Path
import mujoco
import numpy as np


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

XML = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
    / "ur5e_rh8d_index_cad_spatial_v35j1_inextensible.xml"
)


SPOOL = "index_motor_spool"

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

ACT = "pos_index_tendon"

FLEX = "idx_flex_spatial"
SPOOL_TENDON = "idx_spool_drive"


def oid(model, objtype, name):

    idx = mujoco.mj_name2id(
        model,
        objtype,
        name,
    )

    if idx < 0:
        raise RuntimeError(
            f"Missing {name}"
        )

    return idx


def qadr(model, name):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return int(
        model.jnt_qposadr[jid]
    )


def dadr(model, name):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return int(
        model.jnt_dofadr[jid]
    )


def print_state(
    model,
    data,
    label,
    L0,
):

    a = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACT,
    )

    tf = oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        FLEX,
    )

    ts = oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        SPOOL_TENDON,
    )


    qs = float(
        data.qpos[
            qadr(
                model,
                SPOOL,
            )
        ]
    )


    qm = float(
        data.qpos[
            qadr(
                model,
                MCP,
            )
        ]
    )

    qp = float(
        data.qpos[
            qadr(
                model,
                PIP,
            )
        ]
    )

    qd = float(
        data.qpos[
            qadr(
                model,
                DIP,
            )
        ]
    )


    lf = float(
        data.ten_length[
            tf
        ]
    )

    ls = float(
        data.ten_length[
            ts
        ]
    )


    # Our intended equality:
    #
    # L_flex = L0 - L_spool
    #
    residual = (
        lf
        -
        (
            L0
            -
            ls
        )
    )


    print()
    print(
        "-" * 100
    )

    print(label)

    print(
        "-" * 100
    )

    print(
        f"ctrl               = "
        f"{data.ctrl[a]:+.6f} m"
    )

    print(
        f"spool q            = "
        f"{qs:+.6f} m"
    )

    print(
        f"spool tendon       = "
        f"{ls:+.6f} m"
    )

    print(
        f"flex tendon        = "
        f"{lf:+.6f} m"
    )

    print(
        f"expected flex      = "
        f"{L0 - ls:+.6f} m"
    )

    print(
        f"equality residual  = "
        f"{residual * 1000:+.6f} mm"
    )

    print(
        "angles             = "
        f"{np.degrees(qm):+.4f} / "
        f"{np.degrees(qp):+.4f} / "
        f"{np.degrees(qd):+.4f} deg"
    )

    print(
        f"actuator force     = "
        f"{data.actuator_force[a]:+.6f}"
    )


def main():

    if not XML.exists():

        raise RuntimeError(
            f"Missing XML:\n{XML}"
        )


    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )

    data = mujoco.MjData(
        model
    )


    # --------------------------------------------
    # IDs
    # --------------------------------------------

    actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACT,
    )

    flex_tid = oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        FLEX,
    )

    spool_tid = oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        SPOOL_TENDON,
    )


    # Explicitly make sure no old flexor spring remains.
    model.tendon_stiffness[
        flex_tid
    ] = 0.0

    model.tendon_damping[
        flex_tid
    ] = 0.0


    mujoco.mj_forward(
        model,
        data,
    )


    L0 = float(
        data.ten_length[
            flex_tid
        ]
    )


    print()
    print(
        "=" * 100
    )

    print(
        "V35J1 TRANSMISSION DIAGNOSTIC"
    )

    print(
        "=" * 100
    )

    print(
        f"XML = {XML}"
    )

    print(
        f"L0  = {L0 * 1000:.6f} mm"
    )

    print(
        f"flex stiffness = "
        f"{model.tendon_stiffness[flex_tid]:.6f}"
    )

    print(
        f"flex damping   = "
        f"{model.tendon_damping[flex_tid]:.6f}"
    )


    # --------------------------------------------
    # Let initial state settle at zero command.
    # --------------------------------------------

    data.ctrl[
        actuator
    ] = 0.0


    for _ in range(
        1000
    ):

        mujoco.mj_step(
            model,
            data,
        )


    print_state(
        model,
        data,
        "ZERO COMMAND AFTER 1 SECOND",
        L0,
    )


    # --------------------------------------------
    # Test a few spool commands.
    # --------------------------------------------

    tests = [
        0.001,
        0.003,
        0.005,
        0.008,
    ]


    for target in tests:

        data.ctrl[
            actuator
        ] = target


        for _ in range(
            1500
        ):

            mujoco.mj_step(
                model,
                data,
            )


        print_state(
            model,
            data,
            f"COMMAND = {target * 1000:.1f} mm",
            L0,
        )


    print()
    print(
        "=" * 100
    )

    print(
        "DIAGNOSTIC COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
