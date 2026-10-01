#!/usr/bin/env python3

from pathlib import Path
import math
import mujoco
import numpy as np


WS = Path.home() / (
    "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
)

MODEL_PATH = (
    WS /
    "src/ur5e_mujoco_description/mjcf/"
    "ur5e_rh8d_index_identified.xml"
)

CONTACT_ENCODER = 1020.0
OPEN_ENCODER = 100.0
METERS_PER_TICK = 6.0e-6


def name(model, objtype, idx):
    n = mujoco.mj_id2name(
        model,
        objtype,
        idx,
    )
    return n if n else ""


def joint_qadr(model, joint_name):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if jid < 0:
        raise RuntimeError(
            f"Joint not found: {joint_name}"
        )

    return model.jnt_qposadr[jid]


def main():

    print()
    print("=" * 78)
    print("RH8D INDEX CONTACT-GEOMETRY AUDIT")
    print("=" * 78)

    print()
    print("Loading:")
    print(MODEL_PATH)

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    model.opt.gravity[:] = 0.0

    spool_jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "index_motor_spool",
    )

    spool_qadr = model.jnt_qposadr[
        spool_jid
    ]

    spool_dof = model.jnt_dofadr[
        spool_jid
    ]

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "pos_index_tendon",
    )

    mcp_q = joint_qadr(
        model,
        "Index_Proximal--palm_assembly:1",
    )

    pip_q = joint_qadr(
        model,
        "Index_Middle--Index_Proximal",
    )

    dip_q = joint_qadr(
        model,
        "Index_Distal--Index_Middle",
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    # --------------------------------------------------------
    # Settle fully open first
    # --------------------------------------------------------

    for _ in range(
        int(2.0 / model.opt.timestep)
    ):

        data.qpos[spool_qadr] = 0.0

        data.qvel[spool_dof] = 0.0

        data.ctrl[actuator_id] = 0.0

        mujoco.mj_step(
            model,
            data,
        )


    # --------------------------------------------------------
    # Move to measured first-contact motor state
    #
    # 1020 ticks corresponds to:
    #
    # (1020 - 100) * 6 um/tick
    # --------------------------------------------------------

    spool_target = (
        CONTACT_ENCODER
        - OPEN_ENCODER
    ) * METERS_PER_TICK


    for _ in range(
        int(3.0 / model.opt.timestep)
    ):

        data.qpos[spool_qadr] = (
            spool_target
        )

        data.qvel[spool_dof] = 0.0

        data.ctrl[actuator_id] = (
            spool_target
        )

        mujoco.mj_step(
            model,
            data,
        )


    print()
    print(
        f"Contact encoder = "
        f"{CONTACT_ENCODER:.1f} ticks"
    )

    print(
        f"Effective spool = "
        f"{spool_target*1000:.3f} mm"
    )

    print()
    print("Finger configuration:")

    print(
        f"  MCP = "
        f"{math.degrees(data.qpos[mcp_q]):.3f} deg"
    )

    print(
        f"  PIP = "
        f"{math.degrees(data.qpos[pip_q]):.3f} deg"
    )

    print(
        f"  DIP = "
        f"{math.degrees(data.qpos[dip_q]):.3f} deg"
    )


    # --------------------------------------------------------
    # INDEX BODIES
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("INDEX BODIES")
    print("=" * 78)

    for bid in range(
        model.nbody
    ):

        n = name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            bid,
        )

        if "Index" in n or "index" in n:

            p = data.xpos[bid]

            print(
                f"{bid:3d} "
                f"{n:45s} "
                f"x={p[0]: .6f} "
                f"y={p[1]: .6f} "
                f"z={p[2]: .6f}"
            )


    # --------------------------------------------------------
    # INDEX GEOMS
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("INDEX GEOMS")
    print("=" * 78)

    for gid in range(
        model.ngeom
    ):

        gname = name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            gid,
        )

        body_id = model.geom_bodyid[
            gid
        ]

        bname = name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        if (
            "Index" in gname
            or "index" in gname
            or "Index" in bname
            or "index" in bname
        ):

            p = data.geom_xpos[
                gid
            ]

            size = model.geom_size[
                gid
            ]

            print()
            print(
                f"geom id    : {gid}"
            )

            print(
                f"geom name  : {gname}"
            )

            print(
                f"body       : {bname}"
            )

            print(
                "world pos   : "
                f"{p[0]: .6f} "
                f"{p[1]: .6f} "
                f"{p[2]: .6f}"
            )

            print(
                "geom size   : "
                + " ".join(
                    f"{x:.6f}"
                    for x in size
                )
            )

            print(
                f"geom type   : "
                f"{int(model.geom_type[gid])}"
            )


    # --------------------------------------------------------
    # INDEX SITES
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("INDEX SITES")
    print("=" * 78)

    for sid in range(
        model.nsite
    ):

        sname = name(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            sid,
        )

        body_id = model.site_bodyid[
            sid
        ]

        bname = name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        if (
            "Index" in sname
            or "index" in sname
            or "Index" in bname
            or "index" in bname
        ):

            p = data.site_xpos[
                sid
            ]

            print(
                f"{sid:3d} "
                f"{sname:32s} "
                f"body={bname:30s} "
                f"x={p[0]: .6f} "
                f"y={p[1]: .6f} "
                f"z={p[2]: .6f}"
            )


    print()
    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
