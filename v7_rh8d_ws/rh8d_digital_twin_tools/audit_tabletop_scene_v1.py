#!/usr/bin/env python3

from pathlib import Path

import mujoco
import numpy as np


WS = Path(__file__).resolve().parents[1]

SCENE = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_scene.xml"
)


ARM_POSE = {
    "shoulder_pan_joint":   1.4927,
    "shoulder_lift_joint": -0.8678,
    "elbow_joint":         -2.43,
    "wrist_1_joint":       -0.6596,
    "wrist_2_joint":       -0.0347,
    "wrist_3_joint":        0.126,
}


HAND_POSE = {
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

    "Middle_Proximal--palm_assembly:1":
        0.0,

    "Ring_Proximal--palm_assembly:1":
        0.0,
}


def oid(
    model,
    typ,
    name,
):

    i = mujoco.mj_name2id(
        model,
        typ,
        name,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing MuJoCo object: {name}"
        )

    return i


def set_qpos(
    model,
    data,
    name,
    value,
):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    qadr = model.jnt_qposadr[
        jid
    ]

    data.qpos[
        qadr
    ] = float(
        value
    )


def geom_world_aabb(
    model,
    data,
    gid,
):

    center = data.geom_xpos[
        gid
    ].copy()

    R = data.geom_xmat[
        gid
    ].reshape(
        3,
        3,
    )


    geom_type = model.geom_type[
        gid
    ]


    # BOX
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:

        half_local = model.geom_size[
            gid
        ].copy()

        half_world = (
            np.abs(R)
            @ half_local
        )


        return (
            center - half_world,
            center + half_world,
        )


    raise RuntimeError(
        "AABB helper currently expects box geom."
    )


def print_joint_anchor(
    model,
    data,
    name,
):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    print(
        f"{name:42s} = "
        f"{data.xanchor[jid]}"
    )


def main():

    print()
    print("=" * 110)
    print(
        "RH8D TABLETOP SCENE AUDIT"
    )
    print("=" * 110)

    print()
    print(
        f"Loading:\n{SCENE}"
    )


    model = mujoco.MjModel.from_xml_path(
        str(
            SCENE
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
    # Apply the known old tabletop startup pose.
    # ========================================================

    for name, value in ARM_POSE.items():

        set_qpos(
            model,
            data,
            name,
            value,
        )


    for name, value in HAND_POSE.items():

        set_qpos(
            model,
            data,
            name,
            value,
        )


    data.qvel[:] = 0.0


    mujoco.mj_forward(
        model,
        data,
    )


    # ========================================================
    # Table body
    # ========================================================

    table_bid = oid(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "table",
    )


    print()
    print("=" * 110)
    print(
        "TABLE BODY"
    )
    print("=" * 110)


    print(
        "XML/local body position =",
        model.body_pos[
            table_bid
        ],
    )


    print(
        "compiled world position =",
        data.xpos[
            table_bid
        ],
    )


    # ========================================================
    # Table top collision
    # ========================================================

    top_gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "table_top_collision",
    )


    top_center = (
        data.geom_xpos[
            top_gid
        ].copy()
    )


    top_min, top_max = (
        geom_world_aabb(
            model,
            data,
            top_gid,
        )
    )


    print()
    print("=" * 110)
    print(
        "TABLE TOP COLLISION"
    )
    print("=" * 110)


    print(
        "local geom pos =",
        model.geom_pos[
            top_gid
        ],
    )


    print(
        "box half-size  =",
        model.geom_size[
            top_gid
        ],
    )


    print(
        "world center   =",
        top_center,
    )


    print(
        "world AABB min =",
        top_min,
    )


    print(
        "world AABB max =",
        top_max,
    )


    print()
    print(
        f"TABLE TOP UPPER Z = "
        f"{top_max[2]:.9f} m"
    )


    # ========================================================
    # Old scene cylinder
    # ========================================================

    cyl_bid = oid(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "cylinder",
    )


    cyl_gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "cylinder",
    )


    cyl_center = (
        data.geom_xpos[
            cyl_gid
        ].copy()
    )


    cyl_R = (
        data.geom_xmat[
            cyl_gid
        ].reshape(
            3,
            3,
        )
    )


    cyl_axis = (
        cyl_R[:, 2]
    )


    radius = float(
        model.geom_size[
            cyl_gid,
            0
        ]
    )


    half_height = float(
        model.geom_size[
            cyl_gid,
            1
        ]
    )


    cyl_bottom = (
        cyl_center
        -
        half_height
        * cyl_axis
    )


    cyl_top = (
        cyl_center
        +
        half_height
        * cyl_axis
    )


    print()
    print("=" * 110)
    print(
        "OLD SCENE CYLINDER"
    )
    print("=" * 110)


    print(
        "body local pos =",
        model.body_pos[
            cyl_bid
        ],
    )


    print(
        "world center   =",
        cyl_center,
    )


    print(
        "axis           =",
        cyl_axis,
    )


    print(
        f"radius         = "
        f"{radius:.6f} m "
        f"({radius*1000:.3f} mm)"
    )


    print(
        f"half-height    = "
        f"{half_height:.6f} m"
    )


    print(
        "bottom point   =",
        cyl_bottom,
    )


    print(
        "top point      =",
        cyl_top,
    )


    print()
    print(
        f"Cylinder bottom Z - table top Z = "
        f"{1000*(cyl_bottom[2]-top_max[2]):+.3f} mm"
    )


    # ========================================================
    # Index mechanical joint anchors at startup pose
    # ========================================================

    print()
    print("=" * 110)
    print(
        "INDEX JOINT CENTRES AT TABLETOP START POSE"
    )
    print("=" * 110)


    print_joint_anchor(
        model,
        data,
        "Index_Proximal--palm_assembly:1",
    )


    print_joint_anchor(
        model,
        data,
        "Index_Middle--Index_Proximal",
    )


    print_joint_anchor(
        model,
        data,
        "Index_Distal--Index_Middle",
    )


    # ========================================================
    # Current index collision geoms in the ORIGINAL scene
    # ========================================================

    print()
    print("=" * 110)
    print(
        "ORIGINAL INDEX COLLISION GEOMS IN TABLETOP POSE"
    )
    print("=" * 110)


    for name in [
        "collision_idx_prox",
        "collision_idx_mid",
        "collision_idx_tip",
    ]:

        gid = oid(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )


        print()
        print(
            name
        )

        print(
            "  world center =",
            data.geom_xpos[
                gid
            ],
        )

        print(
            "  radius       =",
            model.geom_size[
                gid,
                0
            ],
        )

        print(
            "  half length  =",
            model.geom_size[
                gid,
                1
            ],
        )


    # ========================================================
    # Useful relative geometry
    # ========================================================

    mcp_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "Index_Proximal--palm_assembly:1",
    )


    mcp = (
        data.xanchor[
            mcp_jid
        ].copy()
    )


    print()
    print("=" * 110)
    print(
        "BOTTLE RELATIVE TO INDEX MCP"
    )
    print("=" * 110)


    delta = (
        cyl_center
        - mcp
    )


    print(
        "MCP -> cylinder center vector =",
        delta,
    )


    print(
        f"3D center distance = "
        f"{1000*np.linalg.norm(delta):.3f} mm"
    )


    print()
    print("=" * 110)


if __name__ == "__main__":
    main()
