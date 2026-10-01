#!/usr/bin/env python3

from pathlib import Path
import mujoco
import numpy as np


XML = Path(
    "src/ur5e_mujoco_description/mjcf/"
    "ur5e_rh8d_index_contact_cylinder_v1.xml"
)


LINKS = [
    (
        "Index_Proximal",
        "Index_Middle",
        "collision_idx_prox",
    ),
    (
        "Index_Middle",
        "Index_Distal",
        "collision_idx_mid",
    ),
]


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
            f"Missing {name}"
        )

    return i


model = mujoco.MjModel.from_xml_path(
    str(XML)
)


print()
print("=" * 100)
print("INDEX COLLISION / MECHANICAL-LINK ALIGNMENT")
print("=" * 100)


for body_name, child_name, geom_name in LINKS:

    bid = oid(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    child_bid = oid(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        child_name,
    )

    gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )


    child_pos = (
        model.body_pos[
            child_bid
        ].copy()
    )

    geom_pos = (
        model.geom_pos[
            gid
        ].copy()
    )

    radius = float(
        model.geom_size[
            gid,
            0
        ]
    )

    half_length = float(
        model.geom_size[
            gid,
            1
        ]
    )


    total_half_extent = (
        radius
        + half_length
    )


    capsule_z_min = (
        geom_pos[2]
        - total_half_extent
    )

    capsule_z_max = (
        geom_pos[2]
        + total_half_extent
    )


    link_length = float(
        np.linalg.norm(
            child_pos
        )
    )


    expected_midpoint = (
        0.5
        * child_pos
    )


    print()
    print("-" * 100)
    print(f"LINK: {body_name} -> {child_name}")
    print("-" * 100)

    print(
        "next joint / child-body position =",
        child_pos,
    )

    print(
        f"mechanical joint spacing = "
        f"{link_length*1000:.3f} mm"
    )

    print()
    print(
        f"collision geom = {geom_name}"
    )

    print(
        "geom local center =",
        geom_pos,
    )

    print(
        f"radius            = "
        f"{radius*1000:.3f} mm"
    )

    print(
        f"cylinder half-len = "
        f"{half_length*1000:.3f} mm"
    )

    print(
        f"total end-to-end  = "
        f"{2*total_half_extent*1000:.3f} mm"
    )

    print()
    print(
        f"capsule local-Z coverage = "
        f"{capsule_z_min*1000:+.3f} "
        f"to "
        f"{capsule_z_max*1000:+.3f} mm"
    )

    print(
        f"current joint Z          = "
        f"{0.0:+.3f} mm"
    )

    print(
        f"next joint Z             = "
        f"{child_pos[2]*1000:+.3f} mm"
    )

    print(
        f"mechanical-link midpoint = "
        f"{expected_midpoint[2]*1000:+.3f} mm"
    )

    center_error = (
        geom_pos[2]
        - expected_midpoint[2]
    )

    print()
    print(
        f"collision-center Z error "
        f"relative to joint midpoint = "
        f"{center_error*1000:+.3f} mm"
    )


print()
print("=" * 100)
