#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


WS = Path(__file__).resolve().parents[1]

SRC = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_v1.xml"
)

DST = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_collisionfix_v2.xml"
)


# ============================================================
# Actual child-body vectors from MJCF audit
#
# parent joint -> next joint
# ============================================================

PROX_LINK = np.array(
    [
        6.24645e-05,
        0.0,
        -2.58748e-02,
    ],
    dtype=float,
)

MID_LINK = np.array(
    [
        6.24630e-05,
        0.0,
        -2.58748e-02,
    ],
    dtype=float,
)


# Approximate visual-mesh lateral offset.
PHALANX_OFFSET = np.array(
    [
        -0.0050625,
        0.0,
        0.0,
    ],
    dtype=float,
)


PROX_RADIUS = 0.007780
MID_RADIUS  = 0.007000


CONFIG = {

    "collision_idx_prox": {
        "link": PROX_LINK,
        "radius": PROX_RADIUS,
        "rgba": "1 0 0 0.25",
    },

    "collision_idx_mid": {
        "link": MID_LINK,
        "radius": MID_RADIUS,
        "rgba": "1 0.5 0 0.25",
    },
}


def unit(v):

    n = np.linalg.norm(v)

    if n < 1e-12:
        raise RuntimeError(
            "Zero-length link vector."
        )

    return v / n


def fmt(values):

    return " ".join(
        f"{float(v):.9f}"
        for v in values
    )


def capsule_fromto(
    link_vector,
    radius,
):

    L = float(
        np.linalg.norm(
            link_vector
        )
    )

    u = unit(
        link_vector
    )

    if 2.0 * radius >= L:

        raise RuntimeError(
            "Capsule radius is too large "
            "for mechanical link length."
        )


    # MuJoCo fromto defines the cylindrical centre-line
    # endpoints. The spherical caps extend one radius
    # beyond those endpoints.
    #
    # Therefore inset each endpoint by radius so the
    # OUTER capsule approximately spans joint -> joint.

    start = (
        PHALANX_OFFSET
        +
        radius * u
    )

    end = (
        PHALANX_OFFSET
        +
        (L - radius) * u
    )


    return (
        start,
        end,
        L,
    )


def main():

    tree = ET.parse(
        SRC
    )

    root = tree.getroot()


    found = set()


    print()
    print("=" * 100)
    print(
        "BUILD INDEX COLLISION FIX V2"
    )
    print("=" * 100)


    for geom in root.iter(
        "geom"
    ):

        name = geom.get(
            "name"
        )

        if name not in CONFIG:
            continue


        cfg = CONFIG[
            name
        ]


        old_fromto = geom.get(
            "fromto"
        )

        old_pos = geom.get(
            "pos"
        )

        old_size = geom.get(
            "size"
        )


        start, end, L = (
            capsule_fromto(
                cfg["link"],
                cfg["radius"],
            )
        )


        fromto = np.concatenate(
            [
                start,
                end,
            ]
        )


        # IMPORTANT:
        # with fromto, do not define pos.
        if "pos" in geom.attrib:

            del geom.attrib[
                "pos"
            ]


        geom.set(
            "fromto",
            fmt(
                fromto
            ),
        )


        # For a fromto capsule only the radius is required.
        geom.set(
            "size",
            f"{cfg['radius']:.9f}",
        )


        # Diagnostic transparency so we can visually compare
        # collision shape with the mesh.
        geom.set(
            "rgba",
            cfg["rgba"],
        )


        print()
        print(
            name
        )

        print(
            f"  old fromto = {old_fromto}"
        )

        print(
            f"  old pos    = {old_pos}"
        )

        print(
            f"  old size   = {old_size}"
        )

        print()
        print(
            f"  link length = "
            f"{L*1000:.3f} mm"
        )

        print(
            f"  radius      = "
            f"{cfg['radius']*1000:.3f} mm"
        )

        print(
            "  new start   = "
            + fmt(
                start
            )
        )

        print(
            "  new end     = "
            + fmt(
                end
            )
        )

        print(
            "  new fromto  = "
            + geom.get(
                "fromto"
            )
        )

        found.add(
            name
        )


    missing = (
        set(
            CONFIG.keys()
        )
        - found
    )


    if missing:

        raise RuntimeError(
            "Missing collision geoms: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


    tree.write(
        DST,
        encoding="unicode",
    )


    # ========================================================
    # Compile and inspect what MuJoCo actually produced
    # ========================================================

    model = mujoco.MjModel.from_xml_path(
        str(
            DST
        )
    )


    print()
    print("=" * 100)
    print(
        "MUJOCO COMPILED GEOMETRY"
    )
    print("=" * 100)


    for name in CONFIG:

        gid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )


        if gid < 0:

            raise RuntimeError(
                f"Compiled geom missing: {name}"
            )


        radius = (
            model.geom_size[
                gid,
                0
            ]
        )

        half_length = (
            model.geom_size[
                gid,
                1
            ]
        )


        total_length = (
            2.0
            * (
                radius
                +
                half_length
            )
        )


        print()
        print(
            name
        )

        print(
            "  compiled center =",
            model.geom_pos[
                gid
            ],
        )

        print(
            f"  radius          = "
            f"{radius*1000:.3f} mm"
        )

        print(
            f"  half-length     = "
            f"{half_length*1000:.3f} mm"
        )

        print(
            f"  outer length    = "
            f"{total_length*1000:.3f} mm"
        )


    print()
    print(
        "Compiled successfully:"
    )

    print(
        DST
    )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
