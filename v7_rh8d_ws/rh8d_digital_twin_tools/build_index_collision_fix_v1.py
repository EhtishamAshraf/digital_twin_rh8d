#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco


WS = Path(__file__).resolve().parents[1]

SRC = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_v1.xml"
)

DST = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_collisionfix_v1.xml"
)


# ============================================================
# Mechanical joint spacing
# ============================================================

LINK_LENGTH = 0.0258748


# ============================================================
# Collision geometry
#
# Capsule total length:
#
#     L = 2 * (half_length + radius)
#
# Choose L = mechanical joint spacing.
#
# Centre is halfway between consecutive joints.
# ============================================================

MIDPOINT_Z = -LINK_LENGTH / 2.0


PROX_RADIUS = 0.007780

PROX_HALF_LENGTH = (
    LINK_LENGTH / 2.0
    - PROX_RADIUS
)


MID_RADIUS = 0.007000

MID_HALF_LENGTH = (
    LINK_LENGTH / 2.0
    - MID_RADIUS
)


# x is aligned with the actual visual phalanx mesh.
#
# Existing visual meshes are around:
#
#   x = -0.00506 m
#   z = -0.01294 m
#
PHALANX_X = -0.0050625


UPDATES = {

    "collision_idx_prox": {
        "pos": (
            PHALANX_X,
            0.0,
            MIDPOINT_Z,
        ),

        "size": (
            PROX_RADIUS,
            PROX_HALF_LENGTH,
        ),
    },

    "collision_idx_mid": {
        "pos": (
            PHALANX_X,
            0.0,
            MIDPOINT_Z,
        ),

        "size": (
            MID_RADIUS,
            MID_HALF_LENGTH,
        ),
    },
}


def fmt3(values):

    return " ".join(
        f"{v:.9f}"
        for v in values
    )


def fmt2(values):

    return " ".join(
        f"{v:.9f}"
        for v in values
    )


def main():

    tree = ET.parse(
        SRC
    )

    root = tree.getroot()


    found = set()


    for geom in root.iter(
        "geom"
    ):

        name = geom.get(
            "name"
        )

        if name not in UPDATES:
            continue


        cfg = UPDATES[
            name
        ]


        old_pos = geom.get(
            "pos"
        )

        old_size = geom.get(
            "size"
        )


        geom.set(
            "pos",
            fmt3(
                cfg[
                    "pos"
                ]
            ),
        )


        geom.set(
            "size",
            fmt2(
                cfg[
                    "size"
                ]
            ),
        )


        # Make collision shape slightly visible in this
        # diagnostic model.
        if name == "collision_idx_prox":

            geom.set(
                "rgba",
                "1 0 0 0.20",
            )

        else:

            geom.set(
                "rgba",
                "1 0.5 0 0.20",
            )


        print()
        print(
            f"{name}"
        )

        print(
            f"  old pos  = {old_pos}"
        )

        print(
            f"  new pos  = "
            f"{geom.get('pos')}"
        )

        print(
            f"  old size = {old_size}"
        )

        print(
            f"  new size = "
            f"{geom.get('size')}"
        )


        found.add(
            name
        )


    missing = (
        set(
            UPDATES.keys()
        )
        - found
    )


    if missing:

        raise RuntimeError(
            "Missing geoms: "
            + ", ".join(
                missing
            )
        )


    tree.write(
        DST,
        encoding="unicode",
    )


    # Compile test.
    model = mujoco.MjModel.from_xml_path(
        str(
            DST
        )
    )


    print()
    print("=" * 100)
    print(
        "CORRECTED INDEX COLLISION MODEL"
    )
    print("=" * 100)


    print(
        f"Mechanical spacing = "
        f"{LINK_LENGTH*1000:.3f} mm"
    )

    print(
        f"Expected midpoint Z = "
        f"{MIDPOINT_Z*1000:.3f} mm"
    )


    print()
    print(
        "Proximal:"
    )

    print(
        f"  radius      = "
        f"{PROX_RADIUS*1000:.3f} mm"
    )

    print(
        f"  half-length = "
        f"{PROX_HALF_LENGTH*1000:.3f} mm"
    )

    print(
        f"  total length = "
        f"{2*(PROX_RADIUS+PROX_HALF_LENGTH)*1000:.3f} mm"
    )


    print()
    print(
        "Middle:"
    )

    print(
        f"  radius      = "
        f"{MID_RADIUS*1000:.3f} mm"
    )

    print(
        f"  half-length = "
        f"{MID_HALF_LENGTH*1000:.3f} mm"
    )

    print(
        f"  total length = "
        f"{2*(MID_RADIUS+MID_HALF_LENGTH)*1000:.3f} mm"
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
