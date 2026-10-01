#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


WS = Path(__file__).resolve().parents[1]

SRC = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_collisionfix_v2.xml"
)

DST = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_tabletop_sysid_v1.xml"
)


# ============================================================
# Initial tabletop bottle geometry
# ============================================================

BOTTLE_X = 0.475
BOTTLE_Y = -0.340

BOTTLE_RADIUS = 0.0346
BOTTLE_HALF_HEIGHT = 0.075

SUPPORT_TOP_Z = 0.724

BOTTLE_Z = (
    SUPPORT_TOP_Z
    + BOTTLE_HALF_HEIGHT
)


# Simple dedicated calibration support.
SUPPORT_HALF_X = 0.30
SUPPORT_HALF_Y = 0.30
SUPPORT_HALF_Z = 0.010

SUPPORT_CENTER_Z = (
    SUPPORT_TOP_Z
    - SUPPORT_HALF_Z
)


FIXTURE_NAME = "index_contact_fixture"
SUPPORT_NAME = "index_sysid_support"


def main():

    tree = ET.parse(
        SRC
    )

    root = tree.getroot()


    # ========================================================
    # Find bottle fixture
    # ========================================================

    fixture = None

    for geom in root.iter(
        "geom"
    ):

        if geom.get(
            "name"
        ) == FIXTURE_NAME:

            fixture = geom
            break


    if fixture is None:

        raise RuntimeError(
            f'Could not find geom "{FIXTURE_NAME}"'
        )


    print()
    print("=" * 100)
    print("BUILD TABLETOP INDEX SYSID V1")
    print("=" * 100)

    print()
    print("Old bottle attributes:")

    for key in [
        "type",
        "pos",
        "size",
        "zaxis",
        "quat",
        "fromto",
        "friction",
        "solref",
    ]:

        if key in fixture.attrib:

            print(
                f"  {key:10s} = "
                f"{fixture.attrib[key]}"
            )


    # ========================================================
    # Convert old floating cylinder to an upright bottle
    # ========================================================

    # Remove all alternate orientation / placement forms
    # which might conflict.
    for attr in [
        "quat",
        "euler",
        "axisangle",
        "xyaxes",
        "fromto",
    ]:

        if attr in fixture.attrib:

            del fixture.attrib[
                attr
            ]


    fixture.set(
        "type",
        "cylinder",
    )

    fixture.set(
        "pos",
        (
            f"{BOTTLE_X:.9f} "
            f"{BOTTLE_Y:.9f} "
            f"{BOTTLE_Z:.9f}"
        ),
    )

    fixture.set(
        "zaxis",
        "0 0 1",
    )

    fixture.set(
        "size",
        (
            f"{BOTTLE_RADIUS:.9f} "
            f"{BOTTLE_HALF_HEIGHT:.9f}"
        ),
    )


    # Candidate-A contact law kept frozen for now.
    fixture.set(
        "friction",
        "1.2 0.005 0.0001",
    )

    fixture.set(
        "solref",
        "-28000 -1.0",
    )

    fixture.set(
        "solimp",
        "0.95 0.95 0.001 0.5 2",
    )

    fixture.set(
        "contype",
        "1",
    )

    fixture.set(
        "conaffinity",
        "1",
    )

    fixture.set(
        "condim",
        "3",
    )

    fixture.set(
        "rgba",
        "0.1 0.65 0.25 0.65",
    )


    # ========================================================
    # Add dedicated support surface
    # ========================================================

    worldbody = root.find(
        "worldbody"
    )


    if worldbody is None:

        raise RuntimeError(
            "Could not find <worldbody>"
        )


    # Remove old copy if script is rerun.
    for geom in list(
        worldbody
    ):

        if (
            geom.tag == "geom"
            and
            geom.get("name")
            == SUPPORT_NAME
        ):

            worldbody.remove(
                geom
            )


    support = ET.Element(
        "geom"
    )


    support.set(
        "name",
        SUPPORT_NAME,
    )

    support.set(
        "type",
        "box",
    )

    support.set(
        "pos",
        (
            f"{BOTTLE_X:.9f} "
            f"{BOTTLE_Y:.9f} "
            f"{SUPPORT_CENTER_Z:.9f}"
        ),
    )

    support.set(
        "size",
        (
            f"{SUPPORT_HALF_X:.9f} "
            f"{SUPPORT_HALF_Y:.9f} "
            f"{SUPPORT_HALF_Z:.9f}"
        ),
    )

    support.set(
        "rgba",
        "0.45 0.45 0.45 0.35",
    )

    support.set(
        "friction",
        "1.0 0.005 0.0001",
    )

    support.set(
        "contype",
        "1",
    )

    support.set(
        "conaffinity",
        "1",
    )

    support.set(
        "condim",
        "3",
    )


    worldbody.append(
        support
    )


    tree.write(
        DST,
        encoding="unicode",
    )


    # ========================================================
    # Compile
    # ========================================================

    model = mujoco.MjModel.from_xml_path(
        str(
            DST
        )
    )


    bottle_gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FIXTURE_NAME,
    )


    support_gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        SUPPORT_NAME,
    )


    if bottle_gid < 0:
        raise RuntimeError(
            "Compiled bottle geom missing."
        )


    if support_gid < 0:
        raise RuntimeError(
            "Compiled support geom missing."
        )


    bottle_bottom = (
        model.geom_pos[
            bottle_gid,
            2
        ]
        -
        model.geom_size[
            bottle_gid,
            1
        ]
    )


    support_top = (
        model.geom_pos[
            support_gid,
            2
        ]
        +
        model.geom_size[
            support_gid,
            2
        ]
    )


    print()
    print("=" * 100)
    print("COMPILED TABLETOP MODEL")
    print("=" * 100)

    print()
    print(
        "Bottle center =",
        model.geom_pos[
            bottle_gid
        ],
    )

    print(
        "Bottle size   =",
        model.geom_size[
            bottle_gid
        ],
    )

    print(
        f"Bottle bottom = "
        f"{bottle_bottom:.9f} m"
    )


    print()
    print(
        "Support center =",
        model.geom_pos[
            support_gid
        ],
    )

    print(
        "Support size   =",
        model.geom_size[
            support_gid
        ],
    )

    print(
        f"Support top    = "
        f"{support_top:.9f} m"
    )


    print()
    print(
        f"Bottom - support = "
        f"{1000*(bottle_bottom-support_top):+.6f} mm"
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
