#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()


SOURCE = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_configdep_geom_v2.xml"
)


OUTPUT = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_configdep_geom_v3.xml"
)


BOTTLE_NAME = "index_contact_fixture"


# ============================================================
# Previously calibrated tabletop contact geometry.
#
# This was obtained from the earlier tabletop XY refinement,
# rather than chosen from the current visual inspection.
# ============================================================

BOTTLE_POS = np.array(
    [
        0.503422950,
       -0.372517762,
        0.799000000,
    ],
    dtype=float,
)


def main():

    if not SOURCE.exists():

        raise RuntimeError(
            f"Missing V2 source:\n{SOURCE}"
        )


    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()


    bottle = None

    for geom in root.iter("geom"):

        if (
            geom.attrib.get("name")
            == BOTTLE_NAME
        ):

            bottle = geom
            break


    if bottle is None:

        raise RuntimeError(
            f"Could not find geom: "
            f"{BOTTLE_NAME}"
        )


    old_pos = np.array(
        [
            float(x)
            for x in bottle.attrib[
                "pos"
            ].split()
        ],
        dtype=float,
    )


    bottle.set(
        "pos",
        " ".join(
            f"{x:.9f}"
            for x in BOTTLE_POS
        ),
    )


    tree.write(
        OUTPUT,
        encoding="unicode",
        xml_declaration=True,
    )


    # Compile immediately so we know V3 is valid.
    model = mujoco.MjModel.from_xml_path(
        str(
            OUTPUT
        )
    )


    gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        BOTTLE_NAME,
    )


    if gid < 0:

        raise RuntimeError(
            "Bottle disappeared after build."
        )


    compiled_pos = (
        model.geom_pos[
            gid
        ].copy()
    )


    delta = (
        BOTTLE_POS
        - old_pos
    )


    print()
    print("=" * 90)
    print(
        "RH8D CONFIGURATION-DEPENDENT "
        "FIXTURE V3"
    )
    print("=" * 90)


    print()
    print(
        "Old seed bottle position:"
    )

    print(
        f"  {old_pos}"
    )


    print()
    print(
        "Restored calibrated bottle position:"
    )

    print(
        f"  {BOTTLE_POS}"
    )


    print()
    print(
        "XY correction:"
    )

    print(
        f"  dx = "
        f"{delta[0]*1000.0:+.3f} mm"
    )

    print(
        f"  dy = "
        f"{delta[1]*1000.0:+.3f} mm"
    )


    print()
    print(
        "Compiled MuJoCo position:"
    )

    print(
        f"  {compiled_pos}"
    )


    print()
    print(
        "Output model:"
    )

    print(
        OUTPUT
    )


    print()
    print(
        "MuJoCo compile: PASS"
    )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
