#!/usr/bin/env python3

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

SOURCE = (
    MJCF
    / "ur5e_rh8d_index_segmented_v36a2.xml"
)

OUTPUT = (
    MJCF
    / "ur5e_rh8d_index_segmented_v36a4.xml"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v36a4_rigid_equalities"
)


EQUALITIES = [
    "idx_v36_eq_seg1",
    "idx_v36_eq_seg2",
    "idx_v36_eq_seg3",
]


# timestep = 0.001 s
#
# With MuJoCo refsafe enabled, 0.002 is the
# stiffest meaningful positive-format time constant.
SOLREF = "0.002 1"

# Make the equality constraint very high impedance.
# This is numerical constraint enforcement, not
# a physical RH8D parameter.
SOLIMP = "0.9999 0.9999 0.001 0.5 2"


def find_named_equality(root, name):

    equality = root.find(
        "equality"
    )

    if equality is None:
        raise RuntimeError(
            "Missing <equality>"
        )

    for e in equality.findall(
        "tendon"
    ):

        if e.get("name") == name:
            return e

    raise RuntimeError(
        f"Missing equality tendon: {name}"
    )


def main():

    if not SOURCE.exists():
        raise RuntimeError(
            f"Missing source:\n{SOURCE}"
        )


    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()


    print()
    print("=" * 120)
    print("V36A4 — NEAR-RIGID SEGMENT EQUALITY CONSTRAINTS")
    print("=" * 120)

    print(
        f"SOURCE:\n{SOURCE}"
    )

    print(
        f"\nOUTPUT:\n{OUTPUT}"
    )

    print()
    print(
        "NO PHYSICAL PARAMETER CHANGES."
    )

    print(
        "Only numerical equality enforcement changes."
    )


    for name in EQUALITIES:

        eq = find_named_equality(
            root,
            name,
        )

        old_ref = eq.get(
            "solref"
        )

        old_imp = eq.get(
            "solimp"
        )

        eq.set(
            "solref",
            SOLREF,
        )

        eq.set(
            "solimp",
            SOLIMP,
        )


        print()
        print(
            name
        )

        print(
            f"  solref: "
            f"{old_ref} -> {SOLREF}"
        )

        print(
            f"  solimp: "
            f"{old_imp} -> {SOLIMP}"
        )


    try:
        ET.indent(
            tree,
            space="  ",
        )
    except AttributeError:
        pass


    tree.write(
        OUTPUT,
        encoding="utf-8",
        xml_declaration=True,
    )


    # ------------------------------------------------------------
    # Compile check
    # ------------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(OUTPUT)
    )

    data = mujoco.MjData(
        model
    )

    mujoco.mj_forward(
        model,
        data,
    )


    shutil.copy2(
        OUTPUT,
        OUTDIR / OUTPUT.name,
    )


    print()
    print("=" * 120)

    print(
        "V36A4 COMPILED SUCCESSFULLY"
    )

    print(
        "Guide friction remains ZERO / ZERO."
    )

    print(
        "Segment stiffnesses unchanged."
    )

    print(
        "Hidden armatures unchanged."
    )

    print(
        "Contact model unchanged."
    )

    print(
        f"\nArtifacts:\n{OUTDIR}"
    )

    print("=" * 120)


if __name__ == "__main__":
    main()
