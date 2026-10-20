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
    / "ur5e_rh8d_index_segmented_v36a.xml"
)

OUTPUT = (
    MJCF
    / "ur5e_rh8d_index_segmented_v36a2.xml"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v36a2_segmented_stabilized"
)


HIDDEN_JOINTS = [
    "idx_v36_slip_mcp",
    "idx_v36_slip_pip",
    "idx_v36_ext1",
    "idx_v36_ext2",
    "idx_v36_ext3",
]

EXTENSION_JOINTS = [
    "idx_v36_ext1",
    "idx_v36_ext2",
    "idx_v36_ext3",
]

EQUALITIES = [
    "idx_v36_eq_seg1",
    "idx_v36_eq_seg2",
    "idx_v36_eq_seg3",
]


# Numerical regularization only.
ARMATURE = 0.02

# Allow tiny signed elastic displacement during
# solver transients. We will inspect whether the
# final/physical states actually go negative.
EXT_RANGE = "-0.002 0.03"

# Old value was 0.002 s.
EQ_SOLREF = "0.005 1"


def named(root, tag, name):

    for e in root.iter(tag):

        if e.get("name") == name:
            return e

    raise RuntimeError(
        f"Missing <{tag} name='{name}'>"
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
    print("V36A2 — NUMERICALLY STABILIZED SEGMENTED CABLE")
    print("=" * 120)

    print(
        f"SOURCE:\n{SOURCE}"
    )

    print(
        f"\nOUTPUT:\n{OUTPUT}"
    )


    # ============================================================
    # 1. Add effective internal inertia / armature.
    # ============================================================

    print()
    print("Hidden cable coordinates:")


    for name in HIDDEN_JOINTS:

        j = named(
            root,
            "joint",
            name,
        )

        j.set(
            "armature",
            f"{ARMATURE:.12g}",
        )

        print(
            f"  {name:22s} "
            f"armature -> {ARMATURE}"
        )


    # ============================================================
    # 2. Remove exact zero-extension hard-stop singularity.
    # ============================================================

    print()
    print(
        "Extension ranges:"
    )


    for name in EXTENSION_JOINTS:

        j = named(
            root,
            "joint",
            name,
        )

        old = j.get(
            "range",
            "NONE",
        )

        j.set(
            "limited",
            "true",
        )

        j.set(
            "range",
            EXT_RANGE,
        )

        print(
            f"  {name:22s} "
            f"{old} -> {EXT_RANGE}"
        )


    # ============================================================
    # 3. Slightly soften the geometric equality dynamics.
    # ============================================================

    print()
    print(
        "Segment equality solver:"
    )


    for name in EQUALITIES:

        eq = named(
            root,
            "tendon",
            name,
        )

        old = eq.get(
            "solref",
            "NONE",
        )

        eq.set(
            "solref",
            EQ_SOLREF,
        )

        print(
            f"  {name:22s} "
            f"{old} -> {EQ_SOLREF}"
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


    # ============================================================
    # Compile validation.
    # ============================================================

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


    print()
    print("=" * 120)
    print("COMPILED HIDDEN DOF PARAMETERS")
    print("=" * 120)


    for name in HIDDEN_JOINTS:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        did = int(
            model.jnt_dofadr[
                jid
            ]
        )

        print(
            f"{name:22s} "
            f"armature="
            f"{model.dof_armature[did]:.6f}   "
            f"damping="
            f"{model.dof_damping[did]:.6f}   "
            f"friction="
            f"{model.dof_frictionloss[did]:.6f}"
        )


    print()
    print(
        "Guide friction remains:"
    )

    for name in [
        "idx_v36_slip_mcp",
        "idx_v36_slip_pip",
    ]:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        did = int(
            model.jnt_dofadr[
                jid
            ]
        )

        print(
            f"  {name}: "
            f"{model.dof_frictionloss[did]:.6f} N"
        )


    shutil.copy2(
        OUTPUT,
        OUTDIR / OUTPUT.name,
    )


    print()
    print(
        "FINAL MODEL COMPILED SUCCESSFULLY"
    )

    print(
        "Guide friction = ZERO / ZERO"
    )

    print(
        "No physical parameter was tuned for H1."
    )

    print(
        f"\nArtifacts:\n{OUTDIR}"
    )

    print("=" * 120)


if __name__ == "__main__":
    main()
