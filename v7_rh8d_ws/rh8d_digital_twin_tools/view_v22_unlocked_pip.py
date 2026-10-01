#!/usr/bin/env python3

from pathlib import Path

import test_index_pip_passive_ablation_v22 as v22
import scan_and_view_index_contact_friction_v18 as viewer_tools


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()

MJCF_DIR = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

CONTACT_BAG = Path(
    "~/rh8d_system_id/index_contact_20260911_164310"
).expanduser().resolve()


TEMPLATE = (
    MJCF_DIR
    / "tmp_v22b_template.xml"
)

MODEL = (
    MJCF_DIR
    / "tmp_v22b_unlocked_pip.xml"
)


def main():

    # Freeze the existing V5 fixture registration.
    v22.ab.patch_replay_modules()


    # Build trusted constant-tendon template.
    v22.create_template(
        TEMPLATE
    )


    baseline_properties = (
        v22.pip_properties(
            TEMPLATE
        )
    )


    case = next(
        c
        for c in v22.CASES
        if (
            c["name"]
            ==
            "NO_PIP_SPRING_OR_FRICTION"
        )
    )


    v22.build_case(
        TEMPLATE,
        MODEL,
        case,
        baseline_properties,
    )


    compiled = (
        v22.pip_properties(
            MODEL
        )
    )


    print()
    print("=" * 100)
    print("V22B — VISUALIZING UNLOCKED PIP COUNTERFACTUAL")
    print("=" * 100)

    print()
    print("THIS IS NOT THE FINAL MODEL.")
    print("It is only the V22 causal diagnostic.")

    print()
    print(
        f"PIP stiffness = "
        f"{compiled['stiffness']:.6f} Nm/rad"
    )

    print(
        f"PIP frictionloss = "
        f"{compiled['frictionloss']:.6f} Nm"
    )

    print(
        f"PIP damping = "
        f"{compiled['damping']:.6f} Nms/rad"
    )

    print()
    print("Frozen:")
    print("  bottle position = V5 unchanged")
    print("  contact mu = 0.05")
    print("  contact stiffness = 1000 N/m")
    print("  constant tendon = 5 / 4 / 1 mm")
    print("  motor model unchanged")

    old_mu = (
        viewer_tools.base.CONTACT_MU
    )

    old_k = (
        viewer_tools.base.CONTACT_STIFFNESS
    )


    try:

        viewer_tools.base.CONTACT_MU = (
            0.05
        )

        viewer_tools.base.CONTACT_STIFFNESS = (
            1000.0
        )


        viewer_tools.visual_replay(
            MODEL,
            CONTACT_BAG,
        )


    finally:

        viewer_tools.base.CONTACT_MU = (
            old_mu
        )

        viewer_tools.base.CONTACT_STIFFNESS = (
            old_k
        )


    try:
        TEMPLATE.unlink()
    except FileNotFoundError:
        pass


    try:
        MODEL.unlink()
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
