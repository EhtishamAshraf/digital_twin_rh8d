#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np

import test_index_middle_mesh_collision_v28 as v28
import scan_index_fixed_support_v30 as v30


# ============================================================
# V33 — REAL PHALANX MESH + FIXED SUPPORT
#
# V28 mesh test was run BEFORE V29 discovered large support
# recoil. Therefore its negative result was confounded.
#
# V33 retests the real CAD-derived middle phalanx collision
# under the corrected V30 boundary condition.
#
# NO artificial skin.
# NO pad.
# NO bottle motion.
# ============================================================


CASE = {
    "name": "V33_MESH_FIXED",
    "yield": True,
    "arms_mm": [4.0, 8.0, 1.5],
}


HW = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


MJCF_DIR = v30.v24.MJCF_DIR

MODEL_PATH = (
    MJCF_DIR
    / "ur5e_rh8d_index_mesh_fixed_support_v33.xml"
)


def run_contact(contact_bag):

    # Build actual middle-phalanx visual mesh as collision.
    # The old simplified capsule is disabled.
    v28.build_model(
        MODEL_PATH,
        True,
    )


    law = v30.v24.StatefulTransmission()
    law.install()


    # Critical correction discovered in V29.
    clamp = v30.FixedSupport()
    clamp.install()


    # V28 recorder understands collision_idx_mid_mesh.
    recorder = v28.Recorder(
        v28.MID_MESH_COLLISION
    )

    recorder.install()


    old_scale = (
        v30.v24.base.METERS_PER_TICK
    )

    old_tendon = (
        v30.v24.base.TENDON_STIFFNESS
    )

    old_mu = (
        v30.v24.base.CONTACT_MU
    )

    old_k = (
        v30.v24.base.CONTACT_STIFFNESS
    )


    try:

        v30.v24.base.METERS_PER_TICK = (
            v30.v24.SCALE
        )

        v30.v24.base.TENDON_STIFFNESS = (
            v30.v24.TENDON_K
        )

        v30.v24.base.CONTACT_MU = 0.05

        v30.v24.base.CONTACT_STIFFNESS = (
            v30.v24.CONTACT_K
        )


        law.configure(
            CASE,
            "CONTACT",
        )


        metrics, _ = (
            v30.v24.base.run_contact(
                MODEL_PATH,
                contact_bag,
            )
        )


    finally:

        recorder.restore()
        clamp.restore()
        law.restore()

        v30.v24.base.METERS_PER_TICK = (
            old_scale
        )

        v30.v24.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v30.v24.base.CONTACT_MU = (
            old_mu
        )

        v30.v24.base.CONTACT_STIFFNESS = (
            old_k
        )


    return (
        metrics,
        v28.analyse(
            recorder.rows
        ),
    )


def visual_replay(contact_bag):

    # Rebuild exact V33 model.
    v28.build_model(
        MODEL_PATH,
        True,
    )


    law = v30.v24.StatefulTransmission()
    law.install()


    clamp = v30.FixedSupport()
    clamp.install()


    old_scale = (
        v30.v24.viewer_tools.base.METERS_PER_TICK
    )

    old_tendon = (
        v30.v24.viewer_tools.base.TENDON_STIFFNESS
    )

    old_mu = (
        v30.v24.viewer_tools.base.CONTACT_MU
    )

    old_k = (
        v30.v24.viewer_tools.base.CONTACT_STIFFNESS
    )


    try:

        v30.v24.viewer_tools.base.METERS_PER_TICK = (
            v30.v24.SCALE
        )

        v30.v24.viewer_tools.base.TENDON_STIFFNESS = (
            v30.v24.TENDON_K
        )

        v30.v24.viewer_tools.base.CONTACT_MU = 0.05

        v30.v24.viewer_tools.base.CONTACT_STIFFNESS = (
            v30.v24.CONTACT_K
        )


        law.configure(
            CASE,
            "CONTACT",
        )


        v30.v24.viewer_tools.visual_replay(
            MODEL_PATH,
            contact_bag,
        )


    finally:

        clamp.restore()
        law.restore()

        v30.v24.viewer_tools.base.METERS_PER_TICK = (
            old_scale
        )

        v30.v24.viewer_tools.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v30.v24.viewer_tools.base.CONTACT_MU = (
            old_mu
        )

        v30.v24.viewer_tools.base.CONTACT_STIFFNESS = (
            old_k
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    parser.add_argument(
        "--no-viewer",
        action="store_true",
    )

    args = parser.parse_args()


    contact_bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )


    v30.v24.ab.patch_replay_modules()


    print()
    print("=" * 116)
    print(
        "V33 — ACTUAL PHALANX MESH + FIXED SUPPORT"
    )
    print("=" * 116)

    print()
    print(
        "Frozen:"
    )

    print(
        "  support = FIXED"
    )

    print(
        "  post-yield transmission = 4 / 8 / 1.5 mm"
    )

    print(
        "  bottle = V5"
    )

    print(
        "  mu = 0.05"
    )

    print(
        "  contact k = 1000 N/m"
    )

    print(
        "  NO artificial skin"
    )

    print()
    print(
        "Only change:"
    )

    print(
        "  simplified middle capsule "
        "-> actual existing phalanx CAD mesh"
    )


    metrics, retention = run_contact(
        contact_bag
    )


    pose = np.asarray(
        metrics[
            "pose"
        ],
        dtype=float,
    )


    rmse = float(
        np.sqrt(
            np.mean(
                (
                    pose
                    -
                    HW
                )
                ** 2
            )
        )
    )


    print()
    print("=" * 116)
    print(
        "V33 RESULT"
    )
    print("=" * 116)

    print()
    print(
        "Hardware:"
    )

    print(
        "  14.536 / 23.811 / 9.327 deg"
    )

    print(
        "Simulation:"
    )

    print(
        "  "
        f"{pose[0]:.3f} / "
        f"{pose[1]:.3f} / "
        f"{pose[2]:.3f} deg"
    )

    print(
        f"RMSE = "
        f"{rmse:.3f} deg"
    )


    print()
    print(
        "Contact retention:"
    )

    print(
        f"  middle total = "
        f"{retention['middle_duration_s']:.3f} s"
    )

    print(
        f"  tip total = "
        f"{retention['tip_duration_s']:.3f} s"
    )

    print(
        f"  middle+tip overlap = "
        f"{retention['overlap_s']:.3f} s"
    )

    print(
        f"  longest continuous overlap = "
        f"{retention['longest_overlap_s']:.3f} s"
    )

    print(
        f"  middle after first tip = "
        f"{retention['middle_after_tip_s']:.3f} s"
    )

    print(
        f"  effective mu = "
        f"{retention['effective_mu']:.3f}"
    )


    print()
    print("=" * 116)
    print(
        "V33 DECISION"
    )
    print("=" * 116)


    if (
        retention[
            "longest_overlap_s"
        ]
        >= 0.15
    ):

        print()
        print(
            "THE ACTUAL CAD PHALANX SURFACE + "
            "CORRECTED FIXED SUPPORT RECOVERS WRAPPING."
        )

        print()
        print(
            "Do not use V32/V32C artificial skin."
        )


    else:

        print()
        print(
            "THE ACTUAL CAD SURFACE STILL DOES NOT "
            "PRODUCE PERSISTENT WRAPPING."
        )

        print()
        print(
            "This result is now much stronger than V28 "
            "because support recoil has been removed."
        )

        print()
        print(
            "If the GUI also shows the physical middle "
            "phalanx peeling away, STOP simulator-only "
            "contact architecture tuning."
        )

        print()
        print(
            "The next useful data must come from the real "
            "hardware contact surface/compliance/contact path."
        )


    if not args.no_viewer:

        print()
        print("=" * 116)
        print(
            "V33 GUI — REAL PHALANX COLLISION"
        )
        print("=" * 116)

        print()
        print(
            "There are NO protruding skin patches."
        )

        print(
            "If contact happens, the actual visible "
            "phalanx surface should be at the cylinder."
        )

        visual_replay(
            contact_bag
        )


    print()
    print("=" * 116)
    print(
        "V33 COMPLETE"
    )
    print("=" * 116)


if __name__ == "__main__":
    main()
