#!/usr/bin/env python3

from pathlib import Path
import math

import fit_index_passive_contact_v35b2 as b2
import scan_index_zero_margin_damping_v35e2b as e2b


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = (
    WS /
    "src/ur5e_mujoco_description/mjcf"
)


SOURCE_OPTIONS = [
    MJCF / "ur5e_rh8d_index_cad_spatial_v35i1_best.xml",
    MJCF / "ur5e_rh8d_index_cad_spatial_v35h1_best.xml",
    MJCF / "ur5e_rh8d_index_cad_spatial_v35b2_selected_v35e2a_zero_margin.xml",
]


J1 = (
    MJCF /
    "ur5e_rh8d_index_cad_spatial_v35j1_inextensible.xml"
)


BAG = Path(
    "/home/ehtisham/rh8d_system_id/"
    "index_contact_20260911_164310"
)


def get_source():

    for p in SOURCE_OPTIONS:

        if p.exists():
            return p.resolve()

    raise RuntimeError(
        "No baseline source XML found."
    )


def show(
    title,
    result,
):

    pose = result[
        "pose"
    ]

    print()
    print(
        "=" * 100
    )

    print(
        title
    )

    print(
        "=" * 100
    )

    print(
        "pose = "
        f"{pose[0]:.3f} / "
        f"{pose[1]:.3f} / "
        f"{pose[2]:.3f} deg"
    )

    print(
        f"RMSE = "
        f"{result['rmse']:.3f} deg"
    )

    print(
        "contact transitions mid/tip = "
        f"{result.get('mid_transitions')} / "
        f"{result.get('tip_transitions')}"
    )

    print(
        "target gaps mid/tip = "
        f"{result.get('target_mid_gap_mm'):+.3f} / "
        f"{result.get('target_tip_gap_mm'):+.3f} mm"
    )

    print(
        "overlap longest = "
        f"{result.get('overlap_longest'):.3f} s"
    )


def main():

    source = get_source()

    if not BAG.exists():

        raise RuntimeError(
            f"Bag missing: {BAG}"
        )

    if not J1.exists():

        raise RuntimeError(
            "Run patch_index_inextensible_v35j1.py first."
        )


    print(
        f"BASELINE = {source}"
    )

    print(
        f"J1       = {J1}"
    )


    b2.v30.v24.ab.patch_replay_modules()


    contact_replay = (
        b2.v30.v24.base.contact_replay
    )


    # Keep the same contact environment used previously.
    old_cd = float(
        contact_replay.CONTACT_DAMPING
    )

    old_mu = (
        b2.v35.CONTACT_MU
    )

    old_k = (
        b2.v35.CONTACT_K
    )


    contact_replay.CONTACT_DAMPING = 20.0
    b2.v35.CONTACT_MU = 0.05
    b2.v35.CONTACT_K = 1000.0


    try:

        print()
        print(
            "Running baseline..."
        )

        baseline = e2b.run_one(
            20.0,
            source,
            BAG,
            contact_replay,
        )


        print()
        print(
            "Running J1 near-inextensible flexor..."
        )

        j1 = e2b.run_one(
            20.0,
            J1,
            BAG,
            contact_replay,
        )


    finally:

        contact_replay.CONTACT_DAMPING = (
            old_cd
        )

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_k
        )


    show(
        "A — CURRENT BASELINE",
        baseline,
    )

    show(
        "B — J1 NEAR-INEXTENSIBLE FLEXOR",
        j1,
    )


    print()
    print(
        "=" * 100
    )

    print(
        "OPENING J1 IN MUJOCO"
    )

    print(
        "=" * 100
    )

    print(
        "Watch specifically:"
    )

    print(
        "  1. first contact"
    )

    print(
        "  2. whether it rebounds"
    )

    print(
        "  3. whether DIP continues wrapping"
    )

    print(
        "  4. whether fingertip remains on bottle"
    )

    print()


    b2.v35.CONTACT_MU = 0.05
    b2.v35.CONTACT_K = 1000.0


    try:

        b2.visual_replay(
            J1,
            BAG,
        )

    finally:

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_k
        )


if __name__ == "__main__":

    main()
