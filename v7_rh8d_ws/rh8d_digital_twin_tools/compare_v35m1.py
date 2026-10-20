#!/usr/bin/env python3

from pathlib import Path

import fit_index_passive_contact_v35b2 as b2
import scan_index_zero_margin_damping_v35e2b as e2b


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

BASELINE = (
    MJCF
    / "ur5e_rh8d_index_cad_spatial_v35i1_best.xml"
)

M1 = (
    MJCF
    / "ur5e_rh8d_index_cad_spatial_v35m1_passive_return.xml"
)

BAG = Path(
    "/home/ehtisham/rh8d_system_id/"
    "index_contact_20260911_164310"
)


def show(
    title,
    result,
):

    q = result["pose"]

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)

    print(
        f"pose = "
        f"{q[0]:.3f} / "
        f"{q[1]:.3f} / "
        f"{q[2]:.3f} deg"
    )

    print(
        f"RMSE = "
        f"{result['rmse']:.3f} deg"
    )

    print(
        "transitions mid/tip = "
        f"{result.get('mid_transitions')} / "
        f"{result.get('tip_transitions')}"
    )

    print(
        "gaps mid/tip = "
        f"{result.get('target_mid_gap_mm'):+.3f} / "
        f"{result.get('target_tip_gap_mm'):+.3f} mm"
    )

    print(
        "overlap = "
        f"{result.get('overlap_longest'):.3f} s"
    )


def main():

    if not BASELINE.exists():
        raise RuntimeError(
            f"Baseline missing:\n{BASELINE}"
        )

    if not M1.exists():
        raise RuntimeError(
            f"M1 missing:\n{M1}"
        )


    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
    )


    old_cd = float(
        contact_replay.CONTACT_DAMPING
    )

    old_mu = float(
        b2.v35.CONTACT_MU
    )

    old_k = float(
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

        a = e2b.run_one(
            20.0,
            BASELINE,
            BAG,
            contact_replay,
        )


        print()
        print(
            "Running M1 passive-return model..."
        )

        b = e2b.run_one(
            20.0,
            M1,
            BAG,
            contact_replay,
        )


    finally:

        contact_replay.CONTACT_DAMPING = old_cd
        b2.v35.CONTACT_MU = old_mu
        b2.v35.CONTACT_K = old_k


    show(
        "A — CURRENT I1",
        a,
    )

    show(
        "B — M1 PASSIVE RETURN TENDON",
        b,
    )


    print()
    print("=" * 110)
    print("OPENING M1")
    print("=" * 110)

    print(
        "Watch:"
    )

    print(
        "  1. first middle contact"
    )

    print(
        "  2. whether the finger bounces away"
    )

    print(
        "  3. whether DIP continues wrapping"
    )

    print(
        "  4. final middle + tip contact"
    )


    b2.v35.CONTACT_MU = 0.05
    b2.v35.CONTACT_K = 1000.0


    try:

        b2.visual_replay(
            M1,
            BAG,
        )

    finally:

        b2.v35.CONTACT_MU = old_mu
        b2.v35.CONTACT_K = old_k


if __name__ == "__main__":
    main()
