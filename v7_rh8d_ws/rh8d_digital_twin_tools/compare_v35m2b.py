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

M2B = (
    MJCF
    / "ur5e_rh8d_index_cad_spatial_v35m2b_local_fixed_returns.xml"
)

BAG = Path(
    "/home/ehtisham/rh8d_system_id/"
    "index_contact_20260911_164310"
)


def show(title, r):

    q = r["pose"]

    print()
    print("=" * 112)
    print(title)
    print("=" * 112)

    print(
        f"pose = "
        f"{q[0]:.3f} / "
        f"{q[1]:.3f} / "
        f"{q[2]:.3f} deg"
    )

    print(
        f"RMSE = "
        f"{r['rmse']:.3f} deg"
    )

    print(
        "transitions mid/tip = "
        f"{r.get('mid_transitions')} / "
        f"{r.get('tip_transitions')}"
    )

    print(
        "gaps mid/tip = "
        f"{r.get('target_mid_gap_mm'):+.3f} / "
        f"{r.get('target_tip_gap_mm'):+.3f} mm"
    )

    print(
        "overlap = "
        f"{r.get('overlap_longest'):.3f} s"
    )

    print(
        "tip force median = "
        f"{r.get('positive_tip_median')}"
    )


def main():

    if not BASELINE.exists():

        raise RuntimeError(
            f"Missing baseline:\n{BASELINE}"
        )

    if not M2B.exists():

        raise RuntimeError(
            f"Missing M2B:\n{M2B}"
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
            "Running I1 baseline..."
        )

        a = e2b.run_one(
            20.0,
            BASELINE,
            BAG,
            contact_replay,
        )


        print()
        print(
            "Running M2B local elastic returns..."
        )

        b = e2b.run_one(
            20.0,
            M2B,
            BAG,
            contact_replay,
        )


    finally:

        contact_replay.CONTACT_DAMPING = old_cd
        b2.v35.CONTACT_MU = old_mu
        b2.v35.CONTACT_K = old_k


    show(
        "A — I1 BASELINE",
        a,
    )

    show(
        "B — M2B LOCAL ELASTIC RETURNS",
        b,
    )


    print()
    print("=" * 112)
    print("OPENING M2B VISUAL REPLAY")
    print("=" * 112)

    print(
        "Watch especially:"
    )

    print(
        "  1. Does DIP flex more naturally?"
    )

    print(
        "  2. Does middle phalanx reach the cylinder?"
    )

    print(
        "  3. Does fingertip reach the cylinder?"
    )

    print(
        "  4. Is touch/rebound reduced?"
    )


    b2.v35.CONTACT_MU = 0.05
    b2.v35.CONTACT_K = 1000.0


    try:

        b2.visual_replay(
            M2B,
            BAG,
        )

    finally:

        b2.v35.CONTACT_MU = old_mu
        b2.v35.CONTACT_K = old_k


if __name__ == "__main__":
    main()
