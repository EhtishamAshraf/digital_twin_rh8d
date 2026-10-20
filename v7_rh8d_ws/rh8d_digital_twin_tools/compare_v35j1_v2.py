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

BAG = Path(
    "/home/ehtisham/rh8d_system_id/"
    "index_contact_20260911_164310"
)


BASELINE_CANDIDATES = [
    MJCF / "ur5e_rh8d_index_cad_spatial_v35i1_best.xml",
    MJCF / "ur5e_rh8d_index_cad_spatial_v35h1_best.xml",
    MJCF / "ur5e_rh8d_index_cad_spatial_v35b2_selected_v35e2a_zero_margin.xml",
]


J1 = (
    MJCF
    / "ur5e_rh8d_index_cad_spatial_v35j1_inextensible.xml"
)


def find_baseline():

    for p in BASELINE_CANDIDATES:

        if p.exists():
            return p.resolve()

    raise RuntimeError(
        "No baseline XML found."
    )


def show(title, result):

    pose = result["pose"]

    print()
    print("=" * 105)
    print(title)
    print("=" * 105)

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


def run_model(
    xml,
    tendon_k,
    contact_replay,
):

    # ---------------------------------------------------------
    # CRITICAL:
    #
    # b2.run_candidate() does NOT trust the XML tendon
    # stiffness. It writes v35.TENDON_K into the lower replay
    # stack.
    #
    # Therefore explicitly select the correct architecture.
    # ---------------------------------------------------------

    old_v35_k = float(
        b2.v35.TENDON_K
    )

    old_base_k = float(
        b2.v30.v24.base.TENDON_STIFFNESS
    )


    b2.v35.TENDON_K = float(
        tendon_k
    )

    b2.v30.v24.base.TENDON_STIFFNESS = float(
        tendon_k
    )


    try:

        return e2b.run_one(
            20.0,
            xml,
            BAG,
            contact_replay,
        )

    finally:

        b2.v35.TENDON_K = (
            old_v35_k
        )

        b2.v30.v24.base.TENDON_STIFFNESS = (
            old_base_k
        )


def visual_j1(
    xml,
):

    # Same issue exists in b2.visual_replay():
    # it also copies v35.TENDON_K into viewer replay.

    old_v35_k = float(
        b2.v35.TENDON_K
    )

    old_viewer_k = float(
        b2.v30.v24.viewer_tools.base.TENDON_STIFFNESS
    )


    b2.v35.TENDON_K = 0.0

    b2.v30.v24.viewer_tools.base.TENDON_STIFFNESS = 0.0


    try:

        b2.visual_replay(
            xml,
            BAG,
        )

    finally:

        b2.v35.TENDON_K = (
            old_v35_k
        )

        b2.v30.v24.viewer_tools.base.TENDON_STIFFNESS = (
            old_viewer_k
        )


def main():

    baseline = find_baseline()


    if not BAG.exists():

        raise RuntimeError(
            f"Bag not found: {BAG}"
        )


    if not J1.exists():

        raise RuntimeError(
            f"J1 XML not found: {J1}"
        )


    print()
    print("=" * 105)
    print("V35J1 V2 — CORRECT ARCHITECTURE TEST")
    print("=" * 105)

    print(
        f"BASELINE:\n{baseline}"
    )

    print(
        f"\nJ1:\n{J1}"
    )

    print()
    print(
        "Baseline runtime tendon stiffness = 2100 N/m"
    )

    print(
        "J1 runtime tendon stiffness       = 0 N/m"
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

    old_contact_k = float(
        b2.v35.CONTACT_K
    )


    contact_replay.CONTACT_DAMPING = 20.0

    b2.v35.CONTACT_MU = 0.05

    b2.v35.CONTACT_K = 1000.0


    try:

        print()
        print(
            "Running ORIGINAL elastic-flexor baseline..."
        )

        result_a = run_model(
            baseline,
            2100.0,
            contact_replay,
        )


        print()
        print(
            "Running J1 with ACTUAL zero flexor stiffness..."
        )

        result_b = run_model(
            J1,
            0.0,
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
            old_contact_k
        )


    show(
        "A — ORIGINAL ELASTIC FLEXOR",
        result_a,
    )

    show(
        "B — J1 INEXTENSIBLE FLEXOR",
        result_b,
    )


    print()
    print("=" * 105)
    print("OPENING CORRECT J1 IN MUJOCO")
    print("=" * 105)

    print(
        "Look only at these four things:"
    )

    print(
        "  1. Does the finger now move?"
    )

    print(
        "  2. Does first contact cause rebound?"
    )

    print(
        "  3. Does DIP continue flexing after middle contact?"
    )

    print(
        "  4. Does tip remain against the bottle?"
    )

    print()


    b2.v35.CONTACT_MU = 0.05

    b2.v35.CONTACT_K = 1000.0


    try:

        visual_j1(
            J1
        )

    finally:

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_contact_k
        )


if __name__ == "__main__":
    main()
