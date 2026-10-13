#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import fit_index_passive_contact_v35b2 as b2
import replay_index_zero_margin_v35e2a as e2a
import scan_index_zero_margin_damping_v35e2b as e2b


# ================================================================
# V35F2 — ZERO-MARGIN DYNAMIC CONTACT-STIFFNESS SCAN
#
# VARIABLE ONLY:
#     direct-format contact stiffness
#
# FROZEN:
#     margin = 0
#     damping = 20
#     mu = 0.05
#     solimp unchanged
#     CAD tendon
#     V35B2 passives
#     bottle registration
#     real H1 encoder replay
#
# NO angle control.
# NO contact-triggered mechanics.
#
# Purpose:
# Determine whether the very large contact reaction seen in F1
# is caused by excessive DYNAMIC direct stiffness.
# ================================================================


K_VALUES = [
    1.0,
    2.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    200.0,
    300.0,
    500.0,
    700.0,
    1000.0,
]

DAMPING = 20.0
MU = 0.05

HW_FORCE = 0.14563


def fmt(
    value,
    width=8,
    precision=3,
):

    if not np.isfinite(value):
        return f"{'nan':>{width}s}"

    return f"{value:{width}.{precision}f}"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    args = parser.parse_args()

    bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )

    if not bag.exists():

        raise RuntimeError(
            f"Missing bag:\n{bag}"
        )


    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
    )

    zero_margin_xml = (
        e2a.make_zero_margin_xml()
    )


    old_damping = float(
        contact_replay.CONTACT_DAMPING
    )

    old_mu = b2.v35.CONTACT_MU
    old_k = b2.v35.CONTACT_K


    contact_replay.CONTACT_DAMPING = (
        DAMPING
    )

    b2.v35.CONTACT_MU = (
        MU
    )


    rows = []


    try:

        print()
        print("=" * 160)
        print(
            "V35F2 — ZERO-MARGIN DYNAMIC CONTACT-STIFFNESS SCAN"
        )
        print("=" * 160)

        print()
        print(
            "FROZEN:"
        )

        print(
            "  total contact margin = 0.000 mm"
        )

        print(
            f"  direct damping = {DAMPING}"
        )

        print(
            f"  mu = {MU}"
        )

        print(
            f"  hardware FTS reference = "
            f"{HW_FORCE:.5f} N"
        )

        print()
        print(
            f"VARIABLE ONLY:"
        )

        print(
            f"  direct stiffness = "
            f"{K_VALUES}"
        )

        print()


        for index, k in enumerate(
            K_VALUES,
            start=1,
        ):

            # ----------------------------------------------
            # This is the ONLY changing parameter.
            # ----------------------------------------------

            b2.v35.CONTACT_K = (
                float(k)
            )

            print(
                f"[{index:02d}/"
                f"{len(K_VALUES):02d}] "
                f"k={k}"
            )


            row = e2b.run_one(
                DAMPING,
                zero_margin_xml,
                bag,
                contact_replay,
            )

            row["k"] = float(k)

            rows.append(
                row
            )


            print(
                f"   RMSE="
                f"{row['rmse']:.3f} deg, "
                f"Ftip+median="
                f"{row['positive_tip_median']:.4f} N, "
                f"Ftip-max="
                f"{row['max_tip_force']:.4f} N, "
                f"longest-overlap="
                f"{row['overlap_longest']:.3f} s, "
                f"final gaps="
                f"{row['final_mid_gap_mm']:+.3f}/"
                f"{row['final_tip_gap_mm']:+.3f} mm"
            )


    finally:

        contact_replay.CONTACT_DAMPING = (
            old_damping
        )

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_k
        )


    print()
    print("=" * 160)
    print(
        "V35F2 — COMPARISON TABLE"
    )
    print("=" * 160)

    print()

    print(
        "     k   RMSE   Ftip+med   FtipMax   Ferr    "
        "ovlpHold longest  midTr tipTr  "
        "E_mid  E_tip  E_F0.1   Tmid Ttip   "
        "finalMid finalTip"
    )

    print(
        "         [deg]     [N]       [N]      [N]       "
        "[s]      [s]                    "
        "[ticks] [ticks] [ticks]                 "
        "[mm]     [mm]"
    )

    print("-" * 160)


    for row in rows:

        print(
            f"{row['k']:6.1f} "
            f"{row['rmse']:6.3f} "
            f"{fmt(row['positive_tip_median'],10,4)} "
            f"{fmt(row['max_tip_force'],9,4)} "
            f"{fmt(row['force_error'],8,4)} "
            f"{row['overlap_total']:8.3f} "
            f"{row['overlap_longest']:7.3f} "
            f"{row['mid_transitions']:6d} "
            f"{row['tip_transitions']:5d} "
            f"{fmt(row['first_mid_encoder'],7,1)} "
            f"{fmt(row['first_tip_encoder'],7,1)} "
            f"{fmt(row['force_onset_encoder'],8,1)} "
            f"{str(row['target_mid']):>5s} "
            f"{str(row['target_tip']):>5s} "
            f"{row['final_mid_gap_mm']:+8.3f} "
            f"{row['final_tip_gap_mm']:+8.3f}"
        )


    # ============================================================
    # Show useful rankings without automatically declaring
    # a "winner".
    # ============================================================

    print()
    print("=" * 160)
    print(
        "V35F2 — RANKING BY CONTACT-FORCE ERROR"
    )
    print("=" * 160)

    print()

    for rank, row in enumerate(
        sorted(
            rows,
            key=lambda r:
                r["force_error"],
        )[:6],
        start=1,
    ):

        print(
            f"#{rank}: "
            f"k={row['k']:.1f}, "
            f"Ftip+median="
            f"{row['positive_tip_median']:.5f} N, "
            f"error="
            f"{row['force_error']:.5f} N, "
            f"RMSE="
            f"{row['rmse']:.3f} deg, "
            f"longest overlap="
            f"{row['overlap_longest']:.3f} s"
        )


    print()
    print("=" * 160)
    print(
        "V35F2 INTERPRETATION"
    )
    print("=" * 160)

    print()

    print(
        "If reducing k strongly reduces contact force/bounce "
        "and increases sustained overlap:"
    )

    print(
        "  -> dynamic contact stiffness was still a major "
        "part of the impact problem."
    )

    print()

    print(
        "If force remains far above hardware OR wrapping "
        "topology remains wrong over the entire scan:"
    )

    print(
        "  -> CLOSE contact-law tuning."
    )

    print(
        "  -> move to global passive/hysteretic "
        "MCP/PIP/DIP identification."
    )

    print()

    print(
        "No candidate should be selected from force alone."
    )

    print(
        "Pose, force, overlap, onset, final gaps and "
        "visual behavior must agree together."
    )

    print()
    print("=" * 160)
    print(
        "V35F2 COMPLETE"
    )
    print("=" * 160)


if __name__ == "__main__":
    main()
