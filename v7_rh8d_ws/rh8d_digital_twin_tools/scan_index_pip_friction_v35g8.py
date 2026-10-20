#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import scan_index_contact_friction_v35g7 as g7


# ================================================================
# V35G8 — PIP BREAKAWAY-FRICTION / LOAD-REDISTRIBUTION SCAN
#
# Hardware target:
#
# MCP moves
# -> middle touches
# -> PIP continues wrapping
# -> middle becomes constrained
# -> PIP stops
# -> DIP rotates
# -> distal becomes constrained
# -> DIP stops
#
# G7 proved that increasing bottle/finger contact friction does NOT
# generate this mechanism.
#
# G8 changes ONLY PIP joint frictionloss.
# ================================================================


SCALE = 5.70e-6
CONTACT_MU = 0.05

PIP_FRICTIONS = [
    0.00125,
    0.00200,
    0.00300,
    0.00400,
    0.00600,
    0.00800,
]

PIP_JOINT = "Index_Middle--Index_Proximal"


def tag(value):
    return (
        f"{value:.5f}"
        .replace(".", "p")
    )


def make_candidate(
    source_xml: Path,
    out_dir: Path,
    friction: float,
) -> Path:

    tree = ET.parse(source_xml)
    root = tree.getroot()

    joint = None

    for element in root.iter("joint"):
        if element.get("name") == PIP_JOINT:
            joint = element
            break

    if joint is None:
        raise RuntimeError(
            f"Could not find PIP joint: {PIP_JOINT}"
        )

    old = joint.get("frictionloss")

    joint.set(
        "frictionloss",
        f"{friction:.10g}",
    )

    path = (
        out_dir
        / f"candidate_pip_friction_{tag(friction)}.xml"
    )

    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(
        f"    PIP frictionloss: "
        f"{old} -> {friction:.6f} Nm"
    )

    return path


def mechanics_score(r):
    """
    Mechanics dominates the score.
    RMSE is only a late penalty.
    """

    score = 0.0

    if not r["sequence_ok"]:
        score += 1000.0

    if not r["mid_retained_to_tip"]:
        score += 300.0

    # We want middle contact to survive toward distal contact.
    score += (
        150.0
        *
        max(
            0.0,
            1.0 - r["mid_duty_mid_to_tip"],
        )
    )

    # We specifically want DIP redistribution.
    # A tiny DIP/PIP ratio is bad.
    ratio = r["dip_over_pip_mid_to_tip"]

    if np.isfinite(ratio):
        score += (
            100.0
            *
            max(
                0.0,
                1.0 - ratio,
            )
        )
    else:
        score += 200.0

    # Post-tip DIP should not remain almost frozen.
    post_ratio = r["dip_over_pip_post_tip"]

    if np.isfinite(post_ratio):
        score += (
            70.0
            *
            max(
                0.0,
                1.0 - post_ratio,
            )
        )
    else:
        score += 100.0

    # Penalize repeated touch-bounce cycles.
    score += (
        25.0
        *
        max(
            0,
            r["e2b_mid_transitions_hold"] - 1,
        )
    )

    score += (
        25.0
        *
        max(
            0,
            r["e2b_tip_transitions_hold"] - 1,
        )
    )

    # Target penetration penalty.
    score += (
        100.0
        *
        max(
            0.0,
            -r["target_mid_gap_mm"],
        )
    )

    score += (
        100.0
        *
        max(
            0.0,
            -r["target_tip_gap_mm"],
        )
    )

    # Strong force spike penalty.
    fmax = r["e2b_hold_tip_force_max_N"]

    if np.isfinite(fmax):
        score += (
            10.0
            *
            max(
                0.0,
                fmax - 2.0,
            )
        )

    # RMSE deliberately weak.
    score += (
        2.0
        *
        r["rmse_deg"]
    )

    return float(score)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            / "rh8d_results"
            / "v35g8_pip_friction"
        ),
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
    )

    args = parser.parse_args()

    bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )

    out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        out_dir / Path(__file__).name,
    )

    # ------------------------------------------------------------
    # Use G7's validated zero-margin model creator.
    # ------------------------------------------------------------

    g7.b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    baseline_xml = (
        g7.make_zero_margin_model(
            out_dir
        )
    )

    # ------------------------------------------------------------
    # Frozen global physics.
    # ------------------------------------------------------------

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    old_k = float(
        g7.b2.v35.TENDON_K
    )

    old_mu = float(
        g7.b2.v35.CONTACT_MU
    )

    old_ck = float(
        g7.b2.v35.CONTACT_K
    )

    old_cd = float(
        contact_replay.CONTACT_DAMPING
    )

    rows = []
    replay_data = {}

    try:

        g7.b2.v35.METERS_PER_TICK = SCALE
        g7.b2.v35.TENDON_K = 2100.0
        g7.b2.v35.CONTACT_MU = CONTACT_MU
        g7.b2.v35.CONTACT_K = 1000.0

        contact_replay.CONTACT_DAMPING = 20.0

        print()
        print("=" * 145)
        print(
            "V35G8 — PIP BREAKAWAY-FRICTION / "
            "LOAD-REDISTRIBUTION SCAN"
        )
        print("=" * 145)

        print()
        print(f"H1 bag:\n  {bag}")
        print(f"Output:\n  {out_dir}")

        print()
        print("FROZEN:")
        print(
            f"  transmission scale = "
            f"{SCALE*1e6:.3f} um/tick"
        )
        print("  tendon stiffness   = 2100 N/m")
        print("  contact mu         = 0.05")
        print("  contact k          = 1000")
        print("  contact damping    = 20")
        print("  margin/gap         = 0")
        print("  PIP stiffness      = 0.0015")
        print("  DIP stiffness      = 0.0475")
        print()
        print(
            f"VARIABLE PIP frictionloss = "
            f"{PIP_FRICTIONS}"
        )

        for friction in PIP_FRICTIONS:

            print()
            print("-" * 145)

            print(
                f"PIP frictionloss = "
                f"{friction:.6f} Nm"
            )

            print("-" * 145)

            candidate = make_candidate(
                baseline_xml,
                out_dir,
                friction,
            )

            (
                result,
                trace,
                qpos_matrix,
                times,
            ) = g7.run_scale(
                SCALE,
                CONTACT_MU,
                candidate,
                bag,
                contact_replay,
            )

            result[
                "pip_frictionloss_Nm"
            ] = float(friction)

            result[
                "mechanical_score_g8"
            ] = mechanics_score(
                result
            )

            rows.append(
                result
            )

            name = (
                f"pipfr_{tag(friction)}"
            )

            g7.save_trace_csv(
                out_dir
                / f"{name}_timeseries.csv",
                trace,
            )

            np.savez_compressed(
                out_dir
                / f"{name}_trajectory.npz",
                times=times,
                qpos=qpos_matrix,
            )

            replay_data[
                float(friction)
            ] = (
                candidate,
                qpos_matrix,
                times,
            )

            print(
                "Target pose MCP/PIP/DIP = "
                f"{result['target_mcp_deg']:.3f} / "
                f"{result['target_pip_deg']:.3f} / "
                f"{result['target_dip_deg']:.3f} deg"
            )

            print(
                f"RMSE = "
                f"{result['rmse_deg']:.3f} deg"
            )

            print()
            print("MIDDLE -> FIRST TIP:")

            print(
                f"  dPIP = "
                f"{result['delta_pip_mid_to_tip_deg']:+.3f} deg"
            )

            print(
                f"  dDIP = "
                f"{result['delta_dip_mid_to_tip_deg']:+.3f} deg"
            )

            print(
                f"  DIP/PIP ratio = "
                f"{result['dip_over_pip_mid_to_tip']:.3f}"
            )

            print()
            print("AFTER FIRST TIP:")

            print(
                f"  PIP speed = "
                f"{result['pip_speed_post_tip_deg_s']:.3f} deg/s"
            )

            print(
                f"  DIP speed = "
                f"{result['dip_speed_post_tip_deg_s']:.3f} deg/s"
            )

            print(
                f"  DIP/PIP ratio = "
                f"{result['dip_over_pip_post_tip']:.3f}"
            )

            print()
            print("CONTACT:")

            print(
                f"  retained middle->tip = "
                f"{result['mid_retained_to_tip']}"
            )

            print(
                f"  middle duty = "
                f"{100*result['mid_duty_mid_to_tip']:.1f}%"
            )

            print(
                f"  transitions mid/tip = "
                f"{result['e2b_mid_transitions_hold']} / "
                f"{result['e2b_tip_transitions_hold']}"
            )

            print(
                f"  target contact mid/tip = "
                f"{result['target_mid_contact']} / "
                f"{result['target_tip_contact']}"
            )

            print(
                f"  target gaps mid/tip = "
                f"{result['target_mid_gap_mm']:+.3f} / "
                f"{result['target_tip_gap_mm']:+.3f} mm"
            )

            print(
                f"  tip force median/max = "
                f"{result['e2b_hold_tip_force_median_N']:.3f} / "
                f"{result['e2b_hold_tip_force_max_N']:.3f} N"
            )

            print(
                f"  G8 mechanics score = "
                f"{result['mechanical_score_g8']:.3f}"
            )

    finally:

        g7.b2.v35.METERS_PER_TICK = old_scale
        g7.b2.v35.TENDON_K = old_k
        g7.b2.v35.CONTACT_MU = old_mu
        g7.b2.v35.CONTACT_K = old_ck

        contact_replay.CONTACT_DAMPING = old_cd

    # ------------------------------------------------------------
    # Save summary.
    # ------------------------------------------------------------

    summary_path = (
        out_dir
        / "event_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(
        rows,
        key=lambda r:
            r["mechanical_score_g8"],
    )

    print()
    print("=" * 145)
    print(
        "V35G8 — MECHANICS-FIRST SUMMARY"
    )
    print("=" * 145)

    print()
    print(
        "PIPfr     score   RMSE   "
        "dPIP    dDIP   ratio1  ratio2  "
        "retain  midTr tipTr  gaps(mid/tip)  Fmax"
    )

    print("-" * 145)

    for r in ranked:

        print(
            f"{r['pip_frictionloss_Nm']:8.5f} "
            f"{r['mechanical_score_g8']:7.2f} "
            f"{r['rmse_deg']:6.3f} "
            f"{r['delta_pip_mid_to_tip_deg']:+7.3f} "
            f"{r['delta_dip_mid_to_tip_deg']:+7.3f} "
            f"{r['dip_over_pip_mid_to_tip']:7.3f} "
            f"{r['dip_over_pip_post_tip']:7.3f} "
            f"{str(r['mid_retained_to_tip']):>6s} "
            f"{r['e2b_mid_transitions_hold']:5d} "
            f"{r['e2b_tip_transitions_hold']:5d} "
            f"{r['target_mid_gap_mm']:+6.3f}/"
            f"{r['target_tip_gap_mm']:+6.3f} "
            f"{r['e2b_hold_tip_force_max_N']:6.2f}"
        )

    best = ranked[0]

    best_friction = float(
        best["pip_frictionloss_Nm"]
    )

    with (
        out_dir
        / "selected_for_visual_check.txt"
    ).open("w") as f:

        f.write(
            "V35G8 mechanically selected candidate\n"
        )

        f.write(
            "=====================================\n\n"
        )

        f.write(
            f"PIP frictionloss = "
            f"{best_friction:.8f} Nm\n"
        )

        f.write(
            f"scale = "
            f"{SCALE*1e6:.3f} um/tick\n"
        )

        f.write(
            f"RMSE = "
            f"{best['rmse_deg']:.6f} deg\n"
        )

        f.write(
            f"mechanical score = "
            f"{best['mechanical_score_g8']:.6f}\n\n"
        )

        f.write(
            "NOT ACCEPTED UNTIL VISUALLY VERIFIED.\n"
        )

    with (
        out_dir
        / "manifest.txt"
    ).open("w") as f:

        f.write(
            "V35G8 — PIP breakaway-friction scan\n\n"
        )

        f.write(
            f"scale = {SCALE*1e6:.3f} um/tick\n"
        )

        f.write(
            f"contact mu = {CONTACT_MU}\n"
        )

        f.write(
            "PIP frictionloss candidates:\n"
        )

        for x in PIP_FRICTIONS:
            f.write(
                f"  {x:.8f} Nm\n"
            )

    # ------------------------------------------------------------
    # Mandatory visual validation.
    # ------------------------------------------------------------

    print()
    print(
        "SELECTED FOR VISUAL CHECK:"
    )

    print(
        f"  PIP frictionloss = "
        f"{best_friction:.6f} Nm"
    )

    print(
        f"  RMSE = "
        f"{best['rmse_deg']:.3f} deg"
    )

    print()
    print(
        "THIS CANDIDATE IS NOT ACCEPTED FROM THE NUMBERS."
    )

    print(
        "Now visually compare the motion against the hardware."
    )

    if not args.no_gui:

        (
            candidate_xml,
            qpos_matrix,
            times,
        ) = replay_data[
            best_friction
        ]

        g7.exact_gui_replay(
            candidate_xml,
            qpos_matrix,
            times,
            (
                f"PIP frictionloss="
                f"{best_friction:.6f} Nm"
            ),
        )

    print()
    print("=" * 145)
    print("V35G8 COMPLETE")
    print("=" * 145)

    print(
        f"\nAll outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
