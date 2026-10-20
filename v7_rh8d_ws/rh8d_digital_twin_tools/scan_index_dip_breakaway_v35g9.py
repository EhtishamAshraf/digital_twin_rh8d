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
# V35G9 — DIP BREAKAWAY-FRICTION / SEQUENTIAL-WRAP SCAN
#
# Critical hardware mechanism:
#
# MCP moves
# -> middle touches
# -> PIP keeps wrapping
# -> middle becomes constrained
# -> PIP slows/stops
# -> DIP THEN rotates
# -> distal becomes constrained
# -> DIP stops
#
# G6/G8 showed that the current simulated DIP is already strongly
# flexed at FIRST MIDDLE CONTACT and then hardly moves afterward.
#
# Therefore G9 tests a DIP breakaway threshold.
#
# VARIABLE:
#   DIP frictionloss only.
#
# RMSE IS NOT THE PRIMARY SELECTION CRITERION.
# ================================================================


SCALE = 5.70e-6
CONTACT_MU = 0.05

DIP_FRICTIONS = [
    0.0000,
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0030,
    0.0040,
]

DIP_JOINT = "Index_Distal--Index_Middle"


def tag(x):
    return (
        f"{x:.5f}"
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
        if element.get("name") == DIP_JOINT:
            joint = element
            break

    if joint is None:
        raise RuntimeError(
            f"Could not find DIP joint: {DIP_JOINT}"
        )

    old = joint.get(
        "frictionloss",
        "0",
    )

    joint.set(
        "frictionloss",
        f"{friction:.10g}",
    )

    destination = (
        out_dir
        / f"candidate_dip_friction_{tag(friction)}.xml"
    )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(
        f"    DIP frictionloss: "
        f"{old} -> {friction:.6f} Nm"
    )

    return destination


def first_contact_index(trace, field):

    for i, row in enumerate(trace):
        if bool(row[field]):
            return i

    return None


def calc_extra_metrics(trace):

    times = np.asarray(
        [r["time_s"] for r in trace],
        dtype=float,
    )

    pip = np.asarray(
        [r["pip_deg"] for r in trace],
        dtype=float,
    )

    dip = np.asarray(
        [r["dip_deg"] for r in trace],
        dtype=float,
    )

    mid = np.asarray(
        [r["mid_contact"] for r in trace],
        dtype=bool,
    )

    tip = np.asarray(
        [r["tip_contact"] for r in trace],
        dtype=bool,
    )

    i_mid = first_contact_index(
        trace,
        "mid_contact",
    )

    i_tip = first_contact_index(
        trace,
        "tip_contact",
    )

    result = {
        "pip_at_first_mid_deg": np.nan,
        "dip_at_first_mid_deg": np.nan,
        "pip_at_first_tip_deg": np.nan,
        "dip_at_first_tip_deg": np.nan,
        "dip_motion_last_200ms_before_mid_deg": np.nan,
        "pip_motion_last_200ms_before_mid_deg": np.nan,
        "dip_motion_mid_to_tip_deg": np.nan,
        "pip_motion_mid_to_tip_deg": np.nan,
        "dip_activation_ratio": np.nan,
        "dip_speed_pre_mid_deg_s_g9": np.nan,
        "dip_speed_mid_to_tip_deg_s_g9": np.nan,
        "pip_speed_pre_mid_deg_s_g9": np.nan,
        "pip_speed_mid_to_tip_deg_s_g9": np.nan,
    }

    if i_mid is None:
        return result

    result[
        "pip_at_first_mid_deg"
    ] = float(pip[i_mid])

    result[
        "dip_at_first_mid_deg"
    ] = float(dip[i_mid])

    t_mid = float(times[i_mid])

    # ------------------------------------------------------------
    # Last 200 ms before middle contact.
    # This measures premature DIP flexion immediately before contact.
    # ------------------------------------------------------------

    pre_indices = np.flatnonzero(
        (times >= t_mid - 0.200)
        &
        (times <= t_mid)
    )

    if len(pre_indices) >= 2:

        a = int(pre_indices[0])
        b = int(pre_indices[-1])

        dip_pre_motion = (
            float(dip[b])
            -
            float(dip[a])
        )

        pip_pre_motion = (
            float(pip[b])
            -
            float(pip[a])
        )

        dt = max(
            float(times[b] - times[a]),
            1e-9,
        )

        result[
            "dip_motion_last_200ms_before_mid_deg"
        ] = dip_pre_motion

        result[
            "pip_motion_last_200ms_before_mid_deg"
        ] = pip_pre_motion

        result[
            "dip_speed_pre_mid_deg_s_g9"
        ] = abs(
            dip_pre_motion / dt
        )

        result[
            "pip_speed_pre_mid_deg_s_g9"
        ] = abs(
            pip_pre_motion / dt
        )

    # ------------------------------------------------------------
    # First middle -> first distal contact.
    # ------------------------------------------------------------

    if (
        i_tip is not None
        and
        i_tip > i_mid
    ):

        result[
            "pip_at_first_tip_deg"
        ] = float(pip[i_tip])

        result[
            "dip_at_first_tip_deg"
        ] = float(dip[i_tip])

        dip_after = (
            float(dip[i_tip])
            -
            float(dip[i_mid])
        )

        pip_after = (
            float(pip[i_tip])
            -
            float(pip[i_mid])
        )

        result[
            "dip_motion_mid_to_tip_deg"
        ] = dip_after

        result[
            "pip_motion_mid_to_tip_deg"
        ] = pip_after

        dt_after = max(
            float(
                times[i_tip]
                -
                times[i_mid]
            ),
            1e-9,
        )

        result[
            "dip_speed_mid_to_tip_deg_s_g9"
        ] = abs(
            dip_after
            /
            dt_after
        )

        result[
            "pip_speed_mid_to_tip_deg_s_g9"
        ] = abs(
            pip_after
            /
            dt_after
        )

        pre_speed = result[
            "dip_speed_pre_mid_deg_s_g9"
        ]

        post_speed = result[
            "dip_speed_mid_to_tip_deg_s_g9"
        ]

        if (
            np.isfinite(pre_speed)
            and
            np.isfinite(post_speed)
        ):

            result[
                "dip_activation_ratio"
            ] = (
                post_speed
                /
                max(
                    pre_speed,
                    1e-6,
                )
            )

    return result


def mechanics_score(r):
    """
    Lower is better.

    Key idea:
      DIP should NOT already be strongly flexing before middle contact.
      DIP should become more active AFTER middle constraint/contact.

    Visual validation remains mandatory.
    """

    score = 0.0

    if not r["sequence_ok"]:
        score += 1000.0

    if not r["mid_retained_to_tip"]:
        score += 350.0

    # Repeated contact losses/bounce are strongly penalized.
    score += (
        60.0
        *
        max(
            0,
            r["e2b_mid_transitions_hold"] - 1,
        )
    )

    score += (
        60.0
        *
        max(
            0,
            r["e2b_tip_transitions_hold"] - 1,
        )
    )

    # ------------------------------------------------------------
    # Premature DIP activity:
    # penalize DIP motion in 200 ms BEFORE middle contact.
    # ------------------------------------------------------------

    pre_dip = abs(
        r[
            "dip_motion_last_200ms_before_mid_deg"
        ]
    )

    if np.isfinite(pre_dip):
        score += (
            20.0
            *
            pre_dip
        )
    else:
        score += 100.0

    # ------------------------------------------------------------
    # Reward DIP activation AFTER middle contact.
    # ------------------------------------------------------------

    post_dip = r[
        "dip_motion_mid_to_tip_deg"
    ]

    if np.isfinite(post_dip):

        # We want meaningful positive DIP rotation.
        score += (
            80.0
            *
            max(
                0.0,
                0.5 - post_dip,
            )
        )

    else:
        score += 200.0

    activation = r[
        "dip_activation_ratio"
    ]

    if np.isfinite(activation):

        # Hardware mechanism requires relative DIP activation.
        # Ratio > 1 means DIP becomes more active after middle contact.
        score += (
            100.0
            *
            max(
                0.0,
                1.0 - activation,
            )
        )

    else:
        score += 150.0

    # ------------------------------------------------------------
    # Target contact/gaps.
    # ------------------------------------------------------------

    if not r["target_mid_contact"]:
        score += 100.0

    if not r["target_tip_contact"]:
        score += 100.0

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

    # Large impact force.
    fmax = r[
        "e2b_hold_tip_force_max_N"
    ]

    if np.isfinite(fmax):
        score += (
            10.0
            *
            max(
                0.0,
                fmax - 2.0,
            )
        )

    # RMSE is deliberately weak.
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
            / "v35g9_dip_breakaway"
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
        out_dir
        / Path(__file__).name,
    )

    g7.b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    baseline_xml = (
        g7.make_zero_margin_model(
            out_dir
        )
    )

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
        print("=" * 150)
        print(
            "V35G9 — DIP BREAKAWAY-FRICTION / "
            "SEQUENTIAL-WRAP SCAN"
        )
        print("=" * 150)

        print()
        print("HARDWARE MECHANISM:")
        print(
            "  middle touch -> PIP wrap -> middle constraint "
            "-> PIP stops -> DIP rotates -> distal constraint -> DIP stops"
        )

        print()
        print("FROZEN:")
        print(
            f"  transmission scale = "
            f"{SCALE*1e6:.3f} um/tick"
        )
        print("  contact mu          = 0.05")
        print("  tendon stiffness    = 2100 N/m")
        print("  MCP k               = 0.0045")
        print("  PIP k               = 0.0015")
        print("  PIP frictionloss    = 0.00125")
        print("  DIP k               = 0.0475")
        print("  contact k           = 1000")
        print("  contact damping     = 20")
        print("  margin/gap          = 0")

        print()
        print(
            f"VARIABLE DIP frictionloss = "
            f"{DIP_FRICTIONS}"
        )

        for friction in DIP_FRICTIONS:

            print()
            print("-" * 150)

            print(
                f"DIP frictionloss = "
                f"{friction:.6f} Nm"
            )

            print("-" * 150)

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
                "dip_frictionloss_Nm"
            ] = float(
                friction
            )

            extra = calc_extra_metrics(
                trace
            )

            result.update(
                extra
            )

            result[
                "mechanical_score_g9"
            ] = mechanics_score(
                result
            )

            rows.append(
                result
            )

            name = (
                f"dipfr_{tag(friction)}"
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
            print("AT FIRST MIDDLE CONTACT:")

            print(
                f"  PIP = "
                f"{result['pip_at_first_mid_deg']:.3f} deg"
            )

            print(
                f"  DIP = "
                f"{result['dip_at_first_mid_deg']:.3f} deg"
            )

            print()
            print("LAST 200 ms BEFORE MIDDLE CONTACT:")

            print(
                f"  dPIP = "
                f"{result['pip_motion_last_200ms_before_mid_deg']:+.3f} deg"
            )

            print(
                f"  dDIP = "
                f"{result['dip_motion_last_200ms_before_mid_deg']:+.3f} deg"
            )

            print(
                f"  DIP speed = "
                f"{result['dip_speed_pre_mid_deg_s_g9']:.3f} deg/s"
            )

            print()
            print("MIDDLE -> FIRST DISTAL CONTACT:")

            print(
                f"  dPIP = "
                f"{result['pip_motion_mid_to_tip_deg']:+.3f} deg"
            )

            print(
                f"  dDIP = "
                f"{result['dip_motion_mid_to_tip_deg']:+.3f} deg"
            )

            print(
                f"  DIP speed = "
                f"{result['dip_speed_mid_to_tip_deg_s_g9']:.3f} deg/s"
            )

            print(
                f"  DIP activation ratio = "
                f"{result['dip_activation_ratio']:.3f}"
            )

            print()
            print("CONTACT STABILITY:")

            print(
                f"  middle retained to tip = "
                f"{result['mid_retained_to_tip']}"
            )

            print(
                f"  transitions middle/tip = "
                f"{result['e2b_mid_transitions_hold']} / "
                f"{result['e2b_tip_transitions_hold']}"
            )

            print(
                f"  target contacts = "
                f"{result['target_mid_contact']} / "
                f"{result['target_tip_contact']}"
            )

            print(
                f"  target gaps = "
                f"{result['target_mid_gap_mm']:+.3f} / "
                f"{result['target_tip_gap_mm']:+.3f} mm"
            )

            print(
                f"  tip force median/max = "
                f"{result['e2b_hold_tip_force_median_N']:.3f} / "
                f"{result['e2b_hold_tip_force_max_N']:.3f} N"
            )

            print(
                f"  G9 mechanics score = "
                f"{result['mechanical_score_g9']:.3f}"
            )

    finally:

        g7.b2.v35.METERS_PER_TICK = old_scale
        g7.b2.v35.TENDON_K = old_k
        g7.b2.v35.CONTACT_MU = old_mu
        g7.b2.v35.CONTACT_K = old_ck

        contact_replay.CONTACT_DAMPING = old_cd

    summary = (
        out_dir
        / "event_summary.csv"
    )

    with summary.open(
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
            r[
                "mechanical_score_g9"
            ],
    )

    print()
    print("=" * 150)
    print(
        "V35G9 — SEQUENTIAL-MECHANICS SUMMARY"
    )
    print("=" * 150)

    print()
    print(
        "DIPfr    RMSE  DIP@MID  preDIP  "
        "postDIP activation retain trM trT "
        "gapM   gapT   Fmax   score"
    )

    print("-" * 150)

    for r in ranked:

        print(
            f"{r['dip_frictionloss_Nm']:7.4f} "
            f"{r['rmse_deg']:6.3f} "
            f"{r['dip_at_first_mid_deg']:7.2f} "
            f"{r['dip_motion_last_200ms_before_mid_deg']:+7.3f} "
            f"{r['dip_motion_mid_to_tip_deg']:+7.3f} "
            f"{r['dip_activation_ratio']:10.3f} "
            f"{str(r['mid_retained_to_tip']):>6s} "
            f"{r['e2b_mid_transitions_hold']:3d} "
            f"{r['e2b_tip_transitions_hold']:3d} "
            f"{r['target_mid_gap_mm']:+6.3f} "
            f"{r['target_tip_gap_mm']:+6.3f} "
            f"{r['e2b_hold_tip_force_max_N']:6.2f} "
            f"{r['mechanical_score_g9']:7.2f}"
        )

    best = ranked[0]

    best_friction = float(
        best[
            "dip_frictionloss_Nm"
        ]
    )

    with (
        out_dir
        / "selected_for_visual_check.txt"
    ).open("w") as f:

        f.write(
            "V35G9 mechanically selected candidate\n"
        )

        f.write(
            "=====================================\n\n"
        )

        f.write(
            f"DIP frictionloss = "
            f"{best_friction:.8f} Nm\n"
        )

        f.write(
            f"scale = "
            f"{SCALE*1e6:.3f} um/tick\n"
        )

        f.write(
            f"DIP at first middle contact = "
            f"{best['dip_at_first_mid_deg']:.6f} deg\n"
        )

        f.write(
            f"DIP activation ratio = "
            f"{best['dip_activation_ratio']:.6f}\n"
        )

        f.write(
            f"RMSE = "
            f"{best['rmse_deg']:.6f} deg\n\n"
        )

        f.write(
            "NOT ACCEPTED UNTIL VISUAL VALIDATION.\n"
        )

    with (
        out_dir
        / "manifest.txt"
    ).open("w") as f:

        f.write(
            "V35G9 — DIP breakaway-friction scan\n\n"
        )

        f.write(
            f"scale = "
            f"{SCALE*1e6:.3f} um/tick\n"
        )

        f.write(
            "contact mu = 0.05\n"
        )

        f.write(
            "PIP frictionloss = 0.00125 Nm\n"
        )

        f.write(
            "DIP stiffness = 0.0475 Nm/rad\n\n"
        )

        f.write(
            "DIP frictionloss candidates:\n"
        )

        for x in DIP_FRICTIONS:
            f.write(
                f"  {x:.8f} Nm\n"
            )

    print()
    print(
        "SELECTED FOR MANDATORY VISUAL CHECK:"
    )

    print(
        f"  DIP frictionloss = "
        f"{best_friction:.6f} Nm"
    )

    print(
        f"  DIP at first middle contact = "
        f"{best['dip_at_first_mid_deg']:.3f} deg"
    )

    print(
        f"  DIP activation ratio = "
        f"{best['dip_activation_ratio']:.3f}"
    )

    print(
        f"  RMSE = "
        f"{best['rmse_deg']:.3f} deg"
    )

    print()
    print(
        "VISUAL ACCEPTANCE SEQUENCE:"
    )

    print(
        "  MCP -> middle touch -> PIP wrap -> "
        "middle constraint -> PIP stop -> "
        "DIP rotation -> distal constraint -> DIP stop"
    )

    print()
    print(
        "Reject it if the finger still performs "
        "touch -> bounce -> touch -> bounce."
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
                f"DIP frictionloss="
                f"{best_friction:.6f} Nm"
            ),
        )

    print()
    print("=" * 150)
    print("V35G9 COMPLETE")
    print("=" * 150)

    print(
        f"\nAll outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
