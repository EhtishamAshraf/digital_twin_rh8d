#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import scan_index_contact_friction_v35g7 as g7


SCALE = 5.70e-6
CONTACT_MU = 0.05
TENDON_K = 2100.0

TENDON_DAMPINGS = [
    2.0,
    5.0,
    10.0,
    20.0,
    40.0,
    80.0,
]

TENDON_NAME = "idx_flex_spatial"


def tag(x):
    return f"{x:.1f}".replace(".", "p")


def make_candidate(source_xml: Path, out_dir: Path, damping: float) -> Path:
    tree = ET.parse(source_xml)
    root = tree.getroot()

    tendon = None

    for element in root.iter("spatial"):
        if element.get("name") == TENDON_NAME:
            tendon = element
            break

    if tendon is None:
        raise RuntimeError(
            f"Could not find tendon {TENDON_NAME}"
        )

    old = tendon.get("damping")

    tendon.set(
        "damping",
        f"{damping:.10g}",
    )

    destination = (
        out_dir
        / f"candidate_tendon_damping_{tag(damping)}.xml"
    )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(
        f"    tendon damping: {old} -> {damping}"
    )

    return destination


def count_contact_episodes(trace, field):
    flags = np.asarray(
        [bool(r[field]) for r in trace],
        dtype=bool,
    )

    episodes = 0
    prev = False

    for value in flags:
        if value and not prev:
            episodes += 1
        prev = value

    return episodes


def first_contact_index(trace, field):
    for i, row in enumerate(trace):
        if bool(row[field]):
            return i
    return None


def extra_metrics(trace):
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

    i_mid = first_contact_index(
        trace,
        "mid_contact",
    )

    i_tip = first_contact_index(
        trace,
        "tip_contact",
    )

    result = {
        "middle_contact_episodes": count_contact_episodes(
            trace,
            "mid_contact",
        ),
        "tip_contact_episodes": count_contact_episodes(
            trace,
            "tip_contact",
        ),
        "pip_at_first_mid_deg": np.nan,
        "dip_at_first_mid_deg": np.nan,
        "pip_at_first_tip_deg": np.nan,
        "dip_at_first_tip_deg": np.nan,
        "delta_pip_mid_to_tip_deg_g10": np.nan,
        "delta_dip_mid_to_tip_deg_g10": np.nan,
    }

    if i_mid is not None:
        result["pip_at_first_mid_deg"] = float(
            pip[i_mid]
        )
        result["dip_at_first_mid_deg"] = float(
            dip[i_mid]
        )

    if (
        i_mid is not None
        and
        i_tip is not None
        and
        i_tip > i_mid
    ):
        result["pip_at_first_tip_deg"] = float(
            pip[i_tip]
        )
        result["dip_at_first_tip_deg"] = float(
            dip[i_tip]
        )

        result[
            "delta_pip_mid_to_tip_deg_g10"
        ] = float(
            pip[i_tip] - pip[i_mid]
        )

        result[
            "delta_dip_mid_to_tip_deg_g10"
        ] = float(
            dip[i_tip] - dip[i_mid]
        )

    return result


def mechanics_score(r):
    score = 0.0

    if not r["sequence_ok"]:
        score += 1000.0

    if not r["mid_retained_to_tip"]:
        score += 300.0

    # Strongly penalize repeated impact/rebound episodes.
    score += (
        120.0
        *
        max(
            0,
            r["middle_contact_episodes"] - 1,
        )
    )

    score += (
        120.0
        *
        max(
            0,
            r["tip_contact_episodes"] - 1,
        )
    )

    # Still want some PIP wrap after middle contact.
    dpip = r["delta_pip_mid_to_tip_deg_g10"]

    if np.isfinite(dpip):
        score += (
            50.0
            *
            max(
                0.0,
                0.2 - dpip,
            )
        )
    else:
        score += 100.0

    # We want meaningful DIP redistribution too.
    ddip = r["delta_dip_mid_to_tip_deg_g10"]

    if np.isfinite(ddip):
        score += (
            80.0
            *
            max(
                0.0,
                0.25 - ddip,
            )
        )
    else:
        score += 150.0

    # Target contact penalties.
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

    # Force spikes.
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
            / "v35g10_tendon_damping"
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

    g7.b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    baseline_xml = g7.make_zero_margin_model(
        out_dir
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
        g7.b2.v35.TENDON_K = TENDON_K
        g7.b2.v35.CONTACT_MU = CONTACT_MU
        g7.b2.v35.CONTACT_K = 1000.0

        contact_replay.CONTACT_DAMPING = 20.0

        print()
        print("=" * 145)
        print(
            "V35G10 — TENDON DAMPING / REBOUND-SUPPRESSION SCAN"
        )
        print("=" * 145)

        print()
        print("FROZEN:")
        print(
            f"  transmission scale = "
            f"{SCALE*1e6:.3f} um/tick"
        )
        print(
            f"  tendon stiffness   = {TENDON_K:.1f} N/m"
        )
        print(
            "  contact mu         = 0.05"
        )
        print(
            "  contact k          = 1000"
        )
        print(
            "  contact damping    = 20"
        )
        print(
            "  PIP frictionloss   = 0.00125"
        )
        print(
            "  DIP frictionloss   = 0"
        )

        print()
        print(
            f"VARIABLE tendon damping = {TENDON_DAMPINGS}"
        )

        for damping in TENDON_DAMPINGS:

            print()
            print("-" * 145)
            print(
                f"Tendon damping = {damping:.3f}"
            )
            print("-" * 145)

            candidate = make_candidate(
                baseline_xml,
                out_dir,
                damping,
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
                "tendon_damping"
            ] = float(damping)

            result.update(
                extra_metrics(trace)
            )

            result[
                "mechanical_score_g10"
            ] = mechanics_score(
                result
            )

            rows.append(
                result
            )

            name = (
                f"tdamp_{tag(damping)}"
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
                float(damping)
            ] = (
                candidate,
                qpos_matrix,
                times,
            )

            print(
                "Target pose MCP/PIP/DIP = "
                f"{result['target_mcp_deg']:.3f} / "
                f"{result['target_pip_deg']:.3f} / "
                f"{result['target_dip_deg']:.3f}"
            )

            print(
                f"RMSE = "
                f"{result['rmse_deg']:.3f} deg"
            )

            print(
                f"contact episodes middle/tip = "
                f"{result['middle_contact_episodes']} / "
                f"{result['tip_contact_episodes']}"
            )

            print(
                "first middle pose PIP/DIP = "
                f"{result['pip_at_first_mid_deg']:.3f} / "
                f"{result['dip_at_first_mid_deg']:.3f}"
            )

            print(
                "middle->tip dPIP/dDIP = "
                f"{result['delta_pip_mid_to_tip_deg_g10']:+.3f} / "
                f"{result['delta_dip_mid_to_tip_deg_g10']:+.3f}"
            )

            print(
                f"hold transitions middle/tip = "
                f"{result['e2b_mid_transitions_hold']} / "
                f"{result['e2b_tip_transitions_hold']}"
            )

            print(
                f"target contacts middle/tip = "
                f"{result['target_mid_contact']} / "
                f"{result['target_tip_contact']}"
            )

            print(
                f"target gaps middle/tip = "
                f"{result['target_mid_gap_mm']:+.3f} / "
                f"{result['target_tip_gap_mm']:+.3f} mm"
            )

            print(
                f"tip force median/max = "
                f"{result['e2b_hold_tip_force_median_N']:.3f} / "
                f"{result['e2b_hold_tip_force_max_N']:.3f} N"
            )

            print(
                f"G10 mechanics score = "
                f"{result['mechanical_score_g10']:.3f}"
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
            r["mechanical_score_g10"],
    )

    print()
    print("=" * 145)
    print(
        "V35G10 — MECHANICS-FIRST SUMMARY"
    )
    print("=" * 145)

    print()
    print(
        "Tdamp  RMSE  episodes(M/T) "
        "DIP@MID  dPIP   dDIP   trM trT "
        "gapM   gapT   Fmax   score"
    )

    print("-" * 145)

    for r in ranked:
        print(
            f"{r['tendon_damping']:6.1f} "
            f"{r['rmse_deg']:6.3f} "
            f"{r['middle_contact_episodes']:3d}/"
            f"{r['tip_contact_episodes']:3d} "
            f"{r['dip_at_first_mid_deg']:7.2f} "
            f"{r['delta_pip_mid_to_tip_deg_g10']:+6.3f} "
            f"{r['delta_dip_mid_to_tip_deg_g10']:+6.3f} "
            f"{r['e2b_mid_transitions_hold']:3d} "
            f"{r['e2b_tip_transitions_hold']:3d} "
            f"{r['target_mid_gap_mm']:+6.3f} "
            f"{r['target_tip_gap_mm']:+6.3f} "
            f"{r['e2b_hold_tip_force_max_N']:6.2f} "
            f"{r['mechanical_score_g10']:7.2f}"
        )

    best = ranked[0]

    best_damping = float(
        best["tendon_damping"]
    )

    with (
        out_dir
        / "selected_for_visual_check.txt"
    ).open("w") as f:

        f.write(
            "V35G10 mechanically selected candidate\n"
        )
        f.write(
            "======================================\n\n"
        )
        f.write(
            f"tendon damping = "
            f"{best_damping:.6f}\n"
        )
        f.write(
            f"RMSE = "
            f"{best['rmse_deg']:.6f} deg\n"
        )
        f.write(
            f"middle episodes = "
            f"{best['middle_contact_episodes']}\n"
        )
        f.write(
            f"tip episodes = "
            f"{best['tip_contact_episodes']}\n\n"
        )
        f.write(
            "NOT ACCEPTED UNTIL VISUAL CHECK.\n"
        )

    print()
    print(
        "SELECTED FOR MANDATORY VISUAL CHECK:"
    )
    print(
        f"  tendon damping = "
        f"{best_damping:.3f}"
    )
    print(
        f"  RMSE = "
        f"{best['rmse_deg']:.3f} deg"
    )

    print()
    print(
        "Reject if you still see:"
    )
    print(
        "  touch -> bounce -> touch -> bounce"
    )

    if not args.no_gui:

        (
            candidate_xml,
            qpos_matrix,
            times,
        ) = replay_data[
            best_damping
        ]

        g7.exact_gui_replay(
            candidate_xml,
            qpos_matrix,
            times,
            (
                f"tendon damping="
                f"{best_damping:.3f}"
            ),
        )

    print()
    print("=" * 145)
    print("V35G10 COMPLETE")
    print("=" * 145)

    print(
        f"\nAll outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
