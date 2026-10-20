#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import fit_index_passive_contact_v35b2 as b2
import replay_index_zero_margin_v35e2a as e2a
import scan_index_zero_margin_damping_v35e2b as e2b


# ================================================================
# V35G3 — COUPLED PIP x DIP PASSIVE-STIFFNESS / WRAP SCAN
#
# PURPOSE
# -------
# Test the local coupled passive-mechanics region identified by
# V35G1 (DIP-only) and V35G2 (PIP-only), while preserving the
# current physically defensible H1 setup.
#
# VARIABLES ONLY
# --------------
#   PIP stiffness
#   DIP stiffness
#
# FROZEN
# ------
#   MCP passive mechanics        = V35B2 selected model
#   PIP frictionloss             = V35B2 selected model
#   DIP frictionloss             = V35B2 selected model
#   CAD spatial tendon           = frozen
#   tendon scale/stiffness       = frozen by existing replay stack
#   H1 bottle pose/support       = frozen
#   measured H1 encoder replay   = frozen
#   contact margin/gap           = 0 / 0
#   contact mu                   = 0.05
#   direct contact stiffness     = 1000
#   direct contact damping       = 20
#
# IMPORTANT
# ---------
# This script does NOT select a winner from RMSE alone.
# The shortlist is lexicographic and mechanics-first:
#
#   1) middle contact occurs before tip contact
#   2) simultaneous middle+tip contact actually occurs
#   3) target frame retains more physical contact
#   4) fewer contact transitions/losses
#   5) longer continuous middle+tip overlap
#   6) smaller geometric separation/penetration near target/final
#   7) force proximity to H1 hardware median
#   8) angle RMSE only as a late tie-breaker
#
# Final acceptance still requires native MuJoCo GUI inspection:
#
#   middle contacts bottle -> remains constrained -> distal wraps.
# ================================================================


PIP_K_VALUES = [
    0.0015,
    0.0020,
    0.0025,
    0.0030,
    0.0035,
]

DIP_K_VALUES = [
    0.0300,
    0.0350,
    0.0400,
    0.0475,
]


CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

PIP_JOINT_NAME = "Index_Middle--Index_Proximal"
DIP_JOINT_NAME = "Index_Distal--Index_Middle"

HW_FORCE = 0.14563


CSV_FIELDS = [
    "rank",
    "pip_k",
    "dip_k",
    "mcp_deg",
    "pip_deg",
    "dip_deg",
    "rmse_deg",
    "first_mid_tick",
    "first_tip_tick",
    "force_onset_tick",
    "sequence_ok",
    "ever_both",
    "overlap_total_s",
    "overlap_longest_s",
    "mid_transitions",
    "tip_transitions",
    "target_mid",
    "target_tip",
    "target_contacts",
    "target_mid_gap_mm",
    "target_tip_gap_mm",
    "target_gap_abs_sum_mm",
    "tip_force_median_N",
    "tip_force_max_N",
    "force_error_N",
    "final_mid_gap_mm",
    "final_tip_gap_mm",
    "max_penetration_mm",
    "xml",
]


def safe_float(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return float(default)
    return value


def finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def find_unique_joint(root, name):
    matches = [
        joint
        for joint in root.iter("joint")
        if joint.get("name") == name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one joint named {name!r}, "
            f"found {len(matches)}"
        )

    return matches[0]


def make_candidate(
    source_xml: Path,
    pip_k: float,
    dip_k: float,
) -> Path:
    """Patch ONLY PIP and DIP stiffness in the zero-margin model."""

    source_xml = source_xml.expanduser().resolve()

    tree = ET.parse(source_xml)
    root = tree.getroot()

    pip_joint = find_unique_joint(root, PIP_JOINT_NAME)
    dip_joint = find_unique_joint(root, DIP_JOINT_NAME)

    old_pip_k = pip_joint.get("stiffness")
    old_dip_k = dip_joint.get("stiffness")

    pip_joint.set("stiffness", f"{pip_k:.10g}")
    dip_joint.set("stiffness", f"{dip_k:.10g}")

    pip_tag = f"{pip_k:.4f}".replace(".", "p")
    dip_tag = f"{dip_k:.4f}".replace(".", "p")

    out = source_xml.parent / (
        source_xml.stem
        + f"_v35g3_pipk_{pip_tag}_dipk_{dip_tag}.xml"
    )

    tree.write(
        out,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(
        f"    PIP stiffness: {old_pip_k} -> {pip_k:.5f} Nm/rad"
    )
    print(
        f"    DIP stiffness: {old_dip_k} -> {dip_k:.5f} Nm/rad"
    )

    return out


def target_gap_abs_sum(row):
    vals = [
        row["target_mid_gap_mm"],
        row["target_tip_gap_mm"],
    ]
    return float(
        sum(abs(v) for v in vals if np.isfinite(v))
    )


def max_penetration_mm(row):
    """Largest negative physical gap magnitude among target/final checks."""

    gaps = [
        row["target_mid_gap_mm"],
        row["target_tip_gap_mm"],
        row["final_mid_gap_mm"],
        row["final_tip_gap_mm"],
    ]

    penetration = [
        max(0.0, -float(g))
        for g in gaps
        if np.isfinite(g)
    ]

    return float(max(penetration)) if penetration else np.nan


def mechanics_rank_key(row):
    """
    Mechanics-first lexicographic key.

    There is deliberately no arbitrary weighted scalar score.
    Lower tuple is better.
    """

    total_transitions = (
        int(row["mid_transitions"])
        + int(row["tip_transitions"])
    )

    force_error = row["force_error_N"]
    if not np.isfinite(force_error):
        force_error = np.inf

    return (
        0 if row["sequence_ok"] else 1,
        0 if row["ever_both"] else 1,
        2 - int(row["target_contacts"]),
        total_transitions,
        -float(row["overlap_longest_s"]),
        float(row["target_gap_abs_sum_mm"]),
        float(row["max_penetration_mm"]),
        float(force_error),
        float(row["rmse_deg"]),
    )


def save_csv(path: Path, rows):
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(key, "")
                    for key in CSV_FIELDS
                }
            )

    return path


def print_matrix(rows, value_key, title, fmt="7.3f"):
    lookup = {
        (row["pip_k"], row["dip_k"]): row[value_key]
        for row in rows
    }

    print()
    print(title)
    print(
        "PIP\\DIP "
        + " ".join(f"{k:>9.4f}" for k in DIP_K_VALUES)
    )

    for pip_k in PIP_K_VALUES:
        vals = []
        for dip_k in DIP_K_VALUES:
            value = lookup[(float(pip_k), float(dip_k))]
            if isinstance(value, (bool, np.bool_)):
                vals.append(f"{str(bool(value)):>9s}")
            elif np.isfinite(float(value)):
                vals.append(f"{float(value):9.3f}")
            else:
                vals.append(f"{'nan':>9s}")

        print(
            f"{pip_k:7.4f} "
            + " ".join(vals)
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "V35G3 coupled PIP x DIP passive-stiffness scan "
            "for RH8D H1 wrapping mechanics."
        )
    )

    parser.add_argument(
        "contact_bag",
        type=Path,
        help="H1 rosbag directory used by the existing replay stack",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("v35g3_pip_dip_wrap_results.csv"),
        help="output CSV path",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="number of mechanics-first shortlist rows to print",
    )

    args = parser.parse_args()

    bag = args.contact_bag.expanduser().resolve()
    if not bag.exists():
        raise RuntimeError(f"Missing contact bag:\n{bag}")

    if args.top < 1:
        raise RuntimeError("--top must be >= 1")

    # Same patch/import path used by G1/G2.
    b2.v30.v24.ab.patch_replay_modules()
    contact_replay = b2.v30.v24.base.contact_replay

    # Generate/use the same zero-margin source model used by E2A/E2B/G1/G2.
    zero_margin_xml = e2a.make_zero_margin_xml()

    old_mu = b2.v35.CONTACT_MU
    old_contact_k = b2.v35.CONTACT_K
    old_damping = float(contact_replay.CONTACT_DAMPING)

    b2.v35.CONTACT_MU = CONTACT_MU
    b2.v35.CONTACT_K = CONTACT_K
    contact_replay.CONTACT_DAMPING = CONTACT_DAMPING

    rows = []
    total = len(PIP_K_VALUES) * len(DIP_K_VALUES)

    try:
        print()
        print("=" * 176)
        print("V35G3 — COUPLED PIP x DIP PASSIVE-STIFFNESS / WRAP SCAN")
        print("=" * 176)
        print()
        print(f"H1 bag:\n  {bag}")
        print(f"Zero-margin source XML:\n  {zero_margin_xml}")
        print()
        print("FROZEN:")
        print("  MCP mechanics       = V35B2 selected model")
        print("  PIP frictionloss    = V35B2 selected model")
        print("  CAD tendon          = frozen")
        print("  H1 bottle/support   = frozen")
        print("  measured encoder    = replayed")
        print("  margin / gap        = 0 / 0")
        print(f"  contact mu          = {CONTACT_MU}")
        print(f"  direct stiffness    = {CONTACT_K}")
        print(f"  direct damping      = {CONTACT_DAMPING}")
        print()
        print("VARIABLES:")
        print(f"  PIP stiffness = {PIP_K_VALUES}")
        print(f"  DIP stiffness = {DIP_K_VALUES}")
        print(f"  total candidates = {total}")
        print()

        candidate_index = 0

        for pip_k in PIP_K_VALUES:
            for dip_k in DIP_K_VALUES:
                candidate_index += 1

                print()
                print("-" * 176)
                print(
                    f"[{candidate_index:02d}/{total:02d}] "
                    f"PIP k={pip_k:.5f}, DIP k={dip_k:.5f} Nm/rad"
                )
                print("-" * 176)

                candidate_xml = make_candidate(
                    zero_margin_xml,
                    pip_k,
                    dip_k,
                )

                result = e2b.run_one(
                    CONTACT_DAMPING,
                    candidate_xml,
                    bag,
                    contact_replay,
                )

                pose = np.asarray(result["pose"], dtype=float)

                first_mid = safe_float(result["first_mid_encoder"])
                first_tip = safe_float(result["first_tip_encoder"])

                sequence_ok = bool(
                    finite(first_mid)
                    and finite(first_tip)
                    and first_mid < first_tip
                )

                tip_force_med = safe_float(result["positive_tip_median"])
                force_error = (
                    abs(tip_force_med - HW_FORCE)
                    if np.isfinite(tip_force_med)
                    else np.inf
                )

                row = {
                    "rank": "",
                    "pip_k": float(pip_k),
                    "dip_k": float(dip_k),
                    "mcp_deg": float(pose[0]),
                    "pip_deg": float(pose[1]),
                    "dip_deg": float(pose[2]),
                    "rmse_deg": float(result["rmse"]),
                    "first_mid_tick": first_mid,
                    "first_tip_tick": first_tip,
                    "force_onset_tick": safe_float(
                        result["force_onset_encoder"]
                    ),
                    "sequence_ok": sequence_ok,
                    "ever_both": bool(result["overlap_longest"] > 0.0),
                    "overlap_total_s": float(result["overlap_total"]),
                    "overlap_longest_s": float(result["overlap_longest"]),
                    "mid_transitions": int(result["mid_transitions"]),
                    "tip_transitions": int(result["tip_transitions"]),
                    "target_mid": bool(result["target_mid"]),
                    "target_tip": bool(result["target_tip"]),
                    "target_contacts": int(bool(result["target_mid"]))
                    + int(bool(result["target_tip"])),
                    "target_mid_gap_mm": float(result["target_mid_gap_mm"]),
                    "target_tip_gap_mm": float(result["target_tip_gap_mm"]),
                    "target_gap_abs_sum_mm": np.nan,
                    "tip_force_median_N": tip_force_med,
                    "tip_force_max_N": safe_float(result["max_tip_force"]),
                    "force_error_N": float(force_error),
                    "final_mid_gap_mm": float(result["final_mid_gap_mm"]),
                    "final_tip_gap_mm": float(result["final_tip_gap_mm"]),
                    "max_penetration_mm": np.nan,
                    "xml": str(candidate_xml),
                }

                row["target_gap_abs_sum_mm"] = target_gap_abs_sum(row)
                row["max_penetration_mm"] = max_penetration_mm(row)

                rows.append(row)

                print(
                    f"    target pose MCP/PIP/DIP = "
                    f"{row['mcp_deg']:.3f} / "
                    f"{row['pip_deg']:.3f} / "
                    f"{row['dip_deg']:.3f} deg"
                )
                print(f"    angle RMSE = {row['rmse_deg']:.3f} deg")
                print(
                    f"    first middle/tip = "
                    f"{row['first_mid_tick']:.1f} / "
                    f"{row['first_tip_tick']:.1f} ticks; "
                    f"middle->tip={row['sequence_ok']}"
                )
                print(
                    f"    overlap total/longest = "
                    f"{row['overlap_total_s']:.3f} / "
                    f"{row['overlap_longest_s']:.3f} s"
                )
                print(
                    f"    transitions middle/tip = "
                    f"{row['mid_transitions']} / {row['tip_transitions']}"
                )
                print(
                    f"    target contact middle/tip = "
                    f"{row['target_mid']} / {row['target_tip']}"
                )
                print(
                    f"    target gaps middle/tip = "
                    f"{row['target_mid_gap_mm']:+.3f} / "
                    f"{row['target_tip_gap_mm']:+.3f} mm"
                )
                print(
                    f"    tip force median/max = "
                    f"{row['tip_force_median_N']:.4f} / "
                    f"{row['tip_force_max_N']:.4f} N"
                )
                print(
                    f"    final gaps middle/tip = "
                    f"{row['final_mid_gap_mm']:+.3f} / "
                    f"{row['final_tip_gap_mm']:+.3f} mm"
                )
                print(
                    f"    max target/final penetration = "
                    f"{row['max_penetration_mm']:.3f} mm"
                )

    finally:
        b2.v35.CONTACT_MU = old_mu
        b2.v35.CONTACT_K = old_contact_k
        contact_replay.CONTACT_DAMPING = old_damping

    # Mechanics-first ordering, explicitly NOT a weighted RMSE score.
    ranked = sorted(rows, key=mechanics_rank_key)

    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    print()
    print("=" * 190)
    print("V35G3 — ALL 20 CANDIDATES (GRID ORDER)")
    print("=" * 190)
    print()
    print(
        " PIP_k   DIP_k    MCP     PIP     DIP    RMSE  "
        "E_mid E_tip seq both  ovLong midTr tipTr "
        "Tmid Ttip TgapMid TgapTip Fmed  maxPen"
    )
    print(
        "[Nm/r]  [Nm/r]   [deg]   [deg]   [deg]   [deg] "
        "[tick][tick]          [s]                  "
        "[mm]    [mm]    [N]    [mm]"
    )
    print("-" * 190)

    for row in rows:
        print(
            f"{row['pip_k']:7.4f} "
            f"{row['dip_k']:7.4f} "
            f"{row['mcp_deg']:7.2f} "
            f"{row['pip_deg']:7.2f} "
            f"{row['dip_deg']:7.2f} "
            f"{row['rmse_deg']:7.3f} "
            f"{row['first_mid_tick']:5.0f} "
            f"{row['first_tip_tick']:5.0f} "
            f"{str(row['sequence_ok']):>4s} "
            f"{str(row['ever_both']):>4s} "
            f"{row['overlap_longest_s']:7.3f} "
            f"{row['mid_transitions']:5d} "
            f"{row['tip_transitions']:5d} "
            f"{str(row['target_mid']):>4s} "
            f"{str(row['target_tip']):>4s} "
            f"{row['target_mid_gap_mm']:+8.3f} "
            f"{row['target_tip_gap_mm']:+8.3f} "
            f"{row['tip_force_median_N']:6.3f} "
            f"{row['max_penetration_mm']:7.3f}"
        )

    print_matrix(
        rows,
        "overlap_longest_s",
        "V35G3 — LONGEST SIMULTANEOUS MIDDLE+TIP OVERLAP [s]",
    )

    print_matrix(
        rows,
        "rmse_deg",
        "V35G3 — H1 TARGET-POSE RMSE [deg]",
    )

    print()
    print("=" * 190)
    print("V35G3 — MECHANICS-FIRST SHORTLIST")
    print("=" * 190)
    print()
    print(
        "Rank PIP_k   DIP_k   seq both Tcontacts  ovLong  "
        "midTr tipTr TgapAbs maxPen F_err  RMSE  pose(M/P/D)"
    )
    print("-" * 190)

    for row in ranked[: min(args.top, len(ranked))]:
        force_error = row["force_error_N"]
        force_error_text = (
            f"{force_error:6.3f}"
            if np.isfinite(force_error)
            else "   inf"
        )

        print(
            f"{row['rank']:4d} "
            f"{row['pip_k']:7.4f} "
            f"{row['dip_k']:7.4f} "
            f"{str(row['sequence_ok']):>4s} "
            f"{str(row['ever_both']):>4s} "
            f"{row['target_contacts']:9d} "
            f"{row['overlap_longest_s']:7.3f} "
            f"{row['mid_transitions']:5d} "
            f"{row['tip_transitions']:5d} "
            f"{row['target_gap_abs_sum_mm']:7.3f} "
            f"{row['max_penetration_mm']:6.3f} "
            f"{force_error_text} "
            f"{row['rmse_deg']:6.3f} "
            f"{row['mcp_deg']:.2f}/"
            f"{row['pip_deg']:.2f}/"
            f"{row['dip_deg']:.2f}"
        )

    csv_path = save_csv(args.csv, ranked)

    print()
    print(f"Saved ranked CSV:\n  {csv_path}")
    print()
    print("NEXT DECISION:")
    print(
        "  Open ONLY the ~3 most informative candidates in native MuJoCo GUI."
    )
    print(
        "  Accept a candidate only if the visible mechanism is:"
    )
    print(
        "      middle contacts bottle"
        " -> middle remains constrained"
        " -> distal wraps smoothly"
    )
    print(
        "  Do not accept a candidate merely because RMSE is lower."
    )
    print(
        "  If no local PIP x DIP combination produces that mechanism, STOP "
        "spring-only tuning and move to missing passive mechanics "
        "(friction/damping/hysteresis/compliance/preload) rather than H2."
    )
    print()
    print("=" * 190)
    print("V35G3 COMPLETE")
    print("=" * 190)


if __name__ == "__main__":
    main()
