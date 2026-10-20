#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import fit_index_passive_contact_v35b2 as b2
import replay_index_zero_margin_v35e2a as e2a
import scan_index_zero_margin_damping_v35e2b as e2b


# ================================================================
# V35G4 — EFFECTIVE ENCODER->TENDON DISPLACEMENT SCALE SCAN
#
# PURPOSE
# -------
# Re-identify the effective transmission scale after the contact
# model and CAD tendon routing have been improved.
#
# IMPORTANT:
# METERS_PER_TICK is NOT treated as a measured mechanical constant.
# It is an identified effective transmission parameter.
#
# VARIABLE ONLY
# -------------
#   encoder -> tendon displacement scale [m/tick]
#
# FROZEN
# ------
#   V35B2 selected passive model
#   CAD tendon
#   tendon stiffness = 2100 N/m
#   H1 bottle/support
#   measured H1 encoder replay
#   margin/gap = 0
#   contact mu = 0.05
#   contact stiffness = 1000
#   contact damping = 20
#
# Goal:
# Determine whether additional effective tendon shortening makes
# the hardware-like contact sequence emerge naturally:
#
# MCP motion
# -> middle contacts
# -> PIP continues wrapping
# -> middle becomes constrained
# -> additional tendon displacement goes to DIP
# -> distal wraps
# ================================================================


SCALES_M_PER_TICK = [
    4.80e-6,
    5.10e-6,
    5.30e-6,
    5.50e-6,
    5.70e-6,
    6.00e-6,
]

TENDON_K = 2100.0

CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

HW_POSE = np.asarray(e2a.HW_POSE, dtype=float)
HW_FORCE = float(e2a.HW_FORCE_MEDIAN)

CONTACT_GEOMS = {
    "collision_idx_prox",
    "collision_idx_mid_mesh",
    "collision_idx_tip",
}

FIXTURE_NAME = "index_contact_fixture"

CSV_FIELDS = [
    "scale_m_per_tick",
    "scale_um_per_tick",
    "mcp_deg",
    "pip_deg",
    "dip_deg",
    "mcp_error_deg",
    "pip_error_deg",
    "dip_error_deg",
    "rmse_deg",
    "first_mid_tick",
    "first_tip_tick",
    "sequence_ok",
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
]


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return float(default)


def finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def make_zero_margin_model(out_dir: Path) -> Path:
    """
    Build the zero-margin model directly inside the experiment folder.

    This avoids scattering newly generated XML files elsewhere.
    """

    source = Path(b2.SELECTED_MODEL).expanduser().resolve()

    if not source.exists():
        raise RuntimeError(
            f"Missing V35B2 selected source model:\n{source}"
        )

    target = out_dir / "v35g4_zero_margin_source.xml"

    tree = ET.parse(source)
    root = tree.getroot()

    # IMPORTANT:
    # This XML is written into the experiment output directory.
    # The original model uses relative mesh paths such as ../meshes/...
    # which only work while the XML lives inside the package mjcf folder.
    #
    # Convert the mesh directory to an absolute path so the copied XML
    # remains self-contained and loadable from the results directory.
    package_dir = source.parent.parent
    meshes_dir = (package_dir / "meshes").resolve()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)

    compiler.set("meshdir", str(meshes_dir))

    print(f"    absolute meshdir = {meshes_dir}")

    wanted = CONTACT_GEOMS | {FIXTURE_NAME}
    found = set()

    for geom in root.iter("geom"):
        name = geom.get("name")

        if name in wanted:
            geom.set("margin", "0")
            geom.set("gap", "0")
            found.add(name)

    missing = wanted - found

    if missing:
        raise RuntimeError(
            "Could not find required contact geoms:\n"
            + "\n".join(sorted(missing))
        )

    tree.write(
        target,
        encoding="utf-8",
        xml_declaration=True,
    )

    shutil.copy2(
        source,
        out_dir / "v35g4_original_selected_model.xml",
    )

    return target


def max_penetration_mm(row):
    gaps = [
        row["target_mid_gap_mm"],
        row["target_tip_gap_mm"],
        row["final_mid_gap_mm"],
        row["final_tip_gap_mm"],
    ]

    vals = [
        max(0.0, -float(g))
        for g in gaps
        if np.isfinite(g)
    ]

    return max(vals) if vals else np.nan


def save_manifest(
    out_dir: Path,
    bag: Path,
    model_xml: Path,
):
    path = out_dir / "manifest.txt"

    with path.open("w") as f:
        f.write("V35G4 — transmission scale scan\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"H1 bag:\n{bag}\n\n")
        f.write(f"zero-margin model:\n{model_xml}\n\n")

        f.write("VARIABLE:\n")
        for s in SCALES_M_PER_TICK:
            f.write(
                f"  {s:.9e} m/tick "
                f"({s * 1e6:.3f} um/tick)\n"
            )

        f.write("\nFROZEN:\n")
        f.write(f"  tendon stiffness = {TENDON_K} N/m\n")
        f.write(f"  contact mu = {CONTACT_MU}\n")
        f.write(f"  contact stiffness = {CONTACT_K}\n")
        f.write(f"  contact damping = {CONTACT_DAMPING}\n")
        f.write("  margin/gap = 0/0\n")
        f.write("  passive model = V35B2 selected\n")
        f.write("  CAD tendon = frozen\n")
        f.write("  H1 bottle/support = frozen\n")
        f.write("  measured H1 encoder replay = frozen\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home()
        / "rh8d_results"
        / "v35g4_transmission_scale",
    )

    args = parser.parse_args()

    bag = args.contact_bag.expanduser().resolve()

    if not bag.exists():
        raise RuntimeError(f"Missing H1 bag:\n{bag}")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = b2.v30.v24.base.contact_replay

    model_xml = make_zero_margin_model(out_dir)

    save_manifest(
        out_dir,
        bag,
        model_xml,
    )

    old_scale = float(b2.v35.METERS_PER_TICK)
    old_tendon_k = float(b2.v35.TENDON_K)
    old_mu = float(b2.v35.CONTACT_MU)
    old_contact_k = float(b2.v35.CONTACT_K)
    old_damping = float(contact_replay.CONTACT_DAMPING)

    rows = []

    try:
        b2.v35.TENDON_K = TENDON_K
        b2.v35.CONTACT_MU = CONTACT_MU
        b2.v35.CONTACT_K = CONTACT_K
        contact_replay.CONTACT_DAMPING = CONTACT_DAMPING

        print()
        print("=" * 170)
        print(
            "V35G4 — EFFECTIVE ENCODER->TENDON "
            "TRANSMISSION-SCALE SCAN"
        )
        print("=" * 170)

        print()
        print(f"H1 bag:\n  {bag}")
        print(f"Output folder:\n  {out_dir}")
        print(f"Zero-margin model:\n  {model_xml}")

        print()
        print("FROZEN:")
        print("  passive model      = V35B2 selected")
        print("  CAD tendon         = frozen")
        print(f"  tendon stiffness   = {TENDON_K:.1f} N/m")
        print("  H1 bottle/support  = frozen")
        print("  measured encoder   = replayed")
        print("  margin / gap       = 0 / 0")
        print(f"  contact mu         = {CONTACT_MU}")
        print(f"  contact stiffness  = {CONTACT_K}")
        print(f"  contact damping    = {CONTACT_DAMPING}")

        print()
        print("VARIABLE:")
        print(
            "  scale [um/tick] = "
            + str(
                [
                    round(v * 1e6, 3)
                    for v in SCALES_M_PER_TICK
                ]
            )
        )

        for i, scale in enumerate(
            SCALES_M_PER_TICK,
            start=1,
        ):
            print()
            print("-" * 170)
            print(
                f"[{i:02d}/{len(SCALES_M_PER_TICK):02d}] "
                f"scale = {scale * 1e6:.3f} um/tick"
            )
            print("-" * 170)

            b2.v35.METERS_PER_TICK = float(scale)

            result = e2b.run_one(
                CONTACT_DAMPING,
                model_xml,
                bag,
                contact_replay,
            )

            pose = np.asarray(
                result["pose"],
                dtype=float,
            )

            err = pose - HW_POSE

            first_mid = safe_float(
                result["first_mid_encoder"]
            )

            first_tip = safe_float(
                result["first_tip_encoder"]
            )

            sequence_ok = bool(
                finite(first_mid)
                and finite(first_tip)
                and first_mid < first_tip
            )

            target_mid_gap = float(
                result["target_mid_gap_mm"]
            )

            target_tip_gap = float(
                result["target_tip_gap_mm"]
            )

            force_med = safe_float(
                result["positive_tip_median"]
            )

            force_error = (
                abs(force_med - HW_FORCE)
                if np.isfinite(force_med)
                else np.nan
            )

            row = {
                "scale_m_per_tick": float(scale),
                "scale_um_per_tick": float(scale * 1e6),

                "mcp_deg": float(pose[0]),
                "pip_deg": float(pose[1]),
                "dip_deg": float(pose[2]),

                "mcp_error_deg": float(err[0]),
                "pip_error_deg": float(err[1]),
                "dip_error_deg": float(err[2]),

                "rmse_deg": float(result["rmse"]),

                "first_mid_tick": first_mid,
                "first_tip_tick": first_tip,
                "sequence_ok": sequence_ok,

                "overlap_total_s": float(
                    result["overlap_total"]
                ),

                "overlap_longest_s": float(
                    result["overlap_longest"]
                ),

                "mid_transitions": int(
                    result["mid_transitions"]
                ),

                "tip_transitions": int(
                    result["tip_transitions"]
                ),

                "target_mid": bool(
                    result["target_mid"]
                ),

                "target_tip": bool(
                    result["target_tip"]
                ),

                "target_contacts":
                    int(bool(result["target_mid"]))
                    + int(bool(result["target_tip"])),

                "target_mid_gap_mm":
                    target_mid_gap,

                "target_tip_gap_mm":
                    target_tip_gap,

                "target_gap_abs_sum_mm":
                    abs(target_mid_gap)
                    + abs(target_tip_gap),

                "tip_force_median_N":
                    force_med,

                "tip_force_max_N":
                    safe_float(
                        result["max_tip_force"]
                    ),

                "force_error_N":
                    force_error,

                "final_mid_gap_mm":
                    float(
                        result["final_mid_gap_mm"]
                    ),

                "final_tip_gap_mm":
                    float(
                        result["final_tip_gap_mm"]
                    ),

                "max_penetration_mm":
                    np.nan,
            }

            row["max_penetration_mm"] = (
                max_penetration_mm(row)
            )

            rows.append(row)

            print(
                "    target pose MCP/PIP/DIP = "
                f"{pose[0]:.3f} / "
                f"{pose[1]:.3f} / "
                f"{pose[2]:.3f} deg"
            )

            print(
                "    errors vs HW             = "
                f"{err[0]:+.3f} / "
                f"{err[1]:+.3f} / "
                f"{err[2]:+.3f} deg"
            )

            print(
                f"    angle RMSE              = "
                f"{row['rmse_deg']:.3f} deg"
            )

            print(
                "    first middle/tip         = "
                f"{first_mid:.1f} / "
                f"{first_tip:.1f} ticks"
            )

            print(
                f"    middle->tip sequence     = "
                f"{sequence_ok}"
            )

            print(
                "    target contact mid/tip   = "
                f"{row['target_mid']} / "
                f"{row['target_tip']}"
            )

            print(
                "    target gaps mid/tip      = "
                f"{target_mid_gap:+.3f} / "
                f"{target_tip_gap:+.3f} mm"
            )

            print(
                "    overlap total/longest    = "
                f"{row['overlap_total_s']:.3f} / "
                f"{row['overlap_longest_s']:.3f} s"
            )

            print(
                "    transitions mid/tip      = "
                f"{row['mid_transitions']} / "
                f"{row['tip_transitions']}"
            )

            print(
                "    tip force median/max     = "
                f"{row['tip_force_median_N']:.4f} / "
                f"{row['tip_force_max_N']:.4f} N"
            )

            print(
                "    final gaps mid/tip       = "
                f"{row['final_mid_gap_mm']:+.3f} / "
                f"{row['final_tip_gap_mm']:+.3f} mm"
            )

            print(
                "    max target/final pen.    = "
                f"{row['max_penetration_mm']:.3f} mm"
            )

    finally:
        b2.v35.METERS_PER_TICK = old_scale
        b2.v35.TENDON_K = old_tendon_k
        b2.v35.CONTACT_MU = old_mu
        b2.v35.CONTACT_K = old_contact_k
        contact_replay.CONTACT_DAMPING = old_damping

    csv_path = out_dir / "v35g4_results.csv"

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print()
    print("=" * 170)
    print("V35G4 — SCALE TREND SUMMARY")
    print("=" * 170)

    print()
    print(
        "scale   MCP     PIP     DIP    "
        "eMCP   ePIP   eDIP   RMSE  "
        "Tmid Ttip  Gmid   Gtip   "
        "ovLong midTr tipTr maxPen"
    )

    print(
        "[um/t] [deg]   [deg]   [deg]  "
        "[deg]  [deg]  [deg]  [deg]       "
        "      [mm]   [mm]    [s]             [mm]"
    )

    print("-" * 170)

    for row in rows:
        print(
            f"{row['scale_um_per_tick']:6.2f} "
            f"{row['mcp_deg']:7.2f} "
            f"{row['pip_deg']:7.2f} "
            f"{row['dip_deg']:7.2f} "
            f"{row['mcp_error_deg']:+6.2f} "
            f"{row['pip_error_deg']:+6.2f} "
            f"{row['dip_error_deg']:+6.2f} "
            f"{row['rmse_deg']:6.3f} "
            f"{str(row['target_mid']):>5s} "
            f"{str(row['target_tip']):>5s} "
            f"{row['target_mid_gap_mm']:+7.3f} "
            f"{row['target_tip_gap_mm']:+7.3f} "
            f"{row['overlap_longest_s']:7.3f} "
            f"{row['mid_transitions']:5d} "
            f"{row['tip_transitions']:5d} "
            f"{row['max_penetration_mm']:7.3f}"
        )

    print()
    print("Interpretation:")
    print(
        "  1. If increasing scale moves BOTH PIP and DIP "
        "toward hardware while preserving physical contact, "
        "the effective transmission scale must be re-identified."
    )
    print(
        "  2. If MCP simply overshoots or contact becomes violent "
        "without stable wrapping, scale is NOT the missing mechanism."
    )
    print(
        "  3. Do NOT select from RMSE alone."
    )
    print(
        "  4. Desired mechanism remains: middle contacts -> "
        "PIP wraps -> middle becomes constrained -> DIP wraps."
    )

    print()
    print("Generated files:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p}")

    print()
    print("=" * 170)
    print("V35G4 COMPLETE")
    print("=" * 170)


if __name__ == "__main__":
    main()
