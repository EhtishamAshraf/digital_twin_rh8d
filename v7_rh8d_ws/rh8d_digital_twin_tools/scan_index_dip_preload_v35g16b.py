#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import scan_index_contact_friction_v35g7 as g7
import scan_index_dip_stiffness_v35g15b as g15b


# ================================================================
# G16B — DIP SPRING PRELOAD / SPRINGREF TEST
#
# Real hardware:
#   passive spring exists across every finger joint.
#
# Current model:
#   MCP springref = -0.10
#   PIP springref = -0.10
#   DIP springref = 0
#
# G15B showed:
#   increasing DIP stiffness delays premature DIP flexion,
#   BUT subsequently prevents correct DIP takeover.
#
# G16B separates:
#   - initial spring preload
#   - incremental spring stiffness
#
# All candidates preserve approximately the SAME passive DIP spring
# torque at the hardware target DIP angle (9.327 deg).
#
# NO:
#   contact-triggered switching
#   joint locking
#   prescribed trajectories
#   tendon topology changes
#   collision changes
# ================================================================


SCALE = 5.70e-6
TENDON_K = 2100.0

CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

DIP_JOINT = "Index_Distal--Index_Middle"

BASE_K = 0.0475
TARGET_DIP_DEG = 9.3270
TARGET_DIP_RAD = math.radians(TARGET_DIP_DEG)

TARGET_PASSIVE_TORQUE = (
    BASE_K
    *
    TARGET_DIP_RAD
)


SPRINGREFS = [
    0.00,
    -0.05,
    -0.10,
    -0.15,
    -0.20,
    -0.30,
]


def candidate_k(
    springref,
):
    return (
        TARGET_PASSIVE_TORQUE
        /
        (
            TARGET_DIP_RAD
            -
            springref
        )
    )


def find_joint(
    root,
    name,
):

    for joint in root.iter("joint"):

        if joint.get("name") == name:
            return joint

    raise RuntimeError(
        f"Joint not found: {name}"
    )


def make_variant(
    baseline_xml,
    out_dir,
    springref,
):

    k = candidate_k(
        springref
    )

    tree = ET.parse(
        baseline_xml
    )

    root = tree.getroot()

    dip = find_joint(
        root,
        DIP_JOINT,
    )

    dip.set(
        "stiffness",
        f"{k:.12g}",
    )

    dip.set(
        "springref",
        f"{springref:.12g}",
    )

    tag = (
        f"ref_{springref:+.3f}"
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )

    path = (
        out_dir
        /
        f"model_{tag}.xml"
    )

    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )

    return (
        path,
        k,
    )


def mechanics_score(
    r,
):

    score = 0.0

    # Contact loss / rebound is our primary failure.
    score += (
        600.0
        *
        max(
            0,
            r["middle_episodes"] - 1,
        )
    )

    score += (
        600.0
        *
        max(
            0,
            r["tip_episodes"] - 1,
        )
    )

    if not r["target_mid_contact"]:
        score += 400.0

    if not r["target_tip_contact"]:
        score += 400.0

    dpip = r[
        "delta_pip_middle_to_tip_deg"
    ]

    ddip = r[
        "delta_dip_middle_to_tip_deg"
    ]

    delay = r[
        "middle_to_tip_delay_s"
    ]

    # Want meaningful PIP wrapping after middle contact.
    if np.isfinite(dpip):

        score += (
            100.0
            *
            max(
                0.0,
                1.0 - dpip,
            )
        )

    else:
        score += 300.0

    # Most important:
    # DIP should ultimately flex, not extend.
    if np.isfinite(ddip):

        score += (
            250.0
            *
            max(
                0.0,
                0.15 - ddip,
            )
        )

    else:
        score += 300.0

    # Middle->tip should not be almost simultaneous.
    if np.isfinite(delay):

        score += (
            150.0
            *
            max(
                0.0,
                0.10 - delay,
            )
        )

    else:
        score += 200.0

    # RMSE still matters, after mechanics.
    score += (
        10.0
        *
        r["rmse_deg"]
    )

    return float(
        score
    )


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
            /
            "rh8d_results"
            /
            "v35g16b_dip_preload"
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
        /
        Path(__file__).name,
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

    old_tendon_k = float(
        g7.b2.v35.TENDON_K
    )

    old_mu = float(
        g7.b2.v35.CONTACT_MU
    )

    old_contact_k = float(
        g7.b2.v35.CONTACT_K
    )

    old_contact_damping = float(
        contact_replay.CONTACT_DAMPING
    )

    rows = []
    trajectories = {}

    try:

        g7.b2.v35.METERS_PER_TICK = (
            SCALE
        )

        g7.b2.v35.TENDON_K = (
            TENDON_K
        )

        g7.b2.v35.CONTACT_MU = (
            CONTACT_MU
        )

        g7.b2.v35.CONTACT_K = (
            CONTACT_K
        )

        contact_replay.CONTACT_DAMPING = (
            CONTACT_DAMPING
        )

        print()
        print("=" * 150)

        print(
            "G16B — DIP SPRING PRELOAD TEST"
        )

        print("=" * 150)

        print()
        print(
            f"hardware target DIP = "
            f"{TARGET_DIP_DEG:.3f} deg"
        )

        print(
            f"iso-target passive torque = "
            f"{TARGET_PASSIVE_TORQUE:.6f} Nm"
        )

        print()

        print(
            "springref    stiffness      "
            "open_preload"
        )

        for ref in SPRINGREFS:

            k = candidate_k(
                ref
            )

            preload = (
                k
                *
                abs(ref)
            )

            print(
                f"{ref:+.3f} rad   "
                f"{k:.6f} Nm/rad   "
                f"{preload:.6f} Nm"
            )

        for ref in SPRINGREFS:

            (
                xml,
                k,
            ) = make_variant(
                baseline_xml,
                out_dir,
                ref,
            )

            preload = (
                k
                *
                abs(ref)
            )

            print()
            print("-" * 150)

            print(
                f"DIP springref = "
                f"{ref:+.3f} rad"
            )

            print(
                f"DIP stiffness = "
                f"{k:.6f} Nm/rad"
            )

            print(
                f"open preload = "
                f"{preload:.6f} Nm"
            )

            print("-" * 150)

            (
                result,
                trace,
                qpos,
                times,
            ) = g7.run_scale(
                SCALE,
                CONTACT_MU,
                xml,
                bag,
                contact_replay,
            )

            row = dict(
                result
            )

            row.update(
                g15b.mechanics(
                    trace
                )
            )

            row[
                "dip_springref_rad"
            ] = ref

            row[
                "dip_stiffness"
            ] = k

            row[
                "dip_open_preload_Nm"
            ] = preload

            row[
                "mechanics_score_g16b"
            ] = mechanics_score(
                row
            )

            rows.append(
                row
            )

            trajectories[
                ref
            ] = (
                xml,
                qpos,
                times,
            )

            tag = (
                f"{ref:+.3f}"
                .replace("+", "p")
                .replace("-", "m")
                .replace(".", "p")
            )

            g7.save_trace_csv(
                out_dir
                /
                f"springref_{tag}_timeseries.csv",
                trace,
            )

            np.savez_compressed(
                out_dir
                /
                f"springref_{tag}_trajectory.npz",
                times=times,
                qpos=qpos,
            )

            print(
                "target pose = "
                f"{row['target_mcp_deg']:.3f} / "
                f"{row['target_pip_deg']:.3f} / "
                f"{row['target_dip_deg']:.3f}"
            )

            print(
                f"RMSE = "
                f"{row['rmse_deg']:.3f} deg"
            )

            print(
                "episodes M/T = "
                f"{row['middle_episodes']} / "
                f"{row['tip_episodes']}"
            )

            print(
                "first-middle pose = "
                f"{row['mcp_first_middle_deg']:.3f} / "
                f"{row['pip_first_middle_deg']:.3f} / "
                f"{row['dip_first_middle_deg']:.3f}"
            )

            print(
                f"middle->tip delay = "
                f"{row['middle_to_tip_delay_s']:.4f} s"
            )

            print(
                "middle->tip dPIP/dDIP = "
                f"{row['delta_pip_middle_to_tip_deg']:+.3f} / "
                f"{row['delta_dip_middle_to_tip_deg']:+.3f} deg"
            )

            print(
                f"mechanics score = "
                f"{row['mechanics_score_g16b']:.2f}"
            )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

        g7.b2.v35.TENDON_K = (
            old_tendon_k
        )

        g7.b2.v35.CONTACT_MU = (
            old_mu
        )

        g7.b2.v35.CONTACT_K = (
            old_contact_k
        )

        contact_replay.CONTACT_DAMPING = (
            old_contact_damping
        )

    with (
        out_dir
        /
        "g16b_summary.csv"
    ).open(
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
        writer.writerows(
            rows
        )

    ranked = sorted(
        rows,
        key=lambda r:
            r[
                "mechanics_score_g16b"
            ],
    )

    print()
    print("=" * 170)
    print(
        "G16B — MECHANICS-FIRST SUMMARY"
    )
    print("=" * 170)

    print(
        "ref_rad   K       preload   RMSE "
        "epM epT DIP@M  delay   dPIP   dDIP "
        "targetM targetT score"
    )

    print("-" * 170)

    for r in ranked:

        print(
            f"{r['dip_springref_rad']:+7.3f} "
            f"{r['dip_stiffness']:7.4f} "
            f"{r['dip_open_preload_Nm']:8.5f} "
            f"{r['rmse_deg']:5.3f} "
            f"{r['middle_episodes']:3d} "
            f"{r['tip_episodes']:3d} "
            f"{r['dip_first_middle_deg']:6.2f} "
            f"{r['middle_to_tip_delay_s']:7.4f} "
            f"{r['delta_pip_middle_to_tip_deg']:+6.2f} "
            f"{r['delta_dip_middle_to_tip_deg']:+6.2f} "
            f"{str(r['target_mid_contact']):>7s} "
            f"{str(r['target_tip_contact']):>7s} "
            f"{r['mechanics_score_g16b']:7.2f}"
        )

    # ------------------------------------------------------------
    # GUI:
    # always show baseline + three best NON-baseline candidates.
    # ------------------------------------------------------------

    if not args.no_gui:

        baseline = next(
            r
            for r in rows
            if abs(
                r[
                    "dip_springref_rad"
                ]
            ) < 1e-12
        )

        nonbaseline = [
            r
            for r in ranked
            if abs(
                r[
                    "dip_springref_rad"
                ]
            ) > 1e-12
        ]

        visual = (
            [baseline]
            +
            nonbaseline[:3]
        )

        print()
        print("=" * 120)
        print(
            "MANDATORY GUI VALIDATION"
        )
        print("=" * 120)

        print(
            "Baseline first, then the three best preload candidates."
        )

        for i, row in enumerate(
            visual,
            start=1,
        ):

            ref = float(
                row[
                    "dip_springref_rad"
                ]
            )

            (
                xml,
                qpos,
                times,
            ) = trajectories[
                ref
            ]

            print()
            print("#" * 130)

            print(
                f"GUI {i}/{len(visual)}"
            )

            print(
                f"DIP springref = "
                f"{ref:+.3f} rad"
            )

            print(
                f"DIP stiffness = "
                f"{row['dip_stiffness']:.6f}"
            )

            print(
                "episodes M/T = "
                f"{row['middle_episodes']} / "
                f"{row['tip_episodes']}"
            )

            print(
                "middle->tip dPIP/dDIP = "
                f"{row['delta_pip_middle_to_tip_deg']:+.3f} / "
                f"{row['delta_dip_middle_to_tip_deg']:+.3f}"
            )

            print(
                f"RMSE = "
                f"{row['rmse_deg']:.3f}"
            )

            print("#" * 130)

            g7.exact_gui_replay(
                xml,
                qpos,
                times,
                (
                    f"G16B springref "
                    f"{ref:+.3f}"
                ),
            )

    with (
        out_dir
        /
        "manifest.txt"
    ).open("w") as f:

        f.write(
            "V35G16B — DIP preload/springref experiment\n\n"
        )

        f.write(
            "Physical hypothesis:\n"
            "real RH8D has a passive spring across each joint.\n"
            "Current DIP has no spring preload while MCP/PIP use "
            "springref=-0.10 rad.\n\n"
        )

        f.write(
            "Each DIP candidate preserves the baseline passive "
            "spring torque at the hardware target DIP angle.\n"
        )

        f.write(
            "Only stiffness/springref coupling changes.\n\n"
        )

        for r in rows:

            f.write(
                f"springref={r['dip_springref_rad']:+.4f}, "
                f"k={r['dip_stiffness']:.8f}, "
                f"preload={r['dip_open_preload_Nm']:.8f}\n"
            )

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
