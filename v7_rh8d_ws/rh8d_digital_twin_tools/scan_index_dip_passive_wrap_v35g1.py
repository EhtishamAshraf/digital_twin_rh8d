#!/usr/bin/env python3

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import fit_index_passive_contact_v35b2 as b2
import replay_index_zero_margin_v35e2a as e2a
import scan_index_zero_margin_damping_v35e2b as e2b


# ================================================================
# V35G1 — DIP PASSIVE-STIFFNESS / WRAPPING SCAN
#
# Purpose:
#   Determine whether excessive DIP passive resistance is
#   preventing the hardware-like sequence:
#
#       middle contact
#           ->
#       middle remains constrained
#           ->
#       distal phalanx wraps
#           ->
#       sustained middle + tip contact
#
# VARIABLE ONLY:
#   DIP joint stiffness
#
# FROZEN:
#   CAD spatial tendon
#   MCP stiffness
#   PIP stiffness
#   PIP friction
#   tendon scale
#   tendon stiffness
#   bottle pose
#   fixed support
#   measured hardware encoder replay
#   contact margin = 0
#   contact mu = 0.05
#   direct contact stiffness = 1000
#   direct contact damping = 20
#
# NO prescribed MCP/PIP/DIP trajectories.
# NO contact-triggered transmission changes.
# ================================================================


DIP_K_VALUES = [
    0.0475,   # current V35B2 baseline
    0.0300,
    0.0200,
    0.0150,
    0.0100,
    0.0075,
    0.0050,
    0.0025,
]


CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

DIP_JOINT_NAME = (
    "Index_Distal--Index_Middle"
)


def make_candidate(
    source_xml: Path,
    dip_k: float,
) -> Path:

    source_xml = (
        source_xml
        .expanduser()
        .resolve()
    )

    tree = ET.parse(
        source_xml
    )

    root = tree.getroot()

    matches = []

    for joint in root.iter(
        "joint"
    ):

        if joint.get(
            "name"
        ) == DIP_JOINT_NAME:

            matches.append(
                joint
            )


    if len(matches) != 1:

        raise RuntimeError(
            f"Expected exactly one joint named "
            f"{DIP_JOINT_NAME!r}, "
            f"found {len(matches)}"
        )


    joint = matches[0]

    old_k = joint.get(
        "stiffness"
    )

    joint.set(
        "stiffness",
        f"{dip_k:.10g}",
    )


    tag = (
        f"{dip_k:.4f}"
        .replace(
            ".",
            "p",
        )
    )


    out = (
        source_xml.parent
        /
        (
            source_xml.stem
            +
            f"_v35g1_dipk_{tag}.xml"
        )
    )


    tree.write(
        out,
        encoding="utf-8",
        xml_declaration=True,
    )


    print(
        f"    XML DIP stiffness: "
        f"{old_k} -> {dip_k}"
    )

    return out


def finite(
    value,
):

    return bool(
        np.isfinite(
            float(value)
        )
    )


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
            f"Missing contact bag:\n{bag}"
        )


    # ------------------------------------------------------------
    # Patch/import the same replay stack used by previous tests.
    # ------------------------------------------------------------

    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
    )


    # ------------------------------------------------------------
    # This is the zero-margin model used in E2A/E2B/F1/F2.
    # ------------------------------------------------------------

    zero_margin_xml = (
        e2a.make_zero_margin_xml()
    )


    print()
    print("=" * 150)
    print(
        "V35G1 — DIP PASSIVE-STIFFNESS / WRAPPING SCAN"
    )
    print("=" * 150)

    print()
    print(
        f"Zero-margin source model:\n"
        f"{zero_margin_xml}"
    )

    print()
    print(
        "FROZEN:"
    )

    print(
        "  contact margin        = 0.000 mm"
    )

    print(
        f"  contact mu            = {CONTACT_MU}"
    )

    print(
        f"  direct stiffness      = {CONTACT_K}"
    )

    print(
        f"  direct damping        = {CONTACT_DAMPING}"
    )

    print(
        "  MCP stiffness         = unchanged V35B2"
    )

    print(
        "  PIP stiffness         = unchanged V35B2"
    )

    print(
        "  PIP friction          = unchanged V35B2"
    )

    print(
        "  CAD tendon            = unchanged"
    )

    print(
        "  bottle pose           = unchanged H1"
    )

    print(
        "  encoder replay        = measured hardware encoder"
    )

    print()
    print(
        "VARIABLE ONLY:"
    )

    print(
        f"  DIP stiffness = {DIP_K_VALUES}"
    )

    print()


    # ------------------------------------------------------------
    # Save globals.
    # ------------------------------------------------------------

    old_mu = (
        b2.v35.CONTACT_MU
    )

    old_k = (
        b2.v35.CONTACT_K
    )

    old_damping = float(
        contact_replay.CONTACT_DAMPING
    )


    b2.v35.CONTACT_MU = (
        CONTACT_MU
    )

    b2.v35.CONTACT_K = (
        CONTACT_K
    )

    contact_replay.CONTACT_DAMPING = (
        CONTACT_DAMPING
    )


    rows = []


    try:

        for index, dip_k in enumerate(
            DIP_K_VALUES,
            start=1,
        ):

            print()
            print("-" * 150)

            print(
                f"[{index:02d}/"
                f"{len(DIP_K_VALUES):02d}] "
                f"DIP stiffness = {dip_k:.5f} Nm/rad"
            )

            print("-" * 150)


            candidate_xml = (
                make_candidate(
                    zero_margin_xml,
                    dip_k,
                )
            )


            result = e2b.run_one(
                CONTACT_DAMPING,
                candidate_xml,
                bag,
                contact_replay,
            )


            pose = np.asarray(
                result[
                    "pose"
                ],
                dtype=float,
            )


            sequence_ok = bool(
                finite(
                    result[
                        "first_mid_encoder"
                    ]
                )
                and
                finite(
                    result[
                        "first_tip_encoder"
                    ]
                )
                and
                (
                    result[
                        "first_mid_encoder"
                    ]
                    <
                    result[
                        "first_tip_encoder"
                    ]
                )
            )


            row = {
                "dip_k":
                    float(
                        dip_k
                    ),

                "xml":
                    candidate_xml,

                "mcp":
                    float(
                        pose[0]
                    ),

                "pip":
                    float(
                        pose[1]
                    ),

                "dip":
                    float(
                        pose[2]
                    ),

                "rmse":
                    float(
                        result[
                            "rmse"
                        ]
                    ),

                "first_mid_encoder":
                    float(
                        result[
                            "first_mid_encoder"
                        ]
                    ),

                "first_tip_encoder":
                    float(
                        result[
                            "first_tip_encoder"
                        ]
                    ),

                "force_onset_encoder":
                    float(
                        result[
                            "force_onset_encoder"
                        ]
                    ),

                "target_mid":
                    bool(
                        result[
                            "target_mid"
                        ]
                    ),

                "target_tip":
                    bool(
                        result[
                            "target_tip"
                        ]
                    ),

                "target_mid_gap":
                    float(
                        result[
                            "target_mid_gap_mm"
                        ]
                    ),

                "target_tip_gap":
                    float(
                        result[
                            "target_tip_gap_mm"
                        ]
                    ),

                "tip_force_median":
                    float(
                        result[
                            "positive_tip_median"
                        ]
                    ),

                "tip_force_max":
                    float(
                        result[
                            "max_tip_force"
                        ]
                    ),

                "overlap_total":
                    float(
                        result[
                            "overlap_total"
                        ]
                    ),

                "overlap_longest":
                    float(
                        result[
                            "overlap_longest"
                        ]
                    ),

                "mid_transitions":
                    int(
                        result[
                            "mid_transitions"
                        ]
                    ),

                "tip_transitions":
                    int(
                        result[
                            "tip_transitions"
                        ]
                    ),

                "final_mid_gap":
                    float(
                        result[
                            "final_mid_gap_mm"
                        ]
                    ),

                "final_tip_gap":
                    float(
                        result[
                            "final_tip_gap_mm"
                        ]
                    ),

                "sequence_ok":
                    sequence_ok,
            }


            rows.append(
                row
            )


            print(
                f"    q target = "
                f"{row['mcp']:.3f} / "
                f"{row['pip']:.3f} / "
                f"{row['dip']:.3f} deg"
            )

            print(
                f"    RMSE     = "
                f"{row['rmse']:.3f} deg"
            )

            print(
                f"    first contacts: "
                f"middle={row['first_mid_encoder']:.1f}, "
                f"tip={row['first_tip_encoder']:.1f} ticks"
            )

            print(
                f"    sequence middle->tip = "
                f"{row['sequence_ok']}"
            )

            print(
                f"    hold overlap: "
                f"total={row['overlap_total']:.3f}s, "
                f"longest={row['overlap_longest']:.3f}s"
            )

            print(
                f"    transitions: "
                f"middle={row['mid_transitions']}, "
                f"tip={row['tip_transitions']}"
            )

            print(
                f"    target contact: "
                f"middle={row['target_mid']}, "
                f"tip={row['target_tip']}"
            )

            print(
                f"    target gaps: "
                f"middle={row['target_mid_gap']:+.3f} mm, "
                f"tip={row['target_tip_gap']:+.3f} mm"
            )

            print(
                f"    final gaps: "
                f"middle={row['final_mid_gap']:+.3f} mm, "
                f"tip={row['final_tip_gap']:+.3f} mm"
            )

            print(
                f"    tip force in contact: "
                f"median={row['tip_force_median']:.4f} N, "
                f"max={row['tip_force_max']:.4f} N"
            )


    finally:

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_k
        )

        contact_replay.CONTACT_DAMPING = (
            old_damping
        )


    # ============================================================
    # Final comparison.
    # ============================================================

    print()
    print("=" * 170)
    print(
        "V35G1 — COMPARISON TABLE"
    )
    print("=" * 170)

    print()

    print(
        " DIP_k    MCP     PIP     DIP    RMSE   "
        "E_mid  E_tip   overlap longest  "
        "midTr tipTr  Tmid Ttip  "
        "TgapMid TgapTip  "
        "FtipMed  finalMid finalTip"
    )

    print(
        " [Nm/r]  [deg]   [deg]   [deg]   [deg]  "
        "[tick] [tick]    [s]     [s]                "
        "      [mm]    [mm]      "
        "[N]       [mm]     [mm]"
    )

    print("-" * 170)


    for r in rows:

        print(
            f"{r['dip_k']:7.4f} "
            f"{r['mcp']:7.2f} "
            f"{r['pip']:7.2f} "
            f"{r['dip']:7.2f} "
            f"{r['rmse']:7.3f} "
            f"{r['first_mid_encoder']:6.1f} "
            f"{r['first_tip_encoder']:6.1f} "
            f"{r['overlap_total']:8.3f} "
            f"{r['overlap_longest']:7.3f} "
            f"{r['mid_transitions']:5d} "
            f"{r['tip_transitions']:5d} "
            f"{str(r['target_mid']):>5s} "
            f"{str(r['target_tip']):>5s} "
            f"{r['target_mid_gap']:+8.3f} "
            f"{r['target_tip_gap']:+8.3f} "
            f"{r['tip_force_median']:8.4f} "
            f"{r['final_mid_gap']:+8.3f} "
            f"{r['final_tip_gap']:+8.3f}"
        )


    print()
    print("=" * 170)
    print(
        "V35G1 — WHAT WE ARE LOOKING FOR"
    )
    print("=" * 170)

    print()

    print(
        "GOOD mechanical direction:"
    )

    print(
        "  middle contacts first"
    )

    print(
        "  -> middle contact is retained"
    )

    print(
        "  -> distal begins flexing/wrapping"
    )

    print(
        "  -> long continuous middle+tip overlap"
    )

    print(
        "  -> fewer contact losses / bounces"
    )

    print(
        "  -> target frame has physical contact"
    )

    print(
        "  -> no large geometric penetration"
    )

    print()

    print(
        "DO NOT choose the lowest RMSE automatically."
    )

    print()

    print(
        "If reducing DIP stiffness improves wrap retention "
        "but DIP angle becomes excessive, we will bracket "
        "the physically useful range next."
    )

    print()

    print(
        "If changing DIP stiffness does NOT improve the "
        "middle-constrained -> distal-wrap sequence, "
        "we will then test PIP passive redistribution."
    )

    print()
    print("=" * 170)
    print(
        "V35G1 COMPLETE"
    )
    print("=" * 170)


if __name__ == "__main__":
    main()
