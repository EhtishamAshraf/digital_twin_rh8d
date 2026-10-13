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
# V35G2 — PIP PASSIVE-STIFFNESS / LOAD-REDISTRIBUTION SCAN
#
# VARIABLE ONLY:
#   PIP stiffness
#
# FROZEN:
#   DIP stiffness = original V35B2 = 0.0475 Nm/rad
#   MCP passive mechanics
#   PIP frictionloss
#   CAD tendon
#   H1 bottle pose
#   measured encoder replay
#   margin = 0
#   mu = 0.05
#   direct contact stiffness = 1000
#   direct contact damping = 20
#
# Question:
#
# Does increased PIP passive resistance allow:
#
#   middle contact
#       ->
#   middle remains supported
#       ->
#   further tendon displacement goes into DIP
#       ->
#   smooth distal wrap
#
# No contact-triggered mechanics.
# No prescribed joint trajectory.
# ================================================================


PIP_K_VALUES = [
    0.0015,   # current V35B2
    0.0030,
    0.0050,
    0.0075,
    0.0100,
    0.0150,
    0.0200,
    0.0310,
]


CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

PIP_JOINT_NAME = (
    "Index_Middle--Index_Proximal"
)

DIP_JOINT_NAME = (
    "Index_Distal--Index_Middle"
)

FROZEN_DIP_K = 0.0475


def make_candidate(
    source_xml: Path,
    pip_k: float,
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

    pip_matches = []
    dip_matches = []

    for joint in root.iter(
        "joint"
    ):

        name = joint.get(
            "name"
        )

        if name == PIP_JOINT_NAME:

            pip_matches.append(
                joint
            )

        if name == DIP_JOINT_NAME:

            dip_matches.append(
                joint
            )


    if len(pip_matches) != 1:

        raise RuntimeError(
            f"Expected one PIP joint "
            f"{PIP_JOINT_NAME!r}, "
            f"found {len(pip_matches)}"
        )


    if len(dip_matches) != 1:

        raise RuntimeError(
            f"Expected one DIP joint "
            f"{DIP_JOINT_NAME!r}, "
            f"found {len(dip_matches)}"
        )


    pip_joint = pip_matches[0]
    dip_joint = dip_matches[0]

    old_pip_k = pip_joint.get(
        "stiffness"
    )

    old_dip_k = dip_joint.get(
        "stiffness"
    )


    # ONLY variable.
    pip_joint.set(
        "stiffness",
        f"{pip_k:.10g}",
    )


    # Explicitly freeze DIP back at V35B2 baseline.
    dip_joint.set(
        "stiffness",
        f"{FROZEN_DIP_K:.10g}",
    )


    tag = (
        f"{pip_k:.4f}"
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
            f"_v35g2_pipk_{tag}.xml"
        )
    )


    tree.write(
        out,
        encoding="utf-8",
        xml_declaration=True,
    )


    print(
        f"    PIP stiffness: "
        f"{old_pip_k} -> {pip_k}"
    )

    print(
        f"    DIP stiffness frozen: "
        f"{old_dip_k} -> {FROZEN_DIP_K}"
    )


    return out


def safe_number(
    x,
):

    try:

        x = float(x)

    except Exception:

        return np.nan

    return x


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


    old_mu = (
        b2.v35.CONTACT_MU
    )

    old_contact_k = (
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

        print()
        print("=" * 160)

        print(
            "V35G2 — PIP PASSIVE-STIFFNESS / "
            "LOAD-REDISTRIBUTION SCAN"
        )

        print("=" * 160)

        print()

        print(
            "FROZEN:"
        )

        print(
            "  margin             = 0 mm"
        )

        print(
            f"  contact mu         = "
            f"{CONTACT_MU}"
        )

        print(
            f"  contact stiffness  = "
            f"{CONTACT_K}"
        )

        print(
            f"  contact damping    = "
            f"{CONTACT_DAMPING}"
        )

        print(
            f"  DIP stiffness      = "
            f"{FROZEN_DIP_K}"
        )

        print(
            "  MCP mechanics      = V35B2"
        )

        print(
            "  PIP frictionloss   = V35B2"
        )

        print(
            "  CAD tendon         = frozen"
        )

        print(
            "  H1 bottle pose     = frozen"
        )

        print()

        print(
            "VARIABLE ONLY:"
        )

        print(
            f"  PIP stiffness = "
            f"{PIP_K_VALUES}"
        )

        print()


        for index, pip_k in enumerate(
            PIP_K_VALUES,
            start=1,
        ):

            print()
            print("-" * 160)

            print(
                f"[{index:02d}/"
                f"{len(PIP_K_VALUES):02d}] "
                f"PIP stiffness = "
                f"{pip_k:.5f} Nm/rad"
            )

            print("-" * 160)


            candidate_xml = (
                make_candidate(
                    zero_margin_xml,
                    pip_k,
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


            first_mid = safe_number(
                result[
                    "first_mid_encoder"
                ]
            )

            first_tip = safe_number(
                result[
                    "first_tip_encoder"
                ]
            )


            sequence_ok = bool(
                np.isfinite(
                    first_mid
                )
                and
                np.isfinite(
                    first_tip
                )
                and
                (
                    first_mid
                    <
                    first_tip
                )
            )


            row = {

                "pip_k":
                    float(
                        pip_k
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

                "first_mid":
                    first_mid,

                "first_tip":
                    first_tip,

                "force_onset":
                    safe_number(
                        result[
                            "force_onset_encoder"
                        ]
                    ),

                "sequence_ok":
                    sequence_ok,

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

                "mid_trans":
                    int(
                        result[
                            "mid_transitions"
                        ]
                    ),

                "tip_trans":
                    int(
                        result[
                            "tip_transitions"
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

                "tip_force_med":
                    safe_number(
                        result[
                            "positive_tip_median"
                        ]
                    ),

                "tip_force_max":
                    safe_number(
                        result[
                            "max_tip_force"
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
            }


            rows.append(
                row
            )


            print(
                f"    target pose = "
                f"{row['mcp']:.3f} / "
                f"{row['pip']:.3f} / "
                f"{row['dip']:.3f} deg"
            )

            print(
                f"    RMSE = "
                f"{row['rmse']:.3f} deg"
            )

            print(
                f"    first middle/tip = "
                f"{row['first_mid']:.1f} / "
                f"{row['first_tip']:.1f} ticks"
            )

            print(
                f"    sequence middle->tip = "
                f"{row['sequence_ok']}"
            )

            print(
                f"    overlap total/longest = "
                f"{row['overlap_total']:.3f} / "
                f"{row['overlap_longest']:.3f} s"
            )

            print(
                f"    transitions middle/tip = "
                f"{row['mid_trans']} / "
                f"{row['tip_trans']}"
            )

            print(
                f"    target contact middle/tip = "
                f"{row['target_mid']} / "
                f"{row['target_tip']}"
            )

            print(
                f"    target gaps middle/tip = "
                f"{row['target_mid_gap']:+.3f} / "
                f"{row['target_tip_gap']:+.3f} mm"
            )

            print(
                f"    contact tip force median/max = "
                f"{row['tip_force_med']:.4f} / "
                f"{row['tip_force_max']:.4f} N"
            )

            print(
                f"    final gaps middle/tip = "
                f"{row['final_mid_gap']:+.3f} / "
                f"{row['final_tip_gap']:+.3f} mm"
            )


    finally:

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_contact_k
        )

        contact_replay.CONTACT_DAMPING = (
            old_damping
        )


    print()
    print("=" * 175)

    print(
        "V35G2 — COMPARISON TABLE"
    )

    print("=" * 175)

    print()

    print(
        " PIP_k    MCP     PIP     DIP    RMSE   "
        "E_mid  E_tip  E_F0.1  seq   "
        "overlap longest midTr tipTr "
        "Tmid Ttip TgapMid TgapTip "
        "FtipMed finalMid finalTip"
    )

    print(
        " [Nm/r]  [deg]   [deg]   [deg]   [deg]  "
        "[tick] [tick] [tick]        "
        "[s]     [s]                  "
        "[mm]    [mm]      [N]      [mm]     [mm]"
    )

    print("-" * 175)


    for r in rows:

        print(
            f"{r['pip_k']:7.4f} "
            f"{r['mcp']:7.2f} "
            f"{r['pip']:7.2f} "
            f"{r['dip']:7.2f} "
            f"{r['rmse']:7.3f} "
            f"{r['first_mid']:6.1f} "
            f"{r['first_tip']:6.1f} "
            f"{r['force_onset']:7.1f} "
            f"{str(r['sequence_ok']):>5s} "
            f"{r['overlap_total']:7.3f} "
            f"{r['overlap_longest']:7.3f} "
            f"{r['mid_trans']:5d} "
            f"{r['tip_trans']:5d} "
            f"{str(r['target_mid']):>5s} "
            f"{str(r['target_tip']):>5s} "
            f"{r['target_mid_gap']:+8.3f} "
            f"{r['target_tip_gap']:+8.3f} "
            f"{r['tip_force_med']:8.4f} "
            f"{r['final_mid_gap']:+8.3f} "
            f"{r['final_tip_gap']:+8.3f}"
        )


    print()
    print("=" * 175)

    print(
        "V35G2 — DECISION RULE"
    )

    print("=" * 175)

    print()

    print(
        "A mechanically promising candidate should show:"
    )

    print(
        "  1. middle contacts before tip"
    )

    print(
        "  2. longer continuous simultaneous contact"
    )

    print(
        "  3. fewer middle/tip contact losses"
    )

    print(
        "  4. increasing DIP flexion without excessive DIP"
    )

    print(
        "  5. PIP no longer keeps taking all extra motion"
    )

    print(
        "  6. target/final gaps move toward retained contact"
    )

    print(
        "  7. no large penetration"
    )

    print()

    print(
        "RMSE alone does NOT select the candidate."
    )

    print()
    print("=" * 175)

    print(
        "V35G2 COMPLETE"
    )

    print("=" * 175)


if __name__ == "__main__":
    main()
