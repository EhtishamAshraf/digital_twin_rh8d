#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import fit_index_passive_contact_v35b2 as b2
import replay_index_zero_margin_v35e2a as e2a
import scan_index_zero_margin_damping_v35e2b as e2b


# ================================================================
# V35H1
#
# JOINT PIP/DIP PASSIVE SEARCH ON CURRENT V35 MODEL
#
# PRESERVED:
#   CAD spatial tendon
#   spool/encoder actuation
#   real hardware encoder replay
#   bottle registration
#   fixed support
#   zero contact margin
#   middle mesh collision
#   MCP passive mechanics
#
# VARIABLES:
#   PIP stiffness
#   DIP stiffness
#
# NO:
#   direct MCP/PIP/DIP angle commands
#   contact-triggered parameter switching
#   hard-coded hardware pose
#
# BEST RESULT OPENS AUTOMATICALLY IN MUJOCO.
# ================================================================


PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"


HW_POSE = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


HW_FORCE = 0.14563


CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0


# ------------------------------------------------
# QUICK 2-D COARSE SEARCH
#
# G1/G2 tested these directions independently.
# H1 now tests combinations.
#
# 5 x 6 = 30 simulations
# followed by 5 x 5 = 25 refinement simulations.
#
# Total ≈ 55.
# ------------------------------------------------

PIP_COARSE = [
    0.0015,
    0.0035,
    0.0075,
    0.0150,
    0.0310,
]


DIP_COARSE = [
    0.0050,
    0.0100,
    0.0200,
    0.0300,
    0.0475,
    0.0600,
]


def fnum(
    x,
    default=np.nan,
):

    try:

        return float(
            x
        )

    except Exception:

        return float(
            default
        )


def patch_candidate(
    source,
    out,
    pip_k,
    dip_k,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    found = {
        PIP: 0,
        DIP: 0,
    }


    for joint in root.iter(
        "joint"
    ):

        name = joint.get(
            "name"
        )

        if name == PIP:

            joint.set(
                "stiffness",
                f"{pip_k:.12g}",
            )

            found[PIP] += 1


        elif name == DIP:

            joint.set(
                "stiffness",
                f"{dip_k:.12g}",
            )

            found[DIP] += 1


    if (
        found[PIP] != 1
        or
        found[DIP] != 1
    ):

        raise RuntimeError(
            f"Joint lookup failed: {found}"
        )


    tree.write(
        out,
        encoding="utf-8",
        xml_declaration=True,
    )


def candidate_score(
    result,
):

    # ------------------------------------------------
    # Angle agreement remains primary.
    # ------------------------------------------------

    score = float(
        result["rmse"]
    )


    # ------------------------------------------------
    # Correct mechanical sequence:
    #
    # middle contact must occur before tip contact.
    # ------------------------------------------------

    first_mid = fnum(
        result.get(
            "first_mid_encoder"
        )
    )

    first_tip = fnum(
        result.get(
            "first_tip_encoder"
        )
    )


    if not (
        np.isfinite(
            first_mid
        )
        and
        np.isfinite(
            first_tip
        )
        and
        first_mid
        <=
        first_tip
    ):

        score += 8.0


    # ------------------------------------------------
    # Sustained simultaneous wrapping.
    # ------------------------------------------------

    overlap = fnum(
        result.get(
            "overlap_longest"
        ),
        0.0,
    )


    if overlap < 0.05:

        score += (
            6.0
            *
            (
                0.05
                -
                max(
                    0.0,
                    overlap,
                )
            )
            /
            0.05
        )

    elif overlap < 0.20:

        score += (
            1.5
            *
            (
                0.20
                -
                overlap
            )
            /
            0.15
        )


    # ------------------------------------------------
    # At video-aligned target both surfaces should
    # physically be contacting the object.
    # ------------------------------------------------

    if not bool(
        result.get(
            "target_mid",
            False,
        )
    ):

        score += 3.0


    if not bool(
        result.get(
            "target_tip",
            False,
        )
    ):

        score += 4.0


    # ------------------------------------------------
    # Large penetration is visually/physically wrong.
    # ------------------------------------------------

    for key in (
        "target_mid_gap_mm",
        "target_tip_gap_mm",
    ):

        gap = fnum(
            result.get(
                key
            )
        )


        if (
            np.isfinite(
                gap
            )
            and
            gap < -0.35
        ):

            score += min(
                8.0,
                4.0
                *
                abs(
                    gap
                    +
                    0.35
                ),
            )


    # ------------------------------------------------
    # Force agreement is secondary because the bottle
    # and FTS/video alignment have uncertainty.
    # ------------------------------------------------

    force = fnum(
        result.get(
            "positive_tip_median"
        )
    )


    if (
        np.isfinite(
            force
        )
        and
        force > 0
    ):

        relative_error = (
            abs(
                force
                -
                HW_FORCE
            )
            /
            HW_FORCE
        )

        score += min(
            3.0,
            0.75
            *
            relative_error,
        )

    else:

        score += 2.0


    # ------------------------------------------------
    # Contact chatter is visually wrong.
    # ------------------------------------------------

    mid_trans = int(
        result.get(
            "mid_transitions",
            0,
        )
    )

    tip_trans = int(
        result.get(
            "tip_transitions",
            0,
        )
    )


    score += (
        0.08
        *
        (
            mid_trans
            +
            tip_trans
        )
    )


    return float(
        score
    )


def run_one(
    model_xml,
    bag,
    contact_replay,
):

    result = e2b.run_one(
        CONTACT_DAMPING,
        model_xml,
        bag,
        contact_replay,
    )


    result = dict(
        result
    )


    result["score"] = (
        candidate_score(
            result
        )
    )


    return result


def neighbourhood(
    best,
):

    values = sorted(
        {
            max(
                1e-5,
                best * scale,
            )
            for scale in (
                0.60,
                0.80,
                1.00,
                1.20,
                1.40,
            )
        }
    )

    return values


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    parser.add_argument(
        "--keep-all",
        action="store_true",
    )

    parser.add_argument(
        "--no-viewer",
        action="store_true",
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


    # ------------------------------------------------
    # Same replay stack as latest V35 tests.
    # ------------------------------------------------

    b2.v30.v24.ab.patch_replay_modules()


    contact_replay = (
        b2.v30.v24.base.contact_replay
    )


    # ------------------------------------------------
    # Regenerate exact latest zero-margin source model.
    # ------------------------------------------------

    source = (
        e2a.make_zero_margin_xml()
        .resolve()
    )


    # Candidate XMLs MUST stay in the same mjcf directory as the source.
    # The model uses meshdir="../meshes", so moving XMLs into a subdirectory
    # breaks all relative mesh paths.
    outdir = source.parent

    outdir.mkdir(
        exist_ok=True
    )


    summary = (
        outdir
        /
        "v35h1_summary.csv"
    )


    best_xml = (
        source.parent
        /
        "ur5e_rh8d_index_cad_spatial_v35h1_best.xml"
    )


    # ------------------------------------------------
    # Freeze current contact model.
    # ------------------------------------------------

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


    def evaluate_grid(
        pip_values,
        dip_values,
        stage,
    ):

        total = (
            len(
                pip_values
            )
            *
            len(
                dip_values
            )
        )


        count = 0


        for pip_k in pip_values:

            for dip_k in dip_values:

                count += 1


                tag = (
                    f"{stage}_"
                    f"pip{pip_k:.6f}_"
                    f"dip{dip_k:.6f}"
                    .replace(
                        ".",
                        "p",
                    )
                )


                xml_path = (
                    outdir
                    /
                    f"{tag}.xml"
                )


                print()

                print(
                    f"[{stage} "
                    f"{count:03d}/"
                    f"{total:03d}] "
                    f"PIP={pip_k:.6g} "
                    f"DIP={dip_k:.6g}"
                )


                try:

                    patch_candidate(
                        source,
                        xml_path,
                        pip_k,
                        dip_k,
                    )


                    result = run_one(
                        xml_path,
                        bag,
                        contact_replay,
                    )


                    pose = np.asarray(
                        result[
                            "pose"
                        ],
                        dtype=float,
                    )


                    row = {

                        "stage":
                            stage,

                        "pip_k":
                            pip_k,

                        "dip_k":
                            dip_k,

                        "score":
                            result[
                                "score"
                            ],

                        "rmse":
                            result[
                                "rmse"
                            ],

                        "mcp":
                            pose[0],

                        "pip":
                            pose[1],

                        "dip":
                            pose[2],

                        "first_mid":
                            fnum(
                                result.get(
                                    "first_mid_encoder"
                                )
                            ),

                        "first_tip":
                            fnum(
                                result.get(
                                    "first_tip_encoder"
                                )
                            ),

                        "overlap":
                            fnum(
                                result.get(
                                    "overlap_longest"
                                ),
                                0.0,
                            ),

                        "target_mid":
                            bool(
                                result.get(
                                    "target_mid",
                                    False,
                                )
                            ),

                        "target_tip":
                            bool(
                                result.get(
                                    "target_tip",
                                    False,
                                )
                            ),

                        "mid_gap":
                            fnum(
                                result.get(
                                    "target_mid_gap_mm"
                                )
                            ),

                        "tip_gap":
                            fnum(
                                result.get(
                                    "target_tip_gap_mm"
                                )
                            ),

                        "tip_force":
                            fnum(
                                result.get(
                                    "positive_tip_median"
                                )
                            ),

                        "mid_trans":
                            int(
                                result.get(
                                    "mid_transitions",
                                    0,
                                )
                            ),

                        "tip_trans":
                            int(
                                result.get(
                                    "tip_transitions",
                                    0,
                                )
                            ),

                        "xml":
                            str(
                                xml_path
                            ),

                        "status":
                            "OK",
                    }


                    rows.append(
                        row
                    )


                    print(
                        "   pose = "
                        f"{pose[0]:.2f} / "
                        f"{pose[1]:.2f} / "
                        f"{pose[2]:.2f} deg"
                    )

                    print(
                        f"   RMSE = "
                        f"{row['rmse']:.3f} deg"
                    )

                    print(
                        f"   score = "
                        f"{row['score']:.3f}"
                    )

                    print(
                        f"   overlap = "
                        f"{row['overlap']:.3f} s"
                    )

                    print(
                        f"   gaps = "
                        f"{row['mid_gap']:+.3f} / "
                        f"{row['tip_gap']:+.3f} mm"
                    )

                    print(
                        f"   Ftip median = "
                        f"{row['tip_force']:.4f} N"
                    )


                except Exception as exc:

                    rows.append(
                        {
                            "stage":
                                stage,

                            "pip_k":
                                pip_k,

                            "dip_k":
                                dip_k,

                            "score":
                                math.inf,

                            "rmse":
                                math.inf,

                            "xml":
                                str(
                                    xml_path
                                ),

                            "status":
                                (
                                    "FAIL: "
                                    +
                                    str(
                                        exc
                                    )
                                ),
                        }
                    )


                    print(
                        f"   FAILED: {exc}"
                    )


    try:

        # ========================================================
        # STAGE 1
        # ========================================================

        print()

        print(
            "=" * 130
        )

        print(
            "V35H1 — COARSE 2-D PIP/DIP SEARCH"
        )

        print(
            "=" * 130
        )


        evaluate_grid(
            PIP_COARSE,
            DIP_COARSE,
            "coarse",
        )


        successful = sorted(
            (
                row
                for row in rows
                if row[
                    "status"
                ]
                ==
                "OK"
            ),
            key=lambda x:
                x[
                    "score"
                ],
        )


        if not successful:

            raise RuntimeError(
                "All coarse candidates failed."
            )


        coarse_best = (
            successful[0]
        )


        # ========================================================
        # STAGE 2
        # ========================================================

        print()

        print(
            "=" * 130
        )

        print(
            "V35H1 — LOCAL REFINEMENT"
        )

        print(
            "=" * 130
        )


        pip_refine = neighbourhood(
            coarse_best[
                "pip_k"
            ]
        )


        dip_refine = neighbourhood(
            coarse_best[
                "dip_k"
            ]
        )


        evaluate_grid(
            pip_refine,
            dip_refine,
            "refine",
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


    successful = sorted(
        (
            row
            for row in rows
            if row[
                "status"
            ]
            ==
            "OK"
        ),
        key=lambda x:
            x[
                "score"
            ],
    )


    if not successful:

        raise RuntimeError(
            "No successful candidates."
        )


    # ------------------------------------------------
    # Save everything.
    # ------------------------------------------------

    fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )


    with summary.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # ------------------------------------------------
    # Print top 10.
    # ------------------------------------------------

    print()

    print(
        "=" * 150
    )

    print(
        "TOP 10 V35H1 CANDIDATES"
    )

    print(
        "=" * 150
    )


    for index, row in enumerate(
        successful[:10],
        start=1,
    ):

        print(
            f"{index:02d}. "
            f"score={row['score']:.3f} "
            f"RMSE={row['rmse']:.3f} "
            f"PIPk={row['pip_k']:.6g} "
            f"DIPk={row['dip_k']:.6g} "
            f"pose="
            f"{row['mcp']:.2f}/"
            f"{row['pip']:.2f}/"
            f"{row['dip']:.2f} "
            f"overlap="
            f"{row['overlap']:.3f}s "
            f"gaps="
            f"{row['mid_gap']:+.3f}/"
            f"{row['tip_gap']:+.3f}mm "
            f"Ftip="
            f"{row['tip_force']:.4f}N"
        )


    best = (
        successful[0]
    )


    shutil.copy2(
        best[
            "xml"
        ],
        best_xml,
    )


    print()

    print(
        "=" * 150
    )

    print(
        "SELECTED V35H1 BEST"
    )

    print(
        "=" * 150
    )


    print(
        f"PIP stiffness : "
        f"{best['pip_k']}"
    )

    print(
        f"DIP stiffness : "
        f"{best['dip_k']}"
    )

    print(
        "Pose          : "
        f"{best['mcp']:.3f} / "
        f"{best['pip']:.3f} / "
        f"{best['dip']:.3f} deg"
    )

    print(
        f"RMSE          : "
        f"{best['rmse']:.3f} deg"
    )

    print(
        f"Score         : "
        f"{best['score']:.3f}"
    )

    print(
        f"Overlap       : "
        f"{best['overlap']:.3f} s"
    )

    print(
        "Target gaps   : "
        f"{best['mid_gap']:+.3f} / "
        f"{best['tip_gap']:+.3f} mm"
    )

    print(
        f"Tip force med : "
        f"{best['tip_force']:.4f} N "
        f"(HW {HW_FORCE:.5f} N)"
    )

    print(
        f"Best XML      : "
        f"{best_xml}"
    )

    print(
        f"Summary CSV   : "
        f"{summary}"
    )


    # ------------------------------------------------
    # Delete non-winning XMLs unless requested.
    # ------------------------------------------------

    if not args.keep_all:

        winning_source = Path(
            best[
                "xml"
            ]
        ).resolve()


        for row in rows:

            candidate = Path(
                row.get(
                    "xml",
                    "",
                )
            )


            if (
                candidate.exists()
                and
                candidate.resolve()
                !=
                winning_source
            ):

                try:

                    candidate.unlink()

                except OSError:

                    pass


    # ------------------------------------------------
    # AUTOMATIC VISUAL INSPECTION.
    # ------------------------------------------------

    if not args.no_viewer:

        print()

        print(
            "=" * 150
        )

        print(
            "OPENING BEST RESULT IN MUJOCO"
        )

        print(
            "=" * 150
        )

        print()

        print(
            "Close the MuJoCo window when you finish inspection."
        )


        b2.v35.CONTACT_MU = (
            CONTACT_MU
        )

        b2.v35.CONTACT_K = (
            CONTACT_K
        )


        try:

            b2.visual_replay(
                best_xml,
                bag,
            )

        finally:

            b2.v35.CONTACT_MU = (
                old_mu
            )

            b2.v35.CONTACT_K = (
                old_k
            )


if __name__ == "__main__":

    main()
