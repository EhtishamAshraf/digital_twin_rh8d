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
import scan_index_zero_margin_damping_v35e2b as e2b


# ============================================================
# V35I1
#
# Goal:
#   suppress repeated contact/rebound while preserving
#   hardware-like final pose/contact.
#
# Fixed:
#   - CAD spatial tendon route
#   - PIP/DIP stiffness from current selected model
#   - bottle pose
#   - zero contact margin
#   - contact stiffness/friction
#
# Scanned:
#   - spatial tendon damping
#   - index spool actuator kv
#   - index joint damping multiplier
# ============================================================


INDEX_JOINTS = [
    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",
]

TENDON_NAME = "idx_flex_spatial"
ACTUATOR_NAME = "pos_index_tendon"


CONTACT_DAMPING = 20.0
CONTACT_MU = 0.05
CONTACT_K = 1000.0


# Keep this relatively small first.
TENDON_DAMPINGS = [
    2.0,
    4.0,
    8.0,
    12.0,
]

ACTUATOR_KVS = [
    20.0,
    35.0,
    55.0,
    80.0,
]

JOINT_DAMPING_MULTIPLIERS = [
    1.0,
    2.0,
    4.0,
]


def patch_model(
    source: Path,
    output: Path,
    tendon_damping: float,
    actuator_kv: float,
    joint_damping_mult: float,
):

    tree = ET.parse(source)
    root = tree.getroot()

    tendon_found = False
    actuator_found = False
    joints_found = 0

    for tendon in root.iter("spatial"):

        if tendon.get("name") == TENDON_NAME:

            tendon.set(
                "damping",
                f"{tendon_damping:.12g}",
            )

            tendon_found = True


    for actuator in root.iter("position"):

        if actuator.get("name") == ACTUATOR_NAME:

            actuator.set(
                "kv",
                f"{actuator_kv:.12g}",
            )

            actuator_found = True


    for joint in root.iter("joint"):

        if joint.get("name") in INDEX_JOINTS:

            old = float(
                joint.get(
                    "damping",
                    "0",
                )
            )

            joint.set(
                "damping",
                f"{old * joint_damping_mult:.12g}",
            )

            joints_found += 1


    if not tendon_found:
        raise RuntimeError(
            "idx_flex_spatial not found"
        )

    if not actuator_found:
        raise RuntimeError(
            "pos_index_tendon not found"
        )

    if joints_found != 3:
        raise RuntimeError(
            f"Expected 3 index joints, found {joints_found}"
        )


    tree.write(
        output,
        encoding="utf-8",
        xml_declaration=True,
    )


def score(result):

    rmse = float(
        result["rmse"]
    )

    score = rmse


    # ------------------------------------------------
    # Explicitly penalize repeated contact chatter.
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

    score += 0.75 * max(
        0,
        mid_trans - 1,
    )

    score += 1.00 * max(
        0,
        tip_trans - 1,
    )


    # ------------------------------------------------
    # Final contact must be maintained.
    # ------------------------------------------------

    mid_gap = float(
        result.get(
            "target_mid_gap_mm",
            math.nan,
        )
    )

    tip_gap = float(
        result.get(
            "target_tip_gap_mm",
            math.nan,
        )
    )


    if math.isfinite(mid_gap):

        score += (
            2.0
            *
            abs(mid_gap)
        )


    if math.isfinite(tip_gap):

        # fingertip gap is especially important
        score += (
            3.0
            *
            abs(tip_gap)
        )


    # ------------------------------------------------
    # Penalize lack of target contact.
    # ------------------------------------------------

    if not bool(
        result.get(
            "target_mid",
            False,
        )
    ):
        score += 4.0


    if not bool(
        result.get(
            "target_tip",
            False,
        )
    ):
        score += 6.0


    # ------------------------------------------------
    # Reward sustained simultaneous contact.
    # ------------------------------------------------

    overlap = float(
        result.get(
            "overlap_longest",
            0.0,
        )
    )

    if overlap < 0.20:
        score += (
            3.0
            *
            (
                0.20
                -
                overlap
            )
            /
            0.20
        )


    return float(score)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "contact_bag",
        type=Path,
    )

    ap.add_argument(
        "--source",
        type=Path,
        default=Path(
            "../src/ur5e_mujoco_description/mjcf/"
            "ur5e_rh8d_index_cad_spatial_v35h1_best.xml"
        ),
    )

    ap.add_argument(
        "--no-viewer",
        action="store_true",
    )

    args = ap.parse_args()


    bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )

    source = (
        args.source
        .expanduser()
        .resolve()
    )


    if not bag.exists():
        raise RuntimeError(
            f"Bag does not exist: {bag}"
        )

    if not source.exists():
        raise RuntimeError(
            f"Source XML does not exist: {source}"
        )


    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
    )


    mjcf_dir = source.parent

    summary_path = (
        mjcf_dir
        /
        "v35i1_dynamic_summary.csv"
    )

    best_xml = (
        mjcf_dir
        /
        "ur5e_rh8d_index_cad_spatial_v35i1_best.xml"
    )


    old_contact_damping = float(
        contact_replay.CONTACT_DAMPING
    )

    old_mu = b2.v35.CONTACT_MU
    old_k = b2.v35.CONTACT_K


    contact_replay.CONTACT_DAMPING = (
        CONTACT_DAMPING
    )

    b2.v35.CONTACT_MU = CONTACT_MU
    b2.v35.CONTACT_K = CONTACT_K


    rows = []

    total = (
        len(TENDON_DAMPINGS)
        *
        len(ACTUATOR_KVS)
        *
        len(JOINT_DAMPING_MULTIPLIERS)
    )

    count = 0


    try:

        for td in TENDON_DAMPINGS:

            for kv in ACTUATOR_KVS:

                for jm in JOINT_DAMPING_MULTIPLIERS:

                    count += 1


                    name = (
                        f"v35i1_td{td:g}_"
                        f"kv{kv:g}_"
                        f"jm{jm:g}"
                        .replace(".", "p")
                    )


                    xml = (
                        mjcf_dir
                        /
                        f"{name}.xml"
                    )


                    print()
                    print(
                        "=" * 110
                    )

                    print(
                        f"[{count:02d}/{total:02d}] "
                        f"tendon damping={td:g}, "
                        f"actuator kv={kv:g}, "
                        f"joint damping x{jm:g}"
                    )


                    patch_model(
                        source,
                        xml,
                        td,
                        kv,
                        jm,
                    )


                    try:

                        result = e2b.run_one(
                            CONTACT_DAMPING,
                            xml,
                            bag,
                            contact_replay,
                        )


                        result = dict(
                            result
                        )


                        pose = np.asarray(
                            result["pose"],
                            dtype=float,
                        )


                        row = {

                            "name": name,

                            "tendon_damping":
                                td,

                            "actuator_kv":
                                kv,

                            "joint_damping_mult":
                                jm,

                            "rmse":
                                float(
                                    result["rmse"]
                                ),

                            "score":
                                score(
                                    result
                                ),

                            "mcp":
                                pose[0],

                            "pip":
                                pose[1],

                            "dip":
                                pose[2],

                            "mid_transitions":
                                int(
                                    result.get(
                                        "mid_transitions",
                                        0,
                                    )
                                ),

                            "tip_transitions":
                                int(
                                    result.get(
                                        "tip_transitions",
                                        0,
                                    )
                                ),

                            "overlap":
                                float(
                                    result.get(
                                        "overlap_longest",
                                        0.0,
                                    )
                                ),

                            "mid_gap":
                                float(
                                    result.get(
                                        "target_mid_gap_mm",
                                        math.nan,
                                    )
                                ),

                            "tip_gap":
                                float(
                                    result.get(
                                        "target_tip_gap_mm",
                                        math.nan,
                                    )
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

                            "xml":
                                str(xml),

                            "status":
                                "OK",
                        }


                        rows.append(
                            row
                        )


                        print(
                            f"pose = "
                            f"{row['mcp']:.2f}/"
                            f"{row['pip']:.2f}/"
                            f"{row['dip']:.2f}"
                        )

                        print(
                            f"RMSE = {row['rmse']:.3f}"
                        )

                        print(
                            f"score = {row['score']:.3f}"
                        )

                        print(
                            "transitions mid/tip = "
                            f"{row['mid_transitions']}/"
                            f"{row['tip_transitions']}"
                        )

                        print(
                            "target gaps mid/tip = "
                            f"{row['mid_gap']:+.3f}/"
                            f"{row['tip_gap']:+.3f} mm"
                        )

                        print(
                            f"overlap = "
                            f"{row['overlap']:.3f}s"
                        )


                    except Exception as exc:

                        rows.append(
                            {
                                "name":
                                    name,

                                "status":
                                    f"FAIL: {exc}",

                                "score":
                                    math.inf,

                                "rmse":
                                    math.inf,

                                "xml":
                                    str(xml),
                            }
                        )

                        print(
                            f"FAILED: {exc}"
                        )


    finally:

        contact_replay.CONTACT_DAMPING = (
            old_contact_damping
        )

        b2.v35.CONTACT_MU = old_mu
        b2.v35.CONTACT_K = old_k


    good = sorted(
        [
            r
            for r in rows
            if r.get(
                "status"
            )
            ==
            "OK"
        ],
        key=lambda r:
            r["score"],
    )


    if not good:

        raise RuntimeError(
            "All V35I1 candidates failed"
        )


    fields = sorted(
        {
            k
            for r in rows
            for k in r.keys()
        }
    )


    with summary_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


    print()
    print(
        "=" * 130
    )

    print(
        "TOP 10 V35I1 DYNAMIC-SETTLING CANDIDATES"
    )

    print(
        "=" * 130
    )


    for i, r in enumerate(
        good[:10],
        1,
    ):

        print(
            f"{i:02d}. "
            f"score={r['score']:.3f} "
            f"RMSE={r['rmse']:.3f} "
            f"td={r['tendon_damping']:g} "
            f"kv={r['actuator_kv']:g} "
            f"jm={r['joint_damping_mult']:g} "
            f"pose="
            f"{r['mcp']:.2f}/"
            f"{r['pip']:.2f}/"
            f"{r['dip']:.2f} "
            f"trans="
            f"{r['mid_transitions']}/"
            f"{r['tip_transitions']} "
            f"gaps="
            f"{r['mid_gap']:+.3f}/"
            f"{r['tip_gap']:+.3f}mm "
            f"overlap="
            f"{r['overlap']:.3f}s"
        )


    best = good[0]


    shutil.copy2(
        best["xml"],
        best_xml,
    )


    print()
    print(
        "=" * 130
    )

    print(
        "SELECTED V35I1 BEST"
    )

    print(
        "=" * 130
    )


    print(
        f"tendon damping      = "
        f"{best['tendon_damping']}"
    )

    print(
        f"actuator kv         = "
        f"{best['actuator_kv']}"
    )

    print(
        f"joint damping mult  = "
        f"{best['joint_damping_mult']}"
    )

    print(
        f"RMSE                = "
        f"{best['rmse']:.3f} deg"
    )

    print(
        f"score               = "
        f"{best['score']:.3f}"
    )

    print(
        "contact transitions = "
        f"{best['mid_transitions']} / "
        f"{best['tip_transitions']}"
    )

    print(
        "target gaps         = "
        f"{best['mid_gap']:+.3f} / "
        f"{best['tip_gap']:+.3f} mm"
    )

    print(
        f"best XML            = "
        f"{best_xml}"
    )

    print(
        f"summary             = "
        f"{summary_path}"
    )


    # Keep only winner.
    winner = Path(
        best["xml"]
    ).resolve()


    for row in rows:

        p = Path(
            row.get(
                "xml",
                "",
            )
        )

        if (
            p.exists()
            and
            p.resolve()
            !=
            winner
        ):

            try:
                p.unlink()
            except OSError:
                pass


    if not args.no_viewer:

        print()
        print(
            "=" * 130
        )

        print(
            "OPENING V35I1 BEST IN MUJOCO"
        )

        print(
            "=" * 130
        )


        b2.v35.CONTACT_MU = CONTACT_MU
        b2.v35.CONTACT_K = CONTACT_K


        try:

            b2.visual_replay(
                best_xml,
                bag,
            )

        finally:

            b2.v35.CONTACT_MU = old_mu
            b2.v35.CONTACT_K = old_k


if __name__ == "__main__":

    main()
