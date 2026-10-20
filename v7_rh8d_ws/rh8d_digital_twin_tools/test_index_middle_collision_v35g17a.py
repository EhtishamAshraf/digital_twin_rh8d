#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import scan_index_contact_friction_v35g7 as g7
import scan_index_dip_stiffness_v35g15b as g15b
import scan_index_dip_preload_v35g16b as g16b


# ================================================================
# G17A — MIDDLE CONTACT REPRESENTATION A/B TEST
#
# Test interaction between:
#
#   middle collision:
#       detailed phalanx mesh
#       existing smooth capsule proxy
#
# and DIP passive spring:
#       original
#       G16B preload candidate (-0.15 rad)
#
# NO:
#   contact-state switching
#   locking
#   prescribed joint trajectories
#   tendon routing changes
#   object pose changes
#
# The existing capsule already belongs to the RH8D model.
# We are NOT inventing H1-specific geometry.
# ================================================================


SCALE = 5.70e-6
TENDON_K = 2100.0

CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

DIP_JOINT = "Index_Distal--Index_Middle"

PRELOAD_REF = -0.15


VARIANTS = [
    (
        "mesh_baseline",
        "mesh",
        0.0,
    ),
    (
        "mesh_preload",
        "mesh",
        PRELOAD_REF,
    ),
    (
        "capsule_baseline",
        "capsule",
        0.0,
    ),
    (
        "capsule_preload",
        "capsule",
        PRELOAD_REF,
    ),
]


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


def find_geom(
    root,
    name,
):
    for geom in root.iter("geom"):
        if geom.get("name") == name:
            return geom

    raise RuntimeError(
        f"Geom not found: {name}"
    )


def make_variant(
    baseline_xml,
    out_dir,
    name,
    middle_mode,
    dip_ref,
):

    tree = ET.parse(
        baseline_xml
    )

    root = tree.getroot()

    # ------------------------------------------------------------
    # DIP spring condition.
    # Preserve same passive torque at hardware target used in G16B.
    # ------------------------------------------------------------

    dip = find_joint(
        root,
        DIP_JOINT,
    )

    dip_k = g16b.candidate_k(
        dip_ref
    )

    dip.set(
        "springref",
        f"{dip_ref:.12g}",
    )

    dip.set(
        "stiffness",
        f"{dip_k:.12g}",
    )

    # ------------------------------------------------------------
    # Middle contact geometry.
    #
    # Baseline:
    #   collision_idx_mid       = disabled capsule
    #   collision_idx_mid_mesh  = active mesh
    #
    # Capsule candidate:
    #   disable old active mesh
    #   enable existing capsule
    #
    # IMPORTANT:
    # rename active capsule to collision_idx_mid_mesh so all
    # validated G7/G12 measurement code continues monitoring
    # the ACTIVE middle contact geometry.
    # ------------------------------------------------------------

    if middle_mode == "capsule":

        mid_mesh = find_geom(
            root,
            "collision_idx_mid_mesh",
        )

        mid_capsule = find_geom(
            root,
            "collision_idx_mid",
        )

        # Disable the detailed mesh, but KEEP its original name.
        # Some existing diagnostics reference these geom names directly.
        mid_mesh.set(
            "contype",
            "0",
        )

        mid_mesh.set(
            "conaffinity",
            "0",
        )

        mid_mesh.set(
            "rgba",
            "0 0.8 1 0.08",
        )

        # Activate the EXISTING smooth capsule.
        # IMPORTANT: keep its canonical name "collision_idx_mid"
        # because replay_index_tabletop_hardware_contact_v1.py
        # explicitly looks up that MuJoCo object.
        mid_capsule.set(
            "contype",
            "2",
        )

        mid_capsule.set(
            "conaffinity",
            "33",
        )

        mid_capsule.set(
            "condim",
            "4",
        )

        mid_capsule.set(
            "margin",
            "0",
        )

        mid_capsule.set(
            "gap",
            "0",
        )

        mid_capsule.set(
            "friction",
            "1.8 0.03 0.001",
        )

        mid_capsule.set(
            "solref",
            "0.05 1.0",
        )

        mid_capsule.set(
            "solimp",
            "0.80 0.95 0.001",
        )

        mid_capsule.set(
            "rgba",
            "1 0.5 0 0.20",
        )

    elif middle_mode != "mesh":

        raise ValueError(
            middle_mode
        )

    path = (
        out_dir
        /
        f"{name}.xml"
    )

    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )

    return (
        path,
        dip_k,
    )


def episode_intervals(
    trace,
    field,
):

    intervals = []

    active = False
    start = None

    for i, row in enumerate(trace):

        now = bool(
            row[field]
        )

        if now and not active:

            start = i
            active = True

        elif active and not now:

            end = i - 1

            intervals.append(
                (
                    start,
                    end,
                )
            )

            active = False
            start = None

    if active:

        intervals.append(
            (
                start,
                len(trace) - 1,
            )
        )

    return intervals


def contact_metrics(
    trace,
):

    times = np.asarray(
        [
            r["time_s"]
            for r in trace
        ],
        dtype=float,
    )

    mid_intervals = episode_intervals(
        trace,
        "mid_contact",
    )

    tip_intervals = episode_intervals(
        trace,
        "tip_contact",
    )

    def duration(interval):

        if interval is None:
            return np.nan

        a, b = interval

        return float(
            times[b]
            -
            times[a]
        )

    first_mid_duration = (
        duration(
            mid_intervals[0]
        )
        if mid_intervals
        else np.nan
    )

    first_tip_duration = (
        duration(
            tip_intervals[0]
        )
        if tip_intervals
        else np.nan
    )

    total_mid = sum(
        duration(x)
        for x in mid_intervals
    )

    total_tip = sum(
        duration(x)
        for x in tip_intervals
    )

    return {
        "first_middle_duration_s":
            first_mid_duration,

        "first_tip_duration_s":
            first_tip_duration,

        "total_middle_contact_s":
            float(total_mid),

        "total_tip_contact_s":
            float(total_tip),
    }


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
            "v35g17a_middle_collision"
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
            "G17A — MIDDLE CONTACT REPRESENTATION A/B TEST"
        )

        print("=" * 150)

        for (
            name,
            middle_mode,
            dip_ref,
        ) in VARIANTS:

            (
                xml,
                dip_k,
            ) = make_variant(
                baseline_xml,
                out_dir,
                name,
                middle_mode,
                dip_ref,
            )

            print()
            print("-" * 150)

            print(
                f"CANDIDATE: {name}"
            )

            print(
                f"middle collision = "
                f"{middle_mode}"
            )

            print(
                f"DIP springref = "
                f"{dip_ref:+.3f} rad"
            )

            print(
                f"DIP stiffness = "
                f"{dip_k:.6f} Nm/rad"
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

            row.update(
                contact_metrics(
                    trace
                )
            )

            row["candidate"] = name
            row["middle_geometry"] = middle_mode
            row["dip_springref"] = dip_ref
            row["dip_stiffness"] = dip_k

            rows.append(
                row
            )

            trajectories[
                name
            ] = (
                xml,
                qpos,
                times,
            )

            g7.save_trace_csv(
                out_dir
                /
                f"{name}_timeseries.csv",
                trace,
            )

            np.savez_compressed(
                out_dir
                /
                f"{name}_trajectory.npz",
                times=times,
                qpos=qpos,
            )

            print()
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
                "target contacts M/T = "
                f"{row['target_mid_contact']} / "
                f"{row['target_tip_contact']}"
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
                f"{row['delta_dip_middle_to_tip_deg']:+.3f}"
            )

            print(
                f"first middle duration = "
                f"{row['first_middle_duration_s']:.4f} s"
            )

            print(
                f"first tip duration = "
                f"{row['first_tip_duration_s']:.4f} s"
            )

            print(
                f"total middle contact = "
                f"{row['total_middle_contact_s']:.4f} s"
            )

            print(
                f"total tip contact = "
                f"{row['total_tip_contact_s']:.4f} s"
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
        "g17a_summary.csv"
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

    print()
    print("=" * 180)

    print(
        "G17A SUMMARY"
    )

    print("=" * 180)

    print(
        "candidate          geom    ref     RMSE "
        "epM epT targetM targetT DIP@M delay "
        "dPIP  dDIP firstMidDur firstTipDur"
    )

    print("-" * 180)

    for r in rows:

        print(
            f"{r['candidate']:<18s} "
            f"{r['middle_geometry']:<7s} "
            f"{r['dip_springref']:+6.3f} "
            f"{r['rmse_deg']:5.3f} "
            f"{r['middle_episodes']:3d} "
            f"{r['tip_episodes']:3d} "
            f"{str(r['target_mid_contact']):>7s} "
            f"{str(r['target_tip_contact']):>7s} "
            f"{r['dip_first_middle_deg']:6.2f} "
            f"{r['middle_to_tip_delay_s']:6.3f} "
            f"{r['delta_pip_middle_to_tip_deg']:+5.2f} "
            f"{r['delta_dip_middle_to_tip_deg']:+5.2f} "
            f"{r['first_middle_duration_s']:10.3f} "
            f"{r['first_tip_duration_s']:10.3f}"
        )

    if not args.no_gui:

        print()
        print("=" * 120)

        print(
            "MANDATORY GUI A/B VALIDATION"
        )

        print("=" * 120)

        print(
            "Close each window manually."
        )

        for i, (
            name,
            middle_mode,
            dip_ref,
        ) in enumerate(
            VARIANTS,
            start=1,
        ):

            row = next(
                r
                for r in rows
                if r[
                    "candidate"
                ]
                ==
                name
            )

            (
                xml,
                qpos,
                times,
            ) = trajectories[
                name
            ]

            print()
            print("#" * 130)

            print(
                f"GUI {i}/{len(VARIANTS)} — "
                f"{name}"
            )

            print(
                f"middle = {middle_mode}"
            )

            print(
                f"DIP ref = {dip_ref:+.3f}"
            )

            print(
                "episodes M/T = "
                f"{row['middle_episodes']} / "
                f"{row['tip_episodes']}"
            )

            print(
                "target M/T = "
                f"{row['target_mid_contact']} / "
                f"{row['target_tip_contact']}"
            )

            print(
                "dPIP/dDIP = "
                f"{row['delta_pip_middle_to_tip_deg']:+.3f} / "
                f"{row['delta_dip_middle_to_tip_deg']:+.3f}"
            )

            print("#" * 130)

            g7.exact_gui_replay(
                xml,
                qpos,
                times,
                (
                    f"G17A {name}"
                ),
            )

    with (
        out_dir
        /
        "manifest.txt"
    ).open("w") as f:

        f.write(
            "V35G17A — middle contact representation A/B\n\n"
        )

        f.write(
            "Question:\n"
            "Does the detailed middle collision mesh prevent "
            "smooth sliding/wrapping against the bottle?\n\n"
        )

        f.write(
            "Existing middle capsule is tested without changing "
            "its original geometric dimensions.\n"
        )

        f.write(
            "No contact-dependent rules or prescribed motion.\n"
        )

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
