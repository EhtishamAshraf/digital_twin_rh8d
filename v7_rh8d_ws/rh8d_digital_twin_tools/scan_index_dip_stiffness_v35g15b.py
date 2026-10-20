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
# G15B — DISTAL PASSIVE-STIFFNESS MECHANISM TEST
#
# Motivation:
# Current simulation reaches first middle contact with DIP already
# strongly flexed (~13.6 deg), while hardware final DIP is ~9.33 deg.
#
# Hypothesis:
# Effective distal passive elastic resistance is too weak / poorly
# represented, causing premature DIP flexion and premature distal
# contact.
#
# ONLY DIP joint stiffness changes.
#
# NO:
#   contact-triggered locking
#   prescribed joint motion
#   tendon topology change
#   geometry change
#   contact parameter tuning
# ================================================================


SCALE = 5.70e-6
TENDON_K = 2100.0

CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

PIP_K = 0.0015

DIP_STIFFNESSES = [
    0.0475,
    0.0550,
    0.0650,
    0.0800,
    0.1000,
    0.1250,
]


PIP_JOINT = "Index_Middle--Index_Proximal"
DIP_JOINT = "Index_Distal--Index_Middle"


def find_joint(root, name):

    for joint in root.iter("joint"):
        if joint.get("name") == name:
            return joint

    raise RuntimeError(
        f"Joint not found: {name}"
    )


def make_variant(
    baseline_xml: Path,
    out_dir: Path,
    dip_k: float,
):

    tree = ET.parse(
        baseline_xml
    )

    root = tree.getroot()

    pip = find_joint(
        root,
        PIP_JOINT,
    )

    dip = find_joint(
        root,
        DIP_JOINT,
    )

    # Keep PIP explicitly frozen.
    pip.set(
        "stiffness",
        f"{PIP_K:.10g}",
    )

    # ONLY variable in G15B.
    dip.set(
        "stiffness",
        f"{dip_k:.10g}",
    )

    tag = (
        f"{dip_k:.4f}"
        .replace(".", "p")
    )

    path = (
        out_dir
        /
        f"model_dipK_{tag}.xml"
    )

    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )

    return path


def episodes(
    trace,
    field,
):

    n = 0
    old = False

    for row in trace:

        new = bool(
            row[field]
        )

        if new and not old:
            n += 1

        old = new

    return n


def first_idx(
    trace,
    field,
):

    for i, row in enumerate(
        trace
    ):
        if bool(row[field]):
            return i

    return None


def mechanics(trace):

    t = np.asarray(
        [
            x["time_s"]
            for x in trace
        ],
        dtype=float,
    )

    mcp = np.asarray(
        [
            x["mcp_deg"]
            for x in trace
        ],
        dtype=float,
    )

    pip = np.asarray(
        [
            x["pip_deg"]
            for x in trace
        ],
        dtype=float,
    )

    dip = np.asarray(
        [
            x["dip_deg"]
            for x in trace
        ],
        dtype=float,
    )

    im = first_idx(
        trace,
        "mid_contact",
    )

    it = first_idx(
        trace,
        "tip_contact",
    )

    r = {
        "middle_episodes":
            episodes(
                trace,
                "mid_contact",
            ),

        "tip_episodes":
            episodes(
                trace,
                "tip_contact",
            ),

        "mcp_first_middle_deg":
            np.nan,

        "pip_first_middle_deg":
            np.nan,

        "dip_first_middle_deg":
            np.nan,

        "middle_to_tip_delay_s":
            np.nan,

        "delta_mcp_middle_to_tip_deg":
            np.nan,

        "delta_pip_middle_to_tip_deg":
            np.nan,

        "delta_dip_middle_to_tip_deg":
            np.nan,

        "dip_pip_motion_ratio":
            np.nan,
    }

    if im is not None:

        r[
            "mcp_first_middle_deg"
        ] = float(
            mcp[im]
        )

        r[
            "pip_first_middle_deg"
        ] = float(
            pip[im]
        )

        r[
            "dip_first_middle_deg"
        ] = float(
            dip[im]
        )

    if (
        im is not None
        and
        it is not None
        and
        it > im
    ):

        dm = float(
            mcp[it]
            -
            mcp[im]
        )

        dp = float(
            pip[it]
            -
            pip[im]
        )

        dd = float(
            dip[it]
            -
            dip[im]
        )

        r[
            "middle_to_tip_delay_s"
        ] = float(
            t[it]
            -
            t[im]
        )

        r[
            "delta_mcp_middle_to_tip_deg"
        ] = dm

        r[
            "delta_pip_middle_to_tip_deg"
        ] = dp

        r[
            "delta_dip_middle_to_tip_deg"
        ] = dd

        r[
            "dip_pip_motion_ratio"
        ] = (
            abs(dd)
            /
            max(
                abs(dp),
                1e-12,
            )
        )

    return r


def rank_key(row):

    clean = (
        row["middle_episodes"] == 1
        and
        row["tip_episodes"] == 1
    )

    target_contacts = (
        bool(
            row[
                "target_mid_contact"
            ]
        )
        and
        bool(
            row[
                "target_tip_contact"
            ]
        )
    )

    delay = row[
        "middle_to_tip_delay_s"
    ]

    if not np.isfinite(delay):
        delay = -1.0

    dpip = row[
        "delta_pip_middle_to_tip_deg"
    ]

    if not np.isfinite(dpip):
        dpip = -999.0

    return (
        not clean,
        not target_contacts,
        (
            row["middle_episodes"]
            +
            row["tip_episodes"]
        ),
        -delay,
        -dpip,
        row["rmse_deg"],
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
            "v35g15b_dip_stiffness"
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
        print("=" * 120)

        print(
            "G15B — DIP PASSIVE STIFFNESS TEST"
        )

        print("=" * 120)

        print()
        print(
            f"PIP stiffness frozen = "
            f"{PIP_K} Nm/rad"
        )

        print(
            "DIP stiffness candidates = "
            +
            ", ".join(
                f"{x:.4f}"
                for x
                in DIP_STIFFNESSES
            )
        )

        for dip_k in (
            DIP_STIFFNESSES
        ):

            xml = make_variant(
                baseline_xml,
                out_dir,
                dip_k,
            )

            print()
            print("-" * 120)

            print(
                f"DIP stiffness = "
                f"{dip_k:.4f} Nm/rad"
            )

            print("-" * 120)

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
                mechanics(
                    trace
                )
            )

            row[
                "pip_stiffness"
            ] = PIP_K

            row[
                "dip_stiffness"
            ] = dip_k

            rows.append(
                row
            )

            trajectories[
                dip_k
            ] = (
                xml,
                qpos,
                times,
            )

            tag = (
                f"{dip_k:.4f}"
                .replace(".", "p")
            )

            g7.save_trace_csv(
                out_dir
                /
                f"dipK_{tag}_timeseries.csv",
                trace,
            )

            np.savez_compressed(
                out_dir
                /
                f"dipK_{tag}_trajectory.npz",
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

            print()
            print(
                "first middle pose = "
                f"{row['mcp_first_middle_deg']:.3f} / "
                f"{row['pip_first_middle_deg']:.3f} / "
                f"{row['dip_first_middle_deg']:.3f}"
            )

            print(
                "middle -> tip:"
            )

            print(
                f"  delay = "
                f"{row['middle_to_tip_delay_s']:.4f} s"
            )

            print(
                f"  dMCP  = "
                f"{row['delta_mcp_middle_to_tip_deg']:+.3f}"
            )

            print(
                f"  dPIP  = "
                f"{row['delta_pip_middle_to_tip_deg']:+.3f}"
            )

            print(
                f"  dDIP  = "
                f"{row['delta_dip_middle_to_tip_deg']:+.3f}"
            )

            print(
                f"  DIP/PIP = "
                f"{row['dip_pip_motion_ratio']:.3f}"
            )

            print(
                "target contacts M/T = "
                f"{row['target_mid_contact']} / "
                f"{row['target_tip_contact']}"
            )

            print(
                "tip force median/max = "
                f"{row['e2b_hold_tip_force_median_N']:.3f} / "
                f"{row['e2b_hold_tip_force_max_N']:.3f} N"
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

    summary = (
        out_dir
        /
        "g15b_summary.csv"
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
        writer.writerows(
            rows
        )

    ranked = sorted(
        rows,
        key=rank_key,
    )

    print()
    print("=" * 150)

    print(
        "G15B — MECHANICS-FIRST SUMMARY"
    )

    print("=" * 150)

    print(
        "DIP_K    RMSE epM epT "
        "DIP@M   delay   dPIP   dDIP  ratio  "
        "targetM targetT Fmax"
    )

    print("-" * 150)

    for r in ranked:

        print(
            f"{r['dip_stiffness']:7.4f} "
            f"{r['rmse_deg']:5.3f} "
            f"{r['middle_episodes']:3d} "
            f"{r['tip_episodes']:3d} "
            f"{r['dip_first_middle_deg']:6.2f} "
            f"{r['middle_to_tip_delay_s']:7.4f} "
            f"{r['delta_pip_middle_to_tip_deg']:+6.2f} "
            f"{r['delta_dip_middle_to_tip_deg']:+6.2f} "
            f"{r['dip_pip_motion_ratio']:5.2f} "
            f"{str(r['target_mid_contact']):>7s} "
            f"{str(r['target_tip_contact']):>7s} "
            f"{r['e2b_hold_tip_force_max_N']:6.2f}"
        )

    with (
        out_dir
        /
        "manifest.txt"
    ).open("w") as f:

        f.write(
            "V35G15B — DIP passive stiffness mechanism test\n\n"
        )

        f.write(
            "Only DIP joint stiffness varied.\n"
        )

        f.write(
            f"PIP stiffness frozen at "
            f"{PIP_K} Nm/rad.\n"
        )

        f.write(
            "Continuous CAD flexor tendon unchanged.\n"
        )

        f.write(
            "Contact model unchanged.\n"
        )

        f.write(
            "No contact-state switching or joint locking.\n\n"
        )

        f.write(
            "Candidates:\n"
        )

        for x in DIP_STIFFNESSES:
            f.write(
                f"  {x:.6f} Nm/rad\n"
            )

    # ========================================================
    # Mandatory visual validation: top three candidates.
    # ========================================================

    if not args.no_gui:

        visual = (
            ranked[:3]
        )

        print()
        print("=" * 120)
        print(
            "MANDATORY GUI VALIDATION"
        )
        print("=" * 120)

        print(
            "Close each window manually to continue."
        )

        for i, row in enumerate(
            visual,
            start=1,
        ):

            dip_k = float(
                row[
                    "dip_stiffness"
                ]
            )

            xml, qpos, times = (
                trajectories[
                    dip_k
                ]
            )

            print()
            print("#" * 120)

            print(
                f"GUI {i}/3"
            )

            print(
                f"DIP stiffness = "
                f"{dip_k:.4f} Nm/rad"
            )

            print(
                f"first-middle DIP = "
                f"{row['dip_first_middle_deg']:.3f} deg"
            )

            print(
                f"middle->tip delay = "
                f"{row['middle_to_tip_delay_s']:.4f} s"
            )

            print(
                "episodes M/T = "
                f"{row['middle_episodes']} / "
                f"{row['tip_episodes']}"
            )

            print(
                f"RMSE = "
                f"{row['rmse_deg']:.3f} deg"
            )

            print("#" * 120)

            g7.exact_gui_replay(
                xml,
                qpos,
                times,
                (
                    f"G15B DIP stiffness "
                    f"{dip_k:.4f}"
                ),
            )

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
