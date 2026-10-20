#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import scan_index_contact_friction_v35g7 as g7


# ================================================================
# V35G14B — CAD DISTAL COLLISION A/B TEST
#
# PURPOSE
# -------
# The current model uses:
#
#   middle contact : CAD-derived mesh collision
#   distal contact : simple capsule
#
# But the real distal finger geometry is already available as
# tip.stl and is already used for visualization.
#
# G14B replaces ONLY the distal capsule collision with the existing
# CAD tip mesh at the exact visual-mesh transform.
#
# NO:
#   joint locking
#   prescribed motion
#   contact-state switching
#   tendon topology changes
#   H1-specific artificial offset
#
# SUCCESS:
#   hardware-like contact sequence + low RMSE
# ================================================================


SCALE = 5.70e-6
TENDON_K = 2100.0

CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0


def find_body(root, name):

    for body in root.iter("body"):
        if body.get("name") == name:
            return body

    raise RuntimeError(
        f"Body not found: {name}"
    )


def find_geom(root, name):

    for geom in root.iter("geom"):
        if geom.get("name") == name:
            return geom

    raise RuntimeError(
        f"Geom not found: {name}"
    )


def make_cad_distal_candidate(
    baseline_xml: Path,
    out_dir: Path,
) -> Path:

    tree = ET.parse(
        baseline_xml
    )

    root = tree.getroot()

    distal_body = find_body(
        root,
        "Index_Distal",
    )

    capsule = find_geom(
        root,
        "collision_idx_tip",
    )

    # ------------------------------------------------------------
    # Disable the old simplified distal capsule.
    # Keep it in the XML for traceability.
    # ------------------------------------------------------------

    capsule.set(
        "contype",
        "0",
    )

    capsule.set(
        "conaffinity",
        "0",
    )

    capsule.set(
        "rgba",
        "1 0 0 0.15",
    )

    # ------------------------------------------------------------
    # Find the existing visual CAD tip geometry.
    #
    # In the source XML:
    #
    #   body = Index_Distal
    #   mesh = tip
    #   pos  = -0.00506253 0 4.01497e-07
    #
    # We copy its geometry/transform rather than inventing one.
    # ------------------------------------------------------------

    visual_tip = None

    for geom in distal_body.findall(
        "geom"
    ):

        if geom.get("mesh") == "tip":
            visual_tip = geom
            break

    if visual_tip is None:
        raise RuntimeError(
            "Could not find existing CAD tip visual mesh "
            "inside Index_Distal."
        )

    cad_collision = ET.Element(
        "geom"
    )

    cad_collision.set(
        "name",
        "collision_idx_tip_cad",
    )

    cad_collision.set(
        "type",
        "mesh",
    )

    cad_collision.set(
        "mesh",
        "tip",
    )

    # Exact transform from the visual CAD mesh.
    cad_collision.set(
        "pos",
        visual_tip.get(
            "pos",
            "0 0 0",
        ),
    )

    if visual_tip.get(
        "quat"
    ) is not None:

        cad_collision.set(
            "quat",
            visual_tip.get(
                "quat"
            ),
        )

    cad_collision.set(
        "mass",
        "0",
    )

    cad_collision.set(
        "contype",
        "2",
    )

    cad_collision.set(
        "conaffinity",
        "33",
    )

    cad_collision.set(
        "condim",
        "4",
    )

    cad_collision.set(
        "friction",
        "1.8 0.03 0.001",
    )

    cad_collision.set(
        "solref",
        "0.05 1.0",
    )

    cad_collision.set(
        "solimp",
        "0.80 0.95 0.001",
    )

    cad_collision.set(
        "margin",
        "0",
    )

    cad_collision.set(
        "gap",
        "0",
    )

    # Visible enough for debugging.
    cad_collision.set(
        "rgba",
        "0 1 0 0.20",
    )

    distal_body.append(
        cad_collision
    )

    destination = (
        out_dir
        /
        "v35g14b_cad_distal_collision.xml"
    )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )

    return destination


def contact_episodes(
    trace,
    field,
):

    count = 0
    previous = False

    for row in trace:

        value = bool(
            row[field]
        )

        if value and not previous:
            count += 1

        previous = value

    return int(count)


def first_contact_index(
    trace,
    field,
):

    for i, row in enumerate(
        trace
    ):

        if bool(
            row[field]
        ):
            return i

    return None


def metrics(trace):

    mcp = np.asarray(
        [
            r["mcp_deg"]
            for r in trace
        ],
        dtype=float,
    )

    pip = np.asarray(
        [
            r["pip_deg"]
            for r in trace
        ],
        dtype=float,
    )

    dip = np.asarray(
        [
            r["dip_deg"]
            for r in trace
        ],
        dtype=float,
    )

    times = np.asarray(
        [
            r["time_s"]
            for r in trace
        ],
        dtype=float,
    )

    mid = np.asarray(
        [
            bool(
                r["mid_contact"]
            )
            for r in trace
        ],
        dtype=bool,
    )

    tip = np.asarray(
        [
            bool(
                r["tip_contact"]
            )
            for r in trace
        ],
        dtype=bool,
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
        "middle_episodes":
            contact_episodes(
                trace,
                "mid_contact",
            ),

        "tip_episodes":
            contact_episodes(
                trace,
                "tip_contact",
            ),

        "mcp_first_mid":
            np.nan,

        "pip_first_mid":
            np.nan,

        "dip_first_mid":
            np.nan,

        "mcp_first_tip":
            np.nan,

        "pip_first_tip":
            np.nan,

        "dip_first_tip":
            np.nan,

        "delta_pip_mid_tip":
            np.nan,

        "delta_dip_mid_tip":
            np.nan,

        "dip_pip_ratio":
            np.nan,

        "delay_mid_to_tip_s":
            np.nan,

        "max_overlap_s":
            0.0,
    }

    if i_mid is not None:

        result[
            "mcp_first_mid"
        ] = float(
            mcp[i_mid]
        )

        result[
            "pip_first_mid"
        ] = float(
            pip[i_mid]
        )

        result[
            "dip_first_mid"
        ] = float(
            dip[i_mid]
        )

    if (
        i_mid is not None
        and
        i_tip is not None
        and
        i_tip > i_mid
    ):

        result[
            "mcp_first_tip"
        ] = float(
            mcp[i_tip]
        )

        result[
            "pip_first_tip"
        ] = float(
            pip[i_tip]
        )

        result[
            "dip_first_tip"
        ] = float(
            dip[i_tip]
        )

        dpip = float(
            pip[i_tip]
            -
            pip[i_mid]
        )

        ddip = float(
            dip[i_tip]
            -
            dip[i_mid]
        )

        result[
            "delta_pip_mid_tip"
        ] = dpip

        result[
            "delta_dip_mid_tip"
        ] = ddip

        result[
            "dip_pip_ratio"
        ] = (
            abs(ddip)
            /
            max(
                abs(dpip),
                1e-12,
            )
        )

        result[
            "delay_mid_to_tip_s"
        ] = float(
            times[i_tip]
            -
            times[i_mid]
        )

    # ------------------------------------------------------------
    # Longest continuous simultaneous middle+distal contact.
    # ------------------------------------------------------------

    overlap = (
        mid
        &
        tip
    )

    start = None
    longest = 0.0

    for i, active in enumerate(
        overlap
    ):

        if (
            active
            and
            start is None
        ):
            start = i

        elif (
            not active
            and
            start is not None
        ):

            end = i - 1

            longest = max(
                longest,
                float(
                    times[end]
                    -
                    times[start]
                ),
            )

            start = None

    if start is not None:

        longest = max(
            longest,
            float(
                times[-1]
                -
                times[start]
            ),
        )

    result[
        "max_overlap_s"
    ] = longest

    return result


def score(row):
    """
    Mechanics-first ranking.

    This is ONLY for GUI ordering.
    It cannot automatically accept a candidate.
    """

    value = 0.0

    value += (
        500.0
        *
        max(
            0,
            row[
                "middle_episodes"
            ]
            -
            1,
        )
    )

    value += (
        500.0
        *
        max(
            0,
            row[
                "tip_episodes"
            ]
            -
            1,
        )
    )

    if not row[
        "mid_retained_to_tip"
    ]:
        value += 500.0

    if not row[
        "target_mid_contact"
    ]:
        value += 250.0

    if not row[
        "target_tip_contact"
    ]:
        value += 250.0

    dpip = row[
        "delta_pip_mid_tip"
    ]

    ddip = row[
        "delta_dip_mid_tip"
    ]

    if np.isfinite(
        dpip
    ):
        value += (
            100.0
            *
            max(
                0.0,
                0.5
                -
                dpip,
            )
        )
    else:
        value += 300.0

    if np.isfinite(
        ddip
    ):
        value += (
            150.0
            *
            max(
                0.0,
                0.3
                -
                ddip,
            )
        )
    else:
        value += 300.0

    fmax = row[
        "e2b_hold_tip_force_max_N"
    ]

    if np.isfinite(
        fmax
    ):
        value += (
            15.0
            *
            max(
                0.0,
                fmax
                -
                2.0,
            )
        )

    # RMSE matters, but mechanics is first.
    value += (
        10.0
        *
        row[
            "rmse_deg"
        ]
    )

    return float(
        value
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
            "v35g14b_cad_distal_collision"
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

    cad_xml = (
        make_cad_distal_candidate(
            baseline_xml,
            out_dir,
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

    candidates = [
        (
            "baseline_capsule",
            baseline_xml,
        ),

        (
            "cad_tip_mesh",
            cad_xml,
        ),
    ]

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
            "V35G14B — DISTAL COLLISION GEOMETRY A/B TEST"
        )

        print("=" * 150)

        print()
        print(
            "A: current distal capsule"
        )

        print(
            "B: existing CAD tip.stl used as distal collision"
        )

        print()
        print(
            "NO tendon/joint/contact-state mechanism is modified."
        )

        for (
            name,
            xml,
        ) in candidates:

            print()
            print("-" * 150)

            print(
                f"RUNNING: {name}"
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

            extra = metrics(
                trace
            )

            row = dict(
                result
            )

            row.update(
                extra
            )

            row[
                "candidate"
            ] = name

            row[
                "mechanical_score_g14"
            ] = score(
                row
            )

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
                "RESULT"
            )

            print(
                "  target MCP/PIP/DIP = "
                f"{row['target_mcp_deg']:.3f} / "
                f"{row['target_pip_deg']:.3f} / "
                f"{row['target_dip_deg']:.3f}"
            )

            print(
                f"  RMSE = "
                f"{row['rmse_deg']:.3f} deg"
            )

            print()
            print(
                "  contact episodes M/T = "
                f"{row['middle_episodes']} / "
                f"{row['tip_episodes']}"
            )

            print(
                f"  middle->tip delay = "
                f"{row['delay_mid_to_tip_s']:.4f} s"
            )

            print(
                f"  middle->tip dPIP = "
                f"{row['delta_pip_mid_tip']:+.3f} deg"
            )

            print(
                f"  middle->tip dDIP = "
                f"{row['delta_dip_mid_tip']:+.3f} deg"
            )

            print(
                f"  DIP/PIP = "
                f"{row['dip_pip_ratio']:.3f}"
            )

            print(
                f"  max overlap = "
                f"{row['max_overlap_s']:.3f} s"
            )

            print(
                f"  target contacts M/T = "
                f"{row['target_mid_contact']} / "
                f"{row['target_tip_contact']}"
            )

            print(
                f"  tip force median/max = "
                f"{row['e2b_hold_tip_force_median_N']:.3f} / "
                f"{row['e2b_hold_tip_force_max_N']:.3f} N"
            )

            print(
                f"  mechanics score = "
                f"{row['mechanical_score_g14']:.3f}"
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
        "g14b_summary.csv"
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
        key=lambda r:
            r[
                "mechanical_score_g14"
            ],
    )

    print()
    print("=" * 150)

    print(
        "G14B COMPARISON"
    )

    print("=" * 150)

    print()

    print(
        "candidate         RMSE  epM epT "
        "delay    dPIP   dDIP ratio  Fmax"
    )

    for r in ranked:

        print(
            f"{r['candidate']:<16s} "
            f"{r['rmse_deg']:5.3f} "
            f"{r['middle_episodes']:3d} "
            f"{r['tip_episodes']:3d} "
            f"{r['delay_mid_to_tip_s']:7.4f} "
            f"{r['delta_pip_mid_tip']:+6.3f} "
            f"{r['delta_dip_mid_tip']:+6.3f} "
            f"{r['dip_pip_ratio']:5.3f} "
            f"{r['e2b_hold_tip_force_max_N']:6.2f}"
        )

    with (
        out_dir
        /
        "manifest.txt"
    ).open("w") as f:

        f.write(
            "V35G14B — distal collision geometry A/B test\n\n"
        )

        f.write(
            "Baseline:\n"
            "  middle collision = CAD phalanx mesh\n"
            "  distal collision = capsule\n\n"
        )

        f.write(
            "Candidate:\n"
            "  middle collision = CAD phalanx mesh\n"
            "  distal collision = existing CAD tip.stl\n\n"
        )

        f.write(
            "No tendon topology change.\n"
        )

        f.write(
            "No joint locking.\n"
        )

        f.write(
            "No state-dependent behavior.\n"
        )

        f.write(
            "No H1-specific geometry offset.\n"
        )

        f.write(
            "No parameter fitting in this test.\n"
        )

    # ------------------------------------------------------------
    # Mandatory GUI comparison.
    # CAD candidate first, baseline second.
    # ------------------------------------------------------------

    if not args.no_gui:

        gui_order = [
            "cad_tip_mesh",
            "baseline_capsule",
        ]

        for i, name in enumerate(
            gui_order,
            start=1,
        ):

            xml, qpos, times = (
                trajectories[
                    name
                ]
            )

            row = next(
                r
                for r in rows
                if r[
                    "candidate"
                ]
                ==
                name
            )

            print()
            print("#" * 150)

            print(
                f"GUI {i}/2: {name}"
            )

            print(
                f"RMSE = "
                f"{row['rmse_deg']:.3f}"
            )

            print(
                "episodes M/T = "
                f"{row['middle_episodes']} / "
                f"{row['tip_episodes']}"
            )

            print(
                "middle->tip dPIP/dDIP = "
                f"{row['delta_pip_mid_tip']:+.3f} / "
                f"{row['delta_dip_mid_tip']:+.3f}"
            )

            print("#" * 150)

            g7.exact_gui_replay(
                xml,
                qpos,
                times,
                (
                    f"G14B {name}"
                ),
            )

    print()
    print("=" * 150)

    print(
        "V35G14B COMPLETE"
    )

    print("=" * 150)

    print(
        f"\nAll results:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
