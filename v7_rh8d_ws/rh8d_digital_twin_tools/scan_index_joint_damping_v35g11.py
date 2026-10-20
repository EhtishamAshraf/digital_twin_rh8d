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
# V35G11 — INDEX JOINT VISCOUS-DAMPING / REBOUND SCAN
#
# VARIABLE:
#   common viscous damping applied to MCP, PIP, DIP
#
# FROZEN:
#   transmission scale = 5.70 um/tick
#   tendon stiffness   = 2100 N/m
#   tendon damping     = 2
#   contact mu         = 0.05
#   contact k          = 1000
#   contact damping    = 20
#   margin/gap         = 0
#
#   MCP k              = 0.0045
#   PIP k              = 0.0015
#   DIP k              = 0.0475
#
#   MCP frictionloss   = 0.002
#   PIP frictionloss   = 0.00125
#   DIP frictionloss   = 0
#
# ACCEPTANCE:
#   mechanics first.
#   Repeated contact episodes are a hard warning.
#   RMSE alone can NEVER select the final model.
# ================================================================


SCALE = 5.70e-6
CONTACT_MU = 0.05
TENDON_K = 2100.0
TENDON_DAMPING = 2.0
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

JOINT_DAMPINGS = [
    0.015,
    0.030,
    0.050,
    0.080,
    0.120,
    0.200,
]

INDEX_JOINTS = [
    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",
]

TENDON_NAME = "idx_flex_spatial"


def tag(x):
    return f"{x:.3f}".replace(".", "p")


def make_candidate(
    source_xml: Path,
    out_dir: Path,
    damping: float,
) -> Path:

    tree = ET.parse(source_xml)
    root = tree.getroot()

    found = {}

    for joint in root.iter("joint"):
        name = joint.get("name")

        if name in INDEX_JOINTS:
            found[name] = joint

    missing = [
        name
        for name in INDEX_JOINTS
        if name not in found
    ]

    if missing:
        raise RuntimeError(
            "Missing index joints:\n"
            + "\n".join(missing)
        )

    print()
    print(
        f"    setting common index damping = "
        f"{damping:.6f}"
    )

    for name in INDEX_JOINTS:

        joint = found[name]

        old = joint.get(
            "damping",
            "0",
        )

        joint.set(
            "damping",
            f"{damping:.10g}",
        )

        print(
            f"      {name}: "
            f"{old} -> {damping:.6f}"
        )

    # Explicitly preserve tendon damping at baseline.
    tendon = None

    for spatial in root.iter("spatial"):
        if spatial.get("name") == TENDON_NAME:
            tendon = spatial
            break

    if tendon is None:
        raise RuntimeError(
            f"Missing tendon: {TENDON_NAME}"
        )

    tendon.set(
        "damping",
        f"{TENDON_DAMPING:.10g}",
    )

    destination = (
        out_dir
        / f"candidate_joint_damping_{tag(damping)}.xml"
    )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )

    return destination


def first_contact_index(
    trace,
    field,
):

    for i, row in enumerate(trace):
        if bool(row[field]):
            return i

    return None


def contact_episodes(
    trace,
    field,
):

    flags = np.asarray(
        [
            bool(row[field])
            for row in trace
        ],
        dtype=bool,
    )

    count = 0
    previous = False

    for value in flags:

        if value and not previous:
            count += 1

        previous = value

    return int(count)


def extra_metrics(trace):

    times = np.asarray(
        [
            row["time_s"]
            for row in trace
        ],
        dtype=float,
    )

    mcp = np.asarray(
        [
            row["mcp_deg"]
            for row in trace
        ],
        dtype=float,
    )

    pip = np.asarray(
        [
            row["pip_deg"]
            for row in trace
        ],
        dtype=float,
    )

    dip = np.asarray(
        [
            row["dip_deg"]
            for row in trace
        ],
        dtype=float,
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
        "middle_contact_episodes":
            contact_episodes(
                trace,
                "mid_contact",
            ),

        "tip_contact_episodes":
            contact_episodes(
                trace,
                "tip_contact",
            ),

        "mcp_at_first_mid_deg":
            np.nan,

        "pip_at_first_mid_deg":
            np.nan,

        "dip_at_first_mid_deg":
            np.nan,

        "mcp_at_first_tip_deg":
            np.nan,

        "pip_at_first_tip_deg":
            np.nan,

        "dip_at_first_tip_deg":
            np.nan,

        "delta_mcp_mid_to_tip_deg_g11":
            np.nan,

        "delta_pip_mid_to_tip_deg_g11":
            np.nan,

        "delta_dip_mid_to_tip_deg_g11":
            np.nan,

        "pip_speed_100ms_after_mid_deg_s":
            np.nan,

        "dip_speed_100ms_after_mid_deg_s":
            np.nan,

        "dip_over_pip_100ms_after_mid":
            np.nan,

        "max_abs_pip_speed_deg_s":
            float(
                np.max(
                    np.abs(
                        [
                            row[
                                "pip_vel_deg_s"
                            ]
                            for row in trace
                        ]
                    )
                )
            ),

        "max_abs_dip_speed_deg_s":
            float(
                np.max(
                    np.abs(
                        [
                            row[
                                "dip_vel_deg_s"
                            ]
                            for row in trace
                        ]
                    )
                )
            ),
    }

    if i_mid is None:
        return result

    result[
        "mcp_at_first_mid_deg"
    ] = float(mcp[i_mid])

    result[
        "pip_at_first_mid_deg"
    ] = float(pip[i_mid])

    result[
        "dip_at_first_mid_deg"
    ] = float(dip[i_mid])

    t_mid = float(
        times[i_mid]
    )

    post_mid = np.flatnonzero(
        (times >= t_mid)
        &
        (times <= t_mid + 0.100)
    )

    if len(post_mid) >= 3:

        p = pip[post_mid]
        d = dip[post_mid]
        t = times[post_mid]

        pip_speed = np.gradient(
            p,
            t,
        )

        dip_speed = np.gradient(
            d,
            t,
        )

        pip_med = float(
            np.median(
                np.abs(pip_speed)
            )
        )

        dip_med = float(
            np.median(
                np.abs(dip_speed)
            )
        )

        result[
            "pip_speed_100ms_after_mid_deg_s"
        ] = pip_med

        result[
            "dip_speed_100ms_after_mid_deg_s"
        ] = dip_med

        result[
            "dip_over_pip_100ms_after_mid"
        ] = (
            dip_med
            /
            max(
                pip_med,
                1e-9,
            )
        )

    if (
        i_tip is not None
        and
        i_tip > i_mid
    ):

        result[
            "mcp_at_first_tip_deg"
        ] = float(mcp[i_tip])

        result[
            "pip_at_first_tip_deg"
        ] = float(pip[i_tip])

        result[
            "dip_at_first_tip_deg"
        ] = float(dip[i_tip])

        result[
            "delta_mcp_mid_to_tip_deg_g11"
        ] = float(
            mcp[i_tip]
            -
            mcp[i_mid]
        )

        result[
            "delta_pip_mid_to_tip_deg_g11"
        ] = float(
            pip[i_tip]
            -
            pip[i_mid]
        )

        result[
            "delta_dip_mid_to_tip_deg_g11"
        ] = float(
            dip[i_tip]
            -
            dip[i_mid]
        )

    return result


def mechanical_score(r):
    """
    Lower is better.

    IMPORTANT:
    score is only for ranking candidates to inspect.
    It is NOT automatic acceptance.
    """

    score = 0.0

    if not r["sequence_ok"]:
        score += 2000.0

    # Hardly acceptable if contacts repeatedly break.
    score += (
        300.0
        *
        max(
            0,
            r[
                "middle_contact_episodes"
            ]
            - 1,
        )
    )

    score += (
        300.0
        *
        max(
            0,
            r[
                "tip_contact_episodes"
            ]
            - 1,
        )
    )

    if not r[
        "mid_retained_to_tip"
    ]:
        score += 500.0

    # Target-frame physical contact.
    if not r[
        "target_mid_contact"
    ]:
        score += 150.0

    if not r[
        "target_tip_contact"
    ]:
        score += 150.0

    # Penalize penetration.
    score += (
        120.0
        *
        max(
            0.0,
            -r[
                "target_mid_gap_mm"
            ],
        )
    )

    score += (
        120.0
        *
        max(
            0.0,
            -r[
                "target_tip_gap_mm"
            ],
        )
    )

    # Need PIP wrap after middle contact.
    dpip = r[
        "delta_pip_mid_to_tip_deg_g11"
    ]

    if np.isfinite(dpip):

        score += (
            80.0
            *
            max(
                0.0,
                0.25 - dpip,
            )
        )

    else:
        score += 200.0

    # Need some DIP participation after middle contact.
    ddip = r[
        "delta_dip_mid_to_tip_deg_g11"
    ]

    if np.isfinite(ddip):

        score += (
            100.0
            *
            max(
                0.0,
                0.20 - ddip,
            )
        )

    else:
        score += 250.0

    # Reward relative DIP activity after middle contact.
    ratio = r[
        "dip_over_pip_100ms_after_mid"
    ]

    if np.isfinite(ratio):

        score += (
            80.0
            *
            max(
                0.0,
                0.5 - ratio,
            )
        )

    else:
        score += 100.0

    # Large force spike penalty.
    fmax = r[
        "e2b_hold_tip_force_max_N"
    ]

    if np.isfinite(fmax):

        score += (
            12.0
            *
            max(
                0.0,
                fmax - 2.0,
            )
        )

    # RMSE deliberately weak.
    score += (
        2.0
        *
        r[
            "rmse_deg"
        ]
    )

    return float(score)


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
            / "rh8d_results"
            / "v35g11_joint_damping"
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
        / Path(__file__).name,
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

    old_k = float(
        g7.b2.v35.TENDON_K
    )

    old_mu = float(
        g7.b2.v35.CONTACT_MU
    )

    old_ck = float(
        g7.b2.v35.CONTACT_K
    )

    old_cd = float(
        contact_replay.CONTACT_DAMPING
    )

    rows = []
    replay_data = {}

    try:

        g7.b2.v35.METERS_PER_TICK = SCALE
        g7.b2.v35.TENDON_K = TENDON_K
        g7.b2.v35.CONTACT_MU = CONTACT_MU
        g7.b2.v35.CONTACT_K = CONTACT_K

        contact_replay.CONTACT_DAMPING = (
            CONTACT_DAMPING
        )

        print()
        print("=" * 155)

        print(
            "V35G11 — INDEX JOINT VISCOUS-DAMPING / "
            "REBOUND-SUPPRESSION SCAN"
        )

        print("=" * 155)

        print()
        print("HARDWARE TARGET:")
        print(
            "  MCP -> middle touch -> PIP wrap -> "
            "middle constraint -> PIP slows/stops -> "
            "DIP rotates -> distal constraint -> DIP stops"
        )

        print()
        print("HARD REJECTION:")
        print(
            "  touch -> bounce -> touch -> bounce"
        )

        print()
        print("FROZEN:")
        print(
            f"  scale               = "
            f"{SCALE*1e6:.3f} um/tick"
        )
        print(
            f"  tendon stiffness    = "
            f"{TENDON_K:.1f} N/m"
        )
        print(
            f"  tendon damping      = "
            f"{TENDON_DAMPING:.1f}"
        )
        print(
            f"  contact mu          = "
            f"{CONTACT_MU}"
        )
        print(
            f"  contact stiffness   = "
            f"{CONTACT_K}"
        )
        print(
            f"  contact damping     = "
            f"{CONTACT_DAMPING}"
        )
        print(
            "  MCP/PIP/DIP springs = "
            "0.0045 / 0.0015 / 0.0475"
        )
        print(
            "  MCP/PIP/DIP frictionloss = "
            "0.002 / 0.00125 / 0"
        )

        print()
        print(
            "VARIABLE common MCP/PIP/DIP damping:"
        )

        print(
            "  "
            + ", ".join(
                str(x)
                for x in JOINT_DAMPINGS
            )
        )

        for damping in JOINT_DAMPINGS:

            print()
            print("-" * 155)

            print(
                f"COMMON INDEX JOINT DAMPING = "
                f"{damping:.6f}"
            )

            print("-" * 155)

            candidate = make_candidate(
                baseline_xml,
                out_dir,
                damping,
            )

            (
                result,
                trace,
                qpos_matrix,
                times,
            ) = g7.run_scale(
                SCALE,
                CONTACT_MU,
                candidate,
                bag,
                contact_replay,
            )

            result[
                "joint_damping"
            ] = float(damping)

            result.update(
                extra_metrics(
                    trace
                )
            )

            result[
                "mechanical_score_g11"
            ] = mechanical_score(
                result
            )

            result[
                "single_episode_mid"
            ] = bool(
                result[
                    "middle_contact_episodes"
                ]
                == 1
            )

            result[
                "single_episode_tip"
            ] = bool(
                result[
                    "tip_contact_episodes"
                ]
                == 1
            )

            result[
                "no_rebound_episode_count"
            ] = bool(
                result[
                    "single_episode_mid"
                ]
                and
                result[
                    "single_episode_tip"
                ]
            )

            rows.append(
                result
            )

            name = (
                f"jdamp_{tag(damping)}"
            )

            g7.save_trace_csv(
                out_dir
                / f"{name}_timeseries.csv",
                trace,
            )

            np.savez_compressed(
                out_dir
                / f"{name}_trajectory.npz",
                times=times,
                qpos=qpos_matrix,
            )

            replay_data[
                float(damping)
            ] = (
                candidate,
                qpos_matrix,
                times,
            )

            print(
                "target pose MCP/PIP/DIP = "
                f"{result['target_mcp_deg']:.3f} / "
                f"{result['target_pip_deg']:.3f} / "
                f"{result['target_dip_deg']:.3f} deg"
            )

            print(
                f"RMSE = "
                f"{result['rmse_deg']:.3f} deg"
            )

            print()
            print(
                "FIRST MIDDLE CONTACT:"
            )

            print(
                "  pose MCP/PIP/DIP = "
                f"{result['mcp_at_first_mid_deg']:.3f} / "
                f"{result['pip_at_first_mid_deg']:.3f} / "
                f"{result['dip_at_first_mid_deg']:.3f}"
            )

            print()
            print(
                "MIDDLE -> FIRST TIP:"
            )

            print(
                f"  dMCP = "
                f"{result['delta_mcp_mid_to_tip_deg_g11']:+.3f}"
            )

            print(
                f"  dPIP = "
                f"{result['delta_pip_mid_to_tip_deg_g11']:+.3f}"
            )

            print(
                f"  dDIP = "
                f"{result['delta_dip_mid_to_tip_deg_g11']:+.3f}"
            )

            print()
            print(
                "100 ms AFTER MIDDLE CONTACT:"
            )

            print(
                f"  PIP speed = "
                f"{result['pip_speed_100ms_after_mid_deg_s']:.3f} deg/s"
            )

            print(
                f"  DIP speed = "
                f"{result['dip_speed_100ms_after_mid_deg_s']:.3f} deg/s"
            )

            print(
                f"  DIP/PIP   = "
                f"{result['dip_over_pip_100ms_after_mid']:.3f}"
            )

            print()
            print(
                "CONTACT EPISODES:"
            )

            print(
                f"  middle = "
                f"{result['middle_contact_episodes']}"
            )

            print(
                f"  tip    = "
                f"{result['tip_contact_episodes']}"
            )

            print(
                f"  NO repeated episode count = "
                f"{result['no_rebound_episode_count']}"
            )

            print()
            print(
                "TARGET:"
            )

            print(
                f"  contacts middle/tip = "
                f"{result['target_mid_contact']} / "
                f"{result['target_tip_contact']}"
            )

            print(
                f"  gaps middle/tip = "
                f"{result['target_mid_gap_mm']:+.3f} / "
                f"{result['target_tip_gap_mm']:+.3f} mm"
            )

            print(
                f"  force median/max = "
                f"{result['e2b_hold_tip_force_median_N']:.3f} / "
                f"{result['e2b_hold_tip_force_max_N']:.3f} N"
            )

            print(
                f"  G11 mechanics score = "
                f"{result['mechanical_score_g11']:.3f}"
            )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

        g7.b2.v35.TENDON_K = (
            old_k
        )

        g7.b2.v35.CONTACT_MU = (
            old_mu
        )

        g7.b2.v35.CONTACT_K = (
            old_ck
        )

        contact_replay.CONTACT_DAMPING = (
            old_cd
        )

    # ------------------------------------------------------------
    # Save complete numerical table.
    # ------------------------------------------------------------

    summary_path = (
        out_dir
        / "event_summary.csv"
    )

    with summary_path.open(
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

    # ------------------------------------------------------------
    # Mechanics-first ranking.
    #
    # Candidates with one middle + one tip episode always rank
    # ahead of bouncing candidates.
    # ------------------------------------------------------------

    ranked = sorted(
        rows,
        key=lambda r:
            (
                not r[
                    "no_rebound_episode_count"
                ],
                r[
                    "middle_contact_episodes"
                ]
                +
                r[
                    "tip_contact_episodes"
                ],
                r[
                    "mechanical_score_g11"
                ],
            ),
    )

    print()
    print("=" * 155)

    print(
        "V35G11 — MECHANICS-FIRST SUMMARY"
    )

    print("=" * 155)

    print()

    print(
        "damp   RMSE  epM epT clean  "
        "PIP@M  DIP@M   dPIP   dDIP  "
        "ratio100  gapM   gapT   Fmax   score"
    )

    print("-" * 155)

    for r in ranked:

        print(
            f"{r['joint_damping']:5.3f} "
            f"{r['rmse_deg']:6.3f} "
            f"{r['middle_contact_episodes']:3d} "
            f"{r['tip_contact_episodes']:3d} "
            f"{str(r['no_rebound_episode_count']):>5s} "
            f"{r['pip_at_first_mid_deg']:6.2f} "
            f"{r['dip_at_first_mid_deg']:6.2f} "
            f"{r['delta_pip_mid_to_tip_deg_g11']:+7.3f} "
            f"{r['delta_dip_mid_to_tip_deg_g11']:+7.3f} "
            f"{r['dip_over_pip_100ms_after_mid']:8.3f} "
            f"{r['target_mid_gap_mm']:+6.3f} "
            f"{r['target_tip_gap_mm']:+6.3f} "
            f"{r['e2b_hold_tip_force_max_N']:6.2f} "
            f"{r['mechanical_score_g11']:8.2f}"
        )

    # ------------------------------------------------------------
    # Select up to three visually informative candidates.
    #
    # Candidate 1: mechanics-first best
    # Candidate 2: next distinct damping
    # Candidate 3: next distinct damping
    #
    # These are NOT accepted automatically.
    # ------------------------------------------------------------

    visual_rows = []

    for row in ranked:

        visual_rows.append(
            row
        )

        if len(
            visual_rows
        ) >= 3:
            break

    with (
        out_dir
        / "visual_candidates.txt"
    ).open("w") as f:

        f.write(
            "V35G11 visual candidates\n"
        )

        f.write(
            "========================\n\n"
        )

        f.write(
            "These are NOT automatically accepted.\n"
        )

        f.write(
            "Hardware-like visual behavior is mandatory.\n\n"
        )

        for i, row in enumerate(
            visual_rows,
            start=1,
        ):

            f.write(
                f"{i}. damping = "
                f"{row['joint_damping']:.6f}\n"
            )

            f.write(
                f"   middle episodes = "
                f"{row['middle_contact_episodes']}\n"
            )

            f.write(
                f"   tip episodes = "
                f"{row['tip_contact_episodes']}\n"
            )

            f.write(
                f"   RMSE = "
                f"{row['rmse_deg']:.6f} deg\n"
            )

            f.write(
                f"   mechanics score = "
                f"{row['mechanical_score_g11']:.6f}\n\n"
            )

    with (
        out_dir
        / "manifest.txt"
    ).open("w") as f:

        f.write(
            "V35G11 — index joint viscous damping scan\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"H1 bag:\n{bag}\n\n"
        )

        f.write(
            f"scale = "
            f"{SCALE*1e6:.3f} um/tick\n"
        )

        f.write(
            f"tendon stiffness = "
            f"{TENDON_K:.3f} N/m\n"
        )

        f.write(
            f"tendon damping = "
            f"{TENDON_DAMPING:.3f}\n"
        )

        f.write(
            f"contact mu = "
            f"{CONTACT_MU}\n"
        )

        f.write(
            f"contact stiffness = "
            f"{CONTACT_K}\n"
        )

        f.write(
            f"contact damping = "
            f"{CONTACT_DAMPING}\n\n"
        )

        f.write(
            "common MCP/PIP/DIP damping candidates:\n"
        )

        for value in JOINT_DAMPINGS:

            f.write(
                f"  {value:.6f}\n"
            )

    print()
    print("=" * 155)

    print(
        "MANDATORY VISUAL VALIDATION"
    )

    print("=" * 155)

    print()
    print(
        "Up to three mechanics-first candidates will now "
        "be replayed sequentially."
    )

    print()
    print(
        "For EACH GUI:"
    )

    print(
        "  - watch the complete motion"
    )

    print(
        "  - close the GUI manually when finished"
    )

    print(
        "  - the next candidate will then open"
    )

    print()
    print(
        "ACCEPT ONLY IF YOU SEE:"
    )

    print(
        "  MCP"
    )

    print(
        "  -> middle touch"
    )

    print(
        "  -> PIP continues wrapping"
    )

    print(
        "  -> middle remains constrained"
    )

    print(
        "  -> PIP slows/stops"
    )

    print(
        "  -> DIP visibly rotates"
    )

    print(
        "  -> distal becomes constrained"
    )

    print(
        "  -> DIP stops"
    )

    print()
    print(
        "REJECT:"
    )

    print(
        "  touch -> bounce -> touch -> bounce"
    )

    if not args.no_gui:

        for i, row in enumerate(
            visual_rows,
            start=1,
        ):

            damping = float(
                row[
                    "joint_damping"
                ]
            )

            (
                candidate_xml,
                qpos_matrix,
                times,
            ) = replay_data[
                damping
            ]

            print()
            print("#" * 155)

            print(
                f"VISUAL CANDIDATE "
                f"{i}/{len(visual_rows)}"
            )

            print(
                f"common MCP/PIP/DIP damping = "
                f"{damping:.6f}"
            )

            print(
                f"middle/tip episodes = "
                f"{row['middle_contact_episodes']} / "
                f"{row['tip_contact_episodes']}"
            )

            print(
                f"RMSE = "
                f"{row['rmse_deg']:.3f} deg"
            )

            print("#" * 155)

            g7.exact_gui_replay(
                candidate_xml,
                qpos_matrix,
                times,
                (
                    f"G11 candidate {i}: "
                    f"joint damping="
                    f"{damping:.6f}"
                ),
            )

    print()
    print("=" * 155)

    print(
        "V35G11 COMPLETE"
    )

    print("=" * 155)

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
