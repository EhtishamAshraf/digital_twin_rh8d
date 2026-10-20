#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

import scan_index_contact_friction_v35g7 as g7


# ================================================================
# V35G13A — CONTACT COMPLIANCE / STIFFNESS SCAN
#
# G12 diagnosis:
#   tendon routing/moment arms are present,
#   but first simultaneous contact produces large constraint torque
#   and the finger redistributes through rebound/free motion.
#
# G13A changes ONLY contact stiffness.
#
# Success requires BOTH:
#   1. hardware-like sequential wrapping
#   2. low pose RMSE
#
# RMSE alone never selects the final model.
# ================================================================


SCALE = 5.70e-6
TENDON_K = 2100.0
CONTACT_MU = 0.05
CONTACT_DAMPING = 20.0

CONTACT_STIFFNESSES = [
    1000.0,
    500.0,
    250.0,
    125.0,
    60.0,
]


def tag(x):
    return (
        f"{x:.1f}"
        .replace(".", "p")
    )


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
    count = 0
    previous = False

    for row in trace:
        value = bool(row[field])

        if value and not previous:
            count += 1

        previous = value

    return count


def extra_metrics(trace):

    times = np.asarray(
        [r["time_s"] for r in trace],
        dtype=float,
    )

    mcp = np.asarray(
        [r["mcp_deg"] for r in trace],
        dtype=float,
    )

    pip = np.asarray(
        [r["pip_deg"] for r in trace],
        dtype=float,
    )

    dip = np.asarray(
        [r["dip_deg"] for r in trace],
        dtype=float,
    )

    mid = np.asarray(
        [bool(r["mid_contact"]) for r in trace],
        dtype=bool,
    )

    tip = np.asarray(
        [bool(r["tip_contact"]) for r in trace],
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

        "mcp_at_first_mid_deg": np.nan,
        "pip_at_first_mid_deg": np.nan,
        "dip_at_first_mid_deg": np.nan,

        "mcp_at_first_tip_deg": np.nan,
        "pip_at_first_tip_deg": np.nan,
        "dip_at_first_tip_deg": np.nan,

        "delta_mcp_mid_to_tip_deg": np.nan,
        "delta_pip_mid_to_tip_deg_g13": np.nan,
        "delta_dip_mid_to_tip_deg_g13": np.nan,

        "dip_over_pip_mid_to_tip_g13": np.nan,

        "simultaneous_contact_longest_s": 0.0,
    }

    if i_mid is not None:

        result[
            "mcp_at_first_mid_deg"
        ] = float(mcp[i_mid])

        result[
            "pip_at_first_mid_deg"
        ] = float(pip[i_mid])

        result[
            "dip_at_first_mid_deg"
        ] = float(dip[i_mid])

    if (
        i_mid is not None
        and
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

        dmcp = float(
            mcp[i_tip]
            -
            mcp[i_mid]
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
            "delta_mcp_mid_to_tip_deg"
        ] = dmcp

        result[
            "delta_pip_mid_to_tip_deg_g13"
        ] = dpip

        result[
            "delta_dip_mid_to_tip_deg_g13"
        ] = ddip

        result[
            "dip_over_pip_mid_to_tip_g13"
        ] = (
            abs(ddip)
            /
            max(
                abs(dpip),
                1e-9,
            )
        )

    # ------------------------------------------------------------
    # Longest uninterrupted simultaneous middle+tip contact.
    # ------------------------------------------------------------

    overlap = (
        mid
        &
        tip
    )

    longest = 0.0
    start = None

    for i, value in enumerate(overlap):

        if value and start is None:
            start = i

        if (
            not value
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
        "simultaneous_contact_longest_s"
    ] = longest

    return result


def mechanics_score(r):
    """
    Lower is better.

    Only used for deciding which trajectories to inspect.
    It cannot automatically accept a candidate.
    """

    score = 0.0

    # ------------------------------------------------------------
    # Repeated contact is the main failure we are attacking.
    # ------------------------------------------------------------

    score += (
        500.0
        *
        max(
            0,
            r["middle_contact_episodes"] - 1,
        )
    )

    score += (
        500.0
        *
        max(
            0,
            r["tip_contact_episodes"] - 1,
        )
    )

    if not r["mid_retained_to_tip"]:
        score += 500.0

    # ------------------------------------------------------------
    # Must reach physical target contacts.
    # ------------------------------------------------------------

    if not r["target_mid_contact"]:
        score += 250.0

    if not r["target_tip_contact"]:
        score += 250.0

    # ------------------------------------------------------------
    # Excess penetration.
    # ------------------------------------------------------------

    score += (
        120.0
        *
        max(
            0.0,
            -r["target_mid_gap_mm"],
        )
    )

    score += (
        120.0
        *
        max(
            0.0,
            -r["target_tip_gap_mm"],
        )
    )

    # ------------------------------------------------------------
    # Need PIP continuation after first middle contact.
    # ------------------------------------------------------------

    dpip = r[
        "delta_pip_mid_to_tip_deg_g13"
    ]

    if np.isfinite(dpip):

        score += (
            80.0
            *
            max(
                0.0,
                0.30 - dpip,
            )
        )

    else:
        score += 200.0

    # ------------------------------------------------------------
    # Need DIP participation, not a frozen DIP.
    # ------------------------------------------------------------

    ddip = r[
        "delta_dip_mid_to_tip_deg_g13"
    ]

    if np.isfinite(ddip):

        score += (
            120.0
            *
            max(
                0.0,
                0.20 - ddip,
            )
        )

    else:
        score += 250.0

    ratio = r[
        "dip_over_pip_mid_to_tip_g13"
    ]

    if np.isfinite(ratio):

        score += (
            100.0
            *
            max(
                0.0,
                0.30 - ratio,
            )
        )

    else:
        score += 100.0

    # ------------------------------------------------------------
    # Large force spikes.
    # ------------------------------------------------------------

    fmax = r[
        "e2b_hold_tip_force_max_N"
    ]

    if np.isfinite(fmax):

        score += (
            15.0
            *
            max(
                0.0,
                fmax - 2.0,
            )
        )

    # ------------------------------------------------------------
    # RMSE matters, but only after mechanics.
    # ------------------------------------------------------------

    score += (
        8.0
        *
        r["rmse_deg"]
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
            /
            "rh8d_results"
            /
            "v35g13a_contact_compliance"
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


    # ------------------------------------------------------------
    # Reuse G7 zero-margin model and validated recorder.
    # ------------------------------------------------------------

    g7.b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    model_xml = (
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
    replay_data = {}


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

        contact_replay.CONTACT_DAMPING = (
            CONTACT_DAMPING
        )


        print()
        print("=" * 150)

        print(
            "V35G13A — COMPLIANT CONTACT / "
            "STIFFNESS SCAN"
        )

        print("=" * 150)

        print()
        print(
            "G12 hypothesis:"
        )

        print(
            "  rigid/strong contact reaction is ejecting "
            "the finger before smooth PIP->DIP redistribution."
        )

        print()
        print("FROZEN:")

        print(
            f"  scale             = "
            f"{SCALE*1e6:.3f} um/tick"
        )

        print(
            f"  tendon stiffness  = "
            f"{TENDON_K:.1f} N/m"
        )

        print(
            "  tendon damping    = 2"
        )

        print(
            f"  contact mu        = "
            f"{CONTACT_MU}"
        )

        print(
            f"  contact damping   = "
            f"{CONTACT_DAMPING}"
        )

        print(
            "  joint damping     = "
            "0.015 / 0.015 / 0.015"
        )

        print()
        print(
            "VARIABLE contact stiffness:"
        )

        print(
            "  "
            +
            ", ".join(
                str(x)
                for x in CONTACT_STIFFNESSES
            )
        )


        for contact_k in (
            CONTACT_STIFFNESSES
        ):

            print()
            print("-" * 150)

            print(
                f"CONTACT STIFFNESS = "
                f"{contact_k:.1f}"
            )

            print("-" * 150)


            g7.b2.v35.CONTACT_K = (
                float(contact_k)
            )


            (
                result,
                trace,
                qpos_matrix,
                times,
            ) = g7.run_scale(
                SCALE,
                CONTACT_MU,
                model_xml,
                bag,
                contact_replay,
            )


            result[
                "contact_stiffness_g13"
            ] = float(
                contact_k
            )


            result.update(
                extra_metrics(
                    trace
                )
            )


            result[
                "mechanical_score_g13"
            ] = mechanics_score(
                result
            )


            result[
                "single_middle_episode"
            ] = bool(
                result[
                    "middle_contact_episodes"
                ]
                == 1
            )

            result[
                "single_tip_episode"
            ] = bool(
                result[
                    "tip_contact_episodes"
                ]
                == 1
            )

            result[
                "single_contact_sequence"
            ] = bool(
                result[
                    "single_middle_episode"
                ]
                and
                result[
                    "single_tip_episode"
                ]
            )


            rows.append(
                result
            )


            name = (
                f"contactK_{tag(contact_k)}"
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
                qpos=qpos_matrix,
            )


            replay_data[
                float(contact_k)
            ] = (
                qpos_matrix,
                times,
            )


            print(
                "Target pose MCP/PIP/DIP = "
                f"{result['target_mcp_deg']:.3f} / "
                f"{result['target_pip_deg']:.3f} / "
                f"{result['target_dip_deg']:.3f}"
            )

            print(
                f"RMSE = "
                f"{result['rmse_deg']:.3f} deg"
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
                f"  single sequence = "
                f"{result['single_contact_sequence']}"
            )


            print()
            print(
                "FIRST MIDDLE CONTACT:"
            )

            print(
                "  pose = "
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
                f"{result['delta_mcp_mid_to_tip_deg']:+.3f}"
            )

            print(
                f"  dPIP = "
                f"{result['delta_pip_mid_to_tip_deg_g13']:+.3f}"
            )

            print(
                f"  dDIP = "
                f"{result['delta_dip_mid_to_tip_deg_g13']:+.3f}"
            )

            print(
                f"  DIP/PIP = "
                f"{result['dip_over_pip_mid_to_tip_g13']:.3f}"
            )


            print()
            print(
                "CONTACT STABILITY:"
            )

            print(
                f"  longest simultaneous contact = "
                f"{result['simultaneous_contact_longest_s']:.3f} s"
            )

            print(
                f"  target contacts = "
                f"{result['target_mid_contact']} / "
                f"{result['target_tip_contact']}"
            )

            print(
                f"  target gaps = "
                f"{result['target_mid_gap_mm']:+.3f} / "
                f"{result['target_tip_gap_mm']:+.3f} mm"
            )

            print(
                f"  tip force median/max = "
                f"{result['e2b_hold_tip_force_median_N']:.3f} / "
                f"{result['e2b_hold_tip_force_max_N']:.3f} N"
            )

            print(
                f"  mechanics score = "
                f"{result['mechanical_score_g13']:.3f}"
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


    # ------------------------------------------------------------
    # Save summary.
    # ------------------------------------------------------------

    summary_path = (
        out_dir
        /
        "event_summary.csv"
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
    # Single contact sequence ranks before bouncing candidates.
    # ------------------------------------------------------------

    ranked = sorted(
        rows,
        key=lambda r:
            (
                not r[
                    "single_contact_sequence"
                ],

                r[
                    "middle_contact_episodes"
                ]
                +
                r[
                    "tip_contact_episodes"
                ],

                r[
                    "mechanical_score_g13"
                ],
            ),
    )


    print()
    print("=" * 150)

    print(
        "V35G13A — MECHANICS-FIRST SUMMARY"
    )

    print("=" * 150)

    print()

    print(
        "Kcontact  RMSE epM epT clean "
        "PIP@M DIP@M dPIP dDIP ratio "
        "overlap gapM gapT Fmax score"
    )

    print("-" * 150)


    for r in ranked:

        print(
            f"{r['contact_stiffness_g13']:8.1f} "
            f"{r['rmse_deg']:5.3f} "
            f"{r['middle_contact_episodes']:3d} "
            f"{r['tip_contact_episodes']:3d} "
            f"{str(r['single_contact_sequence']):>5s} "
            f"{r['pip_at_first_mid_deg']:5.2f} "
            f"{r['dip_at_first_mid_deg']:5.2f} "
            f"{r['delta_pip_mid_to_tip_deg_g13']:+5.2f} "
            f"{r['delta_dip_mid_to_tip_deg_g13']:+5.2f} "
            f"{r['dip_over_pip_mid_to_tip_g13']:5.2f} "
            f"{r['simultaneous_contact_longest_s']:6.3f} "
            f"{r['target_mid_gap_mm']:+5.2f} "
            f"{r['target_tip_gap_mm']:+5.2f} "
            f"{r['e2b_hold_tip_force_max_N']:5.2f} "
            f"{r['mechanical_score_g13']:7.2f}"
        )


    # ------------------------------------------------------------
    # Choose three most informative candidates for GUI.
    # ------------------------------------------------------------

    visual_rows = (
        ranked[:3]
    )


    with (
        out_dir
        /
        "visual_candidates.txt"
    ).open("w") as f:

        f.write(
            "V35G13A VISUAL CANDIDATES\n"
        )

        f.write(
            "=========================\n\n"
        )

        f.write(
            "NOT automatically accepted.\n"
        )

        f.write(
            "Both hardware-like behavior and low RMSE are required.\n\n"
        )

        for i, row in enumerate(
            visual_rows,
            start=1,
        ):

            f.write(
                f"Candidate {i}\n"
            )

            f.write(
                f"  contact stiffness = "
                f"{row['contact_stiffness_g13']:.3f}\n"
            )

            f.write(
                f"  middle episodes = "
                f"{row['middle_contact_episodes']}\n"
            )

            f.write(
                f"  tip episodes = "
                f"{row['tip_contact_episodes']}\n"
            )

            f.write(
                f"  RMSE = "
                f"{row['rmse_deg']:.6f} deg\n"
            )

            f.write(
                f"  dPIP middle->tip = "
                f"{row['delta_pip_mid_to_tip_deg_g13']:.6f} deg\n"
            )

            f.write(
                f"  dDIP middle->tip = "
                f"{row['delta_dip_mid_to_tip_deg_g13']:.6f} deg\n\n"
            )


    with (
        out_dir
        /
        "manifest.txt"
    ).open("w") as f:

        f.write(
            "V35G13A — compliant contact stiffness scan\n\n"
        )

        f.write(
            f"scale = "
            f"{SCALE*1e6:.3f} um/tick\n"
        )

        f.write(
            f"tendon stiffness = "
            f"{TENDON_K:.1f} N/m\n"
        )

        f.write(
            f"contact mu = "
            f"{CONTACT_MU}\n"
        )

        f.write(
            f"contact damping = "
            f"{CONTACT_DAMPING}\n\n"
        )

        f.write(
            "contact stiffness candidates:\n"
        )

        for x in (
            CONTACT_STIFFNESSES
        ):

            f.write(
                f"  {x:.3f}\n"
            )

        f.write(
            "\nSuccess requires:\n"
        )

        f.write(
            "  hardware-like sequential wrapping\n"
        )

        f.write(
            "  AND low pose RMSE\n"
        )


    # ------------------------------------------------------------
    # Mandatory GUI validation.
    # ------------------------------------------------------------

    print()
    print("=" * 150)

    print(
        "MANDATORY VISUAL VALIDATION"
    )

    print("=" * 150)

    print()
    print(
        "Three mechanics-first candidates will be replayed."
    )

    print()
    print(
        "Close each GUI manually to open the next candidate."
    )

    print()
    print(
        "ACCEPTANCE SEQUENCE:"
    )

    print(
        "  MCP"
    )

    print(
        "  -> middle touches"
    )

    print(
        "  -> PIP continues wrapping"
    )

    print(
        "  -> middle stays constrained"
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
        "HARD REJECTION:"
    )

    print(
        "  touch -> bounce -> touch -> bounce"
    )


    if not args.no_gui:

        # run_scale reads this global during replay.
        for i, row in enumerate(
            visual_rows,
            start=1,
        ):

            contact_k = float(
                row[
                    "contact_stiffness_g13"
                ]
            )

            (
                qpos_matrix,
                times,
            ) = replay_data[
                contact_k
            ]


            print()
            print("#" * 150)

            print(
                f"VISUAL CANDIDATE "
                f"{i}/{len(visual_rows)}"
            )

            print(
                f"contact stiffness = "
                f"{contact_k:.1f}"
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

            print(
                f"middle->tip dPIP/dDIP = "
                f"{row['delta_pip_mid_to_tip_deg_g13']:+.3f} / "
                f"{row['delta_dip_mid_to_tip_deg_g13']:+.3f} deg"
            )

            print("#" * 150)


            g7.exact_gui_replay(
                model_xml,
                qpos_matrix,
                times,
                (
                    f"G13A candidate {i}: "
                    f"contact K="
                    f"{contact_k:.1f}"
                ),
            )


    print()
    print("=" * 150)

    print(
        "V35G13A COMPLETE"
    )

    print("=" * 150)

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
