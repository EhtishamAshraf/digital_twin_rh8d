#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import mujoco
import numpy as np

import fit_index_contact_v1 as bagbase
import scan_index_contact_friction_v35g7 as g7
import scan_index_dip_stiffness_v35g15b as g15b
import test_index_breakaway_hierarchy_v35g20c as g20c


# ================================================================
# G20D — MCP BACKDRIVE / MIDDLE-CONTACT RETENTION TEST
#
# G20C result:
#   coordinated breakaway hierarchy successfully produces
#
#       MCP -> PIP -> DIP
#
#   in free space and removes repeated TIP contact for the
#   medium/strong candidates.
#
# Remaining failure:
#   middle contact still occurs twice.
#
# Hypothesis:
#   MCP was made sufficiently easy to flex that middle-contact
#   reaction can also back-drive MCP, allowing the middle phalanx
#   to detach.
#
# Freeze successful PIP/DIP hierarchy:
#   PIP frictionloss = 0.003 Nm
#   DIP frictionloss = 0.006 Nm
#
# Test only MCP resistance.
#
# NO:
#   contact-dependent switching
#   joint locking
#   tendon routing changes
#   spring changes
#   collision changes
#   trajectory prescription
# ================================================================


SCALE = 5.70e-6
CONTACT_MU = 0.05

PIP_FRICTION = 0.003
DIP_FRICTION = 0.006

MCP_VALUES = [
    0.00035,
    0.00070,
    0.00100,
    0.00130,
    0.00160,
    0.00200,
]


def candidate_name(x):

    return (
        f"mcp_{x:.5f}"
        .replace(".", "p")
    )


def episodes(
    trace,
    field,
):

    result = []

    active = False
    start = None

    for i, row in enumerate(trace):

        now = bool(
            row[field]
        )

        if now and not active:

            active = True
            start = i

        elif active and not now:

            result.append(
                (
                    start,
                    i - 1,
                )
            )

            active = False
            start = None

    if active:

        result.append(
            (
                start,
                len(trace) - 1,
            )
        )

    return result


def q_at(
    times,
    qpos,
    t,
):

    times = np.asarray(
        times,
        dtype=float,
    )

    qpos = np.asarray(
        qpos,
        dtype=float,
    )

    out = np.empty(
        qpos.shape[1],
        dtype=float,
    )

    for j in range(
        qpos.shape[1]
    ):

        out[j] = np.interp(
            t,
            times,
            qpos[:, j],
        )

    return out


def joint_qadr(
    model,
    name,
):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:

        raise RuntimeError(
            f"Joint not found: {name}"
        )

    return int(
        model.jnt_qposadr[
            jid
        ]
    )


def pose_deg(
    q,
    addresses,
):

    return np.asarray(
        [
            np.rad2deg(
                q[a]
            )
            for a in addresses
        ],
        dtype=float,
    )


def first_middle_episode_metrics(
    xml,
    trace,
    qpos,
    times,
):

    mids = episodes(
        trace,
        "mid_contact",
    )

    if not mids:

        return {
            "first_mid_duration_s":
                np.nan,

            "first_mid_dMCP_deg":
                np.nan,

            "first_mid_dPIP_deg":
                np.nan,

            "first_mid_dDIP_deg":
                np.nan,

            "first_mid_end_mcp_deg":
                np.nan,

            "first_mid_end_pip_deg":
                np.nan,

            "first_mid_end_dip_deg":
                np.nan,

            "mcp_backdrive_first_mid_deg":
                np.nan,

            "gap_to_second_middle_s":
                np.nan,
        }

    model = mujoco.MjModel.from_xml_path(
        str(xml)
    )

    addresses = [
        joint_qadr(
            model,
            g20c.MCP,
        ),
        joint_qadr(
            model,
            g20c.PIP,
        ),
        joint_qadr(
            model,
            g20c.DIP,
        ),
    ]

    a, b = mids[0]

    t0 = float(
        trace[a]["time_s"]
    )

    t1 = float(
        trace[b]["time_s"]
    )

    q0 = q_at(
        times,
        qpos,
        t0,
    )

    q1 = q_at(
        times,
        qpos,
        t1,
    )

    p0 = pose_deg(
        q0,
        addresses,
    )

    p1 = pose_deg(
        q1,
        addresses,
    )

    delta = (
        p1
        -
        p0
    )

    # ------------------------------------------------------------
    # Find minimum MCP angle during the first middle-contact
    # episode. Flexion is positive; a drop below the contact-start
    # MCP angle is backdrive / extension.
    # ------------------------------------------------------------

    times_np = np.asarray(
        times,
        dtype=float,
    )

    qpos_np = np.asarray(
        qpos,
        dtype=float,
    )

    mask = (
        (times_np >= t0)
        &
        (times_np <= t1)
    )

    mcp_series = np.rad2deg(
        qpos_np[
            mask,
            addresses[0],
        ]
    )

    if len(mcp_series):

        minimum_mcp = float(
            np.min(
                mcp_series
            )
        )

        backdrive = max(
            0.0,
            float(
                p0[0]
                -
                minimum_mcp
            ),
        )

    else:

        backdrive = np.nan

    if len(mids) >= 2:

        second_start = float(
            trace[
                mids[1][0]
            ][
                "time_s"
            ]
        )

        gap = (
            second_start
            -
            t1
        )

    else:

        gap = np.nan

    return {
        "first_mid_duration_s":
            float(
                t1 - t0
            ),

        "first_mid_dMCP_deg":
            float(
                delta[0]
            ),

        "first_mid_dPIP_deg":
            float(
                delta[1]
            ),

        "first_mid_dDIP_deg":
            float(
                delta[2]
            ),

        "first_mid_end_mcp_deg":
            float(
                p1[0]
            ),

        "first_mid_end_pip_deg":
            float(
                p1[1]
            ),

        "first_mid_end_dip_deg":
            float(
                p1[2]
            ),

        "mcp_backdrive_first_mid_deg":
            float(
                backdrive
            ),

        "gap_to_second_middle_s":
            float(
                gap
            )
            if np.isfinite(
                gap
            )
            else np.nan,
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
            "v35g20d_mcp_backdrive"
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

    (
        states,
        force_data,
        encoder_data,
    ) = bagbase.read_bag(
        bag
    )

    encoder_data = np.asarray(
        encoder_data,
        dtype=float,
    ).copy()

    encoder_data[:, 1] = (
        bagbase.median_filter_5(
            encoder_data[:, 1]
        )
    )

    transitions = (
        bagbase.compress_states(
            states
        )
    )

    (
        approach_start,
        regulation_start,
        hold_start,
        hold_end,
    ) = bagbase.find_intervals(
        transitions
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

    old_mu = float(
        g7.b2.v35.CONTACT_MU
    )

    rows = []
    trajectories = {}

    try:

        g7.b2.v35.METERS_PER_TICK = (
            SCALE
        )

        g7.b2.v35.CONTACT_MU = (
            CONTACT_MU
        )

        print()
        print("=" * 160)
        print(
            "G20D — MCP BACKDRIVE / "
            "MIDDLE-CONTACT RETENTION"
        )
        print("=" * 160)

        print()
        print(
            f"Frozen PIP frictionloss = "
            f"{PIP_FRICTION:.6f} Nm"
        )

        print(
            f"Frozen DIP frictionloss = "
            f"{DIP_FRICTION:.6f} Nm"
        )

        for mcp_friction in MCP_VALUES:

            name = candidate_name(
                mcp_friction
            )

            candidate = {
                "name":
                    name,

                "mcp":
                    mcp_friction,

                "pip":
                    PIP_FRICTION,

                "dip":
                    DIP_FRICTION,
            }

            print()
            print("=" * 160)

            print(
                f"CANDIDATE {name}"
            )

            print(
                "frictionloss MCP/PIP/DIP = "
                f"{mcp_friction:.6f} / "
                f"{PIP_FRICTION:.6f} / "
                f"{DIP_FRICTION:.6f}"
            )

            print("=" * 160)

            # ====================================================
            # FREE SPACE
            # ====================================================

            free_xml = (
                out_dir
                /
                f"{name}_free.xml"
            )

            g20c.make_model(
                baseline_xml,
                free_xml,
                candidate,
                free_space=True,
            )

            (
                free_result,
                free_trace,
                free_qpos,
                free_times,
            ) = g7.run_scale(
                SCALE,
                CONTACT_MU,
                free_xml,
                bag,
                contact_replay,
            )

            fm = g20c.free_space_metrics(
                free_xml,
                free_times,
                free_qpos,
                encoder_data,
                approach_start,
                regulation_start,
            )

            print()
            print(
                "FREE SPACE:"
            )

            print(
                "  1deg activation MCP/PIP/DIP = "
                f"{fm['mcp_activation_1deg_tick']:.1f} / "
                f"{fm['pip_activation_1deg_tick']:.1f} / "
                f"{fm['dip_activation_1deg_tick']:.1f}"
            )

            print(
                "  3deg activation MCP/PIP/DIP = "
                f"{fm['mcp_activation_3deg_tick']:.1f} / "
                f"{fm['pip_activation_3deg_tick']:.1f} / "
                f"{fm['dip_activation_3deg_tick']:.1f}"
            )

            print(
                "  ordering 1deg / 3deg = "
                f"{fm['order_1deg_pass']} / "
                f"{fm['order_3deg_pass']}"
            )

            print(
                "  @300 ticks MCP/PIP/DIP = "
                f"{fm.get('mcp_at_300_deg', np.nan):+.3f} / "
                f"{fm.get('pip_at_300_deg', np.nan):+.3f} / "
                f"{fm.get('dip_at_300_deg', np.nan):+.3f}"
            )

            # ====================================================
            # H1 CONTACT
            # ====================================================

            contact_xml = (
                out_dir
                /
                f"{name}_contact.xml"
            )

            g20c.make_model(
                baseline_xml,
                contact_xml,
                candidate,
                free_space=False,
            )

            (
                result,
                trace,
                qpos,
                times,
            ) = g7.run_scale(
                SCALE,
                CONTACT_MU,
                contact_xml,
                bag,
                contact_replay,
            )

            mechanics = (
                g15b.mechanics(
                    trace
                )
            )

            episode = (
                first_middle_episode_metrics(
                    contact_xml,
                    trace,
                    qpos,
                    times,
                )
            )

            row = dict(
                result
            )

            row.update(
                mechanics
            )

            row.update(
                episode
            )

            row.update(
                {
                    "candidate":
                        name,

                    "mcp_frictionloss":
                        mcp_friction,

                    "pip_frictionloss":
                        PIP_FRICTION,

                    "dip_frictionloss":
                        DIP_FRICTION,

                    "free_order_1deg":
                        fm[
                            "order_1deg_pass"
                        ],

                    "free_order_3deg":
                        fm[
                            "order_3deg_pass"
                        ],

                    "free_mcp_activation_1deg":
                        fm[
                            "mcp_activation_1deg_tick"
                        ],

                    "free_pip_activation_1deg":
                        fm[
                            "pip_activation_1deg_tick"
                        ],

                    "free_dip_activation_1deg":
                        fm[
                            "dip_activation_1deg_tick"
                        ],
                }
            )

            rows.append(
                row
            )

            trajectories[
                name
            ] = (
                contact_xml,
                qpos,
                times,
            )

            g7.save_trace_csv(
                out_dir
                /
                f"{name}_contact_timeseries.csv",
                trace,
            )

            print()
            print(
                "H1 CONTACT:"
            )

            print(
                f"  target pose = "
                f"{row['target_mcp_deg']:.3f} / "
                f"{row['target_pip_deg']:.3f} / "
                f"{row['target_dip_deg']:.3f}"
            )

            print(
                f"  RMSE = "
                f"{row['rmse_deg']:.3f} deg"
            )

            print(
                "  episodes M/T = "
                f"{row['middle_episodes']} / "
                f"{row['tip_episodes']}"
            )

            print(
                "  target contacts M/T = "
                f"{row['target_mid_contact']} / "
                f"{row['target_tip_contact']}"
            )

            print(
                "  first-middle pose = "
                f"{row['mcp_first_middle_deg']:.3f} / "
                f"{row['pip_first_middle_deg']:.3f} / "
                f"{row['dip_first_middle_deg']:.3f}"
            )

            print()
            print(
                "FIRST MIDDLE EPISODE:"
            )

            print(
                f"  duration = "
                f"{row['first_mid_duration_s']:.4f} s"
            )

            print(
                "  dMCP/dPIP/dDIP = "
                f"{row['first_mid_dMCP_deg']:+.3f} / "
                f"{row['first_mid_dPIP_deg']:+.3f} / "
                f"{row['first_mid_dDIP_deg']:+.3f} deg"
            )

            print(
                f"  MCP backdrive = "
                f"{row['mcp_backdrive_first_mid_deg']:.3f} deg"
            )

            print(
                "  first-middle END pose = "
                f"{row['first_mid_end_mcp_deg']:.3f} / "
                f"{row['first_mid_end_pip_deg']:.3f} / "
                f"{row['first_mid_end_dip_deg']:.3f}"
            )

            print(
                f"  gap to second middle = "
                f"{row['gap_to_second_middle_s']:.4f} s"
            )

            print()
            print(
                "MIDDLE -> TIP:"
            )

            print(
                f"  delay = "
                f"{row['middle_to_tip_delay_s']:.4f} s"
            )

            print(
                "  dPIP/dDIP = "
                f"{row['delta_pip_middle_to_tip_deg']:+.3f} / "
                f"{row['delta_dip_middle_to_tip_deg']:+.3f}"
            )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

        g7.b2.v35.CONTACT_MU = (
            old_mu
        )

    # ============================================================
    # Save
    # ============================================================

    csv_path = (
        out_dir
        /
        "g20d_summary.csv"
    )

    with csv_path.open(
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

    # ============================================================
    # Rank:
    #   1. preserve free-space staging
    #   2. middle stays in contact
    #   3. tip stays in contact
    #   4. both contacts present at target
    #   5. reduce MCP backdrive
    #   6. RMSE
    # ============================================================

    ranked = sorted(
        rows,
        key=lambda r: (
            not r[
                "free_order_1deg"
            ],

            not r[
                "free_order_3deg"
            ],

            not (
                r[
                    "middle_episodes"
                ]
                ==
                1
                and
                r[
                    "target_mid_contact"
                ]
            ),

            not (
                r[
                    "tip_episodes"
                ]
                ==
                1
                and
                r[
                    "target_tip_contact"
                ]
            ),

            not (
                r[
                    "target_mid_contact"
                ]
                and
                r[
                    "target_tip_contact"
                ]
            ),

            r[
                "mcp_backdrive_first_mid_deg"
            ]
            if np.isfinite(
                r[
                    "mcp_backdrive_first_mid_deg"
                ]
            )
            else 999.0,

            r[
                "rmse_deg"
            ],
        ),
    )

    print()
    print("=" * 190)
    print(
        "G20D FINAL SUMMARY"
    )
    print("=" * 190)

    print(
        "MCPfric  ord1 ord3 "
        "epM epT targetM targetT "
        "midDur backdrive "
        "PIP@M DIP@M "
        "delay dPIP dDIP RMSE"
    )

    print("-" * 190)

    for r in ranked:

        print(
            f"{r['mcp_frictionloss']:7.5f} "
            f"{str(r['free_order_1deg']):>5s} "
            f"{str(r['free_order_3deg']):>5s} "
            f"{r['middle_episodes']:3d} "
            f"{r['tip_episodes']:3d} "
            f"{str(r['target_mid_contact']):>7s} "
            f"{str(r['target_tip_contact']):>7s} "
            f"{r['first_mid_duration_s']:7.3f} "
            f"{r['mcp_backdrive_first_mid_deg']:9.3f} "
            f"{r['pip_first_middle_deg']:6.2f} "
            f"{r['dip_first_middle_deg']:6.2f} "
            f"{r['middle_to_tip_delay_s']:6.3f} "
            f"{r['delta_pip_middle_to_tip_deg']:+6.2f} "
            f"{r['delta_dip_middle_to_tip_deg']:+6.2f} "
            f"{r['rmse_deg']:6.3f}"
        )

    # ============================================================
    # GUI — top three mechanically ranked H1 candidates.
    # ============================================================

    if not args.no_gui:

        visual = ranked[:3]

        print()
        print("=" * 140)
        print(
            "MANDATORY G20D GUI VALIDATION"
        )
        print("=" * 140)

        print(
            "Close each GUI manually."
        )

        print(
            "Watch specifically whether FIRST middle contact "
            "is retained while PIP wraps."
        )

        for i, r in enumerate(
            visual,
            start=1,
        ):

            name = r[
                "candidate"
            ]

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
                f"GUI {i}/{len(visual)} — {name}"
            )

            print(
                f"MCP frictionloss = "
                f"{r['mcp_frictionloss']:.6f}"
            )

            print(
                "episodes M/T = "
                f"{r['middle_episodes']} / "
                f"{r['tip_episodes']}"
            )

            print(
                f"MCP first-contact backdrive = "
                f"{r['mcp_backdrive_first_mid_deg']:.3f} deg"
            )

            print(
                f"RMSE = "
                f"{r['rmse_deg']:.3f}"
            )

            print("#" * 130)

            g7.exact_gui_replay(
                xml,
                qpos,
                times,
                (
                    f"G20D {name}"
                ),
            )

    with (
        out_dir
        /
        "manifest.txt"
    ).open("w") as f:

        f.write(
            "V35G20D — MCP backdrive/contact-retention test\n\n"
        )

        f.write(
            "G20C established that MCP<PIP<DIP "
            "breakaway hierarchy fixes free-space staging "
            "and can eliminate repeated distal contact.\n"
        )

        f.write(
            "G20D freezes PIP=0.003 and DIP=0.006 Nm "
            "and varies only MCP frictionloss.\n"
        )

        f.write(
            "Purpose: determine whether low MCP resistance "
            "allows contact reaction to back-drive MCP and "
            "cause first middle-contact loss.\n"
        )

        f.write(
            "No contact-state rules, locks, tendon changes, "
            "spring changes or collision changes.\n"
        )

    print()
    print(
        f"Saved:\n  {csv_path}"
    )

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
