#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import fit_index_contact_v1 as bagbase
import scan_index_contact_friction_v35g7 as g7
import scan_index_dip_stiffness_v35g15b as g15b


# =====================================================================
# G20C — PASSIVE BREAKAWAY HIERARCHY TEST
#
# G20A proved:
#   - the current spatial tendon CAN redistribute
#   - MCP alone can be driven
#   - PIP alone can be driven
#   - DIP alone can be driven
#
# Problem:
#   with all joints free, MCP/PIP/DIP activate almost together.
#
# Hypothesis:
#   the passive breakaway hierarchy is wrong.
#
# Desired qualitative hierarchy:
#
#       MCP easiest
#           <
#       PIP intermediate
#           <
#       DIP hardest
#
# This is NOT contact-conditioned switching.
# frictionloss is configuration-independent Coulomb/breakaway resistance.
#
# NO:
#   contact switching
#   joint locking
#   prescribed phalanx trajectories
#   tendon topology change
#   spring stiffness changes
#   contact changes
# =====================================================================


SCALE = 5.70e-6
CONTACT_MU = 0.05

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

FIXTURE = "index_contact_fixture"


# ---------------------------------------------------------------------
# IMPORTANT:
#
# Existing near-open spring torques are approximately:
#
#   MCP  = 0.0045 * 0.10 = 0.00045 Nm
#   PIP  = 0.0015 * 0.10 = 0.00015 Nm
#   DIP  = ~0 near q=0
#
# These friction values therefore generate increasingly stronger
# total breakaway resistance from MCP -> PIP -> DIP.
# ---------------------------------------------------------------------

CANDIDATES = [
    {
        "name": "baseline",
        "mcp": 0.00200,
        "pip": 0.00125,
        "dip": 0.00000,
    },

    {
        "name": "hierarchy_mild",
        "mcp": 0.00075,
        "pip": 0.00200,
        "dip": 0.00300,
    },

    {
        "name": "hierarchy_medium",
        "mcp": 0.00050,
        "pip": 0.00250,
        "dip": 0.00450,
    },

    {
        "name": "hierarchy_strong",
        "mcp": 0.00035,
        "pip": 0.00300,
        "dip": 0.00600,
    },

    {
        "name": "hierarchy_high_dip",
        "mcp": 0.00035,
        "pip": 0.00300,
        "dip": 0.00800,
    },
]


ACTIVATION_THRESHOLDS_DEG = [
    1.0,
    3.0,
]


CHECKPOINTS = [
    300.0,
    500.0,
    700.0,
    850.0,
    1000.0,
    1027.0,
]


def find_joint(root, name):

    for joint in root.iter("joint"):

        if joint.get("name") == name:
            return joint

    raise RuntimeError(
        f"Joint not found: {name}"
    )


def find_geom(root, name):

    for geom in root.iter("geom"):

        if geom.get("name") == name:
            return geom

    raise RuntimeError(
        f"Geom not found: {name}"
    )


def make_model(
    source,
    destination,
    candidate,
    free_space,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    values = {
        MCP: candidate["mcp"],
        PIP: candidate["pip"],
        DIP: candidate["dip"],
    }

    for name, friction in values.items():

        joint = find_joint(
            root,
            name,
        )

        joint.set(
            "frictionloss",
            f"{friction:.12g}",
        )

    if free_space:

        fixture = find_geom(
            root,
            FIXTURE,
        )

        fixture.set(
            "contype",
            "0",
        )

        fixture.set(
            "conaffinity",
            "0",
        )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )


def joint_qadr(model, name):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(
            f"MuJoCo joint not found: {name}"
        )

    return int(
        model.jnt_qposadr[jid]
    )


def q_at_time(
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


def encoder_at_time(
    encoder,
    t,
):

    return float(
        np.interp(
            t,
            encoder[:, 0],
            encoder[:, 1],
        )
    )


def first_tick_for_angle(
    times,
    angle_deg,
    encoder,
    start,
    end,
    threshold,
):

    times = np.asarray(
        times,
        dtype=float,
    )

    angle_deg = np.asarray(
        angle_deg,
        dtype=float,
    )

    mask = (
        (times >= start)
        &
        (times <= end)
    )

    ids = np.flatnonzero(
        mask
        &
        (
            angle_deg
            >=
            threshold
        )
    )

    if not len(ids):
        return np.nan

    i = int(
        ids[0]
    )

    return encoder_at_time(
        encoder,
        times[i],
    )


def time_for_encoder(
    encoder,
    target,
    start,
    end,
):

    mask = (
        (encoder[:, 0] >= start)
        &
        (encoder[:, 0] <= end)
        &
        (encoder[:, 1] >= target)
    )

    ids = np.flatnonzero(
        mask
    )

    if not len(ids):
        return np.nan

    i = int(
        ids[0]
    )

    if i == 0:
        return float(
            encoder[i, 0]
        )

    t0, e0 = encoder[
        i - 1
    ]

    t1, e1 = encoder[i]

    if (
        e1 != e0
        and
        e0 <= target <= e1
    ):

        a = (
            target
            -
            e0
        ) / (
            e1
            -
            e0
        )

        return float(
            t0
            +
            a
            *
            (
                t1
                -
                t0
            )
        )

    return float(
        t1
    )


def free_space_metrics(
    xml,
    times,
    qpos,
    encoder,
    approach_start,
    regulation_start,
):

    model = mujoco.MjModel.from_xml_path(
        str(xml)
    )

    adr = {
        "mcp": joint_qadr(
            model,
            MCP,
        ),

        "pip": joint_qadr(
            model,
            PIP,
        ),

        "dip": joint_qadr(
            model,
            DIP,
        ),
    }

    q0 = q_at_time(
        times,
        qpos,
        approach_start,
    )

    reference = {
        key: np.rad2deg(
            q0[a]
        )
        for key, a
        in adr.items()
    }

    times_np = np.asarray(
        times,
        dtype=float,
    )

    qpos_np = np.asarray(
        qpos,
        dtype=float,
    )

    angles = {}

    for key, a in adr.items():

        angles[key] = (
            np.rad2deg(
                qpos_np[:, a]
            )
            -
            reference[key]
        )

    out = {}

    for threshold in (
        ACTIVATION_THRESHOLDS_DEG
    ):

        suffix = (
            f"{threshold:g}"
            .replace(".", "p")
        )

        for key in [
            "mcp",
            "pip",
            "dip",
        ]:

            out[
                f"{key}_activation_{suffix}deg_tick"
            ] = first_tick_for_angle(
                times_np,
                angles[key],
                encoder,
                approach_start,
                regulation_start,
                threshold,
            )

    for tick in CHECKPOINTS:

        t = time_for_encoder(
            encoder,
            tick,
            approach_start,
            regulation_start + 0.3,
        )

        if not np.isfinite(t):
            continue

        q = q_at_time(
            times,
            qpos,
            t,
        )

        for key, a in adr.items():

            angle = (
                np.rad2deg(
                    q[a]
                )
                -
                reference[key]
            )

            out[
                f"{key}_at_{int(tick)}_deg"
            ] = float(
                angle
            )

    # -------------------------------------------------------------
    # Qualitative staging check.
    #
    # We do not know exact hardware activation tick thresholds.
    # Therefore this is only an ordering test, NOT a fit target.
    # -------------------------------------------------------------

    m1 = out[
        "mcp_activation_1deg_tick"
    ]

    p1 = out[
        "pip_activation_1deg_tick"
    ]

    d1 = out[
        "dip_activation_1deg_tick"
    ]

    m3 = out[
        "mcp_activation_3deg_tick"
    ]

    p3 = out[
        "pip_activation_3deg_tick"
    ]

    d3 = out[
        "dip_activation_3deg_tick"
    ]

    out["order_1deg_pass"] = bool(
        np.isfinite(
            [m1, p1, d1]
        ).all()
        and
        m1 < p1 < d1
    )

    out["order_3deg_pass"] = bool(
        np.isfinite(
            [m3, p3, d3]
        ).all()
        and
        m3 < p3 < d3
    )

    if np.isfinite(
        [m1, p1, d1]
    ).all():

        out[
            "activation_separation_1deg"
        ] = float(
            (
                p1 - m1
            )
            +
            (
                d1 - p1
            )
        )

    else:

        out[
            "activation_separation_1deg"
        ] = -999.0

    # Score only ranks experiments.
    # Lower = better.
    score = 0.0

    if not out[
        "order_1deg_pass"
    ]:
        score += 1000.0

    if not out[
        "order_3deg_pass"
    ]:
        score += 500.0

    # Penalize premature distal motion.
    score += (
        10.0
        *
        abs(
            out.get(
                "dip_at_500_deg",
                100.0,
            )
        )
    )

    score += (
        3.0
        *
        abs(
            out.get(
                "pip_at_300_deg",
                100.0,
            )
        )
    )

    # Reward clear staged activation.
    score -= (
        0.1
        *
        max(
            0.0,
            out[
                "activation_separation_1deg"
            ],
        )
    )

    out[
        "free_stage_score"
    ] = float(
        score
    )

    return out


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
            "v35g20c_breakaway_hierarchy"
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

    free_rows = []
    contact_rows = []

    free_trajectories = {}
    contact_trajectories = {}

    try:

        g7.b2.v35.METERS_PER_TICK = (
            SCALE
        )

        g7.b2.v35.CONTACT_MU = (
            CONTACT_MU
        )

        print()
        print("=" * 150)
        print(
            "G20C — PASSIVE BREAKAWAY HIERARCHY"
        )
        print("=" * 150)

        print()
        print(
            "Stage A: bottle disabled; test FREE-SPACE joint activation."
        )

        print(
            "Stage B: same mechanical candidate; restore bottle and test H1."
        )

        for candidate in CANDIDATES:

            name = candidate[
                "name"
            ]

            print()
            print("=" * 150)

            print(
                f"CANDIDATE: {name}"
            )

            print(
                "frictionloss MCP/PIP/DIP = "
                f"{candidate['mcp']:.5f} / "
                f"{candidate['pip']:.5f} / "
                f"{candidate['dip']:.5f} Nm"
            )

            print("=" * 150)

            # =====================================================
            # FREE SPACE
            # =====================================================

            free_xml = (
                out_dir
                /
                f"{name}_free.xml"
            )

            make_model(
                baseline_xml,
                free_xml,
                candidate,
                free_space=True,
            )

            print()
            print(
                "[A] FREE-SPACE RUN"
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

            fm = free_space_metrics(
                free_xml,
                free_times,
                free_qpos,
                encoder_data,
                approach_start,
                regulation_start,
            )

            free_row = {
                "candidate":
                    name,

                "mcp_frictionloss":
                    candidate[
                        "mcp"
                    ],

                "pip_frictionloss":
                    candidate[
                        "pip"
                    ],

                "dip_frictionloss":
                    candidate[
                        "dip"
                    ],

                **fm,
            }

            free_rows.append(
                free_row
            )

            free_trajectories[
                name
            ] = (
                free_xml,
                free_qpos,
                free_times,
            )

            print()
            print(
                "1-deg activation ticks:"
            )

            print(
                "  MCP/PIP/DIP = "
                f"{fm['mcp_activation_1deg_tick']:.1f} / "
                f"{fm['pip_activation_1deg_tick']:.1f} / "
                f"{fm['dip_activation_1deg_tick']:.1f}"
            )

            print(
                f"  correct order = "
                f"{fm['order_1deg_pass']}"
            )

            print()
            print(
                "3-deg activation ticks:"
            )

            print(
                "  MCP/PIP/DIP = "
                f"{fm['mcp_activation_3deg_tick']:.1f} / "
                f"{fm['pip_activation_3deg_tick']:.1f} / "
                f"{fm['dip_activation_3deg_tick']:.1f}"
            )

            print(
                f"  correct order = "
                f"{fm['order_3deg_pass']}"
            )

            print()
            print(
                "free-space joint motion:"
            )

            for tick in [
                300,
                500,
                700,
                850,
            ]:

                print(
                    f"  {tick:4d} ticks: "
                    f"MCP={fm.get(f'mcp_at_{tick}_deg', np.nan):+7.3f} "
                    f"PIP={fm.get(f'pip_at_{tick}_deg', np.nan):+7.3f} "
                    f"DIP={fm.get(f'dip_at_{tick}_deg', np.nan):+7.3f}"
                )

            print(
                f"free-stage score = "
                f"{fm['free_stage_score']:.2f}"
            )

            # =====================================================
            # CONTACT
            # =====================================================

            contact_xml = (
                out_dir
                /
                f"{name}_contact.xml"
            )

            make_model(
                baseline_xml,
                contact_xml,
                candidate,
                free_space=False,
            )

            print()
            print(
                "[B] H1 CONTACT RUN"
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

            row = dict(
                result
            )

            row.update(
                mechanics
            )

            row[
                "candidate"
            ] = name

            row[
                "mcp_frictionloss"
            ] = candidate[
                "mcp"
            ]

            row[
                "pip_frictionloss"
            ] = candidate[
                "pip"
            ]

            row[
                "dip_frictionloss"
            ] = candidate[
                "dip"
            ]

            row[
                "free_order_1deg"
            ] = fm[
                "order_1deg_pass"
            ]

            row[
                "free_order_3deg"
            ] = fm[
                "order_3deg_pass"
            ]

            row[
                "free_stage_score"
            ] = fm[
                "free_stage_score"
            ]

            contact_rows.append(
                row
            )

            contact_trajectories[
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
                "H1 target pose = "
                f"{row['target_mcp_deg']:.3f} / "
                f"{row['target_pip_deg']:.3f} / "
                f"{row['target_dip_deg']:.3f}"
            )

            print(
                f"RMSE = "
                f"{row['rmse_deg']:.3f} deg"
            )

            print(
                "contact episodes M/T = "
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
                "middle->tip:"
            )

            print(
                f"  delay = "
                f"{row['middle_to_tip_delay_s']:.4f} s"
            )

            print(
                f"  dPIP  = "
                f"{row['delta_pip_middle_to_tip_deg']:+.3f} deg"
            )

            print(
                f"  dDIP  = "
                f"{row['delta_dip_middle_to_tip_deg']:+.3f} deg"
            )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

        g7.b2.v35.CONTACT_MU = (
            old_mu
        )

    # =============================================================
    # Save CSVs
    # =============================================================

    with (
        out_dir
        /
        "g20c_free_space_summary.csv"
    ).open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                free_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            free_rows
        )

    with (
        out_dir
        /
        "g20c_contact_summary.csv"
    ).open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                contact_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            contact_rows
        )

    # =============================================================
    # Ranking: FREE-SPACE behavior FIRST.
    # H1 RMSE cannot rescue mechanically wrong free-space behavior.
    # =============================================================

    ranked = sorted(
        contact_rows,
        key=lambda r: (
            not r[
                "free_order_1deg"
            ],

            not r[
                "free_order_3deg"
            ],

            r[
                "free_stage_score"
            ],

            (
                r[
                    "middle_episodes"
                ]
                +
                r[
                    "tip_episodes"
                ]
            ),

            r[
                "rmse_deg"
            ],
        ),
    )

    print()
    print("=" * 180)
    print(
        "G20C FINAL SUMMARY — FREE-SPACE MECHANICS FIRST"
    )
    print("=" * 180)

    print(
        "candidate           "
        "fMCP    fPIP    fDIP    "
        "E1_M   E1_P   E1_D   "
        "ord1 ord3  "
        "epM epT  RMSE  "
        "PIP@M DIP@M dPIP dDIP"
    )

    print("-" * 180)

    free_lookup = {
        r["candidate"]: r
        for r in free_rows
    }

    for r in ranked:

        f = free_lookup[
            r[
                "candidate"
            ]
        ]

        print(
            f"{r['candidate']:<19s} "
            f"{r['mcp_frictionloss']:6.4f} "
            f"{r['pip_frictionloss']:6.4f} "
            f"{r['dip_frictionloss']:6.4f} "
            f"{f['mcp_activation_1deg_tick']:6.1f} "
            f"{f['pip_activation_1deg_tick']:6.1f} "
            f"{f['dip_activation_1deg_tick']:6.1f} "
            f"{str(f['order_1deg_pass']):>5s} "
            f"{str(f['order_3deg_pass']):>5s} "
            f"{r['middle_episodes']:3d} "
            f"{r['tip_episodes']:3d} "
            f"{r['rmse_deg']:6.3f} "
            f"{r['pip_first_middle_deg']:6.2f} "
            f"{r['dip_first_middle_deg']:6.2f} "
            f"{r['delta_pip_middle_to_tip_deg']:+5.2f} "
            f"{r['delta_dip_middle_to_tip_deg']:+5.2f}"
        )

    # =============================================================
    # GUI
    #
    # 1. baseline contact
    # 2. best candidate FREE SPACE
    # 3. same best candidate H1 CONTACT
    # =============================================================

    if not args.no_gui:

        baseline = next(
            r
            for r in contact_rows
            if r[
                "candidate"
            ]
            ==
            "baseline"
        )

        best = next(
            r
            for r in ranked
            if r[
                "candidate"
            ]
            !=
            "baseline"
        )

        best_name = best[
            "candidate"
        ]

        print()
        print("=" * 140)
        print(
            "MANDATORY G20C GUI VALIDATION"
        )
        print("=" * 140)

        print()
        print(
            "GUI 1/3 — baseline H1 contact"
        )

        (
            xml,
            qpos,
            times,
        ) = contact_trajectories[
            "baseline"
        ]

        g7.exact_gui_replay(
            xml,
            qpos,
            times,
            "G20C baseline contact",
        )

        print()
        print(
            f"GUI 2/3 — {best_name} FREE SPACE"
        )

        print(
            "Watch specifically:"
        )

        print(
            "  MCP should begin first."
        )

        print(
            "  PIP should remain quiet longer."
        )

        print(
            "  DIP should remain quiet longest."
        )

        (
            xml,
            qpos,
            times,
        ) = free_trajectories[
            best_name
        ]

        g7.exact_gui_replay(
            xml,
            qpos,
            times,
            (
                f"G20C {best_name} FREE SPACE"
            ),
        )

        print()
        print(
            f"GUI 3/3 — {best_name} H1 CONTACT"
        )

        print(
            "Watch:"
        )

        print(
            "  MCP closes first"
        )

        print(
            "  -> middle contact"
        )

        print(
            "  -> PIP wraps"
        )

        print(
            "  -> distal contact"
        )

        print(
            "  -> DIP wraps"
        )

        print(
            "  -> no contact loss/bounce"
        )

        (
            xml,
            qpos,
            times,
        ) = contact_trajectories[
            best_name
        ]

        g7.exact_gui_replay(
            xml,
            qpos,
            times,
            (
                f"G20C {best_name} H1 CONTACT"
            ),
        )

    # =============================================================
    # Manifest
    # =============================================================

    with (
        out_dir
        /
        "manifest.txt"
    ).open("w") as f:

        f.write(
            "V35G20C — passive breakaway hierarchy test\n\n"
        )

        f.write(
            "G20A proved existing continuous spatial tendon "
            "can redistribute into each remaining free DOF.\n"
        )

        f.write(
            "G20C tests whether the missing behavior is "
            "joint-specific breakaway resistance.\n\n"
        )

        f.write(
            "Only MCP/PIP/DIP frictionloss values vary.\n"
        )

        f.write(
            "Joint spring stiffnesses unchanged.\n"
        )

        f.write(
            "Spatial tendon routing unchanged.\n"
        )

        f.write(
            "Encoder->tendon scale unchanged.\n"
        )

        f.write(
            "Contact model unchanged.\n"
        )

        f.write(
            "No contact-dependent switching or locking.\n"
        )

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
