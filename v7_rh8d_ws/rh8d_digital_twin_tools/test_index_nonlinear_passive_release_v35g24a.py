#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import fit_index_contact_v1 as bagbase
import scan_index_contact_friction_v35g7 as g7
import scan_index_dip_stiffness_v35g15b as g15b
import test_index_breakaway_hierarchy_v35g20c as g20c
import test_index_mcp_backdrive_v35g20d as g20d


# ================================================================
# G24A — NONLINEAR PASSIVE RELEASE LAW
#
# Evidence:
#
# G23A:
#   K=6900 N/m:
#       middle gap 1.577 -> 0.094 s
#   => higher tendon load strongly helps contact retention.
#
# G23B:
#   K=6900 + constant friction hierarchy:
#       free-space staging correct
#       middle gap worsened to 1.201 s
#
# Therefore:
#
#   passive resistance is useful near OPEN configuration
#   but must RELEASE as joints flex.
#
# This test uses a smooth q-dependent passive torque:
#
#       tau_extra(q) = -tau0 * gate(q)
#
# gate = 1 near open
# gate -> 0 smoothly with increasing flexion.
#
# NO:
#   contact detection
#   timing logic
#   joint locking
#   state machine
#   object-dependent switching
#
# This is a GREY-BOX nonlinear passive-joint test.
# ================================================================


SCALE = 5.70e-6
TENDON_K = 6900.0
CONTACT_MU = 0.05

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

TENDON = "idx_flex_spatial"
FIXTURE = "index_contact_fixture"


# ---------------------------------------------------------------
# Baseline physical joint friction restored.
# ---------------------------------------------------------------

BASE_FRICTION = {
    MCP: 0.00200,
    PIP: 0.00125,
    DIP: 0.00000,
}


# ---------------------------------------------------------------
# ONE mechanically motivated nonlinear passive law.
#
# Values come from the load levels that produced useful staging
# in G20C/G23B, but unlike constant Coulomb resistance they decay
# smoothly with joint flexion.
#
# PIP:
#   full additional resistance <= 2 deg
#   zero by 12 deg
#
# DIP:
#   full additional resistance <= 2 deg
#   zero by 10 deg
#
# At H1 first middle contact (~17 deg PIP / ~13 deg DIP in the
# high-load model), these extra torques should already be zero.
# ---------------------------------------------------------------

PASSIVE = {
    PIP: {
        "tau0_Nm": 0.00300,
        "full_until_deg": 2.0,
        "zero_after_deg": 12.0,
    },

    DIP: {
        "tau0_Nm": 0.00600,
        "full_until_deg": 2.0,
        "zero_after_deg": 10.0,
    },
}


def find_joint_xml(
    root,
    name,
):

    for joint in root.iter("joint"):

        if joint.get("name") == name:
            return joint

    raise RuntimeError(
        f"Joint not found: {name}"
    )


def find_spatial(
    root,
    name,
):

    for tendon in root.iter("spatial"):

        if tendon.get("name") == name:
            return tendon

    raise RuntimeError(
        f"Tendon not found: {name}"
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


def make_model(
    source,
    destination,
    free_space,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    # Restore original joint friction.
    for name, value in BASE_FRICTION.items():

        find_joint_xml(
            root,
            name,
        ).set(
            "frictionloss",
            f"{value:.12g}",
        )

    # High contact-supporting tendon load.
    find_spatial(
        root,
        TENDON,
    ).set(
        "stiffness",
        f"{TENDON_K:.12g}",
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


def smooth_gate(
    q_deg,
    full_until_deg,
    zero_after_deg,
):

    if q_deg <= full_until_deg:
        return 1.0

    if q_deg >= zero_after_deg:
        return 0.0

    s = (
        q_deg
        -
        full_until_deg
    ) / (
        zero_after_deg
        -
        full_until_deg
    )

    # cubic smoothstep:
    #
    # 0 -> 1 smoothly with zero end slopes
    smooth = (
        3.0 * s * s
        -
        2.0 * s * s * s
    )

    return (
        1.0
        -
        smooth
    )


def joint_refs(
    model,
):

    refs = {}

    for name in [
        MCP,
        PIP,
        DIP,
    ]:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            raise RuntimeError(
                f"Joint missing: {name}"
            )

        refs[
            name
        ] = {
            "jid":
                int(jid),

            "qadr":
                int(
                    model.jnt_qposadr[
                        jid
                    ]
                ),

            "dadr":
                int(
                    model.jnt_dofadr[
                        jid
                    ]
                ),
        }

    return refs


def extra_torque(
    joint_name,
    q_deg,
):

    if joint_name not in PASSIVE:
        return 0.0

    cfg = PASSIVE[
        joint_name
    ]

    gate = smooth_gate(
        q_deg,
        cfg[
            "full_until_deg"
        ],
        cfg[
            "zero_after_deg"
        ],
    )

    # Positive q = flexion.
    # Negative torque = passive resistance to flexion.
    return (
        -
        cfg[
            "tau0_Nm"
        ]
        *
        gate
    )


def print_passive_law():

    print()
    print("=" * 130)
    print(
        "G24A NONLINEAR PASSIVE TORQUE LAW"
    )
    print("=" * 130)

    print(
        "angle    PIP extra torque    DIP extra torque"
    )

    print("-" * 130)

    for q in [
        0,
        1,
        2,
        4,
        6,
        8,
        10,
        12,
        15,
        20,
    ]:

        print(
            f"{q:5.1f} deg   "
            f"{extra_torque(PIP, q):+10.6f} Nm      "
            f"{extra_torque(DIP, q):+10.6f} Nm"
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
            "v35g24a_nonlinear_passive_release"
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

    out = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        out
        /
        Path(__file__).name,
    )

    # ------------------------------------------------------------
    # Hardware encoder timeline.
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Replay infrastructure.
    # ------------------------------------------------------------

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    source = (
        g7.make_zero_margin_model(
            out
        )
    )

    free_xml = (
        out
        /
        "g24a_free.xml"
    )

    contact_xml = (
        out
        /
        "g24a_contact.xml"
    )

    make_model(
        source,
        free_xml,
        True,
    )

    make_model(
        source,
        contact_xml,
        False,
    )

    # Compile now.
    mujoco.MjModel.from_xml_path(
        str(
            free_xml
        )
    )

    mujoco.MjModel.from_xml_path(
        str(
            contact_xml
        )
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    old_k = float(
        g7.b2.v35.TENDON_K
    )

    original_step = (
        mujoco.mj_step
    )

    refs_cache = {}


    def passive_step(
        model,
        data,
        *step_args,
        **step_kwargs,
    ):

        key = id(
            model
        )

        if key not in refs_cache:

            refs_cache[
                key
            ] = joint_refs(
                model
            )

        refs = refs_cache[
            key
        ]

        # Remove only our previous contribution on these DOFs.
        #
        # In this replay there are no other externally applied
        # MCP/PIP/DIP generalized forces.
        for joint_name in [
            MCP,
            PIP,
            DIP,
        ]:

            dadr = refs[
                joint_name
            ][
                "dadr"
            ]

            data.qfrc_applied[
                dadr
            ] = 0.0

        for joint_name in [
            PIP,
            DIP,
        ]:

            qadr = refs[
                joint_name
            ][
                "qadr"
            ]

            dadr = refs[
                joint_name
            ][
                "dadr"
            ]

            q_deg = math.degrees(
                float(
                    data.qpos[
                        qadr
                    ]
                )
            )

            data.qfrc_applied[
                dadr
            ] += extra_torque(
                joint_name,
                q_deg,
            )

        return original_step(
            model,
            data,
            *step_args,
            **step_kwargs,
        )


    print_passive_law()

    try:

        g7.b2.v35.METERS_PER_TICK = (
            SCALE
        )

        g7.b2.v35.TENDON_K = (
            TENDON_K
        )

        # Activate smooth passive torque law.
        mujoco.mj_step = (
            passive_step
        )

        # ========================================================
        # FREE SPACE
        # ========================================================

        print()
        print("=" * 150)
        print(
            "G24A-A — FREE SPACE"
        )
        print("=" * 150)

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
            replay,
        )

        fm = (
            g20c.free_space_metrics(
                free_xml,
                free_times,
                free_qpos,
                encoder_data,
                approach_start,
                regulation_start,
            )
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
            "free-space checkpoints:"
        )

        for tick in [
            300,
            500,
            700,
            850,
            1000,
        ]:

            print(
                f"  {tick:4d}: "
                f"MCP="
                f"{fm.get(f'mcp_at_{tick}_deg', np.nan):+7.3f} "
                f"PIP="
                f"{fm.get(f'pip_at_{tick}_deg', np.nan):+7.3f} "
                f"DIP="
                f"{fm.get(f'dip_at_{tick}_deg', np.nan):+7.3f}"
            )

        # ========================================================
        # CONTACT
        # ========================================================

        print()
        print("=" * 150)
        print(
            "G24A-B — H1 CONTACT"
        )
        print("=" * 150)

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
            replay,
        )

        mechanics = (
            g15b.mechanics(
                trace
            )
        )

        episode = (
            g20d.first_middle_episode_metrics(
                contact_xml,
                trace,
                qpos,
                times,
            )
        )

        g7.save_trace_csv(
            out
            /
            "g24a_contact_timeseries.csv",
            trace,
        )

        np.savez_compressed(
            out
            /
            "g24a_contact_trajectory.npz",
            times=np.asarray(
                times
            ),
            qpos=np.asarray(
                qpos
            ),
        )

        print()
        print(
            "H1 target pose MCP/PIP/DIP:"
        )

        print(
            f"  {result['target_mcp_deg']:.3f} / "
            f"{result['target_pip_deg']:.3f} / "
            f"{result['target_dip_deg']:.3f}"
        )

        print(
            f"  RMSE = "
            f"{result['rmse_deg']:.3f} deg"
        )

        print()
        print(
            "contact mechanics:"
        )

        print(
            f"  episodes M/T = "
            f"{mechanics['middle_episodes']} / "
            f"{mechanics['tip_episodes']}"
        )

        print(
            "  first-middle pose = "
            f"{mechanics['mcp_first_middle_deg']:.3f} / "
            f"{mechanics['pip_first_middle_deg']:.3f} / "
            f"{mechanics['dip_first_middle_deg']:.3f}"
        )

        print(
            f"  first-middle duration = "
            f"{episode['first_mid_duration_s']:.3f} s"
        )

        print(
            f"  detached middle gap   = "
            f"{episode['gap_to_second_middle_s']:.3f} s"
        )

        print()
        print(
            "middle -> tip:"
        )

        print(
            f"  delay = "
            f"{mechanics['middle_to_tip_delay_s']:.3f} s"
        )

        print(
            f"  dPIP = "
            f"{mechanics['delta_pip_middle_to_tip_deg']:+.3f}"
        )

        print(
            f"  dDIP = "
            f"{mechanics['delta_dip_middle_to_tip_deg']:+.3f}"
        )

        # ========================================================
        # DECISION
        # ========================================================

        staging = bool(
            fm[
                "order_1deg_pass"
            ]
            and
            fm[
                "order_3deg_pass"
            ]
        )

        middle_stable = (
            mechanics[
                "middle_episodes"
            ]
            ==
            1
        )

        tip_ok = (
            mechanics[
                "tip_episodes"
            ]
            <=
            2
        )

        gap = float(
            episode[
                "gap_to_second_middle_s"
            ]
        )

        print()
        print("=" * 150)
        print(
            "G24A DECISION"
        )
        print("=" * 150)

        print(
            f"free-space MCP->PIP->DIP staging = {staging}"
        )

        print(
            f"middle episodes                  = "
            f"{mechanics['middle_episodes']}"
        )

        print(
            f"tip episodes                     = "
            f"{mechanics['tip_episodes']}"
        )

        print(
            f"middle detached gap              = "
            f"{gap:.3f} s"
        )

        print()

        if (
            staging
            and
            middle_stable
            and
            tip_ok
        ):

            print(
                "RESULT: NONLINEAR PASSIVE-RELEASE "
                "HYPOTHESIS STRONGLY SUPPORTED."
            )

            print(
                "Proceed directly to identification/fitting."
            )

        elif (
            staging
            and
            np.isfinite(
                gap
            )
            and
            gap <= 0.15
        ):

            print(
                "RESULT: STRONGLY PROMISING."
            )

            print(
                "Correct staging is preserved and contact "
                "loss remains very short."
            )

            print(
                "Visual verdict determines whether the residual "
                "episode is meaningful or just contact chatter."
            )

        else:

            print(
                "RESULT: THIS SINGLE NONLINEAR LAW IS INSUFFICIENT."
            )

            print(
                "Do NOT scan tau0/cutoff values blindly."
            )

    finally:

        mujoco.mj_step = (
            original_step
        )

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

        g7.b2.v35.TENDON_K = (
            old_k
        )

    print()
    print(
        f"All outputs:\n  {out}"
    )

    if not args.no_gui:

        print()
        print("=" * 150)
        print(
            "GUI 1/2 — G24A FREE SPACE"
        )
        print("=" * 150)

        print(
            "Required:"
        )

        print(
            "  MCP first -> PIP -> DIP"
        )

        g7.exact_gui_replay(
            free_xml,
            free_qpos,
            free_times,
            "G24A nonlinear passive release FREE SPACE",
        )

        print()
        print("=" * 150)
        print(
            "GUI 2/2 — G24A H1 CONTACT"
        )
        print("=" * 150)

        print(
            "Required:"
        )

        print(
            "  middle touches and stays"
        )

        print(
            "  -> PIP wraps"
        )

        print(
            "  -> distal touches"
        )

        print(
            "  -> DIP wraps"
        )

        print(
            "  -> no visible bounce/gap"
        )

        g7.exact_gui_replay(
            contact_xml,
            qpos,
            times,
            "G24A nonlinear passive release H1 CONTACT",
        )


if __name__ == "__main__":
    main()
