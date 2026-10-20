#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import fit_index_contact_v1 as bagbase
import scan_index_contact_friction_v35g7 as g7
import scan_index_dip_stiffness_v35g15b as g15b
import test_index_breakaway_hierarchy_v35g20c as g20c
import test_index_mcp_backdrive_v35g20d as g20d


# ================================================================
# G23B — HIGH TENDON LOAD + PASSIVE HIERARCHY SUFFICIENCY TEST
#
# G20C:
#   passive resistance hierarchy corrected free-space ordering:
#       MCP -> PIP -> DIP
#   but middle contact still detached.
#
# G23A:
#   K = 6900 N/m reduced middle detached gap:
#       1.577 s -> 0.094 s
#   but produced severe distal contact chatter (6 episodes).
#
# G23B asks ONE question:
#
#   Are these two ingredients jointly sufficient?
#
#       1. enough tendon load during contact
#       2. proximal-to-distal passive resistance hierarchy
#
# IMPORTANT:
#
# The frictionloss values used here are NOT claimed as the
# physical RH8D parameters.
#
# They are a diagnostic proxy established by G20C.
#
# If G23B works:
#   DO NOT tune these friction values.
#   Replace the proxy with physically defensible nonlinear
#   transverse-joint spring mechanics.
#
# No contact tuning.
# No geometry changes.
# No joint locks.
# No contact-state switching.
# ================================================================


SCALE = 5.70e-6
TENDON_K = 6900.0
CONTACT_MU = 0.05

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

TENDON = "idx_flex_spatial"
FIXTURE = "index_contact_fixture"


# G20C hierarchy_strong.
FRICTION = {
    MCP: 0.00035,
    PIP: 0.00300,
    DIP: 0.00600,
}


CHECKPOINTS = [
    300,
    500,
    700,
    850,
    1000,
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

    # ------------------------------------------------------------
    # Diagnostic passive hierarchy from G20C.
    # ------------------------------------------------------------

    for joint_name, friction in FRICTION.items():

        joint = find_joint(
            root,
            joint_name,
        )

        joint.set(
            "frictionloss",
            f"{friction:.12g}",
        )

    # ------------------------------------------------------------
    # Contact-load requirement identified by G23A.
    # ------------------------------------------------------------

    tendon = find_spatial(
        root,
        TENDON,
    )

    tendon.set(
        "stiffness",
        f"{TENDON_K:.12g}",
    )

    # ------------------------------------------------------------
    # Free-space diagnostic = same mechanics, bottle disabled.
    # ------------------------------------------------------------

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
            "v35g23b_load_plus_hierarchy"
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
    # Hardware encoder timeline for free-space staging metrics.
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
    # Replay setup.
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
        "g23b_free.xml"
    )

    contact_xml = (
        out
        /
        "g23b_contact.xml"
    )

    make_model(
        source,
        free_xml,
        free_space=True,
    )

    make_model(
        source,
        contact_xml,
        free_space=False,
    )

    # Compile before running.
    import mujoco

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

    try:

        g7.b2.v35.METERS_PER_TICK = (
            SCALE
        )

        g7.b2.v35.TENDON_K = (
            TENDON_K
        )

        # ========================================================
        # A — FREE SPACE
        # ========================================================

        print()
        print("=" * 150)
        print(
            "G23B-A — FREE SPACE"
        )
        print("=" * 150)

        print(
            f"K = {TENDON_K:.1f} N/m"
        )

        print(
            "frictionloss MCP/PIP/DIP = "
            f"{FRICTION[MCP]:.5f} / "
            f"{FRICTION[PIP]:.5f} / "
            f"{FRICTION[DIP]:.5f} Nm"
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
            "free-space joint motion:"
        )

        for tick in CHECKPOINTS:

            print(
                f"  {tick:4d} ticks: "
                f"MCP="
                f"{fm.get(f'mcp_at_{tick}_deg', np.nan):+7.3f} "
                f"PIP="
                f"{fm.get(f'pip_at_{tick}_deg', np.nan):+7.3f} "
                f"DIP="
                f"{fm.get(f'dip_at_{tick}_deg', np.nan):+7.3f}"
            )

        # ========================================================
        # B — H1 CONTACT
        # ========================================================

        print()
        print("=" * 150)
        print(
            "G23B-B — H1 CONTACT"
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
            "g23b_contact_timeseries.csv",
            trace,
        )

        np.savez_compressed(
            out
            /
            "g23b_contact_trajectory.npz",
            times=np.asarray(
                times
            ),
            qpos=np.asarray(
                qpos
            ),
        )

        print()
        print(
            "H1 target pose:"
        )

        print(
            f"  MCP/PIP/DIP = "
            f"{result['target_mcp_deg']:.3f} / "
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
            f"  dPIP  = "
            f"{mechanics['delta_pip_middle_to_tip_deg']:+.3f} deg"
        )

        print(
            f"  dDIP  = "
            f"{mechanics['delta_dip_middle_to_tip_deg']:+.3f} deg"
        )

        # ========================================================
        # DECISION
        # ========================================================

        print()
        print("=" * 150)
        print(
            "G23B DECISION"
        )
        print("=" * 150)

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

        tip_reasonable = (
            mechanics[
                "tip_episodes"
            ]
            <=
            2
        )

        print(
            f"free-space staging MCP->PIP->DIP = "
            f"{staging}"
        )

        print(
            f"single middle-contact episode     = "
            f"{middle_stable}"
        )

        print(
            f"tip episodes <= 2                 = "
            f"{tip_reasonable}"
        )

        print()

        if (
            staging
            and
            middle_stable
            and
            tip_reasonable
        ):

            print(
                "RESULT: JOINT LOAD HIERARCHY + HIGH TENDON LOAD "
                "ARE JOINTLY SUFFICIENT."
            )

            print()
            print(
                "Next step is NOT another scan."
            )

            print(
                "Replace diagnostic Coulomb hierarchy with "
                "physical nonlinear transverse-joint springs."
            )

        elif (
            staging
            and
            middle_stable
        ):

            print(
                "RESULT: MIDDLE-CONTACT PROBLEM SOLVED, "
                "BUT DISTAL/DIP CONTACT STILL NEEDS PHYSICAL MODELING."
            )

            print()
            print(
                "Proceed to nonlinear DIP spring/contact mechanics."
            )

        elif staging:

            print(
                "RESULT: FREE-SPACE STAGING IS CORRECT, "
                "BUT CONTACT LOSS REMAINS."
            )

            print()
            print(
                "Do not tune joint friction further."
            )

            print(
                "The missing mechanism is beyond simple "
                "load hierarchy + tendon magnitude."
            )

        else:

            print(
                "RESULT: HIGH LOAD DESTROYS THE REQUIRED "
                "FREE-SPACE STAGING."
            )

            print()
            print(
                "Final model needs nonlinear/passive mechanics "
                "rather than constant resistance."
            )

    finally:

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

    # ============================================================
    # GUI — free then contact.
    # ============================================================

    if not args.no_gui:

        print()
        print("=" * 150)
        print(
            "GUI 1/2 — G23B FREE SPACE"
        )
        print("=" * 150)

        print(
            "Look only for:"
        )

        print(
            "  MCP clearly begins first"
        )

        print(
            "  then PIP"
        )

        print(
            "  then DIP"
        )

        g7.exact_gui_replay(
            free_xml,
            free_qpos,
            free_times,
            "G23B high-load + hierarchy FREE SPACE",
        )

        print()
        print("=" * 150)
        print(
            "GUI 2/2 — G23B H1 CONTACT"
        )
        print("=" * 150)

        print(
            "Required visual sequence:"
        )

        print(
            "  middle touches"
        )

        print(
            "  -> remains touching"
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
            "  -> no repeated distal chatter"
        )

        g7.exact_gui_replay(
            contact_xml,
            qpos,
            times,
            "G23B high-load + hierarchy H1 CONTACT",
        )


if __name__ == "__main__":
    main()
