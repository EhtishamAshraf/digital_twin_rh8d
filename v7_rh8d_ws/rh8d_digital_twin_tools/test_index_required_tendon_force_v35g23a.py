#!/usr/bin/env python3

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import scan_index_contact_friction_v35g7 as g7
import scan_index_dip_stiffness_v35g15b as g15b
import test_index_mcp_backdrive_v35g20d as g20d


# ================================================================
# G23A — REQUIRED TENDON FORCE COUNTERFACTUAL
#
# G22C measured at peak first-middle contact:
#
# PIP:
#   tendon torque      ~ +0.002177 Nm
#   passive residual   ~ -0.002622 Nm
#   constraint force   ~ -0.005000 Nm
#   bias               ~ -0.000511 Nm
#
# MuJoCo forward dynamics:
#
#   M qacc =
#       qfrc_passive
#     + qfrc_constraint
#     - qfrc_bias
#     + ...
#
# Approximate PIP tendon torque required for zero acceleration:
#
#   tau_required
#     = -(passive_residual + constraint - bias)
#     ~ 0.00711 Nm
#
# Ratio:
#
#   0.00711 / 0.002177 ~ 3.27
#
# Current tendon stiffness:
#
#   2100 N/m
#
# Therefore one mechanically-derived diagnostic candidate:
#
#   2100 * 3.27 ~ 6870 N/m
#
# Rounded:
#
#   6900 N/m
#
# THIS IS NOT YET THE FINAL IDENTIFIED STIFFNESS.
#
# It is a binary causal test:
#
# Does sufficient tendon tension prevent the middle-contact loss?
#
# No contact tuning.
# No joint locking.
# No state-dependent switching.
# No change to object geometry.
# No change to joint friction/springs.
# ================================================================


SCALE = 5.70e-6

BASELINE_K = 2100.0
TEST_K = 6900.0

CONTACT_MU = 0.05

TENDON_NAME = "idx_flex_spatial"


def set_tendon_stiffness(
    source,
    destination,
    stiffness,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    found = False

    for tendon in root.iter(
        "spatial"
    ):

        if tendon.get(
            "name"
        ) == TENDON_NAME:

            tendon.set(
                "stiffness",
                f"{stiffness:.12g}",
            )

            found = True
            break

    if not found:

        raise RuntimeError(
            f"Could not find tendon {TENDON_NAME}"
        )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )


def run_case(
    name,
    stiffness,
    xml,
    bag,
    replay,
):

    print()
    print("=" * 150)

    print(
        f"{name}"
    )

    print(
        f"Tendon stiffness = "
        f"{stiffness:.1f} N/m"
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
        xml,
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
            xml,
            trace,
            qpos,
            times,
        )
    )

    row = {}

    row.update(
        result
    )

    row.update(
        mechanics
    )

    row.update(
        episode
    )

    row[
        "name"
    ] = name

    row[
        "tendon_stiffness_N_m"
    ] = stiffness

    print()

    print(
        "TARGET POSE MCP/PIP/DIP:"
    )

    print(
        f"  "
        f"{result['target_mcp_deg']:.3f} / "
        f"{result['target_pip_deg']:.3f} / "
        f"{result['target_dip_deg']:.3f} deg"
    )

    print(
        f"  RMSE = "
        f"{result['rmse_deg']:.3f} deg"
    )

    print()

    print(
        "CONTACT:"
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
        "MIDDLE -> TIP:"
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

    return (
        row,
        trace,
        qpos,
        times,
    )


def save_summary(
    path,
    rows,
):

    keys = []

    seen = set()

    for row in rows:

        for key in row.keys():

            if key not in seen:

                seen.add(
                    key
                )

                keys.append(
                    key
                )

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow({
                key:
                    row.get(
                        key,
                        np.nan,
                    )
                for key in keys
            })


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
            "v35g23a_required_tendon_force"
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

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    source_xml = (
        g7.make_zero_margin_model(
            out
        )
    )

    baseline_xml = (
        out
        /
        "g23a_baseline_k2100.xml"
    )

    test_xml = (
        out
        /
        "g23a_required_k6900.xml"
    )

    set_tendon_stiffness(
        source_xml,
        baseline_xml,
        BASELINE_K,
    )

    set_tendon_stiffness(
        source_xml,
        test_xml,
        TEST_K,
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    old_k = float(
        g7.b2.v35.TENDON_K
    )

    rows = []

    try:

        g7.b2.v35.METERS_PER_TICK = (
            SCALE
        )

        # ========================================================
        # CONTROL
        # ========================================================

        g7.b2.v35.TENDON_K = (
            BASELINE_K
        )

        (
            baseline,
            baseline_trace,
            baseline_qpos,
            baseline_times,
        ) = run_case(
            "G23A CONTROL — K=2100",
            BASELINE_K,
            baseline_xml,
            bag,
            replay,
        )

        rows.append(
            baseline
        )

        g7.save_trace_csv(
            out
            /
            "baseline_timeseries.csv",
            baseline_trace,
        )

        # ========================================================
        # ONE DERIVED COUNTERFACTUAL
        # ========================================================

        g7.b2.v35.TENDON_K = (
            TEST_K
        )

        (
            test,
            test_trace,
            test_qpos,
            test_times,
        ) = run_case(
            "G23A TEST — FORCE-BALANCE K=6900",
            TEST_K,
            test_xml,
            bag,
            replay,
        )

        rows.append(
            test
        )

        g7.save_trace_csv(
            out
            /
            "k6900_timeseries.csv",
            test_trace,
        )

        np.savez_compressed(
            out
            /
            "k6900_trajectory.npz",
            times=np.asarray(
                test_times
            ),
            qpos=np.asarray(
                test_qpos
            ),
        )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

        g7.b2.v35.TENDON_K = (
            old_k
        )

    summary = (
        out
        /
        "g23a_summary.csv"
    )

    save_summary(
        summary,
        rows,
    )

    print()
    print("=" * 150)
    print(
        "G23A DECISION"
    )
    print("=" * 150)

    b_ep = int(
        baseline[
            "middle_episodes"
        ]
    )

    t_ep = int(
        test[
            "middle_episodes"
        ]
    )

    b_gap = float(
        baseline[
            "gap_to_second_middle_s"
        ]
    )

    t_gap = float(
        test[
            "gap_to_second_middle_s"
        ]
    )

    print(
        f"baseline K=2100:"
    )

    print(
        f"  middle episodes = {b_ep}"
    )

    print(
        f"  detached gap    = {b_gap:.3f} s"
    )

    print()

    print(
        f"derived K=6900:"
    )

    print(
        f"  middle episodes = {t_ep}"
    )

    print(
        f"  detached gap    = {t_gap:.3f} s"
    )

    print()

    if (
        t_ep == 1
    ):

        print(
            "RESULT: TENSION HYPOTHESIS SUPPORTED."
        )

        print(
            "Sufficient flexor tension eliminates "
            "middle-contact loss."
        )

        print(
            "Next model must reproduce this load level "
            "while preserving real free-space staging."
        )

    elif (
        np.isfinite(
            t_gap
        )
        and
        np.isfinite(
            b_gap
        )
        and
        t_gap
        <
        0.5
        *
        b_gap
    ):

        print(
            "RESULT: TENSION HYPOTHESIS PARTIALLY SUPPORTED."
        )

        print(
            "The gap fell by more than 50%, but "
            "contact is still not fully stable."
        )

    else:

        print(
            "RESULT: TENSION HYPOTHESIS REJECTED."
        )

        print(
            "Even the force-balance tendon stiffness "
            "does not cure middle-contact loss."
        )

        print(
            "Do NOT scan tendon stiffness further."
        )

    print()
    print(
        f"Summary:\n  {summary}"
    )

    print()
    print(
        f"All outputs:\n  {out}"
    )

    if not args.no_gui:

        print()
        print("=" * 150)
        print(
            "G23A GUI — K=6900 ONLY"
        )
        print("=" * 150)

        print(
            "Watch:"
        )

        print(
            "  middle touches"
        )

        print(
            "  -> does it remain touching?"
        )

        print(
            "  -> does PIP continue wrapping?"
        )

        print(
            "  -> does DIP eventually participate?"
        )

        g7.exact_gui_replay(
            test_xml,
            test_qpos,
            test_times,
            "G23A force-balance tendon K=6900",
        )


if __name__ == "__main__":
    main()
