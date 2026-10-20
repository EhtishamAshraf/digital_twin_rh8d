#!/usr/bin/env python3

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import scan_index_contact_friction_v35g7 as g7
import scan_index_dip_stiffness_v35g15b as g15b
import test_index_mcp_backdrive_v35g20d as g20d


SCALE = 5.70e-6
TIP_GEOM = "collision_idx_tip"


def disable_tip(source, destination):

    tree = ET.parse(source)
    root = tree.getroot()

    found = False

    for geom in root.iter("geom"):

        if geom.get("name") == TIP_GEOM:

            geom.set("contype", "0")
            geom.set("conaffinity", "0")

            found = True
            break

    if not found:
        raise RuntimeError(
            f"Could not find geom {TIP_GEOM}"
        )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )


def run_case(
    name,
    xml,
    bag,
    replay,
):

    print()
    print("=" * 120)
    print(name)
    print("=" * 120)

    result, trace, qpos, times = (
        g7.run_scale(
            SCALE,
            0.05,
            xml,
            bag,
            replay,
        )
    )

    mech = g15b.mechanics(trace)

    episode = (
        g20d.first_middle_episode_metrics(
            xml,
            trace,
            qpos,
            times,
        )
    )

    print(
        "target pose MCP/PIP/DIP = "
        f"{result['target_mcp_deg']:.3f} / "
        f"{result['target_pip_deg']:.3f} / "
        f"{result['target_dip_deg']:.3f}"
    )

    print(
        f"RMSE = {result['rmse_deg']:.3f} deg"
    )

    print(
        "contact episodes M/T = "
        f"{mech['middle_episodes']} / "
        f"{mech['tip_episodes']}"
    )

    print(
        "first middle pose = "
        f"{mech['mcp_first_middle_deg']:.3f} / "
        f"{mech['pip_first_middle_deg']:.3f} / "
        f"{mech['dip_first_middle_deg']:.3f}"
    )

    print(
        "first middle duration = "
        f"{episode['first_mid_duration_s']:.3f} s"
    )

    print(
        "gap to second middle = "
        f"{episode['gap_to_second_middle_s']:.3f} s"
    )

    return {
        "result": result,
        "mechanics": mech,
        "episode": episode,
        "trace": trace,
        "qpos": qpos,
        "times": times,
        "xml": xml,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "bag",
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            / "rh8d_results"
            / "v35g22b_no_tip_contact"
        ),
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
    )

    args = parser.parse_args()

    bag = args.bag.expanduser().resolve()

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
        out / Path(__file__).name,
    )

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    original_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    g7.b2.v35.METERS_PER_TICK = SCALE

    try:

        baseline_xml = (
            g7.make_zero_margin_model(out)
        )

        no_tip_xml = (
            out
            / "g22b_no_tip_contact.xml"
        )

        disable_tip(
            baseline_xml,
            no_tip_xml,
        )

        baseline = run_case(
            "G22B CONTROL — ORIGINAL CONTACT",
            baseline_xml,
            bag,
            replay,
        )

        no_tip = run_case(
            "G22B TEST — DISTAL COLLISION DISABLED",
            no_tip_xml,
            bag,
            replay,
        )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            original_scale
        )

    print()
    print("=" * 120)
    print("G22B DECISION")
    print("=" * 120)

    b_ep = baseline[
        "mechanics"
    ][
        "middle_episodes"
    ]

    n_ep = no_tip[
        "mechanics"
    ][
        "middle_episodes"
    ]

    b_gap = baseline[
        "episode"
    ][
        "gap_to_second_middle_s"
    ]

    n_gap = no_tip[
        "episode"
    ][
        "gap_to_second_middle_s"
    ]

    print(
        f"baseline middle episodes = {b_ep}"
    )

    print(
        f"no-tip   middle episodes = {n_ep}"
    )

    print(
        f"baseline middle gap = {b_gap:.3f} s"
    )

    print(
        f"no-tip   middle gap = {n_gap:.3f} s"
    )

    print()

    if n_ep == 1:

        print(
            "RESULT: removing distal contact prevents "
            "middle-contact loss."
        )

        print(
            "=> bounce is caused by the distal-contact "
            "constraint / simultaneous-contact geometry."
        )

    else:

        print(
            "RESULT: middle contact still breaks without "
            "distal contact."
        )

        print(
            "=> bounce originates at middle-contact mechanics, "
            "not distal contact."
        )

    if not args.no_gui:

        print()
        print(
            "GUI — NO-TIP DIAGNOSTIC RUN"
        )

        print(
            "Watch only whether the middle phalanx "
            "touches and then separates."
        )

        g7.exact_gui_replay(
            no_tip[
                "xml"
            ],
            no_tip[
                "qpos"
            ],
            no_tip[
                "times"
            ],
            "G22B distal collision disabled",
        )

    print()
    print(
        f"All outputs:\n  {out}"
    )


if __name__ == "__main__":
    main()
