#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import scan_index_contact_friction_v35g7 as g7
import scan_index_dip_stiffness_v35g15b as g15b


SCALE = 5.70e-6
TENDON_K = 2100.0

CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

TIMESTEPS = [
    0.001,
    0.0005,
    0.00025,
]


def make_variant(
    source_xml: Path,
    out_dir: Path,
    timestep: float,
):

    tree = ET.parse(source_xml)
    root = tree.getroot()

    option = root.find("option")

    if option is None:
        raise RuntimeError(
            "No <option> element found."
        )

    option.set(
        "timestep",
        f"{timestep:.10g}",
    )

    name = (
        f"model_dt_{timestep*1000:.3f}ms"
        .replace(".", "p")
        +
        ".xml"
    )

    path = out_dir / name

    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )

    return path


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
            "v35g16a_timestep_audit"
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
        out_dir / Path(__file__).name,
    )

    g7.b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    base_xml = g7.make_zero_margin_model(
        out_dir
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

        g7.b2.v35.METERS_PER_TICK = SCALE
        g7.b2.v35.TENDON_K = TENDON_K
        g7.b2.v35.CONTACT_MU = CONTACT_MU
        g7.b2.v35.CONTACT_K = CONTACT_K

        contact_replay.CONTACT_DAMPING = (
            CONTACT_DAMPING
        )

        print()
        print("=" * 130)
        print(
            "G16A — NUMERICAL TIMESTEP CONVERGENCE AUDIT"
        )
        print("=" * 130)

        print()
        print(
            "NO physical parameter changes."
        )

        for dt in TIMESTEPS:

            xml = make_variant(
                base_xml,
                out_dir,
                dt,
            )

            print()
            print("-" * 130)
            print(
                f"TIMESTEP = {dt*1000:.3f} ms"
            )
            print("-" * 130)

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

            row = dict(result)

            row.update(
                g15b.mechanics(
                    trace
                )
            )

            row["timestep_s"] = dt
            row["timestep_ms"] = dt * 1000.0

            rows.append(row)

            trajectories[dt] = (
                xml,
                qpos,
                times,
            )

            tag = (
                f"{dt*1000:.3f}"
                .replace(".", "p")
            )

            g7.save_trace_csv(
                out_dir
                /
                f"dt_{tag}ms_timeseries.csv",
                trace,
            )

            print(
                "target MCP/PIP/DIP = "
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

            print(
                "first-middle pose = "
                f"{row['mcp_first_middle_deg']:.3f} / "
                f"{row['pip_first_middle_deg']:.3f} / "
                f"{row['dip_first_middle_deg']:.3f}"
            )

            print(
                f"middle->tip delay = "
                f"{row['middle_to_tip_delay_s']:.4f} s"
            )

            print(
                "middle->tip dPIP/dDIP = "
                f"{row['delta_pip_middle_to_tip_deg']:+.3f} / "
                f"{row['delta_dip_middle_to_tip_deg']:+.3f} deg"
            )

            print(
                "target contacts M/T = "
                f"{row['target_mid_contact']} / "
                f"{row['target_tip_contact']}"
            )

    finally:

        g7.b2.v35.METERS_PER_TICK = old_scale
        g7.b2.v35.TENDON_K = old_tendon_k
        g7.b2.v35.CONTACT_MU = old_mu
        g7.b2.v35.CONTACT_K = old_contact_k

        contact_replay.CONTACT_DAMPING = (
            old_contact_damping
        )

    with (
        out_dir
        /
        "g16a_summary.csv"
    ).open(
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
        writer.writerows(rows)

    print()
    print("=" * 150)
    print("G16A SUMMARY")
    print("=" * 150)

    print(
        "dt_ms  RMSE  epM epT "
        "DIP@M  delay   dPIP   dDIP"
    )

    for r in rows:

        print(
            f"{r['timestep_ms']:5.3f} "
            f"{r['rmse_deg']:5.3f} "
            f"{r['middle_episodes']:3d} "
            f"{r['tip_episodes']:3d} "
            f"{r['dip_first_middle_deg']:6.2f} "
            f"{r['middle_to_tip_delay_s']:7.4f} "
            f"{r['delta_pip_middle_to_tip_deg']:+6.2f} "
            f"{r['delta_dip_middle_to_tip_deg']:+6.2f}"
        )

    if not args.no_gui:

        for i, dt in enumerate(
            TIMESTEPS,
            start=1,
        ):

            xml, qpos, times = (
                trajectories[dt]
            )

            print()
            print("#" * 120)
            print(
                f"GUI {i}/3 — timestep "
                f"{dt*1000:.3f} ms"
            )
            print("#" * 120)

            g7.exact_gui_replay(
                xml,
                qpos,
                times,
                (
                    f"G16A timestep "
                    f"{dt*1000:.3f} ms"
                ),
            )

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
