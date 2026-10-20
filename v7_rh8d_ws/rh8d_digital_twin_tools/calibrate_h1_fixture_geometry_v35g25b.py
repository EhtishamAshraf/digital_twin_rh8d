#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import scan_index_contact_friction_v35g7 as g7
import audit_index_tendon_transmission_v35g12 as g12


# =====================================================================
# G25B — GEOMETRIC FIXTURE CALIBRATION FROM REAL FIRST-CONTACT POSE
#
# Hardware G25A:
#
#   first middle contact:
#       encoder = 673.573 ticks
#       MCP     = 18.543 deg
#       PIP     =  6.161 deg
#       DIP     =  5.276 deg
#
# Current simulation:
#
#       encoder = 870.963 ticks
#       MCP     = 16.937 deg
#       PIP     = 17.145 deg
#       DIP     = 13.608 deg
#
# We DO NOT tune the bottle until simulated contact occurs at 673.
#
# Instead:
#
#   1. Obtain the complete simulated robot configuration near
#      encoder=673 from the normal H1 replay.
#
#   2. Replace ONLY MCP/PIP/DIP with the measured hardware
#      first-contact angles.
#
#   3. Geometrically solve the bottle radial translation required
#      for the real middle phalanx pose to be exactly touching.
#
#   4. Freeze that calibrated bottle geometry.
#
#   5. Replay the original H1 model once.
#
# This distinguishes:
#
#   fixture geometry error
#       from
#   wrong pre-contact finger mechanics.
#
# NO passive-law tuning.
# NO contact-parameter tuning.
# NO tendon-stiffness tuning.
# =====================================================================


HW_ENCODER = 673.573

HW_MCP = 18.543
HW_PIP = 6.161
HW_DIP = 5.276

TRUSTED_FINAL = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)

TARGET_BAG_TIME = 38.970486

SCALE = 5.70e-6
TENDON_K = 2100.0
CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0


def nearest_index(
    values,
    target,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    return int(
        np.argmin(
            np.abs(
                values - target
            )
        )
    )


def find_geom_xml(
    root,
    name,
):

    for geom in root.iter("geom"):

        if geom.get("name") == name:
            return geom

    raise RuntimeError(
        f"Geom not found in XML: {name}"
    )


def parse_pos(
    geom,
):

    raw = geom.get("pos")

    if raw is None:
        raise RuntimeError(
            "Fixture geom has no explicit pos attribute."
        )

    values = np.asarray(
        [
            float(x)
            for x in raw.split()
        ],
        dtype=float,
    )

    if len(values) != 3:
        raise RuntimeError(
            f"Unexpected fixture pos: {raw}"
        )

    return values


def write_shifted_model(
    source_xml,
    destination_xml,
    direction_xy,
    shift_m,
):

    tree = ET.parse(
        source_xml
    )

    root = tree.getroot()

    fixture = find_geom_xml(
        root,
        g12.FIXTURE_NAME,
    )

    pos = parse_pos(
        fixture
    )

    pos[0] += (
        shift_m
        *
        direction_xy[0]
    )

    pos[1] += (
        shift_m
        *
        direction_xy[1]
    )

    fixture.set(
        "pos",
        (
            f"{pos[0]:.12f} "
            f"{pos[1]:.12f} "
            f"{pos[2]:.12f}"
        ),
    )

    tree.write(
        destination_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    return pos


def set_hardware_finger_pose(
    model,
    data,
    q_reference,
):

    if len(q_reference) != model.nq:
        raise RuntimeError(
            "Reference qpos dimension mismatch."
        )

    data.qpos[:] = (
        q_reference
    )

    refs = g12.get_joint_refs(
        model
    )

    data.qpos[
        refs["mcp"]["qadr"]
    ] = math.radians(
        HW_MCP
    )

    data.qpos[
        refs["pip"]["qadr"]
    ] = math.radians(
        HW_PIP
    )

    data.qpos[
        refs["dip"]["qadr"]
    ] = math.radians(
        HW_DIP
    )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    return refs


def geom_distance(
    model,
    data,
    middle_gid,
    fixture_gid,
):

    fromto = np.zeros(
        6,
        dtype=float,
    )

    distance = float(
        mujoco.mj_geomDistance(
            model,
            data,
            middle_gid,
            fixture_gid,
            0.20,
            fromto,
        )
    )

    return (
        distance,
        fromto,
    )


def static_distance_for_shift(
    source_xml,
    temporary_xml,
    direction_xy,
    shift_m,
    q_reference,
):

    write_shifted_model(
        source_xml,
        temporary_xml,
        direction_xy,
        shift_m,
    )

    model = mujoco.MjModel.from_xml_path(
        str(
            temporary_xml
        )
    )

    data = mujoco.MjData(
        model
    )

    set_hardware_finger_pose(
        model,
        data,
        q_reference,
    )

    middle_gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        g12.CONTACT_GEOMS["mid"],
    )

    fixture_gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        g12.FIXTURE_NAME,
    )

    if (
        middle_gid < 0
        or
        fixture_gid < 0
    ):
        raise RuntimeError(
            "Could not resolve middle/fixture geometry."
        )

    distance, fromto = geom_distance(
        model,
        data,
        middle_gid,
        fixture_gid,
    )

    return distance


def first_episode(
    rows,
    key,
):

    flags = np.asarray(
        [
            bool(
                r[
                    key
                ]
            )
            for r in rows
        ],
        dtype=bool,
    )

    starts = g12.contact_episode_starts(
        flags
    )

    ends = g12.contact_episode_ends(
        flags
    )

    return (
        starts,
        ends,
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
            "v35g25b_fixture_geometry"
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

    # -------------------------------------------------------------
    # Replay infrastructure.
    # -------------------------------------------------------------

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    source_xml = (
        g7.make_zero_margin_model(
            out
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
        replay.CONTACT_DAMPING
    )

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

        replay.CONTACT_DAMPING = (
            CONTACT_DAMPING
        )

        # =========================================================
        # STEP 1 — baseline replay only to obtain complete qpos
        # around the real hardware contact encoder.
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G25B STEP 1 — BASELINE CONFIGURATION AT HARDWARE CONTACT ENCODER"
        )
        print("=" * 150)

        (
            baseline_rows,
            baseline_qpos,
            baseline_times,
            baseline_summary,
        ) = g12.run_audit(
            source_xml,
            bag,
            replay,
        )

        baseline_encoder = np.asarray(
            [
                r[
                    "encoder_tick"
                ]
                for r in baseline_rows
            ],
            dtype=float,
        )

        hw_idx = nearest_index(
            baseline_encoder,
            HW_ENCODER,
        )

        baseline_at_hw = (
            baseline_rows[
                hw_idx
            ]
        )

        q_reference = np.asarray(
            baseline_qpos[
                hw_idx
            ],
            dtype=float,
        ).copy()

        print()
        print(
            f"hardware first-contact encoder = "
            f"{HW_ENCODER:.3f}"
        )

        print()
        print(
            "REAL hardware pose there:"
        )

        print(
            f"  MCP/PIP/DIP = "
            f"{HW_MCP:.3f} / "
            f"{HW_PIP:.3f} / "
            f"{HW_DIP:.3f} deg"
        )

        print()
        print(
            "CURRENT simulation pose at same encoder:"
        )

        print(
            f"  MCP/PIP/DIP = "
            f"{baseline_at_hw['mcp_deg']:.3f} / "
            f"{baseline_at_hw['pip_deg']:.3f} / "
            f"{baseline_at_hw['dip_deg']:.3f} deg"
        )

        print(
            f"  current middle gap = "
            f"{baseline_at_hw['mid_gap_mm']:+.3f} mm"
        )

        # =========================================================
        # STEP 2 — static hardware-pose geometry.
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G25B STEP 2 — STATIC GEOMETRY AT MEASURED HARDWARE CONTACT POSE"
        )
        print("=" * 150)

        model0 = mujoco.MjModel.from_xml_path(
            str(
                source_xml
            )
        )

        data0 = mujoco.MjData(
            model0
        )

        set_hardware_finger_pose(
            model0,
            data0,
            q_reference,
        )

        middle_gid = mujoco.mj_name2id(
            model0,
            mujoco.mjtObj.mjOBJ_GEOM,
            g12.CONTACT_GEOMS["mid"],
        )

        fixture_gid = mujoco.mj_name2id(
            model0,
            mujoco.mjtObj.mjOBJ_GEOM,
            g12.FIXTURE_NAME,
        )

        if (
            middle_gid < 0
            or
            fixture_gid < 0
        ):
            raise RuntimeError(
                "Could not resolve geometry IDs."
            )

        distance0, fromto = geom_distance(
            model0,
            data0,
            middle_gid,
            fixture_gid,
        )

        fixture_center = np.asarray(
            data0.geom_xpos[
                fixture_gid
            ],
            dtype=float,
        )

        middle_center = np.asarray(
            data0.geom_xpos[
                middle_gid
            ],
            dtype=float,
        )

        # ---------------------------------------------------------
        # Prefer the closest point on fixture returned by
        # mj_geomDistance.
        #
        # fromto:
        #   first xyz  = closest point on geom1
        #   second xyz = closest point on geom2
        #
        # fixture is geom2 here.
        # ---------------------------------------------------------

        fixture_surface_point = np.asarray(
            fromto[
                3:6
            ],
            dtype=float,
        )

        radial = (
            fixture_surface_point
            -
            fixture_center
        )

        radial[
            2
        ] = 0.0

        radial_norm = float(
            np.linalg.norm(
                radial
            )
        )

        # Fallback if closest-point output is degenerate.
        if radial_norm < 1e-8:

            radial = (
                middle_center
                -
                fixture_center
            )

            radial[
                2
            ] = 0.0

            radial_norm = float(
                np.linalg.norm(
                    radial
                )
            )

        if radial_norm < 1e-8:

            raise RuntimeError(
                "Could not determine bottle radial direction."
            )

        direction_xy = (
            radial[
                :2
            ]
            /
            radial_norm
        )

        print(
            f"static middle-to-bottle signed distance = "
            f"{1000.0 * distance0:+.3f} mm"
        )

        print(
            "radial translation direction xy = "
            f"[{direction_xy[0]:+.6f}, "
            f"{direction_xy[1]:+.6f}]"
        )

        # =========================================================
        # STEP 3 — solve one scalar geometric translation.
        #
        # This is NOT dynamics fitting.
        # We solve:
        #
        #   geom_distance(hardware contact pose) = 0
        #
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G25B STEP 3 — SOLVE FIXTURE RADIAL OFFSET"
        )
        print("=" * 150)

        temp_xml = (
            out
            /
            "_g25b_tmp.xml"
        )

        # Determine bracket.
        #
        # Positive shift = bottle moves toward middle finger.

        if distance0 >= 0.0:

            lo = 0.0
            flo = distance0

            hi = max(
                0.001,
                1.5
                *
                distance0,
            )

            fhi = static_distance_for_shift(
                source_xml,
                temp_xml,
                direction_xy,
                hi,
                q_reference,
            )

            while (
                fhi > 0.0
                and
                hi < 0.030
            ):

                hi *= 1.8

                fhi = static_distance_for_shift(
                    source_xml,
                    temp_xml,
                    direction_xy,
                    hi,
                    q_reference,
                )

            if fhi > 0.0:

                raise RuntimeError(
                    "Could not bracket geometric contact "
                    "within +30 mm fixture motion."
                )

        else:

            hi = 0.0
            fhi = distance0

            lo = -max(
                0.001,
                1.5
                *
                abs(
                    distance0
                ),
            )

            flo = static_distance_for_shift(
                source_xml,
                temp_xml,
                direction_xy,
                lo,
                q_reference,
            )

            while (
                flo < 0.0
                and
                abs(
                    lo
                ) < 0.030
            ):

                lo *= 1.8

                flo = static_distance_for_shift(
                    source_xml,
                    temp_xml,
                    direction_xy,
                    lo,
                    q_reference,
                )

            if flo < 0.0:

                raise RuntimeError(
                    "Could not bracket geometric contact "
                    "within -30 mm fixture motion."
                )

        # Bisection on static geometry.
        #
        # Very cheap and deterministic; this is not a parameter
        # sweep.

        for _ in range(
            22
        ):

            mid = (
                0.5
                *
                (
                    lo + hi
                )
            )

            fmid = static_distance_for_shift(
                source_xml,
                temp_xml,
                direction_xy,
                mid,
                q_reference,
            )

            if fmid > 0.0:

                lo = mid
                flo = fmid

            else:

                hi = mid
                fhi = fmid

        shift_m = (
            0.5
            *
            (
                lo + hi
            )
        )

        calibrated_xml = (
            out
            /
            "g25b_hardware_contact_calibrated.xml"
        )

        calibrated_pos = write_shifted_model(
            source_xml,
            calibrated_xml,
            direction_xy,
            shift_m,
        )

        final_static_distance = (
            static_distance_for_shift(
                source_xml,
                temp_xml,
                direction_xy,
                shift_m,
                q_reference,
            )
        )

        print(
            f"required radial fixture shift = "
            f"{1000.0 * shift_m:+.3f} mm"
        )

        print(
            "calibrated fixture pos = "
            f"{calibrated_pos[0]:+.9f}, "
            f"{calibrated_pos[1]:+.9f}, "
            f"{calibrated_pos[2]:+.9f}"
        )

        print(
            f"static residual distance = "
            f"{1e6 * final_static_distance:+.3f} um"
        )

        # =========================================================
        # STEP 4 — replay H1 ONCE with frozen calibrated geometry.
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G25B STEP 4 — DYNAMIC H1 REPLAY WITH CALIBRATED FIXTURE"
        )
        print("=" * 150)

        (
            rows,
            qpos,
            times,
            summary,
        ) = g12.run_audit(
            calibrated_xml,
            bag,
            replay,
        )

        mid_starts, mid_ends = first_episode(
            rows,
            "mid_contact",
        )

        tip_starts, tip_ends = first_episode(
            rows,
            "tip_contact",
        )

        if not mid_starts:

            raise RuntimeError(
                "No middle contact after calibration."
            )

        first = int(
            mid_starts[
                0
            ]
        )

        first_end = int(
            mid_ends[
                0
            ]
        )

        first_row = rows[
            first
        ]

        first_end_row = rows[
            first_end
        ]

        first_duration = (
            first_end_row[
                "time_s"
            ]
            -
            first_row[
                "time_s"
            ]
        )

        if len(
            mid_starts
        ) >= 2:

            second = int(
                mid_starts[
                    1
                ]
            )

            detached_gap = (
                rows[
                    second
                ][
                    "time_s"
                ]
                -
                first_end_row[
                    "time_s"
                ]
            )

        else:

            detached_gap = (
                0.0
            )

        print()
        print(
            "CALIBRATED dynamic first middle contact:"
        )

        print(
            f"  encoder = "
            f"{first_row['encoder_tick']:.3f} ticks"
        )

        print(
            f"  pose    = "
            f"{first_row['mcp_deg']:.3f} / "
            f"{first_row['pip_deg']:.3f} / "
            f"{first_row['dip_deg']:.3f} deg"
        )

        print()
        print(
            "HARDWARE first middle contact:"
        )

        print(
            f"  encoder = "
            f"{HW_ENCODER:.3f} ticks"
        )

        print(
            f"  pose    = "
            f"{HW_MCP:.3f} / "
            f"{HW_PIP:.3f} / "
            f"{HW_DIP:.3f} deg"
        )

        print()
        print(
            "FIRST-CONTACT ERRORS:"
        )

        print(
            f"  encoder error = "
            f"{first_row['encoder_tick'] - HW_ENCODER:+.3f} ticks"
        )

        print(
            f"  MCP error     = "
            f"{first_row['mcp_deg'] - HW_MCP:+.3f} deg"
        )

        print(
            f"  PIP error     = "
            f"{first_row['pip_deg'] - HW_PIP:+.3f} deg"
        )

        print(
            f"  DIP error     = "
            f"{first_row['dip_deg'] - HW_DIP:+.3f} deg"
        )

        print()
        print(
            "CONTACT CONTINUITY:"
        )

        print(
            f"  middle episodes       = "
            f"{len(mid_starts)}"
        )

        print(
            f"  tip episodes          = "
            f"{len(tip_starts)}"
        )

        print(
            f"  first-middle duration = "
            f"{first_duration:.3f} s"
        )

        print(
            f"  detached middle gap   = "
            f"{detached_gap:.3f} s"
        )

        # =========================================================
        # Final trusted target comparison.
        # =========================================================

        row_times = np.asarray(
            [
                r[
                    "time_s"
                ]
                for r in rows
            ],
            dtype=float,
        )

        target_idx = nearest_index(
            row_times,
            TARGET_BAG_TIME,
        )

        target_row = rows[
            target_idx
        ]

        simulated_target = np.asarray(
            [
                target_row[
                    "mcp_deg"
                ],
                target_row[
                    "pip_deg"
                ],
                target_row[
                    "dip_deg"
                ],
            ],
            dtype=float,
        )

        final_error = (
            simulated_target
            -
            TRUSTED_FINAL
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    final_error
                    **
                    2
                )
            )
        )

        print()
        print(
            "H1 TRUSTED CONTACT-POSE CHECK:"
        )

        print(
            "  simulation = "
            f"{simulated_target[0]:.3f} / "
            f"{simulated_target[1]:.3f} / "
            f"{simulated_target[2]:.3f}"
        )

        print(
            "  hardware   = "
            f"{TRUSTED_FINAL[0]:.3f} / "
            f"{TRUSTED_FINAL[1]:.3f} / "
            f"{TRUSTED_FINAL[2]:.3f}"
        )

        print(
            "  error      = "
            f"{final_error[0]:+.3f} / "
            f"{final_error[1]:+.3f} / "
            f"{final_error[2]:+.3f}"
        )

        print(
            f"  RMSE       = "
            f"{rmse:.3f} deg"
        )

        # =========================================================
        # DECISION.
        # =========================================================

        encoder_error = abs(
            first_row[
                "encoder_tick"
            ]
            -
            HW_ENCODER
        )

        pip_contact_error = abs(
            first_row[
                "pip_deg"
            ]
            -
            HW_PIP
        )

        dip_contact_error = abs(
            first_row[
                "dip_deg"
            ]
            -
            HW_DIP
        )

        print()
        print("=" * 150)
        print(
            "G25B DECISION"
        )
        print("=" * 150)

        if (
            encoder_error <= 40.0
            and
            pip_contact_error <= 4.0
            and
            dip_contact_error <= 4.0
        ):

            print(
                "RESULT: FIXTURE GEOMETRY EXPLAINS MOST "
                "OF THE FIRST-CONTACT MISMATCH."
            )

            print(
                "Next: retain calibrated geometry and "
                "revisit contact-load mechanics."
            )

        else:

            print(
                "RESULT: GEOMETRY ALONE CANNOT EXPLAIN "
                "THE FIRST-CONTACT MISMATCH."
            )

            print()

            print(
                "The bottle now touches the CAD middle phalanx "
                "at the measured hardware pose by construction,"
            )

            print(
                "but the dynamically simulated finger reaches "
                "that surface with the wrong MCP/PIP/DIP state."
            )

            print()

            print(
                "=> pre-contact tendon/passive mechanics must "
                "be identified before further contact tuning."
            )

        if len(
            mid_starts
        ) > 1:

            print()
            print(
                "Middle-contact bounce also still exists."
            )

        else:

            print()
            print(
                "Middle contact remains continuous in this run."
            )

        # =========================================================
        # Save compact result.
        # =========================================================

        result_path = (
            out
            /
            "g25b_summary.csv"
        )

        with result_path.open(
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "fixture_shift_mm",
                    "fixture_direction_x",
                    "fixture_direction_y",
                    "static_residual_um",
                    "hardware_contact_encoder",
                    "sim_contact_encoder",
                    "hardware_contact_mcp_deg",
                    "hardware_contact_pip_deg",
                    "hardware_contact_dip_deg",
                    "sim_contact_mcp_deg",
                    "sim_contact_pip_deg",
                    "sim_contact_dip_deg",
                    "middle_episodes",
                    "tip_episodes",
                    "middle_detached_gap_s",
                    "target_rmse_deg",
                ],
            )

            writer.writeheader()

            writer.writerow({
                "fixture_shift_mm":
                    1000.0
                    *
                    shift_m,

                "fixture_direction_x":
                    direction_xy[
                        0
                    ],

                "fixture_direction_y":
                    direction_xy[
                        1
                    ],

                "static_residual_um":
                    1e6
                    *
                    final_static_distance,

                "hardware_contact_encoder":
                    HW_ENCODER,

                "sim_contact_encoder":
                    first_row[
                        "encoder_tick"
                    ],

                "hardware_contact_mcp_deg":
                    HW_MCP,

                "hardware_contact_pip_deg":
                    HW_PIP,

                "hardware_contact_dip_deg":
                    HW_DIP,

                "sim_contact_mcp_deg":
                    first_row[
                        "mcp_deg"
                    ],

                "sim_contact_pip_deg":
                    first_row[
                        "pip_deg"
                    ],

                "sim_contact_dip_deg":
                    first_row[
                        "dip_deg"
                    ],

                "middle_episodes":
                    len(
                        mid_starts
                    ),

                "tip_episodes":
                    len(
                        tip_starts
                    ),

                "middle_detached_gap_s":
                    detached_gap,

                "target_rmse_deg":
                    rmse,
            })

        np.savez_compressed(
            out
            /
            "g25b_calibrated_trajectory.npz",
            times=np.asarray(
                times
            ),
            qpos=np.asarray(
                qpos
            ),
        )

        print()
        print(
            f"Summary:\n  {result_path}"
        )

        print()
        print(
            f"Calibrated XML:\n  {calibrated_xml}"
        )

        print()
        print(
            f"All outputs:\n  {out}"
        )

        # =========================================================
        # GUI
        # =========================================================

        if not args.no_gui:

            print()
            print("=" * 150)
            print(
                "G25B GUI — HARDWARE-POSE CALIBRATED FIXTURE"
            )
            print("=" * 150)

            print(
                "Watch:"
            )

            print(
                "  1. when middle first touches"
            )

            print(
                "  2. whether PIP is still already too flexed"
            )

            print(
                "  3. whether middle stays or bounces away"
            )

            print(
                "  4. what DIP does after middle contact"
            )

            g7.exact_gui_replay(
                calibrated_xml,
                qpos,
                times,
                "G25B hardware-contact geometry calibrated",
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

        replay.CONTACT_DAMPING = (
            old_cd
        )


if __name__ == "__main__":
    main()
