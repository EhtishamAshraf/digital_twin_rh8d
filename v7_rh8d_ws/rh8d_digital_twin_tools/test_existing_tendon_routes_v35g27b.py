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
# G27B — EXISTING RH8D TENDON-ROUTE AUDIT
#
# We do NOT invent another pulley/wrap geometry.
#
# Compare two routing families ALREADY PRESENT in the RH8D model:
#
# NATIVE:
#   v35_spool_site
#   -> palm_flexor_origin_idx
#   -> idx_guide_p
#   -> idx_guide_m
#   -> idx_guide_d
#   -> idx_anchor
#
# This is the original RH8D spatial-tendon architecture.
#
# CFG2:
#   v35_spool_site
#   -> cfg2_idx_origin
#   -> cfg2_idx_guide_p
#   -> cfg2_idx_guide_m
#   -> cfg2_idx_guide_d
#   -> cfg2_idx_anchor
#
# This alternate guide family also already exists in the XML.
#
# For each route:
#   1. preserve open-state tendon preload
#   2. audit signed MCP/PIP/DIP moment arms over measured HW range
#   3. reject any sign reversal or collapse
#
# ONLY the best passing route gets ONE H1 dynamic run.
#
# No spring tuning.
# No friction tuning.
# No contact tuning.
# No tendon stiffness tuning.
# =====================================================================


TENDON_NAME = "idx_flex_spatial"

ROUTES = {
    "NATIVE": [
        "v35_spool_site",
        "palm_flexor_origin_idx",
        "idx_guide_p",
        "idx_guide_m",
        "idx_guide_d",
        "idx_anchor",
    ],

    "CFG2": [
        "v35_spool_site",
        "cfg2_idx_origin",
        "cfg2_idx_guide_p",
        "cfg2_idx_guide_m",
        "cfg2_idx_guide_d",
        "cfg2_idx_anchor",
    ],
}


# Measured hardware free-space closing configurations.
FREE_SPACE = [
    # tick, MCP, PIP, DIP
    (100.0,     0.000,   0.000,   0.000),
    (500.0,    16.952,   0.968,  -0.428),
    (1000.0,   39.472,  -0.907,  -0.343),
    (1500.0,   57.291,   3.654,  -4.019),
    (2000.0,   73.632,   4.667,   2.012),
    (2500.0,   73.632,  27.119,   3.190),
    (3000.0,   77.347,  48.581,   7.810),
    (3500.0,   80.074,  70.872,  -7.199),
]


# CAD-digitized low-angle effective moment arms.
CAD_TARGET_MM = np.array([
    4.776,
    4.496,
    4.669,
])


HW_CONTACT_ENCODER = 673.573

HW_CONTACT_POSE = np.array([
    18.543,
    6.161,
    5.276,
])


TRUSTED_FINAL = np.array([
    14.5355,
    23.8110,
    9.3270,
])


TARGET_TIME = 38.970486


def nearest_index(values, target):

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


def spatial_element(root):

    for tendon_root in root.iter("tendon"):

        for spatial in tendon_root.findall("spatial"):

            if spatial.get("name") == TENDON_NAME:

                return spatial

    raise RuntimeError(
        f"Tendon not found: {TENDON_NAME}"
    )


def springlength_values(spatial):

    raw = spatial.get("springlength")

    if raw is None:
        return None

    return [
        float(x)
        for x in raw.split()
    ]


def shift_springlength(
    spatial,
    original,
    delta,
):

    if original is None:

        raise RuntimeError(
            "idx_flex_spatial has no explicit springlength."
        )

    values = []

    for x in original:

        if x >= 0.0:
            values.append(
                x + delta
            )
        else:
            values.append(
                x
            )

    spatial.set(
        "springlength",
        " ".join(
            f"{x:.12g}"
            for x in values
        ),
    )


def all_site_names(model):

    names = set()

    for sid in range(
        model.nsite
    ):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            sid,
        )

        if name:
            names.add(name)

    return names


def tendon_J(
    model,
    data,
    tid,
):

    raw = np.asarray(
        data.ten_J,
        dtype=float,
    )

    if (
        raw.ndim == 2
        and
        raw.shape
        ==
        (
            model.ntendon,
            model.nv,
        )
    ):

        return raw[
            tid,
            :
        ].copy()

    if (
        raw.ndim == 1
        and
        raw.size
        ==
        model.ntendon
        *
        model.nv
    ):

        return raw.reshape(
            model.ntendon,
            model.nv,
        )[
            tid,
            :
        ].copy()

    J = np.zeros(
        model.nv,
        dtype=float,
    )

    rownnz = np.asarray(
        model.ten_J_rownnz,
        dtype=int,
    )

    rowadr = np.asarray(
        model.ten_J_rowadr,
        dtype=int,
    )

    colind = np.asarray(
        model.ten_J_colind,
        dtype=int,
    )

    nnz = int(
        rownnz[
            tid
        ]
    )

    adr = int(
        rowadr[
            tid
        ]
    )

    for k in range(nnz):

        i = adr + k

        J[
            int(
                colind[
                    i
                ]
            )
        ] = float(
            raw[
                i
            ]
        )

    return J


def build_route(
    flat_xml,
    route_name,
    route_sites,
    old_open_length,
    output_dir,
):

    tree = ET.parse(
        flat_xml
    )

    root = tree.getroot()

    spatial = spatial_element(
        root
    )

    original_springlength = (
        springlength_values(
            spatial
        )
    )

    for child in list(
        spatial
    ):

        spatial.remove(
            child
        )

    for site_name in route_sites:

        spatial.append(
            ET.Element(
                "site",
                {
                    "site":
                        site_name,
                },
            )
        )

    preadjust = (
        output_dir
        /
        f"g27b_{route_name.lower()}_preadjust.xml"
    )

    tree.write(
        preadjust,
        encoding="utf-8",
        xml_declaration=True,
    )

    model_pre = mujoco.MjModel.from_xml_path(
        str(
            preadjust
        )
    )

    data_pre = mujoco.MjData(
        model_pre
    )

    data_pre.qpos[:] = (
        model_pre.qpos0
    )

    data_pre.qvel[:] = 0.0

    mujoco.mj_forward(
        model_pre,
        data_pre,
    )

    tid = mujoco.mj_name2id(
        model_pre,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    new_open_length = float(
        data_pre.ten_length[
            tid
        ]
    )

    delta = (
        new_open_length
        -
        old_open_length
    )

    # Re-open and preserve the same OPEN preload/extension.
    tree = ET.parse(
        preadjust
    )

    root = tree.getroot()

    spatial = spatial_element(
        root
    )

    shift_springlength(
        spatial,
        original_springlength,
        delta,
    )

    final_xml = (
        output_dir
        /
        f"g27b_{route_name.lower()}_candidate.xml"
    )

    tree.write(
        final_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    # Compile final candidate.
    final_model = mujoco.MjModel.from_xml_path(
        str(
            final_xml
        )
    )

    return (
        final_xml,
        new_open_length,
        delta,
    )


def static_audit(
    route_name,
    xml,
):

    model = mujoco.MjModel.from_xml_path(
        str(
            xml
        )
    )

    data = mujoco.MjData(
        model
    )

    refs = g12.get_joint_refs(
        model
    )

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    rows = []

    for (
        tick,
        mcp_deg,
        pip_deg,
        dip_deg,
    ) in FREE_SPACE:

        data.qpos[:] = (
            model.qpos0
        )

        data.qvel[:] = 0.0

        data.qpos[
            refs[
                "mcp"
            ][
                "qadr"
            ]
        ] = math.radians(
            mcp_deg
        )

        data.qpos[
            refs[
                "pip"
            ][
                "qadr"
            ]
        ] = math.radians(
            pip_deg
        )

        data.qpos[
            refs[
                "dip"
            ][
                "qadr"
            ]
        ] = math.radians(
            dip_deg
        )

        mujoco.mj_forward(
            model,
            data,
        )

        J = tendon_J(
            model,
            data,
            tid,
        )

        jm = (
            1000.0
            *
            float(
                J[
                    refs[
                        "mcp"
                    ][
                        "dadr"
                    ]
                ]
            )
        )

        jp = (
            1000.0
            *
            float(
                J[
                    refs[
                        "pip"
                    ][
                        "dadr"
                    ]
                ]
            )
        )

        jd = (
            1000.0
            *
            float(
                J[
                    refs[
                        "dip"
                    ][
                        "dadr"
                    ]
                ]
            )
        )

        rows.append({
            "route":
                route_name,

            "encoder_tick":
                tick,

            "mcp_deg":
                mcp_deg,

            "pip_deg":
                pip_deg,

            "dip_deg":
                dip_deg,

            "mcp_J_mm_rad":
                jm,

            "pip_J_mm_rad":
                jp,

            "dip_J_mm_rad":
                jd,

            "tendon_length_mm":
                1000.0
                *
                float(
                    data.ten_length[
                        tid
                    ]
                ),
        })

    # --------------------------------------------------------------
    # Physical validity:
    #
    # Positive flexion requires:
    #
    #     tau = -F * J
    #
    # so J must remain NEGATIVE for a positive tendon force.
    # --------------------------------------------------------------

    values = np.asarray(
        [
            [
                r[
                    "mcp_J_mm_rad"
                ],
                r[
                    "pip_J_mm_rad"
                ],
                r[
                    "dip_J_mm_rad"
                ],
            ]
            for r in rows
        ],
        dtype=float,
    )

    moving = values[
        1:,
        :
    ]

    sign_pass = bool(
        np.all(
            moving < 0.0
        )
    )

    min_abs = np.min(
        np.abs(
            moving
        ),
        axis=0,
    )

    collapse_pass = bool(
        np.all(
            min_abs >= 1.0
        )
    )

    # Open-pose agreement with CAD digitization.
    open_abs = np.abs(
        values[
            0,
            :
        ]
    )

    cad_error = (
        open_abs
        -
        CAD_TARGET_MM
    )

    cad_rmse = float(
        np.sqrt(
            np.mean(
                cad_error
                **
                2
            )
        )
    )

    full_pass = bool(
        sign_pass
        and
        collapse_pass
    )

    metrics = {
        "route":
            route_name,

        "sign_pass":
            sign_pass,

        "collapse_pass":
            collapse_pass,

        "full_pass":
            full_pass,

        "min_abs_mcp_mm":
            float(
                min_abs[
                    0
                ]
            ),

        "min_abs_pip_mm":
            float(
                min_abs[
                    1
                ]
            ),

        "min_abs_dip_mm":
            float(
                min_abs[
                    2
                ]
            ),

        "open_mcp_mm":
            float(
                open_abs[
                    0
                ]
            ),

        "open_pip_mm":
            float(
                open_abs[
                    1
                ]
            ),

        "open_dip_mm":
            float(
                open_abs[
                    2
                ]
            ),

        "cad_rmse_mm":
            cad_rmse,
    }

    return (
        rows,
        metrics,
    )


def episode_indices(
    rows,
    key,
):

    flags = np.asarray(
        [
            bool(
                row[
                    key
                ]
            )
            for row in rows
        ],
        dtype=bool,
    )

    return (
        g12.contact_episode_starts(
            flags
        ),
        g12.contact_episode_ends(
            flags
        ),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g27b_existing_tendon_routes"
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

    source_xml = (
        args.xml
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

    # --------------------------------------------------------------
    # Flatten source model.
    # --------------------------------------------------------------

    source_model = mujoco.MjModel.from_xml_path(
        str(
            source_xml
        )
    )

    flat_xml = (
        out
        /
        "g27b_flat_source.xml"
    )

    mujoco.mj_saveLastXML(
        str(
            flat_xml
        ),
        source_model,
    )

    source_model = mujoco.MjModel.from_xml_path(
        str(
            flat_xml
        )
    )

    source_data = mujoco.MjData(
        source_model
    )

    source_data.qpos[:] = (
        source_model.qpos0
    )

    mujoco.mj_forward(
        source_model,
        source_data,
    )

    source_tid = mujoco.mj_name2id(
        source_model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    old_open_length = float(
        source_data.ten_length[
            source_tid
        ]
    )

    available_sites = all_site_names(
        source_model
    )

    print()
    print("=" * 150)
    print(
        "G27B — EXISTING RH8D TENDON ROUTES"
    )
    print("=" * 150)

    print(
        f"current open tendon length = "
        f"{1000.0 * old_open_length:.6f} mm"
    )

    # --------------------------------------------------------------
    # Verify all requested existing sites.
    # --------------------------------------------------------------

    for route_name, route_sites in ROUTES.items():

        missing = [
            site
            for site in route_sites
            if site not in available_sites
        ]

        if missing:

            raise RuntimeError(
                f"{route_name}: missing sites: "
                f"{missing}"
            )

    # --------------------------------------------------------------
    # Build + static audit both EXISTING route families.
    # --------------------------------------------------------------

    all_rows = []

    metrics = []

    route_xmls = {}

    for route_name, route_sites in ROUTES.items():

        print()
        print("=" * 150)
        print(
            f"G27B ROUTE — {route_name}"
        )
        print("=" * 150)

        print(
            " -> ".join(
                route_sites
            )
        )

        (
            candidate_xml,
            new_open_length,
            delta,
        ) = build_route(
            flat_xml,
            route_name,
            route_sites,
            old_open_length,
            out,
        )

        route_xmls[
            route_name
        ] = candidate_xml

        print()
        print(
            f"open path length = "
            f"{1000.0 * new_open_length:.6f} mm"
        )

        print(
            f"springlength compensation = "
            f"{1000.0 * delta:+.6f} mm"
        )

        (
            rows,
            m,
        ) = static_audit(
            route_name,
            candidate_xml,
        )

        all_rows.extend(
            rows
        )

        metrics.append(
            m
        )

        print()
        print(
            "tick     MCP-J      PIP-J      DIP-J"
        )

        print("-" * 80)

        for row in rows:

            print(
                f"{row['encoder_tick']:4.0f}   "
                f"{row['mcp_J_mm_rad']:+9.3f} "
                f"{row['pip_J_mm_rad']:+9.3f} "
                f"{row['dip_J_mm_rad']:+9.3f}"
            )

        print()
        print(
            f"all flexion signs negative = "
            f"{m['sign_pass']}"
        )

        print(
            "minimum |J| MCP/PIP/DIP = "
            f"{m['min_abs_mcp_mm']:.3f} / "
            f"{m['min_abs_pip_mm']:.3f} / "
            f"{m['min_abs_dip_mm']:.3f} mm"
        )

        print(
            "open |J| MCP/PIP/DIP = "
            f"{m['open_mcp_mm']:.3f} / "
            f"{m['open_pip_mm']:.3f} / "
            f"{m['open_dip_mm']:.3f} mm"
        )

        print(
            f"CAD moment-arm RMSE = "
            f"{m['cad_rmse_mm']:.3f} mm"
        )

        print(
            f"FULL-RANGE PASS = "
            f"{m['full_pass']}"
        )

    # --------------------------------------------------------------
    # Save geometry results.
    # --------------------------------------------------------------

    audit_csv = (
        out
        /
        "g27b_route_audit.csv"
    )

    with audit_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                all_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )

    metrics_csv = (
        out
        /
        "g27b_route_metrics.csv"
    )

    with metrics_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                metrics[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            metrics
        )

    # --------------------------------------------------------------
    # Select ONLY among physically passing routes.
    # --------------------------------------------------------------

    passing = [
        m
        for m in metrics
        if m[
            "full_pass"
        ]
    ]

    print()
    print("=" * 150)
    print(
        "G27B STATIC DECISION"
    )
    print("=" * 150)

    if not passing:

        print(
            "RESULT: NEITHER EXISTING GUIDE ROUTE PASSES "
            "THE FULL-RANGE MOMENT-ARM TEST."
        )

        print()
        print(
            "STOP — no H1 dynamics will be run."
        )

        print(
            "This means a true persistent wrap/contact surface "
            "must be reconstructed from the CAD."
        )

        print()
        print(
            f"Audit:\n  {audit_csv}"
        )

        print(
            f"Metrics:\n  {metrics_csv}"
        )

        return

    best = min(
        passing,
        key=lambda x:
            x[
                "cad_rmse_mm"
            ],
    )

    best_name = best[
        "route"
    ]

    best_xml = route_xmls[
        best_name
    ]

    print(
        f"Selected route = "
        f"{best_name}"
    )

    print(
        "Reason:"
    )

    print(
        "  - no torque-direction reversal"
    )

    print(
        "  - no moment-arm collapse"
    )

    print(
        "  - smallest error relative to CAD-digitized "
        "low-angle moment arms"
    )

    # --------------------------------------------------------------
    # ONE dynamic H1 run.
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        f"G27B H1 DYNAMIC RUN — {best_name}"
    )
    print("=" * 150)

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    (
        rows,
        qpos,
        times,
        summary,
    ) = g12.run_audit(
        best_xml,
        bag,
        replay,
    )

    mid_starts, mid_ends = episode_indices(
        rows,
        "mid_contact",
    )

    tip_starts, tip_ends = episode_indices(
        rows,
        "tip_contact",
    )

    if not mid_starts:

        raise RuntimeError(
            "Selected route produced no middle contact."
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

    first_pose = np.array([
        first_row[
            "mcp_deg"
        ],
        first_row[
            "pip_deg"
        ],
        first_row[
            "dip_deg"
        ],
    ])

    first_pose_error = (
        first_pose
        -
        HW_CONTACT_POSE
    )

    first_pose_rmse = float(
        np.sqrt(
            np.mean(
                first_pose_error
                **
                2
            )
        )
    )

    first_duration = (
        rows[
            first_end
        ][
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

        detached_gap = (
            rows[
                mid_starts[
                    1
                ]
            ][
                "time_s"
            ]
            -
            rows[
                first_end
            ][
                "time_s"
            ]
        )

    else:

        detached_gap = 0.0

    print()
    print(
        "FIRST MIDDLE CONTACT"
    )

    print(
        f"  simulation encoder = "
        f"{first_row['encoder_tick']:.3f}"
    )

    print(
        f"  hardware encoder   = "
        f"{HW_CONTACT_ENCODER:.3f}"
    )

    print(
        f"  encoder error      = "
        f"{first_row['encoder_tick'] - HW_CONTACT_ENCODER:+.3f}"
    )

    print()

    print(
        "  simulation pose = "
        f"{first_pose[0]:.3f} / "
        f"{first_pose[1]:.3f} / "
        f"{first_pose[2]:.3f}"
    )

    print(
        "  hardware pose   = "
        f"{HW_CONTACT_POSE[0]:.3f} / "
        f"{HW_CONTACT_POSE[1]:.3f} / "
        f"{HW_CONTACT_POSE[2]:.3f}"
    )

    print(
        f"  pose RMSE       = "
        f"{first_pose_rmse:.3f} deg"
    )

    print()
    print(
        "CONTACT CONTINUITY"
    )

    print(
        f"  middle episodes = "
        f"{len(mid_starts)}"
    )

    print(
        f"  tip episodes    = "
        f"{len(tip_starts)}"
    )

    print(
        f"  first duration  = "
        f"{first_duration:.3f} s"
    )

    print(
        f"  detached gap    = "
        f"{detached_gap:.3f} s"
    )

    # --------------------------------------------------------------
    # Trusted final pose.
    # --------------------------------------------------------------

    row_times = np.asarray(
        [
            row[
                "time_s"
            ]
            for row in rows
        ],
        dtype=float,
    )

    ti = nearest_index(
        row_times,
        TARGET_TIME,
    )

    target = rows[
        ti
    ]

    sim_final = np.array([
        target[
            "mcp_deg"
        ],
        target[
            "pip_deg"
        ],
        target[
            "dip_deg"
        ],
    ])

    final_error = (
        sim_final
        -
        TRUSTED_FINAL
    )

    final_rmse = float(
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
        "TRUSTED FINAL H1"
    )

    print(
        "  simulation = "
        f"{sim_final[0]:.3f} / "
        f"{sim_final[1]:.3f} / "
        f"{sim_final[2]:.3f}"
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
        f"{final_rmse:.3f} deg"
    )

    # --------------------------------------------------------------
    # Summary.
    # --------------------------------------------------------------

    summary_csv = (
        out
        /
        "g27b_dynamic_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "selected_route",
                "cad_moment_arm_rmse_mm",
                "first_contact_encoder",
                "first_contact_encoder_error",
                "first_contact_pose_rmse_deg",
                "middle_episodes",
                "tip_episodes",
                "middle_detached_gap_s",
                "final_rmse_deg",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "selected_route":
                best_name,

            "cad_moment_arm_rmse_mm":
                best[
                    "cad_rmse_mm"
                ],

            "first_contact_encoder":
                first_row[
                    "encoder_tick"
                ],

            "first_contact_encoder_error":
                first_row[
                    "encoder_tick"
                ]
                -
                HW_CONTACT_ENCODER,

            "first_contact_pose_rmse_deg":
                first_pose_rmse,

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

            "final_rmse_deg":
                final_rmse,
        })

    np.savez_compressed(
        out
        /
        "g27b_selected_trajectory.npz",
        times=np.asarray(
            times
        ),
        qpos=np.asarray(
            qpos
        ),
    )

    print()
    print("=" * 150)
    print(
        "G27B DECISION"
    )
    print("=" * 150)

    if len(
        mid_starts
    ) == 1:

        print(
            "Selected existing route keeps middle contact continuous."
        )

    else:

        print(
            "Selected route fixes full-range transmission geometry,"
        )

        print(
            "but H1 middle-contact bounce remains."
        )

    print()
    print(
        f"Selected XML:\n  {best_xml}"
    )

    print(
        f"Geometry audit:\n  {audit_csv}"
    )

    print(
        f"Summary:\n  {summary_csv}"
    )

    print(
        f"All outputs:\n  {out}"
    )

    if not args.no_gui:

        print()
        print("=" * 150)
        print(
            f"G27B GUI — {best_name} ROUTE"
        )
        print("=" * 150)

        print(
            "Check:"
        )

        print(
            "  MCP-first approach"
        )

        print(
            "  -> middle first contact"
        )

        print(
            "  -> whether middle stays"
        )

        print(
            "  -> PIP wrapping"
        )

        print(
            "  -> distal contact"
        )

        print(
            "  -> DIP participation"
        )

        g7.exact_gui_replay(
            best_xml,
            qpos,
            times,
            f"G27B existing route {best_name}",
        )


if __name__ == "__main__":
    main()
