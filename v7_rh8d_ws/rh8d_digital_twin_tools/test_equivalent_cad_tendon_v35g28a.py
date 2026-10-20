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
# G28A — CAD-EQUIVALENT CONSTANT-MOMENT-ARM TENDON
#
# G26C:
#   current V35 spatial tendon reverses MCP/PIP torque direction.
#
# G27A:
#   guessed cylindrical wrap was not persistent.
#
# G27B:
#   NATIVE route = wrong sign
#   CFG2 route   = MCP moment-arm collapse
#
# Therefore we stop guessing hidden proprietary pulley geometry.
#
# Instead use a reduced-order continuous cable model derived directly
# from the externally digitized CAD tendon path:
#
#   L =
#       +1.0       * q_spool
#       -0.004776  * q_MCP
#       -0.004496  * q_PIP
#       -0.004669  * q_DIP
#
# The coefficients are CAD-derived effective moment arms.
#
# This is one continuous tendon, not independent joint actuators.
#
# NO:
#   contact tuning
#   friction hierarchy
#   spring fitting
#   nonlinear passive law
#   parameter scan
#   state/contact switching
#
# Existing tendon stiffness/damping are preserved.
# =====================================================================


TENDON_NAME = "idx_flex_spatial"

SPOOL_NAME = "index_motor_spool"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"


# CAD-derived effective moment arms [m/rad].
MCP_COEF = -0.004776
PIP_COEF = -0.004496
DIP_COEF = -0.004669

SPOOL_COEF = +1.0


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


H1_PRECONTACT = [
    # encoder, MCP, PIP, DIP
    (416.1, 10.620, 3.994, 1.911),
    (611.7, 18.094, 4.526, 3.724),
    (673.573, 18.543, 6.161, 5.276),
]


HW_CONTACT_ENCODER = 673.573

HW_CONTACT_POSE = np.array(
    [
        18.543,
        6.161,
        5.276,
    ],
    dtype=float,
)


TRUSTED_FINAL = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


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


def find_index_tendon(root):

    for tendon_root in root.iter("tendon"):

        children = list(
            tendon_root
        )

        for i, child in enumerate(children):

            if child.get("name") == TENDON_NAME:

                return (
                    tendon_root,
                    i,
                    child,
                )

    raise RuntimeError(
        f"Tendon not found: {TENDON_NAME}"
    )


def find_geom(root, name):

    for geom in root.iter("geom"):

        if geom.get("name") == name:

            return geom

    raise RuntimeError(
        f"Geom not found: {name}"
    )


def tendon_jacobian_row(
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
        raw.shape == (
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

        idx = (
            adr + k
        )

        J[
            int(
                colind[
                    idx
                ]
            )
        ] = float(
            raw[
                idx
            ]
        )

    return J


def make_fixed_tendon_model(
    source_xml,
    temp_xml,
    final_xml,
):

    # -------------------------------------------------------------
    # Flatten first.
    # -------------------------------------------------------------

    source_model = mujoco.MjModel.from_xml_path(
        str(
            source_xml
        )
    )

    source_data = mujoco.MjData(
        source_model
    )

    source_data.qpos[:] = (
        source_model.qpos0
    )

    source_data.qvel[:] = 0.0

    mujoco.mj_forward(
        source_model,
        source_data,
    )

    old_tid = mujoco.mj_name2id(
        source_model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    old_length = float(
        source_data.ten_length[
            old_tid
        ]
    )

    old_reference = (
        g12.tendon_spring_reference(
            source_model,
            old_tid,
        )
    )

    old_extension = (
        g12.signed_extension(
            old_length,
            old_reference,
        )
    )

    flat_xml = (
        final_xml.parent
        /
        "g28a_flat_source.xml"
    )

    mujoco.mj_saveLastXML(
        str(
            flat_xml
        ),
        source_model,
    )

    tree = ET.parse(
        flat_xml
    )

    root = tree.getroot()

    (
        tendon_parent,
        tendon_index,
        old_element,
    ) = find_index_tendon(
        root
    )

    # Preserve physical tendon material parameters.
    stiffness = old_element.get(
        "stiffness",
        "0",
    )

    damping = old_element.get(
        "damping",
        "0",
    )

    frictionloss = old_element.get(
        "frictionloss",
        "0",
    )

    tendon_parent.remove(
        old_element
    )

    fixed = ET.Element(
        "fixed",
        {
            "name":
                TENDON_NAME,

            "stiffness":
                stiffness,

            "damping":
                damping,

            "frictionloss":
                frictionloss,

            # Temporary; corrected after compile.
            "springlength":
                "0",
        },
    )

    ET.SubElement(
        fixed,
        "joint",
        {
            "joint":
                SPOOL_NAME,

            "coef":
                f"{SPOOL_COEF:.12g}",
        },
    )

    ET.SubElement(
        fixed,
        "joint",
        {
            "joint":
                MCP_NAME,

            "coef":
                f"{MCP_COEF:.12g}",
        },
    )

    ET.SubElement(
        fixed,
        "joint",
        {
            "joint":
                PIP_NAME,

            "coef":
                f"{PIP_COEF:.12g}",
        },
    )

    ET.SubElement(
        fixed,
        "joint",
        {
            "joint":
                DIP_NAME,

            "coef":
                f"{DIP_COEF:.12g}",
        },
    )

    tendon_parent.insert(
        tendon_index,
        fixed,
    )

    tree.write(
        temp_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    # -------------------------------------------------------------
    # Compile temporary model to obtain its open fixed-tendon length.
    # -------------------------------------------------------------

    temp_model = mujoco.MjModel.from_xml_path(
        str(
            temp_xml
        )
    )

    temp_data = mujoco.MjData(
        temp_model
    )

    temp_data.qpos[:] = (
        temp_model.qpos0
    )

    temp_data.qvel[:] = 0.0

    mujoco.mj_forward(
        temp_model,
        temp_data,
    )

    temp_tid = mujoco.mj_name2id(
        temp_model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    fixed_open_length = float(
        temp_data.ten_length[
            temp_tid
        ]
    )

    # Preserve the old open-state extension.
    desired_reference = (
        fixed_open_length
        -
        old_extension
    )

    # Old extension is essentially zero in this model. Avoid an
    # unnecessary tiny negative reference caused by floating point.
    if (
        abs(
            old_extension
        )
        <
        1e-5
        and
        abs(
            desired_reference
        )
        <
        1e-5
    ):

        desired_reference = 0.0

    # -------------------------------------------------------------
    # Write final spring reference.
    # -------------------------------------------------------------

    tree = ET.parse(
        temp_xml
    )

    root = tree.getroot()

    (
        _,
        _,
        fixed_element,
    ) = find_index_tendon(
        root
    )

    fixed_element.set(
        "springlength",
        f"{desired_reference:.12g}",
    )

    tree.write(
        final_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    # Final compile.
    final_model = mujoco.MjModel.from_xml_path(
        str(
            final_xml
        )
    )

    return {
        "old_open_length":
            old_length,

        "old_reference":
            old_reference,

        "old_extension":
            old_extension,

        "fixed_open_length":
            fixed_open_length,

        "fixed_reference":
            desired_reference,

        "stiffness":
            float(
                stiffness
            ),

        "damping":
            float(
                damping
            ),
    }


def disable_fixture(
    source,
    destination,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    fixture = find_geom(
        root,
        g12.FIXTURE_NAME,
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
            "v35g28a_equivalent_cad_tendon"
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

    temp_xml = (
        out
        /
        "g28a_fixed_temp.xml"
    )

    candidate_xml = (
        out
        /
        "g28a_equivalent_cad_tendon.xml"
    )

    free_xml = (
        out
        /
        "g28a_equivalent_cad_tendon_free.xml"
    )

    # =============================================================
    # STEP 1 — BUILD EQUIVALENT CONTINUOUS TENDON
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G28A STEP 1 — BUILD CAD-EQUIVALENT CONTINUOUS TENDON"
    )
    print("=" * 150)

    info = make_fixed_tendon_model(
        source_xml,
        temp_xml,
        candidate_xml,
    )

    print()
    print(
        "Tendon equation:"
    )

    print(
        "  L = "
        "+1.000000*q_spool "
        "-0.004776*q_MCP "
        "-0.004496*q_PIP "
        "-0.004669*q_DIP"
    )

    print()

    print(
        f"preserved stiffness = "
        f"{info['stiffness']:.3f} N/m"
    )

    print(
        f"preserved damping   = "
        f"{info['damping']:.3f}"
    )

    print()

    print(
        f"old open extension  = "
        f"{1000.0 * info['old_extension']:+.6f} mm"
    )

    print(
        f"new open length     = "
        f"{1000.0 * info['fixed_open_length']:+.6f} mm"
    )

    print(
        f"new spring reference= "
        f"{1000.0 * info['fixed_reference']:+.6f} mm"
    )

    # =============================================================
    # STEP 2 — FULL-RANGE STATIC AUDIT
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G28A STEP 2 — FULL-RANGE MOMENT-ARM AUDIT"
    )
    print("=" * 150)

    model = mujoco.MjModel.from_xml_path(
        str(
            candidate_xml
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

    static_rows = []

    print()
    print(
        "tick    MCP     PIP     DIP      "
        "J_MCP     J_PIP     J_DIP"
    )

    print("-" * 120)

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

        J = tendon_jacobian_row(
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

        static_rows.append({
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

        print(
            f"{tick:4.0f} "
            f"{mcp_deg:7.2f} "
            f"{pip_deg:7.2f} "
            f"{dip_deg:7.2f} "
            f"{jm:+10.4f} "
            f"{jp:+10.4f} "
            f"{jd:+10.4f}"
        )

    Jvalues = np.asarray(
        [
            [
                row[
                    "mcp_J_mm_rad"
                ],
                row[
                    "pip_J_mm_rad"
                ],
                row[
                    "dip_J_mm_rad"
                ],
            ]
            for row in static_rows
        ],
        dtype=float,
    )

    expected = np.array(
        [
            1000.0 * MCP_COEF,
            1000.0 * PIP_COEF,
            1000.0 * DIP_COEF,
        ]
    )

    max_J_error = float(
        np.max(
            np.abs(
                Jvalues
                -
                expected[
                    None,
                    :
                ]
            )
        )
    )

    routing_pass = bool(
        np.all(
            Jvalues < 0.0
        )
        and
        max_J_error < 1e-6
    )

    print()
    print(
        f"maximum coefficient error = "
        f"{max_J_error:.9f} mm/rad"
    )

    print(
        f"FULL-RANGE ROUTING PASS   = "
        f"{routing_pass}"
    )

    if not routing_pass:

        raise RuntimeError(
            "Equivalent tendon coefficients did not compile "
            "as expected."
        )

    static_csv = (
        out
        /
        "g28a_full_range_moment_arms.csv"
    )

    with static_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                static_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            static_rows
        )

    # =============================================================
    # STEP 3 — H1 PRE-CONTACT, BOTTLE DISABLED
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G28A STEP 3 — H1 PRE-CONTACT VALIDATION"
    )
    print("=" * 150)

    disable_fixture(
        candidate_xml,
        free_xml,
    )

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    (
        free_rows,
        free_qpos,
        free_times,
        free_summary,
    ) = g12.run_audit(
        free_xml,
        bag,
        replay,
    )

    free_encoder = np.asarray(
        [
            row[
                "encoder_tick"
            ]
            for row in free_rows
        ],
        dtype=float,
    )

    errors = []

    print()
    print(
        "encoder        hardware MCP/PIP/DIP       "
        "simulation MCP/PIP/DIP       error"
    )

    print("-" * 150)

    for (
        tick,
        hm,
        hp,
        hd,
    ) in H1_PRECONTACT:

        i = nearest_index(
            free_encoder,
            tick,
        )

        row = free_rows[
            i
        ]

        sim = np.array(
            [
                row[
                    "mcp_deg"
                ],
                row[
                    "pip_deg"
                ],
                row[
                    "dip_deg"
                ],
            ],
            dtype=float,
        )

        hw = np.array(
            [
                hm,
                hp,
                hd,
            ],
            dtype=float,
        )

        e = (
            sim - hw
        )

        errors.extend(
            e.tolist()
        )

        print(
            f"{tick:8.3f}     "
            f"{hm:7.3f}/{hp:7.3f}/{hd:7.3f}     "
            f"{sim[0]:7.3f}/{sim[1]:7.3f}/{sim[2]:7.3f}     "
            f"{e[0]:+7.3f}/{e[1]:+7.3f}/{e[2]:+7.3f}"
        )

    precontact_rmse = float(
        np.sqrt(
            np.mean(
                np.asarray(
                    errors
                )
                **
                2
            )
        )
    )

    print()
    print(
        f"H1 pre-contact RMSE = "
        f"{precontact_rmse:.3f} deg"
    )

    # =============================================================
    # STEP 4 — ONE H1 CONTACT RUN
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G28A STEP 4 — ONE H1 CONTACT RUN"
    )
    print("=" * 150)

    (
        rows,
        qpos,
        times,
        summary,
    ) = g12.run_audit(
        candidate_xml,
        bag,
        replay,
    )

    mid_starts, mid_ends = (
        episode_indices(
            rows,
            "mid_contact",
        )
    )

    tip_starts, tip_ends = (
        episode_indices(
            rows,
            "tip_contact",
        )
    )

    if not mid_starts:

        print()
        print(
            "NO MIDDLE CONTACT OCCURRED."
        )

        first_encoder = np.nan
        first_pose_rmse = np.nan
        detached_gap = np.nan
        first_duration = np.nan

    else:

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

        first_pose = np.array(
            [
                first_row[
                    "mcp_deg"
                ],
                first_row[
                    "pip_deg"
                ],
                first_row[
                    "dip_deg"
                ],
            ],
            dtype=float,
        )

        first_pose_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        first_pose
                        -
                        HW_CONTACT_POSE
                    )
                    **
                    2
                )
            )
        )

        first_encoder = float(
            first_row[
                "encoder_tick"
            ]
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
            f"{first_encoder:.3f}"
        )

        print(
            f"  hardware encoder   = "
            f"{HW_CONTACT_ENCODER:.3f}"
        )

        print(
            f"  encoder error      = "
            f"{first_encoder - HW_CONTACT_ENCODER:+.3f}"
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

    # =============================================================
    # STEP 5 — TRUSTED FINAL POSE
    # =============================================================

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

    sim_final = np.array(
        [
            target[
                "mcp_deg"
            ],
            target[
                "pip_deg"
            ],
            target[
                "dip_deg"
            ],
        ],
        dtype=float,
    )

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

    # =============================================================
    # DECISION
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G28A DECISION"
    )
    print("=" * 150)

    print(
        "Full-range transmission geometry = PASS"
    )

    print(
        f"H1 pre-contact RMSE              = "
        f"{precontact_rmse:.3f} deg"
    )

    print(
        f"middle episodes                 = "
        f"{len(mid_starts)}"
    )

    print(
        f"tip episodes                    = "
        f"{len(tip_starts)}"
    )

    if np.isfinite(
        detached_gap
    ):

        print(
            f"middle detached gap             = "
            f"{detached_gap:.3f} s"
        )

    print(
        f"final H1 RMSE                   = "
        f"{final_rmse:.3f} deg"
    )

    print()

    if (
        precontact_rmse <= 3.0
        and
        len(mid_starts) == 1
    ):

        print(
            "RESULT: CAD-EQUIVALENT TENDON IS A STRONG "
            "NEW ARCHITECTURE BASELINE."
        )

        print(
            "Freeze transmission architecture."
        )

    else:

        print(
            "RESULT: TRANSMISSION GEOMETRY IS NOW PHYSICALLY "
            "VALID FULL-RANGE, BUT LOW-ANGLE H1 MECHANICS "
            "STILL REQUIRE IDENTIFICATION."
        )

        print()

        print(
            "This is the desired outcome if bounce remains:"
        )

        print(
            "we can now identify pretension/passive joint mechanics "
            "without fitting against a tendon whose moment arms "
            "reverse at large flexion."
        )

    # =============================================================
    # SAVE
    # =============================================================

    summary_csv = (
        out
        /
        "g28a_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mcp_coef_m_rad",
                "pip_coef_m_rad",
                "dip_coef_m_rad",
                "tendon_stiffness",
                "old_open_extension_mm",
                "precontact_rmse_deg",
                "first_contact_encoder",
                "first_contact_pose_rmse_deg",
                "middle_episodes",
                "tip_episodes",
                "middle_detached_gap_s",
                "final_rmse_deg",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "mcp_coef_m_rad":
                MCP_COEF,

            "pip_coef_m_rad":
                PIP_COEF,

            "dip_coef_m_rad":
                DIP_COEF,

            "tendon_stiffness":
                info[
                    "stiffness"
                ],

            "old_open_extension_mm":
                1000.0
                *
                info[
                    "old_extension"
                ],

            "precontact_rmse_deg":
                precontact_rmse,

            "first_contact_encoder":
                first_encoder,

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
        "g28a_contact_trajectory.npz",
        times=np.asarray(
            times
        ),
        qpos=np.asarray(
            qpos
        ),
    )

    print()
    print(
        f"Candidate XML:\n  "
        f"{candidate_xml}"
    )

    print()
    print(
        f"Moment-arm audit:\n  "
        f"{static_csv}"
    )

    print()
    print(
        f"Summary:\n  "
        f"{summary_csv}"
    )

    print()
    print(
        f"All outputs:\n  "
        f"{out}"
    )

    # =============================================================
    # GUI
    # =============================================================

    if not args.no_gui:

        print()
        print("=" * 150)
        print(
            "G28A GUI — CAD-EQUIVALENT CONTINUOUS TENDON"
        )
        print("=" * 150)

        print(
            "Check:"
        )

        print(
            "  1. MCP-first approach"
        )

        print(
            "  2. first middle-contact pose"
        )

        print(
            "  3. whether middle stays or rebounds"
        )

        print(
            "  4. PIP wrapping after middle contact"
        )

        print(
            "  5. distal contact / DIP motion"
        )

        g7.exact_gui_replay(
            candidate_xml,
            qpos,
            times,
            "G28A CAD-equivalent constant-moment-arm tendon",
        )


if __name__ == "__main__":
    main()
