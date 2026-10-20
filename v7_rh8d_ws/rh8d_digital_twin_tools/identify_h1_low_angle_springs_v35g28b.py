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
# G28B — LOW-ANGLE PHYSICAL SPRING IDENTIFICATION
#
# Architecture is FROZEN from G28A:
#
#   one continuous equivalent tendon
#   CAD-derived moment arms
#   K = existing identified 2100 N/m
#
# Identify ONLY conservative joint spring parameters from externally
# measured H1 approach configurations.
#
# Passive resistance:
#
#       R(q) = preload + k*q
#
# with:
#
#       preload >= 0
#       k       >= 0
#
# Equivalent MuJoCo spring:
#
#       tau = -k * (q - qref)
#
#       qref = -preload/k
#
# Existing dry friction is removed for this experiment so spring
# mechanics are tested independently.
#
# NO:
#   contact tuning
#   tendon K tuning
#   friction hierarchy
#   contact-conditioned switching
#   artificial actuators
# =====================================================================


SCALE = 5.70e-6

TENDON_NAME = "idx_flex_spatial"
SPOOL_NAME = "index_motor_spool"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

JOINT_NAMES = {
    "mcp": MCP_NAME,
    "pip": PIP_NAME,
    "dip": DIP_NAME,
}


# H1 measured approach trajectory.
#
# First two are strictly pre-contact.
# Third is the earliest visible middle contact and is treated as
# zero-load contact onset, NOT a loaded contact state.
#
H1_ID_POINTS = [
    # encoder, MCP, PIP, DIP
    (416.1,   10.620, 3.994, 1.911),
    (611.7,   18.094, 4.526, 3.724),
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


def find_joint_xml(root, name):

    for joint in root.iter("joint"):

        if joint.get("name") == name:
            return joint

    raise RuntimeError(
        f"Joint not found: {name}"
    )


def find_geom_xml(root, name):

    for geom in root.iter("geom"):

        if geom.get("name") == name:
            return geom

    raise RuntimeError(
        f"Geom not found: {name}"
    )


def make_identification_xml(
    source,
    destination,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    # Remove existing finger springs/friction so inverse torque
    # calculation contains tendon + gravity only.
    for name in JOINT_NAMES.values():

        joint = find_joint_xml(
            root,
            name,
        )

        joint.set(
            "stiffness",
            "0",
        )

        joint.set(
            "springref",
            "0",
        )

        joint.set(
            "frictionloss",
            "0",
        )

    # Disable bottle for identification.
    fixture = find_geom_xml(
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


def make_candidate_xml(
    source,
    destination,
    fitted,
    free_space=False,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    for key, name in JOINT_NAMES.items():

        joint = find_joint_xml(
            root,
            name,
        )

        k = float(
            fitted[
                key
            ][
                "k_Nm_rad"
            ]
        )

        qref = float(
            fitted[
                key
            ][
                "qref_rad"
            ]
        )

        joint.set(
            "stiffness",
            f"{k:.12g}",
        )

        joint.set(
            "springref",
            f"{qref:.12g}",
        )

        # Test conservative spring model independently.
        joint.set(
            "frictionloss",
            "0",
        )

    if free_space:

        fixture = find_geom_xml(
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


def tendon_jacobian_row(
    model,
    data,
    tendon_id,
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
            tendon_id,
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

        return (
            raw.reshape(
                model.ntendon,
                model.nv,
            )[
                tendon_id,
                :
            ].copy()
        )

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
            tendon_id
        ]
    )

    adr = int(
        rowadr[
            tendon_id
        ]
    )

    for k in range(nnz):

        i = (
            adr
            +
            k
        )

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


def fit_nonnegative_preload_spring(
    q,
    resistance,
):

    q = np.asarray(
        q,
        dtype=float,
    )

    resistance = np.asarray(
        resistance,
        dtype=float,
    )

    # Model:
    #
    # resistance = preload + k*q
    #
    # preload >= 0
    # k       >= 0
    #
    X = np.column_stack(
        [
            np.ones_like(
                q
            ),
            q,
        ]
    )

    candidates = []

    # Unconstrained solution.
    beta, *_ = np.linalg.lstsq(
        X,
        resistance,
        rcond=None,
    )

    if np.all(
        beta >= 0.0
    ):

        candidates.append(
            beta
        )

    # preload-only boundary.
    candidates.append(
        np.array(
            [
                max(
                    float(
                        np.mean(
                            resistance
                        )
                    ),
                    0.0,
                ),
                0.0,
            ]
        )
    )

    # stiffness-only boundary.
    denom = float(
        np.dot(
            q,
            q,
        )
    )

    if denom > 1e-15:

        k_only = max(
            float(
                np.dot(
                    q,
                    resistance,
                )
                /
                denom
            ),
            0.0,
        )

    else:

        k_only = 0.0

    candidates.append(
        np.array(
            [
                0.0,
                k_only,
            ]
        )
    )

    # Zero.
    candidates.append(
        np.zeros(
            2
        )
    )

    beta = min(
        candidates,
        key=lambda b:
            float(
                np.sum(
                    (
                        X @ b
                        -
                        resistance
                    )
                    **
                    2
                )
            ),
    )

    preload = float(
        beta[
            0
        ]
    )

    k = float(
        beta[
            1
        ]
    )

    prediction = (
        X
        @
        beta
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    prediction
                    -
                    resistance
                )
                **
                2
            )
        )
    )

    if k > 1e-12:

        qref = (
            -
            preload
            /
            k
        )

    else:

        qref = 0.0

    return {
        "preload_Nm":
            preload,

        "k_Nm_rad":
            k,

        "qref_rad":
            qref,

        "qref_deg":
            math.degrees(
                qref
            ),

        "rmse_Nm":
            rmse,
    }


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
        help=(
            "G28A equivalent CAD tendon XML"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g28b_h1_low_angle_springs"
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

    id_xml = (
        out
        /
        "g28b_identification_model.xml"
    )

    contact_xml = (
        out
        /
        "g28b_identified_springs_contact.xml"
    )

    free_xml = (
        out
        /
        "g28b_identified_springs_free.xml"
    )

    make_identification_xml(
        source_xml,
        id_xml,
    )

    # =============================================================
    # STEP 1 — recover arm/base state + open spool reference
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G28B STEP 1 — PREPARE H1 STATIC IDENTIFICATION STATES"
    )
    print("=" * 150)

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    g7.b2.v35.METERS_PER_TICK = (
        SCALE
    )

    try:

        (
            replay_rows,
            replay_qpos,
            replay_times,
            replay_summary,
        ) = g12.run_audit(
            id_xml,
            bag,
            replay,
        )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

    encoder = np.asarray(
        [
            row[
                "encoder_tick"
            ]
            for row in replay_rows
        ],
        dtype=float,
    )

    model = mujoco.MjModel.from_xml_path(
        str(
            id_xml
        )
    )

    data = mujoco.MjData(
        model
    )

    refs = g12.get_joint_refs(
        model
    )

    spool_jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_NAME,
    )

    tendon_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    if spool_jid < 0:
        raise RuntimeError(
            "Index spool joint not found."
        )

    if tendon_id < 0:
        raise RuntimeError(
            "Index tendon not found."
        )

    spool_qadr = int(
        model.jnt_qposadr[
            spool_jid
        ]
    )

    open_idx = nearest_index(
        encoder,
        100.0,
    )

    open_encoder = float(
        encoder[
            open_idx
        ]
    )

    open_spool = float(
        replay_qpos[
            open_idx
        ][
            spool_qadr
        ]
    )

    tendon_k = float(
        model.tendon_stiffness[
            tendon_id
        ]
    )

    spring_reference = (
        g12.tendon_spring_reference(
            model,
            tendon_id,
        )
    )

    print(
        f"open encoder    = "
        f"{open_encoder:.3f}"
    )

    print(
        f"open spool qpos = "
        f"{open_spool:+.9f} m"
    )

    print(
        f"tendon K        = "
        f"{tendon_k:.3f} N/m"
    )

    print(
        "No tendon parameter will be changed."
    )

    # =============================================================
    # STEP 2 — inverse spring resistance at measured H1 poses
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G28B STEP 2 — INVERSE PASSIVE RESISTANCE"
    )
    print("=" * 150)

    samples = []

    print()
    print(
        "enc      MCP     PIP     DIP      "
        "Ften      R_MCP      R_PIP      R_DIP"
    )

    print("-" * 150)

    for (
        tick,
        mcp_deg,
        pip_deg,
        dip_deg,
    ) in H1_ID_POINTS:

        i = nearest_index(
            encoder,
            tick,
        )

        # Preserve actual H1 arm/base configuration.
        data.qpos[:] = np.asarray(
            replay_qpos[
                i
            ],
            dtype=float,
        )

        data.qvel[:] = 0.0

        # Use identified effective encoder displacement directly,
        # not model-dependent spool following.
        data.qpos[
            spool_qadr
        ] = (
            open_spool
            +
            SCALE
            *
            (
                tick
                -
                open_encoder
            )
        )

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
            tendon_id,
        )

        extension = (
            g12.signed_extension(
                float(
                    data.ten_length[
                        tendon_id
                    ]
                ),
                spring_reference,
            )
        )

        force = (
            tendon_k
            *
            extension
        )

        sample = {
            "encoder_tick":
                tick,

            "mcp_deg":
                mcp_deg,

            "pip_deg":
                pip_deg,

            "dip_deg":
                dip_deg,

            "extension_mm":
                1000.0
                *
                extension,

            "tendon_force_N":
                force,
        }

        for joint in [
            "mcp",
            "pip",
            "dip",
        ]:

            dadr = refs[
                joint
            ][
                "dadr"
            ]

            # qfrc from positive tendon tension:
            #
            #     tau = -J^T F
            #
            tendon_tau = (
                -
                float(
                    J[
                        dadr
                    ]
                )
                *
                force
            )

            bias = float(
                data.qfrc_bias[
                    dadr
                ]
            )

            # Static equilibrium:
            #
            #   bias = tendon_tau - R
            #
            # therefore:
            #
            #   R = tendon_tau - bias
            #
            resistance = (
                tendon_tau
                -
                bias
            )

            sample[
                f"{joint}_tendon_tau_Nm"
            ] = tendon_tau

            sample[
                f"{joint}_bias_Nm"
            ] = bias

            sample[
                f"{joint}_resistance_Nm"
            ] = resistance

        samples.append(
            sample
        )

        print(
            f"{tick:7.1f} "
            f"{mcp_deg:7.3f} "
            f"{pip_deg:7.3f} "
            f"{dip_deg:7.3f} "
            f"{force:+8.4f} "
            f"{sample['mcp_resistance_Nm']:+10.6f} "
            f"{sample['pip_resistance_Nm']:+10.6f} "
            f"{sample['dip_resistance_Nm']:+10.6f}"
        )

    inverse_csv = (
        out
        /
        "g28b_inverse_passive_resistance.csv"
    )

    with inverse_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                samples[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            samples
        )

    # =============================================================
    # STEP 3 — fit physically constrained preload + stiffness
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G28B STEP 3 — IDENTIFIED TRANSVERSE SPRINGS"
    )
    print("=" * 150)

    fitted = {}

    for joint in [
        "mcp",
        "pip",
        "dip",
    ]:

        q = np.radians(
            np.asarray(
                [
                    sample[
                        f"{joint}_deg"
                    ]
                    for sample in samples
                ],
                dtype=float,
            )
        )

        resistance = np.asarray(
            [
                sample[
                    f"{joint}_resistance_Nm"
                ]
                for sample in samples
            ],
            dtype=float,
        )

        # Negative required resistance would contradict this
        # simple extension-return spring model.
        if np.any(
            resistance < -0.002
        ):

            raise RuntimeError(
                f"{joint.upper()} requires substantial negative "
                "resistance. Linear return-spring model invalid."
            )

        resistance = np.maximum(
            resistance,
            0.0,
        )

        fit = (
            fit_nonnegative_preload_spring(
                q,
                resistance,
            )
        )

        fitted[
            joint
        ] = fit

        print()
        print(
            joint.upper()
        )

        print(
            f"  preload resistance = "
            f"{fit['preload_Nm']:.6f} Nm"
        )

        print(
            f"  stiffness          = "
            f"{fit['k_Nm_rad']:.6f} Nm/rad"
        )

        print(
            f"  springref          = "
            f"{fit['qref_deg']:.3f} deg"
        )

        print(
            f"  torque-fit RMSE    = "
            f"{fit['rmse_Nm']:.6f} Nm"
        )

    # Broad sanity check only — not tuning bounds.
    for joint, fit in fitted.items():

        if (
            fit[
                "k_Nm_rad"
            ]
            >
            1.0
        ):

            raise RuntimeError(
                f"{joint.upper()} identified stiffness "
                "exceeds 1 Nm/rad. Stop and inspect data."
            )

        if (
            abs(
                fit[
                    "qref_deg"
                ]
            )
            >
            45.0
        ):

            raise RuntimeError(
                f"{joint.upper()} springref is implausibly large."
            )

    # =============================================================
    # STEP 4 — build exactly ONE candidate
    # =============================================================

    make_candidate_xml(
        source_xml,
        contact_xml,
        fitted,
        free_space=False,
    )

    make_candidate_xml(
        source_xml,
        free_xml,
        fitted,
        free_space=True,
    )

    mujoco.MjModel.from_xml_path(
        str(
            contact_xml
        )
    )

    mujoco.MjModel.from_xml_path(
        str(
            free_xml
        )
    )

    # =============================================================
    # STEP 5 — dynamic pre-contact validation
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G28B STEP 4 — DYNAMIC H1 APPROACH VALIDATION"
    )
    print("=" * 150)

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

    trajectory_errors = []

    print()
    print(
        "encoder       hardware MCP/PIP/DIP       "
        "simulation MCP/PIP/DIP       error"
    )

    print("-" * 150)

    for (
        tick,
        hm,
        hp,
        hd,
    ) in H1_ID_POINTS:

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
            ]
        )

        hw = np.array(
            [
                hm,
                hp,
                hd,
            ]
        )

        e = (
            sim
            -
            hw
        )

        trajectory_errors.extend(
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
                    trajectory_errors
                )
                **
                2
            )
        )
    )

    print()
    print(
        f"dynamic approach RMSE = "
        f"{precontact_rmse:.3f} deg"
    )

    # =============================================================
    # STEP 6 — ONE CONTACT RUN
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G28B STEP 5 — ONE H1 CONTACT RUN"
    )
    print("=" * 150)

    (
        rows,
        qpos,
        times,
        summary,
    ) = g12.run_audit(
        contact_xml,
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

        raise RuntimeError(
            "No middle contact."
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
        ]
    )

    contact_error = (
        first_pose
        -
        HW_CONTACT_POSE
    )

    contact_rmse = float(
        np.sqrt(
            np.mean(
                contact_error
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
        f"{contact_rmse:.3f} deg"
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
    # Trusted final H1
    # =============================================================

    row_times = np.asarray(
        [
            row[
                "time_s"
            ]
            for row in rows
        ]
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
        ]
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
    # Decision
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G28B DECISION"
    )
    print("=" * 150)

    print(
        f"approach RMSE          = "
        f"{precontact_rmse:.3f} deg"
    )

    print(
        f"contact pose RMSE      = "
        f"{contact_rmse:.3f} deg"
    )

    print(
        f"contact encoder error  = "
        f"{first_row['encoder_tick'] - HW_CONTACT_ENCODER:+.1f} ticks"
    )

    print(
        f"middle episodes        = "
        f"{len(mid_starts)}"
    )

    print(
        f"detached gap           = "
        f"{detached_gap:.3f} s"
    )

    print(
        f"final RMSE             = "
        f"{final_rmse:.3f} deg"
    )

    print()

    if (
        precontact_rmse <= 3.0
        and
        contact_rmse <= 4.0
        and
        len(
            mid_starts
        ) == 1
    ):

        print(
            "RESULT: LOW-ANGLE CONSERVATIVE SPRING MODEL PASSES H1."
        )

        print(
            "Next: test the same parameters against free-space "
            "closing data and then held-out H2 WITHOUT RETUNING."
        )

    elif (
        precontact_rmse <= 3.0
        and
        len(
            mid_starts
        ) > 1
    ):

        print(
            "RESULT: PRE-CONTACT MECHANICS ARE FIXED, "
            "BUT CONTACT BOUNCE REMAINS."
        )

        print()

        print(
            "Freeze these joint springs."
        )

        print(
            "Next variable class: tendon load/compliance only."
        )

    else:

        print(
            "RESULT: A MEMORYLESS LINEAR TRANSVERSE-SPRING MODEL "
            "DOES NOT REPRODUCE THE H1 APPROACH DYNAMICALLY."
        )

        print()

        print(
            "Do NOT manually tune these springs."
        )

        print(
            "Next model class must include measured "
            "friction/hysteresis or nonlinear passive mechanics."
        )

    # =============================================================
    # Save
    # =============================================================

    summary_csv = (
        out
        /
        "g28b_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mcp_preload_Nm",
                "mcp_k_Nm_rad",
                "mcp_qref_deg",
                "pip_preload_Nm",
                "pip_k_Nm_rad",
                "pip_qref_deg",
                "dip_preload_Nm",
                "dip_k_Nm_rad",
                "dip_qref_deg",
                "approach_rmse_deg",
                "first_contact_encoder",
                "first_contact_pose_rmse_deg",
                "middle_episodes",
                "tip_episodes",
                "middle_gap_s",
                "final_rmse_deg",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "mcp_preload_Nm":
                fitted[
                    "mcp"
                ][
                    "preload_Nm"
                ],

            "mcp_k_Nm_rad":
                fitted[
                    "mcp"
                ][
                    "k_Nm_rad"
                ],

            "mcp_qref_deg":
                fitted[
                    "mcp"
                ][
                    "qref_deg"
                ],

            "pip_preload_Nm":
                fitted[
                    "pip"
                ][
                    "preload_Nm"
                ],

            "pip_k_Nm_rad":
                fitted[
                    "pip"
                ][
                    "k_Nm_rad"
                ],

            "pip_qref_deg":
                fitted[
                    "pip"
                ][
                    "qref_deg"
                ],

            "dip_preload_Nm":
                fitted[
                    "dip"
                ][
                    "preload_Nm"
                ],

            "dip_k_Nm_rad":
                fitted[
                    "dip"
                ][
                    "k_Nm_rad"
                ],

            "dip_qref_deg":
                fitted[
                    "dip"
                ][
                    "qref_deg"
                ],

            "approach_rmse_deg":
                precontact_rmse,

            "first_contact_encoder":
                first_row[
                    "encoder_tick"
                ],

            "first_contact_pose_rmse_deg":
                contact_rmse,

            "middle_episodes":
                len(
                    mid_starts
                ),

            "tip_episodes":
                len(
                    tip_starts
                ),

            "middle_gap_s":
                detached_gap,

            "final_rmse_deg":
                final_rmse,
        })

    np.savez_compressed(
        out
        /
        "g28b_contact_trajectory.npz",
        times=np.asarray(
            times
        ),
        qpos=np.asarray(
            qpos
        ),
    )

    print()
    print(
        f"Inverse resistance:\n  "
        f"{inverse_csv}"
    )

    print()
    print(
        f"Candidate XML:\n  "
        f"{contact_xml}"
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

    if not args.no_gui:

        print()
        print("=" * 150)
        print(
            "G28B GUI — IDENTIFIED LOW-ANGLE SPRINGS"
        )
        print("=" * 150)

        print(
            "Required visual sequence:"
        )

        print(
            "  MCP dominates approach"
        )

        print(
            "  -> middle contacts with PIP/DIP still modest"
        )

        print(
            "  -> middle stays constrained"
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

        g7.exact_gui_replay(
            contact_xml,
            qpos,
            times,
            "G28B identified low-angle transverse springs",
        )


if __name__ == "__main__":
    main()
