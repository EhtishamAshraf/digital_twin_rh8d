#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import mujoco
import numpy as np

import scan_index_contact_friction_v35g7 as g7
import audit_index_tendon_transmission_v35g12 as g12


# =====================================================================
# G26A — FORCE BALANCE AT THE MEASURED HARDWARE FIRST-CONTACT POSE
#
# NO MODEL TUNING.
#
# G25A hardware:
#
#   encoder = 673.573 ticks
#
#   MCP = 18.543 deg
#   PIP =  6.161 deg
#   DIP =  5.276 deg
#
# G25B geometrically calibrated the bottle so the CAD middle
# phalanx is touching the bottle at exactly this measured pose.
#
# G26A asks:
#
#   At that REAL-like first-contact geometry:
#
#   1. What are the tendon moment arms?
#   2. What tendon force currently exists?
#   3. What generalized torque does a representative contact
#      force produce at MCP/PIP/DIP?
#   4. What tendon force is required to balance that load?
#   5. What equivalent tendon stiffness would produce it at the
#      measured encoder/configuration?
#
# This is an inverse-mechanics audit, NOT a simulation fit.
#
# We use 0.263 N as the reference contact normal force because
# G22C measured approximately 0.263 N at the peak of the first
# bad middle-contact episode.
#
# Since contact torque scales linearly with normal force, we also
# report torque per 1 N.
# =====================================================================


HW_ENCODER = 673.573

HW_MCP_DEG = 18.543
HW_PIP_DEG = 6.161
HW_DIP_DEG = 5.276


REFERENCE_NORMAL_FORCE_N = 0.262843


TENDON_NAME = "idx_flex_spatial"


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
                values
                -
                target
            )
        )
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

    # Dense representation.
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

        return (
            raw[
                tendon_id,
                :
            ].copy()
        )

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

    # Sparse representation fallback.
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

    for k in range(
        nnz
    ):

        idx = (
            adr
            +
            k
        )

        col = int(
            colind[
                idx
            ]
        )

        J[
            col
        ] = float(
            raw[
                idx
            ]
        )

    return J


def set_hardware_pose(
    model,
    data,
    q_reference,
):

    data.qpos[:] = np.asarray(
        q_reference,
        dtype=float,
    )

    data.qvel[:] = 0.0

    refs = g12.get_joint_refs(
        model
    )

    data.qpos[
        refs[
            "mcp"
        ][
            "qadr"
        ]
    ] = math.radians(
        HW_MCP_DEG
    )

    data.qpos[
        refs[
            "pip"
        ][
            "qadr"
        ]
    ] = math.radians(
        HW_PIP_DEG
    )

    data.qpos[
        refs[
            "dip"
        ][
            "qadr"
        ]
    ] = math.radians(
        HW_DIP_DEG
    )

    mujoco.mj_forward(
        model,
        data,
    )

    return refs


def safe_unit(
    vector,
):

    vector = np.asarray(
        vector,
        dtype=float,
    )

    norm = float(
        np.linalg.norm(
            vector
        )
    )

    if norm < 1e-12:

        raise RuntimeError(
            "Cannot normalize near-zero vector."
        )

    return (
        vector
        /
        norm
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
            "G25B hardware-contact calibrated XML"
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
            "v35g26a_hardware_contact_force_balance"
        ),
    )

    args = parser.parse_args()

    bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )

    xml = (
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

    if not xml.exists():

        raise FileNotFoundError(
            xml
        )

    # --------------------------------------------------------------
    # Use the same baseline replay infrastructure to recover all
    # non-finger qpos values at the measured encoder.
    #
    # Then overwrite only MCP/PIP/DIP with hardware measurements.
    # --------------------------------------------------------------

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    print()
    print("=" * 150)
    print(
        "G26A — HARDWARE FIRST-CONTACT FORCE-BALANCE AUDIT"
    )
    print("=" * 150)

    print()
    print(
        "No parameters are being changed."
    )

    print(
        f"hardware encoder = "
        f"{HW_ENCODER:.3f} ticks"
    )

    print(
        "hardware pose    = "
        f"{HW_MCP_DEG:.3f} / "
        f"{HW_PIP_DEG:.3f} / "
        f"{HW_DIP_DEG:.3f} deg"
    )

    print()

    (
        replay_rows,
        replay_qpos,
        replay_times,
        replay_summary,
    ) = g12.run_audit(
        xml,
        bag,
        replay,
    )

    encoders = np.asarray(
        [
            row[
                "encoder_tick"
            ]
            for row in replay_rows
        ],
        dtype=float,
    )

    idx = nearest_index(
        encoders,
        HW_ENCODER,
    )

    reference_row = replay_rows[
        idx
    ]

    q_reference = np.asarray(
        replay_qpos[
            idx
        ],
        dtype=float,
    ).copy()

    print(
        "Baseline simulation at same encoder:"
    )

    print(
        f"  actual nearest encoder = "
        f"{reference_row['encoder_tick']:.3f}"
    )

    print(
        "  MCP/PIP/DIP = "
        f"{reference_row['mcp_deg']:.3f} / "
        f"{reference_row['pip_deg']:.3f} / "
        f"{reference_row['dip_deg']:.3f}"
    )

    # --------------------------------------------------------------
    # Static hardware configuration.
    # --------------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(
            xml
        )
    )

    data = mujoco.MjData(
        model
    )

    refs = set_hardware_pose(
        model,
        data,
        q_reference,
    )

    tendon_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    if tendon_id < 0:

        raise RuntimeError(
            f"Tendon not found: {TENDON_NAME}"
        )

    middle_gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        g12.CONTACT_GEOMS[
            "mid"
        ],
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
            "Middle or fixture geom missing."
        )

    middle_bid = int(
        model.geom_bodyid[
            middle_gid
        ]
    )

    # --------------------------------------------------------------
    # Exact closest points between middle and bottle at the
    # G25B-calibrated hardware pose.
    # --------------------------------------------------------------

    fromto = np.zeros(
        6,
        dtype=float,
    )

    geom_distance = float(
        mujoco.mj_geomDistance(
            model,
            data,
            middle_gid,
            fixture_gid,
            0.05,
            fromto,
        )
    )

    middle_point = np.asarray(
        fromto[
            0:3
        ],
        dtype=float,
    )

    fixture_point = np.asarray(
        fromto[
            3:6
        ],
        dtype=float,
    )

    contact_point = (
        0.5
        *
        (
            middle_point
            +
            fixture_point
        )
    )

    fixture_center = np.asarray(
        data.geom_xpos[
            fixture_gid
        ],
        dtype=float,
    )

    fixture_R = np.asarray(
        data.geom_xmat[
            fixture_gid
        ],
        dtype=float,
    ).reshape(
        3,
        3,
    )

    cylinder_axis = safe_unit(
        fixture_R[
            :,
            2
        ]
    )

    radial = (
        fixture_point
        -
        fixture_center
    )

    # Remove component along cylinder axis.
    radial = (
        radial
        -
        cylinder_axis
        *
        float(
            np.dot(
                radial,
                cylinder_axis,
            )
        )
    )

    outward_normal = safe_unit(
        radial
    )

    print()
    print("=" * 150)
    print(
        "CONTACT GEOMETRY AT MEASURED HARDWARE POSE"
    )
    print("=" * 150)

    print(
        f"middle/fixture distance = "
        f"{1e6 * geom_distance:+.3f} um"
    )

    print(
        "contact point xyz [m] = "
        f"{contact_point[0]:+.6f}, "
        f"{contact_point[1]:+.6f}, "
        f"{contact_point[2]:+.6f}"
    )

    print(
        "outward bottle normal = "
        f"{outward_normal[0]:+.6f}, "
        f"{outward_normal[1]:+.6f}, "
        f"{outward_normal[2]:+.6f}"
    )

    # --------------------------------------------------------------
    # Contact Jacobian.
    # --------------------------------------------------------------

    jacp = np.zeros(
        (
            3,
            model.nv,
        ),
        dtype=float,
    )

    jacr = np.zeros(
        (
            3,
            model.nv,
        ),
        dtype=float,
    )

    mujoco.mj_jac(
        model,
        data,
        jacp,
        jacr,
        contact_point,
        middle_bid,
    )

    # Contact force on finger points away from bottle.
    force_per_1N = (
        outward_normal
        *
        1.0
    )

    force_reference = (
        outward_normal
        *
        REFERENCE_NORMAL_FORCE_N
    )

    tau_contact_per_1N = (
        jacp.T
        @
        force_per_1N
    )

    tau_contact_reference = (
        jacp.T
        @
        force_reference
    )

    # --------------------------------------------------------------
    # Tendon Jacobian / moment arms.
    # --------------------------------------------------------------

    tendon_J = tendon_jacobian_row(
        model,
        data,
        tendon_id,
    )

    tendon_length = float(
        data.ten_length[
            tendon_id
        ]
    )

    tendon_velocity = float(
        data.ten_velocity[
            tendon_id
        ]
    )

    spring_reference = (
        g12.tendon_spring_reference(
            model,
            tendon_id,
        )
    )

    extension = (
        g12.signed_extension(
            tendon_length,
            spring_reference,
        )
    )

    # Read actual K/D from XML model if possible.
    tendon_k = float(
        model.tendon_stiffness[
            tendon_id
        ]
    )

    tendon_damping = float(
        model.tendon_damping[
            tendon_id
        ]
    )

    current_spring_force = (
        tendon_k
        *
        extension
    )

    current_damping_force = (
        tendon_damping
        *
        tendon_velocity
    )

    current_total_force_est = (
        current_spring_force
        +
        current_damping_force
    )

    print()
    print("=" * 150)
    print(
        "TENDON STATE AT MEASURED HARDWARE POSE"
    )
    print("=" * 150)

    print(
        f"tendon length        = "
        f"{1000.0 * tendon_length:.6f} mm"
    )

    print(
        f"spring reference     = "
        f"{1000.0 * spring_reference:.6f} mm"
    )

    print(
        f"signed extension     = "
        f"{1000.0 * extension:+.6f} mm"
    )

    print(
        f"model stiffness      = "
        f"{tendon_k:.3f} N/m"
    )

    print(
        f"estimated spring F   = "
        f"{current_spring_force:+.6f} N"
    )

    print(
        f"estimated damping F  = "
        f"{current_damping_force:+.6f} N"
    )

    # --------------------------------------------------------------
    # Generalized torque decomposition.
    #
    # qfrc_passive currently contains:
    #
    #   tendon spring + joint spring/damping
    #
    # Estimate tendon contribution explicitly, then subtract it
    # to obtain non-tendon passive residual.
    # --------------------------------------------------------------

    output_rows = []

    print()
    print("=" * 150)
    print(
        "JOINT FORCE BALANCE AT REAL-LIKE FIRST CONTACT"
    )
    print("=" * 150)

    print(
        "joint  momentArm  tendonTau/N   currentTendon   "
        "jointPassive   contact/N   contact@0.263N   bias"
    )

    print("-" * 150)

    required_forces = {}

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

        moment = float(
            tendon_J[
                dadr
            ]
        )

        # Same convention used in G22C.
        tendon_tau_per_N = (
            -
            moment
        )

        current_tendon_tau = (
            tendon_tau_per_N
            *
            current_total_force_est
        )

        passive_total = float(
            data.qfrc_passive[
                dadr
            ]
        )

        passive_without_tendon = (
            passive_total
            -
            current_tendon_tau
        )

        contact_per_N = float(
            tau_contact_per_1N[
                dadr
            ]
        )

        contact_reference = float(
            tau_contact_reference[
                dadr
            ]
        )

        bias = float(
            data.qfrc_bias[
                dadr
            ]
        )

        # Static balance:
        #
        # tendon_perN * F
        # + passive_without_tendon
        # + contact_tau
        # - bias
        # = 0
        #
        # Solve for F.
        numerator = -(
            passive_without_tendon
            +
            contact_reference
            -
            bias
        )

        if abs(
            tendon_tau_per_N
        ) > 1e-10:

            required_force = (
                numerator
                /
                tendon_tau_per_N
            )

        else:

            required_force = np.nan

        required_forces[
            joint
        ] = required_force

        if (
            np.isfinite(
                required_force
            )
            and
            extension > 1e-9
        ):

            equivalent_k = (
                required_force
                /
                extension
            )

        else:

            equivalent_k = np.nan

        print(
            f"{joint.upper():4s} "
            f"{1000.0 * abs(moment):9.4f}mm "
            f"{tendon_tau_per_N:+11.6f} "
            f"{current_tendon_tau:+13.6f} "
            f"{passive_without_tendon:+12.6f} "
            f"{contact_per_N:+11.6f} "
            f"{contact_reference:+15.6f} "
            f"{bias:+10.6f}"
        )

        output_rows.append({
            "joint":
                joint,

            "moment_arm_mm":
                1000.0
                *
                abs(
                    moment
                ),

            "tendon_tau_per_N_Nm":
                tendon_tau_per_N,

            "current_tendon_force_N":
                current_total_force_est,

            "current_tendon_tau_Nm":
                current_tendon_tau,

            "passive_without_tendon_Nm":
                passive_without_tendon,

            "contact_tau_per_1N_Nm":
                contact_per_N,

            "contact_tau_reference_Nm":
                contact_reference,

            "bias_Nm":
                bias,

            "required_tendon_force_N":
                required_force,

            "equivalent_stiffness_N_m":
                equivalent_k,
        })

    print()
    print("=" * 150)
    print(
        "REQUIRED TENDON FORCE FOR ZERO STATIC JOINT ACCELERATION"
    )
    print("=" * 150)

    for joint in [
        "mcp",
        "pip",
        "dip",
    ]:

        force = required_forces[
            joint
        ]

        row = next(
            r
            for r in output_rows
            if r[
                "joint"
            ]
            ==
            joint
        )

        print(
            f"{joint.upper():4s}: "
            f"F_required = "
            f"{force:+.4f} N"
        )

        print(
            f"      equivalent K at this extension = "
            f"{row['equivalent_stiffness_N_m']:+.1f} N/m"
        )

    # --------------------------------------------------------------
    # Which joint determines the minimum common tendon force?
    # --------------------------------------------------------------

    positive_requirements = [
        value
        for value in required_forces.values()
        if (
            np.isfinite(
                value
            )
            and
            value > 0.0
        )
    ]

    if positive_requirements:

        common_required = max(
            positive_requirements
        )

    else:

        common_required = np.nan

    if (
        np.isfinite(
            common_required
        )
        and
        extension > 1e-9
    ):

        common_k = (
            common_required
            /
            extension
        )

    else:

        common_k = np.nan

    print()
    print("=" * 150)
    print(
        "COMMON TENDON REQUIREMENT"
    )
    print("=" * 150)

    print(
        f"current estimated tendon force = "
        f"{current_total_force_est:.4f} N"
    )

    print(
        f"minimum common force to satisfy all positive "
        f"joint requirements = "
        f"{common_required:.4f} N"
    )

    print(
        f"equivalent K at current extension = "
        f"{common_k:.1f} N/m"
    )

    if (
        np.isfinite(
            common_required
        )
        and
        abs(
            current_total_force_est
        ) > 1e-9
    ):

        ratio = (
            common_required
            /
            abs(
                current_total_force_est
            )
        )

    else:

        ratio = np.nan

    print(
        f"force ratio required/current = "
        f"{ratio:.3f} x"
    )

    # --------------------------------------------------------------
    # Also report expected contact torque for the measured H1
    # fingertip-force scale purely as a sensitivity number.
    #
    # We DO NOT assume FTS == middle normal force.
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "CONTACT-TORQUE SENSITIVITY"
    )
    print("=" * 150)

    print(
        "The following is linear scaling only."
    )

    print(
        "We are NOT assuming the fingertip FTS equals "
        "middle-phalanx normal force."
    )

    print()

    for force_N in [
        0.10,
        0.14563,
        0.262843,
        0.50,
        1.00,
    ]:

        print(
            f"Fn={force_N:.5f} N:"
        )

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

            tau = (
                tau_contact_per_1N[
                    dadr
                ]
                *
                force_N
            )

            print(
                f"  {joint.upper():4s} "
                f"{tau:+.6f} Nm"
            )

    # --------------------------------------------------------------
    # Save.
    # --------------------------------------------------------------

    csv_path = (
        out
        /
        "g26a_hardware_contact_force_balance.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                output_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    summary_path = (
        out
        /
        "g26a_summary.txt"
    )

    with summary_path.open(
        "w"
    ) as f:

        f.write(
            "G26A hardware-contact force-balance audit\n"
        )

        f.write(
            f"encoder={HW_ENCODER:.3f}\n"
        )

        f.write(
            "hardware_pose_deg="
            f"{HW_MCP_DEG:.3f},"
            f"{HW_PIP_DEG:.3f},"
            f"{HW_DIP_DEG:.3f}\n"
        )

        f.write(
            f"geom_distance_um="
            f"{1e6 * geom_distance:.6f}\n"
        )

        f.write(
            f"tendon_extension_mm="
            f"{1000.0 * extension:.6f}\n"
        )

        f.write(
            f"current_tendon_force_N="
            f"{current_total_force_est:.6f}\n"
        )

        f.write(
            f"common_required_force_N="
            f"{common_required:.6f}\n"
        )

        f.write(
            f"equivalent_common_K_N_m="
            f"{common_k:.6f}\n"
        )

        f.write(
            f"required_current_ratio="
            f"{ratio:.6f}\n"
        )

    print()
    print(
        f"CSV:\n  {csv_path}"
    )

    print()
    print(
        f"Summary:\n  {summary_path}"
    )

    print()
    print(
        f"All outputs:\n  {out}"
    )

    print()
    print("=" * 150)
    print(
        "G26A STOPPING RULE"
    )
    print("=" * 150)

    print(
        "Do NOT change any model parameter from this script."
    )

    print()

    print(
        "Next candidate will be calculated from:"
    )

    print(
        "  1. measured hardware pre-contact staging,"
    )

    print(
        "  2. this hardware-pose force balance,"
    )

    print(
        "  3. the known continuous spatial tendon topology."
    )


if __name__ == "__main__":
    main()
