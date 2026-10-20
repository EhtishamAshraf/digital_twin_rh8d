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


# ================================================================
# G22C — FIRST MIDDLE-CONTACT DYNAMICS AUTOPSY
#
# NO FITTING.
# NO MODEL PARAMETER CHANGE.
#
# Question:
#
# Why does the simulated middle phalanx:
#
#     touch bottle
#     -> remain briefly
#     -> separate
#     -> leave a large gap
#     -> recontact
#
# G22B already proved the failure remains even with distal
# collision disabled, therefore the cause exists at the
# MIDDLE contact itself.
#
# We inspect:
#
#   q, qdot, qacc
#   tendon force / tendon Jacobian / tendon torques
#   passive generalized forces
#   constraint generalized forces
#   bias generalized forces
#   contact position
#   contact normal
#   normal/tangential force
#   middle-point radial velocity relative to bottle
#   physical gap velocity/acceleration
#
# Positive gap/radial velocity means MOVING AWAY from bottle.
# ================================================================


OUT_NAME = "v35g22c_first_middle_contact_autopsy"


def tendon_jacobian_row(
    model,
    data,
    tid,
):

    raw = np.asarray(
        data.ten_J,
        dtype=float,
    )

    J = np.zeros(
        model.nv,
        dtype=float,
    )

    expected = (
        model.ntendon
        *
        model.nv
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

        J[:] = raw[
            tid,
            :
        ]

        return J

    if (
        raw.ndim == 1
        and
        raw.size == expected
    ):

        J[:] = (
            raw.reshape(
                model.ntendon,
                model.nv,
            )[
                tid,
                :
            ]
        )

        return J

    # Sparse representation.
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

    cols = colind[
        adr:
        adr + nnz
    ]

    vals = raw[
        adr:
        adr + nnz
    ]

    for col, value in zip(
        cols,
        vals,
    ):

        J[
            int(col)
        ] = float(
            value
        )

    return J


def safe_unit(v):

    v = np.asarray(
        v,
        dtype=float,
    )

    n = float(
        np.linalg.norm(
            v
        )
    )

    if n < 1e-12:

        return np.array(
            [
                np.nan,
                np.nan,
                np.nan,
            ],
            dtype=float,
        )

    return (
        v / n
    )


def contact_records(
    model,
    data,
    middle_gid,
    fixture_gid,
):

    records = []

    for ci in range(
        int(
            data.ncon
        )
    ):

        c = data.contact[
            ci
        ]

        g1 = int(
            c.geom1
        )

        g2 = int(
            c.geom2
        )

        if not (
            (
                g1 == middle_gid
                and
                g2 == fixture_gid
            )
            or
            (
                g2 == middle_gid
                and
                g1 == fixture_gid
            )
        ):

            continue

        force6 = np.zeros(
            6,
            dtype=float,
        )

        mujoco.mj_contactForce(
            model,
            data,
            ci,
            force6,
        )

        frame = np.asarray(
            c.frame,
            dtype=float,
        ).reshape(
            3,
            3,
        )

        records.append({
            "index":
                ci,

            "position":
                np.asarray(
                    c.pos,
                    dtype=float,
                ).copy(),

            "normal_raw":
                frame[
                    0,
                    :
                ].copy(),

            "normal_force":
                abs(
                    float(
                        force6[
                            0
                        ]
                    )
                ),

            "tangent_force":
                float(
                    math.hypot(
                        force6[
                            1
                        ],
                        force6[
                            2
                        ],
                    )
                ),

            "dist":
                float(
                    c.dist
                ),
        })

    return records


def weighted_contact(
    records,
):

    if not records:

        return None

    weights = np.asarray(
        [
            max(
                r[
                    "normal_force"
                ],
                1e-9,
            )
            for r in records
        ],
        dtype=float,
    )

    positions = np.asarray(
        [
            r[
                "position"
            ]
            for r in records
        ],
        dtype=float,
    )

    p = np.average(
        positions,
        axis=0,
        weights=weights,
    )

    # Use strongest contact for contact-frame normal.
    strongest = max(
        records,
        key=lambda r:
            r[
                "normal_force"
            ],
    )

    return {
        "position":
            p,

        "normal_raw":
            strongest[
                "normal_raw"
            ],

        "normal_force":
            float(
                sum(
                    r[
                        "normal_force"
                    ]
                    for r in records
                )
            ),

        "tangent_force":
            float(
                sum(
                    r[
                        "tangent_force"
                    ]
                    for r in records
                )
            ),

        "min_dist":
            float(
                min(
                    r[
                        "dist"
                    ]
                    for r in records
                )
            ),

        "count":
            len(
                records
            ),
    }


def nearest_index(
    times,
    target,
):

    times = np.asarray(
        times,
        dtype=float,
    )

    return int(
        np.argmin(
            np.abs(
                times
                -
                target
            )
        )
    )


def save_csv(
    path,
    rows,
):

    if not rows:
        return

    fieldnames = []

    seen = set()

    for row in rows:

        for key in row.keys():

            if key.startswith(
                "_"
            ):
                continue

            if key not in seen:

                seen.add(
                    key
                )

                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:

            writer.writerow({
                key:
                    row.get(
                        key,
                        np.nan,
                    )
                for key in fieldnames
            })


def print_joint_dynamics(
    label,
    row,
):

    print()
    print(
        label
    )

    print(
        "joint   tendon_est    passive_total   "
        "passive_resid   constraint      bias      actuator      qacc"
    )

    print(
        "-" * 118
    )

    for joint in [
        "mcp",
        "pip",
        "dip",
    ]:

        tendon = float(
            row[
                f"{joint}_tendon_tau_total_est_Nm"
            ]
        )

        passive = float(
            row[
                f"{joint}_qfrc_passive_Nm"
            ]
        )

        residual = (
            passive
            -
            tendon
        )

        constraint = float(
            row[
                f"{joint}_qfrc_constraint_Nm"
            ]
        )

        bias = float(
            row[
                f"{joint}_qfrc_bias_Nm"
            ]
        )

        actuator = float(
            row[
                f"{joint}_qfrc_actuator_Nm"
            ]
        )

        qacc = float(
            row[
                f"{joint}_qacc_deg_s2"
            ]
        )

        print(
            f"{joint.upper():4s} "
            f"{tendon:+12.6f} "
            f"{passive:+14.6f} "
            f"{residual:+14.6f} "
            f"{constraint:+12.6f} "
            f"{bias:+10.6f} "
            f"{actuator:+10.6f} "
            f"{qacc:+11.2f}"
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
            OUT_NAME
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
        out_dir
        /
        Path(__file__).name,
    )

    # ------------------------------------------------------------
    # Exact G12 baseline configuration.
    # ------------------------------------------------------------

    g7.b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    model_xml = (
        g12.make_baseline_model(
            out_dir
        )
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    old_tendon = float(
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

    extra_rows = []

    refs = {
        "initialized":
            False,
    }

    original_mj_step = (
        mujoco.mj_step
    )


    def extra_capture_step(
        model,
        data,
        *step_args,
        **step_kwargs,
    ):

        # Run the REAL MuJoCo integrator.
        result = original_mj_step(
            model,
            data,
            *step_args,
            **step_kwargs,
        )

        local = (
            g7.e2a.replay_locals()
        )

        if local is None:
            return result

        if not refs[
            "initialized"
        ]:

            refs[
                "initialized"
            ] = True

            refs[
                "joints"
            ] = (
                g12.get_joint_refs(
                    model
                )
            )

            refs[
                "tid"
            ] = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_TENDON,
                g12.TENDON_NAME,
            )

            refs[
                "middle_gid"
            ] = g7.e2a.gid(
                model,
                g12.CONTACT_GEOMS[
                    "mid"
                ],
            )

            refs[
                "fixture_gid"
            ] = g7.e2a.gid(
                model,
                g12.FIXTURE_NAME,
            )

            refs[
                "middle_bid"
            ] = int(
                model.geom_bodyid[
                    refs[
                        "middle_gid"
                    ]
                ]
            )

            refs[
                "spring_ref"
            ] = (
                g12.tendon_spring_reference(
                    model,
                    refs[
                        "tid"
                    ],
                )
            )

        tid = refs[
            "tid"
        ]

        J = tendon_jacobian_row(
            model,
            data,
            tid,
        )

        tendon_length = float(
            data.ten_length[
                tid
            ]
        )

        tendon_velocity = float(
            data.ten_velocity[
                tid
            ]
        )

        extension = (
            g12.signed_extension(
                tendon_length,
                refs[
                    "spring_ref"
                ],
            )
        )

        spring_force = (
            g12.TENDON_K
            *
            extension
        )

        damping_force = (
            g12.TENDON_DAMPING
            *
            tendon_velocity
        )

        q = np.asarray(
            g7.e2a.q_deg(
                data,
                g7.e2a.joint_info(
                    model
                ),
            ),
            dtype=float,
        )

        row = {
            "time_s":
                float(
                    local[
                        "time"
                    ]
                ),

            "encoder_tick":
                float(
                    local[
                        "encoder"
                    ]
                ),

            "mcp_deg":
                float(
                    q[
                        0
                    ]
                ),

            "pip_deg":
                float(
                    q[
                        1
                    ]
                ),

            "dip_deg":
                float(
                    q[
                        2
                    ]
                ),

            "tendon_length_m":
                tendon_length,

            "tendon_velocity_m_s":
                tendon_velocity,

            "tendon_extension_mm":
                1000.0
                *
                extension,

            "tendon_spring_force_est_N":
                spring_force,

            "tendon_damping_force_est_N":
                damping_force,
        }

        # --------------------------------------------------------
        # MCP/PIP/DIP dynamics.
        # --------------------------------------------------------

        for joint in [
            "mcp",
            "pip",
            "dip",
        ]:

            ref = refs[
                "joints"
            ][
                joint
            ]

            dadr = ref[
                "dadr"
            ]

            moment = float(
                J[
                    dadr
                ]
            )

            tendon_tau_spring = (
                -
                spring_force
                *
                moment
            )

            tendon_tau_damping = (
                -
                damping_force
                *
                moment
            )

            row[
                f"{joint}_moment_arm_mm"
            ] = (
                1000.0
                *
                abs(
                    moment
                )
            )

            row[
                f"{joint}_qvel_deg_s"
            ] = math.degrees(
                float(
                    data.qvel[
                        dadr
                    ]
                )
            )

            row[
                f"{joint}_qacc_deg_s2"
            ] = math.degrees(
                float(
                    data.qacc[
                        dadr
                    ]
                )
            )

            row[
                f"{joint}_tendon_tau_spring_est_Nm"
            ] = (
                tendon_tau_spring
            )

            row[
                f"{joint}_tendon_tau_damping_est_Nm"
            ] = (
                tendon_tau_damping
            )

            row[
                f"{joint}_tendon_tau_total_est_Nm"
            ] = (
                tendon_tau_spring
                +
                tendon_tau_damping
            )

            row[
                f"{joint}_qfrc_passive_Nm"
            ] = float(
                data.qfrc_passive[
                    dadr
                ]
            )

            row[
                f"{joint}_qfrc_constraint_Nm"
            ] = float(
                data.qfrc_constraint[
                    dadr
                ]
            )

            row[
                f"{joint}_qfrc_bias_Nm"
            ] = float(
                data.qfrc_bias[
                    dadr
                ]
            )

            row[
                f"{joint}_qfrc_actuator_Nm"
            ] = float(
                data.qfrc_actuator[
                    dadr
                ]
            )

            row[
                f"{joint}_qfrc_applied_Nm"
            ] = float(
                data.qfrc_applied[
                    dadr
                ]
            )

        # --------------------------------------------------------
        # Exact middle-bottle contact records.
        # --------------------------------------------------------

        contacts = contact_records(
            model,
            data,
            refs[
                "middle_gid"
            ],
            refs[
                "fixture_gid"
            ],
        )

        c = weighted_contact(
            contacts
        )

        row[
            "middle_contact_count"
        ] = len(
            contacts
        )

        row[
            "middle_contact_active"
        ] = bool(
            contacts
        )

        if c is None:

            for key in [
                "contact_x_m",
                "contact_y_m",
                "contact_z_m",
                "contact_local_mid_x_m",
                "contact_local_mid_y_m",
                "contact_local_mid_z_m",
                "contact_normal_force_N",
                "contact_tangent_force_N",
                "contact_dist_mm",
                "contact_normal_alignment",
                "middle_radial_velocity_m_s",
                "middle_radial_accel_Jqacc_m_s2",
            ]:

                row[
                    key
                ] = np.nan

        else:

            point = np.asarray(
                c[
                    "position"
                ],
                dtype=float,
            )

            fixture_center = np.asarray(
                data.geom_xpos[
                    refs[
                        "fixture_gid"
                    ]
                ],
                dtype=float,
            )

            fixture_R = np.asarray(
                data.geom_xmat[
                    refs[
                        "fixture_gid"
                    ]
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
                point
                -
                fixture_center
            )

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

            outward = safe_unit(
                radial
            )

            contact_normal = safe_unit(
                c[
                    "normal_raw"
                ]
            )

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
                point,
                refs[
                    "middle_bid"
                ],
            )

            point_velocity = (
                jacp
                @
                np.asarray(
                    data.qvel,
                    dtype=float,
                )
            )

            # This omits Jdot*qdot and is therefore diagnostic
            # only. A finite-difference radial acceleration is
            # calculated later from actual point velocity.
            point_accel_Jqacc = (
                jacp
                @
                np.asarray(
                    data.qacc,
                    dtype=float,
                )
            )

            body_pos = np.asarray(
                data.xpos[
                    refs[
                        "middle_bid"
                    ]
                ],
                dtype=float,
            )

            body_R = np.asarray(
                data.xmat[
                    refs[
                        "middle_bid"
                    ]
                ],
                dtype=float,
            ).reshape(
                3,
                3,
            )

            local_point = (
                body_R.T
                @
                (
                    point
                    -
                    body_pos
                )
            )

            row[
                "contact_x_m"
            ] = float(
                point[
                    0
                ]
            )

            row[
                "contact_y_m"
            ] = float(
                point[
                    1
                ]
            )

            row[
                "contact_z_m"
            ] = float(
                point[
                    2
                ]
            )

            row[
                "contact_local_mid_x_m"
            ] = float(
                local_point[
                    0
                ]
            )

            row[
                "contact_local_mid_y_m"
            ] = float(
                local_point[
                    1
                ]
            )

            row[
                "contact_local_mid_z_m"
            ] = float(
                local_point[
                    2
                ]
            )

            row[
                "contact_normal_force_N"
            ] = float(
                c[
                    "normal_force"
                ]
            )

            row[
                "contact_tangent_force_N"
            ] = float(
                c[
                    "tangent_force"
                ]
            )

            row[
                "contact_dist_mm"
            ] = (
                1000.0
                *
                float(
                    c[
                        "min_dist"
                    ]
                )
            )

            row[
                "contact_normal_alignment"
            ] = abs(
                float(
                    np.dot(
                        contact_normal,
                        outward,
                    )
                )
            )

            # IMPORTANT SIGN:
            #
            # positive = point moving radially AWAY from bottle
            # negative = point moving INTO bottle
            row[
                "middle_radial_velocity_m_s"
            ] = float(
                np.dot(
                    outward,
                    point_velocity,
                )
            )

            row[
                "middle_radial_accel_Jqacc_m_s2"
            ] = float(
                np.dot(
                    outward,
                    point_accel_Jqacc,
                )
            )

        extra_rows.append(
            row
        )

        return result


    print()
    print("=" * 150)
    print(
        "G22C — FIRST MIDDLE-CONTACT DYNAMICS AUTOPSY"
    )
    print("=" * 150)

    print()
    print(
        "NO PARAMETERS ARE BEING FIT OR CHANGED."
    )

    print(
        "Known failing H1 baseline only."
    )

    try:

        g7.b2.v35.METERS_PER_TICK = (
            g12.SCALE
        )

        g7.b2.v35.TENDON_K = (
            g12.TENDON_K
        )

        g7.b2.v35.CONTACT_MU = (
            g12.CONTACT_MU
        )

        g7.b2.v35.CONTACT_K = (
            g12.CONTACT_K
        )

        contact_replay.CONTACT_DAMPING = (
            g12.CONTACT_DAMPING
        )

        # --------------------------------------------------------
        # Nested step interception:
        #
        # G12 performs its normal complete audit.
        # Our wrapper observes the SAME physical simulation step
        # and adds contact-point/dynamics diagnostics.
        # --------------------------------------------------------

        mujoco.mj_step = (
            extra_capture_step
        )

        (
            base_rows,
            qpos,
            times,
            replay_summary,
        ) = g12.run_audit(
            model_xml,
            bag,
            contact_replay,
        )

    finally:

        mujoco.mj_step = (
            original_mj_step
        )

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

        g7.b2.v35.TENDON_K = (
            old_tendon
        )

        g7.b2.v35.CONTACT_MU = (
            old_mu
        )

        g7.b2.v35.CONTACT_K = (
            old_contact_k
        )

        contact_replay.CONTACT_DAMPING = (
            old_contact_damping
        )

    # ------------------------------------------------------------
    # Merge G12 rows with G22C diagnostics.
    # ------------------------------------------------------------

    if not extra_rows:
        raise RuntimeError(
            "G22C captured no simulation rows."
        )

    extra_times = np.asarray(
        [
            r[
                "time_s"
            ]
            for r in extra_rows
        ],
        dtype=float,
    )

    rows = []

    for base in base_rows:

        t = float(
            base[
                "time_s"
            ]
        )

        j = nearest_index(
            extra_times,
            t,
        )

        merged = dict(
            base
        )

        merged.update(
            extra_rows[
                j
            ]
        )

        rows.append(
            merged
        )

    times_arr = np.asarray(
        [
            r[
                "time_s"
            ]
            for r in rows
        ],
        dtype=float,
    )

    # ------------------------------------------------------------
    # Physical middle gap derivatives.
    #
    # These remain meaningful even immediately before/after
    # contact.
    #
    # positive gap velocity = opening/separating.
    # ------------------------------------------------------------

    gaps_m = (
        1e-3
        *
        np.asarray(
            [
                r[
                    "mid_gap_mm"
                ]
                for r in rows
            ],
            dtype=float,
        )
    )

    if len(
        rows
    ) >= 3:

        gap_vel = np.gradient(
            gaps_m,
            times_arr,
        )

        gap_acc = np.gradient(
            gap_vel,
            times_arr,
        )

    else:

        gap_vel = np.full(
            len(
                rows
            ),
            np.nan,
        )

        gap_acc = np.full(
            len(
                rows
            ),
            np.nan,
        )

    for i, row in enumerate(
        rows
    ):

        row[
            "mid_gap_velocity_m_s"
        ] = float(
            gap_vel[
                i
            ]
        )

        row[
            "mid_gap_accel_m_s2"
        ] = float(
            gap_acc[
                i
            ]
        )

    # ------------------------------------------------------------
    # Finite-difference middle-point radial acceleration.
    #
    # Only evaluated while contact persists.
    # ------------------------------------------------------------

    for row in rows:

        row[
            "middle_radial_accel_fd_m_s2"
        ] = np.nan

    for i in range(
        1,
        len(
            rows
        ),
    ):

        a = rows[
            i - 1
        ]

        b = rows[
            i
        ]

        if not (
            a[
                "middle_contact_active"
            ]
            and
            b[
                "middle_contact_active"
            ]
        ):
            continue

        va = a[
            "middle_radial_velocity_m_s"
        ]

        vb = b[
            "middle_radial_velocity_m_s"
        ]

        dt = (
            b[
                "time_s"
            ]
            -
            a[
                "time_s"
            ]
        )

        if (
            np.isfinite(
                va
            )
            and
            np.isfinite(
                vb
            )
            and
            dt > 0
        ):

            b[
                "middle_radial_accel_fd_m_s2"
            ] = (
                vb - va
            ) / dt

    # ------------------------------------------------------------
    # Find first middle-contact episode.
    # ------------------------------------------------------------

    flags = np.asarray(
        [
            bool(
                r[
                    "mid_contact"
                ]
            )
            for r in rows
        ],
        dtype=bool,
    )

    starts = (
        g12.contact_episode_starts(
            flags
        )
    )

    ends = (
        g12.contact_episode_ends(
            flags
        )
    )

    if not starts:

        raise RuntimeError(
            "No middle contact found."
        )

    start = int(
        starts[
            0
        ]
    )

    end = int(
        ends[
            0
        ]
    )

    t0 = float(
        rows[
            start
        ][
            "time_s"
        ]
    )

    t1 = float(
        rows[
            end
        ][
            "time_s"
        ]
    )

    episode_indices = list(
        range(
            start,
            end + 1,
        )
    )

    force_values = np.asarray(
        [
            (
                rows[
                    i
                ][
                    "contact_normal_force_N"
                ]
                if np.isfinite(
                    rows[
                        i
                    ][
                        "contact_normal_force_N"
                    ]
                )
                else -np.inf
            )
            for i in episode_indices
        ],
        dtype=float,
    )

    peak = episode_indices[
        int(
            np.argmax(
                force_values
            )
        )
    ]

    pre = nearest_index(
        times_arr,
        t0 - 0.020,
    )

    early = nearest_index(
        times_arr,
        t0 + 0.010,
    )

    mid_episode = nearest_index(
        times_arr,
        t0
        +
        0.5
        *
        (
            t1 - t0
        ),
    )

    before_end = nearest_index(
        times_arr,
        max(
            t0,
            t1 - 0.010,
        ),
    )

    # ------------------------------------------------------------
    # SAVE.
    # ------------------------------------------------------------

    csv_path = (
        out_dir
        /
        "g22c_first_middle_contact_autopsy.csv"
    )

    save_csv(
        csv_path,
        rows,
    )

    np.savez_compressed(
        out_dir
        /
        "g22c_trajectory.npz",
        times=np.asarray(
            times
        ),
        qpos=np.asarray(
            qpos
        ),
    )

    # Compact ±100 ms pre and 100 ms post-separation.
    window_rows = [
        r
        for r in rows
        if (
            t0 - 0.100
            <=
            r[
                "time_s"
            ]
            <=
            t1 + 0.100
        )
    ]

    save_csv(
        out_dir
        /
        "g22c_first_episode_window.csv",
        window_rows,
    )

    # ------------------------------------------------------------
    # REPORT.
    # ------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "G22C FIRST MIDDLE-CONTACT EVENT"
    )
    print("=" * 150)

    print(
        f"first contact start : "
        f"{t0:.6f} s"
    )

    print(
        f"encoder at start    : "
        f"{rows[start]['encoder_tick']:.3f}"
    )

    print(
        f"first contact end   : "
        f"{t1:.6f} s"
    )

    print(
        f"duration            : "
        f"{t1 - t0:.6f} s"
    )

    if len(
        starts
    ) >= 2:

        print(
            f"second contact start: "
            f"{rows[starts[1]]['time_s']:.6f} s"
        )

        print(
            f"detached gap time   : "
            f"{rows[starts[1]]['time_s'] - t1:.6f} s"
        )

    print()
    print(
        "IMPORTANT SIGN:"
    )

    print(
        "  gap velocity > 0   = middle is moving AWAY from bottle"
    )

    print(
        "  radial velocity > 0= contact point is moving AWAY from bottle"
    )

    print()

    print(
        "stage       rel_ms enc     "
        "MCP     PIP     DIP    "
        "vMCP    vPIP    vDIP    "
        "gap_mm  gapVel_mm/s  "
        "radVel_mm/s  Fn_N"
    )

    print(
        "-" * 150
    )

    stages = [
        (
            "pre",
            pre,
        ),
        (
            "touch",
            start,
        ),
        (
            "+10ms",
            early,
        ),
        (
            "peakF",
            peak,
        ),
        (
            "middle",
            mid_episode,
        ),
        (
            "end-10",
            before_end,
        ),
        (
            "end",
            end,
        ),
    ]

    for label, i in stages:

        r = rows[
            i
        ]

        rv = r[
            "middle_radial_velocity_m_s"
        ]

        fn = r[
            "contact_normal_force_N"
        ]

        print(
            f"{label:<8s} "
            f"{1000.0*(r['time_s']-t0):+7.1f} "
            f"{r['encoder_tick']:7.1f} "
            f"{r['mcp_deg']:7.2f} "
            f"{r['pip_deg']:7.2f} "
            f"{r['dip_deg']:7.2f} "
            f"{r['mcp_qvel_deg_s']:+7.2f} "
            f"{r['pip_qvel_deg_s']:+7.2f} "
            f"{r['dip_qvel_deg_s']:+7.2f} "
            f"{r['mid_gap_mm']:+7.3f} "
            f"{1000.0*r['mid_gap_velocity_m_s']:+11.3f} "
            f"{1000.0*rv if np.isfinite(rv) else np.nan:+11.3f} "
            f"{fn if np.isfinite(fn) else 0.0:7.3f}"
        )

    # ------------------------------------------------------------
    # Contact geometry.
    # ------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "CONTACT GEOMETRY"
    )
    print("=" * 150)

    for label, i in [
        (
            "touch",
            start,
        ),
        (
            "peak force",
            peak,
        ),
        (
            "before separation",
            before_end,
        ),
    ]:

        r = rows[
            i
        ]

        print()
        print(
            label
        )

        print(
            "  world point xyz [m] = "
            f"{r['contact_x_m']:+.6f}, "
            f"{r['contact_y_m']:+.6f}, "
            f"{r['contact_z_m']:+.6f}"
        )

        print(
            "  middle-local xyz [m]= "
            f"{r['contact_local_mid_x_m']:+.6f}, "
            f"{r['contact_local_mid_y_m']:+.6f}, "
            f"{r['contact_local_mid_z_m']:+.6f}"
        )

        print(
            f"  contact normal force = "
            f"{r['contact_normal_force_N']:.6f} N"
        )

        print(
            f"  tangential force     = "
            f"{r['contact_tangent_force_N']:.6f} N"
        )

        print(
            f"  MuJoCo contact dist  = "
            f"{r['contact_dist_mm']:+.6f} mm"
        )

        print(
            f"  normal/radial align  = "
            f"{r['contact_normal_alignment']:.6f}"
        )

        print(
            f"  radial velocity      = "
            f"{1000.0*r['middle_radial_velocity_m_s']:+.6f} mm/s"
        )

        print(
            f"  radial accel FD      = "
            f"{r['middle_radial_accel_fd_m_s2']:+.6f} m/s^2"
        )

    # ------------------------------------------------------------
    # Generalized-force decomposition.
    # ------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "GENERALIZED-FORCE DECOMPOSITION"
    )
    print("=" * 150)

    print_joint_dynamics(
        "20 ms BEFORE FIRST CONTACT",
        rows[
            pre
        ],
    )

    print_joint_dynamics(
        "AT FIRST CONTACT",
        rows[
            start
        ],
    )

    print_joint_dynamics(
        "AT PEAK FIRST-CONTACT FORCE",
        rows[
            peak
        ],
    )

    print_joint_dynamics(
        "10 ms BEFORE CONTACT LOSS",
        rows[
            before_end
        ],
    )

    # ------------------------------------------------------------
    # Constraint-force jump.
    # ------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "CONSTRAINT TORQUE CHANGE: PRE-CONTACT -> PEAK CONTACT"
    )
    print("=" * 150)

    for joint in [
        "mcp",
        "pip",
        "dip",
    ]:

        before = float(
            rows[
                pre
            ][
                f"{joint}_qfrc_constraint_Nm"
            ]
        )

        after = float(
            rows[
                peak
            ][
                f"{joint}_qfrc_constraint_Nm"
            ]
        )

        print(
            f"{joint.upper():4s}: "
            f"{before:+.6f} -> "
            f"{after:+.6f} Nm   "
            f"delta={after-before:+.6f} Nm"
        )

    # ------------------------------------------------------------
    # Basic mechanical diagnosis flags.
    # ------------------------------------------------------------

    end_gap_velocity = float(
        rows[
            end
        ][
            "mid_gap_velocity_m_s"
        ]
    )

    pre_end_gap_velocity = float(
        rows[
            before_end
        ][
            "mid_gap_velocity_m_s"
        ]
    )

    print()
    print("=" * 150)
    print(
        "G22C MECHANICAL FLAGS"
    )
    print("=" * 150)

    if (
        pre_end_gap_velocity > 0
        or
        end_gap_velocity > 0
    ):

        print(
            "SEPARATION CONFIRMED:"
        )

        print(
            "  middle gap is already OPENING "
            "before/at loss of contact."
        )

    else:

        print(
            "Gap derivative does not clearly show "
            "outward motion before separation."
        )

    pip_peak_constraint = float(
        rows[
            peak
        ][
            "pip_qfrc_constraint_Nm"
        ]
    )

    pip_peak_tendon = float(
        rows[
            peak
        ][
            "pip_tendon_tau_total_est_Nm"
        ]
    )

    mcp_peak_constraint = float(
        rows[
            peak
        ][
            "mcp_qfrc_constraint_Nm"
        ]
    )

    mcp_peak_tendon = float(
        rows[
            peak
        ][
            "mcp_tendon_tau_total_est_Nm"
        ]
    )

    print()
    print(
        "Peak contact comparison:"
    )

    print(
        f"  MCP tendon / constraint = "
        f"{mcp_peak_tendon:+.6f} / "
        f"{mcp_peak_constraint:+.6f} Nm"
    )

    print(
        f"  PIP tendon / constraint = "
        f"{pip_peak_tendon:+.6f} / "
        f"{pip_peak_constraint:+.6f} Nm"
    )

    print()
    print(
        "Do NOT change parameters yet."
    )

    print(
        "The next model change will be chosen ONLY "
        "after interpreting these numbers."
    )

    with (
        out_dir
        /
        "manifest.txt"
    ).open(
        "w"
    ) as f:

        f.write(
            "G22C first-middle-contact dynamics autopsy\n"
        )

        f.write(
            "No model fitting or parameter changes.\n"
        )

        f.write(
            "G22B proved distal contact is not required "
            "for the middle-contact bounce.\n"
        )

        f.write(
            "Purpose: determine the mechanical source "
            "of first middle-contact separation.\n"
        )

    print()
    print(
        f"Full CSV:\n  {csv_path}"
    )

    print()
    print(
        "Compact contact window:\n  "
        f"{out_dir / 'g22c_first_episode_window.csv'}"
    )

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )

    # ------------------------------------------------------------
    # Mandatory visual confirmation.
    # ------------------------------------------------------------

    if not args.no_gui:

        print()
        print("=" * 150)
        print(
            "G22C GUI — EXACT DYNAMIC BASELINE TRAJECTORY"
        )
        print("=" * 150)

        print(
            "Watch only the FIRST middle contact and "
            "the direction in which it leaves the bottle."
        )

        g7.exact_gui_replay(
            model_xml,
            qpos,
            times,
            "G22C first-middle-contact autopsy",
        )


if __name__ == "__main__":
    main()
