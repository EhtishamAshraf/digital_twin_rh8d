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


# ================================================================
# V35G12 — TENDON TRANSMISSION / ENERGY / REDISTRIBUTION AUDIT
#
# NO PARAMETER FITTING.
#
# Question:
#
# When the middle phalanx becomes constrained, why does further
# tendon shortening NOT redistribute strongly into DIP rotation?
#
# Hardware target:
#
# MCP
# -> middle touches
# -> PIP wraps
# -> middle constrained
# -> PIP slows/stops
# -> DIP becomes active
# -> distal touches
# -> DIP stops
#
# Final success criterion:
#
#   hardware-like mechanism + low pose RMSE
#
# ================================================================


SCALE = 5.70e-6

TENDON_K = 2100.0
TENDON_DAMPING = 2.0

CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

TENDON_NAME = "idx_flex_spatial"

JOINT_NAMES = {
    "mcp": "Index_Proximal--palm_assembly:1",
    "pip": "Index_Middle--Index_Proximal",
    "dip": "Index_Distal--Index_Middle",
    "spool": "index_motor_spool",
}

CONTACT_GEOMS = {
    "mid": "collision_idx_mid_mesh",
    "tip": "collision_idx_tip",
}

FIXTURE_NAME = "index_contact_fixture"


def get_joint_refs(model):

    refs = {}

    for key, name in JOINT_NAMES.items():

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            raise RuntimeError(
                f"Missing joint: {name}"
            )

        refs[key] = {
            "jid": jid,
            "qadr": int(
                model.jnt_qposadr[jid]
            ),
            "dadr": int(
                model.jnt_dofadr[jid]
            ),
        }

    return refs


def tendon_spring_reference(model, tid):

    value = np.asarray(
        model.tendon_lengthspring[tid],
        dtype=float,
    ).reshape(-1)

    # Current model uses identical lower/upper spring references.
    if len(value) == 1:
        return float(value[0])

    if abs(value[1] - value[0]) < 1e-12:
        return float(value[0])

    # Generic deadband handling if the model changes later.
    return (
        float(value[0]),
        float(value[1]),
    )


def signed_extension(length, spring_ref):

    if isinstance(
        spring_ref,
        tuple,
    ):

        low, high = spring_ref

        if length < low:
            return length - low

        if length > high:
            return length - high

        return 0.0

    return (
        float(length)
        -
        float(spring_ref)
    )


def contact_episode_starts(flags):

    flags = np.asarray(
        flags,
        dtype=bool,
    )

    starts = []

    previous = False

    for i, value in enumerate(flags):

        if value and not previous:
            starts.append(i)

        previous = value

    return starts


def contact_episode_ends(flags):

    flags = np.asarray(
        flags,
        dtype=bool,
    )

    ends = []

    previous = False

    for i, value in enumerate(flags):

        if previous and not value:
            ends.append(i - 1)

        previous = value

    if len(flags) and flags[-1]:
        ends.append(
            len(flags) - 1
        )

    return ends


def median_window(
    rows,
    indices,
    field,
):

    if len(indices) == 0:
        return np.nan

    values = np.asarray(
        [
            rows[i][field]
            for i in indices
        ],
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.median(values)
    )


def make_baseline_model(
    out_dir,
):

    source = (
        Path(
            g7.b2.SELECTED_MODEL
        )
        .expanduser()
        .resolve()
    )

    tree = ET.parse(source)
    root = tree.getroot()

    # Absolute mesh path because generated XML is moved into results.
    package_dir = source.parent.parent

    meshes_dir = (
        package_dir
        / "meshes"
    ).resolve()

    compiler = root.find(
        "compiler"
    )

    if compiler is None:
        compiler = ET.Element(
            "compiler"
        )
        root.insert(
            0,
            compiler,
        )

    compiler.set(
        "meshdir",
        str(meshes_dir),
    )

    # ------------------------------------------------------------
    # Restore exact baseline index mechanics explicitly.
    # ------------------------------------------------------------

    joint_values = {
        JOINT_NAMES["mcp"]: {
            "damping": "0.015",
            "frictionloss": "0.002",
            "stiffness": "0.0045",
            "springref": "-0.10",
        },

        JOINT_NAMES["pip"]: {
            "damping": "0.015",
            "frictionloss": "0.00125",
            "stiffness": "0.0015",
            "springref": "-0.10",
        },

        JOINT_NAMES["dip"]: {
            "damping": "0.015",
            "frictionloss": "0",
            "stiffness": "0.0475",
        },
    }

    found_joints = set()

    for joint in root.iter(
        "joint"
    ):

        name = joint.get(
            "name"
        )

        if name in joint_values:

            for attr, value in (
                joint_values[name]
                .items()
            ):

                joint.set(
                    attr,
                    value,
                )

            found_joints.add(
                name
            )

    missing = (
        set(
            joint_values
        )
        -
        found_joints
    )

    if missing:
        raise RuntimeError(
            "Missing index joints:\n"
            +
            "\n".join(
                sorted(missing)
            )
        )

    # ------------------------------------------------------------
    # Restore exact index tendon.
    # ------------------------------------------------------------

    tendon = None

    for spatial in root.iter(
        "spatial"
    ):

        if (
            spatial.get("name")
            ==
            TENDON_NAME
        ):

            tendon = spatial
            break

    if tendon is None:
        raise RuntimeError(
            f"Missing tendon: "
            f"{TENDON_NAME}"
        )

    tendon.set(
        "stiffness",
        f"{TENDON_K:.10g}",
    )

    tendon.set(
        "damping",
        f"{TENDON_DAMPING:.10g}",
    )

    tendon.set(
        "frictionloss",
        "0",
    )

    # ------------------------------------------------------------
    # Zero physical contact margin/gap.
    # ------------------------------------------------------------

    wanted = {
        FIXTURE_NAME,
        "collision_idx_prox",
        "collision_idx_mid_mesh",
        "collision_idx_tip",
    }

    found = set()

    for geom in root.iter(
        "geom"
    ):

        name = geom.get(
            "name"
        )

        if name in wanted:

            geom.set(
                "margin",
                "0",
            )

            geom.set(
                "gap",
                "0",
            )

            found.add(
                name
            )

    missing = (
        wanted
        -
        found
    )

    if missing:
        raise RuntimeError(
            "Missing contact geoms:\n"
            +
            "\n".join(
                sorted(missing)
            )
        )

    destination = (
        out_dir
        /
        "v35g12_audit_model.xml"
    )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )

    shutil.copy2(
        source,
        out_dir
        /
        "v35g12_original_selected_model.xml",
    )

    return destination


def run_audit(
    model_xml,
    bag,
    contact_replay,
):

    rows = []

    refs = {
        "initialized": False,
    }

    true_step = (
        mujoco.mj_step
    )


    def capture_step(
        model,
        data,
        *args,
        **kwargs,
    ):

        out = true_step(
            model,
            data,
            *args,
            **kwargs,
        )

        local = (
            g7.e2a.replay_locals()
        )

        if local is None:
            return out


        if not refs[
            "initialized"
        ]:

            refs[
                "initialized"
            ] = True

            refs[
                "model"
            ] = model

            refs[
                "joints"
            ] = get_joint_refs(
                model
            )

            refs[
                "tendon_id"
            ] = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_TENDON,
                TENDON_NAME,
            )

            if refs[
                "tendon_id"
            ] < 0:
                raise RuntimeError(
                    f"Missing tendon "
                    f"{TENDON_NAME}"
                )

            refs[
                "spring_ref"
            ] = (
                tendon_spring_reference(
                    model,
                    refs[
                        "tendon_id"
                    ],
                )
            )

            refs[
                "fixture"
            ] = g7.e2a.gid(
                model,
                FIXTURE_NAME,
            )

            refs[
                "mid_gid"
            ] = g7.e2a.gid(
                model,
                CONTACT_GEOMS[
                    "mid"
                ],
            )

            refs[
                "tip_gid"
            ] = g7.e2a.gid(
                model,
                CONTACT_GEOMS[
                    "tip"
                ],
            )

            refs[
                "q_joints"
            ] = (
                g7.e2a.joint_info(
                    model
                )
            )


        tid = refs[
            "tendon_id"
        ]

        joints = refs[
            "joints"
        ]


        forces, active = (
            g7.e2a.contact_forces(
                model,
                data,
                refs[
                    "fixture"
                ],
            )
        )


        q = np.asarray(
            g7.e2a.q_deg(
                data,
                refs[
                    "q_joints"
                ],
            ),
            dtype=float,
        )


        # --------------------------------------------------------
        # Tendon state.
        # --------------------------------------------------------

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

        extension = signed_extension(
            tendon_length,
            refs[
                "spring_ref"
            ],
        )


        # Linear spring estimate.
        #
        # We log spring and damping components separately rather
        # than claiming a perfect reconstruction of MuJoCo's
        # internal tendon force.
        spring_force_est = (
            TENDON_K
            *
            extension
        )

        damping_force_est = (
            TENDON_DAMPING
            *
            tendon_velocity
        )

        elastic_energy = (
            0.5
            *
            TENDON_K
            *
            extension
            *
            extension
        )

        damping_power = (
            TENDON_DAMPING
            *
            tendon_velocity
            *
            tendon_velocity
        )


        # --------------------------------------------------------
        # Tendon Jacobian / moment arms.
        #
        # dL/dq has units m/rad.
        #
        # Its magnitude is the instantaneous tendon moment arm.
        # --------------------------------------------------------

        # --------------------------------------------------------
        # MuJoCo tendon Jacobian.
        #
        # Depending on the MuJoCo Python build, data.ten_J can be
        # exposed either as:
        #
        #   (ntendon, nv)
        #
        # or as one flattened array of length ntendon*nv.
        #
        # Normalize it here before extracting the index-tendon row.
        # --------------------------------------------------------

        # --------------------------------------------------------
        # MuJoCo tendon Jacobian.
        #
        # Depending on MuJoCo configuration, ten_J can be:
        #
        #   1. dense:  ntendon x nv
        #   2. dense flattened: ntendon*nv
        #   3. sparse CSR-like storage:
        #        ten_J
        #        ten_J_rownnz
        #        ten_J_rowadr
        #        ten_J_colind
        #
        # This model uses sparse storage, so reconstruct only
        # the row belonging to idx_flex_spatial.
        # --------------------------------------------------------

        ten_J_all = np.asarray(
            data.ten_J,
            dtype=float,
        )

        J = np.zeros(
            model.nv,
            dtype=float,
        )

        dense_expected = (
            model.ntendon
            *
            model.nv
        )

        if (
            ten_J_all.ndim == 2
            and
            ten_J_all.shape == (
                model.ntendon,
                model.nv,
            )
        ):

            J[:] = ten_J_all[
                tid,
                :
            ]

        elif (
            ten_J_all.ndim == 1
            and
            ten_J_all.size == dense_expected
        ):

            J[:] = (
                ten_J_all
                .reshape(
                    model.ntendon,
                    model.nv,
                )[
                    tid,
                    :
                ]
            )

        else:

            # Sparse tendon Jacobian.
            required = [
                "ten_J_rownnz",
                "ten_J_rowadr",
                "ten_J_colind",
            ]

            missing = [
                name
                for name in required
                if not hasattr(
                    model,
                    name,
                )
            ]

            if missing:

                raise RuntimeError(
                    "ten_J appears sparse "
                    f"(size={ten_J_all.size}, "
                    f"dense expected={dense_expected}), "
                    "but sparse metadata is missing on mjModel: "
                    + ", ".join(missing)
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

            if nnz < 0:

                raise RuntimeError(
                    f"Invalid tendon Jacobian nnz={nnz}"
                )

            if adr < 0:

                raise RuntimeError(
                    f"Invalid tendon Jacobian row address={adr}"
                )

            if (
                adr + nnz
                >
                ten_J_all.size
            ):

                raise RuntimeError(
                    "Sparse tendon Jacobian row exceeds "
                    "ten_J storage: "
                    f"adr={adr}, nnz={nnz}, "
                    f"size={ten_J_all.size}"
                )

            cols = colind[
                adr:
                adr + nnz
            ]

            vals = ten_J_all[
                adr:
                adr + nnz
            ]

            if len(cols) != len(vals):

                raise RuntimeError(
                    "Sparse tendon Jacobian column/value "
                    "length mismatch"
                )

            for col, value in zip(
                cols,
                vals,
            ):

                col = int(
                    col
                )

                if (
                    col < 0
                    or
                    col >= model.nv
                ):

                    raise RuntimeError(
                        "Invalid tendon Jacobian column: "
                        f"{col}, nv={model.nv}"
                    )

                J[
                    col
                ] = float(
                    value
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
                float(q[0]),

            "pip_deg":
                float(q[1]),

            "dip_deg":
                float(q[2]),

            "mid_contact":
                bool(
                    active[
                        CONTACT_GEOMS[
                            "mid"
                        ]
                    ]
                ),

            "tip_contact":
                bool(
                    active[
                        CONTACT_GEOMS[
                            "tip"
                        ]
                    ]
                ),

            "mid_force_N":
                float(
                    forces[
                        CONTACT_GEOMS[
                            "mid"
                        ]
                    ]
                ),

            "tip_force_N":
                float(
                    forces[
                        CONTACT_GEOMS[
                            "tip"
                        ]
                    ]
                ),

            "mid_gap_mm":
                1000.0
                *
                float(
                    g7.e2a.physical_distance(
                        model,
                        data,
                        refs[
                            "mid_gid"
                        ],
                        refs[
                            "fixture"
                        ],
                    )
                ),

            "tip_gap_mm":
                1000.0
                *
                float(
                    g7.e2a.physical_distance(
                        model,
                        data,
                        refs[
                            "tip_gid"
                        ],
                        refs[
                            "fixture"
                        ],
                    )
                ),

            "tendon_length_m":
                tendon_length,

            "tendon_velocity_m_s":
                tendon_velocity,

            "tendon_extension_m":
                extension,

            "tendon_extension_mm":
                1000.0
                *
                extension,

            "tendon_spring_force_est_N":
                spring_force_est,

            "tendon_damping_force_est_N":
                damping_force_est,

            "tendon_elastic_energy_J":
                elastic_energy,

            "tendon_damping_power_W":
                damping_power,
        }


        # --------------------------------------------------------
        # Per-DOF mechanics.
        # --------------------------------------------------------

        for key in (
            "mcp",
            "pip",
            "dip",
            "spool",
        ):

            ref = joints[
                key
            ]

            dadr = ref[
                "dadr"
            ]

            qvel = float(
                data.qvel[
                    dadr
                ]
            )

            moment = float(
                J[
                    dadr
                ]
            )

            # Contribution from this DOF to tendon length velocity.
            contribution = (
                moment
                *
                qvel
            )

            # Spring-only generalized torque estimate.
            #
            # Q = -F * dL/dq
            tendon_tau_spring = (
                -
                spring_force_est
                *
                moment
            )

            row[
                f"{key}_qvel_native"
            ] = qvel

            if key != "spool":

                row[
                    f"{key}_vel_deg_s"
                ] = math.degrees(
                    qvel
                )

            else:

                row[
                    "spool_vel_m_s"
                ] = qvel

                row[
                    "spool_qpos_m"
                ] = float(
                    data.qpos[
                        ref[
                            "qadr"
                        ]
                    ]
                )

            row[
                f"{key}_dL_dq"
            ] = moment

            row[
                f"{key}_moment_arm_mm"
            ] = (
                1000.0
                *
                abs(moment)
            )

            row[
                f"{key}_tendon_velocity_contribution_m_s"
            ] = contribution

            row[
                f"{key}_tendon_tau_spring_est_Nm"
            ] = tendon_tau_spring

            row[
                f"{key}_qfrc_passive_Nm"
            ] = float(
                data.qfrc_passive[
                    dadr
                ]
            )

            row[
                f"{key}_qfrc_constraint_Nm"
            ] = float(
                data.qfrc_constraint[
                    dadr
                ]
            )


        joint_sum = (
            row[
                "mcp_tendon_velocity_contribution_m_s"
            ]
            +
            row[
                "pip_tendon_velocity_contribution_m_s"
            ]
            +
            row[
                "dip_tendon_velocity_contribution_m_s"
            ]
        )

        row[
            "index_joint_tendon_velocity_sum_m_s"
        ] = joint_sum

        row[
            "spool_plus_index_velocity_sum_m_s"
        ] = (
            joint_sum
            +
            row[
                "spool_tendon_velocity_contribution_m_s"
            ]
        )


        # Full state only for visual replay.
        row[
            "_qpos"
        ] = np.asarray(
            data.qpos,
            dtype=float,
        ).copy()


        rows.append(
            row
        )

        return out


    old_step = (
        mujoco.mj_step
    )

    mujoco.mj_step = (
        capture_step
    )

    try:

        summary = g7.e2b.run_one(
            CONTACT_DAMPING,
            model_xml,
            bag,
            contact_replay,
        )

    finally:

        mujoco.mj_step = (
            old_step
        )


    if not rows:
        raise RuntimeError(
            "No G12 trajectory captured."
        )


    qpos = np.vstack(
        [
            row[
                "_qpos"
            ]
            for row in rows
        ]
    )

    times = np.asarray(
        [
            row[
                "time_s"
            ]
            for row in rows
        ],
        dtype=float,
    )

    return (
        rows,
        qpos,
        times,
        summary,
    )


def save_timeseries(
    path,
    rows,
):

    fields = [
        key
        for key in rows[0].keys()
        if key != "_qpos"
    ]

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    key:
                        row[key]
                    for key in fields
                }
            )


def phase_indices(
    times,
    start_time,
    end_time,
):

    return np.flatnonzero(
        (times >= start_time)
        &
        (times <= end_time)
    )


def build_event_report(
    rows,
    contact_replay,
):

    times = np.asarray(
        [
            r["time_s"]
            for r in rows
        ],
        dtype=float,
    )

    mid = np.asarray(
        [
            r["mid_contact"]
            for r in rows
        ],
        dtype=bool,
    )

    tip = np.asarray(
        [
            r["tip_contact"]
            for r in rows
        ],
        dtype=bool,
    )

    mid_starts = (
        contact_episode_starts(
            mid
        )
    )

    mid_ends = (
        contact_episode_ends(
            mid
        )
    )

    tip_starts = (
        contact_episode_starts(
            tip
        )
    )

    tip_ends = (
        contact_episode_ends(
            tip
        )
    )


    event_rows = []


    def add_event(
        name,
        index,
    ):

        if index is None:
            return

        r = rows[
            index
        ]

        event_rows.append(
            {
                "event":
                    name,

                "time_s":
                    r[
                        "time_s"
                    ],

                "encoder_tick":
                    r[
                        "encoder_tick"
                    ],

                "mcp_deg":
                    r[
                        "mcp_deg"
                    ],

                "pip_deg":
                    r[
                        "pip_deg"
                    ],

                "dip_deg":
                    r[
                        "dip_deg"
                    ],

                "mid_contact":
                    r[
                        "mid_contact"
                    ],

                "tip_contact":
                    r[
                        "tip_contact"
                    ],

                "tendon_extension_mm":
                    r[
                        "tendon_extension_mm"
                    ],

                "tendon_spring_force_est_N":
                    r[
                        "tendon_spring_force_est_N"
                    ],

                "tendon_elastic_energy_J":
                    r[
                        "tendon_elastic_energy_J"
                    ],

                "mcp_moment_arm_mm":
                    r[
                        "mcp_moment_arm_mm"
                    ],

                "pip_moment_arm_mm":
                    r[
                        "pip_moment_arm_mm"
                    ],

                "dip_moment_arm_mm":
                    r[
                        "dip_moment_arm_mm"
                    ],

                "mcp_dLdt_m_s":
                    r[
                        "mcp_tendon_velocity_contribution_m_s"
                    ],

                "pip_dLdt_m_s":
                    r[
                        "pip_tendon_velocity_contribution_m_s"
                    ],

                "dip_dLdt_m_s":
                    r[
                        "dip_tendon_velocity_contribution_m_s"
                    ],

                "mcp_tau_tendon_est_Nm":
                    r[
                        "mcp_tendon_tau_spring_est_Nm"
                    ],

                "pip_tau_tendon_est_Nm":
                    r[
                        "pip_tendon_tau_spring_est_Nm"
                    ],

                "dip_tau_tendon_est_Nm":
                    r[
                        "dip_tendon_tau_spring_est_Nm"
                    ],

                "mcp_constraint_Nm":
                    r[
                        "mcp_qfrc_constraint_Nm"
                    ],

                "pip_constraint_Nm":
                    r[
                        "pip_qfrc_constraint_Nm"
                    ],

                "dip_constraint_Nm":
                    r[
                        "dip_qfrc_constraint_Nm"
                    ],
            }
        )


    add_event(
        "first_middle_contact",
        (
            mid_starts[0]
            if mid_starts
            else None
        ),
    )

    add_event(
        "first_tip_contact",
        (
            tip_starts[0]
            if tip_starts
            else None
        ),
    )

    if mid_ends:
        add_event(
            "first_middle_contact_end",
            mid_ends[0],
        )

    if tip_ends:
        add_event(
            "first_tip_contact_end",
            tip_ends[0],
        )

    if len(
        mid_starts
    ) >= 2:
        add_event(
            "second_middle_contact",
            mid_starts[1],
        )

    if len(
        tip_starts
    ) >= 2:
        add_event(
            "second_tip_contact",
            tip_starts[1],
        )


    target_index = int(
        np.argmin(
            np.abs(
                times
                -
                float(
                    contact_replay.TARGET_BAG_TIME
                )
            )
        )
    )

    add_event(
        "hardware_target_time",
        target_index,
    )


    # ------------------------------------------------------------
    # Phase summaries.
    # ------------------------------------------------------------

    phases = []


    def summarize_phase(
        name,
        indices,
    ):

        if len(indices) < 2:
            return

        first = rows[
            int(indices[0])
        ]

        last = rows[
            int(indices[-1])
        ]

        phases.append(
            {
                "phase":
                    name,

                "start_time_s":
                    first[
                        "time_s"
                    ],

                "end_time_s":
                    last[
                        "time_s"
                    ],

                "duration_s":
                    last[
                        "time_s"
                    ]
                    -
                    first[
                        "time_s"
                    ],

                "delta_mcp_deg":
                    last[
                        "mcp_deg"
                    ]
                    -
                    first[
                        "mcp_deg"
                    ],

                "delta_pip_deg":
                    last[
                        "pip_deg"
                    ]
                    -
                    first[
                        "pip_deg"
                    ],

                "delta_dip_deg":
                    last[
                        "dip_deg"
                    ]
                    -
                    first[
                        "dip_deg"
                    ],

                "delta_tendon_extension_mm":
                    last[
                        "tendon_extension_mm"
                    ]
                    -
                    first[
                        "tendon_extension_mm"
                    ],

                "median_tendon_force_est_N":
                    median_window(
                        rows,
                        indices,
                        "tendon_spring_force_est_N",
                    ),

                "median_mcp_moment_arm_mm":
                    median_window(
                        rows,
                        indices,
                        "mcp_moment_arm_mm",
                    ),

                "median_pip_moment_arm_mm":
                    median_window(
                        rows,
                        indices,
                        "pip_moment_arm_mm",
                    ),

                "median_dip_moment_arm_mm":
                    median_window(
                        rows,
                        indices,
                        "dip_moment_arm_mm",
                    ),

                "median_abs_mcp_dLdt":
                    median_window(
                        rows,
                        indices,
                        "mcp_tendon_velocity_contribution_m_s",
                    ),

                "median_abs_pip_dLdt":
                    float(
                        np.median(
                            np.abs(
                                [
                                    rows[i][
                                        "pip_tendon_velocity_contribution_m_s"
                                    ]
                                    for i in indices
                                ]
                            )
                        )
                    ),

                "median_abs_dip_dLdt":
                    float(
                        np.median(
                            np.abs(
                                [
                                    rows[i][
                                        "dip_tendon_velocity_contribution_m_s"
                                    ]
                                    for i in indices
                                ]
                            )
                        )
                    ),

                "median_abs_spool_dLdt":
                    float(
                        np.median(
                            np.abs(
                                [
                                    rows[i][
                                        "spool_tendon_velocity_contribution_m_s"
                                    ]
                                    for i in indices
                                ]
                            )
                        )
                    ),

                "median_pip_constraint_Nm":
                    median_window(
                        rows,
                        indices,
                        "pip_qfrc_constraint_Nm",
                    ),

                "median_dip_constraint_Nm":
                    median_window(
                        rows,
                        indices,
                        "dip_qfrc_constraint_Nm",
                    ),
            }
        )


    # 150 ms before first middle contact.
    if mid_starts:

        i_mid = (
            mid_starts[0]
        )

        t_mid = times[
            i_mid
        ]

        summarize_phase(
            "150ms_before_middle",
            phase_indices(
                times,
                t_mid - 0.150,
                t_mid,
            ),
        )


    # First middle -> first tip.
    if (
        mid_starts
        and
        tip_starts
        and
        tip_starts[0]
        >
        mid_starts[0]
    ):

        summarize_phase(
            "first_middle_to_first_tip",
            np.arange(
                mid_starts[0],
                tip_starts[0] + 1,
            ),
        )


    # First simultaneous-contact segment.
    overlap = (
        mid
        &
        tip
    )

    overlap_starts = (
        contact_episode_starts(
            overlap
        )
    )

    overlap_ends = (
        contact_episode_ends(
            overlap
        )
    )

    if (
        overlap_starts
        and
        overlap_ends
    ):

        summarize_phase(
            "first_simultaneous_contact",
            np.arange(
                overlap_starts[0],
                overlap_ends[0] + 1,
            ),
        )


    # First fully free bounce period after simultaneous contact.
    if (
        overlap_ends
        and
        len(
            mid_starts
        ) >= 2
    ):

        start = (
            overlap_ends[0]
            + 1
        )

        end = (
            mid_starts[1]
            - 1
        )

        if end > start:

            summarize_phase(
                "bounce_free_motion",
                np.arange(
                    start,
                    end + 1,
                ),
            )


    # Second contact / settling phase.
    if len(
        mid_starts
    ) >= 2:

        start = (
            mid_starts[1]
        )

        end = min(
            len(rows) - 1,
            start
            +
            int(
                round(
                    0.300
                    /
                    max(
                        np.median(
                            np.diff(
                                times
                            )
                        ),
                        1e-6,
                    )
                )
            ),
        )

        summarize_phase(
            "300ms_after_second_middle",
            np.arange(
                start,
                end + 1,
            ),
        )


    return (
        event_rows,
        phases,
        {
            "middle_episode_count":
                len(
                    mid_starts
                ),

            "tip_episode_count":
                len(
                    tip_starts
                ),
        },
    )


def write_csv(
    path,
    rows,
):

    if not rows:
        return

    with path.open(
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
        writer.writerows(
            rows
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
            "v35g12_tendon_transmission_audit"
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


    g7.b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        g7.b2.v30.v24.base.contact_replay
    )


    model_xml = make_baseline_model(
        out_dir
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

        contact_replay.CONTACT_DAMPING = (
            CONTACT_DAMPING
        )


        print()
        print("=" * 150)

        print(
            "V35G12 — TENDON TRANSMISSION / "
            "ENERGY / REDISTRIBUTION AUDIT"
        )

        print("=" * 150)

        print()
        print(
            "NO PARAMETERS ARE BEING FIT."
        )

        print()
        print(
            "Question:"
        )

        print(
            "  Why does PIP/DIP redistribution occur "
            "through bounce instead of while the "
            "middle stays constrained?"
        )


        (
            rows,
            qpos,
            times,
            replay_summary,
        ) = run_audit(
            model_xml,
            bag,
            contact_replay,
        )


    finally:

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


    save_timeseries(
        out_dir
        /
        "v35g12_timeseries.csv",
        rows,
    )


    np.savez_compressed(
        out_dir
        /
        "v35g12_trajectory.npz",
        times=times,
        qpos=qpos,
    )


    (
        events,
        phases,
        topology,
    ) = build_event_report(
        rows,
        contact_replay,
    )


    write_csv(
        out_dir
        /
        "v35g12_events.csv",
        events,
    )

    write_csv(
        out_dir
        /
        "v35g12_phases.csv",
        phases,
    )


    # ------------------------------------------------------------
    # Terminal summary.
    # ------------------------------------------------------------

    print()
    print("=" * 150)

    print(
        "CONTACT TOPOLOGY"
    )

    print("=" * 150)

    print(
        f"middle contact episodes = "
        f"{topology['middle_episode_count']}"
    )

    print(
        f"tip contact episodes    = "
        f"{topology['tip_episode_count']}"
    )


    print()
    print("=" * 150)

    print(
        "EVENT STATES"
    )

    print("=" * 150)

    for e in events:

        print()
        print(
            f"{e['event']}:"
        )

        print(
            "  pose MCP/PIP/DIP = "
            f"{e['mcp_deg']:.3f} / "
            f"{e['pip_deg']:.3f} / "
            f"{e['dip_deg']:.3f} deg"
        )

        print(
            "  contact mid/tip = "
            f"{e['mid_contact']} / "
            f"{e['tip_contact']}"
        )

        print(
            "  tendon extension = "
            f"{e['tendon_extension_mm']:+.4f} mm"
        )

        print(
            "  tendon spring-force estimate = "
            f"{e['tendon_spring_force_est_N']:+.3f} N"
        )

        print(
            "  elastic energy = "
            f"{1000*e['tendon_elastic_energy_J']:.4f} mJ"
        )

        print(
            "  moment arms MCP/PIP/DIP = "
            f"{e['mcp_moment_arm_mm']:.3f} / "
            f"{e['pip_moment_arm_mm']:.3f} / "
            f"{e['dip_moment_arm_mm']:.3f} mm"
        )

        print(
            "  tendon dL/dt contributions MCP/PIP/DIP = "
            f"{1e3*e['mcp_dLdt_m_s']:+.4f} / "
            f"{1e3*e['pip_dLdt_m_s']:+.4f} / "
            f"{1e3*e['dip_dLdt_m_s']:+.4f} mm/s"
        )

        print(
            "  tendon torque est MCP/PIP/DIP = "
            f"{e['mcp_tau_tendon_est_Nm']:+.5f} / "
            f"{e['pip_tau_tendon_est_Nm']:+.5f} / "
            f"{e['dip_tau_tendon_est_Nm']:+.5f} Nm"
        )

        print(
            "  constraint torque MCP/PIP/DIP = "
            f"{e['mcp_constraint_Nm']:+.5f} / "
            f"{e['pip_constraint_Nm']:+.5f} / "
            f"{e['dip_constraint_Nm']:+.5f} Nm"
        )


    print()
    print("=" * 150)

    print(
        "PHASE REDISTRIBUTION"
    )

    print("=" * 150)

    for p in phases:

        print()
        print(
            f"{p['phase']}:"
        )

        print(
            "  ΔMCP/PIP/DIP = "
            f"{p['delta_mcp_deg']:+.3f} / "
            f"{p['delta_pip_deg']:+.3f} / "
            f"{p['delta_dip_deg']:+.3f} deg"
        )

        print(
            "  Δtendon extension = "
            f"{p['delta_tendon_extension_mm']:+.4f} mm"
        )

        print(
            "  median moment arms MCP/PIP/DIP = "
            f"{p['median_mcp_moment_arm_mm']:.3f} / "
            f"{p['median_pip_moment_arm_mm']:.3f} / "
            f"{p['median_dip_moment_arm_mm']:.3f} mm"
        )

        print(
            "  median |dL/dt| PIP/DIP = "
            f"{1e3*p['median_abs_pip_dLdt']:.4f} / "
            f"{1e3*p['median_abs_dip_dLdt']:.4f} mm/s"
        )

        ratio = (
            p[
                "median_abs_dip_dLdt"
            ]
            /
            max(
                p[
                    "median_abs_pip_dLdt"
                ],
                1e-12,
            )
        )

        print(
            "  DIP/PIP tendon-motion contribution ratio = "
            f"{ratio:.3f}"
        )

        print(
            "  median constraint torque PIP/DIP = "
            f"{p['median_pip_constraint_Nm']:+.5f} / "
            f"{p['median_dip_constraint_Nm']:+.5f} Nm"
        )


    # ------------------------------------------------------------
    # Manifest.
    # ------------------------------------------------------------

    with (
        out_dir
        /
        "manifest.txt"
    ).open(
        "w"
    ) as f:

        f.write(
            "V35G12 — tendon transmission / energy audit\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"H1 bag:\n{bag}\n\n"
        )

        f.write(
            f"scale = "
            f"{SCALE*1e6:.3f} um/tick\n"
        )

        f.write(
            f"tendon stiffness = "
            f"{TENDON_K:.3f} N/m\n"
        )

        f.write(
            f"tendon damping = "
            f"{TENDON_DAMPING:.3f}\n"
        )

        f.write(
            f"contact mu = "
            f"{CONTACT_MU}\n"
        )

        f.write(
            f"contact stiffness = "
            f"{CONTACT_K}\n"
        )

        f.write(
            f"contact damping = "
            f"{CONTACT_DAMPING}\n\n"
        )

        f.write(
            "NO PARAMETER FITTING IN G12.\n\n"
        )

        f.write(
            "Final success criterion:\n"
        )

        f.write(
            "  1. hardware-like sequential wrap\n"
        )

        f.write(
            "  2. stable middle + distal constraint\n"
        )

        f.write(
            "  3. low joint-angle RMSE\n"
        )


    print()
    print("=" * 150)

    print(
        "MANDATORY VISUAL REPLAY"
    )

    print("=" * 150)

    print()
    print(
        "The exact baseline trajectory will now open."
    )

    print()
    print(
        "Use the numerical event/phase audit above "
        "together with what you see."
    )

    print()
    print(
        "Current model is expected to show the known "
        "bounce; G12 is diagnosing WHY."
    )


    if not args.no_gui:

        g7.exact_gui_replay(
            model_xml,
            qpos,
            times,
            "G12 baseline transmission audit",
        )


    print()
    print("=" * 150)

    print(
        "V35G12 COMPLETE"
    )

    print("=" * 150)

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
