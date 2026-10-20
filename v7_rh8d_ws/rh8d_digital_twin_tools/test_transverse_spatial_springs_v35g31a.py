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
import test_spatial_prescribed_spool_v35g29b as g29b


# =====================================================================
# G31A — RH8D TRANSVERSE SPATIAL RETURN SPRINGS
#
# Inspired architecturally by the Aero Hand, but NOT parameter-copied.
#
# RH8D facts preserved:
#
#   - one continuous flexor spatial tendon: idx_flex_spatial
#   - independent MCP / PIP / DIP joints
#   - measured hardware encoder drives the virtual spool
#
# New physically observed mechanism:
#
#   - one transverse spring crossing each RH8D joint
#   - represented as a MuJoCo <spatial> spring tendon
#
# Joint scalar stiffness/friction is removed to avoid double-counting.
# Joint damping is preserved.
#
# Spring K and preload are IDENTIFIED from H1 measured approach states.
#
# NO:
#   - fixed tendon
#   - PIP/DIP equality
#   - state-dependent switching
#   - contact-conditioned mechanics
#   - parameter scan
#   - artificial DIP actuator
# =====================================================================


MAIN_TENDON = "idx_flex_spatial"
SPOOL_JOINT = "index_motor_spool"

JOINTS = {
    "mcp": "Index_Proximal--palm_assembly:1",
    "pip": "Index_Middle--Index_Proximal",
    "dip": "Index_Distal--Index_Middle",
}

SPRINGS = {
    "mcp": "idx_mcp_return_spring",
    "pip": "idx_pip_return_spring",
    "dip": "idx_dip_return_spring",
}


# --------------------------------------------------------------
# Geometry of the OBSERVED transverse joint spring.
#
# h = offset from hinge axis toward dorsal side
# s = attachment distance along parent/child links.
#
# These define spring geometry, not fitted dynamic parameters.
# --------------------------------------------------------------

H_OFFSET = 0.0040
S_OFFSET = 0.0040


SCALE = 5.70e-6

ID_POINTS = [
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


def nearest(values, target):

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


def unit(v):

    v = np.asarray(
        v,
        dtype=float,
    )

    n = float(
        np.linalg.norm(v)
    )

    if n < 1e-12:
        raise RuntimeError(
            "Cannot normalize zero vector."
        )

    return v / n


def find_body_xml(root, name):

    for body in root.iter("body"):

        if body.get("name") == name:
            return body

    raise RuntimeError(
        f"Body not found: {name}"
    )


def find_joint_xml(root, name):

    for joint in root.iter("joint"):

        if joint.get("name") == name:
            return joint

    raise RuntimeError(
        f"Joint not found: {name}"
    )


def find_tendon_root(root):
    """
    Return ONLY the top-level MuJoCo <tendon> section.

    IMPORTANT:
    A flattened XML can also contain a <tendon> element inside
    <default>.  spatial tendons are NOT valid children there.
    """

    tendon_root = root.find("./tendon")

    if tendon_root is None:
        raise RuntimeError(
            "Top-level <mujoco><tendon> section missing."
        )

    return tendon_root


def find_geom_xml(root, name):

    for geom in root.iter("geom"):

        if geom.get("name") == name:
            return geom

    raise RuntimeError(
        f"Geom not found: {name}"
    )


def local_from_world(
    body_pos,
    body_R,
    world_point,
):

    return (
        body_R.T
        @
        (
            world_point
            -
            body_pos
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

        return raw.reshape(
            model.ntendon,
            model.nv,
        )[
            tendon_id,
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
            tendon_id
        ]
    )

    adr = int(
        rowadr[
            tendon_id
        ]
    )

    for k in range(nnz):

        idx = adr + k

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


def fit_nonnegative_force_model(
    delta_length,
    force_required,
):
    """
    Fit

        F = F0 + K * delta_length

    with

        F0 >= 0
        K  >= 0

    Only two unknowns, therefore the constrained optimum can
    be obtained by checking the unconstrained solution and
    both boundaries.
    """

    x = np.asarray(
        delta_length,
        dtype=float,
    )

    y = np.asarray(
        force_required,
        dtype=float,
    )

    X = np.column_stack(
        [
            np.ones_like(x),
            x,
        ]
    )

    candidates = []

    beta, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    if np.all(beta >= 0.0):

        candidates.append(beta)

    # F0-only
    candidates.append(
        np.array(
            [
                max(
                    float(
                        np.mean(y)
                    ),
                    0.0,
                ),
                0.0,
            ]
        )
    )

    # K-only
    denom = float(
        np.dot(x, x)
    )

    if denom > 1e-15:

        k_only = max(
            float(
                np.dot(
                    x,
                    y,
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

    candidates.append(
        np.zeros(2)
    )

    beta = min(
        candidates,
        key=lambda b:
            float(
                np.sum(
                    (
                        X @ b
                        -
                        y
                    )
                    **
                    2
                )
            ),
    )

    prediction = (
        X @ beta
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    prediction
                    -
                    y
                )
                **
                2
            )
        )
    )

    return (
        float(beta[0]),
        float(beta[1]),
        rmse,
    )


def build_geometry(
    source_xml,
    destination_xml,
):

    # --------------------------------------------------------------
    # Flatten model first.
    # --------------------------------------------------------------

    source_model = mujoco.MjModel.from_xml_path(
        str(source_xml)
    )

    flat_xml = (
        destination_xml.parent
        /
        "g31a_flat_source.xml"
    )

    mujoco.mj_saveLastXML(
        str(flat_xml),
        source_model,
    )

    model = mujoco.MjModel.from_xml_path(
        str(flat_xml)
    )

    data = mujoco.MjData(model)

    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    # Mandatory topology check.
    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        MAIN_TENDON,
    )

    if tid < 0:
        raise RuntimeError(
            "idx_flex_spatial is missing."
        )

    # --------------------------------------------------------------
    # XML modification.
    # --------------------------------------------------------------

    tree = ET.parse(flat_xml)
    root = tree.getroot()

    tendon_root = find_tendon_root(root)

    geometry_report = []

    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        joint_name = JOINTS[key]

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if jid < 0:
            raise RuntimeError(
                f"Joint not found: {joint_name}"
            )

        child_bid = int(
            model.jnt_bodyid[
                jid
            ]
        )

        parent_bid = int(
            model.body_parentid[
                child_bid
            ]
        )

        child_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            child_bid,
        )

        parent_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            parent_bid,
        )

        Rc = np.asarray(
            data.xmat[
                child_bid
            ]
        ).reshape(
            3,
            3,
        )

        Rp = np.asarray(
            data.xmat[
                parent_bid
            ]
        ).reshape(
            3,
            3,
        )

        pc = np.asarray(
            data.xpos[
                child_bid
            ]
        )

        pp = np.asarray(
            data.xpos[
                parent_bid
            ]
        )

        joint_local = np.asarray(
            model.jnt_pos[
                jid
            ],
            dtype=float,
        )

        center = (
            pc
            +
            Rc @ joint_local
        )

        axis = unit(
            Rc
            @
            np.asarray(
                model.jnt_axis[
                    jid
                ],
                dtype=float,
            )
        )

        # RH8D finger extends approximately along child-local -Z.
        distal = (
            Rc
            @
            np.array(
                [
                    0.0,
                    0.0,
                    -1.0,
                ]
            )
        )

        distal = (
            distal
            -
            axis
            *
            np.dot(
                distal,
                axis,
            )
        )

        distal = unit(distal)

        # The CAD flexor route is at negative local-Y.
        # Therefore positive local-Y is the return-spring/dorsal side.
        dorsal = (
            Rc
            @
            np.array(
                [
                    0.0,
                    1.0,
                    0.0,
                ]
            )
        )

        dorsal = (
            dorsal
            -
            axis
            *
            np.dot(
                dorsal,
                axis,
            )
        )

        dorsal = unit(dorsal)

        parent_world = (
            center
            -
            S_OFFSET
            *
            distal
            +
            H_OFFSET
            *
            dorsal
        )

        child_world = (
            center
            +
            S_OFFSET
            *
            distal
            +
            H_OFFSET
            *
            dorsal
        )

        parent_local = local_from_world(
            pp,
            Rp,
            parent_world,
        )

        child_local = local_from_world(
            pc,
            Rc,
            child_world,
        )

        parent_site_name = (
            f"g31a_{key}_spring_parent"
        )

        child_site_name = (
            f"g31a_{key}_spring_child"
        )

        parent_body_xml = find_body_xml(
            root,
            parent_name,
        )

        child_body_xml = find_body_xml(
            root,
            child_name,
        )

        ET.SubElement(
            parent_body_xml,
            "site",
            {
                "name":
                    parent_site_name,

                "pos":
                    " ".join(
                        f"{x:.12g}"
                        for x in parent_local
                    ),

                "size":
                    "0.001",

                "rgba":
                    "0 1 0 1",

                "group":
                    "4",
            },
        )

        ET.SubElement(
            child_body_xml,
            "site",
            {
                "name":
                    child_site_name,

                "pos":
                    " ".join(
                        f"{x:.12g}"
                        for x in child_local
                    ),

                "size":
                    "0.001",

                "rgba":
                    "0 1 0 1",

                "group":
                    "4",
            },
        )

        # ----------------------------------------------------------
        # Actual passive spring is now a SPATIAL tendon.
        #
        # stiffness=0 here only because K is identified later.
        # ----------------------------------------------------------

        spring = ET.SubElement(
            tendon_root,
            "spatial",
            {
                "name":
                    SPRINGS[key],

                "stiffness":
                    "0",

                "springlength":
                    "0",

                "width":
                    "0.0012",

                "rgba":
                    "0 1 0 1",

                "group":
                    "2",
            },
        )

        ET.SubElement(
            spring,
            "site",
            {
                "site":
                    parent_site_name,
            },
        )

        ET.SubElement(
            spring,
            "site",
            {
                "site":
                    child_site_name,
            },
        )

        # ----------------------------------------------------------
        # Remove old scalar joint spring + dry friction.
        #
        # Damping is deliberately preserved.
        # ----------------------------------------------------------

        joint_xml = find_joint_xml(
            root,
            joint_name,
        )

        joint_xml.set(
            "stiffness",
            "0",
        )

        joint_xml.set(
            "springref",
            "0",
        )

        joint_xml.set(
            "frictionloss",
            "0",
        )

        geometry_report.append({
            "joint":
                key,

            "parent_body":
                parent_name,

            "child_body":
                child_name,

            "parent_site":
                parent_site_name,

            "child_site":
                child_site_name,

            "h_offset_mm":
                1000.0
                *
                H_OFFSET,

            "s_offset_mm":
                1000.0
                *
                S_OFFSET,
        })

    tree.write(
        destination_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    # Compile gate.
    mujoco.MjModel.from_xml_path(
        str(destination_xml)
    )

    return geometry_report


def geometry_audit(
    xml_path,
):

    model = mujoco.MjModel.from_xml_path(
        str(xml_path)
    )

    data = mujoco.MjData(model)

    refs = g12.get_joint_refs(
        model
    )

    print()
    print("=" * 150)
    print(
        "G31A STEP 1 — TRANSVERSE-SPRING GEOMETRY AUDIT"
    )
    print("=" * 150)

    print()
    print(
        "The spring must LENGTHEN for positive flexion."
    )

    print(
        "Therefore dL/dq must remain positive."
    )

    results = {}

    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        tid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            SPRINGS[key],
        )

        if tid < 0:
            raise RuntimeError(
                f"Spring missing: {SPRINGS[key]}"
            )

        values = []

        print()
        print(
            key.upper()
        )

        print(
            "  angle      length       dL/dq"
        )

        print(
            "  --------------------------------"
        )

        for deg in [
            0.0,
            5.0,
            10.0,
            20.0,
            30.0,
            45.0,
        ]:

            data.qpos[:] = model.qpos0
            data.qvel[:] = 0.0

            data.qpos[
                refs[
                    key
                ][
                    "qadr"
                ]
            ] = math.radians(
                deg
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

            moment = float(
                J[
                    refs[
                        key
                    ][
                        "dadr"
                    ]
                ]
            )

            length = float(
                data.ten_length[
                    tid
                ]
            )

            values.append(
                (
                    deg,
                    length,
                    moment,
                )
            )

            print(
                f"  {deg:5.1f}   "
                f"{1000.0*length:9.4f} mm   "
                f"{1000.0*moment:+8.4f} mm/rad"
            )

        moments = np.asarray(
            [
                x[
                    2
                ]
                for x in values
            ]
        )

        lengths = np.asarray(
            [
                x[
                    1
                ]
                for x in values
            ]
        )

        if np.any(
            moments <= 0.001
        ):

            raise RuntimeError(
                f"{key.upper()} transverse spring does not "
                "maintain a positive restoring moment arm."
            )

        if np.any(
            np.diff(
                lengths
            )
            <=
            0.0
        ):

            raise RuntimeError(
                f"{key.upper()} spring does not lengthen "
                "monotonically with flexion."
            )

        results[
            key
        ] = {
            "open_length":
                float(
                    lengths[
                        0
                    ]
                ),
        }

    print()
    print(
        "GEOMETRY RESULT: PASS"
    )

    print(
        "All three transverse springs produce passive "
        "extension-resisting torque."
    )

    return results


def disable_fixture(
    source,
    destination,
):

    tree = ET.parse(source)
    root = tree.getroot()

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


def identify_springs(
    xml_path,
    bag,
    geometry_info,
):

    print()
    print("=" * 150)
    print(
        "G31A STEP 2 — IDENTIFY SPRING K + PRELOAD FROM H1"
    )
    print("=" * 150)

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    g7.b2.v35.METERS_PER_TICK = SCALE

    try:

        (
            replay_rows,
            replay_qpos,
            replay_times,
            replay_summary,
        ) = g12.run_audit(
            xml_path,
            bag,
            replay,
        )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

    encoder = np.asarray(
        [
            r[
                "encoder_tick"
            ]
            for r in replay_rows
        ],
        dtype=float,
    )

    model = mujoco.MjModel.from_xml_path(
        str(xml_path)
    )

    data = mujoco.MjData(model)

    refs = g12.get_joint_refs(
        model
    )

    main_tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        MAIN_TENDON,
    )

    spool_jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_JOINT,
    )

    spool_qadr = int(
        model.jnt_qposadr[
            spool_jid
        ]
    )

    open_i = nearest(
        encoder,
        100.0,
    )

    open_encoder = float(
        encoder[
            open_i
        ]
    )

    open_spool = float(
        replay_qpos[
            open_i
        ][
            spool_qadr
        ]
    )

    main_K = float(
        model.tendon_stiffness[
            main_tid
        ]
    )

    main_ref = (
        g12.tendon_spring_reference(
            model,
            main_tid,
        )
    )

    datasets = {
        "mcp": [],
        "pip": [],
        "dip": [],
    }

    print()
    print(
        "enc      Fmain       R_MCP       R_PIP       R_DIP"
    )

    print("-" * 90)

    for (
        tick,
        mcp_deg,
        pip_deg,
        dip_deg,
    ) in ID_POINTS:

        i = nearest(
            encoder,
            tick,
        )

        data.qpos[:] = np.asarray(
            replay_qpos[
                i
            ],
            dtype=float,
        )

        data.qvel[:] = 0.0

        # Direct measured-spool mechanics input.
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

        hardware = {
            "mcp": mcp_deg,
            "pip": pip_deg,
            "dip": dip_deg,
        }

        for key in [
            "mcp",
            "pip",
            "dip",
        ]:

            data.qpos[
                refs[
                    key
                ][
                    "qadr"
                ]
            ] = math.radians(
                hardware[
                    key
                ]
            )

        mujoco.mj_forward(
            model,
            data,
        )

        main_J = tendon_jacobian_row(
            model,
            data,
            main_tid,
        )

        main_extension = (
            g12.signed_extension(
                float(
                    data.ten_length[
                        main_tid
                    ]
                ),
                main_ref,
            )
        )

        main_force = max(
            0.0,
            main_K
            *
            main_extension,
        )

        required = {}

        for key in [
            "mcp",
            "pip",
            "dip",
        ]:

            dadr = refs[
                key
            ][
                "dadr"
            ]

            # Main flexor torque.
            main_tau = (
                -
                float(
                    main_J[
                        dadr
                    ]
                )
                *
                main_force
            )

            bias = float(
                data.qfrc_bias[
                    dadr
                ]
            )

            # Required passive opposing torque.
            R = (
                main_tau
                -
                bias
            )

            spring_tid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_TENDON,
                SPRINGS[key],
            )

            spring_J = tendon_jacobian_row(
                model,
                data,
                spring_tid,
            )

            spring_moment = float(
                spring_J[
                    dadr
                ]
            )

            spring_length = float(
                data.ten_length[
                    spring_tid
                ]
            )

            if spring_moment <= 0.0:

                raise RuntimeError(
                    f"{key.upper()} spring moment arm "
                    "became non-restoring."
                )

            if R <= 0.0:

                raise RuntimeError(
                    f"{key.upper()} required passive torque "
                    f"is non-positive at encoder {tick:.1f}."
                )

            required_force = (
                R
                /
                spring_moment
            )

            delta_L = (
                spring_length
                -
                geometry_info[
                    key
                ][
                    "open_length"
                ]
            )

            datasets[
                key
            ].append(
                {
                    "encoder":
                        tick,

                    "q_deg":
                        hardware[
                            key
                        ],

                    "length":
                        spring_length,

                    "delta_length":
                        delta_L,

                    "moment":
                        spring_moment,

                    "required_torque":
                        R,

                    "required_force":
                        required_force,
                }
            )

            required[
                key
            ] = R

        print(
            f"{tick:7.1f} "
            f"{main_force:+9.4f} "
            f"{required['mcp']:+11.6f} "
            f"{required['pip']:+11.6f} "
            f"{required['dip']:+11.6f}"
        )

    fitted = {}

    print()
    print("=" * 150)
    print(
        "IDENTIFIED SPATIAL SPRINGS"
    )
    print("=" * 150)

    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        samples = datasets[
            key
        ]

        delta_L = [
            s[
                "delta_length"
            ]
            for s in samples
        ]

        force = [
            s[
                "required_force"
            ]
            for s in samples
        ]

        (
            F0,
            K,
            force_rmse,
        ) = fit_nonnegative_force_model(
            delta_L,
            force,
        )

        if K <= 1e-6:

            raise RuntimeError(
                f"{key.upper()} identification produced "
                "zero spring stiffness."
            )

        open_length = (
            geometry_info[
                key
            ][
                "open_length"
            ]
        )

        rest_length = (
            open_length
            -
            F0
            /
            K
        )

        if rest_length <= 0.0:

            raise RuntimeError(
                f"{key.upper()} fitted spring rest length "
                "is non-physical."
            )

        if rest_length >= open_length:

            # Equal is okay only for F0 ~ 0.
            if F0 > 1e-6:
                raise RuntimeError(
                    f"{key.upper()} preload/rest-length "
                    "relationship is inconsistent."
                )

        # Broad physical sanity gate only.
        if K > 20000.0:

            raise RuntimeError(
                f"{key.upper()} K={K:.1f} N/m is excessive; "
                "refuse candidate."
            )

        fitted[
            key
        ] = {
            "F0":
                F0,

            "K":
                K,

            "rest_length":
                rest_length,

            "force_rmse":
                force_rmse,
        }

        print()
        print(
            key.upper()
        )

        print(
            f"  open spring length = "
            f"{1000.0*open_length:.4f} mm"
        )

        print(
            f"  preload F0        = "
            f"{F0:.4f} N"
        )

        print(
            f"  stiffness K       = "
            f"{K:.2f} N/m"
        )

        print(
            f"  rest length       = "
            f"{1000.0*rest_length:.4f} mm"
        )

        print(
            f"  force-fit RMSE    = "
            f"{force_rmse:.4f} N"
        )

    return fitted


def apply_fitted_springs(
    source_xml,
    destination_xml,
    fitted,
):

    tree = ET.parse(source_xml)
    root = tree.getroot()

    for spatial in root.iter(
        "spatial"
    ):

        name = spatial.get(
            "name"
        )

        for key in [
            "mcp",
            "pip",
            "dip",
        ]:

            if name == SPRINGS[key]:

                spatial.set(
                    "stiffness",
                    f"{fitted[key]['K']:.12g}",
                )

                spatial.set(
                    "springlength",
                    f"{fitted[key]['rest_length']:.12g}",
                )

    tree.write(
        destination_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    mujoco.MjModel.from_xml_path(
        str(destination_xml)
    )


def run_prescribed_spool(
    xml,
    bag,
    replay,
):

    clamp = g29b.PrescribedSpool()

    clamp.install()

    try:

        result = g12.run_audit(
            xml,
            bag,
            replay,
        )

        drift = (
            1000.0
            *
            max(
                clamp.post_step_drifts,
                default=0.0,
            )
        )

    finally:

        clamp.restore()

    return result, drift


def episodes(
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
            "v35g31a_transverse_spatial_springs"
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

    geometry_xml = (
        out
        /
        "g31a_transverse_springs_geometry.xml"
    )

    contact_xml = (
        out
        /
        "g31a_identified_transverse_springs.xml"
    )

    free_xml = (
        out
        /
        "g31a_identified_transverse_springs_free.xml"
    )

    # --------------------------------------------------------------
    # Absolute architecture gate.
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "G31A — RH8D TRANSVERSE SPATIAL RETURN SPRINGS"
    )
    print("=" * 150)

    g29b.verify_spatial_tendon(
        source_xml
    )

    print()
    print(
        "Architecture:"
    )

    print(
        "  measured encoder"
    )

    print(
        "      -> prescribed virtual spool"
    )

    print(
        "      -> idx_flex_spatial"
    )

    print(
        "      -> independent MCP / PIP / DIP"
    )

    print(
        "      + three transverse SPATIAL return springs"
    )

    print()
    print(
        "No Aero PIP/DIP equality constraint is copied."
    )

    # --------------------------------------------------------------
    # Build physically observed spring topology.
    # --------------------------------------------------------------

    geometry_report = build_geometry(
        source_xml,
        geometry_xml,
    )

    geometry_info = geometry_audit(
        geometry_xml
    )

    # --------------------------------------------------------------
    # Identify K + preload.
    # --------------------------------------------------------------

    fitted = identify_springs(
        geometry_xml,
        bag,
        geometry_info,
    )

    apply_fitted_springs(
        geometry_xml,
        contact_xml,
        fitted,
    )

    disable_fixture(
        contact_xml,
        free_xml,
    )

    # --------------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------------

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    g7.b2.v35.METERS_PER_TICK = SCALE

    try:

        print()
        print("=" * 150)
        print(
            "G31A STEP 3 — DYNAMIC H1 APPROACH"
        )
        print("=" * 150)

        (
            (
                free_rows,
                free_qpos,
                free_times,
                free_summary,
            ),
            free_drift,
        ) = run_prescribed_spool(
            free_xml,
            bag,
            replay,
        )

        enc = np.asarray(
            [
                r[
                    "encoder_tick"
                ]
                for r in free_rows
            ]
        )

        errors = []

        print()
        print(
            "encoder       HW MCP/PIP/DIP          "
            "SIM MCP/PIP/DIP          error"
        )

        print("-" * 140)

        for (
            tick,
            hm,
            hp,
            hd,
        ) in ID_POINTS:

            i = nearest(
                enc,
                tick,
            )

            r = free_rows[
                i
            ]

            sim = np.array(
                [
                    r[
                        "mcp_deg"
                    ],
                    r[
                        "pip_deg"
                    ],
                    r[
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

            e = sim - hw

            errors.extend(
                e.tolist()
            )

            print(
                f"{tick:8.3f}     "
                f"{hm:7.3f}/{hp:7.3f}/{hd:7.3f}     "
                f"{sim[0]:7.3f}/{sim[1]:7.3f}/{sim[2]:7.3f}     "
                f"{e[0]:+7.3f}/{e[1]:+7.3f}/{e[2]:+7.3f}"
            )

        approach_rmse = float(
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
            f"APPROACH RMSE = "
            f"{approach_rmse:.3f} deg"
        )

        print()
        print("=" * 150)
        print(
            "G31A STEP 4 — ONE H1 CONTACT RUN"
        )
        print("=" * 150)

        (
            (
                rows,
                qpos,
                times,
                summary,
            ),
            contact_drift,
        ) = run_prescribed_spool(
            contact_xml,
            bag,
            replay,
        )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

    mid_starts, mid_ends = episodes(
        rows,
        "mid_contact",
    )

    tip_starts, tip_ends = episodes(
        rows,
        "tip_contact",
    )

    if not mid_starts:

        raise RuntimeError(
            "No middle contact occurred."
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

    first_error = (
        first_pose
        -
        HW_CONTACT_POSE
    )

    first_rmse = float(
        np.sqrt(
            np.mean(
                first_error
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
        f"  encoder = "
        f"{first_row['encoder_tick']:.3f}"
    )

    print(
        f"  hardware= "
        f"{HW_CONTACT_ENCODER:.3f}"
    )

    print(
        f"  error   = "
        f"{first_row['encoder_tick'] - HW_CONTACT_ENCODER:+.3f}"
    )

    print()

    print(
        "  simulation = "
        f"{first_pose[0]:.3f} / "
        f"{first_pose[1]:.3f} / "
        f"{first_pose[2]:.3f}"
    )

    print(
        "  hardware   = "
        f"{HW_CONTACT_POSE[0]:.3f} / "
        f"{HW_CONTACT_POSE[1]:.3f} / "
        f"{HW_CONTACT_POSE[2]:.3f}"
    )

    print(
        "  error      = "
        f"{first_error[0]:+.3f} / "
        f"{first_error[1]:+.3f} / "
        f"{first_error[2]:+.3f}"
    )

    print(
        f"  pose RMSE  = "
        f"{first_rmse:.3f} deg"
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
    # Final trusted H1.
    # --------------------------------------------------------------

    row_times = np.asarray(
        [
            r[
                "time_s"
            ]
            for r in rows
        ]
    )

    ti = nearest(
        row_times,
        TARGET_TIME,
    )

    final = np.array(
        [
            rows[
                ti
            ][
                "mcp_deg"
            ],
            rows[
                ti
            ][
                "pip_deg"
            ],
            rows[
                ti
            ][
                "dip_deg"
            ],
        ]
    )

    final_error = (
        final
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
        f"{final[0]:.3f} / "
        f"{final[1]:.3f} / "
        f"{final[2]:.3f}"
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
    # Save.
    # --------------------------------------------------------------

    summary_csv = (
        out
        /
        "g31a_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        fields = [
            "approach_rmse_deg",
            "first_contact_encoder",
            "first_contact_encoder_error",
            "first_contact_pose_rmse_deg",
            "middle_episodes",
            "tip_episodes",
            "middle_gap_s",
            "final_rmse_deg",
        ]

        for key in [
            "mcp",
            "pip",
            "dip",
        ]:

            fields.extend(
                [
                    f"{key}_spring_K_N_m",
                    f"{key}_spring_preload_N",
                    f"{key}_spring_rest_mm",
                ]
            )

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        result = {
            "approach_rmse_deg":
                approach_rmse,

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
                first_rmse,

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
        }

        for key in [
            "mcp",
            "pip",
            "dip",
        ]:

            result[
                f"{key}_spring_K_N_m"
            ] = fitted[
                key
            ][
                "K"
            ]

            result[
                f"{key}_spring_preload_N"
            ] = fitted[
                key
            ][
                "F0"
            ]

            result[
                f"{key}_spring_rest_mm"
            ] = (
                1000.0
                *
                fitted[
                    key
                ][
                    "rest_length"
                ]
            )

        writer.writerow(
            result
        )

    np.savez_compressed(
        out
        /
        "g31a_contact_trajectory.npz",
        times=np.asarray(
            times
        ),
        qpos=np.asarray(
            qpos
        ),
    )

    # --------------------------------------------------------------
    # Hard decision gate.
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "G31A DECISION"
    )
    print("=" * 150)

    print(
        f"approach RMSE         = "
        f"{approach_rmse:.3f} deg"
    )

    print(
        f"first-contact RMSE    = "
        f"{first_rmse:.3f} deg"
    )

    print(
        f"first-contact enc err = "
        f"{first_row['encoder_tick'] - HW_CONTACT_ENCODER:+.1f} ticks"
    )

    print(
        f"middle detached gap   = "
        f"{detached_gap:.3f} s"
    )

    print(
        f"final H1 RMSE         = "
        f"{final_rmse:.3f} deg"
    )

    print()

    if (
        approach_rmse <= 3.0
        and
        first_rmse <= 4.0
    ):

        print(
            "PASS: physical transverse-spring architecture "
            "reproduces H1 approach/load sharing."
        )

        if detached_gap <= 0.25:

            print(
                "Contact retention also passes."
            )

            print(
                "FREEZE H1 mechanics."
            )

        else:

            print(
                "Approach mechanics pass but contact bounce remains."
            )

            print(
                "FREEZE the spring model and change ONLY "
                "contact-pad/compliance mechanics."
            )

    else:

        print(
            "FAIL: direct transverse extension-spring geometry "
            "is insufficient."
        )

        print(
            "Do NOT scan K, preload, friction, or geometry."
        )

        print(
            "The one remaining Aero-inspired refinement is "
            "a wrapped spatial spring around the joint guide, "
            "using the actual spring path."
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

    # --------------------------------------------------------------
    # GUI
    # --------------------------------------------------------------

    if not args.no_gui:

        view_xml = (
            out
            /
            "g31a_view.xml"
        )

        g29b.make_view_xml(
            contact_xml,
            view_xml,
        )

        print()
        print("=" * 150)
        print(
            "G31A GUI — MAIN + RETURN SPATIAL TENDONS"
        )
        print("=" * 150)

        print(
            "You should see:"
        )

        print(
            "  main index spatial flexor tendon"
        )

        print(
            "  + THREE green transverse return-spring tendons"
        )

        print(
            "  one crossing MCP, one PIP, one DIP."
        )

        g7.exact_gui_replay(
            view_xml,
            qpos,
            times,
            "G31A physical transverse spatial springs",
        )


if __name__ == "__main__":
    main()
