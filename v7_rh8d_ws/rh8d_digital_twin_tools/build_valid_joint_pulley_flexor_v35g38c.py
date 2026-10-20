#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import audit_index_tendon_transmission_v35g12 as g12
import build_persistent_flexor_wrap_v35g38a as g38a


# =====================================================================
# G38C — VALID OUTSIDE-SITE SPATIAL-TENDON PULLEYS
#
# Purpose:
#   Correct the geometric defect in G38A/G38B:
#
#       tendon sites must remain OUTSIDE the wrapping cylinder.
#
#   Entry and exit sites are constructed so their direct chord
#   intersects the cylinder.  MuJoCo must therefore use the geom
#   wrapping solution rather than silently reverting to a straight
#   site-to-site segment.
#
# NO:
#   H1 fitting
#   friction tuning
#   spring fitting
#   contact tuning
#   fixed tendon
#   PIP/DIP equality
#
# idx_flex_spatial remains one continuous MuJoCo <spatial> tendon.
# =====================================================================


MAIN_TENDON = "idx_flex_spatial"

JOINTS = {
    "mcp":
        "Index_Proximal--palm_assembly:1",

    "pip":
        "Index_Middle--Index_Proximal",

    "dip":
        "Index_Distal--Index_Middle",
}


CROSSINGS = {
    "mcp":
        (
            "v35_p02_mcp_parent",
            "v35_p03_mcp_child",
        ),

    "pip":
        (
            "v35_p04_pip_parent",
            "v35_p05_pip_child",
        ),

    "dip":
        (
            "v35_p06_dip_parent",
            "v35_p07_dip_child",
        ),
}


# Low-angle RH8D CAD moment arms already measured before the
# large-flexion route becomes invalid.
PULLEY_RADIUS = {
    "mcp": 0.00477625,
    "pip": 0.004495583,
    "dip": 0.004925074,
}


# Routing sites must be OUTSIDE pulley.
OUTSIDE_CLEARANCE = 0.00150


# Two outside sites are placed +/- this angular amount about
# the flexor side.
#
# With Rsite = R + 1.5 mm and 50 deg:
#
#     closest chord radius = Rsite*cos(50deg) < R
#
# so the direct chord intersects the pulley.
#
# This is a topology choice, not an H1 fitted parameter.
SITE_ANGLE_DEG = 50.0


GUIDE_HALF_LENGTH = 0.012


# Sidesite is also definitely outside pulley.
SIDE_CLEARANCE = 0.0030


MEASURED_POSES = [
    (100,   0.000,  0.000,  0.000),
    (500,  16.952,  0.968, -0.428),
    (1000, 39.472, -0.907, -0.343),
    (1500, 57.291,  3.654, -4.019),
    (2000, 73.632,  4.667,  2.012),
    (2500, 73.632, 27.119,  3.190),
    (3000, 77.347, 48.581,  7.810),
    (3500, 80.074, 70.872, -7.199),
]


TEST_ANGLES = [
    0,
    5,
    15,
    30,
    45,
    60,
    75,
    85,
]


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


def project_perp(
    v,
    axis,
):

    v = np.asarray(
        v,
        dtype=float,
    )

    axis = np.asarray(
        axis,
        dtype=float,
    )

    return (
        v
        -
        axis
        *
        float(
            np.dot(
                v,
                axis,
            )
        )
    )


def find_body_xml(
    root,
    name,
):

    for body in root.iter(
        "body"
    ):

        if body.get(
            "name"
        ) == name:

            return body

    raise RuntimeError(
        f"Body not found: {name}"
    )


def find_main_spatial(
    root,
):

    tendon_root = root.find(
        "./tendon"
    )

    if tendon_root is None:

        raise RuntimeError(
            "Top-level <tendon> missing."
        )

    for spatial in tendon_root.findall(
        "spatial"
    ):

        if spatial.get(
            "name"
        ) == MAIN_TENDON:

            return spatial

    raise RuntimeError(
        "idx_flex_spatial missing."
    )


def local_from_world(
    body_pos,
    body_R,
    point,
):

    return (
        body_R.T
        @
        (
            point
            -
            body_pos
        )
    )


def replace_crossing(
    spatial,
    old_parent,
    old_child,
    new_entry,
    guide,
    sidesite,
    new_exit,
):

    children = list(
        spatial
    )

    ip = None
    ic = None

    for i, elem in enumerate(
        children
    ):

        if (
            elem.tag == "site"
            and
            elem.get("site")
            ==
            old_parent
        ):

            ip = i

        if (
            elem.tag == "site"
            and
            elem.get("site")
            ==
            old_child
        ):

            ic = i

    if (
        ip is None
        or
        ic is None
    ):

        raise RuntimeError(
            f"Could not locate crossing "
            f"{old_parent} -> {old_child}"
        )

    if ic != ip + 1:

        raise RuntimeError(
            f"Crossing {old_parent} -> {old_child} "
            "is not consecutive."
        )

    # Remove old parent/child path elements.
    spatial.remove(
        children[
            ic
        ]
    )

    spatial.remove(
        children[
            ip
        ]
    )

    # Insert:
    #
    # entry -> wrapped geom -> exit
    #
    spatial.insert(
        ip,
        ET.Element(
            "site",
            {
                "site":
                    new_entry,
            },
        ),
    )

    spatial.insert(
        ip + 1,
        ET.Element(
            "geom",
            {
                "geom":
                    guide,

                "sidesite":
                    sidesite,
            },
        ),
    )

    spatial.insert(
        ip + 2,
        ET.Element(
            "site",
            {
                "site":
                    new_exit,
            },
        ),
    )


def joint_geometry(
    model,
    data,
    jid,
):

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

    Rc = np.asarray(
        data.xmat[
            child_bid
        ],
        dtype=float,
    ).reshape(
        3,
        3,
    )

    Rp = np.asarray(
        data.xmat[
            parent_bid
        ],
        dtype=float,
    ).reshape(
        3,
        3,
    )

    pc = np.asarray(
        data.xpos[
            child_bid
        ],
        dtype=float,
    )

    pp = np.asarray(
        data.xpos[
            parent_bid
        ],
        dtype=float,
    )

    joint_local = np.asarray(
        model.jnt_pos[
            jid
        ],
        dtype=float,
    )

    axis_local = unit(
        np.asarray(
            model.jnt_axis[
                jid
            ],
            dtype=float,
        )
    )

    center = (
        pc
        +
        Rc
        @
        joint_local
    )

    axis = unit(
        Rc
        @
        axis_local
    )

    return {
        "child_bid":
            child_bid,

        "parent_bid":
            parent_bid,

        "Rc":
            Rc,

        "Rp":
            Rp,

        "pc":
            pc,

        "pp":
            pp,

        "joint_local":
            joint_local,

        "axis_local":
            axis_local,

        "center":
            center,

        "axis":
            axis,
    }


def original_flexor_side(
    model,
    data,
    key,
):

    parent_name, child_name = (
        CROSSINGS[
            key
        ]
    )

    psid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        parent_name,
    )

    csid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        child_name,
    )

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        JOINTS[
            key
        ],
    )

    info = joint_geometry(
        model,
        data,
        jid,
    )

    p = np.asarray(
        data.site_xpos[
            psid
        ],
        dtype=float,
    )

    c = np.asarray(
        data.site_xpos[
            csid
        ],
        dtype=float,
    )

    # Use both existing CAD endpoints.
    rp = project_perp(
        p
        -
        info[
            "center"
        ],
        info[
            "axis"
        ],
    )

    rc = project_perp(
        c
        -
        info[
            "center"
        ],
        info[
            "axis"
        ],
    )

    # Align their directions before averaging.
    if (
        np.dot(
            rp,
            rc,
        )
        <
        0
    ):

        rc = -rc

    candidate = (
        rp
        +
        rc
    )

    if (
        np.linalg.norm(
            candidate
        )
        <
        1e-8
    ):

        candidate = rp

    return unit(
        candidate
    )


def add_joint_pulley(
    tree,
    model,
    key,
    side_sign,
    orientation_sign,
):

    root = tree.getroot()

    data = mujoco.MjData(
        model
    )

    data.qpos[:] = (
        model.qpos0
    )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        JOINTS[
            key
        ],
    )

    info = joint_geometry(
        model,
        data,
        jid,
    )

    side = (
        float(
            side_sign
        )
        *
        original_flexor_side(
            model,
            data,
            key,
        )
    )

    tangent = unit(
        np.cross(
            info[
                "axis"
            ],
            side,
        )
    )

    R = float(
        PULLEY_RADIUS[
            key
        ]
    )

    Rsite = (
        R
        +
        OUTSIDE_CLEARANCE
    )

    alpha = math.radians(
        SITE_ANGLE_DEG
    )

    # Two points outside the circle.
    #
    # Their direct chord crosses through the pulley.
    entry_direction = unit(
        math.cos(
            alpha
        )
        *
        side
        +
        float(
            orientation_sign
        )
        *
        math.sin(
            alpha
        )
        *
        tangent
    )

    exit_direction = unit(
        math.cos(
            alpha
        )
        *
        side
        -
        float(
            orientation_sign
        )
        *
        math.sin(
            alpha
        )
        *
        tangent
    )

    entry_world = (
        info[
            "center"
        ]
        +
        Rsite
        *
        entry_direction
    )

    exit_world = (
        info[
            "center"
        ]
        +
        Rsite
        *
        exit_direction
    )

    sidesite_world = (
        info[
            "center"
        ]
        +
        (
            R
            +
            SIDE_CLEARANCE
        )
        *
        side
    )

    entry_local = local_from_world(
        info[
            "pp"
        ],
        info[
            "Rp"
        ],
        entry_world,
    )

    exit_local = local_from_world(
        info[
            "pc"
        ],
        info[
            "Rc"
        ],
        exit_world,
    )

    side_local = local_from_world(
        info[
            "pc"
        ],
        info[
            "Rc"
        ],
        sidesite_world,
    )

    parent_body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        info[
            "parent_bid"
        ],
    )

    child_body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        info[
            "child_bid"
        ],
    )

    parent_body = find_body_xml(
        root,
        parent_body_name,
    )

    child_body = find_body_xml(
        root,
        child_body_name,
    )

    entry_name = (
        f"g38c_{key}_entry"
    )

    exit_name = (
        f"g38c_{key}_exit"
    )

    sidesite_name = (
        f"g38c_{key}_side"
    )

    guide_name = (
        f"g38c_{key}_pulley"
    )

    ET.SubElement(
        parent_body,
        "site",
        {
            "name":
                entry_name,

            "pos":
                " ".join(
                    f"{x:.12g}"
                    for x in entry_local
                ),

            "size":
                "0.00035",

            "rgba":
                "1 0.5 0 0",

            "group":
                "4",
        },
    )

    ET.SubElement(
        child_body,
        "site",
        {
            "name":
                exit_name,

            "pos":
                " ".join(
                    f"{x:.12g}"
                    for x in exit_local
                ),

            "size":
                "0.00035",

            "rgba":
                "1 0.5 0 0",

            "group":
                "4",
        },
    )

    ET.SubElement(
        child_body,
        "site",
        {
            "name":
                sidesite_name,

            "pos":
                " ".join(
                    f"{x:.12g}"
                    for x in side_local
                ),

            "size":
                "0.00035",

            "rgba":
                "1 1 0 0",

            "group":
                "4",
        },
    )

    p1 = (
        info[
            "joint_local"
        ]
        -
        GUIDE_HALF_LENGTH
        *
        info[
            "axis_local"
        ]
    )

    p2 = (
        info[
            "joint_local"
        ]
        +
        GUIDE_HALF_LENGTH
        *
        info[
            "axis_local"
        ]
    )

    ET.SubElement(
        child_body,
        "geom",
        {
            "name":
                guide_name,

            "type":
                "cylinder",

            "fromto":
                " ".join(
                    f"{x:.12g}"
                    for x in np.concatenate(
                        [
                            p1,
                            p2,
                        ]
                    )
                ),

            "size":
                f"{R:.12g}",

            # Very faint blue for debugging.
            "rgba":
                "0.1 0.5 1 0.08",

            "contype":
                "0",

            "conaffinity":
                "0",

            "group":
                "2",
        },
    )

    spatial = find_main_spatial(
        root
    )

    old_parent, old_child = (
        CROSSINGS[
            key
        ]
    )

    replace_crossing(
        spatial,
        old_parent,
        old_child,
        entry_name,
        guide_name,
        sidesite_name,
        exit_name,
    )

    # Pure geometric sanity.
    #
    # Both endpoints must be outside.
    #
    # Their zero-pose direct chord must cut through pulley.
    endpoint_radius = Rsite

    chord_radius = (
        Rsite
        *
        abs(
            math.cos(
                alpha
            )
        )
    )

    if not (
        endpoint_radius
        >
        R
        and
        chord_radius
        <
        R
    ):

        raise RuntimeError(
            f"{key.upper()} pulley topology is not geometrically "
            "forced to wrap."
        )

    return {
        "R":
            R,

        "Rsite":
            Rsite,

        "chord_radius":
            chord_radius,

        "side_sign":
            side_sign,

        "orientation_sign":
            orientation_sign,
    }


def evaluate_joint(
    xml,
    key,
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
        MAIN_TENDON,
    )

    values = []

    for angle in TEST_ANGLES:

        data.qpos[:] = (
            model.qpos0
        )

        data.qvel[:] = 0.0

        data.qpos[
            refs[
                key
            ][
                "qadr"
            ]
        ] = math.radians(
            angle
        )

        mujoco.mj_forward(
            model,
            data,
        )

        J = g38a.tendon_jacobian_row(
            model,
            data,
            tid,
        )

        values.append(
            float(
                J[
                    refs[
                        key
                    ][
                        "dadr"
                    ]
                ]
            )
        )

    target = (
        -
        abs(
            PULLEY_RADIUS[
                key
            ]
        )
    )

    sign_failures = sum(
        1
        for value in values
        if value >= 0.0
    )

    leverage_failures = sum(
        1
        for value in values
        if abs(
            value
        )
        <
        0.003
    )

    target_error = float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(
                        values
                    )
                    -
                    target
                )
                **
                2
            )
        )
    )

    # Strongly prioritize topology validity.
    score = (
        1000.0
        *
        sign_failures
        +
        100.0
        *
        leverage_failures
        +
        1000.0
        *
        target_error
    )

    return {
        "values":
            values,

        "sign_failures":
            sign_failures,

        "leverage_failures":
            leverage_failures,

        "target_error":
            target_error,

        "score":
            score,
    }


def select_branch(
    current_tree,
    current_xml,
    key,
    out,
):

    model = mujoco.MjModel.from_xml_path(
        str(
            current_xml
        )
    )

    candidates = []

    print()
    print(
        f"{key.upper()} BRANCH TEST"
    )

    print(
        " side  orient    "
        "J0      J30      J60      J85    "
        "signFail  leverageFail"
    )

    print(
        "-" * 100
    )

    for side_sign in [
        +1,
        -1,
    ]:

        for orientation_sign in [
            +1,
            -1,
        ]:

            tree = copy.deepcopy(
                current_tree
            )

            geometry = add_joint_pulley(
                tree,
                model,
                key,
                side_sign,
                orientation_sign,
            )

            temp = (
                out
                /
                (
                    f"g38c_{key}_"
                    f"s{side_sign:+d}_"
                    f"o{orientation_sign:+d}.xml"
                )
            )

            tree.write(
                temp,
                encoding="utf-8",
                xml_declaration=True,
            )

            try:

                result = evaluate_joint(
                    temp,
                    key,
                )

            except Exception as exc:

                print(
                    f" {side_sign:+d}      "
                    f"{orientation_sign:+d}      "
                    f"COMPILE/EVAL FAIL: {exc}"
                )

                continue

            vals = result[
                "values"
            ]

            print(
                f" {side_sign:+d}      "
                f"{orientation_sign:+d}     "
                f"{1000*vals[0]:+7.3f} "
                f"{1000*vals[3]:+7.3f} "
                f"{1000*vals[5]:+7.3f} "
                f"{1000*vals[7]:+7.3f}       "
                f"{result['sign_failures']:2d}          "
                f"{result['leverage_failures']:2d}"
            )

            candidates.append(
                (
                    result[
                        "score"
                    ],
                    tree,
                    temp,
                    geometry,
                    result,
                )
            )

    if not candidates:

        raise RuntimeError(
            f"No valid branch compiled for {key.upper()}"
        )

    candidates.sort(
        key=lambda item:
            item[
                0
            ]
    )

    (
        score,
        best_tree,
        best_xml,
        geometry,
        result,
    ) = candidates[
        0
    ]

    print()
    print(
        f"selected {key.upper()}: "
        f"side={geometry['side_sign']:+d}, "
        f"orientation={geometry['orientation_sign']:+d}"
    )

    print(
        f"  pulley radius       = "
        f"{1000*geometry['R']:.4f} mm"
    )

    print(
        f"  site radius         = "
        f"{1000*geometry['Rsite']:.4f} mm"
    )

    print(
        f"  unwrapped chord min = "
        f"{1000*geometry['chord_radius']:.4f} mm"
    )

    print(
        f"  sign failures       = "
        f"{result['sign_failures']}"
    )

    print(
        f"  leverage failures   = "
        f"{result['leverage_failures']}"
    )

    return (
        best_tree,
        best_xml,
        result,
    )


def preserve_open_extension(
    original_xml,
    candidate_xml,
    final_xml,
):

    old_model = mujoco.MjModel.from_xml_path(
        str(
            original_xml
        )
    )

    old_data = mujoco.MjData(
        old_model
    )

    old_data.qpos[:] = (
        old_model.qpos0
    )

    mujoco.mj_forward(
        old_model,
        old_data,
    )

    old_tid = mujoco.mj_name2id(
        old_model,
        mujoco.mjtObj.mjOBJ_TENDON,
        MAIN_TENDON,
    )

    old_length = float(
        old_data.ten_length[
            old_tid
        ]
    )

    old_ref = (
        g12.tendon_spring_reference(
            old_model,
            old_tid,
        )
    )

    new_model = mujoco.MjModel.from_xml_path(
        str(
            candidate_xml
        )
    )

    new_data = mujoco.MjData(
        new_model
    )

    new_data.qpos[:] = (
        new_model.qpos0
    )

    mujoco.mj_forward(
        new_model,
        new_data,
    )

    new_tid = mujoco.mj_name2id(
        new_model,
        mujoco.mjtObj.mjOBJ_TENDON,
        MAIN_TENDON,
    )

    new_length = float(
        new_data.ten_length[
            new_tid
        ]
    )

    delta = (
        new_length
        -
        old_length
    )

    new_ref = (
        old_ref
        +
        delta
    )

    tree = ET.parse(
        candidate_xml
    )

    spatial = find_main_spatial(
        tree.getroot()
    )

    spatial.set(
        "springlength",
        f"{new_ref:.12g}",
    )

    tree.write(
        final_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    mujoco.MjModel.from_xml_path(
        str(
            final_xml
        )
    )

    print()
    print(
        "OPEN-STATE EXTENSION PRESERVATION"
    )

    print(
        f"  old length = "
        f"{1000*old_length:.6f} mm"
    )

    print(
        f"  new length = "
        f"{1000*new_length:.6f} mm"
    )

    print(
        f"  path delta = "
        f"{1000*delta:+.6f} mm"
    )


def full_audit(
    xml,
    out,
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
        MAIN_TENDON,
    )

    rows = []

    overall_pass = True

    print()
    print("=" * 150)
    print(
        "G38C STEP 2 — FULL 0–85 DEG JOINT AUDIT"
    )
    print("=" * 150)

    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        values = []

        print()
        print(
            key.upper()
        )

        print(
            " angle       J [mm/rad]"
        )

        print(
            " ----------------------"
        )

        for angle in range(
            0,
            86,
            5,
        ):

            data.qpos[:] = (
                model.qpos0
            )

            data.qvel[:] = 0.0

            data.qpos[
                refs[
                    key
                ][
                    "qadr"
                ]
            ] = math.radians(
                angle
            )

            mujoco.mj_forward(
                model,
                data,
            )

            J = g38a.tendon_jacobian_row(
                model,
                data,
                tid,
            )

            value = float(
                J[
                    refs[
                        key
                    ][
                        "dadr"
                    ]
                ]
            )

            values.append(
                value
            )

            print(
                f" {angle:5.1f}      "
                f"{1000*value:+9.4f}"
            )

            rows.append({
                "test":
                    f"{key}_single",

                "tick":
                    "",

                "mcp_deg":
                    angle
                    if key == "mcp"
                    else 0,

                "pip_deg":
                    angle
                    if key == "pip"
                    else 0,

                "dip_deg":
                    angle
                    if key == "dip"
                    else 0,

                "j_mcp_mm_rad":
                    1000
                    *
                    J[
                        refs[
                            "mcp"
                        ][
                            "dadr"
                        ]
                    ],

                "j_pip_mm_rad":
                    1000
                    *
                    J[
                        refs[
                            "pip"
                        ][
                            "dadr"
                        ]
                    ],

                "j_dip_mm_rad":
                    1000
                    *
                    J[
                        refs[
                            "dip"
                        ][
                            "dadr"
                        ]
                    ],
            })

        same_sign = all(
            v < 0.0
            for v in values
        )

        min_abs = min(
            abs(
                v
            )
            for v in values
        )

        useful = (
            min_abs
            >=
            0.003
        )

        print()
        print(
            f"  all flexion sign = "
            f"{same_sign}"
        )

        print(
            f"  min |J| = "
            f"{1000*min_abs:.4f} mm/rad"
        )

        if not (
            same_sign
            and
            useful
        ):

            overall_pass = False

    # =============================================================
    # Real video-13 configurations.
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G38C STEP 3 — VIDEO-13 CONFIGURATION AUDIT"
    )
    print("=" * 150)

    print()
    print(
        "tick     MCP     PIP     DIP      "
        "J_MCP      J_PIP      J_DIP"
    )

    print("-" * 110)

    for (
        tick,
        mcp,
        pip,
        dip,
    ) in MEASURED_POSES:

        pose = {
            "mcp":
                max(
                    0.0,
                    mcp,
                ),

            "pip":
                max(
                    0.0,
                    pip,
                ),

            "dip":
                max(
                    0.0,
                    dip,
                ),
        }

        data.qpos[:] = (
            model.qpos0
        )

        data.qvel[:] = 0.0

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
                pose[
                    key
                ]
            )

        mujoco.mj_forward(
            model,
            data,
        )

        J = g38a.tendon_jacobian_row(
            model,
            data,
            tid,
        )

        jm = float(
            J[
                refs[
                    "mcp"
                ][
                    "dadr"
                ]
            ]
        )

        jp = float(
            J[
                refs[
                    "pip"
                ][
                    "dadr"
                ]
            ]
        )

        jd = float(
            J[
                refs[
                    "dip"
                ][
                    "dadr"
                ]
            ]
        )

        print(
            f"{tick:4d}  "
            f"{pose['mcp']:7.2f} "
            f"{pose['pip']:7.2f} "
            f"{pose['dip']:7.2f}   "
            f"{1000*jm:+9.4f} "
            f"{1000*jp:+9.4f} "
            f"{1000*jd:+9.4f}"
        )

        rows.append({
            "test":
                "video13",

            "tick":
                tick,

            "mcp_deg":
                pose[
                    "mcp"
                ],

            "pip_deg":
                pose[
                    "pip"
                ],

            "dip_deg":
                pose[
                    "dip"
                ],

            "j_mcp_mm_rad":
                1000
                *
                jm,

            "j_pip_mm_rad":
                1000
                *
                jp,

            "j_dip_mm_rad":
                1000
                *
                jd,
        })

        if not (
            jm < -0.003
            and
            jp < -0.003
            and
            jd < -0.003
        ):

            overall_pass = False

    path = (
        out
        /
        "g38c_moment_arm_audit.csv"
    )

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test",
                "tick",
                "mcp_deg",
                "pip_deg",
                "dip_deg",
                "j_mcp_mm_rad",
                "j_pip_mm_rad",
                "j_dip_mm_rad",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print("=" * 150)
    print(
        "G38C DECISION"
    )
    print("=" * 150)

    if overall_pass:

        print(
            "PASS: THE CONTINUOUS RH8D SPATIAL FLEXOR NOW HAS "
            "PERSISTENT FLEXION LEVERAGE OVER THE COMPLETE RANGE."
        )

        print()
        print(
            "FREEZE TRANSMISSION GEOMETRY."
        )

        print(
            "Next = identify MCP -> PIP -> DIP passive recruitment "
            "from free-space video 13."
        )

    else:

        print(
            "FAIL: even a geometrically valid outside-site joint "
            "pulley does not reproduce a persistent RH8D flexor."
        )

        print()
        print(
            "STOP ALL HINGE-PULLEY WORK."
        )

        print(
            "Next = ONE Aero-style multi-guide channel using "
            "RH8D CAD waypoints."
        )

    print()
    print(
        f"Audit CSV:\n  "
        f"{path}"
    )

    return overall_pass


def gui(
    xml,
):

    try:

        import mujoco.viewer

    except Exception as exc:

        print(
            f"GUI unavailable: {exc}"
        )

        return

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

    # Show the demanding video-13 configuration.
    # ---------------------------------------------------------
    # ALWAYS stage the complete robot at the real tabletop
    # experiment configuration before changing the index finger.
    #
    # Never use raw model.qpos0 for RH8D tabletop experiments.
    # ---------------------------------------------------------
    from rh8d_tabletop_stage_v1 import (
        apply_tabletop_stage,
        verify_tabletop_scene,
    )

    apply_tabletop_stage(
        model,
        data,
        reset_to_qpos0=True,
    )

    verify_tabletop_scene(
        model,
        data,
    )

    data.qpos[
        refs[
            "mcp"
        ][
            "qadr"
        ]
    ] = math.radians(
        77.347
    )

    data.qpos[
        refs[
            "pip"
        ][
            "qadr"
        ]
    ] = math.radians(
        48.581
    )

    data.qpos[
        refs[
            "dip"
        ][
            "qadr"
        ]
    ] = math.radians(
        7.810
    )

    mujoco.mj_forward(
        model,
        data,
    )

    print()
    print("=" * 150)
    print(
        "G38C GUI — HIGH FLEXION"
    )
    print("=" * 150)

    print(
        "MAGENTA = real continuous idx_flex_spatial."
    )

    print(
        "Faint blue cylinders are numerical cable guides only."
    )

    print(
        "GUI remains open until you close it."
    )

    mujoco.viewer.launch(
        model,
        data,
    )


def main():

    parser = argparse.ArgumentParser()

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
            "v35g38c_valid_joint_pulley"
        ),
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
    )

    args = parser.parse_args()

    source = (
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

    # Flatten clean source.
    source_model = mujoco.MjModel.from_xml_path(
        str(
            source
        )
    )

    flat = (
        out
        /
        "g38c_flat_source.xml"
    )

    mujoco.mj_saveLastXML(
        str(
            flat
        ),
        source_model,
    )

    current_xml = flat

    current_tree = ET.parse(
        current_xml
    )

    print()
    print("=" * 150)
    print(
        "G38C — VALID OUTSIDE-SITE JOINT PULLEYS"
    )
    print("=" * 150)

    print()
    print(
        "The main index tendon remains one continuous "
        "MuJoCo <spatial> tendon."
    )

    print()
    print("=" * 150)
    print(
        "G38C STEP 1 — SELECT PHYSICAL WRAP BRANCH"
    )
    print("=" * 150)

    # Sequential topology construction only.
    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        (
            selected_tree,
            selected_xml,
            result,
        ) = select_branch(
            current_tree,
            current_xml,
            key,
            out,
        )

        current_tree = (
            selected_tree
        )

        current_xml = (
            out
            /
            f"g38c_selected_through_{key}.xml"
        )

        current_tree.write(
            current_xml,
            encoding="utf-8",
            xml_declaration=True,
        )

        mujoco.MjModel.from_xml_path(
            str(
                current_xml
            )
        )

    final_xml = (
        out
        /
        "g38c_valid_joint_pulley_flexor.xml"
    )

    preserve_open_extension(
        flat,
        current_xml,
        final_xml,
    )

    passed = full_audit(
        final_xml,
        out,
    )

    print()
    print(
        f"Candidate XML:\n  "
        f"{final_xml}"
    )

    print()
    print(
        f"All outputs:\n  "
        f"{out}"
    )

    if not args.no_gui:

        gui(
            final_xml
        )

    if not passed:

        raise SystemExit(2)


if __name__ == "__main__":
    main()
