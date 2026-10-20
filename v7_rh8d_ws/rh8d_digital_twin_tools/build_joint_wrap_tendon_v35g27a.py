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
# G27A — PHYSICAL JOINT-AXIS TENDON WRAPPING
#
# G26C established:
#
#   MCP:
#       J ~ -5 mm/rad at low angle
#       -> reverses to +4.6...+5.0 mm/rad near ~74 deg
#
#   PIP:
#       J ~ -4.5...-5.3 mm/rad
#       -> reverses near ~71 deg
#
#   DIP:
#       remains ~ -4.9 mm/rad
#
# Therefore the straight site-to-site spatial tendon is not a
# mechanically valid full-range transmission.
#
# G27A:
#
#   1. Detects which tendon site-pair crosses MCP/PIP/DIP.
#   2. Adds one non-colliding CYLINDER centered on each joint axis.
#   3. Adds a sidesite on the currently used tendon side.
#   4. Inserts:
#
#          site -> cylinder-wrap -> site
#
#      into the SAME continuous spatial tendon.
#
#   5. Radius comes from existing OPEN configuration geometry:
#
#          max(
#              current |moment arm|,
#              distance(segment, joint axis) + 0.25 mm
#          )
#
#      No radius scan.
#
#   6. Adjusts springlength only by the change in reference tendon
#      path length, preserving the existing open-state preload.
#
#   7. Audits moment arms through the measured full range.
#   8. Runs ONE H1 dynamic validation.
#
# NO:
#   joint locking
#   contact switching
#   artificial DIP actuator
#   contact friction tuning
#   spring tuning
#   parameter scan
# =====================================================================


TENDON_NAME = "idx_flex_spatial"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

JOINT_NAMES = {
    "mcp": MCP_NAME,
    "pip": PIP_NAME,
    "dip": DIP_NAME,
}


# Full-range measured real hardware configurations.
FREE_SPACE = [
    # tick,   MCP,      PIP,      DIP
    (100.0,     0.000,    0.000,    0.000),
    (500.0,    16.952,    0.968,   -0.428),
    (1000.0,   39.472,   -0.907,   -0.343),
    (1500.0,   57.291,    3.654,   -4.019),
    (2000.0,   73.632,    4.667,    2.012),
    (2500.0,   73.632,   27.119,    3.190),
    (3000.0,   77.347,   48.581,    7.810),
    (3500.0,   80.074,   70.872,   -7.199),
]


H1_HW_ENCODER = 673.573

H1_HW_CONTACT = np.array(
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
        raw.shape
        ==
        (
            model.ntendon,
            model.nv,
        )
    ):

        return (
            raw[
                tid,
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
                tid,
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
            tid
        ]
    )

    adr = int(
        rowadr[
            tid
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


def unit(
    x,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    n = float(
        np.linalg.norm(
            x
        )
    )

    if n < 1e-12:

        raise RuntimeError(
            "Cannot normalize zero vector."
        )

    return (
        x / n
    )


def is_descendant(
    model,
    body_id,
    ancestor_id,
):

    b = int(
        body_id
    )

    ancestor_id = int(
        ancestor_id
    )

    while True:

        if b == ancestor_id:
            return True

        if b == 0:
            return False

        b = int(
            model.body_parentid[
                b
            ]
        )


def segment_axis_distance(
    p0,
    p1,
    axis_point,
    axis_direction,
):

    p0 = np.asarray(
        p0,
        dtype=float,
    )

    p1 = np.asarray(
        p1,
        dtype=float,
    )

    c = np.asarray(
        axis_point,
        dtype=float,
    )

    a = unit(
        axis_direction
    )

    d = (
        p1
        -
        p0
    )

    r = (
        p0
        -
        c
    )

    # Projection perpendicular to hinge axis.
    d_perp = (
        d
        -
        a
        *
        np.dot(
            d,
            a,
        )
    )

    r_perp = (
        r
        -
        a
        *
        np.dot(
            r,
            a,
        )
    )

    denominator = float(
        np.dot(
            d_perp,
            d_perp,
        )
    )

    if denominator < 1e-14:

        s = 0.5

    else:

        s = float(
            -
            np.dot(
                r_perp,
                d_perp,
            )
            /
            denominator
        )

        s = float(
            np.clip(
                s,
                0.0,
                1.0,
            )
        )

    closest_segment = (
        p0
        +
        s
        *
        d
    )

    r2 = (
        closest_segment
        -
        c
    )

    r2_perp = (
        r2
        -
        a
        *
        np.dot(
            r2,
            a,
        )
    )

    distance = float(
        np.linalg.norm(
            r2_perp
        )
    )

    return (
        distance,
        closest_segment,
    )


def quat_z_to_axis(
    axis,
):

    axis = unit(
        axis
    )

    z = np.array(
        [
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    dot = float(
        np.dot(
            z,
            axis,
        )
    )

    if dot > 0.999999:

        return np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

    if dot < -0.999999:

        return np.array(
            [
                0.0,
                1.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

    cross = np.cross(
        z,
        axis,
    )

    q = np.array(
        [
            1.0 + dot,
            cross[0],
            cross[1],
            cross[2],
        ],
        dtype=float,
    )

    return unit(
        q
    )


def find_body_element(
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
        f"Body XML not found: {name}"
    )


def find_spatial_element(
    root,
    name,
):

    for tendon_root in root.iter(
        "tendon"
    ):

        for spatial in tendon_root.findall(
            "spatial"
        ):

            if spatial.get(
                "name"
            ) == name:

                return spatial

    raise RuntimeError(
        f"Spatial tendon XML not found: {name}"
    )


def parse_springlength(
    spatial,
):

    raw = spatial.get(
        "springlength"
    )

    if raw is None:

        return None

    values = [
        float(x)
        for x in raw.split()
    ]

    return values


def set_shifted_springlength(
    spatial,
    old_values,
    delta,
):

    if old_values is None:

        print()
        print(
            "WARNING: tendon has no explicit springlength."
        )

        print(
            "Leaving springlength unchanged."
        )

        return

    adjusted = []

    for value in old_values:

        if value >= 0.0:

            adjusted.append(
                value
                +
                delta
            )

        else:

            adjusted.append(
                value
            )

    spatial.set(
        "springlength",
        " ".join(
            f"{x:.12g}"
            for x in adjusted
        ),
    )


def first_episode(
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
            "v35g27a_joint_wrap_tendon"
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

    # ==============================================================
    # STEP 1 — flatten XML.
    # ==============================================================

    print()
    print("=" * 150)
    print(
        "G27A STEP 1 — LOAD / FLATTEN SOURCE MODEL"
    )
    print("=" * 150)

    source_model = mujoco.MjModel.from_xml_path(
        str(
            source_xml
        )
    )

    flat_xml = (
        out
        /
        "g27a_flat_source.xml"
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

    source_data.qvel[:] = 0.0

    mujoco.mj_forward(
        source_model,
        source_data,
    )

    tendon_id = mujoco.mj_name2id(
        source_model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    if tendon_id < 0:

        raise RuntimeError(
            f"Tendon missing: {TENDON_NAME}"
        )

    source_J = tendon_jacobian_row(
        source_model,
        source_data,
        tendon_id,
    )

    source_open_length = float(
        source_data.ten_length[
            tendon_id
        ]
    )

    # ==============================================================
    # STEP 2 — inspect current site chain.
    # ==============================================================

    tree = ET.parse(
        flat_xml
    )

    root = tree.getroot()

    spatial = find_spatial_element(
        root,
        TENDON_NAME,
    )

    original_springlength = (
        parse_springlength(
            spatial
        )
    )

    children = list(
        spatial
    )

    if not children:

        raise RuntimeError(
            "Spatial tendon has no path elements."
        )

    non_site = [
        child
        for child in children
        if child.tag != "site"
    ]

    if non_site:

        print()
        print(
            "Existing tendon path already contains "
            "non-site elements:"
        )

        for child in non_site:

            print(
                f"  <{child.tag}> "
                f"{child.attrib}"
            )

        raise RuntimeError(
            "G27A expects the current tendon to be "
            "the existing site-only routing."
        )

    site_names = [
        child.get(
            "site"
        )
        for child in children
    ]

    if any(
        name is None
        for name in site_names
    ):

        raise RuntimeError(
            "Unnamed tendon site entry."
        )

    print()
    print(
        "CURRENT TENDON SITE CHAIN:"
    )

    for i, name in enumerate(
        site_names
    ):

        sid = mujoco.mj_name2id(
            source_model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )

        if sid < 0:

            raise RuntimeError(
                f"Site missing from model: {name}"
            )

        bid = int(
            source_model.site_bodyid[
                sid
            ]
        )

        body_name = mujoco.mj_id2name(
            source_model,
            mujoco.mjtObj.mjOBJ_BODY,
            bid,
        )

        print(
            f"  [{i:02d}] "
            f"{name:<45s} "
            f"body={body_name}"
        )

    # ==============================================================
    # STEP 3 — identify each joint crossing and derive wrap radius.
    # ==============================================================

    print()
    print("=" * 150)
    print(
        "G27A STEP 2 — IDENTIFY JOINT CROSSINGS"
    )
    print("=" * 150)

    insertions = {}

    wrap_report = []

    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        joint_name = (
            JOINT_NAMES[
                key
            ]
        )

        jid = mujoco.mj_name2id(
            source_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if jid < 0:

            raise RuntimeError(
                f"Joint missing: {joint_name}"
            )

        child_body = int(
            source_model.jnt_bodyid[
                jid
            ]
        )

        body_name = mujoco.mj_id2name(
            source_model,
            mujoco.mjtObj.mjOBJ_BODY,
            child_body,
        )

        crossing_pairs = []

        for i in range(
            len(
                site_names
            )
            -
            1
        ):

            sid0 = mujoco.mj_name2id(
                source_model,
                mujoco.mjtObj.mjOBJ_SITE,
                site_names[
                    i
                ],
            )

            sid1 = mujoco.mj_name2id(
                source_model,
                mujoco.mjtObj.mjOBJ_SITE,
                site_names[
                    i + 1
                ],
            )

            b0 = int(
                source_model.site_bodyid[
                    sid0
                ]
            )

            b1 = int(
                source_model.site_bodyid[
                    sid1
                ]
            )

            d0 = is_descendant(
                source_model,
                b0,
                child_body,
            )

            d1 = is_descendant(
                source_model,
                b1,
                child_body,
            )

            if d0 != d1:

                crossing_pairs.append(
                    (
                        i,
                        sid0,
                        sid1,
                    )
                )

        if len(
            crossing_pairs
        ) != 1:

            print()
            print(
                f"{key.upper()} crossing candidates:"
            )

            for pair in crossing_pairs:

                print(
                    f"  pair index {pair[0]}: "
                    f"{site_names[pair[0]]} -> "
                    f"{site_names[pair[0]+1]}"
                )

            raise RuntimeError(
                f"Expected exactly one tendon crossing for "
                f"{key.upper()}, got {len(crossing_pairs)}"
            )

        pair_index, sid0, sid1 = (
            crossing_pairs[
                0
            ]
        )

        # Avoid two joint wraps being inserted into the exact
        # same site-site interval. MuJoCo requires multiple
        # obstacle geoms to be separated by sites.
        if pair_index in insertions:

            raise RuntimeError(
                f"{key.upper()} shares tendon segment "
                f"{pair_index} with another joint. "
                "Need an intermediate physical via-site first."
            )

        body_R = np.asarray(
            source_data.xmat[
                child_body
            ],
            dtype=float,
        ).reshape(
            3,
            3,
        )

        body_pos = np.asarray(
            source_data.xpos[
                child_body
            ],
            dtype=float,
        )

        joint_local_pos = np.asarray(
            source_model.jnt_pos[
                jid
            ],
            dtype=float,
        )

        joint_local_axis = unit(
            np.asarray(
                source_model.jnt_axis[
                    jid
                ],
                dtype=float,
            )
        )

        joint_world = (
            body_pos
            +
            body_R
            @
            joint_local_pos
        )

        axis_world = unit(
            body_R
            @
            joint_local_axis
        )

        p0 = np.asarray(
            source_data.site_xpos[
                sid0
            ],
            dtype=float,
        )

        p1 = np.asarray(
            source_data.site_xpos[
                sid1
            ],
            dtype=float,
        )

        (
            segment_distance,
            closest_segment,
        ) = segment_axis_distance(
            p0,
            p1,
            joint_world,
            axis_world,
        )

        dadr = int(
            source_model.jnt_dofadr[
                jid
            ]
        )

        open_moment_arm = abs(
            float(
                source_J[
                    dadr
                ]
            )
        )

        # Make cylinder just large enough that the current
        # open tendon segment genuinely wraps.
        radius = max(
            open_moment_arm,
            segment_distance
            +
            0.00025,
        )

        if not (
            0.0025
            <=
            radius
            <=
            0.0080
        ):

            raise RuntimeError(
                f"{key.upper()} derived wrap radius "
                f"{1000*radius:.3f} mm is implausible. "
                "Refusing automatic patch."
            )

        midpoint = (
            0.5
            *
            (
                p0 + p1
            )
        )

        side_world = (
            midpoint
            -
            joint_world
        )

        side_world = (
            side_world
            -
            axis_world
            *
            np.dot(
                side_world,
                axis_world,
            )
        )

        if np.linalg.norm(
            side_world
        ) < 1e-8:

            side_world = (
                closest_segment
                -
                joint_world
            )

            side_world = (
                side_world
                -
                axis_world
                *
                np.dot(
                    side_world,
                    axis_world,
                )
            )

        if np.linalg.norm(
            side_world
        ) < 1e-8:

            trial = np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                ]
            )

            if abs(
                np.dot(
                    trial,
                    axis_world,
                )
            ) > 0.9:

                trial = np.array(
                    [
                        0.0,
                        1.0,
                        0.0,
                    ]
                )

            side_world = np.cross(
                axis_world,
                trial,
            )

        side_world = unit(
            side_world
        )

        side_local_direction = (
            body_R.T
            @
            side_world
        )

        side_local_direction = unit(
            side_local_direction
        )

        side_local_pos = (
            joint_local_pos
            +
            side_local_direction
            *
            (
                2.25
                *
                radius
            )
        )

        wrap_geom_name = (
            f"g27a_{key}_tendon_pulley"
        )

        side_site_name = (
            f"g27a_{key}_tendon_side"
        )

        body_element = find_body_element(
            root,
            body_name,
        )

        quat = quat_z_to_axis(
            joint_local_axis
        )

        ET.SubElement(
            body_element,
            "geom",
            {
                "name":
                    wrap_geom_name,

                "type":
                    "cylinder",

                "pos":
                    " ".join(
                        f"{x:.12g}"
                        for x in joint_local_pos
                    ),

                "quat":
                    " ".join(
                        f"{x:.12g}"
                        for x in quat
                    ),

                "size":
                    (
                        f"{radius:.12g} "
                        f"{0.008:.12g}"
                    ),

                "contype":
                    "0",

                "conaffinity":
                    "0",

                "group":
                    "5",

                "rgba":
                    "0.1 0.7 1 0.30",
            },
        )

        ET.SubElement(
            body_element,
            "site",
            {
                "name":
                    side_site_name,

                "pos":
                    " ".join(
                        f"{x:.12g}"
                        for x in side_local_pos
                    ),

                "size":
                    "0.001",

                "rgba":
                    "1 0.2 0.2 0.8",
            },
        )

        insertions[
            pair_index
        ] = {
            "geom":
                wrap_geom_name,

            "sidesite":
                side_site_name,
        }

        wrap_report.append({
            "joint":
                key,

            "joint_name":
                joint_name,

            "body":
                body_name,

            "site_before":
                site_names[
                    pair_index
                ],

            "site_after":
                site_names[
                    pair_index + 1
                ],

            "segment_axis_distance_mm":
                1000.0
                *
                segment_distance,

            "original_open_moment_arm_mm":
                1000.0
                *
                open_moment_arm,

            "wrap_radius_mm":
                1000.0
                *
                radius,
        })

        print()
        print(
            f"{key.upper()}"
        )

        print(
            f"  joint body       = "
            f"{body_name}"
        )

        print(
            f"  tendon segment   = "
            f"{site_names[pair_index]} "
            f"-> "
            f"{site_names[pair_index+1]}"
        )

        print(
            f"  segment-axis dist= "
            f"{1000.0*segment_distance:.3f} mm"
        )

        print(
            f"  original |J|     = "
            f"{1000.0*open_moment_arm:.3f} mm/rad"
        )

        print(
            f"  pulley radius    = "
            f"{1000.0*radius:.3f} mm"
        )

    # ==============================================================
    # STEP 4 — rebuild tendon path with wrap objects.
    # ==============================================================

    for child in list(
        spatial
    ):

        spatial.remove(
            child
        )

    for i, child in enumerate(
        children
    ):

        spatial.append(
            child
        )

        if i in insertions:

            info = insertions[
                i
            ]

            spatial.append(
                ET.Element(
                    "geom",
                    {
                        "geom":
                            info[
                                "geom"
                            ],

                        "sidesite":
                            info[
                                "sidesite"
                            ],
                    },
                )
            )

    preadjust_xml = (
        out
        /
        "g27a_wrap_preadjust.xml"
    )

    tree.write(
        preadjust_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    # Compile before changing springlength.
    pre_model = mujoco.MjModel.from_xml_path(
        str(
            preadjust_xml
        )
    )

    pre_data = mujoco.MjData(
        pre_model
    )

    pre_data.qpos[:] = (
        pre_model.qpos0
    )

    mujoco.mj_forward(
        pre_model,
        pre_data,
    )

    pre_tid = mujoco.mj_name2id(
        pre_model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    new_open_length = float(
        pre_data.ten_length[
            pre_tid
        ]
    )

    reference_delta = (
        new_open_length
        -
        source_open_length
    )

    print()
    print("=" * 150)
    print(
        "REFERENCE LENGTH PRESERVATION"
    )
    print("=" * 150)

    print(
        f"old open tendon length = "
        f"{1000.0*source_open_length:.6f} mm"
    )

    print(
        f"new open tendon length = "
        f"{1000.0*new_open_length:.6f} mm"
    )

    print(
        f"path-length delta       = "
        f"{1000.0*reference_delta:+.6f} mm"
    )

    # Re-open patched XML and shift explicit springlength by the
    # same path-length delta.
    tree2 = ET.parse(
        preadjust_xml
    )

    root2 = tree2.getroot()

    spatial2 = find_spatial_element(
        root2,
        TENDON_NAME,
    )

    set_shifted_springlength(
        spatial2,
        original_springlength,
        reference_delta,
    )

    candidate_xml = (
        out
        /
        "g27a_joint_wrap_candidate.xml"
    )

    tree2.write(
        candidate_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    candidate_model = mujoco.MjModel.from_xml_path(
        str(
            candidate_xml
        )
    )

    # Save wrap derivation.
    wrap_csv = (
        out
        /
        "g27a_wrap_geometry.csv"
    )

    with wrap_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                wrap_report[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            wrap_report
        )

    # ==============================================================
    # STEP 5 — FULL-RANGE STATIC MOMENT-ARM AUDIT.
    # ==============================================================

    print()
    print("=" * 150)
    print(
        "G27A STEP 3 — FULL-RANGE WRAPPED-TENDON AUDIT"
    )
    print("=" * 150)

    data = mujoco.MjData(
        candidate_model
    )

    refs = g12.get_joint_refs(
        candidate_model
    )

    tid = mujoco.mj_name2id(
        candidate_model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    rows = []

    print()
    print(
        "tick    MCP     PIP     DIP      "
        "J_MCP     J_PIP     J_DIP"
    )

    print("-" * 150)

    for (
        tick,
        mcp_deg,
        pip_deg,
        dip_deg,
    ) in FREE_SPACE:

        data.qpos[:] = (
            candidate_model.qpos0
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
            candidate_model,
            data,
        )

        J = tendon_jacobian_row(
            candidate_model,
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

        row = {
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
        }

        rows.append(
            row
        )

        print(
            f"{tick:4.0f} "
            f"{mcp_deg:7.2f} "
            f"{pip_deg:7.2f} "
            f"{dip_deg:7.2f} "
            f"{jm:+10.3f} "
            f"{jp:+10.3f} "
            f"{jd:+10.3f}"
        )

    audit_csv = (
        out
        /
        "g27a_full_range_moment_arms.csv"
    )

    with audit_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # --------------------------------------------------------------
    # Check against low-angle sign.
    # --------------------------------------------------------------

    routing_pass = True

    print()
    print("=" * 150)
    print(
        "WRAP ROUTING VALIDITY"
    )
    print("=" * 150)

    for joint in [
        "mcp",
        "pip",
        "dip",
    ]:

        key = (
            f"{joint}_J_mm_rad"
        )

        # Use 500-tick point as sign reference.
        reference = float(
            rows[
                1
            ][
                key
            ]
        )

        reference_sign = np.sign(
            reference
        )

        values = np.asarray(
            [
                r[
                    key
                ]
                for r in rows[
                    1:
                ]
            ],
            dtype=float,
        )

        sign_flip = np.any(
            (
                np.sign(
                    values
                )
                !=
                reference_sign
            )
            &
            (
                np.abs(
                    values
                )
                >
                0.25
            )
        )

        minimum_abs = float(
            np.min(
                np.abs(
                    values
                )
            )
        )

        print()
        print(
            joint.upper()
        )

        print(
            f"  sign reference = "
            f"{reference_sign:+.0f}"
        )

        print(
            f"  min |J|        = "
            f"{minimum_abs:.3f} mm/rad"
        )

        print(
            f"  sign reversal  = "
            f"{bool(sign_flip)}"
        )

        if sign_flip:

            routing_pass = False

        if minimum_abs < 2.5:

            routing_pass = False

    print()
    print(
        f"FULL-RANGE ROUTING PASS = "
        f"{routing_pass}"
    )

    if not routing_pass:

        print()
        print(
            "STOP:"
        )

        print(
            "automatic joint-centered wrap did not produce "
            "a valid full-range routing."
        )

        print(
            "Do not run identification on this candidate."
        )

        print()
        print(
            f"Candidate kept for inspection:\n  "
            f"{candidate_xml}"
        )

        return

    # ==============================================================
    # STEP 6 — ONE H1 DYNAMIC RUN, BASELINE PARAMETERS.
    # ==============================================================

    print()
    print("=" * 150)
    print(
        "G27A STEP 4 — ONE H1 DYNAMIC RUN WITH WRAPPED TENDON"
    )
    print("=" * 150)

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    (
        h1_rows,
        qpos,
        times,
        summary,
    ) = g12.run_audit(
        candidate_xml,
        bag,
        replay,
    )

    middle_starts, middle_ends = (
        first_episode(
            h1_rows,
            "mid_contact",
        )
    )

    tip_starts, tip_ends = (
        first_episode(
            h1_rows,
            "tip_contact",
        )
    )

    if not middle_starts:

        raise RuntimeError(
            "Wrapped candidate produced no middle contact."
        )

    first = int(
        middle_starts[
            0
        ]
    )

    first_end = int(
        middle_ends[
            0
        ]
    )

    first_row = (
        h1_rows[
            first
        ]
    )

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

    contact_pose_error = (
        first_pose
        -
        H1_HW_CONTACT
    )

    contact_pose_rmse = float(
        np.sqrt(
            np.mean(
                contact_pose_error
                **
                2
            )
        )
    )

    first_duration = (
        h1_rows[
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
        middle_starts
    ) >= 2:

        detached_gap = (
            h1_rows[
                middle_starts[
                    1
                ]
            ][
                "time_s"
            ]
            -
            h1_rows[
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
        f"{H1_HW_ENCODER:.3f}"
    )

    print(
        f"  encoder error      = "
        f"{first_row['encoder_tick'] - H1_HW_ENCODER:+.3f}"
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
        f"{H1_HW_CONTACT[0]:.3f} / "
        f"{H1_HW_CONTACT[1]:.3f} / "
        f"{H1_HW_CONTACT[2]:.3f}"
    )

    print(
        f"  pose RMSE       = "
        f"{contact_pose_rmse:.3f} deg"
    )

    print()
    print(
        "CONTACT CONTINUITY"
    )

    print(
        f"  middle episodes = "
        f"{len(middle_starts)}"
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
    # Final target.
    # --------------------------------------------------------------

    h1_times = np.asarray(
        [
            row[
                "time_s"
            ]
            for row in h1_rows
        ],
        dtype=float,
    )

    target_idx = nearest_index(
        h1_times,
        TARGET_TIME,
    )

    target_row = (
        h1_rows[
            target_idx
        ]
    )

    sim_final = np.array(
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
        "TRUSTED H1 FINAL"
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

    # ==============================================================
    # Decision.
    # ==============================================================

    print()
    print("=" * 150)
    print(
        "G27A DECISION"
    )
    print("=" * 150)

    print(
        "FULL-RANGE TENDON ROUTING:"
    )

    print(
        "  PASS — no sign reversal / collapse."
    )

    print()
    print(
        "H1 LOW-ANGLE CONTACT:"
    )

    if len(
        middle_starts
    ) == 1:

        print(
            "  middle contact continuous"
        )

    else:

        print(
            "  middle bounce STILL EXISTS"
        )

    print()

    if (
        len(
            middle_starts
        )
        >
        1
    ):

        print(
            "INTERPRETATION:"
        )

        print(
            "  The full-range routing defect is now repaired,"
        )

        print(
            "  but it was NOT the cause of the low-angle H1 bounce."
        )

        print()

        print(
            "  Next identification can now legitimately focus on"
        )

        print(
            "  low-angle transmission/passive mechanics without"
        )

        print(
            "  fitting against a tendon that becomes nonphysical"
        )

        print(
            "  later in the finger range."
        )

    else:

        print(
            "INTERPRETATION:"
        )

        print(
            "  Physical wrap routing also removed the H1 "
            "contact discontinuity."
        )

        print(
            "  Then this routing becomes the new architecture "
            "baseline."
        )

    summary_csv = (
        out
        /
        "g27a_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "routing_pass",
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
            "routing_pass":
                routing_pass,

            "first_contact_encoder":
                first_row[
                    "encoder_tick"
                ],

            "first_contact_encoder_error":
                first_row[
                    "encoder_tick"
                ]
                -
                H1_HW_ENCODER,

            "first_contact_pose_rmse_deg":
                contact_pose_rmse,

            "middle_episodes":
                len(
                    middle_starts
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
        "g27a_h1_trajectory.npz",
        times=np.asarray(
            times
        ),
        qpos=np.asarray(
            qpos
        ),
    )

    print()
    print(
        f"Wrap geometry:\n  {wrap_csv}"
    )

    print()
    print(
        f"Full-range audit:\n  {audit_csv}"
    )

    print()
    print(
        f"Candidate XML:\n  {candidate_xml}"
    )

    print()
    print(
        f"Summary:\n  {summary_csv}"
    )

    print()
    print(
        f"All outputs:\n  {out}"
    )

    if not args.no_gui:

        print()
        print("=" * 150)
        print(
            "G27A GUI — PHYSICAL WRAPPED-TENDON CANDIDATE"
        )
        print("=" * 150)

        print(
            "Check only:"
        )

        print(
            "  1. middle first-contact configuration"
        )

        print(
            "  2. whether middle stays or separates"
        )

        print(
            "  3. PIP wrapping"
        )

        print(
            "  4. distal contact"
        )

        print(
            "  5. DIP motion"
        )

        g7.exact_gui_replay(
            candidate_xml,
            qpos,
            times,
            "G27A physical joint-wrap tendon",
        )


if __name__ == "__main__":
    main()
