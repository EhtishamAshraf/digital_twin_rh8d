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

import audit_index_tendon_transmission_v35g12 as g12
import build_persistent_flexor_wrap_v35g38a as g38a

from rh8d_tabletop_stage_v1 import (
    apply_tabletop_stage,
    verify_tabletop_scene,
)


# =====================================================================
# G39A
#
# RH8D CAD MULTI-GUIDE FLEXOR CHANNEL
#
# Structural transmission experiment ONLY.
#
# IMPORTANT:
#   - idx_flex_spatial remains one real MuJoCo <spatial> tendon.
#   - NO fixed tendon.
#   - NO prescribed MCP/PIP/DIP motion.
#   - NO equality constraints.
#   - NO passive tuning.
#   - NO friction tuning.
#   - NO H1 fitting.
#
# The existing RH8D CAD route supplies the guide locations.
#
# Original:
#
#   p01 -> p02 -> p03 -> p04 -> p05 -> p06 -> p07 -> p08 -> p09
#
# G39A:
#
#   p01
#     -> geom at CAD p02
#     -> p03
#     -> geom at CAD p04
#     -> p05
#     -> geom at CAD p06
#     -> p07
#     -> p08
#     -> p09
#
# Thus the main cable is constrained along the digitized RH8D
# flexor channel similarly to Aero's alternating site/geom/site route.
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


# Each guide replaces an existing CAD waypoint in the tendon path.
#
# prev -> GUIDE -> next
#
GUIDES = {
    "mcp": {
        "prev": "v35_p01_palm_entry",
        "guide": "v35_p02_mcp_parent",
        "next": "v35_p03_mcp_child",
    },

    "pip": {
        "prev": "v35_p03_mcp_child",
        "guide": "v35_p04_pip_parent",
        "next": "v35_p05_pip_child",
    },

    "dip": {
        "prev": "v35_p05_pip_child",
        "guide": "v35_p06_dip_parent",
        "next": "v35_p07_dip_child",
    },
}


# Numerical margin beyond the original CAD-corner chord.
#
# This is NOT fitted to H1.
FORCE_WRAP_MARGIN = 0.00035


# Aero uses small offset guide cylinders.
#
# We refuse a guide larger than this. If the CAD waypoint would need
# something larger, the abstraction fails rather than being tuned.
MAX_GUIDE_RADIUS = 0.00250


# Long enough along the joint axis to prevent axial escape.
GUIDE_HALF_LENGTH = 0.012


# sidesite lies outside the routing cylinder.
SIDE_EXTRA = 0.0020


VIDEO13 = [
    # tick, MCP, PIP, DIP
    (100,   0.000,  0.000,  0.000),
    (500,  16.952,  0.968, -0.428),
    (1000, 39.472, -0.907, -0.343),
    (1500, 57.291,  3.654, -4.019),
    (2000, 73.632,  4.667,  2.012),
    (2500, 73.632, 27.119,  3.190),
    (3000, 77.347, 48.581,  7.810),
    (3500, 80.074, 70.872, -7.199),
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


def project_perpendicular(
    v,
    axis,
):

    v = np.asarray(
        v,
        dtype=float,
    )

    axis = unit(
        axis
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


def segment_axis_distance(
    center,
    axis,
    p0,
    p1,
):
    """
    Minimum radial distance between the infinite cylinder axis
    through 'center' and the finite segment p0 -> p1.

    Everything is projected into the plane normal to the
    cylinder axis.
    """

    r0 = project_perpendicular(
        p0 - center,
        axis,
    )

    r1 = project_perpendicular(
        p1 - center,
        axis,
    )

    d = (
        r1
        -
        r0
    )

    denom = float(
        np.dot(
            d,
            d,
        )
    )

    if denom < 1e-16:

        t = 0.0

    else:

        t = float(
            -np.dot(
                r0,
                d,
            )
            /
            denom
        )

        t = float(
            np.clip(
                t,
                0.0,
                1.0,
            )
        )

    closest = (
        r0
        +
        t
        *
        d
    )

    distance = float(
        np.linalg.norm(
            closest
        )
    )

    endpoint0 = float(
        np.linalg.norm(
            r0
        )
    )

    endpoint1 = float(
        np.linalg.norm(
            r1
        )
    )

    return (
        distance,
        endpoint0,
        endpoint1,
        t,
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


def replace_site_with_geom(
    spatial,
    old_site_name,
    geom_name,
    sidesite_name,
):

    children = list(
        spatial
    )

    index = None

    for i, elem in enumerate(
        children
    ):

        if (
            elem.tag == "site"
            and
            elem.get(
                "site"
            )
            ==
            old_site_name
        ):

            index = i
            break

    if index is None:

        raise RuntimeError(
            f"Tendon site not found: {old_site_name}"
        )

    spatial.remove(
        children[
            index
        ]
    )

    spatial.insert(
        index,
        ET.Element(
            "geom",
            {
                "geom":
                    geom_name,

                "sidesite":
                    sidesite_name,
            },
        ),
    )


def joint_axis_world(
    model,
    data,
    joint_name,
):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if jid < 0:

        raise RuntimeError(
            f"Joint missing: {joint_name}"
        )

    # MuJoCo provides global joint anchor/axis after mj_forward.
    center = np.asarray(
        data.xanchor[
            jid
        ],
        dtype=float,
    )

    axis = unit(
        np.asarray(
            data.xaxis[
                jid
            ],
            dtype=float,
        )
    )

    return (
        jid,
        center,
        axis,
    )


def build_guide_information(
    model,
    data,
    key,
):

    spec = GUIDES[
        key
    ]

    prev_sid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        spec[
            "prev"
        ],
    )

    guide_sid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        spec[
            "guide"
        ],
    )

    next_sid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        spec[
            "next"
        ],
    )

    if min(
        prev_sid,
        guide_sid,
        next_sid,
    ) < 0:

        raise RuntimeError(
            f"{key.upper()} CAD waypoint missing."
        )

    (
        jid,
        joint_center,
        joint_axis,
    ) = joint_axis_world(
        model,
        data,
        JOINTS[
            key
        ],
    )

    prev_world = np.asarray(
        data.site_xpos[
            prev_sid
        ],
        dtype=float,
    )

    guide_world = np.asarray(
        data.site_xpos[
            guide_sid
        ],
        dtype=float,
    )

    next_world = np.asarray(
        data.site_xpos[
            next_sid
        ],
        dtype=float,
    )

    (
        chord_distance,
        d_prev,
        d_next,
        chord_fraction,
    ) = segment_axis_distance(
        guide_world,
        joint_axis,
        prev_world,
        next_world,
    )

    # -------------------------------------------------------------
    # Smallest guide that forces the original straight chord to
    # interact with the CAD waypoint.
    # -------------------------------------------------------------

    radius = (
        chord_distance
        +
        FORCE_WRAP_MARGIN
    )

    endpoint_limit = (
        min(
            d_prev,
            d_next,
        )
        -
        FORCE_WRAP_MARGIN
    )

    if endpoint_limit <= 0.0:

        raise RuntimeError(
            f"{key.upper()}: endpoint geometry invalid."
        )

    if radius >= endpoint_limit:

        raise RuntimeError(
            f"{key.upper()}: no valid guide can be placed at the "
            "existing CAD waypoint without swallowing a neighboring "
            "routing site."
        )

    if radius > MAX_GUIDE_RADIUS:

        raise RuntimeError(
            f"{key.upper()}: required guide radius "
            f"{1000*radius:.3f} mm > "
            f"{1000*MAX_GUIDE_RADIUS:.3f} mm hard limit."
        )

    # -------------------------------------------------------------
    # Choose the physical cable side directly from the CAD waypoint:
    #
    # joint axis -> existing CAD guide location
    #
    # No branch scan.
    # -------------------------------------------------------------

    outward = project_perpendicular(
        guide_world
        -
        joint_center,
        joint_axis,
    )

    outward = unit(
        outward
    )

    host_bid = int(
        model.site_bodyid[
            guide_sid
        ]
    )

    host_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        host_bid,
    )

    host_pos = np.asarray(
        data.xpos[
            host_bid
        ],
        dtype=float,
    )

    host_R = np.asarray(
        data.xmat[
            host_bid
        ],
        dtype=float,
    ).reshape(
        3,
        3,
    )

    guide_local = np.asarray(
        model.site_pos[
            guide_sid
        ],
        dtype=float,
    )

    axis_local = unit(
        host_R.T
        @
        joint_axis
    )

    sidesite_world = (
        guide_world
        +
        (
            radius
            +
            SIDE_EXTRA
        )
        *
        outward
    )

    sidesite_local = (
        host_R.T
        @
        (
            sidesite_world
            -
            host_pos
        )
    )

    return {
        "key":
            key,

        "jid":
            jid,

        "host_bid":
            host_bid,

        "host_name":
            host_name,

        "guide_local":
            guide_local,

        "axis_local":
            axis_local,

        "sidesite_local":
            sidesite_local,

        "radius":
            radius,

        "chord_distance":
            chord_distance,

        "d_prev":
            d_prev,

        "d_next":
            d_next,

        "chord_fraction":
            chord_fraction,

        "old_site":
            spec[
                "guide"
            ],
    }


def build_candidate(
    source,
    destination,
    out,
):

    original_model = mujoco.MjModel.from_xml_path(
        str(
            source
        )
    )

    flat = (
        out
        /
        "g39a_flat_source.xml"
    )

    mujoco.mj_saveLastXML(
        str(
            flat
        ),
        original_model,
    )

    model = mujoco.MjModel.from_xml_path(
        str(
            flat
        )
    )

    data = mujoco.MjData(
        model
    )

    # =============================================================
    # CRITICAL:
    # Always use the real tabletop experimental staging.
    # =============================================================

    apply_tabletop_stage(
        model,
        data,
        reset_to_qpos0=True,
    )

    verify_tabletop_scene(
        model,
        data,
    )

    old_tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        MAIN_TENDON,
    )

    old_length = float(
        data.ten_length[
            old_tid
        ]
    )

    old_reference = (
        g12.tendon_spring_reference(
            model,
            old_tid,
        )
    )

    print()
    print("=" * 150)
    print(
        "G39A STEP 1 — RH8D CAD MULTI-GUIDE CONSTRUCTION"
    )
    print("=" * 150)

    guide_info = {}

    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        info = build_guide_information(
            model,
            data,
            key,
        )

        guide_info[
            key
        ] = info

        print()
        print(
            key.upper()
        )

        print(
            f"  CAD waypoint             = "
            f"{info['old_site']}"
        )

        print(
            f"  host body                = "
            f"{info['host_name']}"
        )

        print(
            f"  straight-chord distance  = "
            f"{1000*info['chord_distance']:.4f} mm"
        )

        print(
            f"  generated guide radius   = "
            f"{1000*info['radius']:.4f} mm"
        )

        print(
            f"  previous endpoint radius = "
            f"{1000*info['d_prev']:.4f} mm"
        )

        print(
            f"  next endpoint radius     = "
            f"{1000*info['d_next']:.4f} mm"
        )

        print(
            "  sidesite                 = outward from joint "
            "through existing CAD flexor waypoint"
        )

    tree = ET.parse(
        flat
    )

    root = tree.getroot()

    spatial = find_main_spatial(
        root
    )

    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        info = guide_info[
            key
        ]

        body_xml = find_body_xml(
            root,
            info[
                "host_name"
            ],
        )

        guide_name = (
            f"g39a_{key}_cad_guide"
        )

        sidesite_name = (
            f"g39a_{key}_cad_side"
        )

        p1 = (
            info[
                "guide_local"
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
                "guide_local"
            ]
            +
            GUIDE_HALF_LENGTH
            *
            info[
                "axis_local"
            ]
        )

        ET.SubElement(
            body_xml,
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
                    f"{info['radius']:.12g}",

                # Numerical cable guide, faint cyan.
                "rgba":
                    "0.05 0.65 1 0.10",

                "contype":
                    "0",

                "conaffinity":
                    "0",

                "group":
                    "2",
            },
        )

        ET.SubElement(
            body_xml,
            "site",
            {
                "name":
                    sidesite_name,

                "pos":
                    " ".join(
                        f"{x:.12g}"
                        for x in info[
                            "sidesite_local"
                        ]
                    ),

                "size":
                    "0.00025",

                # Hide helper site.
                "rgba":
                    "1 1 0 0",

                "group":
                    "4",
            },
        )

        replace_site_with_geom(
            spatial,
            info[
                "old_site"
            ],
            guide_name,
            sidesite_name,
        )

    stage_xml = (
        out
        /
        "g39a_stage.xml"
    )

    tree.write(
        stage_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    # Compile intermediate candidate.
    stage_model = mujoco.MjModel.from_xml_path(
        str(
            stage_xml
        )
    )

    stage_data = mujoco.MjData(
        stage_model
    )

    apply_tabletop_stage(
        stage_model,
        stage_data,
        reset_to_qpos0=True,
    )

    stage_tid = mujoco.mj_name2id(
        stage_model,
        mujoco.mjtObj.mjOBJ_TENDON,
        MAIN_TENDON,
    )

    new_length = float(
        stage_data.ten_length[
            stage_tid
        ]
    )

    # Preserve the original open-state elastic extension.
    delta_length = (
        new_length
        -
        old_length
    )

    new_reference = (
        old_reference
        +
        delta_length
    )

    final_tree = ET.parse(
        stage_xml
    )

    final_spatial = find_main_spatial(
        final_tree.getroot()
    )

    final_spatial.set(
        "springlength",
        f"{new_reference:.12g}",
    )

    final_tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )

    mujoco.MjModel.from_xml_path(
        str(
            destination
        )
    )

    print()
    print(
        "OPEN-STATE PATH CHECK"
    )

    print(
        f"  original length = "
        f"{1000*old_length:.6f} mm"
    )

    print(
        f"  new length      = "
        f"{1000*new_length:.6f} mm"
    )

    print(
        f"  path change     = "
        f"{1000*delta_length:+.6f} mm"
    )


def set_index_pose(
    model,
    data,
    refs,
    mcp,
    pip,
    dip,
):

    # Always begin at the real experimental tabletop configuration.
    apply_tabletop_stage(
        model,
        data,
        reset_to_qpos0=True,
    )

    values = {
        "mcp":
            max(
                0.0,
                float(
                    mcp
                ),
            ),

        "pip":
            max(
                0.0,
                float(
                    pip
                ),
            ),

        "dip":
            max(
                0.0,
                float(
                    dip
                ),
            ),
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
            values[
                key
            ]
        )

    mujoco.mj_forward(
        model,
        data,
    )


def audit_candidate(
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

    if tid < 0:

        raise RuntimeError(
            "idx_flex_spatial disappeared."
        )

    records = []

    overall_pass = True

    print()
    print("=" * 150)
    print(
        "G39A STEP 2 — INDEPENDENT 0–85 DEG MOMENT-ARM AUDIT"
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

            mcp = (
                angle
                if key == "mcp"
                else 0.0
            )

            pip = (
                angle
                if key == "pip"
                else 0.0
            )

            dip = (
                angle
                if key == "dip"
                else 0.0
            )

            set_index_pose(
                model,
                data,
                refs,
                mcp,
                pip,
                dip,
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

            records.append({
                "test":
                    f"{key}_single",

                "tick":
                    "",

                "mcp_deg":
                    mcp,

                "pip_deg":
                    pip,

                "dip_deg":
                    dip,

                "j_mcp_mm_rad":
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
                    ),

                "j_pip_mm_rad":
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
                    ),

                "j_dip_mm_rad":
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
                    ),
            })

        same_sign = all(
            value < 0.0
            for value in values
        )

        minimum = min(
            abs(
                value
            )
            for value in values
        )

        useful = (
            minimum
            >=
            0.003
        )

        print()
        print(
            f"  flexion sign throughout = "
            f"{same_sign}"
        )

        print(
            f"  minimum |J|             = "
            f"{1000*minimum:.4f} mm/rad"
        )

        if not (
            same_sign
            and
            useful
        ):

            overall_pass = False

    # =============================================================
    # Real free-space video configurations.
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G39A STEP 3 — VIDEO-13 MEASURED CONFIGURATION AUDIT"
    )
    print("=" * 150)

    print()
    print(
        "tick     MCP     PIP     DIP      "
        "J_MCP      J_PIP      J_DIP"
    )

    print("-" * 110)

    video_pass = True

    for (
        tick,
        mcp,
        pip,
        dip,
    ) in VIDEO13:

        set_index_pose(
            model,
            data,
            refs,
            mcp,
            pip,
            dip,
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
            f"{max(0,mcp):7.2f} "
            f"{max(0,pip):7.2f} "
            f"{max(0,dip):7.2f}   "
            f"{1000*jm:+9.4f} "
            f"{1000*jp:+9.4f} "
            f"{1000*jd:+9.4f}"
        )

        records.append({
            "test":
                "video13",

            "tick":
                tick,

            "mcp_deg":
                max(
                    0.0,
                    mcp,
                ),

            "pip_deg":
                max(
                    0.0,
                    pip,
                ),

            "dip_deg":
                max(
                    0.0,
                    dip,
                ),

            "j_mcp_mm_rad":
                1000.0
                *
                jm,

            "j_pip_mm_rad":
                1000.0
                *
                jp,

            "j_dip_mm_rad":
                1000.0
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

            video_pass = False

    overall_pass = (
        overall_pass
        and
        video_pass
    )

    csv_path = (
        out
        /
        "g39a_cad_multiguide_audit.csv"
    )

    with csv_path.open(
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
            records
        )

    print()
    print("=" * 150)
    print(
        "G39A DECISION"
    )
    print("=" * 150)

    if overall_pass:

        print(
            "PASS:"
        )

        print(
            "The RH8D CAD multi-guide idx_flex_spatial keeps "
            "flexion-signed useful leverage through 0–85 deg."
        )

        print()
        print(
            "FREEZE MAIN FLEXOR TRANSMISSION GEOMETRY."
        )

        print(
            "NEXT = identify passive MCP -> PIP -> DIP recruitment "
            "from VIDEO 13 only."
        )

    else:

        print(
            "FAIL:"
        )

        print(
            "The existing CAD waypoints cannot be converted into a "
            "persistent multi-guide channel with this minimal "
            "Aero-style topology."
        )

        print()
        print(
            "NO radius scan. NO sidesite scan."
        )

        print(
            "Next = return to the digitized CAD cross-section and "
            "add missing physical guide/wrap locations explicitly."
        )

    print()
    print(
        f"Audit CSV:\n  "
        f"{csv_path}"
    )

    return overall_pass


def open_tabletop_gui(
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

    # =============================================================
    # ABSOLUTELY DO NOT show model.qpos0.
    #
    # Real tabletop/bottle experiment staging first.
    # =============================================================

    apply_tabletop_stage(
        model,
        data,
        reset_to_qpos0=True,
    )

    # Hardware first-middle-contact pose.
    data.qpos[
        refs[
            "mcp"
        ][
            "qadr"
        ]
    ] = math.radians(
        18.543
    )

    data.qpos[
        refs[
            "pip"
        ][
            "qadr"
        ]
    ] = math.radians(
        6.161
    )

    data.qpos[
        refs[
            "dip"
        ][
            "qadr"
        ]
    ] = math.radians(
        5.276
    )

    mujoco.mj_forward(
        model,
        data,
    )

    verify_tabletop_scene(
        model,
        data,
    )

    print()
    print("=" * 150)
    print(
        "G39A TABLETOP GEOMETRY GUI"
    )
    print("=" * 150)

    print(
        "This is the TABLETOP/BOTTLE configuration."
    )

    print(
        "It is NOT model.qpos0."
    )

    print()

    print(
        "Index pose:"
    )

    print(
        "  MCP/PIP/DIP = "
        "18.543 / 6.161 / 5.276 deg"
    )

    print()

    print(
        "MAGENTA = continuous idx_flex_spatial"
    )

    print(
        "Faint cyan = numerical CAD routing guides"
    )

    print()

    print(
        "GUI remains open until YOU close it."
    )

    # STATIC geometry viewer only.
    #
    # DO NOT call mj_step().
    # Otherwise the passive RH8D finger immediately relaxes
    # toward zero and destroys the pose we are trying to inspect.
    import time

    viewer = mujoco.viewer.launch_passive(
        model,
        data,
    )

    print()
    print(
        "STATIC VIEWER: physics is NOT advancing."
    )

    print(
        "The index finger will remain at "
        "18.543 / 6.161 / 5.276 deg."
    )

    print(
        "Close the GUI window when finished."
    )

    while viewer.is_running():

        viewer.sync()

        time.sleep(
            0.02
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
            "v35g39a_cad_multiguide_flexor"
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

    candidate = (
        out
        /
        "g39a_cad_multiguide_flexor.xml"
    )

    print()
    print("=" * 150)
    print(
        "G39A — RH8D CAD MULTI-GUIDE SPATIAL FLEXOR"
    )
    print("=" * 150)

    print()
    print(
        "Main flexor remains idx_flex_spatial."
    )

    print(
        "All geometry/audits use the frozen tabletop staging."
    )

    build_candidate(
        source,
        candidate,
        out,
    )

    passed = audit_candidate(
        candidate,
        out,
    )

    print()
    print(
        f"Candidate XML:\n  "
        f"{candidate}"
    )

    print()
    print(
        f"All outputs:\n  "
        f"{out}"
    )

    if not args.no_gui:

        open_tabletop_gui(
            candidate
        )

    if not passed:

        raise SystemExit(2)


if __name__ == "__main__":
    main()
