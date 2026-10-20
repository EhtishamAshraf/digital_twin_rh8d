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


# =====================================================================
# G38B — CAD-SIDE PERSISTENT FLEXOR WRAPPING
#
# Fixes G38A only.
#
# G38A problems:
#   1. sidesite direction inferred from an ambiguous open-pose midpoint
#   2. wrap radius only slightly above the OPEN distance
#   3. 6-mm half-length was marginal for the RH8D x-offset
#
# G38B:
#   - derives physical flexor side from original CAD crossing at 15 deg
#   - computes full 0..85 deg segment/hinge distance envelope
#   - radius = maximum CAD distance + 0.35 mm
#   - cylinder half-length = 12 mm
#
# NO dynamics.
# NO H1 fitting.
# NO friction.
# NO spring changes.
# NO fixed tendon.
# =====================================================================


MAIN_TENDON = "idx_flex_spatial"

JOINTS = g38a.JOINTS
CROSSINGS = g38a.CROSSINGS
MEASURED_POSES = g38a.MEASURED_POSES

REFERENCE_ANGLE_DEG = 15.0

ENVELOPE_STEP_DEG = 2.0

RADIAL_ENGAGEMENT_MARGIN = 0.00035

GUIDE_HALF_LENGTH = 0.012

SIDE_EXTRA = 0.0030


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
    vector,
    axis,
):

    vector = np.asarray(
        vector,
        dtype=float,
    )

    axis = np.asarray(
        axis,
        dtype=float,
    )

    return (
        vector
        -
        axis
        *
        float(
            np.dot(
                vector,
                axis,
            )
        )
    )


def closest_radial_vector(
    center,
    axis,
    p0,
    p1,
):
    """
    Closest vector from the infinite hinge axis to the finite
    tendon segment p0 -> p1.

    Project both endpoints into the plane normal to the axis.
    The axis then becomes the origin and we find the closest
    point on the projected segment.
    """

    r0 = project_perp(
        p0 - center,
        axis,
    )

    r1 = project_perp(
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

        closest = r0.copy()

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

    return (
        closest,
        float(
            np.linalg.norm(
                closest
            )
        ),
        t,
    )


def set_only_joint(
    model,
    data,
    refs,
    key,
    angle_deg,
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
        angle_deg
    )

    mujoco.mj_forward(
        model,
        data,
    )


def joint_center_axis(
    model,
    data,
    jid,
):

    bid = int(
        model.jnt_bodyid[
            jid
        ]
    )

    R = np.asarray(
        data.xmat[
            bid
        ],
        dtype=float,
    ).reshape(
        3,
        3,
    )

    p = np.asarray(
        data.xpos[
            bid
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
        p
        +
        R
        @
        joint_local
    )

    axis = unit(
        R
        @
        axis_local
    )

    return (
        bid,
        R,
        p,
        joint_local,
        axis_local,
        center,
        axis,
    )


def analyze_cad_crossing(
    model,
    key,
):

    data = mujoco.MjData(
        model
    )

    refs = g12.get_joint_refs(
        model
    )

    joint_name = JOINTS[
        key
    ]

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if jid < 0:

        raise RuntimeError(
            f"Joint not found: {joint_name}"
        )

    parent_site_name, child_site_name = (
        CROSSINGS[
            key
        ]
    )

    parent_sid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        parent_site_name,
    )

    child_sid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        child_site_name,
    )

    if (
        parent_sid < 0
        or
        child_sid < 0
    ):

        raise RuntimeError(
            f"CAD sites missing for {key.upper()}"
        )

    envelope = []

    reference = None

    for angle in np.arange(
        0.0,
        85.0 + 0.001,
        ENVELOPE_STEP_DEG,
    ):

        set_only_joint(
            model,
            data,
            refs,
            key,
            float(
                angle
            ),
        )

        (
            child_bid,
            R,
            body_pos,
            joint_local,
            axis_local,
            center,
            axis,
        ) = joint_center_axis(
            model,
            data,
            jid,
        )

        p0 = np.asarray(
            data.site_xpos[
                parent_sid
            ],
            dtype=float,
        )

        p1 = np.asarray(
            data.site_xpos[
                child_sid
            ],
            dtype=float,
        )

        (
            radial,
            distance,
            segment_fraction,
        ) = closest_radial_vector(
            center,
            axis,
            p0,
            p1,
        )

        envelope.append(
            (
                float(
                    angle
                ),
                distance,
                radial.copy(),
                segment_fraction,
            )
        )

    max_item = max(
        envelope,
        key=lambda x:
            x[
                1
            ],
    )

    # -------------------------------------------------------------
    # Determine physical flexor side at 15 degrees.
    #
    # Use nearest envelope sample then recompute exactly at 15 deg.
    # This avoids the open-pose branch ambiguity seen in G38A.
    # -------------------------------------------------------------

    set_only_joint(
        model,
        data,
        refs,
        key,
        REFERENCE_ANGLE_DEG,
    )

    (
        child_bid,
        R_ref,
        body_pos_ref,
        joint_local,
        axis_local,
        center_ref,
        axis_ref,
    ) = joint_center_axis(
        model,
        data,
        jid,
    )

    p0_ref = np.asarray(
        data.site_xpos[
            parent_sid
        ],
        dtype=float,
    )

    p1_ref = np.asarray(
        data.site_xpos[
            child_sid
        ],
        dtype=float,
    )

    (
        radial_ref,
        distance_ref,
        t_ref,
    ) = closest_radial_vector(
        center_ref,
        axis_ref,
        p0_ref,
        p1_ref,
    )

    physical_side_world = unit(
        radial_ref
    )

    radius = (
        max_item[
            1
        ]
        +
        RADIAL_ENGAGEMENT_MARGIN
    )

    # Sanity: guide should remain inside roughly the RH8D finger
    # cross-section rather than becoming a huge invented pulley.
    if radius > 0.0085:

        raise RuntimeError(
            f"{key.upper()} required guide radius "
            f"{1000*radius:.3f} mm is implausibly large."
        )

    sidesite_world = (
        center_ref
        +
        (
            radius
            +
            SIDE_EXTRA
        )
        *
        physical_side_world
    )

    # Convert physical side to CHILD-BODY coordinates at the
    # reference pose.  The side then rotates naturally with the
    # finger body.
    sidesite_local = (
        R_ref.T
        @
        (
            sidesite_world
            -
            body_pos_ref
        )
    )

    return {
        "joint_name":
            joint_name,

        "jid":
            jid,

        "child_bid":
            child_bid,

        "child_name":
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                child_bid,
            ),

        "joint_local":
            joint_local,

        "axis_local":
            axis_local,

        "parent_site":
            parent_site_name,

        "child_site":
            child_site_name,

        "reference_distance":
            distance_ref,

        "reference_fraction":
            t_ref,

        "max_distance":
            max_item[
                1
            ],

        "max_distance_angle":
            max_item[
                0
            ],

        "radius":
            radius,

        "sidesite_local":
            sidesite_local,

        "envelope":
            envelope,
    }


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
            "Top-level <tendon> section missing."
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


def insert_wrap(
    spatial,
    parent_site,
    child_site,
    guide_name,
    sidesite_name,
):

    children = list(
        spatial
    )

    parent_index = None
    child_index = None

    for i, element in enumerate(
        children
    ):

        if (
            element.tag == "site"
            and
            element.get(
                "site"
            )
            ==
            parent_site
        ):

            parent_index = i

        if (
            element.tag == "site"
            and
            element.get(
                "site"
            )
            ==
            child_site
        ):

            child_index = i

    if (
        parent_index is None
        or
        child_index is None
    ):

        raise RuntimeError(
            f"Could not find {parent_site} -> {child_site}"
        )

    if (
        child_index
        !=
        parent_index
        +
        1
    ):

        raise RuntimeError(
            f"{parent_site} and {child_site} are not consecutive."
        )

    spatial.insert(
        parent_index
        +
        1,
        ET.Element(
            "geom",
            {
                "geom":
                    guide_name,

                "sidesite":
                    sidesite_name,
            },
        ),
    )


def build_candidate(
    source,
    destination,
    out,
):

    source_model = mujoco.MjModel.from_xml_path(
        str(
            source
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
        MAIN_TENDON,
    )

    old_open_length = float(
        source_data.ten_length[
            source_tid
        ]
    )

    old_spring_reference = (
        g12.tendon_spring_reference(
            source_model,
            source_tid,
        )
    )

    flat = (
        out
        /
        "g38b_flat_source.xml"
    )

    mujoco.mj_saveLastXML(
        str(
            flat
        ),
        source_model,
    )

    baseline_model = mujoco.MjModel.from_xml_path(
        str(
            flat
        )
    )

    print()
    print("=" * 150)
    print(
        "G38B STEP 1 — CAD CROSSING ENVELOPES"
    )
    print("=" * 150)

    analyses = {}

    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        info = analyze_cad_crossing(
            baseline_model,
            key,
        )

        analyses[
            key
        ] = info

        print()
        print(
            key.upper()
        )

        print(
            f"  distance at {REFERENCE_ANGLE_DEG:.1f} deg "
            f"= {1000*info['reference_distance']:.4f} mm"
        )

        print(
            f"  max CAD segment/axis distance "
            f"= {1000*info['max_distance']:.4f} mm"
        )

        print(
            f"  occurs at "
            f"{info['max_distance_angle']:.1f} deg"
        )

        print(
            f"  persistent guide radius "
            f"= {1000*info['radius']:.4f} mm"
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

        info = analyses[
            key
        ]

        body_xml = find_body_xml(
            root,
            info[
                "child_name"
            ],
        )

        guide_name = (
            f"g38b_{key}_flexor_wrap"
        )

        sidesite_name = (
            f"g38b_{key}_flexor_side"
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
                    "0.0003",

                # Hide construction helper.
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

                # Mathematical routing object only.
                # Invisible in the final viewer.
                "rgba":
                    "0 0.7 1 0",

                "contype":
                    "0",

                "conaffinity":
                    "0",

                "group":
                    "2",
            },
        )

        insert_wrap(
            spatial,
            info[
                "parent_site"
            ],
            info[
                "child_site"
            ],
            guide_name,
            sidesite_name,
        )

    stage_xml = (
        out
        /
        "g38b_stage.xml"
    )

    tree.write(
        stage_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    # First compile.
    stage_model = mujoco.MjModel.from_xml_path(
        str(
            stage_xml
        )
    )

    stage_data = mujoco.MjData(
        stage_model
    )

    stage_data.qpos[:] = (
        stage_model.qpos0
    )

    mujoco.mj_forward(
        stage_model,
        stage_data,
    )

    stage_tid = mujoco.mj_name2id(
        stage_model,
        mujoco.mjtObj.mjOBJ_TENDON,
        MAIN_TENDON,
    )

    new_open_length = float(
        stage_data.ten_length[
            stage_tid
        ]
    )

    # Preserve original open-state extension for completeness.
    length_delta = (
        new_open_length
        -
        old_open_length
    )

    new_spring_reference = (
        old_spring_reference
        +
        length_delta
    )

    stage_tree = ET.parse(
        stage_xml
    )

    stage_root = stage_tree.getroot()

    stage_spatial = find_main_spatial(
        stage_root
    )

    stage_spatial.set(
        "springlength",
        f"{new_spring_reference:.12g}",
    )

    stage_tree.write(
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
        "REFERENCE-LENGTH PRESERVATION"
    )

    print(
        f"  old open length = "
        f"{1000*old_open_length:.6f} mm"
    )

    print(
        f"  new open length = "
        f"{1000*new_open_length:.6f} mm"
    )

    print(
        f"  path delta      = "
        f"{1000*length_delta:+.6f} mm"
    )

    return analyses


def audit(
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

    records = []

    print()
    print("=" * 150)
    print(
        "G38B STEP 2 — INDEPENDENT FULL-RANGE SWEEP"
    )
    print("=" * 150)

    sweep_pass = True

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

            records.append({
                "test":
                    f"{key}_sweep",

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

        minimum_leverage = min(
            abs(
                x
            )
            for x in values
        )

        same_flexion_sign = all(
            x < 0.0
            for x in values
        )

        useful_leverage = (
            minimum_leverage
            >=
            0.003
        )

        print()
        print(
            f"  sign always flexion = "
            f"{same_flexion_sign}"
        )

        print(
            f"  min |J| = "
            f"{1000*minimum_leverage:.4f} mm/rad"
        )

        if not (
            same_flexion_sign
            and
            useful_leverage
        ):

            sweep_pass = False

    # =============================================================
    # Real video-13 configurations.
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G38B STEP 3 — VIDEO-13 MEASURED POSES"
    )
    print("=" * 150)

    print()
    print(
        "tick     MCP     PIP     DIP      "
        "J_MCP      J_PIP      J_DIP"
    )

    print("-" * 110)

    measured_pass = True

    for (
        tick,
        mcp,
        pip,
        dip,
    ) in MEASURED_POSES:

        data.qpos[:] = (
            model.qpos0
        )

        data.qvel[:] = 0.0

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

        records.append({
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

            measured_pass = False

    passed = (
        sweep_pass
        and
        measured_pass
    )

    print()
    print("=" * 150)
    print(
        "G38B DECISION"
    )
    print("=" * 150)

    if passed:

        print(
            "PASS: MAIN RH8D SPATIAL FLEXOR NOW KEEPS "
            "A PHYSICAL FLEXION SIGN THROUGHOUT THE RANGE."
        )

        print()
        print(
            "STOP transmission-geometry work."
        )

        print(
            "Next = identify sequential MCP -> PIP -> DIP passive "
            "mechanics from video 13 on THIS frozen route."
        )

    else:

        print(
            "FAIL: the hinge-centered wrap abstraction itself "
            "cannot preserve the RH8D route."
        )

        print()
        print(
            "Do NOT enlarge radii or scan sidesites."
        )

        print(
            "Next = Aero-style multi-guide flexor channel using "
            "the existing RH8D CAD sites as fixed waypoints."
        )

    csv_path = (
        out
        /
        "g38b_audit.csv"
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
    print(
        f"Audit:\n  {csv_path}"
    )

    return passed


def open_gui(
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

    # Representative high-flexion video-13 configuration.
    data.qpos[:] = (
        model.qpos0
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
        "G38B GUI — VIDEO-13 HIGH-FLEXION CONFIGURATION"
    )
    print("=" * 150)

    print(
        "Inspect the MAGENTA idx_flex_spatial."
    )

    print(
        "The numerical wrap cylinders are hidden."
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
            "v35g38b_persistent_flexor_wrap"
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
        "g38b_persistent_flexor_wrap.xml"
    )

    print()
    print("=" * 150)
    print(
        "G38B — CAD-SIDE PERSISTENT MAIN FLEXOR"
    )
    print("=" * 150)

    build_candidate(
        source,
        candidate,
        out,
    )

    passed = audit(
        candidate,
        out,
    )

    print()
    print(
        f"Candidate:\n  {candidate}"
    )

    print()
    print(
        f"All outputs:\n  {out}"
    )

    if not args.no_gui:

        open_gui(
            candidate
        )

    if not passed:

        raise SystemExit(2)


if __name__ == "__main__":
    main()
