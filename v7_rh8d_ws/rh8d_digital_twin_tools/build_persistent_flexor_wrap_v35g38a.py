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


# =====================================================================
# G38A — PERSISTENT RH8D MAIN-FLEXOR WRAPPING
#
# Structural experiment only.
#
# PRESERVED:
#   idx_flex_spatial
#   all existing tendon sites
#   all joint passive parameters
#   contact geometry
#
# ADDED:
#   one non-colliding cylindrical guide at MCP/PIP/DIP
#   one sidesite per guide forcing the spatial tendon to remain
#   on the EXISTING CAD flexor side.
#
# NO:
#   dynamics fitting
#   spring tuning
#   friction tuning
#   fixed tendon
#   PIP/DIP equality
#
# Pass criterion:
#   main flexor moment arms remain FLEXION-SIGNED and non-collapsing
#   over the complete physical index-finger range.
# =====================================================================


MAIN_TENDON = "idx_flex_spatial"


JOINTS = {
    "mcp": "Index_Proximal--palm_assembly:1",
    "pip": "Index_Middle--Index_Proximal",
    "dip": "Index_Distal--Index_Middle",
}


# Existing CAD route crossing each joint.
CROSSINGS = {
    "mcp": (
        "v35_p02_mcp_parent",
        "v35_p03_mcp_child",
    ),

    "pip": (
        "v35_p04_pip_parent",
        "v35_p05_pip_child",
    ),

    "dip": (
        "v35_p06_dip_parent",
        "v35_p07_dip_child",
    ),
}


# Measured low-angle CAD moment arms.
CAD_RADIUS = {
    "mcp": 0.00477625,
    "pip": 0.004495583,
    "dip": 0.004925074,
}


# Small penetration margin to ensure the wrapping geom stays active.
WRAP_CLEARANCE = 0.00010

GUIDE_HALF_LENGTH = 0.0060

SIDE_EXTRA = 0.0030


MEASURED_POSES = [
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


def project_perp(v, axis):

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


def find_body_xml(
    root,
    name,
):

    for body in root.iter("body"):

        if body.get("name") == name:
            return body

    raise RuntimeError(
        f"Body not found: {name}"
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


def build_candidate(
    source,
    destination,
):

    source_model = mujoco.MjModel.from_xml_path(
        str(
            source
        )
    )

    flat = (
        destination.parent
        /
        "g38a_flat_source.xml"
    )

    mujoco.mj_saveLastXML(
        str(
            flat
        ),
        source_model,
    )

    model = mujoco.MjModel.from_xml_path(
        str(
            flat
        )
    )

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

    tree = ET.parse(
        flat
    )

    root = tree.getroot()

    tendon_root = root.find(
        "./tendon"
    )

    if tendon_root is None:
        raise RuntimeError(
            "Top-level <tendon> missing."
        )

    spatial = None

    for element in tendon_root.findall(
        "spatial"
    ):

        if element.get(
            "name"
        ) == MAIN_TENDON:

            spatial = element
            break

    if spatial is None:
        raise RuntimeError(
            "idx_flex_spatial is not a spatial tendon."
        )

    print()
    print("=" * 150)
    print(
        "G38A STEP 1 — BUILD PERSISTENT FLEXOR WRAPS"
    )
    print("=" * 150)

    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

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
                f"Joint missing: {joint_name}"
            )

        child_bid = int(
            model.jnt_bodyid[
                jid
            ]
        )

        child_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            child_bid,
        )

        pc = np.asarray(
            data.xpos[
                child_bid
            ],
            dtype=float,
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

        center_world = (
            pc
            +
            Rc
            @
            joint_local
        )

        axis_world = unit(
            Rc
            @
            axis_local
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
                f"Missing CAD crossing sites for {key.upper()}"
            )

        p_world = np.asarray(
            data.site_xpos[
                parent_sid
            ],
            dtype=float,
        )

        c_world = np.asarray(
            data.site_xpos[
                child_sid
            ],
            dtype=float,
        )

        crossing_mid = (
            0.5
            *
            (
                p_world
                +
                c_world
            )
        )

        flexor_side = project_perp(
            crossing_mid
            -
            center_world,
            axis_world,
        )

        flexor_side = unit(
            flexor_side
        )

        radius = (
            CAD_RADIUS[
                key
            ]
            +
            WRAP_CLEARANCE
        )

        side_world = (
            center_world
            +
            (
                radius
                +
                SIDE_EXTRA
            )
            *
            flexor_side
        )

        side_local = local_from_world(
            pc,
            Rc,
            side_world,
        )

        body_xml = find_body_xml(
            root,
            child_name,
        )

        guide_name = (
            f"g38a_{key}_flexor_wrap"
        )

        sidesite_name = (
            f"g38a_{key}_flexor_side"
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
                        for x in side_local
                    ),

                "size":
                    "0.0005",

                # Yellow = side-selection helper.
                "rgba":
                    "1 1 0 0.35",

                "group":
                    "4",
            },
        )

        p1 = (
            joint_local
            -
            GUIDE_HALF_LENGTH
            *
            axis_local
        )

        p2 = (
            joint_local
            +
            GUIDE_HALF_LENGTH
            *
            axis_local
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
                    f"{radius:.12g}",

                # Cyan translucent = numerical cable guide.
                "rgba":
                    "0 0.7 1 0.18",

                "contype":
                    "0",

                "conaffinity":
                    "0",

                "group":
                    "2",
            },
        )

        # ---------------------------------------------------------
        # Insert geom between the EXISTING CAD parent/child sites.
        # ---------------------------------------------------------

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
                parent_site_name
            ):

                parent_index = i

            if (
                element.tag == "site"
                and
                element.get(
                    "site"
                )
                ==
                child_site_name
            ):

                child_index = i

        if (
            parent_index is None
            or
            child_index is None
        ):

            raise RuntimeError(
                f"Tendon crossing pair not found for {key}"
            )

        if child_index <= parent_index:

            raise RuntimeError(
                f"Unexpected tendon ordering for {key}"
            )

        # They should be directly consecutive in the original route.
        if (
            child_index
            !=
            parent_index
            +
            1
        ):

            raise RuntimeError(
                f"{key.upper()} crossing sites are not consecutive. "
                "Refusing automatic modification."
            )

        wrap_element = ET.Element(
            "geom",
            {
                "geom":
                    guide_name,

                "sidesite":
                    sidesite_name,
            },
        )

        spatial.insert(
            parent_index
            +
            1,
            wrap_element,
        )

        print()
        print(
            key.upper()
        )

        print(
            f"  CAD radius       = "
            f"{1000*CAD_RADIUS[key]:.4f} mm"
        )

        print(
            f"  wrap radius      = "
            f"{1000*radius:.4f} mm"
        )

        print(
            f"  parent site      = "
            f"{parent_site_name}"
        )

        print(
            f"  child site       = "
            f"{child_site_name}"
        )

        print(
            "  sidesite         = existing CAD flexor side"
        )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )

    # Compile immediately.
    mujoco.MjModel.from_xml_path(
        str(
            destination
        )
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
            "idx_flex_spatial missing after modification."
        )

    records = []

    print()
    print("=" * 150)
    print(
        "G38A STEP 2 — MEASURED FULL-RANGE MOMENT-ARM AUDIT"
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

        data.qpos[:] = (
            model.qpos0
        )

        data.qvel[:] = 0.0

        # Negative DIP measurements near zero are marker noise.
        # Clamp only for this geometric routing audit because the
        # MuJoCo physical range starts at zero.
        values = {
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
                values[
                    key
                ]
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

        records.append({
            "type":
                "measured_pose",

            "tick":
                tick,

            "mcp_deg":
                values[
                    "mcp"
                ],

            "pip_deg":
                values[
                    "pip"
                ],

            "dip_deg":
                values[
                    "dip"
                ],

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

        print(
            f"{tick:4d}  "
            f"{values['mcp']:7.2f} "
            f"{values['pip']:7.2f} "
            f"{values['dip']:7.2f}   "
            f"{1000*jm:+9.4f} "
            f"{1000*jp:+9.4f} "
            f"{1000*jd:+9.4f}"
        )

    # =============================================================
    # Independent single-joint sweep.
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G38A STEP 3 — INDEPENDENT JOINT SWEEP"
    )
    print("=" * 150)

    sweep_ok = True

    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

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

        for angle in [
            0,
            15,
            30,
            45,
            60,
            75,
            85,
        ]:

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

            J = tendon_jacobian_row(
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

            print(
                f" {angle:5.1f}      "
                f"{1000*value:+9.4f}"
            )

            records.append({
                "type":
                    f"{key}_sweep",

                "tick":
                    "",

                "mcp_deg":
                    angle
                    if
                    key == "mcp"
                    else
                    0,

                "pip_deg":
                    angle
                    if
                    key == "pip"
                    else
                    0,

                "dip_deg":
                    angle
                    if
                    key == "dip"
                    else
                    0,

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

            # Flexion sign must remain negative.
            # Require useful leverage > 3 mm/rad.
            if not (
                value
                <
                -0.003
            ):

                sweep_ok = False

    # =============================================================
    # Measured-pose check.
    # =============================================================

    measured = [
        r
        for r in records
        if
        r[
            "type"
        ]
        ==
        "measured_pose"
    ]

    measured_ok = True

    for r in measured:

        for field in [
            "j_mcp_mm_rad",
            "j_pip_mm_rad",
            "j_dip_mm_rad",
        ]:

            value_mm = float(
                r[
                    field
                ]
            )

            if value_mm >= -3.0:
                measured_ok = False

    full_pass = (
        sweep_ok
        and
        measured_ok
    )

    print()
    print("=" * 150)
    print(
        "G38A DECISION"
    )
    print("=" * 150)

    if full_pass:

        print(
            "PASS: idx_flex_spatial now has persistent "
            "flexion-signed leverage over the full index range."
        )

        print()
        print(
            "The transmission geometry is now suitable for "
            "free-space passive-mechanics identification."
        )

        print()
        print(
            "NEXT: fit MCP/PIP/DIP passive mechanics from VIDEO 13 "
            "sequential recruitment — NOT from H1 contact."
        )

    else:

        print(
            "FAIL: at least one joint still unwraps, reverses sign, "
            "or loses leverage."
        )

        print()
        print(
            "STOP. Do not fit springs or joint limits."
        )

        print(
            "We will correct only the failed guide/sidesite geometry."
        )

    csv_path = (
        out
        /
        "g38a_persistent_wrap_audit.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "type",
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
        f"Audit CSV:\n  "
        f"{csv_path}"
    )

    return full_pass


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
            "v35g38a_persistent_flexor_wrap"
        ),
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
        "g38a_persistent_flexor_wrap.xml"
    )

    print()
    print("=" * 150)
    print(
        "G38A — PERSISTENT MAIN SPATIAL-FLEXOR WRAPS"
    )
    print("=" * 150)

    build_candidate(
        source,
        candidate,
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

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
