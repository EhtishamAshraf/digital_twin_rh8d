#!/usr/bin/env python3

from __future__ import annotations

import itertools
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import build_audit_index_spatial_tendon_v34a as b


# =============================================================================
# V34A4
#
# CAD-CHANNEL SPATIAL TENDON
#
# No cylindrical wrapping geoms.
#
# Instead, approximate the real internal tendon channel using paired guide
# sites on the two bodies adjacent to each revolute joint:
#
#       MCP parent ---> MCP child
#                       |
#                       | same rigid proximal body
#                       v
#       PIP parent ---> PIP child
#                       |
#                       | same rigid middle body
#                       v
#       DIP parent ---> DIP child
#                       |
#                       v
#                    anchor
#
# Only the segment CROSSING a joint produces its tendon moment arm.
#
# This is a genuine MuJoCo spatial tendon.
# Joint angles are NEVER commanded.
# =============================================================================


WS = Path(__file__).resolve().parents[1]

MJCF_DIR = (
    WS
    / "src"
    / "ur5e_mujoco_description"
    / "mjcf"
)

BASE = (
    MJCF_DIR
    / "ur5e_rh8d_index_sysid_v5.xml"
)

OUT = (
    MJCF_DIR
    / "ur5e_rh8d_index_spatial_v34a5.xml"
)

TMP = (
    MJCF_DIR
    / ".ur5e_rh8d_index_spatial_v34a5_tmp.xml"
)


# -------------------------------------------------------------------------
# Names
# -------------------------------------------------------------------------

ACT = "pos_index_spool_v34"

S_MCP_PARENT = "idx_v34_mcp_parent"
S_MCP_CHILD  = "idx_v34_mcp_child"

S_PIP_PARENT = "idx_v34_pip_parent"
S_PIP_CHILD  = "idx_v34_pip_child"

S_DIP_PARENT = "idx_v34_dip_parent"
S_DIP_CHILD  = "idx_v34_dip_child"


# -------------------------------------------------------------------------
# Structural test poses
# -------------------------------------------------------------------------

POSES = [
    (5.0, 5.0, 2.0),
    (30.0, 10.0, 5.0),
    (45.0, 25.0, 10.0),
    (60.0, 50.0, 20.0),
]


# Previous identified effective transmission.
#
# IMPORTANT:
# only an initialization PRIOR for the new spatial geometry.
# These coefficients are NOT imposed.
TARGET = np.array(
    [
        0.005,
        0.004,
        0.001,
    ],
    dtype=float,
)


# -------------------------------------------------------------------------
# Physically reasonable guide-offset search
# -------------------------------------------------------------------------

MCP_RADII = np.array(
    [
        0.0030,
        0.0040,
        0.0050,
        0.0060,
        0.0070,
    ]
)

PIP_RADII = np.array(
    [
        0.0025,
        0.0035,
        0.0045,
        0.0055,
        0.0065,
    ]
)

DIP_RADII = np.array(
    [
        0.0008,
        0.0012,
        0.0016,
        0.0020,
        0.0024,
    ]
)


# Distance on each side of the joint at which the effective internal
# channel guide points are placed.
#
# This is NOT a pulley radius.
#
# It represents the finite distance over which the physical cable bends
# through the internal printed guide geometry visible in the CAD/photos.
DELTA_VALUES = np.array(
    [
        0.0030,
        0.0045,
        0.0060,
    ]
)


# Cable plane x coordinate.
#
# Existing RH8D index guide sites are around x=-10 mm.
# V34A5:
# hardware/CAD registration.
#
# The earlier -10 mm value came from old manually-created guide sites.
# The actual visual/collision phalanx center is close to -5 mm.
# Put the active flexor in that internal cable plane.
X_CABLE = -0.00505


# =============================================================================
# HELPERS
# =============================================================================

def vec3(text):
    return np.asarray(
        [
            float(x)
            for x in text.split()
        ],
        dtype=float,
    )


def fmt(v):
    return " ".join(
        f"{x:.10g}"
        for x in v
    )


def quat_to_R(q):
    """
    MuJoCo XML quaternion = w x y z.
    """

    q = np.asarray(
        q,
        dtype=float,
    )

    q = q / np.linalg.norm(q)

    w, x, y, z = q

    return np.array(
        [
            [
                1 - 2*(y*y + z*z),
                2*(x*y - z*w),
                2*(x*z + y*w),
            ],
            [
                2*(x*y + z*w),
                1 - 2*(x*x + z*z),
                2*(y*z - x*w),
            ],
            [
                2*(x*z - y*w),
                2*(y*z + x*w),
                1 - 2*(x*x + y*y),
            ],
        ],
        dtype=float,
    )


def add_site(
    body,
    name,
    pos,
    rgba="1 1 0 1",
):

    ET.SubElement(
        body,
        "site",
        {
            "name": name,
            "pos": fmt(
                np.asarray(
                    pos,
                    dtype=float,
                )
            ),
            "size": "0.0012",
            "rgba": rgba,
            "group": "4",
        },
    )


def remove_generated_sites(root):

    names = {
        S_MCP_PARENT,
        S_MCP_CHILD,
        S_PIP_PARENT,
        S_PIP_CHILD,
        S_DIP_PARENT,
        S_DIP_CHILD,
        b.SPOOL_SITE,
    }

    for parent in root.iter():

        for child in list(
            parent
        ):

            if (
                child.tag == "site"
                and
                child.attrib.get(
                    "name"
                )
                in names
            ):

                parent.remove(
                    child
                )


def remove_old_index_actuators(root):
    """
    Remove any direct position actuator acting on:
      MCP / PIP / DIP / spool

    Then V34A4 adds ONE actuator:
      motor spool only.
    """

    actuator = root.find(
        ".//actuator"
    )

    if actuator is None:

        actuator = ET.SubElement(
            root,
            "actuator",
        )


    target_joints = {
        b.MCP,
        b.PIP,
        b.DIP,
        b.SPOOL,
    }


    for child in list(
        actuator
    ):

        if (
            child.attrib.get(
                "joint"
            )
            in target_joints
        ):

            actuator.remove(
                child
            )


    ET.SubElement(
        actuator,
        "position",
        {
            "name": ACT,
            "joint": b.SPOOL,
            "kp": "5000",
            "kv": "20",
            "ctrllimited": "true",
            "ctrlrange": "0 0.025",
        },
    )



def retarget_index_actuator_force_sensor(root):
    """
    The V5 MJCF contains:

        <actuatorfrc name="frc_index"
                     actuator="pos_index_tendon"/>

    V34A4 intentionally removes pos_index_tendon because it directly
    actuated an index joint.

    The only index actuator in V34A4 is now the physical motor/spool
    actuator pos_index_spool_v34, so frc_index must follow that actuator.
    """

    sensor_block = root.find(".//sensor")

    if sensor_block is None:
        return

    changed = False

    for sensor in sensor_block:

        if (
            sensor.tag == "actuatorfrc"
            and
            sensor.attrib.get("name") == "frc_index"
        ):
            old = sensor.attrib.get("actuator")

            sensor.set(
                "actuator",
                ACT,
            )

            print(
                "[V34A4 BUILD] Retargeted frc_index: "
                f"{old} -> {ACT}"
            )

            changed = True

    if not changed:
        print(
            "[V34A4 BUILD] WARNING: "
            "frc_index actuator-force sensor not found."
        )



def remove_old_index_tendon(root):

    tendon_block = root.find(
        ".//tendon"
    )

    if tendon_block is None:

        tendon_block = ET.SubElement(
            root,
            "tendon",
        )


    for child in list(
        tendon_block
    ):

        if (
            child.attrib.get(
                "name"
            )
            ==
            b.TENDON
        ):

            tendon_block.remove(
                child
            )


    return tendon_block


# =============================================================================
# GUIDE GEOMETRY
# =============================================================================

def compute_sites(
    root,
    signs,
    radii,
    delta,
):

    sm, sp, sd = signs

    rm, rp, rd = radii


    palm = b.body_named(
        root,
        "palm_assembly:1",
    )

    prox = b.body_named(
        root,
        "Index_Proximal",
    )

    middle = b.body_named(
        root,
        "Index_Middle",
    )

    distal = b.body_named(
        root,
        "Index_Distal",
    )


    # ------------------------------------------------------------------
    # MCP
    #
    # Index_Proximal has a nontrivial fixed orientation relative to palm.
    # Define both MCP guide points first in the INDEX-PROXIMAL reference
    # convention and transform the parent-side point into palm coordinates.
    # ------------------------------------------------------------------

    prox_body_pos = vec3(
        prox.attrib[
            "pos"
        ]
    )


    prox_quat = np.asarray(
        [
            float(x)
            for x
            in prox.attrib.get(
                "quat",
                "1 0 0 0",
            ).split()
        ],
        dtype=float,
    )


    R_prox = quat_to_R(
        prox_quat
    )


    mcp_parent_in_index_frame = np.array(
        [
            X_CABLE,
            sm * rm,
            +delta,
        ]
    )


    mcp_parent_palm = (
        prox_body_pos
        +
        R_prox
        @
        mcp_parent_in_index_frame
    )


    mcp_child = np.array(
        [
            X_CABLE,
            sm * rm,
            -delta,
        ]
    )


    # ------------------------------------------------------------------
    # PIP
    #
    # Index_Middle body origin IS the PIP joint location in the proximal
    # body frame.
    # ------------------------------------------------------------------

    pip_joint_in_prox = vec3(
        middle.attrib[
            "pos"
        ]
    )


    pip_parent = (
        pip_joint_in_prox
        +
        np.array(
            [
                X_CABLE,
                sp * rp,
                +delta,
            ]
        )
    )


    pip_child = np.array(
        [
            X_CABLE,
            sp * rp,
            -delta,
        ]
    )


    # ------------------------------------------------------------------
    # DIP
    # ------------------------------------------------------------------

    dip_joint_in_middle = vec3(
        distal.attrib[
            "pos"
        ]
    )


    dip_parent = (
        dip_joint_in_middle
        +
        np.array(
            [
                X_CABLE,
                sd * rd,
                +delta,
            ]
        )
    )


    dip_child = np.array(
        [
            X_CABLE,
            sd * rd,
            -delta,
        ]
    )


    return {
        S_MCP_PARENT:
            (
                palm,
                mcp_parent_palm,
            ),

        S_MCP_CHILD:
            (
                prox,
                mcp_child,
            ),

        S_PIP_PARENT:
            (
                prox,
                pip_parent,
            ),

        S_PIP_CHILD:
            (
                middle,
                pip_child,
            ),

        S_DIP_PARENT:
            (
                middle,
                dip_parent,
            ),

        S_DIP_CHILD:
            (
                distal,
                dip_child,
            ),
    }


# =============================================================================
# BUILD MODEL
# =============================================================================

def build(
    signs,
    radii,
    delta,
    stiffness=0.0,
    damping=0.0,
    springlength=None,
):

    tree = ET.parse(
        BASE
    )

    root = tree.getroot()


    b.make_index_visuals_solid(
        root
    )


    b.disable_v5_fixture_for_free_audit(
        root
    )


    remove_generated_sites(
        root
    )


    remove_old_index_actuators(
        root
    )


    retarget_index_actuator_force_sensor(
        root
    )


    tendon_block = remove_old_index_tendon(
        root
    )


    # ------------------------------------------------------------------
    # Moving motor-spool site
    # ------------------------------------------------------------------

    spool_body = b.body_named(
        root,
        b.SPOOL_BODY,
    )


    add_site(
        spool_body,
        b.SPOOL_SITE,
        np.zeros(
            3
        ),
        rgba="1 0 1 1",
    )


    palm_origin = b.site_named(
        root,
        b.PALM_ORIGIN,
    )


    spool_pos = vec3(
        spool_body.attrib.get(
            "pos",
            "0 0 0",
        )
    )


    palm_origin_pos = vec3(
        palm_origin.attrib[
            "pos"
        ]
    )


    spool_axis = (
        spool_pos
        -
        palm_origin_pos
    )


    spool_axis /= np.linalg.norm(
        spool_axis
    )


    b.joint_named(
        root,
        b.SPOOL,
    ).set(
        "axis",
        fmt(
            spool_axis
        ),
    )


    # ------------------------------------------------------------------
    # Joint-channel guide pairs
    # ------------------------------------------------------------------

    sites = compute_sites(
        root,
        signs,
        radii,
        delta,
    )


    for name, (
        body,
        pos,
    ) in sites.items():

        add_site(
            body,
            name,
            pos,
        )


    # ------------------------------------------------------------------
    # Spatial tendon
    # ------------------------------------------------------------------

    attrs = {
        "name": b.TENDON,
        "rgba": "1 0 1 1",
        "width": "0.0015",
        "stiffness":
            f"{stiffness:.10g}",
        "damping":
            f"{damping:.10g}",
        "frictionloss":
            "0",
    }


    if (
        springlength
        is not None
    ):

        attrs[
            "springlength"
        ] = (
            f"{springlength:.10g}"
        )


    tendon = ET.SubElement(
        tendon_block,
        "spatial",
        attrs,
    )


    sequence = [
        b.SPOOL_SITE,
        b.PALM_ORIGIN,

        S_MCP_PARENT,
        S_MCP_CHILD,

        S_PIP_PARENT,
        S_PIP_CHILD,

        S_DIP_PARENT,
        S_DIP_CHILD,

        b.ANCHOR,
    ]


    for name in sequence:

        ET.SubElement(
            tendon,
            "site",
            {
                "site":
                    name,
            },
        )


    return tree


def compile_tree(
    tree,
):

    tree.write(
        TMP,
        encoding="utf-8",
        xml_declaration=True,
    )


    model = mujoco.MjModel.from_xml_path(
        str(
            TMP
        )
    )


    data = mujoco.MjData(
        model
    )


    return (
        model,
        data,
    )


# =============================================================================
# SEARCH
# =============================================================================

def search():

    # Create one compiled model first.
    tree = build(
        signs=(-1, -1, -1),
        radii=(
            0.005,
            0.004,
            0.001,
        ),
        delta=0.0045,
    )


    model, data = compile_tree(
        tree
    )


    tid = b.id_tendon(
        model,
        b.TENDON,
    )


    site_ids = {
        name:
            b.id_site(
                model,
                name,
            )
        for name in (
            S_MCP_PARENT,
            S_MCP_CHILD,
            S_PIP_PARENT,
            S_PIP_CHILD,
            S_DIP_PARENT,
            S_DIP_CHILD,
        )
    }


    # Body IDs / base transforms
    palm_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "palm_assembly:1",
    )

    prox_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "Index_Proximal",
    )

    mid_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "Index_Middle",
    )

    dst_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "Index_Distal",
    )


    prox_body_pos = model.body_pos[
        prox_id
    ].copy()


    prox_R = np.zeros(
        9
    )


    mujoco.mju_quat2Mat(
        prox_R,
        model.body_quat[
            prox_id
        ],
    )


    prox_R = prox_R.reshape(
        3,
        3,
    )


    pip_joint_pos = model.body_pos[
        mid_id
    ].copy()


    dip_joint_pos = model.body_pos[
        dst_id
    ].copy()


    def set_candidate(
        signs,
        radii,
        delta,
    ):

        sm, sp, sd = signs

        rm, rp, rd = radii


        # MCP parent in palm body coordinates.
        conceptual_parent = np.array(
            [
                X_CABLE,
                sm * rm,
                +delta,
            ]
        )


        model.site_pos[
            site_ids[
                S_MCP_PARENT
            ]
        ] = (
            prox_body_pos
            +
            prox_R
            @
            conceptual_parent
        )


        model.site_pos[
            site_ids[
                S_MCP_CHILD
            ]
        ] = np.array(
            [
                X_CABLE,
                sm * rm,
                -delta,
            ]
        )


        model.site_pos[
            site_ids[
                S_PIP_PARENT
            ]
        ] = (
            pip_joint_pos
            +
            np.array(
                [
                    X_CABLE,
                    sp * rp,
                    +delta,
                ]
            )
        )


        model.site_pos[
            site_ids[
                S_PIP_CHILD
            ]
        ] = np.array(
            [
                X_CABLE,
                sp * rp,
                -delta,
            ]
        )


        model.site_pos[
            site_ids[
                S_DIP_PARENT
            ]
        ] = (
            dip_joint_pos
            +
            np.array(
                [
                    X_CABLE,
                    sd * rd,
                    +delta,
                ]
            )
        )


        model.site_pos[
            site_ids[
                S_DIP_CHILD
            ]
        ] = np.array(
            [
                X_CABLE,
                sd * rd,
                -delta,
            ]
        )


    rows = []


    total = (
        8
        *
        len(
            MCP_RADII
        )
        *
        len(
            PIP_RADII
        )
        *
        len(
            DIP_RADII
        )
        *
        len(
            DELTA_VALUES
        )
    )


    print()
    print("=" * 110)
    print(
        "V34A5 CAD-REGISTERED ROUTE SEARCH"
    )
    print("=" * 110)

    print(
        f"Candidates = {total}"
    )


    for (
        signs,
        rm,
        rp,
        rd,
        delta,
    ) in itertools.product(
        itertools.product(
            (-1, +1),
            repeat=3,
        ),
        MCP_RADII,
        PIP_RADII,
        DIP_RADII,
        DELTA_VALUES,
    ):

        radii = (
            rm,
            rp,
            rd,
        )


        set_candidate(
            signs,
            radii,
            delta,
        )


        arms_all = []

        gains = []


        valid = True


        for pose in POSES:

            arms, gain = b.moment_arms(
                model,
                data,
                tid,
                pose,
            )


            if (
                np.any(
                    ~np.isfinite(
                        arms
                    )
                )
                or
                not np.isfinite(
                    gain
                )
            ):

                valid = False

                break


            arms_all.append(
                arms
            )

            gains.append(
                gain
            )


        if not valid:

            continue


        A = np.asarray(
            arms_all
        )


        G = np.asarray(
            gains
        )


        # Hard physical requirements:
        #
        # all joints must receive flexion torque
        # throughout tested configuration range.

        if (
            np.any(
                A
                <=
                0.00005
            )
            or
            np.any(
                A
                >
                0.015
            )
            or
            np.any(
                G
                <
                0.8
            )
            or
            np.any(
                G
                >
                1.2
            )
        ):

            continue


        # Near-open prior against trusted old effective transmission.
        norm = np.array(
            [
                0.0025,
                0.0020,
                0.0015,
            ]
        )


        fit = float(
            np.sum(
                (
                    (
                        A[
                            0
                        ]
                        -
                        TARGET
                    )
                    /
                    norm
                )
                **
                2
            )
        )


        # Prefer moment arms that stay well behaved over the range
        # rather than crossing singularities.
        variation = float(
            np.sum(
                np.std(
                    A,
                    axis=0,
                )
                /
                np.maximum(
                    np.mean(
                        A,
                        axis=0,
                    ),
                    1e-6,
                )
            )
        )


        spool_score = float(
            20.0
            *
            np.mean(
                (
                    G
                    -
                    1.0
                )
                **
                2
            )
        )


        # Small regularization toward original effective scales.
        geometry_prior = float(
            0.05
            *
            np.sum(
                (
                    (
                        np.asarray(
                            radii
                        )
                        -
                        TARGET
                    )
                    /
                    0.003
                )
                **
                2
            )
        )


        score = (
            fit
            +
            0.30
            *
            variation
            +
            spool_score
            +
            geometry_prior
        )


        rows.append(
            {
                "score":
                    score,

                "signs":
                    signs,

                "radii":
                    radii,

                "delta":
                    delta,

                "A":
                    A.copy(),

                "G":
                    G.copy(),
            }
        )


    if not rows:

        raise RuntimeError(
            "V34A4 found no all-positive CAD-channel route. "
            "Do NOT tune contact. Inspect coordinate convention."
        )


    rows.sort(
        key=lambda x:
            x[
                "score"
            ]
    )


    print()
    print(
        f"Valid all-positive candidates = "
        f"{len(rows)}"
    )


    print()
    print(
        "Top 12:"
    )


    for rank, row in enumerate(
        rows[
            :12
        ],
        start=1,
    ):

        r = (
            np.asarray(
                row[
                    "radii"
                ]
            )
            *
            1000.0
        )


        a = (
            row[
                "A"
            ][
                0
            ]
            *
            1000.0
        )


        print(
            f" {rank:02d}. "
            f"score={row['score']:.4f} "
            f"signs={row['signs']} "
            f"offsets="
            f"{r[0]:.2f}/"
            f"{r[1]:.2f}/"
            f"{r[2]:.2f} mm "
            f"delta="
            f"{row['delta']*1000:.2f} mm "
            f"near-open arms="
            f"{a[0]:.3f}/"
            f"{a[1]:.3f}/"
            f"{a[2]:.3f} mm "
            f"dL/dx="
            f"{row['G'][0]:.6f}"
        )


    return rows[
        0
    ]


# =============================================================================
# FINAL MODEL + GUI
# =============================================================================

def main():

    print()
    print("=" * 110)
    print(
        "V34A5 — CAD-REGISTERED SPATIAL TENDON"
    )
    print("=" * 110)

    print()
    print(
        "Real spatial tendon."
    )

    print(
        "Paired guide sites represent the internal tendon channel "
        "around MCP/PIP/DIP."
    )

    print(
        "No wrap cylinders."
    )

    print(
        "No contact-triggered transmission."
    )

    print(
        "No direct joint actuator."
    )

    print(
        "Only the hidden motor spool is commanded."
    )


    best = search()


    signs = best[
        "signs"
    ]

    radii = best[
        "radii"
    ]

    delta = best[
        "delta"
    ]


    # ------------------------------------------------------------------
    # Measure open tendon length.
    # ------------------------------------------------------------------

    tree0 = build(
        signs,
        radii,
        delta,
    )


    model0, data0 = compile_tree(
        tree0
    )


    tid0 = b.id_tendon(
        model0,
        b.TENDON,
    )


    b.set_finger_pose(
        model0,
        data0,
        (
            0,
            0,
            0,
        ),
        spool_m=0.0,
    )


    L0 = b.tendon_length(
        model0,
        data0,
        tid0,
    )


    # ------------------------------------------------------------------
    # Final elastic tendon.
    # ------------------------------------------------------------------

    tree = build(
        signs,
        radii,
        delta,
        stiffness=b.TENDON_STIFFNESS,
        damping=b.TENDON_DAMPING,
        springlength=L0,
    )


    tree.write(
        OUT,
        encoding="utf-8",
        xml_declaration=True,
    )


    print()
    print("=" * 110)
    print(
        "SELECTED V34A5 ROUTE"
    )
    print("=" * 110)


    print(
        f"signs MCP/PIP/DIP = "
        f"{signs}"
    )


    print(
        "effective channel offsets = "
        f"{radii[0]*1000:.3f} / "
        f"{radii[1]*1000:.3f} / "
        f"{radii[2]*1000:.3f} mm"
    )


    print(
        f"guide half-span delta = "
        f"{delta*1000:.3f} mm"
    )


    print(
        f"open tendon length = "
        f"{L0*1000:.3f} mm"
    )


    print(
        f"saved model:\n  {OUT}"
    )


    # ------------------------------------------------------------------
    # Final audit.
    # ------------------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(
            OUT
        )
    )


    data = mujoco.MjData(
        model
    )


    tid = b.id_tendon(
        model,
        b.TENDON,
    )


    print()
    print("=" * 110)
    print(
        "CONFIGURATION-DEPENDENT MOMENT-ARM AUDIT"
    )
    print("=" * 110)


    for pose in POSES:

        arms, gain = b.moment_arms(
            model,
            data,
            tid,
            pose,
        )


        print(
            f" q="
            f"{pose[0]:5.1f}/"
            f"{pose[1]:5.1f}/"
            f"{pose[2]:5.1f} deg "
            f"arms="
            f"{arms[0]*1000:7.3f}/"
            f"{arms[1]*1000:7.3f}/"
            f"{arms[2]*1000:7.3f} mm "
            f"dL/dx="
            f"{gain:.6f}"
        )


    # ------------------------------------------------------------------
    # Native GUI.
    # ------------------------------------------------------------------

    data = mujoco.MjData(
        model
    )


    frozen = b.create_fixed_state(
        model,
        data,
    )


    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACT,
    )


    if actuator_id < 0:

        raise RuntimeError(
            f"Missing actuator {ACT}"
        )


    print()
    print("=" * 110)
    print(
        "V34A5 NATIVE MUJOCO VISUALIZATION"
    )
    print("=" * 110)

    print()
    print(
        "Magenta = ACTIVE spatial tendon."
    )

    print(
        "Yellow sites = effective CAD-channel guide points."
    )

    print(
        "Bottle is non-colliding in this architecture audit."
    )

    print()
    print(
        "Watch whether a single spool pull naturally flexes "
        "MCP, then PIP, then DIP."
    )


    viewer = mujoco.viewer.launch_passive(
        model,
        data,
    )


    try:

        middle_body = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "Index_Middle",
        )


        mujoco.mj_forward(
            model,
            data,
        )


        if middle_body >= 0:

            viewer.cam.lookat[:] = (
                data.xpos[
                    middle_body
                ]
            )

            viewer.cam.distance = 0.22


        total_time = 4.0


        steps = max(
            1,
            int(
                total_time
                /
                model.opt.timestep
            ),
        )


        for k in range(
            steps
        ):

            if not viewer.is_running():

                break


            t = (
                k
                *
                model.opt.timestep
            )


            u = min(
                1.0,
                t
                /
                2.5,
            )


            smooth = (
                3*u*u
                -
                2*u*u*u
            )


            data.ctrl[
                actuator_id
            ] = (
                b.VIEW_SPOOL_TARGET
                *
                smooth
            )


            mujoco.mj_step(
                model,
                data,
            )


            b.enforce_fixed_state(
                model,
                data,
                frozen,
            )


            viewer.sync()


            time.sleep(
                model.opt.timestep
            )


        q = np.degrees(
            [
                data.qpos[
                    b.qadr(
                        model,
                        b.MCP,
                    )
                ],

                data.qpos[
                    b.qadr(
                        model,
                        b.PIP,
                    )
                ],

                data.qpos[
                    b.qadr(
                        model,
                        b.DIP,
                    )
                ],
            ]
        )


        print()
        print(
            "Structural pose @ encoder-equivalent 1000 ticks:"
        )


        print(
            f" MCP/PIP/DIP = "
            f"{q[0]:.3f} / "
            f"{q[1]:.3f} / "
            f"{q[2]:.3f} deg"
        )


        print()
        print(
            "This is NOT yet the free-space fit."
        )

        print(
            "It is only the architecture/sign test."
        )


        print()
        print(
            "Viewer remains open for screen recording."
        )

        print(
            "Close the MuJoCo window when finished."
        )


        while viewer.is_running():

            viewer.sync()

            time.sleep(
                0.05
            )


    finally:

        viewer.close()


    try:

        TMP.unlink()

    except FileNotFoundError:

        pass


    print()
    print("=" * 110)
    print(
        "V34A5 COMPLETE"
    )
    print("=" * 110)


if __name__ == "__main__":

    main()
