#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import test_index_middle_mesh_collision_v28 as v28
import scan_index_fixed_support_v30 as v30


# ============================================================
# V35A
#
# CAD-MEASURED SPATIAL TENDON
# INSIDE EXACT V33 CONTACT ENVIRONMENT
#
# NO fixed 5/4/1 joint-coefficient tendon.
# NO V24 stateful transmission.
# NO load-triggered PIP breakaway.
# NO direct MCP/PIP/DIP command.
#
# The existing V33:
#   table
#   cylinder
#   support condition
#   replay initialization
#   middle mesh collision
#
# are preserved.
# ============================================================


WS = v30.v24.WS
MJCF_DIR = v30.v24.MJCF_DIR

MODEL_PATH = (
    MJCF_DIR
    / "ur5e_rh8d_index_cad_spatial_v35a.xml"
)

TMP_PATH = (
    MJCF_DIR
    / ".ur5e_rh8d_index_cad_spatial_v35a_tmp.xml"
)


MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

SPOOL = "index_motor_spool"
TENDON = "idx_flex_spatial"

MID_COLLISION = v28.MID_MESH_COLLISION


# ============================================================
# HARDWARE H1 CONTACT TARGET
#
# Manually checked target from the original contact experiment.
# All 3 joints are retained.
# ============================================================

HW = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


# ============================================================
# CAD DIGITIZATION
#
# Coordinate system:
#
#   origin = MCP
#   +CAD-X = toward fingertip
#   +CAD-Y = FRONT / PALMAR
#
# These are the values YOU just measured from CAD.
# ============================================================

CAD_MCP_X = 0.0

CAD_PIP_X = (
    26.06522233901073
)

CAD_DIP_X = (
    51.947755884979685
)


CAD_POINTS = [

    (
        "v35_p01_palm_entry",
        "palm",
        -10.326628414667073,
        +5.160274460361130,
    ),

    (
        "v35_p02_mcp_parent",
        "palm",
        -2.152748805859891,
        +4.710999857839114,
    ),

    (
        "v35_p03_mcp_child",
        "Index_Proximal",
        +3.517899171168670,
        +4.888825055725025,
    ),

    (
        "v35_p04_pip_parent",
        "Index_Proximal",
        +23.088398128930820,
        +4.479067163844325,
    ),

    (
        "v35_p05_pip_child",
        "Index_Middle",
        +29.385233982275587,
        +4.515848102210469,
    ),

    (
        "v35_p06_dip_parent",
        "Index_Middle",
        +48.538071706028770,
        +5.060570759666121,
    ),

    (
        "v35_p07_dip_child",
        "Index_Distal",
        +52.878222433234020,
        +4.892776726789158,
    ),

    (
        "v35_p08_distal_turn",
        "Index_Distal",
        +55.832552515718090,
        +4.695497148279827,
    ),

    (
        "v35_p09_anchor",
        "Index_Distal",
        +61.654275917275380,
        -4.511288481751857,
    ),
]


# ============================================================
# 3-D REGISTRATION OF THE 2-D CAD SECTION
#
# Actual visual/collision index geometry is centred near
# local x = -5.05 mm.
#
# We do NOT use old manually placed idx_guide_* x values.
# ============================================================

CABLE_X_LOCAL = -0.00505


# V34 structural testing established the numeric local-y
# convention for positive flexion:
#
# physical FRONT/PALMAR = negative MuJoCo local-y.
#
FRONT_TO_LOCAL_Y = -1.0


# ============================================================
# PARAMETERS FROZEN FOR THIS FIRST STRUCTURAL CONTACT TEST
# ============================================================

METERS_PER_TICK = 5.10e-6

TENDON_K = 2100.0

TENDON_DAMPING = 2.0


# Exact V33 runtime contact baseline.
CONTACT_MU = 0.05

CONTACT_K = 1000.0


AUDIT_POSES = [

    (
        5.0,
        5.0,
        2.0,
    ),

    (
        15.0,
        20.0,
        8.0,
    ),

    (
        30.0,
        35.0,
        15.0,
    ),

    (
        45.0,
        55.0,
        25.0,
    ),
]


# ============================================================
# XML HELPERS
# ============================================================


def vec3(
    text,
):

    return np.asarray(
        [
            float(x)
            for x in text.split()
        ],
        dtype=float,
    )


def fmt3(
    value,
):

    return " ".join(
        f"{float(x):.12g}"
        for x in np.asarray(
            value,
            dtype=float,
        )
    )


def quat_to_R(
    q,
):

    q = np.asarray(
        q,
        dtype=float,
    )

    q /= np.linalg.norm(
        q
    )

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


def body_named(
    root,
    name,
):

    value = root.find(
        f".//body[@name='{name}']"
    )

    if value is None:

        raise RuntimeError(
            f"Missing body: {name}"
        )

    return value


def joint_named(
    root,
    name,
):

    value = root.find(
        f".//joint[@name='{name}']"
    )

    if value is None:

        raise RuntimeError(
            f"Missing joint: {name}"
        )

    return value


def tendon_parent(
    root,
):

    value = root.find(
        ".//tendon"
    )

    if value is None:

        raise RuntimeError(
            "Missing <tendon> block."
        )

    return value


def add_site(
    body,
    name,
    pos,
    rgba="1 1 0 1",
    size="0.0015",
):

    ET.SubElement(
        body,
        "site",
        {
            "name":
                name,

            "pos":
                fmt3(
                    pos
                ),

            "size":
                size,

            "rgba":
                rgba,

            "group":
                "4",
        },
    )


def remove_generated_sites(
    root,
):

    names = {
        name
        for (
            name,
            body,
            x,
            y,
        )
        in CAD_POINTS
    }

    names.add(
        "v35_spool_site"
    )


    for parent in root.iter():

        for child in list(
            parent
        ):

            if (
                child.tag
                ==
                "site"
                and
                child.attrib.get(
                    "name"
                )
                in names
            ):

                parent.remove(
                    child
                )


def make_index_visuals_solid(
    root,
):

    for body_name in (
        "Index_Proximal",
        "Index_Middle",
        "Index_Distal",
    ):

        body = body_named(
            root,
            body_name,
        )

        for geom in body.findall(
            "geom"
        ):

            # Keep V33's special mesh collision translucent.
            if (
                geom.attrib.get(
                    "name"
                )
                ==
                MID_COLLISION
            ):

                continue


            if (
                geom.attrib.get(
                    "class"
                )
                !=
                "visual_mesh"
            ):

                continue


            rgba = geom.attrib.get(
                "rgba"
            )

            if rgba is None:

                continue


            values = rgba.split()

            if len(values) != 4:

                continue


            # VISUALIZATION ONLY:
            # make proximal/middle/distal visual meshes transparent
            # so the CAD tendon and distal anchor can be inspected.
            # Physics is unchanged.
            values[
                3
            ] = "0.12"

            geom.set(
                "rgba",
                " ".join(
                    values
                ),
            )


# ============================================================
# CAD -> MUJOCO BODY COORDINATES
# ============================================================


def finger_body_point(
    body_name,
    cad_x_mm,
    cad_y_front_mm,
):

    y = (
        FRONT_TO_LOCAL_Y
        *
        cad_y_front_mm
        *
        1e-3
    )


    if (
        body_name
        ==
        "Index_Proximal"
    ):

        z = (
            -(
                cad_x_mm
                -
                CAD_MCP_X
            )
            *
            1e-3
        )


    elif (
        body_name
        ==
        "Index_Middle"
    ):

        z = (
            -(
                cad_x_mm
                -
                CAD_PIP_X
            )
            *
            1e-3
        )


    elif (
        body_name
        ==
        "Index_Distal"
    ):

        z = (
            -(
                cad_x_mm
                -
                CAD_DIP_X
            )
            *
            1e-3
        )


    else:

        raise RuntimeError(
            f"Unsupported finger body: "
            f"{body_name}"
        )


    return np.array(
        [
            CABLE_X_LOCAL,
            y,
            z,
        ],
        dtype=float,
    )


def palm_point_from_cad(
    root,
    cad_x_mm,
    cad_y_front_mm,
):

    # A palm-side point is first represented in the same
    # MCP/proximal coordinate convention as the CAD.
    #
    # Then the fixed Index_Proximal transform maps it into
    # palm_assembly coordinates.

    prox = body_named(
        root,
        "Index_Proximal",
    )


    body_pos = vec3(
        prox.attrib.get(
            "pos",
            "0 0 0",
        )
    )


    body_quat = np.asarray(
        [
            float(x)
            for x in prox.attrib.get(
                "quat",
                "1 0 0 0",
            ).split()
        ],
        dtype=float,
    )


    R = quat_to_R(
        body_quat
    )


    conceptual = (
        finger_body_point(
            "Index_Proximal",
            cad_x_mm,
            cad_y_front_mm,
        )
    )


    return (
        body_pos
        +
        R
        @
        conceptual
    )


# ============================================================
# BUILD V35A MODEL
# ============================================================


def build_model(
    output_path,
):

    # --------------------------------------------------------
    # CRITICAL:
    #
    # Build the exact V33 parent environment.
    #
    # This preserves:
    #
    # table
    # V5 cylinder
    # real middle mesh collision
    # palm/UR geometry
    #
    # No manual recreation of the tabletop pose.
    # --------------------------------------------------------

    v28.build_model(
        output_path,
        True,
    )


    tree = ET.parse(
        output_path
    )

    root = tree.getroot()


    remove_generated_sites(
        root
    )


    make_index_visuals_solid(
        root
    )


    # VISUALIZATION ONLY:
    # make index collision geoms faint so they do not hide
    # the active tendon or cyan dorsal anchor.
    for body_name in (
        "Index_Proximal",
        "Index_Middle",
        "Index_Distal",
    ):

        body = body_named(
            root,
            body_name,
        )

        for geom in body.findall("geom"):

            name = geom.attrib.get(
                "name",
                "",
            )

            if not name.startswith(
                "collision_idx"
            ):
                continue

            rgba = geom.attrib.get(
                "rgba"
            )

            if rgba is None:
                continue

            vals = rgba.split()

            if len(vals) == 4:
                vals[3] = "0.06"
                geom.set(
                    "rgba",
                    " ".join(vals),
                )


    palm = body_named(
        root,
        "palm_assembly:1",
    )

    prox = body_named(
        root,
        "Index_Proximal",
    )

    middle = body_named(
        root,
        "Index_Middle",
    )

    distal = body_named(
        root,
        "Index_Distal",
    )

    spool_body = body_named(
        root,
        "IndexSpoolDummy",
    )


    route_names = []

    first_palm_point = None


    print()
    print(
        "CAD ROUTE -> MUJOCO BODY SITES"
    )


    for (
        name,
        body_name,
        cad_x,
        cad_y,
    ) in CAD_POINTS:


        if (
            body_name
            ==
            "palm"
        ):

            local = (
                palm_point_from_cad(
                    root,
                    cad_x,
                    cad_y,
                )
            )

            body = palm


            if (
                first_palm_point
                is None
            ):

                first_palm_point = (
                    local.copy()
                )


        else:

            local = (
                finger_body_point(
                    body_name,
                    cad_x,
                    cad_y,
                )
            )


            body = {

                "Index_Proximal":
                    prox,

                "Index_Middle":
                    middle,

                "Index_Distal":
                    distal,

            }[
                body_name
            ]


        if (
            name
            ==
            "v35_p09_anchor"
        ):

            rgba = (
                "0 1 1 1"
            )

            size = (
                "0.0050"
            )

        else:

            rgba = (
                "1 1 0 1"
            )

            size = (
                "0.0015"
            )


        add_site(
            body,
            name,
            local,
            rgba,
            size,
        )


        route_names.append(
            name
        )


        print(
            f"  {name:<24s} "
            f"body={body_name:<16s} "
            f"CAD=({cad_x:+8.3f},"
            f"{cad_y:+8.3f}) mm  "
            f"local="
            f"[{local[0]:+.6f}, "
            f"{local[1]:+.6f}, "
            f"{local[2]:+.6f}] m"
        )


    if (
        first_palm_point
        is None
    ):

        raise RuntimeError(
            "No palm entry in CAD route."
        )


    # --------------------------------------------------------
    # MOVING MOTOR SPOOL
    #
    # Move away from first cable point so positive spool
    # displacement increases tendon length.
    # --------------------------------------------------------

    add_site(
        spool_body,
        "v35_spool_site",
        np.zeros(
            3
        ),
        "1 0 1 1",
        "0.002",
    )


    spool_axis = (
        -first_palm_point
        /
        np.linalg.norm(
            first_palm_point
        )
    )


    joint_named(
        root,
        SPOOL,
    ).set(
        "axis",
        fmt3(
            spool_axis
        ),
    )


    # --------------------------------------------------------
    # REMOVE ONLY OLD INDEX FIXED TENDON.
    # --------------------------------------------------------

    parent = tendon_parent(
        root
    )


    for child in list(
        parent
    ):

        if (
            child.attrib.get(
                "name"
            )
            ==
            TENDON
        ):

            parent.remove(
                child
            )


    # --------------------------------------------------------
    # FIRST PASS:
    # zero-stiffness spatial tendon so we can measure L0.
    # --------------------------------------------------------

    spatial = ET.SubElement(
        parent,
        "spatial",
        {
            "name":
                TENDON,

            "rgba":
                "1 0 1 1",

            "width":
                "0.0015",

            "stiffness":
                "0",

            "damping":
                "0",

            "frictionloss":
                "0",
        },
    )


    ET.SubElement(
        spatial,
        "site",
        {
            "site":
                "v35_spool_site"
        },
    )


    for name in route_names:

        ET.SubElement(
            spatial,
            "site",
            {
                "site":
                    name
            },
        )


    tree.write(
        TMP_PATH,
        encoding="utf-8",
        xml_declaration=True,
    )


    model0 = (
        mujoco.MjModel.from_xml_path(
            str(
                TMP_PATH
            )
        )
    )

    data0 = (
        mujoco.MjData(
            model0
        )
    )


    for joint_name in (
        MCP,
        PIP,
        DIP,
        SPOOL,
    ):

        jid = mujoco.mj_name2id(
            model0,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )


        if jid < 0:

            raise RuntimeError(
                f"Missing joint "
                f"{joint_name}"
            )


        data0.qpos[
            int(
                model0.jnt_qposadr[
                    jid
                ]
            )
        ] = 0.0


    mujoco.mj_forward(
        model0,
        data0,
    )


    tendon_id = (
        mujoco.mj_name2id(
            model0,
            mujoco.mjtObj.mjOBJ_TENDON,
            TENDON,
        )
    )


    L0 = float(
        data0.ten_length[
            tendon_id
        ]
    )


    # --------------------------------------------------------
    # FINAL ELASTIC TENDON
    # --------------------------------------------------------

    spatial.set(
        "stiffness",
        f"{TENDON_K:.12g}",
    )


    spatial.set(
        "damping",
        f"{TENDON_DAMPING:.12g}",
    )


    spatial.set(
        "springlength",
        f"{L0:.12g}",
    )


    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )


    try:

        TMP_PATH.unlink()

    except FileNotFoundError:

        pass


    # Hard compile gate.
    mujoco.MjModel.from_xml_path(
        str(
            output_path
        )
    )


    return (
        L0,
        spool_axis,
    )


# ============================================================
# JACOBIAN / MOMENT-ARM AUDIT
# ============================================================


def joint_info(
    model,
    name,
):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )


    if jid < 0:

        raise RuntimeError(
            f"Missing joint: {name}"
        )


    return (
        int(
            model.jnt_qposadr[
                jid
            ]
        ),

        int(
            model.jnt_dofadr[
                jid
            ]
        ),
    )


def tendon_length(
    model,
    data,
    tendon_id,
):

    mujoco.mj_forward(
        model,
        data,
    )


    return float(
        data.ten_length[
            tendon_id
        ]
    )


def set_pose(
    model,
    data,
    pose_deg,
    spool=0.002,
):

    for (
        joint_name,
        angle_deg,
    ) in zip(
        (
            MCP,
            PIP,
            DIP,
        ),
        pose_deg,
    ):

        qadr, _ = joint_info(
            model,
            joint_name,
        )


        data.qpos[
            qadr
        ] = math.radians(
            float(
                angle_deg
            )
        )


    qadr, _ = joint_info(
        model,
        SPOOL,
    )


    data.qpos[
        qadr
    ] = float(
        spool
    )


    data.qvel[:] = 0.0


    mujoco.mj_forward(
        model,
        data,
    )


def moment_arms(
    model,
    data,
    pose_deg,
):

    tendon_id = (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            TENDON,
        )
    )


    set_pose(
        model,
        data,
        pose_deg,
    )


    eps = 1e-4

    arms = []


    for joint_name in (
        MCP,
        PIP,
        DIP,
    ):

        qadr, _ = joint_info(
            model,
            joint_name,
        )


        q0 = float(
            data.qpos[
                qadr
            ]
        )


        data.qpos[
            qadr
        ] = (
            q0
            +
            eps
        )


        Lp = tendon_length(
            model,
            data,
            tendon_id,
        )


        data.qpos[
            qadr
        ] = (
            q0
            -
            eps
        )


        Lm = tendon_length(
            model,
            data,
            tendon_id,
        )


        data.qpos[
            qadr
        ] = q0


        arms.append(
            -(
                Lp
                -
                Lm
            )
            /
            (
                2.0
                *
                eps
            )
        )


    qadr, _ = joint_info(
        model,
        SPOOL,
    )


    x0 = float(
        data.qpos[
            qadr
        ]
    )


    eps_x = 1e-5


    data.qpos[
        qadr
    ] = (
        x0
        +
        eps_x
    )


    Lp = tendon_length(
        model,
        data,
        tendon_id,
    )


    data.qpos[
        qadr
    ] = (
        x0
        -
        eps_x
    )


    Lm = tendon_length(
        model,
        data,
        tendon_id,
    )


    data.qpos[
        qadr
    ] = x0


    mujoco.mj_forward(
        model,
        data,
    )


    gain = (
        (
            Lp
            -
            Lm
        )
        /
        (
            2.0
            *
            eps_x
        )
    )


    return (
        np.asarray(
            arms,
            dtype=float,
        ),
        float(
            gain
        ),
    )


def audit_model():

    model = mujoco.MjModel.from_xml_path(
        str(
            MODEL_PATH
        )
    )

    data = mujoco.MjData(
        model
    )


    print()
    print("=" * 116)
    print(
        "V35A CAD SPATIAL-TENDON JACOBIAN AUDIT"
    )
    print("=" * 116)


    all_positive = True


    for pose in AUDIT_POSES:

        arms, gain = moment_arms(
            model,
            data,
            pose,
        )


        all_positive = (
            all_positive
            and
            bool(
                np.all(
                    arms
                    >
                    0.0
                )
            )
        )


        print(
            f" q="
            f"{pose[0]:5.1f}/"
            f"{pose[1]:5.1f}/"
            f"{pose[2]:5.1f} deg "
            f" arms="
            f"{arms[0]*1000:+7.3f}/"
            f"{arms[1]*1000:+7.3f}/"
            f"{arms[2]*1000:+7.3f} mm "
            f" dL/dx="
            f"{gain:+.6f}"
        )


    if not all_positive:

        raise RuntimeError(
            "CAD spatial tendon has a non-flexion "
            "moment arm. STOP before contact replay."
        )


# ============================================================
# CONTACT REPLAY — EXACT V33 ENVIRONMENT
# ============================================================


def run_contact(
    contact_bag,
):

    (
        L0,
        spool_axis,
    ) = build_model(
        MODEL_PATH
    )


    audit_model()


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # NO StatefulTransmission is installed here.
    #
    # Only the CAD spatial tendon generates index
    # transmission torques.
    # --------------------------------------------------------


    clamp = v30.FixedSupport()

    clamp.install()


    recorder = v28.Recorder(
        MID_COLLISION
    )

    recorder.install()


    old_scale = (
        v30.v24.base.METERS_PER_TICK
    )

    old_tendon = (
        v30.v24.base.TENDON_STIFFNESS
    )

    old_mu = (
        v30.v24.base.CONTACT_MU
    )

    old_contact_k = (
        v30.v24.base.CONTACT_STIFFNESS
    )


    try:

        v30.v24.base.METERS_PER_TICK = (
            METERS_PER_TICK
        )

        v30.v24.base.TENDON_STIFFNESS = (
            TENDON_K
        )

        v30.v24.base.CONTACT_MU = (
            CONTACT_MU
        )

        v30.v24.base.CONTACT_STIFFNESS = (
            CONTACT_K
        )


        metrics, _ = (
            v30.v24.base.run_contact(
                MODEL_PATH,
                contact_bag,
            )
        )


    finally:

        recorder.restore()

        clamp.restore()


        v30.v24.base.METERS_PER_TICK = (
            old_scale
        )

        v30.v24.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v30.v24.base.CONTACT_MU = (
            old_mu
        )

        v30.v24.base.CONTACT_STIFFNESS = (
            old_contact_k
        )


    retention = v28.analyse(
        recorder.rows
    )


    return (
        metrics,
        retention,
        L0,
        spool_axis,
    )


# ============================================================
# GUI
# ============================================================


def visual_replay(
    contact_bag,
):

    # Rebuild identical V35A XML.
    build_model(
        MODEL_PATH
    )


    clamp = v30.FixedSupport()

    clamp.install()


    old_scale = (
        v30.v24.viewer_tools.base.METERS_PER_TICK
    )

    old_tendon = (
        v30.v24.viewer_tools.base.TENDON_STIFFNESS
    )

    old_mu = (
        v30.v24.viewer_tools.base.CONTACT_MU
    )

    old_contact_k = (
        v30.v24.viewer_tools.base.CONTACT_STIFFNESS
    )


    try:

        v30.v24.viewer_tools.base.METERS_PER_TICK = (
            METERS_PER_TICK
        )

        v30.v24.viewer_tools.base.TENDON_STIFFNESS = (
            TENDON_K
        )

        v30.v24.viewer_tools.base.CONTACT_MU = (
            CONTACT_MU
        )

        v30.v24.viewer_tools.base.CONTACT_STIFFNESS = (
            CONTACT_K
        )


        v30.v24.viewer_tools.visual_replay(
            MODEL_PATH,
            contact_bag,
        )


    finally:

        clamp.restore()


        v30.v24.viewer_tools.base.METERS_PER_TICK = (
            old_scale
        )

        v30.v24.viewer_tools.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v30.v24.viewer_tools.base.CONTACT_MU = (
            old_mu
        )

        v30.v24.viewer_tools.base.CONTACT_STIFFNESS = (
            old_contact_k
        )


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "contact_bag",
        type=Path,
    )


    parser.add_argument(
        "--no-viewer",
        action="store_true",
    )


    args = parser.parse_args()


    contact_bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )


    if not contact_bag.exists():

        raise RuntimeError(
            f"Missing contact bag:\n"
            f"{contact_bag}"
        )


    # Same replay patch used by V33/V30.
    #
    # This is what preserves the known-good experiment
    # initialization.  We do NOT manually recreate UR pose.
    v30.v24.ab.patch_replay_modules()


    print()
    print("=" * 116)
    print(
        "V35A — CAD-MEASURED SPATIAL TENDON "
        "IN EXACT V33 CONTACT ENVIRONMENT"
    )
    print("=" * 116)


    print()
    print(
        "Frozen from V33:"
    )

    print(
        "  table/environment = V33"
    )

    print(
        "  cylinder = frozen V5 reference"
    )

    print(
        "  actual middle phalanx mesh collision = ON"
    )

    print(
        "  upstream support = FIXED at replay initialization"
    )

    print(
        "  contact mu = 0.05"
    )

    print(
        "  contact k = 1000 N/m"
    )


    print()
    print(
        "Removed:"
    )

    print(
        "  fixed 5/4/1 transmission"
    )

    print(
        "  V24/V33 post-yield moment-arm switching"
    )

    print(
        "  load-triggered PIP breakaway law"
    )


    print()
    print(
        "New:"
    )

    print(
        "  CAD-measured FRONT/PALMAR spatial route"
    )

    print(
        "  CAD-measured distal turnaround"
    )

    print(
        "  CAD-measured BACK/DORSAL distal anchor"
    )


    (
        metrics,
        retention,
        L0,
        spool_axis,
    ) = run_contact(
        contact_bag
    )


    pose = np.asarray(
        metrics[
            "pose"
        ],
        dtype=float,
    )


    rmse = float(
        np.sqrt(
            np.mean(
                (
                    pose
                    -
                    HW
                )
                **
                2
            )
        )
    )


    print()
    print("=" * 116)
    print(
        "V35A RESULT"
    )
    print("=" * 116)


    print()
    print(
        f"Open CAD tendon length L0 = "
        f"{L0*1000:.3f} mm"
    )


    print(
        "Spool axis = "
        f"[{spool_axis[0]:+.6f}, "
        f"{spool_axis[1]:+.6f}, "
        f"{spool_axis[2]:+.6f}]"
    )


    print()
    print(
        "Hardware H1 MCP/PIP/DIP:"
    )

    print(
        f"  {HW[0]:.3f} / "
        f"{HW[1]:.3f} / "
        f"{HW[2]:.3f} deg"
    )


    print(
        "Simulation:"
    )

    print(
        f"  {pose[0]:.3f} / "
        f"{pose[1]:.3f} / "
        f"{pose[2]:.3f} deg"
    )


    print(
        f"Angle RMSE = "
        f"{rmse:.3f} deg"
    )


    print()
    print(
        "Contact retention:"
    )

    print(
        f"  middle total = "
        f"{retention['middle_duration_s']:.3f} s"
    )

    print(
        f"  tip total = "
        f"{retention['tip_duration_s']:.3f} s"
    )

    print(
        f"  middle+tip overlap = "
        f"{retention['overlap_s']:.3f} s"
    )

    print(
        f"  longest overlap = "
        f"{retention['longest_overlap_s']:.3f} s"
    )

    print(
        f"  middle after first tip = "
        f"{retention['middle_after_tip_s']:.3f} s"
    )

    print(
        f"  effective mu = "
        f"{retention['effective_mu']:.3f}"
    )


    print()
    print("=" * 116)
    print(
        "V35A DECISION RULE"
    )
    print("=" * 116)


    print()
    print(
        "DO NOT accept/reject this model from angle RMSE alone."
    )

    print(
        "The GUI must show contact causing the configuration."
    )

    print(
        "If a phalanx touches then retreats while the finger "
        "continues toward a numerical target, V35A FAILS."
    )


    if not args.no_viewer:

        print()
        print("=" * 116)
        print(
            "V35A GUI"
        )
        print("=" * 116)


        print()
        print(
            "MAGENTA = active CAD spatial tendon"
        )

        print(
            "YELLOW = CAD route points"
        )

        print(
            "CYAN = measured backside distal anchor"
        )

        print(
            "Cylinder/table/hand initialization = V33 replay"
        )


        visual_replay(
            contact_bag
        )


    print()
    print("=" * 116)
    print(
        "V35A COMPLETE"
    )
    print("=" * 116)


if __name__ == "__main__":

    main()
