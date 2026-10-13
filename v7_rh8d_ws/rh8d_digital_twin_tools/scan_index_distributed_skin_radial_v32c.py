#!/usr/bin/env python3

from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import scan_index_distributed_skin_v32 as v32


# ============================================================
# V32C — CORRECTED RADIAL DISTRIBUTED SKIN
#
# V32B measured the actual middle/bottle contact path:
#
# local x:
#   median = -6.025 mm
#
# local y:
#   median = -6.363 mm
#
# local z:
#   range = -13.046 ... -10.590 mm
#
# Existing middle capsule centerline:
#   x ~= -5.03 mm
#   y = 0
#
# Therefore contact is predominantly on the NEGATIVE-Y side,
# not the negative-X side used accidentally in V32.
#
# V32C changes ONLY skin placement/orientation.
#
# All corrected V30/V31 mechanics remain frozen.
# ============================================================


WS = v32.WS
MJCF_DIR = v32.MJCF_DIR


RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "distributed_skin_radial_v32c"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


v32.RESULT_DIR = RESULT_DIR

v32.SUMMARY = (
    RESULT_DIR
    / "summary.csv"
)

v32.SELECTED_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_distributed_skin_radial_v32c_candidate.xml"
)


# ============================================================
# MEASURED CONTACT PATH FROM V32B
# ============================================================

CONTACT_MEDIAN_X = -0.006025
CONTACT_MEDIAN_Y = -0.006363


# Three zones covering:
#
# measured z:
#   -13.046 ... -10.590 mm
#
# plus modest overlap/margin.
#
v32.PATCH_Z = [
    -0.01310,
    -0.01180,
    -0.01050,
]


# These labels are retained because V32's recorder already
# uses prox/mid/dist field names. In V32C they simply identify
# the three measured z zones.
v32.PATCH_NAMES = [
    "prox",
    "mid",
    "dist",
]


# ============================================================
# SMALL SKIN GEOMETRY
# ============================================================

# Full radial thickness = 0.8 mm.
v32.SKIN_HALF_THICKNESS = 0.00040

# Tangential half-width.
#
# Once oriented correctly, the measured contact path is very
# narrow tangentially. 2.5 mm gives generous coverage without
# creating a huge artificial pad.
v32.SKIN_HALF_WIDTH = 0.00250

# Full z length = 2.0 mm per element.
#
# Three overlapping elements span the measured path.
v32.SKIN_HALF_LENGTH = 0.00100

# Maximum inward compression.
v32.SKIN_TRAVEL = 0.00120

v32.SKIN_DAMPING = 0.75

v32.SKIN_MASS = 0.00020


# ============================================================
# SMALL STIFFNESS TEST ONLY
# ============================================================

v32.CASES = [

    {
        "name":
            "NO_SKIN_CONTROL",

        "skin":
            False,

        "k":
            -1.0,
    },

    {
        "name":
            "RADIAL_SKIN_K250",

        "skin":
            True,

        "k":
            250.0,
    },

    {
        "name":
            "RADIAL_SKIN_K500",

        "skin":
            True,

        "k":
            500.0,
    },

    {
        "name":
            "RADIAL_SKIN_K1000",

        "skin":
            True,

        "k":
            1000.0,
    },
]


# ============================================================
# CORRECTED MODEL BUILDER
# ============================================================

def build_model_radial(
    output,
    case,
):

    # Exact corrected V30/V31 base model.
    v32.v30.v24.build_model(
        output
    )


    if not case[
        "skin"
    ]:

        mujoco.MjModel.from_xml_path(
            str(
                output
            )
        )

        return


    tree = ET.parse(
        output
    )

    root = tree.getroot()


    middle_body, core = (
        v32.middle_body_and_core(
            root
        )
    )


    # ========================================================
    # RIGID CAPSULE CENTERLINE
    # ========================================================

    fromto = np.asarray(
        [
            float(value)
            for value in
            core.attrib[
                "fromto"
            ].split()
        ],
        dtype=float,
    )


    p1 = fromto[
        :3
    ]

    p2 = fromto[
        3:
    ]


    centerline = (
        0.5
        *
        (
            p1
            +
            p2
        )
    )


    center_x = float(
        centerline[
            0
        ]
    )

    center_y = float(
        centerline[
            1
        ]
    )


    radius = float(
        core.attrib[
            "size"
        ].split()[
            0
        ]
    )


    # ========================================================
    # MEASURED RADIAL DIRECTION
    # ========================================================

    radial = np.array(
        [
            CONTACT_MEDIAN_X
            -
            center_x,

            CONTACT_MEDIAN_Y
            -
            center_y,
        ],
        dtype=float,
    )


    radial /= (
        np.linalg.norm(
            radial
        )
    )


    nx = float(
        radial[
            0
        ]
    )

    ny = float(
        radial[
            1
        ]
    )


    yaw = math.atan2(
        ny,
        nx,
    )


    yaw_deg = math.degrees(
        yaw
    )


    # Body local +x points radially OUTWARD.
    #
    # Patch center is halfway through the 0.8-mm layer:
    #
    # rigid radius = 7.0 mm
    #
    # inner skin surface = 7.0 mm
    # outer skin surface = 7.8 mm
    #
    radial_center = (
        radius
        +
        v32.SKIN_HALF_THICKNESS
    )


    patch_x = (
        center_x
        +
        nx
        *
        radial_center
    )


    patch_y = (
        center_y
        +
        ny
        *
        radial_center
    )


    # Quaternion corresponding to rotation about local z.
    qw = math.cos(
        yaw
        /
        2.0
    )

    qz = math.sin(
        yaw
        /
        2.0
    )


    print()
    print("=" * 100)

    print(
        f"[V32C BUILD] "
        f"{case['name']}"
    )

    print("=" * 100)


    print(
        f"capsule center xy = "
        f"[{1000*center_x:+.3f}, "
        f"{1000*center_y:+.3f}] mm"
    )


    print(
        f"measured contact median xy = "
        f"[{1000*CONTACT_MEDIAN_X:+.3f}, "
        f"{1000*CONTACT_MEDIAN_Y:+.3f}] mm"
    )


    print(
        f"radial direction = "
        f"[{nx:+.4f}, {ny:+.4f}]"
    )


    print(
        f"radial angle = "
        f"{yaw_deg:+.2f} deg"
    )


    print(
        f"patch center xy = "
        f"[{1000*patch_x:+.3f}, "
        f"{1000*patch_y:+.3f}] mm"
    )


    print(
        f"skin thickness = "
        f"{2000*v32.SKIN_HALF_THICKNESS:.3f} mm"
    )


    print(
        f"tangential full width = "
        f"{2000*v32.SKIN_HALF_WIDTH:.3f} mm"
    )


    print(
        f"element stiffness = "
        f"{case['k']:.1f} N/m"
    )


    # ========================================================
    # BUILD THREE INDEPENDENT COMPLIANT PATCHES
    # ========================================================

    for label, z in zip(
        v32.PATCH_NAMES,
        v32.PATCH_Z,
    ):

        body_name = (
            f"idx_mid_skin_{label}_body"
        )

        joint_name = (
            f"idx_mid_skin_{label}_slide"
        )

        geom_name = (
            f"collision_idx_mid_skin_{label}"
        )


        patch_body = ET.SubElement(
            middle_body,
            "body",
            {

                "name":
                    body_name,

                "pos":
                    (
                        f"{patch_x:.9f} "
                        f"{patch_y:.9f} "
                        f"{z:.9f}"
                    ),

                # Rotate local +x onto measured radial
                # direction.
                "quat":
                    (
                        f"{qw:.12f} "
                        f"0 "
                        f"0 "
                        f"{qz:.12f}"
                    ),
            },
        )


        # ----------------------------------------------------
        # Positive q moves INWARD toward the rigid core.
        #
        # Local +x points outward, therefore slide axis is -x.
        # ----------------------------------------------------

        ET.SubElement(
            patch_body,
            "joint",
            {

                "name":
                    joint_name,

                "type":
                    "slide",

                "axis":
                    "-1 0 0",

                "limited":
                    "true",

                "range":
                    (
                        f"0 "
                        f"{v32.SKIN_TRAVEL:.9f}"
                    ),

                "springref":
                    "0",

                "stiffness":
                    f"{case['k']:.9f}",

                "damping":
                    f"{v32.SKIN_DAMPING:.9f}",

                "armature":
                    "0.000001",
            },
        )


        # ----------------------------------------------------
        # Local box axes:
        #
        # x = radial thickness
        # y = tangential width
        # z = longitudinal finger direction
        # ----------------------------------------------------

        ET.SubElement(
            patch_body,
            "geom",
            {

                "name":
                    geom_name,

                "type":
                    "box",

                "size":
                    (
                        f"{v32.SKIN_HALF_THICKNESS:.9f} "
                        f"{v32.SKIN_HALF_WIDTH:.9f} "
                        f"{v32.SKIN_HALF_LENGTH:.9f}"
                    ),

                "mass":
                    f"{v32.SKIN_MASS:.9f}",

                "contype":
                    "2",

                "conaffinity":
                    "33",

                "condim":
                    "4",

                # Keep V31 validated friction.
                "friction":
                    "0.05 0.005 0.0001",

                # Match current runtime contact stiffness
                # directly rather than using the old
                # 0.05/1 reference formulation.
                "solref":
                    "-1000 -1",

                "solimp":
                    "0.80 0.95 0.001",

                "margin":
                    "0.001",

                # Faint cyan.
                "rgba":
                    "0 0.8 1 0.12",
            },
        )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


    model = mujoco.MjModel.from_xml_path(
        str(
            output
        )
    )


    print()
    print(
        "Compiled skin elements:"
    )


    for label in v32.PATCH_NAMES:

        geom_name = (
            f"collision_idx_mid_skin_{label}"
        )


        gid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )


        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"idx_mid_skin_{label}_slide",
        )


        if (
            gid < 0
            or
            jid < 0
        ):

            raise RuntimeError(
                f"Failed to compile skin element "
                f"{label}"
            )


        print(
            f"  {label}: "
            f"geom={gid}, joint={jid}"
        )


# ============================================================
# REPLACE ONLY THE BROKEN V32 BUILDER
# ============================================================

v32.build_model = (
    build_model_radial
)


# ============================================================
# RUN EXISTING TEST / RECORDER / GUI INFRASTRUCTURE
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 122)
    print(
        "V32C — CORRECTED RADIAL DISTRIBUTED SKIN"
    )
    print("=" * 122)

    print()
    print(
        "V32B measured 0.0% coverage for the old skin."
    )

    print(
        "V32C rotates/repositions the patches onto the "
        "measured contact path."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Inner V32 headings below are reused from the "
        "validated experiment runner."
    )

    print(
        "The skin geometry itself is V32C."
    )

    v32.main()
