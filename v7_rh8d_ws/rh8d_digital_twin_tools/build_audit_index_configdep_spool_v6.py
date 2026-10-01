#!/usr/bin/env python3

from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


# ============================================================
# PATHS
# ============================================================

WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()

MJCF_DIR = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

SOURCE = (
    MJCF_DIR
    / "ur5e_rh8d_index_configdep_geom_v5.xml"
)

OUTPUT = (
    MJCF_DIR
    / "ur5e_rh8d_index_configdep_spool_v6.xml"
)


# ============================================================
# NAMES
# ============================================================

TENDON = "idx_flex_spatial"

SPOOL_BODY = "IndexSpoolDummy"
SPOOL_JOINT = "index_motor_spool"

SPOOL_SITE = "cfg2_idx_spool"
ORIGIN_SITE = "cfg2_idx_origin"

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"


# ============================================================
# CURRENT PROVISIONAL MECHANICAL PARAMETERS
#
# These are NOT declared final.
#
# Candidate-A tendon stiffness:
#   2100 N/m
#
# Old fixed-tendon damping:
#   2 N s/m
#
# The spring-length interval will be SHIFTED so that the
# spatial tendon is just at the zero-tension boundary at the
# open reference configuration.
#
# This reproduces the meaning of the old:
#
#   springlength="-0.050 0.0"
#
# without incorrectly applying a negative absolute rest
# length to a spatial tendon whose geometric length is > 0.
# ============================================================

TENDON_STIFFNESS = 2100.0
TENDON_DAMPING = 2.0

DEADBAND_WIDTH = 0.050


# ============================================================
# SPOOL GEOMETRY
#
# At spool = 0, place the moving spool site 20 mm in +x
# relative to the fixed palm entry point.
#
# The slide axis is +x.
#
# Therefore positive spool travel increases the straight
# spool-entry segment approximately one-for-one:
#
#       dL / d(spool) ≈ +1
#
# matching the old fixed-tendon coefficient +1.
# ============================================================

SPOOL_SEGMENT_OPEN = 0.020


# ============================================================
# CONFIGURATIONS FOR AUDIT
# ============================================================

Q_OPEN = np.array(
    [0.0, 0.0, 0.0]
)

Q_HW_CONTACT = np.array(
    [14.5355, 23.8110, 9.3270]
)


# ============================================================
# XML HELPERS
# ============================================================

def find_body(root, name):

    for body in root.iter("body"):

        if body.attrib.get("name") == name:
            return body

    raise RuntimeError(
        f"Body not found: {name}"
    )


def find_site(root, name):

    for site in root.iter("site"):

        if site.attrib.get("name") == name:
            return site

    raise RuntimeError(
        f"Site not found: {name}"
    )


def find_tendon(root, name):

    for section in root.findall("tendon"):

        for element in list(section):

            if (
                element.attrib.get("name")
                == name
            ):

                return element

    raise RuntimeError(
        f"Tendon not found: {name}"
    )


def parse_pos(element):

    return np.array(
        [
            float(x)
            for x in element.attrib[
                "pos"
            ].split()
        ],
        dtype=float,
    )


# ============================================================
# MUJOCO HELPERS
# ============================================================

def obj_id(model, objtype, name):

    idx = mujoco.mj_name2id(
        model,
        objtype,
        name,
    )

    if idx < 0:

        raise RuntimeError(
            f"Missing MuJoCo object: {name}"
        )

    return idx


def joint_adr(model, name):

    jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return int(
        model.jnt_qposadr[jid]
    )


def set_finger_configuration(
    model,
    data,
    qpos_reference,
    addresses,
    q_deg,
):

    data.qpos[:] = qpos_reference
    data.qvel[:] = 0.0


    for adr, deg in zip(
        addresses,
        q_deg,
    ):

        data.qpos[adr] = math.radians(
            float(deg)
        )


    mujoco.mj_forward(
        model,
        data,
    )


def tendon_length(
    model,
    data,
    tid,
):

    mujoco.mj_forward(
        model,
        data,
    )

    return float(
        data.ten_length[
            tid
        ]
    )


def derivative_mm_per_rad(
    model,
    data,
    qpos_reference,
    addresses,
    tid,
    q_deg,
):

    set_finger_configuration(
        model,
        data,
        qpos_reference,
        addresses,
        q_deg,
    )


    base = data.qpos.copy()

    eps = 1.0e-5

    result = []


    for adr in addresses:

        q0 = float(
            base[adr]
        )


        if abs(q0) < 2.0 * eps:

            data.qpos[:] = base

            mujoco.mj_forward(
                model,
                data,
            )

            L0 = float(
                data.ten_length[
                    tid
                ]
            )


            data.qpos[
                adr
            ] = q0 + eps

            mujoco.mj_forward(
                model,
                data,
            )

            L1 = float(
                data.ten_length[
                    tid
                ]
            )


            d = (
                L1 - L0
            ) / eps


        else:

            data.qpos[:] = base

            data.qpos[
                adr
            ] = q0 + eps

            mujoco.mj_forward(
                model,
                data,
            )

            Lp = float(
                data.ten_length[
                    tid
                ]
            )


            data.qpos[:] = base

            data.qpos[
                adr
            ] = q0 - eps

            mujoco.mj_forward(
                model,
                data,
            )

            Lm = float(
                data.ten_length[
                    tid
                ]
            )


            d = (
                Lp - Lm
            ) / (
                2.0 * eps
            )


        result.append(
            d * 1000.0
        )


    data.qpos[:] = base

    mujoco.mj_forward(
        model,
        data,
    )


    return np.asarray(
        result,
        dtype=float,
    )


def spool_derivative(
    model,
    data,
    qpos_reference,
    spool_adr,
    tid,
):

    data.qpos[:] = qpos_reference
    data.qvel[:] = 0.0

    data.qpos[
        spool_adr
    ] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


    L0 = float(
        data.ten_length[
            tid
        ]
    )


    eps = 1.0e-5

    data.qpos[
        spool_adr
    ] = eps

    mujoco.mj_forward(
        model,
        data,
    )


    L1 = float(
        data.ten_length[
            tid
        ]
    )


    data.qpos[:] = qpos_reference

    mujoco.mj_forward(
        model,
        data,
    )


    return (
        L1 - L0
    ) / eps


# ============================================================
# BUILD FIRST-PASS SPATIAL SPOOL MODEL
# ============================================================

def build_geometry():

    if not SOURCE.exists():

        raise RuntimeError(
            f"Missing V5 source:\n"
            f"{SOURCE}"
        )


    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()


    origin = find_site(
        root,
        ORIGIN_SITE,
    )


    origin_pos = parse_pos(
        origin
    )


    spool_body = find_body(
        root,
        SPOOL_BODY,
    )


    # Remove an old copy if script is ever rerun on a model
    # that already contains the site.
    for site in list(
        spool_body.findall(
            "site"
        )
    ):

        if (
            site.attrib.get(
                "name"
            )
            == SPOOL_SITE
        ):

            spool_body.remove(
                site
            )


    spool_pos = origin_pos.copy()

    spool_pos[0] += (
        SPOOL_SEGMENT_OPEN
    )


    ET.SubElement(
        spool_body,
        "site",
        {
            "name":
                SPOOL_SITE,

            "pos":
                " ".join(
                    f"{x:.10g}"
                    for x in spool_pos
                ),

            "size":
                "0.0018",

            "rgba":
                "0 0.8 1 1",
        },
    )


    tendon = find_tendon(
        root,
        TENDON,
    )


    if tendon.tag != "spatial":

        raise RuntimeError(
            f"{TENDON} is not spatial "
            f"in V5."
        )


    # Remove existing spool reference if any.
    for child in list(
        tendon
    ):

        if (
            child.tag == "site"
            and
            child.attrib.get(
                "site"
            )
            == SPOOL_SITE
        ):

            tendon.remove(
                child
            )


    # Prepend moving spool site.
    tendon.insert(
        0,
        ET.Element(
            "site",
            {
                "site":
                    SPOOL_SITE
            },
        ),
    )


    # First pass: no elastic force yet.
    tendon.attrib.pop(
        "springlength",
        None,
    )

    tendon.set(
        "stiffness",
        "0",
    )

    tendon.set(
        "damping",
        f"{TENDON_DAMPING:.10g}",
    )


    tree.write(
        OUTPUT,
        encoding="unicode",
        xml_declaration=True,
    )


# ============================================================
# SHIFT SPRING LENGTH TO SPATIAL OPEN LENGTH
# ============================================================

def apply_elasticity():

    # Compile geometry-only model.
    model = mujoco.MjModel.from_xml_path(
        str(
            OUTPUT
        )
    )

    data = mujoco.MjData(
        model
    )

    mujoco.mj_forward(
        model,
        data,
    )


    tid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON,
    )


    open_length = float(
        data.ten_length[
            tid
        ]
    )


    lower = (
        open_length
        - DEADBAND_WIDTH
    )

    upper = (
        open_length
    )


    tree = ET.parse(
        OUTPUT
    )

    root = tree.getroot()

    tendon = find_tendon(
        root,
        TENDON,
    )


    tendon.set(
        "stiffness",
        f"{TENDON_STIFFNESS:.10g}",
    )

    tendon.set(
        "damping",
        f"{TENDON_DAMPING:.10g}",
    )

    tendon.set(
        "springlength",
        (
            f"{lower:.10g} "
            f"{upper:.10g}"
        ),
    )


    tree.write(
        OUTPUT,
        encoding="unicode",
        xml_declaration=True,
    )


    return (
        open_length,
        lower,
        upper,
    )


# ============================================================
# FINAL AUDIT
# ============================================================

def audit(
    initial_open_length,
    spring_lower,
    spring_upper,
):

    model = mujoco.MjModel.from_xml_path(
        str(
            OUTPUT
        )
    )

    data = mujoco.MjData(
        model
    )


    qpos_reference = (
        model.qpos0.copy()
    )


    addresses = [
        joint_adr(
            model,
            MCP,
        ),
        joint_adr(
            model,
            PIP,
        ),
        joint_adr(
            model,
            DIP,
        ),
    ]


    spool_adr = joint_adr(
        model,
        SPOOL_JOINT,
    )


    tid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON,
    )


    set_finger_configuration(
        model,
        data,
        qpos_reference,
        addresses,
        Q_OPEN,
    )


    final_open_length = float(
        data.ten_length[
            tid
        ]
    )


    open_arms = (
        derivative_mm_per_rad(
            model,
            data,
            qpos_reference,
            addresses,
            tid,
            Q_OPEN,
        )
    )


    contact_arms = (
        derivative_mm_per_rad(
            model,
            data,
            qpos_reference,
            addresses,
            tid,
            Q_HW_CONTACT,
        )
    )


    spool_dldx = spool_derivative(
        model,
        data,
        qpos_reference,
        spool_adr,
        tid,
    )


    # --------------------------------------------------------
    # Structural pass gates
    # --------------------------------------------------------

    spool_ok = (
        0.995
        <= spool_dldx
        <= 1.005
    )


    signs_ok = bool(
        np.all(
            open_arms < 0.0
        )
        and
        np.all(
            contact_arms < 0.0
        )
    )


    pip_growth = (
        abs(
            contact_arms[1]
        )
        /
        abs(
            open_arms[1]
        )
    )


    growth_ok = (
        pip_growth
        >= 1.20
    )


    length_ok = (
        abs(
            final_open_length
            - initial_open_length
        )
        < 1.0e-7
    )


    passed = (
        spool_ok
        and
        signs_ok
        and
        growth_ok
        and
        length_ok
    )


    print()
    print("=" * 110)
    print(
        "RH8D INDEX — CONFIGURATION-DEPENDENT "
        "SPOOL/TENDON V6"
    )
    print("=" * 110)


    print()
    print(
        "Spatial tendon open length:"
    )

    print(
        f"  {final_open_length*1000.0:.3f} mm"
    )


    print()
    print(
        "Shifted spring-length interval:"
    )

    print(
        f"  lower = "
        f"{spring_lower*1000.0:.3f} mm"
    )

    print(
        f"  upper = "
        f"{spring_upper*1000.0:.3f} mm"
    )


    print()
    print(
        "Spool transmission:"
    )

    print(
        f"  dL/dspool = "
        f"{spool_dldx:.6f} m/m"
    )

    print(
        f"  target     = +1.000000"
    )


    print()
    print(
        f"{'STATE':<16}"
        f"{'MCP mm/rad':>14}"
        f"{'PIP mm/rad':>14}"
        f"{'DIP mm/rad':>14}"
    )

    print(
        "-" * 60
    )


    print(
        f"{'OPEN':<16}"
        f"{open_arms[0]:>14.3f}"
        f"{open_arms[1]:>14.3f}"
        f"{open_arms[2]:>14.3f}"
    )


    print(
        f"{'HW_CONTACT':<16}"
        f"{contact_arms[0]:>14.3f}"
        f"{contact_arms[1]:>14.3f}"
        f"{contact_arms[2]:>14.3f}"
    )


    print()
    print(
        f"PIP contact/open leverage = "
        f"{pip_growth:.3f} x"
    )


    print()
    print("=" * 110)
    print(
        "STRUCTURAL CHECK"
    )
    print("=" * 110)


    print()
    print(
        f"Spool dL/dx approximately +1 : "
        f"{spool_ok}"
    )

    print(
        f"Correct joint flexion signs   : "
        f"{signs_ok}"
    )

    print(
        f"PIP leverage growth retained  : "
        f"{growth_ok}"
    )

    print(
        f"Spring reference consistent   : "
        f"{length_ok}"
    )


    print()
    print(
        "Output model:"
    )

    print(
        OUTPUT
    )


    print()
    print("=" * 110)


    if passed:

        print(
            "RESULT: PASS — moving motor spool is now "
            "physically coupled to the configuration-"
            "dependent spatial tendon."
        )

        print()
        print(
            "NEXT: run exact hardware free-space and "
            "contact encoder replays. Do not manually "
            "impose MCP/PIP/DIP angles."
        )

    else:

        print(
            "RESULT: DO NOT REPLAY YET — spool/tendon "
            "coupling failed structural checks."
        )


    print()
    print("=" * 110)


def main():

    build_geometry()

    (
        open_length,
        spring_lower,
        spring_upper,
    ) = apply_elasticity()


    audit(
        open_length,
        spring_lower,
        spring_upper,
    )


if __name__ == "__main__":
    main()
