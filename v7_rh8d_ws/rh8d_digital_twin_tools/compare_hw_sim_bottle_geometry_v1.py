#!/usr/bin/env python3

from pathlib import Path
import math

import mujoco
import numpy as np

import fit_index_contact_v1 as base


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    base.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_v1.xml"
)


# ============================================================
# Candidate A
#
# Used only to establish the same OPEN equilibrium that we
# have used in our previous Candidate-A simulations.
# ============================================================

TENDON_STIFFNESS = 2100.0
OPEN_ENCODER = 100.0
SCALE = 5.1e-6

TENDON_NAME = "idx_flex_spatial"

SPOOL_NAME = "index_motor_spool"
ACTUATOR_NAME = "pos_index_tendon"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"

FIXTURE_NAME = "index_contact_fixture"


# ============================================================
# HARDWARE GEOMETRY
#
# Derived from:
#
# measured bottle diameter       = 69.2 mm
# good bottle-center measurement = 2026-09-14
# mean OPEN pink orientation     = 17.4695 deg
#
# These values describe the STATIC bottle geometry relative
# to the hand in its OPEN reference configuration.
# ============================================================

HW_ALONG_MM = 8.202606
HW_SIDE_MM = 51.836696

HW_CENTER_DISTANCE_MM = 52.481671

BOTTLE_RADIUS_MM = 34.6

HW_SURFACE_DISTANCE_MM = (
    HW_CENTER_DISTANCE_MM
    - BOTTLE_RADIUS_MM
)


def oid(
    model,
    typ,
    name,
):

    idx = mujoco.mj_name2id(
        model,
        typ,
        name,
    )

    if idx < 0:

        raise RuntimeError(
            f"Missing MuJoCo object: {name}"
        )

    return idx


def unit(v):

    v = np.asarray(
        v,
        dtype=float,
    )

    n = np.linalg.norm(v)

    if n < 1e-12:

        raise RuntimeError(
            "Cannot normalize zero vector."
        )

    return v / n


def main():

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(
        model
    )


    # Same isolated system-ID condition.
    model.opt.gravity[:] = 0.0


    # --------------------------------------------------------
    # Candidate-A tendon
    # --------------------------------------------------------

    tendon_id = oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    model.tendon_stiffness[
        tendon_id
    ] = TENDON_STIFFNESS


    # --------------------------------------------------------
    # Spool
    # --------------------------------------------------------

    spool_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_NAME,
    )

    spool_qadr = (
        model.jnt_qposadr[
            spool_jid
        ]
    )

    spool_dofadr = (
        model.jnt_dofadr[
            spool_jid
        ]
    )


    actuator_id = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAME,
    )


    # At OPEN_ENCODER = 100, spool displacement is zero.
    open_spool = max(
        0.0,
        (
            OPEN_ENCODER
            - 100.0
        )
        * SCALE,
    )


    # --------------------------------------------------------
    # Establish Candidate-A OPEN equilibrium
    # --------------------------------------------------------

    mujoco.mj_resetData(
        model,
        data,
    )


    for _ in range(
        int(
            2.0
            / model.opt.timestep
        )
    ):

        data.qpos[
            spool_qadr
        ] = open_spool

        data.qvel[
            spool_dofadr
        ] = 0.0

        data.ctrl[
            actuator_id
        ] = open_spool

        mujoco.mj_step(
            model,
            data,
        )


    # --------------------------------------------------------
    # Joint anchors
    #
    # These give us the physical MCP and PIP joint centers in
    # world coordinates.
    # --------------------------------------------------------

    mcp_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        MCP_NAME,
    )

    pip_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        PIP_NAME,
    )


    mcp = (
        data.xanchor[
            mcp_jid
        ].copy()
    )

    pip = (
        data.xanchor[
            pip_jid
        ].copy()
    )


    # --------------------------------------------------------
    # Bottle geometry
    # --------------------------------------------------------

    fixture_gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FIXTURE_NAME,
    )


    bottle_center = (
        data.geom_xpos[
            fixture_gid
        ].copy()
    )


    # Cylinder local Z is its long axis.
    R = (
        data.geom_xmat[
            fixture_gid
        ].reshape(
            3,
            3,
        )
    )


    bottle_axis = unit(
        R[:, 2]
    )


    # --------------------------------------------------------
    # Construct OPEN finger-plane basis
    # --------------------------------------------------------

    forward_raw = (
        pip - mcp
    )


    # Remove any component along the bottle axis.
    #
    # Hardware image effectively observes the finger plane
    # perpendicular to the bottle axis.
    forward_plane = (
        forward_raw
        -
        np.dot(
            forward_raw,
            bottle_axis,
        )
        * bottle_axis
    )


    forward = unit(
        forward_plane
    )


    side = unit(
        np.cross(
            bottle_axis,
            forward,
        )
    )


    bottle_vector = (
        bottle_center
        - mcp
    )


    # Make SIDE positive toward the bottle.
    if np.dot(
        bottle_vector,
        side,
    ) < 0.0:

        side *= -1.0


    # --------------------------------------------------------
    # Decompose bottle center relative to MCP
    # --------------------------------------------------------

    sim_along_m = float(
        np.dot(
            bottle_vector,
            forward,
        )
    )


    sim_side_m = float(
        np.dot(
            bottle_vector,
            side,
        )
    )


    sim_axis_m = float(
        np.dot(
            bottle_vector,
            bottle_axis,
        )
    )


    sim_plane_distance_m = math.sqrt(
        sim_along_m**2
        +
        sim_side_m**2
    )


    sim_along_mm = (
        1000.0
        * sim_along_m
    )

    sim_side_mm = (
        1000.0
        * sim_side_m
    )

    sim_axis_mm = (
        1000.0
        * sim_axis_m
    )

    sim_plane_distance_mm = (
        1000.0
        * sim_plane_distance_m
    )


    sim_surface_distance_mm = (
        sim_plane_distance_mm
        - BOTTLE_RADIUS_MM
    )


    # --------------------------------------------------------
    # Required in-plane correction
    # --------------------------------------------------------

    delta_along_mm = (
        HW_ALONG_MM
        - sim_along_mm
    )


    delta_side_mm = (
        HW_SIDE_MM
        - sim_side_mm
    )


    correction_m = (
        forward
        * (
            delta_along_mm
            / 1000.0
        )
        +
        side
        * (
            delta_side_mm
            / 1000.0
        )
    )


    suggested_center = (
        bottle_center
        + correction_m
    )


    correction_norm_mm = float(
        np.linalg.norm(
            correction_m
        )
        * 1000.0
    )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "HARDWARE vs CURRENT MUJOCO BOTTLE GEOMETRY"
    )
    print("=" * 100)


    print()
    print("OPEN SIMULATION REFERENCE")
    print("-" * 100)

    print(
        "MCP world = "
        f"{mcp[0]:.9f} "
        f"{mcp[1]:.9f} "
        f"{mcp[2]:.9f}"
    )

    print(
        "PIP world = "
        f"{pip[0]:.9f} "
        f"{pip[1]:.9f} "
        f"{pip[2]:.9f}"
    )

    print(
        "open forward axis = "
        f"{forward[0]:.9f} "
        f"{forward[1]:.9f} "
        f"{forward[2]:.9f}"
    )

    print(
        "toward-bottle axis = "
        f"{side[0]:.9f} "
        f"{side[1]:.9f} "
        f"{side[2]:.9f}"
    )

    print(
        "bottle axis = "
        f"{bottle_axis[0]:.9f} "
        f"{bottle_axis[1]:.9f} "
        f"{bottle_axis[2]:.9f}"
    )


    print()
    print("=" * 100)
    print("HARDWARE TARGET — OPEN FINGER FRAME")
    print("=" * 100)

    print(
        f"along finger           = "
        f"{HW_ALONG_MM:+.3f} mm"
    )

    print(
        f"toward bottle          = "
        f"{HW_SIDE_MM:+.3f} mm"
    )

    print(
        f"MCP -> center          = "
        f"{HW_CENTER_DISTANCE_MM:.3f} mm"
    )

    print(
        f"MCP -> bottle surface  = "
        f"{HW_SURFACE_DISTANCE_MM:.3f} mm"
    )


    print()
    print("=" * 100)
    print("CURRENT MUJOCO — OPEN FINGER FRAME")
    print("=" * 100)

    print(
        f"along finger           = "
        f"{sim_along_mm:+.3f} mm"
    )

    print(
        f"toward bottle          = "
        f"{sim_side_mm:+.3f} mm"
    )

    print(
        f"MCP -> center in plane = "
        f"{sim_plane_distance_mm:.3f} mm"
    )

    print(
        f"MCP -> bottle surface  = "
        f"{sim_surface_distance_mm:.3f} mm"
    )

    print(
        f"offset along bottle axis = "
        f"{sim_axis_mm:+.3f} mm"
    )


    print()
    print("=" * 100)
    print("GEOMETRY ERROR: HARDWARE - MUJOCO")
    print("=" * 100)

    print(
        f"required along shift   = "
        f"{delta_along_mm:+.3f} mm"
    )

    print(
        f"required side shift    = "
        f"{delta_side_mm:+.3f} mm"
    )

    print(
        f"total in-plane shift   = "
        f"{correction_norm_mm:.3f} mm"
    )


    print()
    print(
        "Current cylinder center:"
    )

    print(
        "  "
        f"{bottle_center[0]:.9f} "
        f"{bottle_center[1]:.9f} "
        f"{bottle_center[2]:.9f}"
    )


    print()
    print(
        "Hardware-derived provisional cylinder center:"
    )

    print(
        "  "
        f"{suggested_center[0]:.9f} "
        f"{suggested_center[1]:.9f} "
        f"{suggested_center[2]:.9f}"
    )


    print()
    print(
        "IMPORTANT: do NOT modify the XML from this value yet."
    )

    print(
        "We will first inspect the magnitude and direction "
        "of this discrepancy."
    )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
