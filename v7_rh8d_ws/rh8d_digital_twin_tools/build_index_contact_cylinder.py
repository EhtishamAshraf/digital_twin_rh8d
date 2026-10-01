#!/usr/bin/env python3

from pathlib import Path
import math
import re

import mujoco
import numpy as np


WS = Path.home() / (
    "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
)

SOURCE_XML = (
    WS /
    "src/ur5e_mujoco_description/mjcf/"
    "ur5e_rh8d_index_identified.xml"
)

OUTPUT_XML = (
    WS /
    "src/ur5e_mujoco_description/mjcf/"
    "ur5e_rh8d_index_contact_cylinder_v1.xml"
)


OPEN_TICKS = 100.0
CONTACT_TICKS = 1020.0

METERS_PER_TICK = 6.0e-6

APPROACH_RATE = 200.0


# Measured physical bottle diameter = 69.2 mm
BOTTLE_RADIUS = 0.0692 / 2.0


# Long static cylinder so endcaps cannot affect contact.
BOTTLE_HALF_LENGTH = 0.100


TIP_GEOM = "collision_idx_tip"
TIP_SITE = "index_tip_force_site"

MIDDLE_BODY = "Index_Middle"

SPOOL = "index_motor_spool"
ACTUATOR = "pos_index_tendon"


def oid(model, typ, name):

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


def set_ticks(
    model,
    data,
    spool_q,
    spool_dof,
    actuator,
    ticks,
):

    q = max(
        0.0,
        (
            ticks - OPEN_TICKS
        ) * METERS_PER_TICK,
    )

    data.qpos[spool_q] = q
    data.qvel[spool_dof] = 0.0
    data.ctrl[actuator] = q


def get_contact_geometry():

    model = mujoco.MjModel.from_xml_path(
        str(SOURCE_XML)
    )

    data = mujoco.MjData(model)

    model.opt.gravity[:] = 0.0


    spool_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL,
    )

    spool_q = model.jnt_qposadr[
        spool_jid
    ]

    spool_dof = model.jnt_dofadr[
        spool_jid
    ]


    actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR,
    )


    tip_gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        TIP_GEOM,
    )

    tip_sid = oid(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        TIP_SITE,
    )

    middle_bid = oid(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        MIDDLE_BODY,
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    for _ in range(
        int(
            1.5
            / model.opt.timestep
        )
    ):

        set_ticks(
            model,
            data,
            spool_q,
            spool_dof,
            actuator,
            OPEN_TICKS,
        )

        mujoco.mj_step(
            model,
            data,
        )


    wanted = [
        980.0,
        CONTACT_TICKS,
        1060.0,
    ]

    samples = {}

    ticks = OPEN_TICKS


    while ticks <= 1070.0:

        set_ticks(
            model,
            data,
            spool_q,
            spool_dof,
            actuator,
            ticks,
        )

        mujoco.mj_step(
            model,
            data,
        )


        for target in wanted:

            if (
                target not in samples
                and ticks >= target
            ):

                samples[target] = {

                    "geom_center":
                        data.geom_xpos[
                            tip_gid
                        ].copy(),

                    "geom_R":
                        data.geom_xmat[
                            tip_gid
                        ].reshape(
                            3, 3
                        ).copy(),

                    "tip_site":
                        data.site_xpos[
                            tip_sid
                        ].copy(),

                    "middle":
                        data.xpos[
                            middle_bid
                        ].copy(),
                }


        ticks += (
            APPROACH_RATE
            * model.opt.timestep
        )


    p_before = (
        samples[980.0][
            "geom_center"
        ]
    )

    p_contact = (
        samples[
            CONTACT_TICKS
        ]["geom_center"]
    )

    p_after = (
        samples[1060.0][
            "geom_center"
        ]
    )


    # --------------------------------------------------------
    # Direction fingertip moves toward obstacle.
    # --------------------------------------------------------

    motion = (
        p_after - p_before
    )

    motion /= np.linalg.norm(
        motion
    )


    # Obstacle radial outward normal, pointing toward finger.
    radial_normal = -motion


    # --------------------------------------------------------
    # Distal-finger direction.
    # --------------------------------------------------------

    distal = (
        samples[
            CONTACT_TICKS
        ]["tip_site"]
        -
        samples[
            CONTACT_TICKS
        ]["middle"]
    )

    distal /= np.linalg.norm(
        distal
    )


    # --------------------------------------------------------
    # Bottle axis.
    #
    # Physical experiment is approximately planar:
    #
    # distal direction + radial normal lie in flexion plane,
    # bottle axis is perpendicular to that plane.
    # --------------------------------------------------------

    cylinder_axis = np.cross(
        radial_normal,
        distal,
    )

    cylinder_axis /= np.linalg.norm(
        cylinder_axis
    )


    # --------------------------------------------------------
    # Find surface of fingertip capsule in direction toward
    # the bottle.
    # --------------------------------------------------------

    capsule_radius = (
        model.geom_size[
            tip_gid,
            0
        ]
    )

    capsule_half_length = (
        model.geom_size[
            tip_gid,
            1
        ]
    )


    R_tip = (
        samples[
            CONTACT_TICKS
        ]["geom_R"]
    )


    # MuJoCo capsule local axis = Z.
    capsule_axis = R_tip[
        :,
        2
    ]


    toward_bottle = (
        -radial_normal
    )


    support_distance = (
        capsule_radius
        +
        capsule_half_length
        * abs(
            np.dot(
                toward_bottle,
                capsule_axis,
            )
        )
    )


    fingertip_surface = (
        p_contact
        +
        toward_bottle
        * support_distance
    )


    # Cylinder surface satisfies:
    #
    # surface =
    #     cylinder_center
    #     + radius * radial_normal
    #
    # therefore:
    cylinder_center = (
        fingertip_surface
        -
        BOTTLE_RADIUS
        * radial_normal
    )


    return {
        "normal":
            radial_normal,

        "axis":
            cylinder_axis,

        "distal":
            distal,

        "surface":
            fingertip_surface,

        "center":
            cylinder_center,

        "support":
            support_distance,
    }


def line_angle(
    a,
    b,
):

    a = (
        a
        / np.linalg.norm(a)
    )

    b = (
        b
        / np.linalg.norm(b)
    )

    value = np.clip(
        np.dot(a, b),
        -1.0,
        1.0,
    )

    angle = math.degrees(
        math.acos(value)
    )

    return min(
        angle,
        180.0 - angle,
    )


def main():

    print()
    print("=" * 82)
    print(
        "RH8D INDEX CONTACT CYLINDER BUILDER"
    )
    print("=" * 82)

    geometry = (
        get_contact_geometry()
    )


    n = geometry["normal"]
    axis = geometry["axis"]
    distal = geometry["distal"]

    surface = geometry["surface"]
    center = geometry["center"]


    print()
    print(
        f"Measured bottle diameter = "
        f"{2*BOTTLE_RADIUS*1000:.2f} mm"
    )

    print(
        f"Bottle radius            = "
        f"{BOTTLE_RADIUS*1000:.2f} mm"
    )


    print()
    print(
        "Cylinder radial normal:"
    )

    print(
        " ",
        *[
            f"{x:.8f}"
            for x in n
        ]
    )


    print()
    print(
        "Cylinder axis:"
    )

    print(
        " ",
        *[
            f"{x:.8f}"
            for x in axis
        ]
    )


    print()
    print(
        "Distal finger direction:"
    )

    print(
        " ",
        *[
            f"{x:.8f}"
            for x in distal
        ]
    )


    print()
    print(
        "Radial-normal / distal angle = "
        f"{line_angle(n, distal):.2f} deg"
    )


    print()
    print(
        "Fingertip contact surface:"
    )

    print(
        " ",
        *[
            f"{x:.8f}"
            for x in surface
        ]
    )


    print()
    print(
        "Bottle cylinder centre:"
    )

    print(
        " ",
        *[
            f"{x:.8f}"
            for x in center
        ]
    )


    # --------------------------------------------------------
    # Add static identification cylinder.
    #
    # Keep geom name index_contact_fixture so all existing
    # contact fitting scripts can reuse it.
    # --------------------------------------------------------

    xml = SOURCE_XML.read_text()


    cylinder_xml = f'''

    <!--
      RH8D INDEX CONTACT SYSTEM-ID BOTTLE

      Real measured bottle:
          diameter = 69.2 mm

      Static on purpose:
      the water-filled bottle was effectively stationary
      during the clean low-force calibration interval.
    -->

    <geom name="index_contact_fixture"
          type="cylinder"

          pos="{center[0]:.9f}
               {center[1]:.9f}
               {center[2]:.9f}"

          zaxis="{axis[0]:.9f}
                 {axis[1]:.9f}
                 {axis[2]:.9f}"

          size="{BOTTLE_RADIUS:.6f}
                {BOTTLE_HALF_LENGTH:.6f}"

          rgba="0.1 0.65 0.25 0.55"

          contype="1"
          conaffinity="1"

          priority="10"

          condim="3"

          friction="0.8 0.005 0.0001"

          solref="-32000 -320"

          solimp="0.95 0.95 0.001 0.5 2"

          margin="0"/>
'''


    match = re.search(
        r"<worldbody[^>]*>",
        xml,
    )

    if match is None:
        raise RuntimeError(
            "Could not find <worldbody>."
        )


    at = match.end()


    xml = (
        xml[:at]
        + cylinder_xml
        + xml[at:]
    )


    OUTPUT_XML.write_text(
        xml
    )


    print()
    print("Created:")
    print(OUTPUT_XML)


    # --------------------------------------------------------
    # Compile + sanity-check first contact.
    # --------------------------------------------------------

    model = (
        mujoco.MjModel.from_xml_path(
            str(OUTPUT_XML)
        )
    )

    data = mujoco.MjData(
        model
    )

    model.opt.gravity[:] = 0.0


    spool_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL,
    )

    spool_q = model.jnt_qposadr[
        spool_jid
    ]

    spool_dof = model.jnt_dofadr[
        spool_jid
    ]

    actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR,
    )

    fixture_gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "index_contact_fixture",
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    for _ in range(
        int(
            1.5
            / model.opt.timestep
        )
    ):

        set_ticks(
            model,
            data,
            spool_q,
            spool_dof,
            actuator,
            OPEN_TICKS,
        )

        mujoco.mj_step(
            model,
            data,
        )


    first_ticks = None
    other_name = None

    ticks = OPEN_TICKS


    while ticks <= 1150.0:

        set_ticks(
            model,
            data,
            spool_q,
            spool_dof,
            actuator,
            ticks,
        )

        mujoco.mj_step(
            model,
            data,
        )


        for i in range(
            data.ncon
        ):

            c = data.contact[i]


            if (
                c.geom1 == fixture_gid
                or
                c.geom2 == fixture_gid
            ):

                other = (
                    c.geom2
                    if c.geom1
                    == fixture_gid
                    else c.geom1
                )


                other_name = (
                    mujoco.mj_id2name(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        other,
                    )
                )


                first_ticks = ticks

                break


        if first_ticks is not None:
            break


        ticks += (
            APPROACH_RATE
            * model.opt.timestep
        )


    print()
    print("=" * 82)
    print(
        "CYLINDER CONTACT SANITY CHECK"
    )
    print("=" * 82)


    if first_ticks is None:

        print(
            "No fixture contact before "
            "1150 ticks."
        )

    else:

        print(
            f"First simulated contact = "
            f"{first_ticks:.1f} ticks"
        )

        print(
            f"Contacting geom         = "
            f"{other_name}"
        )

        print(
            f"Hardware 0.10-N event   ≈ "
            f"{CONTACT_TICKS:.1f} ticks"
        )

        print(
            f"Difference              = "
            f"{first_ticks-CONTACT_TICKS:+.1f} ticks"
        )


    print()
    print("=" * 82)


if __name__ == "__main__":
    main()
