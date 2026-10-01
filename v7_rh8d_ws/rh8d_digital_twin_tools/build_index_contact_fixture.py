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
    "ur5e_rh8d_index_contact_v1.xml"
)


OPEN_TICKS = 100.0

CONTACT_TICKS = 1020.0

METERS_PER_TICK = 6.0e-6


# Match the hardware contact-search experiment.
APPROACH_RATE_TICKS_S = 200.0


TIP_GEOM_NAME = "collision_idx_tip"

TIP_SITE_NAME = "index_tip_force_site"

SPOOL_JOINT = "index_motor_spool"

ACTUATOR = "pos_index_tendon"


# Small finite patch rather than an infinite plane.
#
# 24 x 24 mm contact face
# 10 mm total thickness.
PATCH_HALF_X = 0.012
PATCH_HALF_Y = 0.012
PATCH_HALF_THICKNESS = 0.005


def obj_id(model, objtype, name):

    idx = mujoco.mj_name2id(
        model,
        objtype,
        name,
    )

    if idx < 0:

        raise RuntimeError(
            f"Could not find: {name}"
        )

    return idx


def set_spool_from_ticks(
    model,
    data,
    actuator_id,
    spool_qadr,
    spool_dof,
    ticks,
):

    spool = max(
        0.0,
        (
            ticks - OPEN_TICKS
        ) * METERS_PER_TICK,
    )

    data.qpos[spool_qadr] = spool

    data.qvel[spool_dof] = 0.0

    data.ctrl[actuator_id] = spool


def simulate_free_approach(
    model,
):

    data = mujoco.MjData(
        model
    )

    model.opt.gravity[:] = 0.0


    spool_jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_JOINT,
    )

    spool_qadr = model.jnt_qposadr[
        spool_jid
    ]

    spool_dof = model.jnt_dofadr[
        spool_jid
    ]


    actuator_id = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR,
    )


    tip_geom_id = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        TIP_GEOM_NAME,
    )


    tip_site_id = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        TIP_SITE_NAME,
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    # --------------------------------------------------------
    # Settle open
    # --------------------------------------------------------

    for _ in range(
        int(
            2.0
            / model.opt.timestep
        )
    ):

        set_spool_from_ticks(
            model,
            data,
            actuator_id,
            spool_qadr,
            spool_dof,
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

    current_ticks = OPEN_TICKS


    # Continue slightly beyond the final requested sample.
    #
    # current_ticks advances in floating-point increments, so
    # stopping exactly at 1060 can occasionally miss the
    # requested 1060-tick sample because of round-off.
    while (
        current_ticks
        <= 1070.0
    ):

        set_spool_from_ticks(
            model,
            data,
            actuator_id,
            spool_qadr,
            spool_dof,
            current_ticks,
        )

        mujoco.mj_step(
            model,
            data,
        )


        for target in wanted:

            if (
                target not in samples
                and current_ticks >= target
            ):

                xmat = (
                    data.geom_xmat[
                        tip_geom_id
                    ]
                    .reshape(3, 3)
                    .copy()
                )

                samples[
                    target
                ] = {

                    "geom_center":
                        data.geom_xpos[
                            tip_geom_id
                        ].copy(),

                    "geom_xmat":
                        xmat,

                    "site":
                        data.site_xpos[
                            tip_site_id
                        ].copy(),
                }


        current_ticks += (
            APPROACH_RATE_TICKS_S
            * model.opt.timestep
        )


    return (
        samples,
        tip_geom_id,
    )


def main():

    print()
    print("=" * 78)
    print(
        "RH8D INDEX CONTACT FIXTURE BUILDER"
    )
    print("=" * 78)

    print()
    print("Source:")
    print(SOURCE_XML)


    base_model = (
        mujoco.MjModel.from_xml_path(
            str(SOURCE_XML)
        )
    )


    samples, tip_geom_id = (
        simulate_free_approach(
            base_model
        )
    )


    p_before = (
        samples[
            980.0
        ]["geom_center"]
    )

    p_contact = (
        samples[
            CONTACT_TICKS
        ]["geom_center"]
    )

    p_after = (
        samples[
            1060.0
        ]["geom_center"]
    )


    # --------------------------------------------------------
    # Local fingertip direction of travel
    # --------------------------------------------------------

    motion = (
        p_after
        - p_before
    )

    motion = (
        motion
        / np.linalg.norm(
            motion
        )
    )


    # Rigid obstacle outward normal points back toward finger.
    normal = -motion


    # --------------------------------------------------------
    # Actual capsule support distance in direction of travel
    # --------------------------------------------------------

    capsule_radius = (
        base_model.geom_size[
            tip_geom_id,
            0,
        ]
    )

    capsule_half_length = (
        base_model.geom_size[
            tip_geom_id,
            1,
        ]
    )


    geom_rotation = (
        samples[
            CONTACT_TICKS
        ]["geom_xmat"]
    )


    # Capsule axis is local Z.
    capsule_axis = (
        geom_rotation[
            :,
            2,
        ]
    )


    support_distance = (
        capsule_radius
        +
        capsule_half_length
        * abs(
            np.dot(
                motion,
                capsule_axis,
            )
        )
    )


    # Surface point of fingertip that reaches obstacle.
    surface_point = (
        p_contact
        +
        motion
        * support_distance
    )


    # Box +Z face is the contact surface.
    #
    # Its centre must therefore be behind the surface by
    # one half-thickness in the -normal direction.
    fixture_center = (
        surface_point
        -
        normal
        * PATCH_HALF_THICKNESS
    )


    print()
    print("Fingertip trajectory:")

    for ticks in [
        980.0,
        1020.0,
        1060.0,
    ]:

        p = samples[
            ticks
        ]["geom_center"]

        print(
            f"  {ticks:6.0f} ticks : "
            f"x={p[0]: .6f} "
            f"y={p[1]: .6f} "
            f"z={p[2]: .6f}"
        )


    print()
    print(
        "Direction of fingertip motion:"
    )

    print(
        "  "
        + " ".join(
            f"{x:.8f}"
            for x in motion
        )
    )


    print()
    print(
        "Fixture outward normal:"
    )

    print(
        "  "
        + " ".join(
            f"{x:.8f}"
            for x in normal
        )
    )


    print()
    print(
        f"Capsule radius       = "
        f"{capsule_radius*1000:.3f} mm"
    )

    print(
        f"Capsule half-length  = "
        f"{capsule_half_length*1000:.3f} mm"
    )

    print(
        f"Support distance     = "
        f"{support_distance*1000:.3f} mm"
    )


    print()
    print(
        "Calculated fingertip contact point:"
    )

    print(
        "  "
        + " ".join(
            f"{x:.8f}"
            for x in surface_point
        )
    )


    print()
    print(
        "Fixture centre:"
    )

    print(
        "  "
        + " ".join(
            f"{x:.8f}"
            for x in fixture_center
        )
    )


    # --------------------------------------------------------
    # Insert finite contact fixture into copied XML
    # --------------------------------------------------------

    xml = SOURCE_XML.read_text()


    fixture_xml = f'''

    <!--
      INDEX CONTACT SYSTEM IDENTIFICATION FIXTURE

      Automatically placed tangent to collision_idx_tip at
      the experimentally measured first-contact motor state
      of approximately {CONTACT_TICKS:.0f} encoder ticks.

      This is a LOCAL tangent approximation of the bottle wall.
    -->

    <geom name="index_contact_fixture"
          type="box"

          pos="{fixture_center[0]:.9f}
               {fixture_center[1]:.9f}
               {fixture_center[2]:.9f}"

          zaxis="{normal[0]:.9f}
                 {normal[1]:.9f}
                 {normal[2]:.9f}"

          size="{PATCH_HALF_X:.6f}
                {PATCH_HALF_Y:.6f}
                {PATCH_HALF_THICKNESS:.6f}"

          rgba="0.15 0.7 0.2 0.65"

          contype="1"
          conaffinity="1"

          priority="10"

          condim="3"

          friction="0.8 0.005 0.0001"

          solref="-2000 -20"

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


    insert_at = match.end()


    xml = (
        xml[:insert_at]
        + fixture_xml
        + xml[insert_at:]
    )


    OUTPUT_XML.write_text(
        xml
    )


    print()
    print(
        "Created:"
    )

    print(
        OUTPUT_XML
    )


    # --------------------------------------------------------
    # Compile generated model
    # --------------------------------------------------------

    contact_model = (
        mujoco.MjModel.from_xml_path(
            str(OUTPUT_XML)
        )
    )


    print()
    print(
        "Generated contact model "
        "compiled successfully."
    )


    # --------------------------------------------------------
    # Contact-onset sanity simulation
    # --------------------------------------------------------

    data = mujoco.MjData(
        contact_model
    )

    contact_model.opt.gravity[:] = 0.0


    spool_jid = obj_id(
        contact_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_JOINT,
    )

    spool_qadr = (
        contact_model.jnt_qposadr[
            spool_jid
        ]
    )

    spool_dof = (
        contact_model.jnt_dofadr[
            spool_jid
        ]
    )

    actuator_id = obj_id(
        contact_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR,
    )

    fixture_gid = obj_id(
        contact_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "index_contact_fixture",
    )


    mujoco.mj_resetData(
        contact_model,
        data,
    )


    for _ in range(
        int(
            2.0
            / contact_model.opt.timestep
        )
    ):

        set_spool_from_ticks(
            contact_model,
            data,
            actuator_id,
            spool_qadr,
            spool_dof,
            OPEN_TICKS,
        )

        mujoco.mj_step(
            contact_model,
            data,
        )


    first_contact_ticks = None

    first_contact_pair = None

    current_ticks = OPEN_TICKS


    while (
        current_ticks
        <= 1150.0
    ):

        set_spool_from_ticks(
            contact_model,
            data,
            actuator_id,
            spool_qadr,
            spool_dof,
            current_ticks,
        )

        mujoco.mj_step(
            contact_model,
            data,
        )


        for i in range(
            data.ncon
        ):

            contact = data.contact[i]

            if (
                contact.geom1
                == fixture_gid
                or
                contact.geom2
                == fixture_gid
            ):

                other_gid = (
                    contact.geom2
                    if contact.geom1
                    == fixture_gid
                    else contact.geom1
                )


                other_name = (
                    mujoco.mj_id2name(
                        contact_model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        other_gid,
                    )
                )


                first_contact_ticks = (
                    current_ticks
                )

                first_contact_pair = (
                    other_name
                )

                break


        if (
            first_contact_ticks
            is not None
        ):

            break


        current_ticks += (
            APPROACH_RATE_TICKS_S
            * contact_model.opt.timestep
        )


    print()
    print("=" * 78)
    print(
        "CONTACT-ONSET SANITY CHECK"
    )
    print("=" * 78)


    if (
        first_contact_ticks
        is None
    ):

        print(
            "NO CONTACT detected before "
            "1150 ticks."
        )

    else:

        print(
            f"First simulated contact = "
            f"{first_contact_ticks:.1f} ticks"
        )

        print(
            f"Contacting geom         = "
            f"{first_contact_pair}"
        )

        print(
            f"Hardware first contact  ≈ "
            f"{CONTACT_TICKS:.1f} ticks"
        )

        print(
            f"Contact onset error      = "
            f"{first_contact_ticks-CONTACT_TICKS:+.1f} ticks"
        )


    print()
    print("=" * 78)
    print("FIXTURE BUILD COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
