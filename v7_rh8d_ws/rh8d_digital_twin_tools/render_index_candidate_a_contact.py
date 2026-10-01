#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

import fit_index_contact_v1 as base


# ============================================================
# CANDIDATE A
# ============================================================

MODEL_PATH = (
    base.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_v1.xml"
)

SCALE = 5.1e-6
TENDON_STIFFNESS = 2100.0

CONTACT_STIFFNESS = 28000.0
CONTACT_DAMPING = 1.0
CONTACT_FRICTION = 1.2

OPEN_ENCODER = 100.0


# Hardware-video alignment
VIDEO_TIME = 50.50
VIDEO_BAG_OFFSET = 11.53

TARGET_BAG_TIME = (
    VIDEO_TIME
    - VIDEO_BAG_OFFSET
)


TENDON_NAME = "idx_flex_spatial"

SPOOL_NAME = "index_motor_spool"
ACTUATOR_NAME = "pos_index_tendon"

FIXTURE_NAME = "index_contact_fixture"


OUTPUT_DIR = (
    base.WS
    / "rh8d_digital_twin_tools/results/"
      "index_sim_contact_visual"
)


WIDTH = 640
HEIGHT = 480


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


def set_spool(
    data,
    qadr,
    dofadr,
    actuator,
    encoder,
):

    spool = max(
        0.0,
        (
            encoder
            - OPEN_ENCODER
        ) * SCALE,
    )

    data.qpos[qadr] = spool
    data.qvel[dofadr] = 0.0
    data.ctrl[actuator] = spool


def get_geom_name(
    model,
    gid,
):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        gid,
    )

    if name is None:
        return f"geom_{gid}"

    return name


def print_fixture_contacts(
    model,
    data,
    fixture_gid,
):

    print()
    print("=" * 100)
    print("ACTIVE CONTACTS WITH BOTTLE")
    print("=" * 100)

    found = False


    for i in range(
        data.ncon
    ):

        c = data.contact[i]

        if (
            c.geom1 != fixture_gid
            and
            c.geom2 != fixture_gid
        ):
            continue


        found = True


        if c.geom1 == fixture_gid:

            other_gid = c.geom2

        else:

            other_gid = c.geom1


        wrench = np.zeros(6)

        mujoco.mj_contactForce(
            model,
            data,
            i,
            wrench,
        )


        force_norm = float(
            np.linalg.norm(
                wrench[:3]
            )
        )


        print()
        print(
            f"contact #{i}"
        )

        print(
            f"  other geom = "
            f"{get_geom_name(model, other_gid)}"
        )

        print(
            f"  distance   = "
            f"{c.dist*1000:.4f} mm"
        )

        print(
            "  position   = "
            f"{c.pos[0]:.6f} "
            f"{c.pos[1]:.6f} "
            f"{c.pos[2]:.6f}"
        )

        print(
            f"  |force|    = "
            f"{force_norm:.6f} N"
        )


    if not found:

        print()
        print(
            "NO ACTIVE CONTACTS WITH CYLINDER"
        )


    print()
    print("=" * 100)


def finger_lookat(
    model,
    data,
    fixture_gid,
):

    points = []


    for body_name in [
        "Index_Proximal",
        "Index_Middle",
        "Index_Distal",
        "Index_TipDummy",
    ]:

        bid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        if bid >= 0:

            points.append(
                data.xpos[
                    bid
                ].copy()
            )


    points.append(
        data.geom_xpos[
            fixture_gid
        ].copy()
    )


    return np.mean(
        np.asarray(points),
        axis=0,
    )


def cylinder_axis(
    data,
    fixture_gid,
):

    R = (
        data.geom_xmat[
            fixture_gid
        ]
        .reshape(3, 3)
    )

    axis = R[:, 2].copy()

    axis /= np.linalg.norm(
        axis
    )

    return axis


def camera_from_axis(
    lookat,
    axis,
    side=1.0,
):

    # Camera positioned along cylinder axis.
    #
    # This approximates the hardware top-down view,
    # because the real bottle axis is approximately
    # perpendicular to the finger flexion plane.

    vec = (
        side
        * axis
    )


    azimuth = math.degrees(
        math.atan2(
            vec[1],
            vec[0],
        )
    )


    elevation = math.degrees(
        math.atan2(
            vec[2],
            math.sqrt(
                vec[0]**2
                +
                vec[1]**2
            ),
        )
    )


    cam = mujoco.MjvCamera()

    mujoco.mjv_defaultCamera(
        cam
    )


    cam.type = (
        mujoco.mjtCamera.mjCAMERA_FREE
    )


    cam.lookat[:] = lookat

    cam.distance = 0.18

    cam.azimuth = azimuth
    cam.elevation = elevation


    return cam


def save_render(
    renderer,
    data,
    camera,
    path,
    title,
    show_contacts=False,
):

    option = mujoco.MjvOption()

    mujoco.mjv_defaultOption(
        option
    )


    if show_contacts:

        option.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
        ] = True

        option.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
        ] = True


    renderer.update_scene(
        data,
        camera=camera,
        scene_option=option,
    )


    rgb = renderer.render()


    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )


    cv2.rectangle(
        bgr,
        (0, 0),
        (WIDTH, 55),
        (0, 0, 0),
        -1,
    )


    cv2.putText(
        bgr,
        title,
        (20, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


    cv2.imwrite(
        str(path),
        bgr,
    )


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            " python3 "
            "render_index_candidate_a_contact.py "
            "/path/to/contact_rosbag"
        )

        sys.exit(1)


    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    (
        states,
        force_data,
        encoder_data,
    ) = base.read_bag(
        bag
    )


    transitions = (
        base.compress_states(
            states
        )
    )


    (
        approach_start,
        regulation_start,
        hold_start,
        hold_end,
    ) = base.find_intervals(
        transitions
    )


    encoder_time = (
        encoder_data[:, 0]
    )


    encoder_position = (
        base.median_filter_5(
            encoder_data[:, 1]
        )
    )


    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(
        model
    )


    model.opt.gravity[:] = 0.0


    tendon_id = oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )


    model.tendon_stiffness[
        tendon_id
    ] = TENDON_STIFFNESS


    fixture_gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FIXTURE_NAME,
    )


    model.geom_friction[
        fixture_gid,
        0
    ] = CONTACT_FRICTION


    model.geom_solref[
        fixture_gid,
        0
    ] = -CONTACT_STIFFNESS

    model.geom_solref[
        fixture_gid,
        1
    ] = -CONTACT_DAMPING


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


    actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAME,
    )


    renderer = mujoco.Renderer(
        model,
        height=HEIGHT,
        width=WIDTH,
    )


    # ========================================================
    # OPEN STATE
    # ========================================================

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

        set_spool(
            data,
            spool_qadr,
            spool_dofadr,
            actuator,
            OPEN_ENCODER,
        )

        mujoco.mj_step(
            model,
            data,
        )


    open_state = (
        data.qpos.copy(),
        data.qvel.copy(),
        data.ctrl.copy(),
    )


    lookat = finger_lookat(
        model,
        data,
        fixture_gid,
    )


    axis = cylinder_axis(
        data,
        fixture_gid,
    )


    # Save from both sides of cylinder axis.
    for side_name, side in [
        ("plus", +1.0),
        ("minus", -1.0),
    ]:

        camera = camera_from_axis(
            lookat,
            axis,
            side,
        )


        save_render(
            renderer,
            data,
            camera,
            OUTPUT_DIR
            / f"index_sim_open_{side_name}.png",
            (
                "SIM OPEN — Candidate A — "
                "encoder 100"
            ),
        )


    # ========================================================
    # REPLAY TO VIDEO-MATCHED HOLD
    # ========================================================

    data.qpos[:] = open_state[0]
    data.qvel[:] = open_state[1]
    data.ctrl[:] = open_state[2]

    mujoco.mj_forward(
        model,
        data,
    )


    elapsed = 0.0

    dt = model.opt.timestep

    target_elapsed = (
        TARGET_BAG_TIME
        - approach_start
    )


    target_encoder = OPEN_ENCODER


    while elapsed <= target_elapsed:

        bag_time = (
            approach_start
            + elapsed
        )


        target_encoder = float(
            np.interp(
                bag_time,
                encoder_time,
                encoder_position,
            )
        )


        set_spool(
            data,
            spool_qadr,
            spool_dofadr,
            actuator,
            target_encoder,
        )


        mujoco.mj_step(
            model,
            data,
        )


        elapsed += dt


    print()
    print("=" * 100)
    print("VIDEO-MATCHED SIMULATION STATE")
    print("=" * 100)

    print(
        f"video time       = "
        f"{VIDEO_TIME:.3f} s"
    )

    print(
        f"bag time         = "
        f"{TARGET_BAG_TIME:.3f} s"
    )

    print(
        f"encoder          = "
        f"{target_encoder:.1f} ticks"
    )

    print(
        f"tendon stiffness = "
        f"{TENDON_STIFFNESS:.1f} N/m"
    )

    print(
        f"scale            = "
        f"{SCALE*1e6:.3f} um/tick"
    )


    print_fixture_contacts(
        model,
        data,
        fixture_gid,
    )


    lookat = finger_lookat(
        model,
        data,
        fixture_gid,
    )


    axis = cylinder_axis(
        data,
        fixture_gid,
    )


    for side_name, side in [
        ("plus", +1.0),
        ("minus", -1.0),
    ]:

        camera = camera_from_axis(
            lookat,
            axis,
            side,
        )


        save_render(
            renderer,
            data,
            camera,
            OUTPUT_DIR
            / f"index_sim_contact_{side_name}.png",
            (
                "SIM CONTACT HOLD — Candidate A — "
                f"encoder {target_encoder:.0f}"
            ),
        )


        save_render(
            renderer,
            data,
            camera,
            OUTPUT_DIR
            / f"index_sim_contact_points_{side_name}.png",
            (
                "SIM CONTACT POINTS — Candidate A — "
                f"encoder {target_encoder:.0f}"
            ),
            show_contacts=True,
        )


    renderer.close()


    print()
    print("=" * 100)
    print("SAVED IMAGES")
    print("=" * 100)

    for name in [
        "index_sim_open_plus.png",
        "index_sim_open_minus.png",
        "index_sim_contact_plus.png",
        "index_sim_contact_minus.png",
        "index_sim_contact_points_plus.png",
        "index_sim_contact_points_minus.png",
    ]:

        print(
            OUTPUT_DIR
            / name
        )


    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
