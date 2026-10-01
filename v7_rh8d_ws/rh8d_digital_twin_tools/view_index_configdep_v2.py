#!/usr/bin/env python3

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()

MODEL_PATH = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_configdep_geom_v2.xml"
)

OUT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "index_configdep_v2/visualization"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

TENDON = "idx_flex_spatial"


# ============================================================
# TABLETOP HARDWARE-CONTACT ARM CONFIGURATION
#
# This is the same UR5e startup pose used by the tabletop
# contact-system-ID setup. It must be restored after qpos0
# because the launch/set_pose layer applied it externally.
# ============================================================

ARM_STARTUP = {
    "shoulder_pan_joint":   1.4927,
    "shoulder_lift_joint": -0.8678,
    "elbow_joint":         -2.4300,
    "wrist_1_joint":       -0.6596,
    "wrist_2_joint":       -0.0347,
    "wrist_3_joint":        0.1260,
}


STATES = [
    (
        "OPEN",
        np.array([0.0, 0.0, 0.0]),
    ),
    (
        "MID",
        np.array([30.0, 15.0, 5.0]),
    ),
    (
        "HW_CONTACT",
        np.array([14.5355, 23.8110, 9.3270]),
    ),
    (
        "FLEXED",
        np.array([50.0, 45.0, 15.0]),
    ),
]


def joint_address(model, name):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    return int(
        model.jnt_qposadr[jid]
    )


def tendon_id(model, name):

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        name,
    )

    if tid < 0:
        raise RuntimeError(
            f"Tendon not found: {name}"
        )

    return tid


def set_state(
    model,
    data,
    qpos0,
    addresses,
    q_deg,
):

    data.qpos[:] = qpos0

    # Restore the physical tabletop UR5e configuration.
    for joint_name, joint_value in ARM_STARTUP.items():

        adr_arm = joint_address(
            model,
            joint_name,
        )

        data.qpos[adr_arm] = joint_value

    # Then impose only the requested index-finger state.
    for adr, angle in zip(
        addresses,
        q_deg,
    ):
        data.qpos[adr] = math.radians(
            float(angle)
        )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


def get_angles_deg(
    data,
    addresses,
):

    return np.degrees(
        [
            data.qpos[adr]
            for adr in addresses
        ]
    )


def render_state(
    model,
    data,
    renderer,
    label,
    q_deg,
    tid,
    addresses,
):

    renderer.update_scene(
        data,
        camera=-1,
    )

    image = renderer.render()

    path = (
        OUT_DIR
        / f"{label.lower()}.png"
    )

    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(
            image
        )

        draw = ImageDraw.Draw(
            img
        )

        angles = get_angles_deg(
            data,
            addresses,
        )

        tendon_len_mm = (
            data.ten_length[tid]
            * 1000.0
        )

        text = (
            f"{label}\n"
            f"MCP = {angles[0]:.2f} deg\n"
            f"PIP = {angles[1]:.2f} deg\n"
            f"DIP = {angles[2]:.2f} deg\n"
            f"Tendon L = {tendon_len_mm:.2f} mm"
        )

        draw.rectangle(
            (8, 8, 240, 100),
            fill=(255, 255, 255),
        )

        draw.multiline_text(
            (15, 15),
            text,
            fill=(0, 0, 0),
            spacing=3,
        )

        img.save(
            path
        )

    except ImportError:

        import imageio.v3 as iio

        iio.imwrite(
            path,
            image,
        )


    print(
        f"Saved: {path}"
    )


def print_state(
    data,
    addresses,
    tid,
    label,
):

    q = get_angles_deg(
        data,
        addresses,
    )

    length_mm = (
        data.ten_length[tid]
        * 1000.0
    )

    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    print(
        f"MCP = {q[0]:8.3f} deg"
    )

    print(
        f"PIP = {q[1]:8.3f} deg"
    )

    print(
        f"DIP = {q[2]:8.3f} deg"
    )

    print(
        f"Tendon length = "
        f"{length_mm:.3f} mm"
    )


def main():

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Missing model:\n{MODEL_PATH}"
        )


    model = mujoco.MjModel.from_xml_path(
        str(
            MODEL_PATH
        )
    )

    data = mujoco.MjData(
        model
    )

    qpos0 = (
        model.qpos0.copy()
    )

    addresses = [
        joint_address(
            model,
            MCP,
        ),
        joint_address(
            model,
            PIP,
        ),
        joint_address(
            model,
            DIP,
        ),
    ]

    tid = tendon_id(
        model,
        TENDON,
    )


    # ========================================================
    # SAVE REPRODUCIBLE IMAGES FIRST
    # ========================================================

    # Match the model's existing offscreen framebuffer.
    # Default framebuffer is 640 x 480.
    renderer = mujoco.Renderer(
        model,
        height=480,
        width=640,
    )


    print()
    print("=" * 80)
    print(
        "SAVING CONFIGURATION-DEPENDENT "
        "TENDON VISUALIZATION FRAMES"
    )
    print("=" * 80)


    for label, q_deg in STATES:

        set_state(
            model,
            data,
            qpos0,
            addresses,
            q_deg,
        )

        print_state(
            data,
            addresses,
            tid,
            label,
        )

        render_state(
            model,
            data,
            renderer,
            label,
            q_deg,
            tid,
            addresses,
        )


    renderer.close()


    # ========================================================
    # INTERACTIVE VIEWER
    # ========================================================

    print()
    print("=" * 80)
    print(
        "INTERACTIVE VIEWER"
    )
    print("=" * 80)

    print()
    print(
        "The viewer cycles automatically through:"
    )

    print(
        "OPEN -> MID -> HW_CONTACT -> FLEXED"
    )

    print()
    print(
        "Each state is held for 4 seconds."
    )

    print(
        "Close the MuJoCo window to exit."
    )


    index = 0

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:


        # Show tendon if supported by this MuJoCo version.
        try:
            viewer.opt.flags[
                mujoco.mjtVisFlag.mjVIS_TENDON
            ] = True
        except Exception:
            pass


        try:
            viewer.opt.flags[
                mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
            ] = False
        except Exception:
            pass


        last_switch = 0.0


        while viewer.is_running():

            now = time.time()

            if (
                last_switch == 0.0
                or
                now - last_switch
                >= 4.0
            ):

                label, q_deg = (
                    STATES[index]
                )

                set_state(
                    model,
                    data,
                    qpos0,
                    addresses,
                    q_deg,
                )

                print_state(
                    data,
                    addresses,
                    tid,
                    label,
                )

                print(
                    f"Viewer state: {label}"
                )

                index = (
                    index + 1
                ) % len(STATES)

                last_switch = now


            mujoco.mj_forward(
                model,
                data,
            )

            viewer.sync()

            time.sleep(
                0.02
            )


if __name__ == "__main__":
    main()
