#!/usr/bin/env python3

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

import build_audit_index_spatial_tendon_v34a as b


WS = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    WS
    / "src"
    / "ur5e_mujoco_description"
    / "mjcf"
    / "ur5e_rh8d_index_spatial_v34a5.xml"
)


# -------------------------------------------------------------------------
# REAL TABLETOP / CONTACT-EXPERIMENT ARM POSE
# -------------------------------------------------------------------------

TABLETOP_POSE = {
    "shoulder_pan_joint":   1.4927,
    "shoulder_lift_joint": -0.8678,
    "elbow_joint":         -2.4300,
    "wrist_1_joint":       -0.6596,
    "wrist_2_joint":       -0.0347,
    "wrist_3_joint":        3.1260,
}


# Full hardware free-space staircase.
ENCODER_SEQUENCE = [
    100,
    500,
    1000,
    1500,
    2000,
    2500,
    3000,
    3500,
    3000,
    2500,
    2000,
    1500,
    1000,
    500,
    100,
]


HOLD_TIME = 1.5


def joint_id(model, name):
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )


def qadr(model, name):
    jid = joint_id(model, name)

    if jid < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    return int(
        model.jnt_qposadr[jid]
    )


def set_joint(model, data, name, value):
    jid = joint_id(
        model,
        name,
    )

    if jid < 0:
        print(
            f"[WARNING] joint not found: {name}"
        )
        return

    data.qpos[
        model.jnt_qposadr[jid]
    ] = value


def get_angles_deg(model, data):

    return np.degrees(
        [
            data.qpos[
                qadr(
                    model,
                    b.MCP,
                )
            ],
            data.qpos[
                qadr(
                    model,
                    b.PIP,
                )
            ],
            data.qpos[
                qadr(
                    model,
                    b.DIP,
                )
            ],
        ]
    )


print()
print("=" * 110)
print(
    "V34A6 — TABLETOP + FULL-RANGE SPATIAL-TENDON AUDIT"
)
print("=" * 110)


if not MODEL_PATH.exists():
    raise RuntimeError(
        f"V34A5 model not found:\n{MODEL_PATH}"
    )


model = mujoco.MjModel.from_xml_path(
    str(
        MODEL_PATH
    )
)

data = mujoco.MjData(
    model
)


# =========================================================================
# 1. SET THE REAL TABLETOP ARM CONFIGURATION
# =========================================================================

for name, value in TABLETOP_POSE.items():
    set_joint(
        model,
        data,
        name,
        value,
    )


# Start index fully open.
set_joint(
    model,
    data,
    b.MCP,
    0.0,
)

set_joint(
    model,
    data,
    b.PIP,
    0.0,
)

set_joint(
    model,
    data,
    b.DIP,
    0.0,
)

set_joint(
    model,
    data,
    b.SPOOL,
    0.0,
)


mujoco.mj_forward(
    model,
    data,
)


# =========================================================================
# 2. MAKE DISTAL TENDON ANCHOR VERY OBVIOUS
# =========================================================================

anchor_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    b.ANCHOR,
)

if anchor_id >= 0:

    model.site_rgba[
        anchor_id
    ] = np.array(
        [
            0.0,
            1.0,
            1.0,
            1.0,
        ]
    )

    model.site_size[
        anchor_id,
        0
    ] = 0.0035

    print()
    print(
        "DISTAL ANCHOR:"
    )

    print(
        "  name =",
        b.ANCHOR,
    )

    print(
        "  local position [m] =",
        model.site_pos[
            anchor_id
        ],
    )

    print(
        "  global position [m] =",
        data.site_xpos[
            anchor_id
        ],
    )

    print(
        "  CYAN sphere in viewer = current distal tendon anchor."
    )

else:

    print(
        "[WARNING] distal anchor site not found."
    )


# =========================================================================
# 3. FIND / SHOW THE V5 BOTTLE FIXTURE
#
# Keep it NON-COLLIDING for this free-space audit.
# =========================================================================

bottle_ids = []

for gid in range(
    model.ngeom
):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        gid,
    )

    if not name:
        continue

    lower = name.lower()

    if (
        "bottle" in lower
        or
        "index_contact_fixture" in lower
    ):

        bottle_ids.append(
            gid
        )

        # Visible green-ish fixture.
        model.geom_rgba[
            gid
        ] = np.array(
            [
                0.2,
                0.8,
                0.2,
                0.45,
            ]
        )


print()
print(
    "BOTTLE/FIXTURE GEOMS FOUND =",
    len(
        bottle_ids
    )
)


for gid in bottle_ids:

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        gid,
    )

    print(
        f"  {name}"
    )

    print(
        "    global xyz =",
        data.geom_xpos[
            gid
        ],
    )

    print(
        "    contype/conaffinity =",
        int(
            model.geom_contype[
                gid
            ]
        ),
        int(
            model.geom_conaffinity[
                gid
            ]
        ),
    )


# =========================================================================
# 4. FREEZE SUPPORT AT THIS REAL TABLETOP POSE
# =========================================================================

frozen = b.create_fixed_state(
    model,
    data,
)


# =========================================================================
# 5. MOTOR SPOOL ACTUATOR
# =========================================================================

actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "pos_index_spool_v34",
)

if actuator_id < 0:

    raise RuntimeError(
        "Missing actuator pos_index_spool_v34"
    )


# =========================================================================
# 6. VIEWER
# =========================================================================

viewer = mujoco.viewer.launch_passive(
    model,
    data,
)


try:

    mujoco.mj_forward(
        model,
        data,
    )


    index_body = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "Index_Middle",
    )


    if index_body >= 0:

        look = data.xpos[
            index_body
        ].copy()


        if bottle_ids:

            bottle_center = np.mean(
                [
                    data.geom_xpos[
                        gid
                    ]
                    for gid in bottle_ids
                ],
                axis=0,
            )

            look = (
                0.5
                *
                (
                    look
                    +
                    bottle_center
                )
            )


        viewer.cam.lookat[:] = look

        viewer.cam.distance = 0.38


    print()
    print("=" * 110)
    print(
        "FULL HARDWARE-RANGE FREE-SPACE SWEEP"
    )
    print("=" * 110)

    print()
    print(
        "The cylinder is shown at the V5/tabletop location but remains "
        "NON-COLLIDING in this audit."
    )

    print(
        "CYAN sphere = distal tendon anchor."
    )

    print(
        "MAGENTA line = active spatial flexor tendon."
    )


    print()
    print(
        "encoder, MCP_deg, PIP_deg, DIP_deg"
    )


    for encoder in ENCODER_SEQUENCE:

        spool_target = (
            b.METERS_PER_TICK
            *
            (
                float(
                    encoder
                )
                -
                b.ENCODER_ZERO
            )
        )


        data.ctrl[
            actuator_id
        ] = spool_target


        nsteps = max(
            1,
            int(
                HOLD_TIME
                /
                model.opt.timestep
            ),
        )


        for _ in range(
            nsteps
        ):

            if not viewer.is_running():
                break


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


        if not viewer.is_running():
            break


        q = get_angles_deg(
            model,
            data,
        )


        print(
            f"{encoder:4d}, "
            f"{q[0]:8.3f}, "
            f"{q[1]:8.3f}, "
            f"{q[2]:8.3f}"
        )


    print()
    print("=" * 110)
    print(
        "V34A6 VIEWER CHECK"
    )
    print("=" * 110)

    print()
    print(
        "Please inspect:"
    )

    print(
        "  - Does hand/table/bottle placement now resemble the real H1/H2 setup?"
    )

    print(
        "  - Is the CYAN anchor actually on the BACKSIDE distal attachment "
        "seen in the CAD/hardware?"
    )

    print(
        "  - Does the MAGENTA tendon lie inside the physical finger volume?"
    )

    print(
        "  - At 3500 ticks, do all MCP/PIP/DIP visibly flex?"
    )

    print()
    print(
        "Viewer remains open for your screen recording."
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


print()
print("=" * 110)
print(
    "V34A6 COMPLETE"
)
print("=" * 110)
