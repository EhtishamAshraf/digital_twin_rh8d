#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import mujoco
import numpy as np

import fit_index_contact_v1 as base


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

VIDEO_TIME = 50.50
VIDEO_BAG_OFFSET = 11.53
TARGET_BAG_TIME = VIDEO_TIME - VIDEO_BAG_OFFSET

TENDON_NAME = "idx_flex_spatial"
SPOOL_NAME = "index_motor_spool"
ACTUATOR_NAME = "pos_index_tendon"

DISTAL_BODY = "Index_Distal"
TIP_BODY = "Index_TipDummy"
TIP_GEOM = "collision_idx_tip"
FIXTURE = "index_contact_fixture"


def oid(model, typ, name):
    idx = mujoco.mj_name2id(
        model,
        typ,
        name,
    )
    if idx < 0:
        raise RuntimeError(f"Missing object: {name}")
    return idx


def set_spool(
    data,
    qadr,
    dofadr,
    actuator,
    encoder,
):
    q = max(
        0.0,
        (encoder - OPEN_ENCODER) * SCALE,
    )

    data.qpos[qadr] = q
    data.qvel[dofadr] = 0.0
    data.ctrl[actuator] = q


def world_to_body_local(
    data,
    body_id,
    point_world,
):
    R = data.xmat[
        body_id
    ].reshape(3, 3)

    p = data.xpos[
        body_id
    ]

    return R.T @ (
        point_world - p
    )


def main():

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python3 audit_index_distal_collision_coverage.py "
            "/path/to/contact_rosbag"
        )
        sys.exit(1)

    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()

    (
        states,
        force_data,
        encoder_data,
    ) = base.read_bag(bag)

    transitions = base.compress_states(
        states
    )

    (
        approach_start,
        regulation_start,
        hold_start,
        hold_end,
    ) = base.find_intervals(
        transitions
    )

    encoder_time = encoder_data[:, 0]
    encoder_position = base.median_filter_5(
        encoder_data[:, 1]
    )

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

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
        FIXTURE,
    )

    tip_gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        TIP_GEOM,
    )

    distal_bid = oid(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        DISTAL_BODY,
    )

    tip_bid = oid(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        TIP_BODY,
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

    spool_q = model.jnt_qposadr[
        spool_jid
    ]

    spool_dof = model.jnt_dofadr[
        spool_jid
    ]

    actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAME,
    )

    # --------------------------------------------------------
    # Print the capsule's nominal extent
    # --------------------------------------------------------

    radius = model.geom_size[
        tip_gid,
        0
    ]

    half_length = model.geom_size[
        tip_gid,
        1
    ]

    center_local = model.geom_pos[
        tip_gid
    ].copy()

    z_min = (
        center_local[2]
        - half_length
        - radius
    )

    z_max = (
        center_local[2]
        + half_length
        + radius
    )

    print()
    print("=" * 100)
    print("DISTAL COLLISION COVERAGE")
    print("=" * 100)

    print()
    print(
        f"Tip capsule radius       = "
        f"{radius*1000:.3f} mm"
    )

    print(
        f"Tip capsule half-length  = "
        f"{half_length*1000:.3f} mm"
    )

    print(
        f"Tip capsule local center = "
        f"{center_local}"
    )

    print(
        f"Approx capsule Z extent  = "
        f"{z_min*1000:.3f} to "
        f"{z_max*1000:.3f} mm"
    )

    print(
        f"Approx end-to-end length = "
        f"{(z_max-z_min)*1000:.3f} mm"
    )

    print()
    print(
        "TipDummy relative to Index_Distal =",
        model.body_pos[
            tip_bid
        ],
    )

    # --------------------------------------------------------
    # Replay to the hardware video-matched hold state
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
        set_spool(
            data,
            spool_q,
            spool_dof,
            actuator,
            OPEN_ENCODER,
        )

        mujoco.mj_step(
            model,
            data,
        )

    dt = model.opt.timestep

    elapsed = 0.0

    target_duration = (
        TARGET_BAG_TIME
        - approach_start
    )

    target_encoder = OPEN_ENCODER

    while elapsed <= target_duration:

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
            spool_q,
            spool_dof,
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
    print("VIDEO-MATCHED CONTACT STATE")
    print("=" * 100)

    print(
        f"encoder = {target_encoder:.1f} ticks"
    )

    print()
    print(
        "Index_Distal world origin =",
        data.xpos[
            distal_bid
        ],
    )

    print(
        "TipDummy world origin      =",
        data.xpos[
            tip_bid
        ],
    )

    print(
        "tip capsule world center   =",
        data.geom_xpos[
            tip_gid
        ],
    )

    # --------------------------------------------------------
    # List every bottle contact and transform it into the
    # Index_Distal local frame.
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("BOTTLE CONTACTS IN INDEX_DISTAL LOCAL COORDINATES")
    print("=" * 100)

    found = False

    local_points = []

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

        other = (
            c.geom2
            if c.geom1 == fixture_gid
            else c.geom1
        )

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            other,
        )

        if name != TIP_GEOM:
            continue

        found = True

        local = world_to_body_local(
            data,
            distal_bid,
            c.pos,
        )

        local_points.append(
            local
        )

        wrench = np.zeros(6)

        mujoco.mj_contactForce(
            model,
            data,
            i,
            wrench,
        )

        print()
        print(
            f"contact #{i}"
        )

        print(
            "  world point =",
            c.pos,
        )

        print(
            "  distal-local point [mm] = "
            f"{local[0]*1000:.3f}  "
            f"{local[1]*1000:.3f}  "
            f"{local[2]*1000:.3f}"
        )

        print(
            f"  |force| = "
            f"{np.linalg.norm(wrench[:3]):.6f} N"
        )

    if not found:
        print("NO TIP/BOTTLE CONTACT FOUND")

    if local_points:

        local_points = np.asarray(
            local_points
        )

        median = np.median(
            local_points,
            axis=0,
        )

        print()
        print("-" * 100)

        print(
            "Median distal-local contact point [mm] = "
            f"{median[0]*1000:.3f}  "
            f"{median[1]*1000:.3f}  "
            f"{median[2]*1000:.3f}"
        )

        print()
        print(
            "Interpretation:"
        )

        print(
            f"  capsule center Z = "
            f"{center_local[2]*1000:.3f} mm"
        )

        print(
            f"  contact Z        = "
            f"{median[2]*1000:.3f} mm"
        )

        if median[2] < (
            center_local[2]
            - half_length
        ):
            print(
                "  contact is near the DISTAL spherical end."
            )

        elif median[2] > (
            center_local[2]
            + half_length
        ):
            print(
                "  contact is near the PROXIMAL spherical end."
            )

        else:
            print(
                "  contact is on the CENTRAL CYLINDRICAL "
                "portion of the capsule."
            )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
