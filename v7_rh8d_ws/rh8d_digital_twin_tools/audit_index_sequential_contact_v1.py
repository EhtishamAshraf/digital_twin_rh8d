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


# ============================================================
# Top geometry candidates from 2D search.
#
# label, center
# ============================================================

CANDIDATES = [

    (
        "F12_S4",
        np.array(
            [0.602564332,
             0.877984809,
             1.570481281]
        ),
    ),

    (
        "F14_S6",
        np.array(
            [0.603789436,
             0.875636806,
             1.571474259]
        ),
    ),

    (
        "F12_S2",
        np.array(
            [0.602964805,
             0.879941518,
             1.570585751]
        ),
    ),

    (
        "F14_S0",
        np.array(
            [0.604990853,
             0.881506932,
             1.571787669]
        ),
    ),

    (
        "F10_S0",
        np.array(
            [0.601739702,
             0.882289520,
             1.569592773]
        ),
    ),

    (
        "F16_S8",
        np.array(
            [0.605014539,
             0.873288804,
             1.572467237]
        ),
    ),
]


# ============================================================
# Candidate A mechanics -- frozen.
# ============================================================

SCALE = 5.1e-6
TENDON_STIFFNESS = 2100.0

OPEN_ENCODER = 100.0

CONTACT_STIFFNESS = 28000.0
CONTACT_DAMPING = 1.0
CONTACT_FRICTION = 1.2


# Hardware
VIDEO_TIME = 50.50
VIDEO_BAG_OFFSET = 11.53

TARGET_BAG_TIME = (
    VIDEO_TIME
    - VIDEO_BAG_OFFSET
)


FIXTURE_NAME = "index_contact_fixture"

TENDON_NAME = "idx_flex_spatial"

SPOOL_NAME = "index_motor_spool"
ACTUATOR_NAME = "pos_index_tendon"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

MID_GEOM = "collision_idx_mid"
TIP_GEOM = "collision_idx_tip"


def oid(model, typ, name):

    i = mujoco.mj_name2id(
        model,
        typ,
        name,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing object: {name}"
        )

    return i


def qadr(model, name):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return model.jnt_qposadr[
        jid
    ]


def set_spool(
    data,
    qadr_spool,
    dof_spool,
    actuator,
    encoder,
):

    q = max(
        0.0,
        (encoder - OPEN_ENCODER)
        * SCALE,
    )

    data.qpos[
        qadr_spool
    ] = q

    data.qvel[
        dof_spool
    ] = 0.0

    data.ctrl[
        actuator
    ] = q


def bottle_contacts(
    model,
    data,
    fixture_gid,
):

    names = []

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

        if name is None:
            name = f"geom_{other}"

        names.append(
            name
        )

    return sorted(
        set(names)
    )


def first_true(
    active,
    encoders,
):

    ids = np.where(
        active
    )[0]

    if len(ids) == 0:
        return None

    return float(
        encoders[
            ids[0]
        ]
    )


def sustained_onset(
    active,
    encoders,
    dt,
    window_s=0.050,
    duty_required=0.80,
):

    active = np.asarray(
        active,
        dtype=float,
    )

    n = max(
        1,
        int(
            round(
                window_s / dt
            )
        ),
    )

    if len(active) < n:
        return None

    duty = np.convolve(
        active,
        np.ones(n),
        mode="valid",
    ) / n

    ids = np.where(
        duty >= duty_required
    )[0]

    if len(ids) == 0:
        return None

    return float(
        encoders[
            ids[0]
        ]
    )


def nearest_index(
    encoders,
    target,
):

    return int(
        np.argmin(
            np.abs(
                encoders
                - target
            )
        )
    )


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            " python3 audit_index_sequential_contact_v1.py "
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
    ) = base.read_bag(
        bag
    )


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


    encoder_time = (
        encoder_data[:, 0]
    )

    encoder_position = (
        base.median_filter_5(
            encoder_data[:, 1]
        )
    )


    print()
    print("=" * 110)
    print(
        "INDEX SEQUENTIAL-CONTACT AUDIT"
    )
    print("=" * 110)

    print()
    print(
        "Hardware qualitative sequence:"
    )

    print(
        "  MCP moves first"
    )

    print(
        "  middle phalanx contacts bottle"
    )

    print(
        "  PIP flexion increases"
    )

    print(
        "  distal/tip contacts bottle"
    )

    print(
        "  DIP flexion increases"
    )


    for label, center in CANDIDATES:


        model = mujoco.MjModel.from_xml_path(
            str(
                MODEL_PATH
            )
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


        model.geom_pos[
            fixture_gid
        ] = center


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


        spool_q = (
            model.jnt_qposadr[
                spool_jid
            ]
        )


        spool_dof = (
            model.jnt_dofadr[
                spool_jid
            ]
        )


        actuator = oid(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            ACTUATOR_NAME,
        )


        mcp_q = qadr(
            model,
            MCP_NAME,
        )

        pip_q = qadr(
            model,
            PIP_NAME,
        )

        dip_q = qadr(
            model,
            DIP_NAME,
        )


        # ----------------------------------------------------
        # Open reference
        # ----------------------------------------------------

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


        open_mcp = math.degrees(
            data.qpos[
                mcp_q
            ]
        )

        open_pip = math.degrees(
            data.qpos[
                pip_q
            ]
        )

        open_dip = math.degrees(
            data.qpos[
                dip_q
            ]
        )


        # ----------------------------------------------------
        # Replay
        # ----------------------------------------------------

        times = []
        encoders = []

        mid_active = []
        tip_active = []
        other_active = []

        mcp_angles = []
        pip_angles = []
        dip_angles = []


        elapsed = 0.0

        dt = model.opt.timestep

        duration = (
            hold_end
            - approach_start
        )


        while elapsed <= duration:

            bag_time = (
                approach_start
                + elapsed
            )


            encoder = float(
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
                encoder,
            )


            mujoco.mj_step(
                model,
                data,
            )


            names = bottle_contacts(
                model,
                data,
                fixture_gid,
            )


            mid_active.append(
                MID_GEOM
                in names
            )


            tip_active.append(
                TIP_GEOM
                in names
            )


            other_active.append(
                any(
                    not n.startswith(
                        "collision_idx_"
                    )
                    for n in names
                )
            )


            times.append(
                bag_time
            )

            encoders.append(
                encoder
            )


            mcp_angles.append(
                math.degrees(
                    data.qpos[
                        mcp_q
                    ]
                )
                - open_mcp
            )


            pip_angles.append(
                math.degrees(
                    data.qpos[
                        pip_q
                    ]
                )
                - open_pip
            )


            dip_angles.append(
                math.degrees(
                    data.qpos[
                        dip_q
                    ]
                )
                - open_dip
            )


            elapsed += dt


        times = np.asarray(
            times
        )

        encoders = np.asarray(
            encoders
        )

        mid_active = np.asarray(
            mid_active,
            dtype=bool,
        )

        tip_active = np.asarray(
            tip_active,
            dtype=bool,
        )

        other_active = np.asarray(
            other_active,
            dtype=bool,
        )

        mcp_angles = np.asarray(
            mcp_angles
        )

        pip_angles = np.asarray(
            pip_angles
        )

        dip_angles = np.asarray(
            dip_angles
        )


        first_mid = first_true(
            mid_active,
            encoders,
        )

        first_tip = first_true(
            tip_active,
            encoders,
        )


        sustained_mid = sustained_onset(
            mid_active,
            encoders,
            dt,
        )

        sustained_tip = sustained_onset(
            tip_active,
            encoders,
            dt,
        )


        hold_mask = (
            (times >= hold_start)
            &
            (times <= hold_end)
        )


        mid_hold_duty = float(
            np.mean(
                mid_active[
                    hold_mask
                ]
            )
        )


        tip_hold_duty = float(
            np.mean(
                tip_active[
                    hold_mask
                ]
            )
        )


        other_hold_duty = float(
            np.mean(
                other_active[
                    hold_mask
                ]
            )
        )


        target_idx = int(
            np.argmin(
                np.abs(
                    times
                    - TARGET_BAG_TIME
                )
            )
        )


        def state_at_encoder(
            encoder_value,
        ):

            if encoder_value is None:

                return None

            i = nearest_index(
                encoders,
                encoder_value,
            )

            return (
                mcp_angles[i],
                pip_angles[i],
                dip_angles[i],
            )


        mid_state = state_at_encoder(
            sustained_mid
        )


        tip_state = state_at_encoder(
            sustained_tip
        )


        print()
        print("=" * 110)

        print(
            f"{label}"
        )

        print(
            "center = "
            f"{center[0]:.9f} "
            f"{center[1]:.9f} "
            f"{center[2]:.9f}"
        )


        print()
        print(
            f"first middle contact      = "
            f"{first_mid}"
        )

        print(
            f"sustained middle contact  = "
            f"{sustained_mid}"
        )

        print(
            f"first tip contact         = "
            f"{first_tip}"
        )

        print(
            f"sustained tip contact     = "
            f"{sustained_tip}"
        )


        print()
        print(
            f"middle hold duty          = "
            f"{100*mid_hold_duty:.1f}%"
        )

        print(
            f"tip hold duty             = "
            f"{100*tip_hold_duty:.1f}%"
        )

        print(
            f"non-index hold duty       = "
            f"{100*other_hold_duty:.1f}%"
        )


        print()
        print(
            "joint pose at sustained middle contact:"
        )

        print(
            f"  {mid_state}"
        )


        print(
            "joint pose at sustained tip contact:"
        )

        print(
            f"  {tip_state}"
        )


        print()
        print(
            "video-matched hold pose:"
        )

        print(
            f"  MCP={mcp_angles[target_idx]:.3f}"
        )

        print(
            f"  PIP={pip_angles[target_idx]:.3f}"
        )

        print(
            f"  DIP={dip_angles[target_idx]:.3f}"
        )


        if (
            sustained_mid is not None
            and
            sustained_tip is not None
        ):

            if sustained_mid < sustained_tip:

                print()
                print(
                    "SEQUENCE = MIDDLE -> TIP   ✓"
                )

            elif sustained_tip < sustained_mid:

                print()
                print(
                    "SEQUENCE = TIP -> MIDDLE   ✗"
                )

            else:

                print()
                print(
                    "SEQUENCE = SIMULTANEOUS"
                )

        else:

            print()
            print(
                "SEQUENCE = incomplete"
            )


    print()
    print("=" * 110)


if __name__ == "__main__":
    main()
