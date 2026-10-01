#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import mujoco
import numpy as np

import fit_index_contact_v1 as base


# ============================================================
# PURPOSE
# ============================================================
#
# Identify calibration-bottle POSITION in the finger plane.
#
# Do NOT use loaded MCP/PIP pose as a fitting objective.
#
# Geometry objectives:
#
#   - no bottle contact in open configuration
#   - no contact with non-index fingers
#   - sustained index-tip contact during real hold
#   - tip contact onset near hardware ~1020 ticks
#
# Candidate-A hand mechanics remain frozen.
# ============================================================


MODEL_PATH = (
    base.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_v1.xml"
)


# ============================================================
# Center prior
#
# Use V2 only as the CENTER of the search.
# It is NOT assumed to be correct.
# ============================================================

CENTER_PRIOR = np.array(
    [
        0.593611823,
        0.884245992,
        1.564105533,
    ],
    dtype=float,
)


# ============================================================
# 2D search grid [mm]
#
# V1 relative to V2 was approximately:
#
#   forward = -3.87 mm
#   side    = -11.63 mm
#
# so this grid contains both V1 and V2 plus the missing
# orthogonal directions.
# ============================================================

FORWARD_OFFSETS_MM = [
     8.0,
    10.0,
    12.0,
    14.0,
    16.0,
]

SIDE_OFFSETS_MM = [
     0.0,
     2.0,
     4.0,
     6.0,
     8.0,
]


# ============================================================
# Candidate A -- frozen
# ============================================================

SCALE = 5.1e-6
TENDON_STIFFNESS = 2100.0

OPEN_ENCODER = 100.0

CONTACT_STIFFNESS = 28000.0
CONTACT_DAMPING = 1.0
CONTACT_FRICTION = 1.2


# ============================================================
# Hardware calibration event
# ============================================================

HW_TIP_CONTACT_ENCODER = 1020.0

HW_MCP = 14.5355
HW_PIP = 23.8110
HW_DIP = 9.3270

VIDEO_TIME = 50.50
VIDEO_BAG_OFFSET = 11.53

TARGET_BAG_TIME = (
    VIDEO_TIME
    - VIDEO_BAG_OFFSET
)

TARGET_HALF_WINDOW = 0.10


# ============================================================
# Names
# ============================================================

FIXTURE_NAME = "index_contact_fixture"

TENDON_NAME = "idx_flex_spatial"

SPOOL_NAME = "index_motor_spool"
ACTUATOR_NAME = "pos_index_tendon"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

FTS_NAME = "index_tip_force"


def oid(
    model,
    typ,
    name,
):

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


def qadr(
    model,
    joint,
):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint,
    )

    return model.jnt_qposadr[
        jid
    ]


def set_spool(
    data,
    spool_qadr,
    spool_dofadr,
    actuator,
    encoder,
):

    displacement = max(
        0.0,
        (
            encoder
            - OPEN_ENCODER
        )
        * SCALE,
    )

    data.qpos[
        spool_qadr
    ] = displacement

    data.qvel[
        spool_dofadr
    ] = 0.0

    data.ctrl[
        actuator
    ] = displacement


def bottle_contacts(
    model,
    data,
    fixture_gid,
):

    names = []


    for i in range(
        data.ncon
    ):

        c = data.contact[
            i
        ]


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
        set(
            names
        )
    )


def is_index(
    name,
):

    return name.startswith(
        "collision_idx_"
    )


def sustained_onset(
    active,
    encoders,
    dt,
    duration=0.050,
    duty_required=0.80,
):

    active = np.asarray(
        active,
        dtype=float,
    )

    encoders = np.asarray(
        encoders,
        dtype=float,
    )


    n = max(
        1,
        int(
            round(
                duration / dt
            )
        ),
    )


    if len(active) < n:
        return None


    kernel = np.ones(
        n,
        dtype=float,
    )


    duty = np.convolve(
        active,
        kernel,
        mode="valid",
    ) / n


    indices = np.where(
        duty >= duty_required
    )[0]


    if len(indices) == 0:
        return None


    return float(
        encoders[
            indices[0]
        ]
    )


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            " python3 search_index_bottle_pose_2d_v4.py "
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


    # ========================================================
    # Establish finger-plane coordinate basis once
    # ========================================================

    basis_model = mujoco.MjModel.from_xml_path(
        str(
            MODEL_PATH
        )
    )

    basis_data = mujoco.MjData(
        basis_model
    )


    basis_model.opt.gravity[:] = 0.0


    fixture_basis = oid(
        basis_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FIXTURE_NAME,
    )


    # Cylinder local Z = cylinder axis.
    mujoco.mj_forward(
        basis_model,
        basis_data,
    )


    cylinder_R = (
        basis_data.geom_xmat[
            fixture_basis
        ].reshape(
            3,
            3,
        )
    )


    cylinder_axis = unit(
        cylinder_R[:, 2]
    )


    mcp_jid = oid(
        basis_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        MCP_NAME,
    )

    pip_jid = oid(
        basis_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        PIP_NAME,
    )


    mcp_world = (
        basis_data.xanchor[
            mcp_jid
        ].copy()
    )

    pip_world = (
        basis_data.xanchor[
            pip_jid
        ].copy()
    )


    forward_raw = (
        pip_world
        - mcp_world
    )


    # Project mechanical MCP->PIP direction into plane
    # perpendicular to bottle axis.
    forward = unit(
        forward_raw
        -
        np.dot(
            forward_raw,
            cylinder_axis,
        )
        * cylinder_axis
    )


    side = unit(
        np.cross(
            cylinder_axis,
            forward,
        )
    )


    # Choose side sign so it points roughly from MCP
    # toward our center prior.
    if np.dot(
        CENTER_PRIOR
        - mcp_world,
        side,
    ) < 0:

        side *= -1.0


    print()
    print("=" * 120)
    print(
        "INDEX BOTTLE POSITION 2D SEARCH V4"
    )
    print("=" * 120)

    print()
    print(
        "Search prior =",
        CENTER_PRIOR,
    )

    print(
        "forward axis =",
        forward,
    )

    print(
        "side axis    =",
        side,
    )

    print(
        "bottle axis  =",
        cylinder_axis,
    )


    total = (
        len(
            FORWARD_OFFSETS_MM
        )
        *
        len(
            SIDE_OFFSETS_MM
        )
    )


    print()
    print(
        f"Testing {total} positions..."
    )


    results = []


    for forward_mm in (
        FORWARD_OFFSETS_MM
    ):


        for side_mm in (
            SIDE_OFFSETS_MM
        ):


            center = (
                CENTER_PRIOR
                +
                forward
                * (
                    forward_mm
                    / 1000.0
                )
                +
                side
                * (
                    side_mm
                    / 1000.0
                )
            )


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


            sensor_id = oid(
                model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                FTS_NAME,
            )


            sensor_adr = (
                model.sensor_adr[
                    sensor_id
                ]
            )


            # =================================================
            # Open equilibrium
            # =================================================

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


            open_contacts = bottle_contacts(
                model,
                data,
                fixture_gid,
            )


            # =================================================
            # Replay
            # =================================================

            times = []
            encoders = []

            tip_active = []
            index_active = []
            other_active = []

            fts_values = []

            mcp_values = []
            pip_values = []
            dip_values = []


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


                has_tip = (
                    "collision_idx_tip"
                    in names
                )


                has_index = any(
                    is_index(
                        name
                    )
                    for name in names
                )


                has_other = any(
                    not is_index(
                        name
                    )
                    for name in names
                )


                xyz = (
                    data.sensordata[
                        sensor_adr:
                        sensor_adr + 3
                    ]
                )


                fts = float(
                    np.linalg.norm(
                        xyz
                    )
                )


                times.append(
                    bag_time
                )

                encoders.append(
                    encoder
                )

                tip_active.append(
                    has_tip
                )

                index_active.append(
                    has_index
                )

                other_active.append(
                    has_other
                )

                fts_values.append(
                    fts
                )


                mcp_values.append(
                    math.degrees(
                        data.qpos[
                            mcp_q
                        ]
                    )
                    - open_mcp
                )


                pip_values.append(
                    math.degrees(
                        data.qpos[
                            pip_q
                        ]
                    )
                    - open_pip
                )


                dip_values.append(
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

            tip_active = np.asarray(
                tip_active,
                dtype=bool,
            )

            index_active = np.asarray(
                index_active,
                dtype=bool,
            )

            other_active = np.asarray(
                other_active,
                dtype=bool,
            )

            fts_values = np.asarray(
                fts_values
            )

            mcp_values = np.asarray(
                mcp_values
            )

            pip_values = np.asarray(
                pip_values
            )

            dip_values = np.asarray(
                dip_values
            )


            # =================================================
            # Sustained onset
            # =================================================

            tip_onset = sustained_onset(
                tip_active,
                encoders,
                dt,
                duration=0.050,
                duty_required=0.80,
            )


            # =================================================
            # Hold / target windows
            # =================================================

            hold_mask = (
                (times >= hold_start)
                &
                (times <= hold_end)
            )


            target_mask = (
                (
                    times
                    >= TARGET_BAG_TIME
                    - TARGET_HALF_WINDOW
                )
                &
                (
                    times
                    <= TARGET_BAG_TIME
                    + TARGET_HALF_WINDOW
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


            tip_target_duty = float(
                np.mean(
                    tip_active[
                        target_mask
                    ]
                )
            )


            index_target_duty = float(
                np.mean(
                    index_active[
                        target_mask
                    ]
                )
            )


            other_target_duty = float(
                np.mean(
                    other_active[
                        target_mask
                    ]
                )
            )


            target_fts = float(
                np.median(
                    fts_values[
                        target_mask
                    ]
                )
            )


            sim_mcp = float(
                np.median(
                    mcp_values[
                        target_mask
                    ]
                )
            )

            sim_pip = float(
                np.median(
                    pip_values[
                        target_mask
                    ]
                )
            )

            sim_dip = float(
                np.median(
                    dip_values[
                        target_mask
                    ]
                )
            )


            # =================================================
            # Geometry score
            #
            # Deliberately NO loaded-pose term.
            # =================================================

            score = 0.0


            if open_contacts:

                score += 1000.0


            score += (
                1000.0
                * other_target_duty
            )


            score += (
                500.0
                * other_hold_duty
            )


            score += (
                300.0
                * (
                    1.0
                    - tip_target_duty
                )
            )


            score += (
                100.0
                * (
                    1.0
                    - tip_hold_duty
                )
            )


            if tip_onset is None:

                onset_error = float(
                    "inf"
                )

                score += 300.0

            else:

                onset_error = abs(
                    tip_onset
                    - HW_TIP_CONTACT_ENCODER
                )

                score += (
                    0.20
                    * onset_error
                )


            # Weak prior only:
            # prefer smaller corrections when two geometries
            # behave equivalently.
            prior_distance_mm = math.sqrt(
                forward_mm**2
                +
                side_mm**2
            )


            score += (
                0.02
                * prior_distance_mm
            )


            useful = (
                not open_contacts

                and

                other_hold_duty
                <= 0.01

                and

                other_target_duty
                <= 0.01

                and

                tip_target_duty
                >= 0.80

                and

                tip_hold_duty
                >= 0.50

                and

                tip_onset is not None

                and

                onset_error
                <= 30.0
            )


            results.append(
                {
                    "forward_mm":
                        forward_mm,

                    "side_mm":
                        side_mm,

                    "center":
                        center.copy(),

                    "open_contacts":
                        open_contacts,

                    "tip_onset":
                        tip_onset,

                    "onset_error":
                        onset_error,

                    "tip_target_duty":
                        tip_target_duty,

                    "index_target_duty":
                        index_target_duty,

                    "other_target_duty":
                        other_target_duty,

                    "tip_hold_duty":
                        tip_hold_duty,

                    "other_hold_duty":
                        other_hold_duty,

                    "target_fts":
                        target_fts,

                    "sim_mcp":
                        sim_mcp,

                    "sim_pip":
                        sim_pip,

                    "sim_dip":
                        sim_dip,

                    "score":
                        score,

                    "useful":
                        useful,
                }
            )


            onset_text = (
                "NONE"
                if tip_onset is None
                else f"{tip_onset:.1f}"
            )


            print(
                f"forward={forward_mm:+5.1f} mm  "
                f"side={side_mm:+5.1f} mm  "
                f"onset={onset_text:>7s}  "
                f"target tip={100*tip_target_duty:5.1f}%  "
                f"hold tip={100*tip_hold_duty:5.1f}%  "
                f"other={100*other_hold_duty:5.1f}%  "
                f"FTS={target_fts:7.4f}"
            )


    # ========================================================
    # Ranking
    # ========================================================

    ranked = sorted(
        results,
        key=lambda r:
            r[
                "score"
            ],
    )


    print()
    print("=" * 120)
    print(
        "TOP 15 GEOMETRY CANDIDATES"
    )
    print("=" * 120)


    for rank, r in enumerate(
        ranked[:15],
        start=1,
    ):

        print()
        print(
            f"#{rank:2d} "
            f"forward={r['forward_mm']:+.1f} mm   "
            f"side={r['side_mm']:+.1f} mm   "
            f"score={r['score']:.3f}"
        )

        print(
            "  center = "
            f"{r['center'][0]:.9f} "
            f"{r['center'][1]:.9f} "
            f"{r['center'][2]:.9f}"
        )

        print(
            f"  sustained tip onset = "
            f"{r['tip_onset']}"
        )

        print(
            f"  target tip duty     = "
            f"{100*r['tip_target_duty']:.1f}%"
        )

        print(
            f"  hold tip duty       = "
            f"{100*r['tip_hold_duty']:.1f}%"
        )

        print(
            f"  target other duty   = "
            f"{100*r['other_target_duty']:.1f}%"
        )

        print(
            f"  hold other duty     = "
            f"{100*r['other_hold_duty']:.1f}%"
        )

        print(
            f"  target FTS          = "
            f"{r['target_fts']:.5f} N"
        )

        print(
            f"  DIAGNOSTIC loaded pose = "
            f"({r['sim_mcp']:.2f}, "
            f"{r['sim_pip']:.2f}, "
            f"{r['sim_dip']:.2f}) deg"
        )


    useful = [
        r
        for r in results
        if r[
            "useful"
        ]
    ]


    print()
    print("=" * 120)
    print(
        "PHYSICALLY USEFUL GEOMETRY CANDIDATES"
    )
    print("=" * 120)


    if not useful:

        print(
            "NONE"
        )

    else:

        for r in sorted(
            useful,
            key=lambda r:
                (
                    r[
                        "onset_error"
                    ],
                    -r[
                        "tip_target_duty"
                    ],
                ),
        ):

            print()
            print(
                f"forward={r['forward_mm']:+.1f} mm   "
                f"side={r['side_mm']:+.1f} mm"
            )

            print(
                "  center = "
                f"{r['center'][0]:.9f} "
                f"{r['center'][1]:.9f} "
                f"{r['center'][2]:.9f}"
            )

            print(
                f"  onset = "
                f"{r['tip_onset']:.1f} ticks"
            )

            print(
                f"  target tip duty = "
                f"{100*r['tip_target_duty']:.1f}%"
            )

            print(
                f"  hold tip duty   = "
                f"{100*r['tip_hold_duty']:.1f}%"
            )

            print(
                f"  loaded pose (diagnostic only) = "
                f"({r['sim_mcp']:.2f}, "
                f"{r['sim_pip']:.2f}, "
                f"{r['sim_dip']:.2f})"
            )


    print()
    print("=" * 120)


if __name__ == "__main__":
    main()
