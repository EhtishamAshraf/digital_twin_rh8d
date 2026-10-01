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
# V1 hardware-derived bottle geometry was TOO CLOSE:
#   unwanted middle/small-finger contact.
#
# V2 hardware-derived geometry was TOO FAR:
#   no contact at the hardware hold encoder.
#
# Therefore search ONLY between those two geometries.
#
# Hand/transmission parameters remain frozen at Candidate A.
# ============================================================


MODEL_PATH = (
    base.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_v1.xml"
)


# TOO CLOSE
CENTER_CLOSE = np.array(
    [
        0.592792826,
        0.896384282,
        1.562587753,
    ],
    dtype=float,
)


# TOO FAR
CENTER_FAR = np.array(
    [
        0.593611823,
        0.884245992,
        1.564105533,
    ],
    dtype=float,
)


# 0 = V1 / close
# 1 = V2 / far
ALPHAS = np.linspace(
    0.0,
    1.0,
    21,
)


# ============================================================
# Candidate A -- completely frozen
# ============================================================

SCALE = 5.1e-6
TENDON_STIFFNESS = 2100.0

OPEN_ENCODER = 100.0

CONTACT_STIFFNESS = 28000.0
CONTACT_DAMPING = 1.0
CONTACT_FRICTION = 1.2


# ============================================================
# Hardware reference
# ============================================================

HW_CONTACT_THRESHOLD_TICKS = 1020.0

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


def joint_qpos(
    model,
    name,
):

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
        )
        * SCALE,
    )

    data.qpos[
        qadr
    ] = spool

    data.qvel[
        dofadr
    ] = 0.0

    data.ctrl[
        actuator
    ] = spool


def contact_names(
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


def is_index_geom(
    name,
):

    return name.startswith(
        "collision_idx_"
    )


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python3 bracket_index_bottle_pose_v3.py "
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
    print("=" * 120)
    print(
        "INDEX BOTTLE POSITION BRACKET V3"
    )
    print("=" * 120)

    print()
    print(
        "Close centre =",
        CENTER_CLOSE,
    )

    print(
        "Far centre   =",
        CENTER_FAR,
    )


    bracket_length_mm = (
        np.linalg.norm(
            CENTER_FAR
            - CENTER_CLOSE
        )
        * 1000.0
    )


    print(
        f"Bracket length = "
        f"{bracket_length_mm:.3f} mm"
    )


    print()
    print(
        "alpha = 0.0 -> too-close V1"
    )

    print(
        "alpha = 1.0 -> too-far V2"
    )

    print()


    results = []


    for alpha in ALPHAS:

        center = (
            CENTER_CLOSE
            +
            alpha
            * (
                CENTER_FAR
                - CENTER_CLOSE
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


        # ----------------------------------------------------
        # Candidate A tendon
        # ----------------------------------------------------

        tendon_id = oid(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            TENDON_NAME,
        )


        model.tendon_stiffness[
            tendon_id
        ] = TENDON_STIFFNESS


        # ----------------------------------------------------
        # Bottle
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Joint / actuator IDs
        # ----------------------------------------------------

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


        mcp_q = joint_qpos(
            model,
            MCP_NAME,
        )

        pip_q = joint_qpos(
            model,
            PIP_NAME,
        )

        dip_q = joint_qpos(
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


        # ----------------------------------------------------
        # OPEN equilibrium
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
                spool_qadr,
                spool_dofadr,
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


        open_contacts = contact_names(
            model,
            data,
            fixture_gid,
        )


        # ----------------------------------------------------
        # Replay real encoder trajectory
        # ----------------------------------------------------

        first_any_encoder = None
        first_index_encoder = None
        first_tip_encoder = None
        first_other_encoder = None
        first_fts_010_encoder = None


        target_mcp = []
        target_pip = []
        target_dip = []
        target_fts = []


        target_names = []

        nearest_target_dt = float(
            "inf"
        )


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
                spool_qadr,
                spool_dofadr,
                actuator,
                encoder,
            )


            mujoco.mj_step(
                model,
                data,
            )


            names = contact_names(
                model,
                data,
                fixture_gid,
            )


            if (
                names
                and first_any_encoder is None
            ):

                first_any_encoder = (
                    encoder
                )


            index_names = [
                n
                for n in names
                if is_index_geom(
                    n
                )
            ]


            other_names = [
                n
                for n in names
                if not is_index_geom(
                    n
                )
            ]


            if (
                index_names
                and
                first_index_encoder is None
            ):

                first_index_encoder = (
                    encoder
                )


            if (
                "collision_idx_tip"
                in names
                and
                first_tip_encoder is None
            ):

                first_tip_encoder = (
                    encoder
                )


            if (
                other_names
                and
                first_other_encoder is None
            ):

                first_other_encoder = (
                    encoder
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


            if (
                fts >= 0.10
                and
                first_fts_010_encoder is None
            ):

                first_fts_010_encoder = (
                    encoder
                )


            if (
                abs(
                    bag_time
                    - TARGET_BAG_TIME
                )
                <= TARGET_HALF_WINDOW
            ):

                target_mcp.append(
                    math.degrees(
                        data.qpos[
                            mcp_q
                        ]
                    )
                    - open_mcp
                )

                target_pip.append(
                    math.degrees(
                        data.qpos[
                            pip_q
                        ]
                    )
                    - open_pip
                )

                target_dip.append(
                    math.degrees(
                        data.qpos[
                            dip_q
                        ]
                    )
                    - open_dip
                )

                target_fts.append(
                    fts
                )


            difference = abs(
                bag_time
                - TARGET_BAG_TIME
            )


            if (
                difference
                < nearest_target_dt
            ):

                nearest_target_dt = (
                    difference
                )

                target_names = (
                    names
                )


            elapsed += dt


        # ----------------------------------------------------
        # Target metrics
        # ----------------------------------------------------

        sim_mcp = float(
            np.median(
                target_mcp
            )
        )

        sim_pip = float(
            np.median(
                target_pip
            )
        )

        sim_dip = float(
            np.median(
                target_dip
            )
        )

        sim_fts = float(
            np.median(
                target_fts
            )
        )


        pose_rmse = math.sqrt(
            (
                (
                    sim_mcp
                    - HW_MCP
                )**2
                +
                (
                    sim_pip
                    - HW_PIP
                )**2
            )
            / 2.0
        )


        target_other = [
            n
            for n in target_names
            if not is_index_geom(
                n
            )
        ]


        target_index = [
            n
            for n in target_names
            if is_index_geom(
                n
            )
        ]


        tip_at_target = (
            "collision_idx_tip"
            in target_names
        )


        index_only_target = (
            bool(
                target_index
            )
            and
            not target_other
        )


        if first_tip_encoder is None:

            tip_onset_error = (
                float(
                    "inf"
                )
            )

        else:

            tip_onset_error = abs(
                first_tip_encoder
                - HW_CONTACT_THRESHOLD_TICKS
            )


        # Geometry ranking:
        #
        # 1. open must be clear
        # 2. avoid non-index contact
        # 3. preferably tip contact at target
        # 4. onset near hardware contact region
        # 5. loaded pose is secondary here
        score = 0.0


        if open_contacts:

            score += 1000.0


        if target_other:

            score += 500.0


        if not target_index:

            score += 300.0


        if not tip_at_target:

            score += 100.0


        if np.isfinite(
            tip_onset_error
        ):

            score += (
                0.10
                * tip_onset_error
            )

        else:

            score += 100.0


        score += (
            0.25
            * pose_rmse
        )


        result = {

            "alpha":
                float(
                    alpha
                ),

            "center":
                center.copy(),

            "open_contacts":
                open_contacts,

            "first_any_encoder":
                first_any_encoder,

            "first_index_encoder":
                first_index_encoder,

            "first_tip_encoder":
                first_tip_encoder,

            "first_other_encoder":
                first_other_encoder,

            "first_fts_010_encoder":
                first_fts_010_encoder,

            "target_contacts":
                target_names,

            "index_only_target":
                index_only_target,

            "tip_at_target":
                tip_at_target,

            "sim_mcp":
                sim_mcp,

            "sim_pip":
                sim_pip,

            "sim_dip":
                sim_dip,

            "sim_fts":
                sim_fts,

            "pose_rmse":
                pose_rmse,

            "score":
                score,
        }


        results.append(
            result
        )


        def fmt(
            x,
        ):

            if x is None:

                return "----"

            return f"{x:7.1f}"


        contact_text = (
            "|".join(
                target_names
            )
            if target_names
            else "NONE"
        )


        print(
            f"alpha={alpha:4.2f}  "
            f"tip_onset={fmt(first_tip_encoder)}  "
            f"other_onset={fmt(first_other_encoder)}  "
            f"target={contact_text:45s}  "
            f"pose=({sim_mcp:5.1f},"
            f"{sim_pip:5.1f},"
            f"{sim_dip:5.1f})  "
            f"FTS={sim_fts:7.4f}"
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
        "TOP BRACKET CANDIDATES"
    )
    print("=" * 120)


    for rank, r in enumerate(
        ranked[:10],
        start=1,
    ):

        print()
        print(
            f"#{rank}  alpha={r['alpha']:.2f}   "
            f"score={r['score']:.3f}"
        )

        print(
            "  center = "
            f"{r['center'][0]:.9f} "
            f"{r['center'][1]:.9f} "
            f"{r['center'][2]:.9f}"
        )

        print(
            f"  first any contact  = "
            f"{r['first_any_encoder']}"
        )

        print(
            f"  first index contact = "
            f"{r['first_index_encoder']}"
        )

        print(
            f"  first TIP contact   = "
            f"{r['first_tip_encoder']}"
        )

        print(
            f"  first other contact = "
            f"{r['first_other_encoder']}"
        )

        print(
            f"  first FTS >= 0.10 N = "
            f"{r['first_fts_010_encoder']}"
        )

        print(
            f"  target contacts     = "
            f"{r['target_contacts']}"
        )

        print(
            f"  loaded pose         = "
            f"MCP {r['sim_mcp']:.3f}, "
            f"PIP {r['sim_pip']:.3f}, "
            f"DIP {r['sim_dip']:.3f} deg"
        )

        print(
            f"  MCP/PIP pose RMSE   = "
            f"{r['pose_rmse']:.3f} deg"
        )

        print(
            f"  target FTS          = "
            f"{r['sim_fts']:.5f} N"
        )


    # ========================================================
    # Strict useful candidates
    # ========================================================

    useful = [
        r
        for r in results
        if (
            not r[
                "open_contacts"
            ]
            and
            r[
                "index_only_target"
            ]
            and
            r[
                "tip_at_target"
            ]
        )
    ]


    print()
    print("=" * 120)
    print(
        "INDEX-ONLY TIP-CONTACT CANDIDATES AT 1029-TICK HOLD"
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
                    abs(
                        (
                            r[
                                "first_tip_encoder"
                            ]
                            if r[
                                "first_tip_encoder"
                            ]
                            is not None
                            else 1e9
                        )
                        -
                        HW_CONTACT_THRESHOLD_TICKS
                    )
                ),
        ):

            print(
                f"alpha={r['alpha']:.2f}   "
                f"tip onset="
                f"{r['first_tip_encoder']}   "
                f"contacts="
                f"{r['target_contacts']}   "
                f"pose=("
                f"{r['sim_mcp']:.2f}, "
                f"{r['sim_pip']:.2f}, "
                f"{r['sim_dip']:.2f})   "
                f"FTS="
                f"{r['sim_fts']:.4f} N"
            )


    print()
    print("=" * 120)


if __name__ == "__main__":
    main()
