#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import mujoco
import numpy as np

import fit_index_contact_v1 as bagbase
import search_index_tabletop_xy_v1 as simbase


MODEL_PATH = (
    simbase.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_tabletop_sysid_v1.xml"
)


# ============================================================
# Frozen tabletop calibration geometry
# ============================================================

BOTTLE_POS = np.array(
    [
         0.503422950,
        -0.372517762,
         0.799000000,
    ],
    dtype=float,
)


# ============================================================
# Current Candidate-A transmission
# ============================================================

SCALE = 5.1e-6
OPEN_ENCODER = 100.0

TENDON_STIFFNESS = 2100.0


# ============================================================
# Diagnostic contact interface
# ============================================================

MU = 0.05

CONTACT_STIFFNESS = 1000.0
CONTACT_DAMPING = 1.0

FORCE_THRESHOLD = 0.005


INDEX_CONTACT_GEOMS = [
    "collision_idx_prox",
    "collision_idx_mid",
    "collision_idx_tip",
]


# ============================================================
# Hardware video / bag alignment
# ============================================================

VIDEO_HOLD_TIME = 50.50
VIDEO_BAG_OFFSET = 11.53

TARGET_BAG_TIME = (
    VIDEO_HOLD_TIME
    - VIDEO_BAG_OFFSET
)


HW_POSE = np.array(
    [
        14.5355,
        23.8110,
         9.3270,
    ],
    dtype=float,
)


def angle_deg(
    model,
    data,
    name,
):

    jid = simbase.oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    qadr = model.jnt_qposadr[
        jid
    ]

    return math.degrees(
        data.qpos[
            qadr
        ]
    )


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            " python3 replay_index_tabletop_hardware_contact_v1.py "
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
    ) = bagbase.read_bag(
        bag
    )


    transitions = bagbase.compress_states(
        states
    )


    (
        approach_start,
        regulation_start,
        hold_start,
        hold_end,
    ) = bagbase.find_intervals(
        transitions
    )


    encoder_time = (
        encoder_data[:, 0]
    )

    encoder_position = (
        bagbase.median_filter_5(
            encoder_data[:, 1]
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


    # ========================================================
    # Candidate-A tendon stiffness
    # ========================================================

    tendon_id = simbase.oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        simbase.TENDON_NAME,
    )


    model.tendon_stiffness[
        tendon_id
    ] = TENDON_STIFFNESS


    fixture_gid = simbase.oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        simbase.FIXTURE_NAME,
    )


    model.geom_pos[
        fixture_gid
    ] = BOTTLE_POS


    # ========================================================
    # Frozen diagnostic contact parameters
    # ========================================================

    contact_gids = [
        fixture_gid
    ]


    for name in INDEX_CONTACT_GEOMS:

        contact_gids.append(
            simbase.oid(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
        )


    for gid in contact_gids:

        model.geom_friction[
            gid
        ] = [
            MU,
            0.001,
            0.0001,
        ]

        model.geom_solref[
            gid
        ] = [
            -CONTACT_STIFFNESS,
            -CONTACT_DAMPING,
        ]


    actuator = simbase.oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        simbase.INDEX_ACTUATOR,
    )


    # ========================================================
    # Known tabletop arm pose
    # ========================================================

    simbase.configure_pose(
        model,
        data,
    )


    # Let the open hand reach equilibrium before replay.
    simbase.settle(
        model,
        data,
        1.0,
    )


    open_mcp = angle_deg(
        model,
        data,
        simbase.MCP_NAME,
    )

    open_pip = angle_deg(
        model,
        data,
        simbase.PIP_NAME,
    )

    open_dip = angle_deg(
        model,
        data,
        simbase.DIP_NAME,
    )


    open_contacts = (
        simbase.bottle_contact_forces(
            model,
            data,
            fixture_gid,
        )
    )


    print()
    print("=" * 125)
    print(
        "REAL HARDWARE ENCODER -> TABLETOP DIGITAL-TWIN REPLAY"
    )
    print("=" * 125)

    print()
    print(
        "Bottle =",
        BOTTLE_POS,
    )

    print(
        f"Tendon stiffness = "
        f"{TENDON_STIFFNESS:.1f} N/m"
    )

    print(
        f"Contact friction = "
        f"{MU:.3f}"
    )

    print(
        f"Contact stiffness = "
        f"{CONTACT_STIFFNESS:.1f} N/m"
    )

    print()
    print(
        f"Approach start    = "
        f"{approach_start:.6f}"
    )

    print(
        f"Regulation start  = "
        f"{regulation_start:.6f}"
    )

    print(
        f"Hold start        = "
        f"{hold_start:.6f}"
    )

    print(
        f"Hold end          = "
        f"{hold_end:.6f}"
    )

    print(
        f"Video target bag time = "
        f"{TARGET_BAG_TIME:.6f}"
    )

    print()
    print(
        "Open contacts =",
        open_contacts,
    )


    rows = []


    t = approach_start

    dt = model.opt.timestep


    while t <= hold_end:

        encoder = float(
            np.interp(
                t,
                encoder_time,
                encoder_position,
            )
        )


        spool = max(
            0.0,
            (
                encoder
                - OPEN_ENCODER
            )
            * SCALE,
        )


        data.ctrl[
            actuator
        ] = spool


        mujoco.mj_step(
            model,
            data,
        )


        forces = (
            simbase.bottle_contact_forces(
                model,
                data,
                fixture_gid,
            )
        )


        rows.append(
            {
                "time":
                    t,

                "encoder":
                    encoder,

                "mcp":
                    (
                        angle_deg(
                            model,
                            data,
                            simbase.MCP_NAME,
                        )
                        - open_mcp
                    ),

                "pip":
                    (
                        angle_deg(
                            model,
                            data,
                            simbase.PIP_NAME,
                        )
                        - open_pip
                    ),

                "dip":
                    (
                        angle_deg(
                            model,
                            data,
                            simbase.DIP_NAME,
                        )
                        - open_dip
                    ),

                "fprox":
                    forces.get(
                        "collision_idx_prox",
                        0.0,
                    ),

                "fmid":
                    forces.get(
                        "collision_idx_mid",
                        0.0,
                    ),

                "ftip":
                    forces.get(
                        "collision_idx_tip",
                        0.0,
                    ),
            }
        )


        t += dt


    # ========================================================
    # Contact events
    # ========================================================

    first_mid = next(
        (
            i
            for i, r in enumerate(rows)
            if r["fmid"] >= FORCE_THRESHOLD
        ),
        None,
    )


    first_tip = next(
        (
            i
            for i, r in enumerate(rows)
            if r["ftip"] >= FORCE_THRESHOLD
        ),
        None,
    )


    target_idx = int(
        np.argmin(
            [
                abs(
                    r["time"]
                    - TARGET_BAG_TIME
                )
                for r in rows
            ]
        )
    )


    target = rows[
        target_idx
    ]


    print()
    print("=" * 125)
    print(
        "CONTACT EVENTS"
    )
    print("=" * 125)


    if first_mid is None:

        print()
        print(
            "Middle contact: NONE"
        )

    else:

        r = rows[
            first_mid
        ]

        print()
        print(
            f"Middle contact:"
        )

        print(
            f"  bag time = "
            f"{r['time']:.6f}"
        )

        print(
            f"  encoder  = "
            f"{r['encoder']:.2f}"
        )

        print(
            f"  pose     = "
            f"({r['mcp']:.3f}, "
            f"{r['pip']:.3f}, "
            f"{r['dip']:.3f}) deg"
        )

        print(
            f"  Fmid     = "
            f"{r['fmid']:.5f} N"
        )


    if first_tip is None:

        print()
        print(
            "Tip contact: NONE"
        )

    else:

        r = rows[
            first_tip
        ]

        print()
        print(
            f"Tip contact:"
        )

        print(
            f"  bag time = "
            f"{r['time']:.6f}"
        )

        print(
            f"  encoder  = "
            f"{r['encoder']:.2f}"
        )

        print(
            f"  pose     = "
            f"({r['mcp']:.3f}, "
            f"{r['pip']:.3f}, "
            f"{r['dip']:.3f}) deg"
        )

        print(
            f"  Fmid     = "
            f"{r['fmid']:.5f} N"
        )

        print(
            f"  Ftip     = "
            f"{r['ftip']:.5f} N"
        )


    # ========================================================
    # Middle -> tip redistribution
    # ========================================================

    print()
    print("=" * 125)
    print(
        "MIDDLE -> TIP REDISTRIBUTION"
    )
    print("=" * 125)


    if (
        first_mid is not None
        and
        first_tip is not None
        and
        first_mid < first_tip
    ):

        a = rows[
            first_mid
        ]

        b = rows[
            first_tip
        ]


        print()
        print(
            f"Encoder interval = "
            f"{a['encoder']:.2f} -> "
            f"{b['encoder']:.2f}"
        )

        print(
            f"delta MCP = "
            f"{b['mcp']-a['mcp']:+.3f} deg"
        )

        print(
            f"delta PIP = "
            f"{b['pip']-a['pip']:+.3f} deg"
        )

        print(
            f"delta DIP = "
            f"{b['dip']-a['dip']:+.3f} deg"
        )


        segment = rows[
            first_mid:
            first_tip + 1
        ]


        duty = np.mean(
            [
                r["fmid"]
                >= FORCE_THRESHOLD
                for r in segment
            ]
        )


        print(
            f"Middle contact duty = "
            f"{100*duty:.1f}%"
        )

    else:

        print()
        print(
            "Valid middle-before-tip sequence "
            "was not observed."
        )


    # ========================================================
    # Hardware video-aligned hold pose
    # ========================================================

    sim_pose = np.array(
        [
            target["mcp"],
            target["pip"],
            target["dip"],
        ]
    )


    error = (
        sim_pose
        - HW_POSE
    )


    print()
    print("=" * 125)
    print(
        "VIDEO-ALIGNED HOLD COMPARISON"
    )
    print("=" * 125)

    print()
    print(
        f"Bag time = "
        f"{target['time']:.6f}"
    )

    print(
        f"Encoder  = "
        f"{target['encoder']:.2f} ticks"
    )

    print()
    print(
        f"SIM pose = "
        f"({sim_pose[0]:.3f}, "
        f"{sim_pose[1]:.3f}, "
        f"{sim_pose[2]:.3f}) deg"
    )

    print(
        f"HW pose  = "
        f"({HW_POSE[0]:.3f}, "
        f"{HW_POSE[1]:.3f}, "
        f"{HW_POSE[2]:.3f}) deg"
    )

    print(
        f"error    = "
        f"({error[0]:+.3f}, "
        f"{error[1]:+.3f}, "
        f"{error[2]:+.3f}) deg"
    )


    rmse = math.sqrt(
        float(
            np.mean(
                error**2
            )
        )
    )


    print(
        f"3-joint pose RMSE = "
        f"{rmse:.3f} deg"
    )

    print()
    print(
        f"SIM forces at target:"
    )

    print(
        f"  middle = "
        f"{target['fmid']:.5f} N"
    )

    print(
        f"  tip    = "
        f"{target['ftip']:.5f} N"
    )


    print()
    print("=" * 125)


if __name__ == "__main__":
    main()
