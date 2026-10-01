#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import mujoco
import numpy as np

import fit_index_contact_v1 as bagbase
import search_index_tabletop_xy_v1 as simbase


WS = simbase.WS

FREE_MODEL = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_sysid_v5.xml"
)

CONTACT_MODEL = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_tabletop_sysid_v1.xml"
)


# ============================================================
# Current Candidate-A parameters -- frozen
# ============================================================

SCALE = 5.1e-6
OPEN_ENCODER = 100.0

TENDON_STIFFNESS = 2100.0

PIP_NAME = "Index_Middle--Index_Proximal"


# ============================================================
# PIP PASSIVE PARAMETER GRID
#
# Includes zero-resistance diagnostic bound.
#
# Current values:
#   stiffness    = 0.031
#   frictionloss = 0.006
# ============================================================

PIP_STIFFNESSES = [
    0.0000,
    0.0030,
    0.0075,
    0.0150,
    0.0230,
    0.0310,
]

PIP_FRICTIONS = [
    0.0000,
    0.0005,
    0.0015,
    0.0030,
    0.0060,
]


# ============================================================
# Hardware free-space physical measurements
#
# DIP deliberately excluded from free-space fitting objective.
# MCP and PIP marker measurements are the stronger signals.
# ============================================================

FREE_TICKS = np.array(
    [
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
    ],
    dtype=float,
)


HW_FREE_MCP = np.array(
    [
         0.00,
        13.44,
        30.85,
        46.46,
        58.88,
        59.85,
        60.97,
        64.35,
        52.24,
        46.77,
        37.93,
        37.92,
        34.04,
        14.58,
         0.00,
    ],
    dtype=float,
)


HW_FREE_PIP = np.array(
    [
         0.00,
         5.55,
        10.52,
        15.50,
        26.67,
        41.75,
        62.23,
        75.27,
        69.47,
        52.50,
        49.78,
        31.35,
        12.98,
         7.93,
         0.00,
    ],
    dtype=float,
)


# ============================================================
# Tabletop/contact calibration configuration
# ============================================================

BOTTLE_POS = np.array(
    [
         0.503422950,
        -0.372517762,
         0.799000000,
    ],
    dtype=float,
)


CONTACT_MU = 0.05
CONTACT_STIFFNESS = 1000.0
CONTACT_DAMPING = 1.0

FORCE_THRESHOLD = 0.005


INDEX_CONTACT_GEOMS = [
    "collision_idx_prox",
    "collision_idx_mid",
    "collision_idx_tip",
]


VIDEO_HOLD_TIME = 50.50
VIDEO_BAG_OFFSET = 11.53

TARGET_BAG_TIME = (
    VIDEO_HOLD_TIME
    - VIDEO_BAG_OFFSET
)


HW_CONTACT_POSE = np.array(
    [
        14.5355,
        23.8110,
         9.3270,
    ],
    dtype=float,
)


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


def angle_deg(
    model,
    data,
    name,
):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return math.degrees(
        data.qpos[
            model.jnt_qposadr[jid]
        ]
    )


def set_pip_passive(
    model,
    stiffness,
    frictionloss,
):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        PIP_NAME,
    )

    dof = model.jnt_dofadr[
        jid
    ]

    model.jnt_stiffness[
        jid
    ] = stiffness

    model.dof_frictionloss[
        dof
    ] = frictionloss


def settle(
    model,
    data,
    seconds,
):

    n = max(
        1,
        int(
            round(
                seconds
                / model.opt.timestep
            )
        ),
    )

    for _ in range(n):

        mujoco.mj_step(
            model,
            data,
        )


# ============================================================
# FREE-SPACE TEST
# ============================================================

def run_free_space(
    pip_k,
    pip_f,
):

    model = mujoco.MjModel.from_xml_path(
        str(FREE_MODEL)
    )

    data = mujoco.MjData(
        model
    )


    # Match previous isolated free-space identification.
    model.opt.gravity[:] = 0.0


    tendon = oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        "idx_flex_spatial",
    )

    model.tendon_stiffness[
        tendon
    ] = TENDON_STIFFNESS


    set_pip_passive(
        model,
        pip_k,
        pip_f,
    )


    actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "pos_index_tendon",
    )


    spool_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "index_motor_spool",
    )

    spool_qadr = model.jnt_qposadr[
        spool_jid
    ]


    mujoco.mj_resetData(
        model,
        data,
    )


    data.qpos[
        spool_qadr
    ] = 0.0

    data.ctrl[
        actuator
    ] = 0.0

    data.qvel[:] = 0.0


    mujoco.mj_forward(
        model,
        data,
    )


    # Establish open equilibrium.
    settle(
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


    sim_mcp = []
    sim_pip = []


    # Sequential closing + opening trajectory.
    # Friction/hysteresis therefore remains path-dependent.
    for ticks in FREE_TICKS:

        displacement = max(
            0.0,
            (
                ticks
                - OPEN_ENCODER
            )
            * SCALE,
        )


        data.ctrl[
            actuator
        ] = displacement


        # Quasi-static plateau.
        settle(
            model,
            data,
            0.80,
        )


        sim_mcp.append(
            angle_deg(
                model,
                data,
                simbase.MCP_NAME,
            )
            - open_mcp
        )


        sim_pip.append(
            angle_deg(
                model,
                data,
                simbase.PIP_NAME,
            )
            - open_pip
        )


    sim_mcp = np.asarray(
        sim_mcp,
        dtype=float,
    )

    sim_pip = np.asarray(
        sim_pip,
        dtype=float,
    )


    mcp_rmse = math.sqrt(
        float(
            np.mean(
                (
                    sim_mcp
                    - HW_FREE_MCP
                ) ** 2
            )
        )
    )


    pip_rmse = math.sqrt(
        float(
            np.mean(
                (
                    sim_pip
                    - HW_FREE_PIP
                ) ** 2
            )
        )
    )


    combined = math.sqrt(
        (
            mcp_rmse**2
            +
            pip_rmse**2
        )
        / 2.0
    )


    return {
        "mcp_rmse":
            mcp_rmse,

        "pip_rmse":
            pip_rmse,

        "combined":
            combined,

        "final_open_mcp":
            sim_mcp[-1],

        "final_open_pip":
            sim_pip[-1],
    }


# ============================================================
# REAL CONTACT REPLAY
# ============================================================

def run_contact(
    pip_k,
    pip_f,
    encoder_time,
    encoder_position,
    approach_start,
    hold_end,
):

    model = mujoco.MjModel.from_xml_path(
        str(CONTACT_MODEL)
    )

    data = mujoco.MjData(
        model
    )


    tendon = oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        simbase.TENDON_NAME,
    )

    model.tendon_stiffness[
        tendon
    ] = TENDON_STIFFNESS


    set_pip_passive(
        model,
        pip_k,
        pip_f,
    )


    fixture = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        simbase.FIXTURE_NAME,
    )

    model.geom_pos[
        fixture
    ] = BOTTLE_POS


    contact_gids = [
        fixture
    ]


    for name in INDEX_CONTACT_GEOMS:

        contact_gids.append(
            oid(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
        )


    for gid in contact_gids:

        model.geom_friction[
            gid
        ] = [
            CONTACT_MU,
            0.001,
            0.0001,
        ]

        model.geom_solref[
            gid
        ] = [
            -CONTACT_STIFFNESS,
            -CONTACT_DAMPING,
        ]


    actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        simbase.INDEX_ACTUATOR,
    )


    simbase.configure_pose(
        model,
        data,
    )


    settle(
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


        displacement = max(
            0.0,
            (
                encoder
                - OPEN_ENCODER
            )
            * SCALE,
        )


        data.ctrl[
            actuator
        ] = displacement


        mujoco.mj_step(
            model,
            data,
        )


        forces = (
            simbase.bottle_contact_forces(
                model,
                data,
                fixture,
            )
        )


        rows.append(
            {
                "time":
                    t,

                "encoder":
                    encoder,

                "mcp":
                    angle_deg(
                        model,
                        data,
                        simbase.MCP_NAME,
                    )
                    - open_mcp,

                "pip":
                    angle_deg(
                        model,
                        data,
                        simbase.PIP_NAME,
                    )
                    - open_pip,

                "dip":
                    angle_deg(
                        model,
                        data,
                        simbase.DIP_NAME,
                    )
                    - open_dip,

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


    first_mid = next(
        (
            i
            for i, r in enumerate(rows)
            if r["fmid"]
            >= FORCE_THRESHOLD
        ),
        None,
    )


    first_tip = next(
        (
            i
            for i, r in enumerate(rows)
            if r["ftip"]
            >= FORCE_THRESHOLD
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


    pose = np.array(
        [
            target["mcp"],
            target["pip"],
            target["dip"],
        ],
        dtype=float,
    )


    error = (
        pose
        - HW_CONTACT_POSE
    )


    pose_rmse = math.sqrt(
        float(
            np.mean(
                error**2
            )
        )
    )


    middle_encoder = None
    tip_encoder = None
    middle_duty = 0.0

    dmcp = None
    dpip = None
    ddip = None


    if first_mid is not None:

        middle_encoder = rows[
            first_mid
        ][
            "encoder"
        ]


    if first_tip is not None:

        tip_encoder = rows[
            first_tip
        ][
            "encoder"
        ]


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


        dmcp = (
            b["mcp"]
            - a["mcp"]
        )

        dpip = (
            b["pip"]
            - a["pip"]
        )

        ddip = (
            b["dip"]
            - a["dip"]
        )


        segment = rows[
            first_mid:
            first_tip + 1
        ]


        middle_duty = float(
            np.mean(
                [
                    r["fmid"]
                    >= FORCE_THRESHOLD
                    for r in segment
                ]
            )
        )


    return {
        "pose":
            pose,

        "error":
            error,

        "rmse":
            pose_rmse,

        "middle_encoder":
            middle_encoder,

        "tip_encoder":
            tip_encoder,

        "middle_duty":
            middle_duty,

        "dmcp":
            dmcp,

        "dpip":
            dpip,

        "ddip":
            ddip,
    }


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            " python3 scan_index_pip_passive_v1.py "
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


    print()
    print("=" * 135)
    print(
        "COMBINED FREE-SPACE + CONTACT PIP PASSIVE SCAN V1"
    )
    print("=" * 135)

    print()
    print(
        f"Candidates = "
        f"{len(PIP_STIFFNESSES) * len(PIP_FRICTIONS)}"
    )

    print(
        "PIP moment arm remains frozen at 4 mm."
    )

    print(
        "MCP and DIP mechanics remain frozen."
    )

    print()


    results = []


    for pip_k in PIP_STIFFNESSES:

        for pip_f in PIP_FRICTIONS:


            free = run_free_space(
                pip_k,
                pip_f,
            )


            contact = run_contact(
                pip_k,
                pip_f,
                encoder_time,
                encoder_position,
                approach_start,
                hold_end,
            )


            r = {
                "k":
                    pip_k,

                "f":
                    pip_f,

                "free":
                    free,

                "contact":
                    contact,
            }


            results.append(
                r
            )


            p = contact[
                "pose"
            ]


            print(
                f"k={pip_k:7.4f}  "
                f"f={pip_f:7.4f}  "
                f"free={free['combined']:6.2f}°  "
                f"contact={contact['rmse']:6.2f}°  "
                f"hold=({p[0]:5.1f},"
                f"{p[1]:5.1f},"
                f"{p[2]:5.1f})"
            )


    # ========================================================
    # Baseline
    # ========================================================

    baseline = min(
        results,
        key=lambda r:
            abs(
                r["k"]
                - 0.031
            )
            +
            abs(
                r["f"]
                - 0.006
            ),
    )


    baseline_free = baseline[
        "free"
    ][
        "combined"
    ]


    print()
    print("=" * 135)
    print(
        "CURRENT BASELINE"
    )
    print("=" * 135)

    print()
    print(
        f"PIP stiffness    = "
        f"{baseline['k']:.4f}"
    )

    print(
        f"PIP frictionloss = "
        f"{baseline['f']:.4f}"
    )

    print(
        f"Free-space RMSE  = "
        f"{baseline_free:.3f} deg"
    )

    print(
        f"Contact RMSE     = "
        f"{baseline['contact']['rmse']:.3f} deg"
    )

    print(
        "Contact hold     = "
        f"{baseline['contact']['pose']}"
    )


    # ========================================================
    # Preserve free-space fit.
    #
    # Allow at most +1 degree deterioration relative to this
    # script's own baseline. This avoids dependence on minor
    # differences from earlier replay implementations.
    # ========================================================

    eligible = [
        r
        for r in results
        if (
            r[
                "free"
            ][
                "combined"
            ]
            <= baseline_free + 1.0
        )
    ]


    eligible = sorted(
        eligible,
        key=lambda r:
            r[
                "contact"
            ][
                "rmse"
            ],
    )


    print()
    print("=" * 135)
    print(
        "FREE-SPACE-PRESERVING CANDIDATES"
    )
    print(
        "(free-space RMSE <= baseline + 1.0 deg)"
    )
    print("=" * 135)


    for rank, r in enumerate(
        eligible[:12],
        start=1,
    ):

        c = r[
            "contact"
        ]

        p = c[
            "pose"
        ]

        e = c[
            "error"
        ]


        print()
        print(
            f"#{rank}  "
            f"k={r['k']:.4f}  "
            f"friction={r['f']:.4f}"
        )

        print(
            f"  free RMSE = "
            f"{r['free']['combined']:.3f} deg"
        )

        print(
            f"    MCP RMSE = "
            f"{r['free']['mcp_rmse']:.3f}"
        )

        print(
            f"    PIP RMSE = "
            f"{r['free']['pip_rmse']:.3f}"
        )

        print(
            f"  contact pose = "
            f"({p[0]:.3f}, "
            f"{p[1]:.3f}, "
            f"{p[2]:.3f})"
        )

        print(
            f"  contact error = "
            f"({e[0]:+.3f}, "
            f"{e[1]:+.3f}, "
            f"{e[2]:+.3f})"
        )

        print(
            f"  contact RMSE = "
            f"{c['rmse']:.3f} deg"
        )

        print(
            f"  middle encoder = "
            f"{c['middle_encoder']}"
        )

        print(
            f"  tip encoder = "
            f"{c['tip_encoder']}"
        )

        print(
            f"  middle duty = "
            f"{100*c['middle_duty']:.1f}%"
        )

        print(
            f"  redistribution "
            f"dMCP/dPIP/dDIP = "
            f"{c['dmcp']} / "
            f"{c['dpip']} / "
            f"{c['ddip']}"
        )


    # ========================================================
    # Zero-resistance diagnostic bound
    # ========================================================

    zero = next(
        r
        for r in results
        if (
            abs(
                r["k"]
            ) < 1e-12
            and
            abs(
                r["f"]
            ) < 1e-12
        )
    )


    print()
    print("=" * 135)
    print(
        "ZERO-PIP-RESISTANCE DIAGNOSTIC BOUND"
    )
    print("=" * 135)

    print()
    print(
        f"Free-space RMSE = "
        f"{zero['free']['combined']:.3f} deg"
    )

    print(
        f"Contact pose = "
        f"{zero['contact']['pose']}"
    )

    print(
        f"Contact RMSE = "
        f"{zero['contact']['rmse']:.3f} deg"
    )

    print(
        f"Contact PIP = "
        f"{zero['contact']['pose'][1]:.3f} deg"
    )


    # ========================================================
    # Interpretation
    # ========================================================

    best = eligible[0]


    print()
    print("=" * 135)
    print(
        "INTERPRETATION"
    )
    print("=" * 135)

    print()
    print(
        f"Best free-space-preserving PIP parameters:"
    )

    print(
        f"  stiffness    = "
        f"{best['k']:.4f}"
    )

    print(
        f"  frictionloss = "
        f"{best['f']:.4f}"
    )

    print(
        f"  free RMSE    = "
        f"{best['free']['combined']:.3f} deg"
    )

    print(
        f"  contact RMSE = "
        f"{best['contact']['rmse']:.3f} deg"
    )

    print(
        f"  contact PIP  = "
        f"{best['contact']['pose'][1]:.3f} deg"
    )


    zero_pip = zero[
        "contact"
    ][
        "pose"
    ][1]


    if zero_pip < 10.0:

        print()
        print(
            "Even removing essentially ALL modeled PIP "
            "passive resistance does not create enough "
            "PIP motion."
        )

        print(
            "This would strongly indicate that the fixed "
            "4-mm PIP transmission / tendon structure is "
            "the next thing to investigate."
        )

    elif (
        best[
            "contact"
        ][
            "pose"
        ][1]
        < 15.0
    ):

        print()
        print(
            "Lower passive resistance helps, but no "
            "free-space-preserving candidate produces "
            "hardware-like PIP motion."
        )

        print(
            "That indicates a free/contact tradeoff that "
            "a constant passive parameter set cannot solve."
        )

    else:

        print()
        print(
            "Passive PIP mechanics alone produce a large "
            "contact improvement without destroying the "
            "free-space fit."
        )

        print(
            "We should refine this passive region before "
            "changing the tendon transmission."
        )


    print()
    print("=" * 135)


if __name__ == "__main__":
    main()
