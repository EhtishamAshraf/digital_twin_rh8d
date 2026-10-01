#!/usr/bin/env python3

from pathlib import Path
import math

import mujoco
import numpy as np


WS = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_sysid_v5.xml"
)


# ============================================================
# Candidate A -- exactly the parameters we are currently using
# ============================================================

SCALE = 5.1e-6
OPEN_TICKS = 100.0

TENDON_STIFFNESS = 2100.0

PIP_STIFFNESS = 0.031
PIP_FRICTION = 0.006


TICKS = np.array(
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


HW_MCP = np.array(
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


HW_PIP = np.array(
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


# This is what we are testing.
DWELL_TIMES = [
    0.8,
    2.0,
    5.0,
]


MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"

SPOOL_NAME = "index_motor_spool"
ACTUATOR_NAME = "pos_index_tendon"

TENDON_NAME = "idx_flex_spatial"


def oid(model, typ, name):

    idx = mujoco.mj_name2id(
        model,
        typ,
        name,
    )

    if idx < 0:
        raise RuntimeError(
            f"Missing object: {name}"
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


def run_one(
    dwell,
):

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(
        model
    )


    # Same isolated free-space assumption as our system-ID work.
    model.opt.gravity[:] = 0.0


    # --------------------------------------------------------
    # Candidate-A tendon
    # --------------------------------------------------------

    tid = oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    model.tendon_stiffness[
        tid
    ] = TENDON_STIFFNESS


    # --------------------------------------------------------
    # Explicitly force current baseline PIP passive mechanics
    # --------------------------------------------------------

    pip_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        PIP_NAME,
    )

    pip_dof = model.jnt_dofadr[
        pip_jid
    ]

    model.jnt_stiffness[
        pip_jid
    ] = PIP_STIFFNESS

    model.dof_frictionloss[
        pip_dof
    ] = PIP_FRICTION


    actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAME,
    )


    spool_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_NAME,
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


    # Give the open configuration plenty of time to settle.
    settle(
        model,
        data,
        3.0,
    )


    open_mcp = angle_deg(
        model,
        data,
        MCP_NAME,
    )

    open_pip = angle_deg(
        model,
        data,
        PIP_NAME,
    )


    sim_mcp = []
    sim_pip = []


    for ticks in TICKS:

        displacement = max(
            0.0,
            (
                ticks
                - OPEN_TICKS
            )
            * SCALE,
        )


        data.ctrl[
            actuator
        ] = displacement


        settle(
            model,
            data,
            dwell,
        )


        sim_mcp.append(
            angle_deg(
                model,
                data,
                MCP_NAME,
            )
            - open_mcp
        )

        sim_pip.append(
            angle_deg(
                model,
                data,
                PIP_NAME,
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
                    - HW_MCP
                ) ** 2
            )
        )
    )


    pip_rmse = math.sqrt(
        float(
            np.mean(
                (
                    sim_pip
                    - HW_PIP
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
        "dwell":
            dwell,

        "mcp":
            sim_mcp,

        "pip":
            sim_pip,

        "mcp_rmse":
            mcp_rmse,

        "pip_rmse":
            pip_rmse,

        "combined":
            combined,
    }


def main():

    print()
    print("=" * 120)
    print(
        "CANDIDATE-A FREE-SPACE BASELINE CONSISTENCY AUDIT"
    )
    print("=" * 120)

    print()
    print(
        f"Scale             = "
        f"{SCALE*1e6:.3f} um/tick"
    )

    print(
        f"Tendon stiffness  = "
        f"{TENDON_STIFFNESS:.1f} N/m"
    )

    print(
        f"PIP stiffness     = "
        f"{PIP_STIFFNESS:.5f} Nm/rad"
    )

    print(
        f"PIP frictionloss  = "
        f"{PIP_FRICTION:.5f} Nm"
    )

    print()


    results = []


    for dwell in DWELL_TIMES:

        r = run_one(
            dwell
        )

        results.append(
            r
        )


        print(
            f"dwell={dwell:4.1f} s  "
            f"MCP RMSE={r['mcp_rmse']:7.3f} deg  "
            f"PIP RMSE={r['pip_rmse']:7.3f} deg  "
            f"combined={r['combined']:7.3f} deg  "
            f"final open=({r['mcp'][-1]:.3f}, "
            f"{r['pip'][-1]:.3f})"
        )


    # ========================================================
    # Print full trajectory for the hardware-like 2 s dwell.
    # ========================================================

    r2 = next(
        r
        for r in results
        if abs(
            r["dwell"]
            - 2.0
        ) < 1e-9
    )


    print()
    print("=" * 120)
    print(
        "2.0 SECOND DWELL TRAJECTORY"
    )
    print("=" * 120)

    print()
    print(
        " step | ticks | HW MCP | SIM MCP | error  "
        "| HW PIP | SIM PIP | error"
    )

    print(
        "-" * 120
    )


    for i in range(
        len(TICKS)
    ):

        em = (
            r2["mcp"][i]
            - HW_MCP[i]
        )

        ep = (
            r2["pip"][i]
            - HW_PIP[i]
        )


        print(
            f"{i:5d} | "
            f"{TICKS[i]:5.0f} | "
            f"{HW_MCP[i]:6.2f} | "
            f"{r2['mcp'][i]:7.2f} | "
            f"{em:+6.2f} | "
            f"{HW_PIP[i]:6.2f} | "
            f"{r2['pip'][i]:7.2f} | "
            f"{ep:+6.2f}"
        )


    print()
    print("=" * 120)
    print(
        "INTERPRETATION"
    )
    print("=" * 120)

    best = min(
        results,
        key=lambda r:
            r["combined"],
    )


    print()
    print(
        f"Best dwell tested = "
        f"{best['dwell']:.1f} s"
    )

    print(
        f"Best combined RMSE = "
        f"{best['combined']:.3f} deg"
    )


    if best[
        "combined"
    ] <= 7.0:

        print()
        print(
            "The earlier poor 12.5-deg baseline was mainly "
            "a replay/settling inconsistency."
        )

        print(
            "Use the hardware-consistent dwell/replay method "
            "before judging PIP parameter candidates."
        )

    else:

        print()
        print(
            "The baseline still does NOT reproduce the "
            "previous ~5-6 deg Candidate-A free-space result."
        )

        print(
            "Do not continue parameter identification yet."
        )

        print(
            "The next step should compare this evaluator "
            "directly against the earlier validated "
            "Candidate-A replay script."
        )


    print()
    print("=" * 120)


if __name__ == "__main__":
    main()
