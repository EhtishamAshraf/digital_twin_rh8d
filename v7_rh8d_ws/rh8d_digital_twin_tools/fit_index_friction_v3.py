#!/usr/bin/env python3

from pathlib import Path
import csv
import math
import itertools
import time

import mujoco
import numpy as np


WS = Path.home() / (
    "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
)

MODEL_PATH = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_sysid_v2.xml"
)

RESULTS_DIR = WS / "rh8d_digital_twin_tools/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "index_v3_friction_grid.csv"


SEQUENCE = [
    ("closing", 100),
    ("closing", 500),
    ("closing", 1000),
    ("closing", 1500),
    ("closing", 2000),
    ("closing", 2500),
    ("closing", 3000),
    ("closing", 3500),

    ("opening", 3000),
    ("opening", 2500),
    ("opening", 2000),
    ("opening", 1500),
    ("opening", 1000),
    ("opening", 500),
    ("opening", 100),
]


# Video-derived physical hardware angles.
# NOT ROS /joint_states.
HARDWARE_REFERENCE = {
    ("closing", 100):  (0.00,  0.00),
    ("closing", 500):  (13.44, 5.55),
    ("closing", 1000): (30.85, 10.52),
    ("closing", 1500): (46.46, 15.50),
    ("closing", 2000): (58.88, 26.67),
    ("closing", 2500): (59.85, 41.75),
    ("closing", 3000): (60.97, 62.23),
    ("closing", 3500): (64.35, 75.27),

    ("opening", 3000): (52.24, 69.47),
    ("opening", 2500): (46.77, 52.50),
    ("opening", 2000): (37.93, 49.78),
    ("opening", 1500): (37.92, 31.35),
    ("opening", 1000): (34.04, 12.98),
    ("opening", 500):  (14.58, 7.93),
    ("opening", 100):  (0.00,  0.00),
}


MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

ACTUATOR_NAME = "pos_index_tendon"

OPEN_TICKS = 100.0


# ============================================================
# SEARCH SPACE
# ============================================================

# Effective motor encoder -> tendon displacement.
SCALE_VALUES = np.array([
    4.8,
    5.1,
    5.4,
    5.7,
    6.0,
]) * 1e-6


# Coulomb/dry joint friction torque [Nm].
#
# MCP intentionally searched over smaller values.
# PIP searched higher because hardware indicates that the PIP
# tends to remain flexed during tendon release.
MCP_FRICTION_VALUES = [
    0.000,
    0.002,
    0.004,
    0.006,
    0.008,
    0.012,
]

PIP_FRICTION_VALUES = [
    0.000,
    0.004,
    0.008,
    0.012,
    0.016,
    0.020,
    0.025,
]


def get_joint_info(model, name):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    return (
        jid,
        model.jnt_qposadr[jid],
        model.jnt_dofadr[jid],
    )


def deg(rad):
    return float(rad) * 180.0 / math.pi


def rmse(values):
    return math.sqrt(
        np.mean(
            np.square(values)
        )
    )


def run_candidate(
    model,
    data,
    aid,
    mcp_qadr,
    pip_qadr,
    dip_qadr,
    mcp_dof,
    pip_dof,
    dip_dof,
    scale,
    mcp_friction,
    pip_friction,
):

    # --------------------------------------------------------
    # Candidate physical parameters
    # --------------------------------------------------------

    model.dof_frictionloss[mcp_dof] = mcp_friction
    model.dof_frictionloss[pip_dof] = pip_friction

    # DIP friction deliberately left zero for now because
    # DIP is not yet reliable enough in our video measurement.
    model.dof_frictionloss[dip_dof] = 0.0

    # Permit the largest searched tendon displacement.
    model.actuator_ctrlrange[aid, 0] = -0.022
    model.actuator_ctrlrange[aid, 1] = 0.0

    mujoco.mj_resetData(
        model,
        data,
    )

    data.ctrl[:] = 0.0

    # Keep the isolated transmission test independent of arm pose.
    model.opt.gravity[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    # --------------------------------------------------------
    # Open-state settling
    # --------------------------------------------------------

    settle_steps = int(
        1.0 / model.opt.timestep
    )

    for _ in range(settle_steps):

        data.ctrl[aid] = 0.0

        mujoco.mj_step(
            model,
            data,
        )

    hold_steps = int(
        2.0 / model.opt.timestep
    )

    average_steps = max(
        1,
        int(
            0.20 / model.opt.timestep
        ),
    )

    mcp_errors = []
    pip_errors = []

    closing_errors = []
    opening_errors = []

    final_mcp = 0.0
    final_pip = 0.0

    for phase, ticks in SEQUENCE:

        ctrl = -(
            float(ticks) - OPEN_TICKS
        ) * scale

        samples = []

        for step in range(hold_steps):

            data.ctrl[aid] = ctrl

            mujoco.mj_step(
                model,
                data,
            )

            if step >= (
                hold_steps - average_steps
            ):

                samples.append([
                    data.qpos[mcp_qadr],
                    data.qpos[pip_qadr],
                    data.qpos[dip_qadr],
                ])

        q = np.mean(
            np.asarray(samples),
            axis=0,
        )

        sim_mcp = deg(q[0])
        sim_pip = deg(q[1])

        hw_mcp, hw_pip = (
            HARDWARE_REFERENCE[
                (phase, ticks)
            ]
        )

        e_mcp = sim_mcp - hw_mcp
        e_pip = sim_pip - hw_pip

        mcp_errors.append(e_mcp)
        pip_errors.append(e_pip)

        pair_errors = [
            e_mcp,
            e_pip,
        ]

        if phase == "closing":
            closing_errors.extend(
                pair_errors
            )
        else:
            opening_errors.extend(
                pair_errors
            )

        if (
            phase == "opening"
            and ticks == 100
        ):
            final_mcp = sim_mcp
            final_pip = sim_pip

    combined = (
        mcp_errors
        + pip_errors
    )

    return {
        "scale_um_per_tick": scale * 1e6,
        "mcp_friction_Nm": mcp_friction,
        "pip_friction_Nm": pip_friction,

        "mcp_rmse_deg": rmse(
            mcp_errors
        ),

        "pip_rmse_deg": rmse(
            pip_errors
        ),

        "combined_rmse_deg": rmse(
            combined
        ),

        "closing_rmse_deg": rmse(
            closing_errors
        ),

        "opening_rmse_deg": rmse(
            opening_errors
        ),

        "final_open_mcp_deg": final_mcp,
        "final_open_pip_deg": final_pip,
    }


def main():

    print()
    print("=" * 76)
    print("RH8D INDEX SYSTEM IDENTIFICATION — V3 FRICTION SEARCH")
    print("=" * 76)

    print()
    print("MuJoCo version:", mujoco.__version__)
    print("Model:")
    print(MODEL_PATH)

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(
        model
    )

    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAME,
    )

    if aid < 0:
        raise RuntimeError(
            "Index tendon actuator not found."
        )

    _, mcp_qadr, mcp_dof = get_joint_info(
        model,
        MCP_NAME,
    )

    _, pip_qadr, pip_dof = get_joint_info(
        model,
        PIP_NAME,
    )

    _, dip_qadr, dip_dof = get_joint_info(
        model,
        DIP_NAME,
    )

    candidates = list(
        itertools.product(
            SCALE_VALUES,
            MCP_FRICTION_VALUES,
            PIP_FRICTION_VALUES,
        )
    )

    print()
    print(
        f"Testing {len(candidates)} parameter combinations..."
    )

    print()
    print(
        "Parameters:"
    )
    print(
        "  encoder/tendon scale"
    )
    print(
        "  MCP dry friction"
    )
    print(
        "  PIP dry friction"
    )

    print()
    print(
        "DIP remains excluded from fitting."
    )

    start = time.time()

    results = []

    for i, (
        scale,
        mcp_friction,
        pip_friction,
    ) in enumerate(
        candidates,
        start=1,
    ):

        result = run_candidate(
            model=model,
            data=data,
            aid=aid,

            mcp_qadr=mcp_qadr,
            pip_qadr=pip_qadr,
            dip_qadr=dip_qadr,

            mcp_dof=mcp_dof,
            pip_dof=pip_dof,
            dip_dof=dip_dof,

            scale=scale,

            mcp_friction=mcp_friction,
            pip_friction=pip_friction,
        )

        results.append(
            result
        )

        if (
            i % 25 == 0
            or i == len(candidates)
        ):

            best_so_far = min(
                results,
                key=lambda x:
                    x[
                        "combined_rmse_deg"
                    ],
            )

            print(
                f"[{i:3d}/{len(candidates)}] "
                f"best = "
                f"{best_so_far['combined_rmse_deg']:.3f} deg"
            )

    results.sort(
        key=lambda x:
            x[
                "combined_rmse_deg"
            ]
    )

    elapsed = (
        time.time() - start
    )

    print()
    print("=" * 76)
    print("TOP 15 PARAMETER SETS")
    print("=" * 76)

    print()

    for rank, r in enumerate(
        results[:15],
        start=1,
    ):

        print(
            f"#{rank:2d}  "
            f"RMSE={r['combined_rmse_deg']:6.3f}°   "
            f"MCP={r['mcp_rmse_deg']:6.3f}°   "
            f"PIP={r['pip_rmse_deg']:6.3f}°   "
            f"close={r['closing_rmse_deg']:6.3f}°   "
            f"open={r['opening_rmse_deg']:6.3f}°"
        )

        print(
            f"     scale="
            f"{r['scale_um_per_tick']:.3f} um/tick   "
            f"MCP friction="
            f"{r['mcp_friction_Nm']:.4f} Nm   "
            f"PIP friction="
            f"{r['pip_friction_Nm']:.4f} Nm"
        )

        print(
            f"     final open: "
            f"MCP={r['final_open_mcp_deg']:.2f}°   "
            f"PIP={r['final_open_pip_deg']:.2f}°"
        )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print()
    print("=" * 76)
    print(
        f"Search completed in {elapsed:.1f} s"
    )
    print()
    print("Saved:")
    print(OUTPUT_CSV)
    print("=" * 76)


if __name__ == "__main__":
    main()
