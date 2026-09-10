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

OUTPUT_CSV = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "index_v4_preload_grid.csv"
)


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


# Video-derived PHYSICAL joint-angle estimates.
# These are NOT ROS /joint_states.
HW = {
    ("closing", 100):  (0.00, 0.00),
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
    ("opening", 100):  (0.00, 0.00),
}


MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

ACTUATOR = "pos_index_tendon"

OPEN_TICKS = 100.0


# ============================================================
# V4 SEARCH SPACE
# ============================================================

# More scale may be required because preload adds opposing torque.
SCALES = np.array([
    5.4,
    5.8,
    6.2,
]) * 1e-6


# Search tightly around the successful V3 region.
MCP_FRICTIONS = [
    0.001,
    0.002,
    0.003,
]

PIP_FRICTIONS = [
    0.006,
    0.008,
    0.010,
]


# Negative springref means:
#
# spring equilibrium lies slightly beyond the physical open stop.
#
# Therefore even near q=0 the passive element still pulls the
# finger toward OPEN.
#
# This represents an EFFECTIVE return preload.
MCP_SPRINGREFS = [
     0.00,
    -0.10,
    -0.20,
    -0.30,
]

PIP_SPRINGREFS = [
     0.00,
    -0.10,
    -0.20,
    -0.30,
]


def joint_info(model, name):

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


def deg(x):
    return float(x) * 180.0 / math.pi


def rmse(x):
    return math.sqrt(
        np.mean(
            np.square(x)
        )
    )


def run(
    model,
    data,
    aid,

    mcp_q,
    pip_q,
    dip_q,

    mcp_dof,
    pip_dof,
    dip_dof,

    scale,

    mcp_friction,
    pip_friction,

    mcp_ref,
    pip_ref,
):

    # ========================================================
    # IDENTIFIED PARAMETERS
    # ========================================================

    model.dof_frictionloss[mcp_dof] = (
        mcp_friction
    )

    model.dof_frictionloss[pip_dof] = (
        pip_friction
    )

    model.dof_frictionloss[dip_dof] = 0.0


    # qpos_spring contains the spring equilibrium configuration.
    #
    # Negative values create effective return preload toward
    # the lower mechanical joint stop q=0.

    model.qpos_spring[mcp_q] = mcp_ref
    model.qpos_spring[pip_q] = pip_ref

    # DIP remains uncalibrated.
    model.qpos_spring[dip_q] = 0.0


    model.actuator_ctrlrange[aid, 0] = -0.022
    model.actuator_ctrlrange[aid, 1] = 0.0


    mujoco.mj_resetData(
        model,
        data,
    )

    data.ctrl[:] = 0.0

    model.opt.gravity[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


    # ========================================================
    # INITIAL OPEN SETTLING
    # ========================================================

    for _ in range(
        int(1.0 / model.opt.timestep)
    ):

        data.ctrl[aid] = 0.0

        mujoco.mj_step(
            model,
            data,
        )


    hold_steps = int(
        2.0 / model.opt.timestep
    )

    average_steps = int(
        0.20 / model.opt.timestep
    )


    mcp_errors = []
    pip_errors = []

    closing_errors = []
    opening_errors = []

    final_open_mcp = 0.0
    final_open_pip = 0.0


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
                hold_steps
                - average_steps
            ):

                samples.append([
                    data.qpos[mcp_q],
                    data.qpos[pip_q],
                ])


        q = np.mean(
            np.asarray(samples),
            axis=0,
        )

        sim_mcp = deg(q[0])
        sim_pip = deg(q[1])

        hw_mcp, hw_pip = HW[
            (phase, ticks)
        ]

        em = sim_mcp - hw_mcp
        ep = sim_pip - hw_pip

        mcp_errors.append(em)
        pip_errors.append(ep)

        if phase == "closing":
            closing_errors += [em, ep]
        else:
            opening_errors += [em, ep]

        if (
            phase == "opening"
            and ticks == 100
        ):
            final_open_mcp = sim_mcp
            final_open_pip = sim_pip


    combined_errors = (
        mcp_errors
        + pip_errors
    )


    # ========================================================
    # IDENTIFICATION SCORE
    #
    # Raw RMSE remains reported.
    #
    # But because returning fully open is an important real
    # hardware property, the final-open state is weighted
    # additionally during parameter selection.
    # ========================================================

    weighted_errors = list(
        combined_errors
    )

    # Add four EXTRA copies of final-open error.
    #
    # It already occurs once in the dataset, therefore its
    # effective total weight becomes 5.
    for _ in range(4):

        weighted_errors.append(
            final_open_mcp
        )

        weighted_errors.append(
            final_open_pip
        )


    return {

        "score_deg":
            rmse(weighted_errors),

        "combined_rmse_deg":
            rmse(combined_errors),

        "mcp_rmse_deg":
            rmse(mcp_errors),

        "pip_rmse_deg":
            rmse(pip_errors),

        "closing_rmse_deg":
            rmse(closing_errors),

        "opening_rmse_deg":
            rmse(opening_errors),

        "final_open_mcp_deg":
            final_open_mcp,

        "final_open_pip_deg":
            final_open_pip,

        "scale_um_per_tick":
            scale * 1e6,

        "mcp_friction_Nm":
            mcp_friction,

        "pip_friction_Nm":
            pip_friction,

        "mcp_springref_rad":
            mcp_ref,

        "pip_springref_rad":
            pip_ref,
    }


def main():

    print()
    print("=" * 78)
    print(
        "RH8D INDEX SYSTEM IDENTIFICATION — "
        "V4 PASSIVE RETURN PRELOAD"
    )
    print("=" * 78)

    print()
    print(
        "MuJoCo version:",
        mujoco.__version__,
    )

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)


    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR,
    )


    _, mcp_q, mcp_dof = joint_info(
        model,
        MCP,
    )

    _, pip_q, pip_dof = joint_info(
        model,
        PIP,
    )

    _, dip_q, dip_dof = joint_info(
        model,
        DIP,
    )


    candidates = list(
        itertools.product(

            SCALES,

            MCP_FRICTIONS,
            PIP_FRICTIONS,

            MCP_SPRINGREFS,
            PIP_SPRINGREFS,
        )
    )


    print()
    print(
        f"Testing {len(candidates)} "
        "parameter combinations..."
    )

    print()
    print(
        "Searching:"
    )

    print(
        "  encoder -> tendon scale"
    )

    print(
        "  MCP/PIP dry friction"
    )

    print(
        "  MCP/PIP passive-return preload"
    )

    print()
    print(
        "Final open configuration receives "
        "extra weight."
    )


    start = time.time()

    results = []


    for i, candidate in enumerate(
        candidates,
        start=1,
    ):

        (
            scale,
            mcp_friction,
            pip_friction,
            mcp_ref,
            pip_ref,
        ) = candidate


        result = run(

            model,
            data,
            aid,

            mcp_q,
            pip_q,
            dip_q,

            mcp_dof,
            pip_dof,
            dip_dof,

            scale,

            mcp_friction,
            pip_friction,

            mcp_ref,
            pip_ref,
        )

        results.append(
            result
        )


        if (
            i % 25 == 0
            or i == len(candidates)
        ):

            best = min(
                results,
                key=lambda r:
                    r["score_deg"],
            )

            print(
                f"[{i:3d}/{len(candidates)}] "
                f"best score="
                f"{best['score_deg']:.3f}°  "
                f"raw RMSE="
                f"{best['combined_rmse_deg']:.3f}°"
            )


    results.sort(
        key=lambda r:
            r["score_deg"]
    )


    print()
    print("=" * 78)
    print("TOP 15 V4 PARAMETER SETS")
    print("=" * 78)


    for rank, r in enumerate(
        results[:15],
        start=1,
    ):

        print()

        print(
            f"#{rank:2d} "
            f"SCORE={r['score_deg']:6.3f}°   "
            f"RMSE={r['combined_rmse_deg']:6.3f}°"
        )

        print(
            f"     MCP={r['mcp_rmse_deg']:6.3f}°   "
            f"PIP={r['pip_rmse_deg']:6.3f}°   "
            f"close={r['closing_rmse_deg']:6.3f}°   "
            f"open={r['opening_rmse_deg']:6.3f}°"
        )

        print(
            f"     scale="
            f"{r['scale_um_per_tick']:.3f} um/tick"
        )

        print(
            f"     friction: "
            f"MCP={r['mcp_friction_Nm']:.4f} Nm   "
            f"PIP={r['pip_friction_Nm']:.4f} Nm"
        )

        print(
            f"     springref: "
            f"MCP={r['mcp_springref_rad']:.3f} rad   "
            f"PIP={r['pip_springref_rad']:.3f} rad"
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
        writer.writerows(results)


    print()
    print("=" * 78)
    print(
        f"Finished in "
        f"{time.time() - start:.1f} s"
    )

    print()
    print("Saved:")
    print(OUTPUT_CSV)
    print("=" * 78)


if __name__ == "__main__":
    main()
