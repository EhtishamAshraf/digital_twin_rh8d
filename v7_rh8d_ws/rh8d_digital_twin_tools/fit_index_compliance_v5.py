#!/usr/bin/env python3

from pathlib import Path
import csv
import itertools
import math
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
      "ur5e_rh8d_index_sysid_v5.xml"
)

RESULT_DIR = WS / "rh8d_digital_twin_tools/results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

GRID_CSV = RESULT_DIR / "index_v5_compliance_grid.csv"
BEST_CSV = RESULT_DIR / "index_v5_best_trajectory.csv"


# ============================================================
# SYNCHRONIZED HARDWARE DATA
#
# encoder_ticks are /dynamic_joint_states values at the same
# synchronized video instants from which physical MCP/PIP
# angles were reconstructed.
# ============================================================

DATA = [

    # phase, label, measured encoder, MCP deg, PIP deg

    ("closing", 100,   100.0,  0.00,  0.00),
    ("closing", 500,   500.0, 13.44,  5.55),
    ("closing", 1000, 1000.0, 30.85, 10.52),
    ("closing", 1500, 1500.0, 46.46, 15.50),
    ("closing", 2000, 2000.0, 58.88, 26.67),
    ("closing", 2500, 2466.0, 59.85, 41.75),
    ("closing", 3000, 2961.0, 60.97, 62.23),
    ("closing", 3500, 3466.0, 64.35, 75.27),

    ("opening", 3000, 3000.0, 52.24, 69.47),
    ("opening", 2500, 2500.0, 46.77, 52.50),
    ("opening", 2000, 2000.0, 37.93, 49.78),
    ("opening", 1500, 1500.0, 37.92, 31.35),
    ("opening", 1000, 1000.0, 34.04, 12.98),
    ("opening", 500,   500.0, 14.58,  7.93),
    ("opening", 100,   100.0,  0.00,  0.00),
]


MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

SPOOL = "index_motor_spool"
TENDON = "idx_flex_spatial"
ACTUATOR = "pos_index_tendon"

OPEN_ENCODER = 100.0


# ============================================================
# V5 SEARCH SPACE
# ============================================================

# Effective encoder -> spool displacement.
SCALES = np.array([
    4.4,
    4.8,
    5.2,
    5.6,
    6.0,
    6.4,
]) * 1e-6


# Tendon deadband before tension develops.
SLACK_VALUES = np.array([
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
]) * 1e-3


# Effective tendon stiffness [N/m].
STIFFNESS_VALUES = [
    500.0,
    800.0,
    1200.0,
    1800.0,
    2800.0,
    4500.0,
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
        model.jnt_qposadr[jid],
        model.jnt_dofadr[jid],
    )


def deg(x):
    return math.degrees(float(x))


def rmse(values):

    return math.sqrt(
        np.mean(
            np.square(values)
        )
    )


def simulate_candidate(
    model,
    data,
    aid,
    tid,
    mcp_q,
    pip_q,
    dip_q,
    spool_q,
    spool_dof,
    scale,
    slack,
    stiffness,
    keep_trajectory=False,
):

    # --------------------------------------------------------
    # Candidate tendon properties
    # --------------------------------------------------------

    model.tendon_stiffness[tid] = stiffness

    # Effectively one-sided tensile tendon:
    #
    # large negative lower bound -> no compression force
    # positive upper value      -> physical slack/deadband
    model.tendon_lengthspring[tid, 0] = -0.050
    model.tendon_lengthspring[tid, 1] = slack

    model.opt.gravity[:] = 0.0

    mujoco.mj_resetData(
        model,
        data,
    )

    data.ctrl[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


    # ========================================================
    # IMPORTANT
    #
    # For this identification stage the spool position is
    # imposed from the REAL measured motor encoder.
    #
    # This prevents temporary simulated motor-servo compliance
    # from being mistaken for tendon compliance.
    # ========================================================

    def impose_spool(target):

        data.qpos[spool_q] = target
        data.qvel[spool_dof] = 0.0

        # Make the temporary actuator target agree with the
        # imposed spool coordinate so it does not fight us.
        data.ctrl[aid] = target


    # settle at real open motor position
    for _ in range(
        int(1.0 / model.opt.timestep)
    ):

        impose_spool(0.0)

        mujoco.mj_step(
            model,
            data,
        )


    HOLD_STEPS = int(
        2.0 / model.opt.timestep
    )

    AVG_STEPS = int(
        0.20 / model.opt.timestep
    )


    all_errors = []
    mcp_errors = []
    pip_errors = []

    closing_errors = []
    opening_errors = []

    trajectory = []

    final_open_mcp = 0.0
    final_open_pip = 0.0


    for (
        phase,
        label_ticks,
        encoder_ticks,
        hw_mcp,
        hw_pip,
    ) in DATA:

        spool_target = (
            encoder_ticks - OPEN_ENCODER
        ) * scale


        samples = []


        for step in range(HOLD_STEPS):

            impose_spool(
                spool_target
            )

            mujoco.mj_step(
                model,
                data,
            )


            if step >= (
                HOLD_STEPS - AVG_STEPS
            ):

                samples.append([
                    data.qpos[mcp_q],
                    data.qpos[pip_q],
                    data.qpos[dip_q],
                    data.ten_length[tid],
                ])


        mean = np.mean(
            np.asarray(samples),
            axis=0,
        )

        sim_mcp = deg(mean[0])
        sim_pip = deg(mean[1])
        sim_dip = deg(mean[2])

        tendon_length_mm = (
            mean[3] * 1000.0
        )


        em = sim_mcp - hw_mcp
        ep = sim_pip - hw_pip


        mcp_errors.append(em)
        pip_errors.append(ep)

        all_errors.extend([
            em,
            ep,
        ])


        if phase == "closing":

            closing_errors.extend([
                em,
                ep,
            ])

        else:

            opening_errors.extend([
                em,
                ep,
            ])


        if (
            phase == "opening"
            and label_ticks == 100
        ):

            final_open_mcp = sim_mcp
            final_open_pip = sim_pip


        if keep_trajectory:

            trajectory.append({

                "phase": phase,
                "label_ticks": label_ticks,

                "measured_encoder_ticks":
                    encoder_ticks,

                "spool_target_mm":
                    spool_target * 1000.0,

                "hardware_mcp_deg":
                    hw_mcp,

                "hardware_pip_deg":
                    hw_pip,

                "sim_mcp_deg":
                    sim_mcp,

                "sim_pip_deg":
                    sim_pip,

                "sim_dip_deg":
                    sim_dip,

                "mcp_error_deg":
                    em,

                "pip_error_deg":
                    ep,

                "tendon_coordinate_mm":
                    tendon_length_mm,
            })


    # ========================================================
    # SCORE
    #
    # Raw RMSE is always retained.
    #
    # Final reopening receives additional weight because a
    # model that fits intermediate points but remains stuck
    # closed is physically unacceptable.
    # ========================================================

    score_errors = list(
        all_errors
    )


    for _ in range(4):

        score_errors.extend([
            final_open_mcp,
            final_open_pip,
        ])


    result = {

        "score_deg":
            rmse(score_errors),

        "combined_rmse_deg":
            rmse(all_errors),

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

        "slack_mm":
            slack * 1000.0,

        "tendon_stiffness_N_per_m":
            stiffness,
    }


    return result, trajectory


def main():

    print()
    print("=" * 80)
    print(
        "RH8D INDEX SYSTEM IDENTIFICATION — "
        "V5 TENDON COMPLIANCE + SLACK"
    )
    print("=" * 80)

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
        ACTUATOR,
    )

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON,
    )


    mcp_q, _ = joint_info(
        model,
        MCP,
    )

    pip_q, _ = joint_info(
        model,
        PIP,
    )

    dip_q, _ = joint_info(
        model,
        DIP,
    )

    spool_q, spool_dof = joint_info(
        model,
        SPOOL,
    )


    candidates = list(
        itertools.product(
            SCALES,
            SLACK_VALUES,
            STIFFNESS_VALUES,
        )
    )


    print()
    print(
        f"Testing {len(candidates)} "
        "parameter combinations..."
    )

    print()
    print("Searching:")
    print("  measured encoder -> spool scale")
    print("  tendon slack/deadband")
    print("  tendon stiffness")

    print()
    print(
        "MCP/PIP friction and passive preload "
        "remain fixed from V4."
    )

    print(
        "DIP remains excluded from the score."
    )


    start = time.time()

    results = []


    for i, (
        scale,
        slack,
        stiffness,
    ) in enumerate(
        candidates,
        start=1,
    ):


        result, _ = simulate_candidate(

            model=model,
            data=data,

            aid=aid,
            tid=tid,

            mcp_q=mcp_q,
            pip_q=pip_q,
            dip_q=dip_q,

            spool_q=spool_q,
            spool_dof=spool_dof,

            scale=scale,
            slack=slack,
            stiffness=stiffness,
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
                f"{best['score_deg']:.3f}°   "
                f"RMSE="
                f"{best['combined_rmse_deg']:.3f}°"
            )


    results.sort(
        key=lambda r:
            r["score_deg"]
    )


    print()
    print("=" * 80)
    print("TOP 15 V5 PARAMETER SETS")
    print("=" * 80)


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
            f"PIP={r['pip_rmse_deg']:6.3f}°"
        )

        print(
            f"     close={r['closing_rmse_deg']:6.3f}°   "
            f"open={r['opening_rmse_deg']:6.3f}°"
        )

        print(
            f"     scale="
            f"{r['scale_um_per_tick']:.3f} um/tick"
        )

        print(
            f"     slack="
            f"{r['slack_mm']:.3f} mm"
        )

        print(
            f"     stiffness="
            f"{r['tendon_stiffness_N_per_m']:.1f} N/m"
        )

        print(
            f"     final open: "
            f"MCP={r['final_open_mcp_deg']:.2f}°   "
            f"PIP={r['final_open_pip_deg']:.2f}°"
        )


    # Save full grid.
    with open(
        GRID_CSV,
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


    # ========================================================
    # Re-run best candidate and save its complete trajectory.
    # ========================================================

    best = results[0]


    _, best_trajectory = simulate_candidate(

        model=model,
        data=data,

        aid=aid,
        tid=tid,

        mcp_q=mcp_q,
        pip_q=pip_q,
        dip_q=dip_q,

        spool_q=spool_q,
        spool_dof=spool_dof,

        scale=
            best["scale_um_per_tick"]
            * 1e-6,

        slack=
            best["slack_mm"]
            * 1e-3,

        stiffness=
            best["tendon_stiffness_N_per_m"],

        keep_trajectory=True,
    )


    with open(
        BEST_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                best_trajectory[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            best_trajectory
        )


    print()
    print("=" * 80)
    print("BEST V5 TRAJECTORY")
    print("=" * 80)


    for r in best_trajectory:

        print(
            f"{r['phase']:7s} "
            f"{int(r['label_ticks']):4d}  "
            f"enc={r['measured_encoder_ticks']:7.1f}  "
            f"HW=({r['hardware_mcp_deg']:6.2f},"
            f"{r['hardware_pip_deg']:6.2f})  "
            f"SIM=({r['sim_mcp_deg']:6.2f},"
            f"{r['sim_pip_deg']:6.2f})"
        )


    print()
    print("=" * 80)

    print(
        f"Finished in "
        f"{time.time() - start:.1f} s"
    )

    print()
    print("Saved:")
    print(GRID_CSV)
    print(BEST_CSV)

    print("=" * 80)


if __name__ == "__main__":
    main()
