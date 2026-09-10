#!/usr/bin/env python3

from pathlib import Path
import csv
import math

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

OUTPUT_CSV = RESULTS_DIR / "index_tendon_v2_sim.csv"


# ============================================================
# HARDWARE EXPERIMENT
# ============================================================

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


# Initial video-derived hardware estimates.
#
# IMPORTANT:
# These are NOT /joint_states.
#
# They were reconstructed from your colored-marker video.
# MCP/PIP are suitable for our first fit.
# DIP is intentionally not scored yet because its marker estimate
# is less reliable with the current video resolution.

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


JOINTS = {
    "mcp": "Index_Proximal--palm_assembly:1",
    "pip": "Index_Middle--Index_Proximal",
    "dip": "Index_Distal--Index_Middle",
}

TENDON_NAME = "idx_flex_spatial"
ACTUATOR_NAME = "pos_index_tendon"


# ============================================================
# CURRENT v1 MOTOR-TICK -> TENDON COMMAND MODEL
# ============================================================

OPEN_TICKS = 100.0

# First system-ID estimate:
#
# 3 micrometres of effective tendon displacement per encoder tick.
#
# This is an IDENTIFICATION PARAMETER.
# It is not claimed to be the real spool geometry.

METERS_PER_TICK = 5.406e-6


def ticks_to_tendon_command(ticks):
    """
    Increasing encoder ticks = closing.

    The fixed tendon was defined with negative joint coefficients,
    therefore flexion corresponds to a negative tendon coordinate.
    """

    displacement = (
        float(ticks) - OPEN_TICKS
    ) * METERS_PER_TICK

    return -displacement


def obj_id(model, obj_type, name):
    idx = mujoco.mj_name2id(
        model,
        obj_type,
        name,
    )

    if idx < 0:
        raise RuntimeError(
            f"MuJoCo object not found: {name}"
        )

    return idx


def rad2deg(x):
    return float(x) * 180.0 / math.pi


def main():

    print()
    print("=" * 76)
    print("RH8D INDEX TENDON DIGITAL-TWIN TEST v2")
    print("=" * 76)

    print()
    print("Loading:")
    print(MODEL_PATH)

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    # ---------------------------------------------------------
    # This is a FREE-SPACE transmission experiment.
    #
    # Gravity is disabled only for this isolated identification
    # test so arm/hand pose does not drift while we measure the
    # finger transmission.
    #
    # We are NOT claiming gravity is absent in the final twin.
    # ---------------------------------------------------------

    model.opt.gravity[:] = 0.0

    print()
    print(f"MuJoCo timestep = {model.opt.timestep:.6f} s")

    actuator_id = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAME,
    )

    tendon_id = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    qadr = {}

    for key, joint_name in JOINTS.items():

        jid = obj_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qadr[key] = model.jnt_qposadr[jid]

    mujoco.mj_resetData(model, data)

    # Start all controls at zero.
    data.ctrl[:] = 0.0

    mujoco.mj_forward(model, data)

    # ---------------------------------------------------------
    # SETTLE OPEN CONFIGURATION
    # ---------------------------------------------------------

    settle_time = 1.0

    settle_steps = int(
        settle_time / model.opt.timestep
    )

    for _ in range(settle_steps):

        data.ctrl[actuator_id] = 0.0

        mujoco.mj_step(
            model,
            data,
        )

    print()
    print("Open configuration after settling:")
    print(
        f"  MCP = {rad2deg(data.qpos[qadr['mcp']]):8.3f} deg"
    )
    print(
        f"  PIP = {rad2deg(data.qpos[qadr['pip']]):8.3f} deg"
    )
    print(
        f"  DIP = {rad2deg(data.qpos[qadr['dip']]):8.3f} deg"
    )

    print()
    print("=" * 76)
    print("STARTING HARDWARE TRAJECTORY")
    print("=" * 76)

    rows = []

    # Hardware experiment held each command for ~2 seconds.
    HOLD_TIME = 2.0

    steps_per_hold = int(
        HOLD_TIME / model.opt.timestep
    )

    # Average final 0.20 seconds rather than using one sample.
    averaging_time = 0.20

    averaging_steps = max(
        1,
        int(
            averaging_time
            / model.opt.timestep
        ),
    )

    mcp_errors = []
    pip_errors = []

    for phase, ticks in SEQUENCE:

        ctrl = ticks_to_tendon_command(
            ticks
        )

        samples = []

        for step in range(steps_per_hold):

            data.ctrl[actuator_id] = ctrl

            mujoco.mj_step(
                model,
                data,
            )

            if step >= (
                steps_per_hold
                - averaging_steps
            ):

                samples.append([
                    data.qpos[qadr["mcp"]],
                    data.qpos[qadr["pip"]],
                    data.qpos[qadr["dip"]],
                    data.ten_length[tendon_id],
                    data.actuator_force[actuator_id],
                ])

        samples = np.asarray(
            samples,
            dtype=float,
        )

        mean_values = np.mean(
            samples,
            axis=0,
        )

        mcp_deg = rad2deg(
            mean_values[0]
        )

        pip_deg = rad2deg(
            mean_values[1]
        )

        dip_deg = rad2deg(
            mean_values[2]
        )

        tendon_mm = (
            mean_values[3] * 1000.0
        )

        actuator_force = (
            mean_values[4]
        )

        hw_mcp, hw_pip = (
            HARDWARE_REFERENCE[
                (phase, ticks)
            ]
        )

        mcp_error = (
            mcp_deg - hw_mcp
        )

        pip_error = (
            pip_deg - hw_pip
        )

        mcp_errors.append(
            mcp_error
        )

        pip_errors.append(
            pip_error
        )

        rows.append({
            "phase": phase,
            "ticks": ticks,
            "tendon_command_m": ctrl,
            "sim_mcp_deg": mcp_deg,
            "sim_pip_deg": pip_deg,
            "sim_dip_deg": dip_deg,
            "hardware_mcp_deg": hw_mcp,
            "hardware_pip_deg": hw_pip,
            "mcp_error_deg": mcp_error,
            "pip_error_deg": pip_error,
            "tendon_length_mm": tendon_mm,
            "actuator_force_N": actuator_force,
        })

        print()
        print(
            f"{phase.upper():7s} "
            f"{ticks:4d} ticks"
        )

        print(
            f"  tendon ctrl = "
            f"{ctrl * 1000.0:8.3f} mm"
        )

        print(
            "  SIM: "
            f"MCP={mcp_deg:7.2f}  "
            f"PIP={pip_deg:7.2f}  "
            f"DIP={dip_deg:7.2f}"
        )

        print(
            "   HW: "
            f"MCP={hw_mcp:7.2f}  "
            f"PIP={hw_pip:7.2f}"
        )

        print(
            "ERROR: "
            f"MCP={mcp_error:+7.2f}  "
            f"PIP={pip_error:+7.2f}"
        )

        print(
            f"  tendon length = "
            f"{tendon_mm:8.3f} mm"
        )

        print(
            f"  actuator force = "
            f"{actuator_force:8.3f} N"
        )

    # ---------------------------------------------------------
    # ERROR SUMMARY
    # ---------------------------------------------------------

    mcp_rmse = math.sqrt(
        np.mean(
            np.square(mcp_errors)
        )
    )

    pip_rmse = math.sqrt(
        np.mean(
            np.square(pip_errors)
        )
    )

    combined_rmse = math.sqrt(
        np.mean(
            np.square(
                mcp_errors
                + pip_errors
            )
        )
    )

    print()
    print("=" * 76)
    print("V2 HARDWARE-vs-SIM ERROR")
    print("=" * 76)

    print()
    print(
        f"MCP RMSE      = {mcp_rmse:.3f} deg"
    )

    print(
        f"PIP RMSE      = {pip_rmse:.3f} deg"
    )

    print(
        f"Combined RMSE = {combined_rmse:.3f} deg"
    )

    print()
    print(
        "DIP is printed but NOT included "
        "in the fit score yet."
    )

    # ---------------------------------------------------------
    # SAVE CSV
    # ---------------------------------------------------------

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Saved:\n{OUTPUT_CSV}"
    )

    print()
    print("=" * 76)
    print("TEST FINISHED")
    print("=" * 76)


if __name__ == "__main__":
    main()
