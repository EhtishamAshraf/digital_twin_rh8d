#!/usr/bin/env python3

from pathlib import Path

import compare_v35_k9000_vs_v36a as base


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

base.V36A = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
    / "ur5e_rh8d_index_segmented_v36a2.xml"
)

base.OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v36a2_segmented_stabilized"
)


if __name__ == "__main__":
    base.main()
