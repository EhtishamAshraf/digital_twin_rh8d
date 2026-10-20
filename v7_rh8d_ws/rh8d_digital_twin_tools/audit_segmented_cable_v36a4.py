#!/usr/bin/env python3

from pathlib import Path

import audit_segmented_cable_v36a3 as audit


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()


audit.XML = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
    / "ur5e_rh8d_index_segmented_v36a4.xml"
)

audit.OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v36a4_rigid_equalities"
)

audit.OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

audit.OUTCSV = (
    audit.OUTDIR
    / "v36a4_internal_trace.csv"
)


if __name__ == "__main__":
    audit.main()
