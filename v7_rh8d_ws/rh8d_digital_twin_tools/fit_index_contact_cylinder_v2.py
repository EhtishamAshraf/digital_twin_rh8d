#!/usr/bin/env python3

import fit_index_contact_v1 as base
import fit_index_contact_v1_1 as fit


# ============================================================
# REAL-BOTTLE CONTACT IDENTIFICATION V2
#
# Keep the validated free-space plant frozen:
#
#   tendon stiffness = 1200 N/m
#   encoder scale     = 6 um/tick
#   passive mechanics unchanged
#
# Only identify the MuJoCo contact law using the measured
# 69.2-mm bottle cylinder.
# ============================================================


base.MODEL_PATH = (
    base.WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_v1.xml"
)


# Previous cylinder test with 32000 / 320 produced a huge
# impact spike, so deliberately search much softer values too.
base.STIFFNESS_VALUES = [
    25.0,
    50.0,
    100.0,
    200.0,
    400.0,
    800.0,
    1600.0,
    3200.0,
    6400.0,
    12800.0,
    24000.0,
    32000.0,
]


base.DAMPING_VALUES = [
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
    320.0,
]


fit.GRID_CSV = (
    base.WS
    / "rh8d_digital_twin_tools/results/"
      "index_contact_cylinder_v2_grid.csv"
)


fit.TRACE_CSV = (
    base.WS
    / "rh8d_digital_twin_tools/results/"
      "index_contact_cylinder_v2_best_trace.csv"
)


if __name__ == "__main__":
    fit.main()
