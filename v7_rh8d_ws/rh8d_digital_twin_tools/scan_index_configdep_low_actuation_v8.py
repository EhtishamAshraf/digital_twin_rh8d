#!/usr/bin/env python3

from pathlib import Path

import scan_index_configdep_scale_stiffness_v7 as v7


# ============================================================
# V8 — LOWER-ACTUATION EXTENSION
#
# V7 showed that:
#
#   - free-space error worsened as stiffness increased
#   - free-space error also worsened toward higher scale
#
# Therefore V7 hit the lower boundary of the tested region.
#
# This experiment extends that region downward ONCE.
#
# All model structure remains frozen.
# ============================================================


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()


# ------------------------------------------------------------
# Extended scale range
# ------------------------------------------------------------

v7.SCALES_UM = [
    4.0,
    4.5,
    5.0,
    5.1,
    5.5,
    6.0,
    6.4,
]


# ------------------------------------------------------------
# Extended stiffness range
#
# Includes:
#
#   300  : very soft diagnostic lower bound
#   600
#   900
#   1200
#   1500
#   2100 : original V6 control
#
# This should bracket the low-stiffness optimum if one exists.
# ------------------------------------------------------------

v7.TENDON_K = [
    300.0,
    600.0,
    900.0,
    1200.0,
    1500.0,
    2100.0,
]


# ------------------------------------------------------------
# Separate outputs from V7
# ------------------------------------------------------------

v7.RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "configdep_low_actuation_v8"
)

v7.RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

v7.SUMMARY = (
    v7.RESULT_DIR
    / "summary.csv"
)

v7.BEST_MODEL = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_configdep_spool_v8_candidate.xml"
)


if __name__ == "__main__":
    v7.main()
