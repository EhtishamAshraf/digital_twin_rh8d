#!/usr/bin/env python3

from pathlib import Path

import test_prescribed_spool_v35p3 as p3


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()


# ============================================================
# P4:
#
# Correct boundary condition is frozen:
#
# measured hardware encoder
#       ↓
# prescribed spool coordinate
#       ↓
# tendon
#       ↓
# dynamic MCP/PIP/DIP + contact
#
# VARIABLE ONLY:
# tendon stiffness.
#
# This is a causal diagnostic, NOT final identification.
# ============================================================

K_VALUES = [
    2100.0,
    2800.0,
    3500.0,
    4200.0,
    5000.0,
    6000.0,
    7000.0,
    8000.0,
    9000.0,
]


p3.CASES = [
    (
        f"PRESCRIBED_K{int(K)}",
        K,
        True,
    )
    for K in K_VALUES
]


p3.OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v35p4_prescribed_spool_tendon_k"
)

p3.OUTCSV = (
    p3.OUTDIR
    / "v35p4_prescribed_spool_tendon_k_summary.csv"
)


if __name__ == "__main__":
    p3.main()
