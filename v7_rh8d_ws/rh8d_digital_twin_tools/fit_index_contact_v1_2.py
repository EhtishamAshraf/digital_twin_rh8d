#!/usr/bin/env python3

from pathlib import Path

import fit_index_contact_v1 as base
import fit_index_contact_v1_1 as fit


# ============================================================
# V1.2
#
# Geometry and complete free-space finger model remain frozen.
#
# We only extend the contact-law search because V1.1 placed
# the optimum exactly at the previous stiffness upper bound.
# ============================================================

base.STIFFNESS_VALUES = [
    16000.0,
    24000.0,
    32000.0,
    48000.0,
    64000.0,
    96000.0,
]

base.DAMPING_VALUES = [
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
      "index_contact_v1_2_grid.csv"
)

fit.TRACE_CSV = (
    base.WS
    / "rh8d_digital_twin_tools/results/"
      "index_contact_v1_2_best_trace.csv"
)


if __name__ == "__main__":
    fit.main()
