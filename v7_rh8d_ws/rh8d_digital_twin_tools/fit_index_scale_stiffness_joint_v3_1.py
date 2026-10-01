#!/usr/bin/env python3

import fit_index_scale_stiffness_joint_v3 as fit


# ============================================================
# V3.1 LOCAL REFINEMENT
#
# V3 found a very promising basin around:
#
#   scale       = 5.2 um/tick
#   tendon k    = 2000 N/m
#   contact     = 24000 / 2.5
#
# Candidate failed our acceptance test only because
# hold std = 0.0309 N versus the 0.0300-N limit.
#
# Refine locally rather than changing model structure.
# ============================================================


fit.SCALE_UM_PER_TICK = [
    5.10,
    5.20,
    5.30,
]


fit.TENDON_STIFFNESS = [
    1900.0,
    2000.0,
    2100.0,
]


CONTACT_STIFFNESSES = [
    20000.0,
    22000.0,
    24000.0,
    26000.0,
    28000.0,
]


CONTACT_DAMPINGS = [
    1.0,
    2.5,
    5.0,
    10.0,
]


fit.CONTACT_LAWS = [
    (k, d)
    for k in CONTACT_STIFFNESSES
    for d in CONTACT_DAMPINGS
]


# Friction audit showed saturation above approximately 1.2.
fit.CONTACT_FRICTION = 1.2


fit.OUT = (
    fit.base.WS
    / "rh8d_digital_twin_tools/results/"
      "index_scale_stiffness_joint_v3_1.csv"
)


if __name__ == "__main__":
    fit.main()
