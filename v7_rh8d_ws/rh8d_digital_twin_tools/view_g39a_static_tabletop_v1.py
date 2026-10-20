#!/usr/bin/env python3

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer

import audit_index_tendon_transmission_v35g12 as g12

from rh8d_tabletop_stage_v1 import (
    apply_tabletop_stage,
    verify_tabletop_scene,
)


XML = (
    Path.home()
    /
    "rh8d_results"
    /
    "v35g39a_cad_multiguide_flexor"
    /
    "g39a_cad_multiguide_flexor.xml"
)


model = mujoco.MjModel.from_xml_path(
    str(XML)
)

data = mujoco.MjData(
    model
)

refs = g12.get_joint_refs(
    model
)


# ================================================================
# REAL TABLETOP EXPERIMENT POSE
# ================================================================

apply_tabletop_stage(
    model,
    data,
    reset_to_qpos0=True,
)


# Hardware first-middle-contact pose.
TARGET = {
    "mcp": 18.543,
    "pip":  6.161,
    "dip":  5.276,
}


for key, deg in TARGET.items():

    data.qpos[
        refs[
            key
        ][
            "qadr"
        ]
    ] = math.radians(
        deg
    )


data.qvel[:] = 0.0

mujoco.mj_forward(
    model,
    data,
)


verify_tabletop_scene(
    model,
    data,
)


print()
print("=" * 100)
print("G39A STATIC TABLETOP VIEW")
print("=" * 100)

print(
    "UR5e = frozen tabletop experiment pose"
)

print(
    "Bottle = present"
)

print(
    "Index MCP/PIP/DIP = "
    "18.543 / 6.161 / 5.276 deg"
)

print()
print(
    "IMPORTANT: mj_step() is NEVER called."
)

print(
    "The finger must remain fixed at this pose."
)

print(
    "Close the GUI when finished."
)


viewer = mujoco.viewer.launch_passive(
    model,
    data,
)


while viewer.is_running():

    # Restore exactly the requested geometry before every sync.
    # This is only a STATIC VIEWER, not a simulation.
    for key, deg in TARGET.items():

        data.qpos[
            refs[
                key
            ][
                "qadr"
            ]
        ] = math.radians(
            deg
        )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    viewer.sync()

    time.sleep(
        0.02
    )
