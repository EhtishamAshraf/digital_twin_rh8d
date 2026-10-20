#!/usr/bin/env python3

import math
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
    "v35g38c_valid_joint_pulley"
    /
    "g38c_valid_joint_pulley_flexor.xml"
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
# REAL TABLETOP STAGING
# ================================================================

apply_tabletop_stage(
    model,
    data,
    reset_to_qpos0=True,
)


# ================================================================
# Put the index at the HARDWARE H1 first-middle-contact pose.
#
# This is much more useful visually than putting the finger at
# 77/49/8 degrees while the bottle experiment is our target.
# ================================================================

contact_pose = {
    "mcp": 18.543,
    "pip":  6.161,
    "dip":  5.276,
}


for key, value in contact_pose.items():

    data.qpos[
        refs[
            key
        ][
            "qadr"
        ]
    ] = math.radians(
        value
    )


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
print("G38C — CORRECT TABLETOP VIEW")
print("=" * 100)

print(
    "Index pose = hardware first-middle-contact pose:"
)

print(
    "  MCP/PIP/DIP = "
    "18.543 / 6.161 / 5.276 deg"
)

print()
print(
    "You should see:"
)

print(
    "  UR5e at the known tabletop experiment pose"
)

print(
    "  green bottle beside the index finger"
)

print(
    "  table in its normal position"
)

print(
    "  magenta spatial flexor inside the RH8D hand"
)

print()
print(
    "GUI stays open until you close it."
)


mujoco.viewer.launch(
    model,
    data,
)
