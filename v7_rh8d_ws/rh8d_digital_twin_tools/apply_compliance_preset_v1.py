#!/usr/bin/env python3

from pathlib import Path
import shutil
import re

WS = Path.home() / "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/digital_twin/v7_rh8d_ws"
MJCF = WS / "src/ur5e_mujoco_description/mjcf/ur5e_rh8d.xml"

BACKUP_DIR = WS / "rh8d_digital_twin_backups_step6"
BACKUP_DIR.mkdir(exist_ok=True)

backup = BACKUP_DIR / "ur5e_rh8d.xml.before_compliance_v1.bak"
shutil.copy2(MJCF, backup)

text = MJCF.read_text()

# ---------------------------------------------------------------------
# 1. Make direct finger actuators compliant.
# Previous tracking preset was strong:
#   kp=80, kv=6, forcerange=-50 50
#
# For contact wrapping, this is too position-stiff.
# This preset keeps free-space motion usable but lets joints yield in contact.
# ---------------------------------------------------------------------

text = text.replace(
    'kp="80" kv="6" forcelimited="true" forcerange="-50 50"',
    'kp="35" kv="3" forcelimited="true" forcerange="-12 12"',
)

# Also handle if some actuator lines still have older low-gain values.
text = text.replace(
    'kp="5" kv="1" forcelimited="true" forcerange="-5 5"',
    'kp="35" kv="3" forcelimited="true" forcerange="-12 12"',
)

# ---------------------------------------------------------------------
# 2. Reduce passive finger joint stiffness.
# We only touch RH8D finger joints, not UR5e arm joints.
# ---------------------------------------------------------------------

finger_joints = [
    "Thumb_Methacarpal--Thumb_axis",
    "Thumb_Proximal--Thumb_Methacarpal",
    "Thumb_Distal--Thumb_Proximal",

    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",

    "Middle_Proximal--palm_assembly:1",
    "Middle_Middle--Middle_Proximal",
    "Middle_Distal--Middle_Middle",

    "Ring_Proximal--palm_assembly:1",
    "Ring_Middle--Ring_Proximal",
    "Ring_Distal--Ring_Middle",

    "Small_Proximal--palm_assembly:1",
    "Small_Middle--Small_Proximal",
    "Small_Distal--Small_Middle",
]

for joint in finger_joints:
    pattern = re.compile(
        rf'(<joint name="{re.escape(joint)}"[^>]*?)\s+damping="[^"]+"\s+stiffness="[^"]+"([^>]*/>)'
    )

    text, count = pattern.subn(
        rf'\1 damping="0.08" stiffness="0.02"\2',
        text,
        count=1,
    )

    if count == 0:
        print(f"WARNING: did not find damping/stiffness for joint: {joint}")

MJCF.write_text(text)

print(f"Backup saved: {backup}")
print(f"Compliance preset v1 applied to: {MJCF}")
print("")
print("Changed staged actuators to: kp=35 kv=3 forcerange=-12 12")
print("Changed finger passive joints to: damping=0.08 stiffness=0.02")
