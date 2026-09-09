#!/usr/bin/env python3

from pathlib import Path
import shutil
import re

WS = Path.home() / "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/digital_twin/v7_rh8d_ws"
MJCF_DIR = WS / "src/ur5e_mujoco_description/mjcf"
BACKUP_DIR = WS / "rh8d_digital_twin_backups_step6"
BACKUP_DIR.mkdir(exist_ok=True)

changed = False

for path in MJCF_DIR.glob("*.xml"):
    text = path.read_text()
    old = text

    # Lower cylinder slightly so it sits more firmly on the table.
    text = text.replace(
        '<body name="cylinder" pos="0.49 -0.348 0.799">',
        '<body name="cylinder" pos="0.49 -0.348 0.791">',
    )

    # Slightly heavier object for stable grasp debugging.
    text = text.replace(
        'mass="0.26"\n                diaginertia="0.000528 0.000528 0.000081"',
        'mass="0.45"\n                diaginertia="0.00091 0.00091 0.00014"',
    )

    if text != old:
        backup = BACKUP_DIR / f"{path.name}.before_cylinder_stability_v5.bak"
        shutil.copy2(path, backup)
        path.write_text(text)
        print(f"Updated: {path}")
        print(f"Backup:  {backup}")
        changed = True

if not changed:
    print("WARNING: cylinder block not found or already changed.")
else:
    print("Cylinder stability v5 applied.")
