#!/usr/bin/env python3

from pathlib import Path
import shutil
import re

WS = Path.home() / "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/digital_twin/v7_rh8d_ws"
MJCF = WS / "src/ur5e_mujoco_description/mjcf/ur5e_rh8d.xml"

BACKUP_DIR = WS / "rh8d_digital_twin_backups_step6"
BACKUP_DIR.mkdir(exist_ok=True)

backup = BACKUP_DIR / "ur5e_rh8d.xml.before_underactuated_contact_v2.bak"
shutil.copy2(MJCF, backup)

text = MJCF.read_text()


def replace_actuator(name, kp, kv, force):
    global text

    pattern = re.compile(
        rf'(<position name="{re.escape(name)}"[^>]*?\n\s*)'
        rf'kp="[^"]+"\s+kv="[^"]+"\s+forcelimited="true"\s+forcerange="[^"]+"',
        re.MULTILINE,
    )

    text, count = pattern.subn(
        rf'\1kp="{kp}" kv="{kv}" forcelimited="true" forcerange="-{force} {force}"',
        text,
        count=1,
    )

    if count == 0:
        print(f"WARNING: actuator not updated: {name}")


# ---------------------------------------------------------------------
# Finger actuator compliance:
#
# Proximal joints contact first, so they must be weakest/yielding.
# Middle joints wrap next, slightly stronger.
# Distal joints are light and should not generate huge impulses.
# ---------------------------------------------------------------------

# Index
replace_actuator("pos_index_tendon", 18, 3, 5)
replace_actuator("pos_index_middle", 16, 3, 4)
replace_actuator("pos_index_distal", 10, 2, 2.5)

# Middle
replace_actuator("pos_middle_tendon", 18, 3, 5)
replace_actuator("pos_middle_middle", 16, 3, 4)
replace_actuator("pos_middle_distal", 10, 2, 2.5)

# Ring
replace_actuator("pos_ring_small_tendon", 18, 3, 5)
replace_actuator("pos_ring_middle", 16, 3, 4)
replace_actuator("pos_ring_distal", 10, 2, 2.5)

# Small
replace_actuator("pos_small_proximal", 18, 3, 5)
replace_actuator("pos_small_middle", 16, 3, 4)
replace_actuator("pos_small_distal", 10, 2, 2.5)

# Thumb
replace_actuator("pos_thumb_tendon", 16, 3, 4)
replace_actuator("pos_thumb_proximal", 18, 3, 5)
replace_actuator("pos_thumb_distal", 10, 2, 2.5)


# ---------------------------------------------------------------------
# Add soft contact attributes to hand collision geoms and cylinder.
# This reduces contact impulse spikes and bouncing.
# ---------------------------------------------------------------------

soft_contact = {
    "solref": "0.05 1.0",
    "solimp": "0.80 0.95 0.001",
    "friction": "1.8 0.03 0.001",
    "condim": "4",
    "margin": "0.001",
}


def add_or_replace_attr(tag, key, value):
    if re.search(rf'\s{key}="[^"]*"', tag):
        return re.sub(rf'\s{key}="[^"]*"', f' {key}="{value}"', tag)
    return tag[:-2] + f' {key}="{value}"/>' if tag.endswith("/>") else tag[:-1] + f' {key}="{value}">'


def soften_geom_tag(match):
    tag = match.group(0)
    for k, v in soft_contact.items():
        tag = add_or_replace_attr(tag, k, v)
    return tag


# All RH8D collision geoms
text = re.sub(
    r'<geom[^>]*class="hand_collision"[^>]*/>',
    soften_geom_tag,
    text,
)

# The simple fixed cylinder object in your file appears as <geom type="cylinder"/>
text = re.sub(
    r'<geom\s+type="cylinder"\s*/>',
    soften_geom_tag,
    text,
    count=1,
)


MJCF.write_text(text)

print(f"Backup saved: {backup}")
print(f"Updated: {MJCF}")
print("")
print("Applied v2:")
print("  Proximal finger force: about ±5")
print("  Middle finger force:   about ±4")
print("  Distal finger force:   about ±2.5")
print("  Soft contact added to hand_collision geoms and cylinder")
