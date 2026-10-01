#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()

SOURCE = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_configdep_geom_v3.xml"
)

OUTPUT = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_configdep_geom_v4.xml"
)

BOTTLE_NAME = "index_contact_fixture"

# Unique collision bit for bottle interaction during index-contact sysid stage
BOTTLE_BIT = 32

ALLOWED_WITH_BOTTLE = {
    "collision_palm",
    "collision_idx_prox",
    "collision_idx_mid",
    "collision_idx_tip",
}


def get_int_attr(elem, key, default):
    return int(elem.attrib.get(key, str(default)))


def main():

    if not SOURCE.exists():
        raise RuntimeError(f"Missing source model:\n{SOURCE}")

    tree = ET.parse(SOURCE)
    root = tree.getroot()

    bottle = None
    updated = []

    for geom in root.iter("geom"):
        name = geom.attrib.get("name", "")

        if name == BOTTLE_NAME:
            bottle = geom

        if name in ALLOWED_WITH_BOTTLE:
            old_aff = get_int_attr(geom, "conaffinity", 1)
            new_aff = old_aff | BOTTLE_BIT
            geom.set("conaffinity", str(new_aff))
            updated.append((name, old_aff, new_aff))

    if bottle is None:
        raise RuntimeError(f"Could not find bottle geom '{BOTTLE_NAME}'")

    old_ct = get_int_attr(bottle, "contype", 1)
    old_ca = get_int_attr(bottle, "conaffinity", 1)

    # Bottle collides only with geoms whose conaffinity includes BOTTLE_BIT.
    bottle.set("contype", str(BOTTLE_BIT))
    bottle.set("conaffinity", "0")

    tree.write(
        OUTPUT,
        encoding="unicode",
        xml_declaration=True,
    )

    model = mujoco.MjModel.from_xml_path(str(OUTPUT))

    gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        BOTTLE_NAME,
    )

    if gid < 0:
        raise RuntimeError("Bottle not found after compile.")

    print()
    print("=" * 100)
    print("RH8D INDEX CONFIGURATION-DEPENDENT CONTACT MASK V4")
    print("=" * 100)

    print()
    print("Source:")
    print(SOURCE)

    print()
    print("Output:")
    print(OUTPUT)

    print()
    print("Bottle mask:")
    print(f"  old contype     = {old_ct}")
    print(f"  old conaffinity = {old_ca}")
    print(f"  new contype     = {BOTTLE_BIT}")
    print(f"  new conaffinity = 0")

    print()
    print("Allowed bottle-contact geoms:")
    for name, old_aff, new_aff in updated:
        print(f"  {name:20s} conaffinity {old_aff} -> {new_aff}")

    print()
    print("MuJoCo compile: PASS")
    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
