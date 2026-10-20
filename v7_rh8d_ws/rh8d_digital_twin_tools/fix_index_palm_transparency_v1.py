#!/usr/bin/env python3

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


INDEX_BODIES = {
    "Index_Proximal",
    "Index_Middle",
    "Index_Distal",
}


def set_alpha(geom, alpha):
    rgba = geom.get("rgba")

    if rgba is None:
        return

    values = [float(x) for x in rgba.split()]

    if len(values) != 4:
        return

    values[3] = alpha

    geom.set(
        "rgba",
        " ".join(f"{x:.6g}" for x in values),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    src = args.xml.expanduser().resolve()
    dst = args.out.expanduser().resolve()

    tree = ET.parse(src)
    root = tree.getroot()

    changed = []

    for body in root.iter("body"):
        name = body.get("name", "")

        if name == "palm_assembly:1":
            alpha = 0.25
        elif name in INDEX_BODIES:
            alpha = 0.35
        else:
            alpha = 1.0

        # Only direct visual meshes belonging to THIS body.
        # Transparency is deliberately NOT inherited from the palm.
        for geom in body.findall("geom"):
            if geom.get("class") != "visual_mesh":
                continue

            old = geom.get("rgba")

            set_alpha(
                geom,
                alpha,
            )

            if geom.get("rgba") != old:
                changed.append(
                    (
                        name,
                        geom.get("name", geom.get("mesh", "?")),
                        alpha,
                    )
                )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tree.write(
        dst,
        encoding="utf-8",
        xml_declaration=True,
    )

    # Compile check.
    mujoco.MjModel.from_xml_path(
        str(dst)
    )

    print("Visualization XML created:")
    print(f"  {dst}")
    print()
    print("Visual policy:")
    print("  palm       alpha = 0.25")
    print("  index      alpha = 0.35")
    print("  all others alpha = 1.00")
    print()
    print(f"Changed {len(changed)} visual geoms.")


if __name__ == "__main__":
    main()
