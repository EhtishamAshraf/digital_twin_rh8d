#!/usr/bin/env python3

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


RETURN_SPRINGS = {
    "idx_mcp_return_spring",
    "idx_pip_return_spring",
    "idx_dip_return_spring",
}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    src = args.xml.expanduser().resolve()
    dst = args.out.expanduser().resolve()

    tree = ET.parse(src)
    root = tree.getroot()

    # ----------------------------------------------------------
    # Tendons
    # ----------------------------------------------------------

    tendon_root = root.find("./tendon")

    if tendon_root is None:
        raise RuntimeError(
            "Top-level <tendon> not found."
        )

    for spatial in tendon_root.findall("spatial"):

        name = spatial.get("name", "")

        if name == "idx_flex_spatial":

            # MAIN FLEXOR = magenta
            spatial.set(
                "rgba",
                "1 0 1 1",
            )

            spatial.set(
                "width",
                "0.0015",
            )

        elif name in RETURN_SPRINGS:

            # RETURN SPRINGS = red/orange
            spatial.set(
                "rgba",
                "1 0.08 0.02 1",
            )

            spatial.set(
                "width",
                "0.0015",
            )

    # ----------------------------------------------------------
    # G33 artificial wrap cylinders
    # ----------------------------------------------------------

    for geom in root.iter("geom"):

        name = geom.get("name", "")

        if (
            name.startswith("g33a_")
            and
            name.endswith("_spring_guide")
        ):

            # BLUE and very transparent.
            geom.set(
                "rgba",
                "0.05 0.35 1 0.18",
            )

    # ----------------------------------------------------------
    # Hide construction sites / yellow balls.
    # They are only routing helpers.
    # ----------------------------------------------------------

    for site in root.iter("site"):

        name = site.get("name", "")

        if name.startswith("g33a_"):

            site.set(
                "rgba",
                "1 1 1 0",
            )

            site.set(
                "size",
                "0.0002",
            )

    # ----------------------------------------------------------
    # Palm + index transparent only.
    # Other fingers remain opaque.
    # ----------------------------------------------------------

    index_bodies = {
        "Index_Proximal",
        "Index_Middle",
        "Index_Distal",
    }

    for body in root.iter("body"):

        name = body.get(
            "name",
            "",
        )

        if name == "palm_assembly:1":
            alpha = 0.22

        elif name in index_bodies:
            alpha = 0.32

        else:
            alpha = 1.0

        for geom in body.findall("geom"):

            if geom.get("class") != "visual_mesh":
                continue

            rgba = geom.get("rgba")

            if rgba is None:
                continue

            values = [
                float(x)
                for x in rgba.split()
            ]

            if len(values) != 4:
                continue

            values[3] = alpha

            geom.set(
                "rgba",
                " ".join(
                    f"{x:.6g}"
                    for x in values
                ),
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

    mujoco.MjModel.from_xml_path(
        str(dst)
    )

    print("Created clear visualization XML:")
    print(f"  {dst}")
    print()
    print("MAGENTA = main spatial flexor")
    print("RED     = passive return springs")
    print("BLUE    = artificial non-colliding wrap guides")


if __name__ == "__main__":
    main()
