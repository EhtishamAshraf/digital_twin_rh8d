#!/usr/bin/env python3

from pathlib import Path
import re

import mujoco


WS = Path(__file__).resolve().parents[1]

SRC = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_v1.xml"
)

DST = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_cylinder_hw_v2.xml"
)

GEOM_NAME = "index_contact_fixture"


# ============================================================
# Hardware-derived bottle centre
#
# From:
# - measured physical diameter = 69.2 mm
# - top-down contact video
# - hardware MCP
# - open proximal-finger orientation
#
# This is still a CALIBRATION fixture, not final validation.
# ============================================================

NEW_CENTER = (
    0.593611823,
    0.884245992,
    1.564105533,
)


def main():

    text = SRC.read_text()


    # Find the complete opening tag containing our fixture.
    pattern = re.compile(
        r'<geom\b(?=[^>]*\bname\s*=\s*["\']'
        + re.escape(GEOM_NAME)
        + r'["\'])[^>]*>',
        flags=re.DOTALL,
    )


    match = pattern.search(text)


    if match is None:

        raise RuntimeError(
            f'Could not find geom "{GEOM_NAME}" in:\n{SRC}'
        )


    old_tag = match.group(0)


    new_pos = (
        f'{NEW_CENTER[0]:.9f} '
        f'{NEW_CENTER[1]:.9f} '
        f'{NEW_CENTER[2]:.9f}'
    )


    pos_pattern = re.compile(
        r'\bpos\s*=\s*["\'][^"\']*["\']'
    )


    if pos_pattern.search(old_tag):

        new_tag = pos_pattern.sub(
            f'pos="{new_pos}"',
            old_tag,
            count=1,
        )

    else:

        # Insert immediately before closing >
        new_tag = (
            old_tag[:-1]
            + f' pos="{new_pos}">'
        )


    new_text = (
        text[:match.start()]
        + new_tag
        + text[match.end():]
    )


    DST.write_text(
        new_text
    )


    print()
    print("=" * 100)
    print("HARDWARE-DERIVED BOTTLE MODEL")
    print("=" * 100)

    print()
    print("Source:")
    print(SRC)

    print()
    print("Output:")
    print(DST)

    print()
    print("Original fixture tag:")
    print(old_tag)

    print()
    print("Updated fixture tag:")
    print(new_tag)


    # --------------------------------------------------------
    # Compile test
    # --------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(DST)
    )

    gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        GEOM_NAME,
    )


    if gid < 0:

        raise RuntimeError(
            "Updated model compiles, but fixture geom "
            "could not be found."
        )


    print()
    print("Compiled successfully.")

    print(
        "MuJoCo local geom position =",
        model.geom_pos[gid],
    )


    # Check which body owns the fixture.
    bid = model.geom_bodyid[
        gid
    ]

    body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        bid,
    )


    print(
        f"Fixture body id = {bid}"
    )

    print(
        f"Fixture parent body = "
        f"{body_name if body_name else 'world'}"
    )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
