#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET

import test_index_serial_cable_v35g21b1 as base


# =====================================================================
# G21C — DISTRIBUTED TENDON-SEGMENT FRICTION
#
# G21B0:
#   validated the zero-friction serial cable representation.
#
# G21B1:
#   frictionloss on hidden slip JOINTS compiled correctly but did not
#   generate a meaningful T0/T1/T2 difference.
#
# G21C:
#   hidden slip joints have ZERO friction.
#   dry friction is applied directly to the serial TENDON segments.
#
# No contact-state logic.
# No finger-joint locking.
# No prescribed finger trajectory.
# No contact changes.
# No passive spring changes.
# =====================================================================


# Put strongest case first so, if ranking is tied, the GUI does not
# simply show the zero-friction control again.
CANDIDATES = [
    (
        "segment_friction_strong",
        0.30,
        0.60,
    ),

    (
        "segment_friction_medium",
        0.20,
        0.40,
    ),

    (
        "segment_friction_mild",
        0.10,
        0.20,
    ),

    (
        "segment_friction_zero",
        0.00,
        0.00,
    ),
]


_original_make_serial_model = (
    base.make_serial_model
)


def find_fixed_tendon(
    root,
    name,
):

    for tendon in root.iter(
        "fixed"
    ):

        if tendon.get(
            "name"
        ) == name:

            return tendon

    raise RuntimeError(
        f"Fixed tendon not found: {name}"
    )


def find_joint(
    root,
    name,
):

    for joint in root.iter(
        "joint"
    ):

        if joint.get(
            "name"
        ) == name:

            return joint

    raise RuntimeError(
        f"Joint not found: {name}"
    )


def make_serial_model(
    source_xml,
    destination,
    transmission,
    friction1,
    friction2,
    free_space,
):

    # -------------------------------------------------------------
    # Generate the already validated stable G21B0/B1 serial model,
    # but with ZERO friction on the two hidden slide coordinates.
    # -------------------------------------------------------------

    _original_make_serial_model(
        source_xml,
        destination,
        transmission,
        0.0,
        0.0,
        free_space,
    )

    tree = ET.parse(
        destination
    )

    root = tree.getroot()

    # -------------------------------------------------------------
    # Explicitly verify/force zero hidden-joint friction.
    # -------------------------------------------------------------

    slip1 = find_joint(
        root,
        base.SLIP1,
    )

    slip2 = find_joint(
        root,
        base.SLIP2,
    )

    slip1.set(
        "frictionloss",
        "0",
    )

    slip2.set(
        "frictionloss",
        "0",
    )

    # -------------------------------------------------------------
    # Apply dry friction DIRECTLY to cable segments.
    #
    # SEG0:
    #   proximal motor/MCP segment
    #   no downstream guide loss yet
    #
    # SEG1:
    #   PIP-region segment
    #
    # SEG2:
    #   distal-region segment
    #
    # frictionloss here has units N because tendon coordinate
    # has units metres.
    # -------------------------------------------------------------

    seg0 = find_fixed_tendon(
        root,
        base.SEG0,
    )

    seg1 = find_fixed_tendon(
        root,
        base.SEG1,
    )

    seg2 = find_fixed_tendon(
        root,
        base.SEG2,
    )

    seg0.set(
        "frictionloss",
        "0",
    )

    seg1.set(
        "frictionloss",
        f"{friction1:.12g}",
    )

    seg2.set(
        "frictionloss",
        f"{friction2:.12g}",
    )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )


# Replace only the architecture under test.
base.CANDIDATES = (
    CANDIDATES
)

base.make_serial_model = (
    make_serial_model
)

# So the experiment folder records this wrapper script rather than
# silently copying G21B1 again.
base.__file__ = __file__


print()
print("=" * 140)
print(
    "G21C — DISTRIBUTED TENDON-SEGMENT FRICTION"
)
print("=" * 140)

print(
    "Hidden slip joint frictionloss = 0 / 0 N"
)

print(
    "The F1/F2 values printed below now mean "
    "SEGMENT TENDON frictionloss, not slip-joint friction."
)

print(
    "Required structural observation: "
    "elastic T0, T1 and T2 must become meaningfully different."
)

print()


base.main()
