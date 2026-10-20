#!/usr/bin/env python3

from pathlib import Path
import copy
import math
import shutil
import xml.etree.ElementTree as ET

import mujoco


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

SOURCE_OPTIONS = [
    MJCF / "ur5e_rh8d_index_cad_spatial_v35i1_best.xml",
    MJCF / "ur5e_rh8d_index_cad_spatial_v35h1_best.xml",
    MJCF / "ur5e_rh8d_index_cad_spatial_v35b2_selected_v35e2a_zero_margin.xml",
]

OUTPUT = (
    MJCF
    / "ur5e_rh8d_index_cad_spatial_v35m1_passive_return.xml"
)

TEMP = (
    MJCF
    / "_v35m1_temp.xml"
)


INDEX_JOINTS = [
    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",
]


# ------------------------------------------------------------
# Initial translational spring stiffness.
#
# This is NOT final calibration.
#
# We deliberately start with a moderate shared elastic element,
# instead of reproducing the old extremely stiff DIP spring.
# ------------------------------------------------------------

RETURN_K = 180.0      # N/m
RETURN_DAMPING = 0.5  # N s/m


def source_xml():

    for p in SOURCE_OPTIONS:

        if p.exists():
            return p.resolve()

    raise RuntimeError(
        "Could not find I1/H1/E2A XML."
    )


def parent_of(root, target):

    for parent in root.iter():

        for child in list(parent):

            if child is target:
                return parent

    return None


def named(root, tag, name):

    for e in root.iter(tag):

        if e.get("name") == name:
            return e

    raise RuntimeError(
        f"Missing <{tag} name='{name}'>"
    )


def parse_pos(site):

    return [
        float(v)
        for v in site.get("pos").split()
    ]


def set_pos(site, xyz):

    site.set(
        "pos",
        " ".join(
            f"{v:.12g}"
            for v in xyz
        ),
    )


def mirrored_site(
    root,
    original_name,
    new_name,
):

    src = named(
        root,
        "site",
        original_name,
    )

    body = parent_of(
        root,
        src,
    )

    if body is None:
        raise RuntimeError(
            f"No body parent for {original_name}"
        )


    xyz = parse_pos(
        src
    )


    # --------------------------------------------------------
    # Current flexor route is on the palmar side:
    # local y is predominantly negative.
    #
    # Mirror across local y=0 to create a first dorsal
    # passive-return routing hypothesis.
    # --------------------------------------------------------

    xyz[1] *= -1.0


    site = ET.Element(
        "site",
        {
            "name":
                new_name,

            "size":
                "0.0015",

            "rgba":
                "0 1 0 1",

            "group":
                "4",
        },
    )

    set_pos(
        site,
        xyz,
    )

    body.append(
        site
    )


def write(tree, path):

    try:
        ET.indent(
            tree,
            space="  ",
        )
    except AttributeError:
        pass

    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )


def get_tendon_length(
    xml,
    tendon_name,
):

    model = mujoco.MjModel.from_xml_path(
        str(xml)
    )

    data = mujoco.MjData(
        model
    )

    mujoco.mj_forward(
        model,
        data,
    )


    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        tendon_name,
    )

    if tid < 0:
        raise RuntimeError(
            f"Tendon missing: {tendon_name}"
        )

    return float(
        data.ten_length[
            tid
        ]
    )


def main():

    source = source_xml()


    print()
    print("=" * 110)
    print("V35M1 — EXPLICIT PASSIVE RETURN TENDON")
    print("=" * 110)

    print(
        f"SOURCE:\n{source}"
    )

    print(
        f"\nOUTPUT:\n{OUTPUT}"
    )


    tree = ET.parse(
        source
    )

    root = tree.getroot()


    # ============================================================
    # 1. Remove the three independent rotational return springs.
    # ============================================================

    print()
    print("Removing torsional joint stiffness:")


    for name in INDEX_JOINTS:

        joint = named(
            root,
            "joint",
            name,
        )

        print(
            f"  {name}: "
            f"{joint.get('stiffness', '0')} -> 0"
        )

        joint.set(
            "stiffness",
            "0",
        )

        # springref no longer matters once stiffness=0.
        joint.set(
            "springref",
            "0",
        )


    # ============================================================
    # 2. Create dorsal passive-return sites by mirroring the
    #    existing CAD flexor route.
    #
    # Do NOT use spool site: passive tendon is anchored to palm.
    # ============================================================

    mapping = [

        (
            "v35_p01_palm_entry",
            "m1_ret_p01_palm",
        ),

        (
            "v35_p02_mcp_parent",
            "m1_ret_p02_mcp_parent",
        ),

        (
            "v35_p03_mcp_child",
            "m1_ret_p03_mcp_child",
        ),

        (
            "v35_p04_pip_parent",
            "m1_ret_p04_pip_parent",
        ),

        (
            "v35_p05_pip_child",
            "m1_ret_p05_pip_child",
        ),

        (
            "v35_p06_dip_parent",
            "m1_ret_p06_dip_parent",
        ),

        (
            "v35_p07_dip_child",
            "m1_ret_p07_dip_child",
        ),

        (
            "v35_p08_distal_turn",
            "m1_ret_p08_distal_turn",
        ),

        (
            "v35_p09_anchor",
            "m1_ret_p09_anchor",
        ),
    ]


    # Remove previous M1 sites if script is rerun.
    for _, new_name in mapping:

        for site in list(
            root.iter("site")
        ):

            if site.get(
                "name"
            ) == new_name:

                p = parent_of(
                    root,
                    site,
                )

                if p is not None:
                    p.remove(
                        site
                    )


    for old, new in mapping:

        mirrored_site(
            root,
            old,
            new,
        )


    # ============================================================
    # 3. Add passive spatial tendon.
    # ============================================================

    tendon_block = root.find(
        "tendon"
    )

    if tendon_block is None:

        raise RuntimeError(
            "Missing <tendon>"
        )


    for t in list(
        tendon_block
    ):

        if t.get(
            "name"
        ) == "idx_return_spatial":

            tendon_block.remove(
                t
            )


    ret = ET.SubElement(
        tendon_block,
        "spatial",
        {
            "name":
                "idx_return_spatial",

            "rgba":
                "0 1 0 1",

            "width":
                "0.0015",

            "stiffness":
                f"{RETURN_K:.12g}",

            "damping":
                f"{RETURN_DAMPING:.12g}",

            "frictionloss":
                "0",
        },
    )


    for _, new_name in mapping:

        ET.SubElement(
            ret,
            "site",
            {
                "site":
                    new_name,
            },
        )


    # ============================================================
    # 4. First write with no springlength, compile, measure the
    #    natural OPEN length exactly.
    # ============================================================

    write(
        tree,
        TEMP,
    )


    L0 = get_tendon_length(
        TEMP,
        "idx_return_spatial",
    )


    print()
    print(
        f"Passive return open length = "
        f"{L0 * 1000:.6f} mm"
    )


    # Rest length = open configuration.
    ret.set(
        "springlength",
        f"{L0:.15g}",
    )


    write(
        tree,
        OUTPUT,
    )


    # ============================================================
    # 5. Final compile check.
    # ============================================================

    model = mujoco.MjModel.from_xml_path(
        str(OUTPUT)
    )

    data = mujoco.MjData(
        model
    )

    mujoco.mj_forward(
        model,
        data,
    )


    rid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        "idx_return_spatial",
    )


    print()
    print(
        "FINAL MODEL COMPILED SUCCESSFULLY"
    )

    print(
        f"return stiffness = "
        f"{model.tendon_stiffness[rid]:.3f} N/m"
    )

    print(
        f"return damping   = "
        f"{model.tendon_damping[rid]:.3f}"
    )

    print(
        f"return length    = "
        f"{data.ten_length[rid] * 1000:.6f} mm"
    )

    print(
        f"return rest      = "
        f"{L0 * 1000:.6f} mm"
    )


    try:
        TEMP.unlink()
    except FileNotFoundError:
        pass


    print()
    print(
        "M1 architecture:"
    )

    print(
        "  flexor spatial tendon      = unchanged"
    )

    print(
        "  flexor motor/spool         = unchanged"
    )

    print(
        "  rotational joint springs  = REMOVED"
    )

    print(
        "  dorsal passive tendon      = ADDED"
    )

    print()
    print(
        "V35M1 ready."
    )


if __name__ == "__main__":
    main()
