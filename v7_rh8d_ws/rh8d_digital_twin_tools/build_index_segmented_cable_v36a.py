#!/usr/bin/env python3

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

TOOLS = WS / "rh8d_digital_twin_tools"
MJCF = WS / "src/ur5e_mujoco_description/mjcf"

OUTDIR = (
    TOOLS
    / "results"
    / "v36a_segmented_zero_friction"
)

SOURCE_OPTIONS = [
    MJCF / "ur5e_rh8d_index_cad_spatial_v35i1_best.xml",
    MJCF / "ur5e_rh8d_index_cad_spatial_v35h1_best.xml",
    MJCF / "ur5e_rh8d_index_cad_spatial_v35b2_selected_v35e2a_zero_margin.xml",
]

OUTPUT = (
    MJCF
    / "ur5e_rh8d_index_segmented_v36a.xml"
)

TEMP = (
    MJCF
    / "_v36a_temp.xml"
)


# ============================================================
# Equivalent whole-cable axial stiffness found in O1.
# ============================================================

TOTAL_K = 9000.0


# ============================================================
# Internal model names
# ============================================================

FULL_TENDON = "idx_flex_spatial"
SPOOL_JOINT = "index_motor_spool"

SLIP1 = "idx_v36_slip_mcp"
SLIP2 = "idx_v36_slip_pip"

EXT1 = "idx_v36_ext1"
EXT2 = "idx_v36_ext2"
EXT3 = "idx_v36_ext3"


SEGMENTS = [
    {
        "label": "MCP",
        "geom": "idx_v36_seg1_geom",
        "state": "idx_v36_seg1_state",
        "sites": [
            "v35_p01_palm_entry",
            "v35_p02_mcp_parent",
            "v35_p03_mcp_child",
        ],
        "ext": EXT1,
    },

    {
        "label": "PIP",
        "geom": "idx_v36_seg2_geom",
        "state": "idx_v36_seg2_state",
        "sites": [
            "v35_p03_mcp_child",
            "v35_p04_pip_parent",
            "v35_p05_pip_child",
        ],
        "ext": EXT2,
    },

    {
        "label": "DIP",
        "geom": "idx_v36_seg3_geom",
        "state": "idx_v36_seg3_state",
        "sites": [
            "v35_p05_pip_child",
            "v35_p06_dip_parent",
            "v35_p07_dip_child",
            "v35_p08_distal_turn",
            "v35_p09_anchor",
        ],
        "ext": EXT3,
    },
]


def find_source():

    for p in SOURCE_OPTIONS:

        if p.exists():
            return p.resolve()

    raise RuntimeError(
        "Could not find I1/H1/E2A source XML."
    )


def named(root, tag, name):

    for e in root.iter(tag):

        if e.get("name") == name:
            return e

    raise RuntimeError(
        f"Missing <{tag} name='{name}'>"
    )


def parent_of(root, target):

    for p in root.iter():

        for c in list(p):

            if c is target:
                return p

    return None


def remove_named(root, tag, name):

    for e in list(root.iter(tag)):

        if e.get("name") == name:

            p = parent_of(
                root,
                e,
            )

            if p is not None:
                p.remove(e)


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


def tendon_id(model, name):

    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        name,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing tendon: {name}"
        )

    return i


def joint_id(model, name):

    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing joint: {name}"
        )

    return i


def add_hidden_joint_body(
    worldbody,
    body_name,
    joint_name,
    range_text,
):

    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": body_name,
        },
    )

    ET.SubElement(
        body,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": "0.0001",
            "diaginertia": "1e-8 1e-8 1e-8",
        },
    )

    joint = ET.SubElement(
        body,
        "joint",
        {
            "name": joint_name,
            "type": "slide",
            "axis": "1 0 0",
            "limited": "true",
            "range": range_text,
            "damping": "0",
            "frictionloss": "0",
        },
    )

    return joint


def main():

    source = find_source()

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print("=" * 120)
    print("V36A — SEGMENTED CABLE / ZERO GUIDE FRICTION")
    print("=" * 120)

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


    worldbody = root.find(
        "worldbody"
    )

    tendon_block = root.find(
        "tendon"
    )

    equality_block = root.find(
        "equality"
    )


    if worldbody is None:
        raise RuntimeError(
            "Missing <worldbody>"
        )

    if tendon_block is None:
        raise RuntimeError(
            "Missing <tendon>"
        )

    if equality_block is None:
        equality_block = ET.SubElement(
            root,
            "equality",
        )


    # ============================================================
    # Read damping from the current selected model BEFORE
    # disabling the original whole-finger tendon.
    # ============================================================

    full = named(
        root,
        "spatial",
        FULL_TENDON,
    )


    old_k = float(
        full.get(
            "stiffness",
            "0",
        )
    )

    total_damping = float(
        full.get(
            "damping",
            "0",
        )
    )


    print()
    print(
        f"Original full tendon XML stiffness = "
        f"{old_k:.3f} N/m"
    )

    print(
        f"Original full tendon damping       = "
        f"{total_damping:.6f}"
    )

    print(
        f"V36 equivalent total stiffness     = "
        f"{TOTAL_K:.3f} N/m"
    )


    # ============================================================
    # The old full tendon stays only as a visual/sensor path.
    #
    # Runtime comparison will also force its stiffness to zero.
    # ============================================================

    full.set(
        "stiffness",
        "0",
    )

    full.set(
        "damping",
        "0",
    )

    full.set(
        "frictionloss",
        "0",
    )

    if "springlength" in full.attrib:
        del full.attrib[
            "springlength"
        ]


    # ============================================================
    # Remove old V36 objects if script is rerun.
    # ============================================================

    hidden_bodies = [
        "idx_v36_slip_mcp_body",
        "idx_v36_slip_pip_body",
        "idx_v36_ext1_body",
        "idx_v36_ext2_body",
        "idx_v36_ext3_body",
    ]


    for name in hidden_bodies:

        remove_named(
            root,
            "body",
            name,
        )


    for s in SEGMENTS:

        remove_named(
            root,
            "spatial",
            s["geom"],
        )

        remove_named(
            root,
            "fixed",
            s["state"],
        )


    for name in [
        "idx_v36_eq_seg1",
        "idx_v36_eq_seg2",
        "idx_v36_eq_seg3",
    ]:

        remove_named(
            root,
            "tendon",
            name,
        )


    # ============================================================
    # Hidden internal cable coordinates.
    #
    # s1, s2:
    #   cable sliding through MCP/PIP guide holes.
    #
    # e1,e2,e3:
    #   elastic elongation of each cable segment.
    # ============================================================

    add_hidden_joint_body(
        worldbody,
        "idx_v36_slip_mcp_body",
        SLIP1,
        "-0.03 0.03",
    )

    add_hidden_joint_body(
        worldbody,
        "idx_v36_slip_pip_body",
        SLIP2,
        "-0.03 0.03",
    )


    ext_joint_elements = {}


    ext_joint_elements[
        EXT1
    ] = add_hidden_joint_body(
        worldbody,
        "idx_v36_ext1_body",
        EXT1,
        "0 0.03",
    )

    ext_joint_elements[
        EXT2
    ] = add_hidden_joint_body(
        worldbody,
        "idx_v36_ext2_body",
        EXT2,
        "0 0.03",
    )

    ext_joint_elements[
        EXT3
    ] = add_hidden_joint_body(
        worldbody,
        "idx_v36_ext3_body",
        EXT3,
        "0 0.03",
    )


    # ============================================================
    # Three purely GEOMETRIC cable portions.
    # ============================================================

    for s in SEGMENTS:

        spatial = ET.SubElement(
            tendon_block,
            "spatial",
            {
                "name": s["geom"],
                "rgba": "0 1 1 0.7",
                "width": "0.0010",
                "stiffness": "0",
                "damping": "0",
                "frictionloss": "0",
            },
        )


        for site_name in s[
            "sites"
        ]:

            ET.SubElement(
                spatial,
                "site",
                {
                    "site": site_name,
                },
            )


    # ============================================================
    # Internal state tendons.
    #
    # Segment 1:
    #
    #   ΔL1 = -u + s1 + e1
    #
    # Segment 2:
    #
    #   ΔL2 = -s1 + s2 + e2
    #
    # Segment 3:
    #
    #   ΔL3 = -s2 + e3
    #
    # Summing:
    #
    #   e1 + e2 + e3
    #     = ΔL1 + ΔL2 + ΔL3 + u
    #
    # so s1/s2 only redistribute cable; they do not create it.
    # ============================================================

    f1 = ET.SubElement(
        tendon_block,
        "fixed",
        {
            "name":
                "idx_v36_seg1_state",
        },
    )

    ET.SubElement(
        f1,
        "joint",
        {
            "joint": SPOOL_JOINT,
            "coef": "-1",
        },
    )

    ET.SubElement(
        f1,
        "joint",
        {
            "joint": SLIP1,
            "coef": "1",
        },
    )

    ET.SubElement(
        f1,
        "joint",
        {
            "joint": EXT1,
            "coef": "1",
        },
    )


    f2 = ET.SubElement(
        tendon_block,
        "fixed",
        {
            "name":
                "idx_v36_seg2_state",
        },
    )

    ET.SubElement(
        f2,
        "joint",
        {
            "joint": SLIP1,
            "coef": "-1",
        },
    )

    ET.SubElement(
        f2,
        "joint",
        {
            "joint": SLIP2,
            "coef": "1",
        },
    )

    ET.SubElement(
        f2,
        "joint",
        {
            "joint": EXT2,
            "coef": "1",
        },
    )


    f3 = ET.SubElement(
        tendon_block,
        "fixed",
        {
            "name":
                "idx_v36_seg3_state",
        },
    )

    ET.SubElement(
        f3,
        "joint",
        {
            "joint": SLIP2,
            "coef": "-1",
        },
    )

    ET.SubElement(
        f3,
        "joint",
        {
            "joint": EXT3,
            "coef": "1",
        },
    )


    # ============================================================
    # Geometry-to-internal-coordinate constraints.
    #
    # MuJoCo tendon equality is expressed relative to the
    # reference lengths, so polycoef starts at ZERO.
    # ============================================================

    for i, s in enumerate(
        SEGMENTS,
        1,
    ):

        ET.SubElement(
            equality_block,
            "tendon",
            {
                "name":
                    f"idx_v36_eq_seg{i}",

                "tendon1":
                    s["geom"],

                "tendon2":
                    s["state"],

                "polycoef":
                    "0 1 0 0 0",

                "solref":
                    "0.002 1",

                "solimp":
                    "0.95 0.99 0.001",
            },
        )


    # ============================================================
    # First temporary compile: measure exact CAD segment lengths.
    # ============================================================

    write(
        tree,
        TEMP,
    )


    model = mujoco.MjModel.from_xml_path(
        str(TEMP)
    )

    data = mujoco.MjData(
        model
    )

    mujoco.mj_forward(
        model,
        data,
    )


    lengths = []


    for s in SEGMENTS:

        tid = tendon_id(
            model,
            s["geom"],
        )

        lengths.append(
            float(
                data.ten_length[
                    tid
                ]
            )
        )


    total_length = sum(
        lengths
    )


    print()
    print("=" * 120)
    print("CAD SEGMENT LENGTHS")
    print("=" * 120)


    for s, L in zip(
        SEGMENTS,
        lengths,
    ):

        print(
            f"{s['label']}: "
            f"{L * 1000.0:.6f} mm"
        )


    print(
        f"TOTAL: "
        f"{total_length * 1000.0:.6f} mm"
    )


    # ============================================================
    # Same cable material/cross-section:
    #
    #   K = EA/L
    #
    # TOTAL_K = EA/L_total
    #
    # therefore:
    #
    #   Ki = TOTAL_K * L_total/Li
    #
    # The same scaling is applied to viscous cable damping.
    # ============================================================

    segment_K = []

    segment_D = []


    for L in lengths:

        Ki = (
            TOTAL_K
            *
            total_length
            /
            L
        )

        segment_K.append(
            Ki
        )


        if total_damping > 0:

            Di = (
                total_damping
                *
                total_length
                /
                L
            )

        else:

            Di = 0.0


        segment_D.append(
            Di
        )


    print()
    print("=" * 120)
    print("SEGMENT ELASTIC PARAMETERS")
    print("=" * 120)


    for s, K, D in zip(
        SEGMENTS,
        segment_K,
        segment_D,
    ):

        print(
            f"{s['label']}: "
            f"K={K:.3f} N/m   "
            f"D={D:.6f} Ns/m"
        )


    # ============================================================
    # Apply elastic parameters to e1/e2/e3.
    # ============================================================

    for s, K, D in zip(
        SEGMENTS,
        segment_K,
        segment_D,
    ):

        j = ext_joint_elements[
            s["ext"]
        ]

        j.set(
            "stiffness",
            f"{K:.15g}",
        )

        j.set(
            "damping",
            f"{D:.15g}",
        )

        j.set(
            "springref",
            "0",
        )


    write(
        tree,
        OUTPUT,
    )


    # ============================================================
    # Final compile verification.
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


    print()
    print("=" * 120)
    print("FINAL V36A COMPILE CHECK")
    print("=" * 120)


    for name in [
        SLIP1,
        SLIP2,
        EXT1,
        EXT2,
        EXT3,
    ]:

        jid = joint_id(
            model,
            name,
        )

        print(
            f"{name:22s} "
            f"stiffness="
            f"{model.jnt_stiffness[jid]:.6f}"
        )


    print()
    print(
        "Guide friction:"
    )

    for name in [
        SLIP1,
        SLIP2,
    ]:

        jid = joint_id(
            model,
            name,
        )

        did = int(
            model.jnt_dofadr[
                jid
            ]
        )

        print(
            f"  {name}: "
            f"{model.dof_frictionloss[did]:.6f} N"
        )


    # Store a copy in the dedicated experiment folder.
    shutil.copy2(
        OUTPUT,
        OUTDIR / OUTPUT.name,
    )


    manifest = (
        OUTDIR
        / "v36a_build_info.txt"
    )


    with manifest.open(
        "w"
    ) as f:

        f.write(
            "V36A segmented cable architecture\n"
        )

        f.write(
            f"source={source}\n"
        )

        f.write(
            f"working_xml={OUTPUT}\n"
        )

        f.write(
            f"TOTAL_K={TOTAL_K}\n"
        )

        f.write(
            f"TOTAL_DAMPING={total_damping}\n"
        )

        f.write(
            f"L_total={total_length}\n"
        )

        for s, L, K, D in zip(
            SEGMENTS,
            lengths,
            segment_K,
            segment_D,
        ):

            f.write(
                f"{s['label']}: "
                f"L={L} K={K} D={D}\n"
            )


    try:
        TEMP.unlink()
    except FileNotFoundError:
        pass


    print()
    print(
        "FINAL MODEL COMPILED SUCCESSFULLY"
    )

    print(
        "Guide friction = ZERO / ZERO"
    )

    print(
        f"Working XML:\n{OUTPUT}"
    )

    print(
        f"\nAll experiment artifacts:\n{OUTDIR}"
    )

    print("=" * 120)


if __name__ == "__main__":
    main()
