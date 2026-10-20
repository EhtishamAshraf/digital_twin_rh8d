#!/usr/bin/env python3

from pathlib import Path
import math
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
    / "ur5e_rh8d_index_cad_spatial_v35m2_local_returns.xml"
)

TEMP = (
    MJCF
    / "_v35m2_temp.xml"
)


K_RETURN = 180.0
D_RETURN = 0.30


JOINTS = [
    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",
]


PAIRS = [
    {
        "label": "MCP",
        "parent": "v35_p02_mcp_parent",
        "child":  "v35_p03_mcp_child",
        "p_new":  "m2_mcp_return_parent",
        "c_new":  "m2_mcp_return_child",
        "tendon": "idx_return_mcp",
    },

    {
        "label": "PIP",
        "parent": "v35_p04_pip_parent",
        "child":  "v35_p05_pip_child",
        "p_new":  "m2_pip_return_parent",
        "c_new":  "m2_pip_return_child",
        "tendon": "idx_return_pip",
    },

    {
        "label": "DIP",
        "parent": "v35_p06_dip_parent",
        "child":  "v35_p07_dip_child",
        "p_new":  "m2_dip_return_parent",
        "c_new":  "m2_dip_return_child",
        "tendon": "idx_return_dip",
    },
]


def find_source():

    for p in SOURCE_OPTIONS:

        if p.exists():
            return p.resolve()

    raise RuntimeError(
        "No I1/H1/E2A source XML found."
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


def xyz(site):

    return [
        float(x)
        for x in site.get("pos").split()
    ]


def add_mirrored_site(
    root,
    source_name,
    new_name,
):

    src = named(
        root,
        "site",
        source_name,
    )

    body = parent_of(
        root,
        src,
    )

    if body is None:
        raise RuntimeError(
            f"No body parent for {source_name}"
        )

    p = xyz(
        src
    )

    # Mirror the flexor site to the opposite side
    # of the finger locally.
    p[1] *= -1.0

    site = ET.Element(
        "site",
        {
            "name": new_name,
            "pos": " ".join(
                f"{v:.12g}"
                for v in p
            ),
            "size": "0.0015",
            "rgba": "0 1 0 1",
            "group": "4",
        },
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


def tendon_id(
    model,
    name,
):

    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        name,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing tendon {name}"
        )

    return i


def joint_qadr(
    model,
    name,
):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(
            f"Missing joint {name}"
        )

    return int(
        model.jnt_qposadr[jid]
    )


def length(
    model,
    data,
    tid,
):

    mujoco.mj_forward(
        model,
        data,
    )

    return float(
        data.ten_length[tid]
    )


def moment_arm(
    model,
    data,
    tid,
    joint,
):

    qa = joint_qadr(
        model,
        joint,
    )

    q0 = float(
        data.qpos[qa]
    )

    eps = 1e-5

    data.qpos[qa] = q0 + eps
    lp = length(
        model,
        data,
        tid,
    )

    data.qpos[qa] = q0 - eps
    lm = length(
        model,
        data,
        tid,
    )

    data.qpos[qa] = q0

    mujoco.mj_forward(
        model,
        data,
    )

    # For return tendon:
    # positive dL/dq is what we want.
    return (
        lp - lm
    ) / (
        2.0 * eps
    )


def set_pose(
    model,
    data,
    deg,
):

    for j, angle in zip(
        JOINTS,
        deg,
    ):

        data.qpos[
            joint_qadr(
                model,
                j,
            )
        ] = math.radians(
            angle
        )

    mujoco.mj_forward(
        model,
        data,
    )


def main():

    source = find_source()

    print()
    print("=" * 112)
    print("V35M2 — THREE LOCAL PASSIVE RETURN TENDONS")
    print("=" * 112)

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
    # 1. Remove artificial rotational stiffness.
    # ============================================================

    print()
    print("Removing index torsional springs:")

    for name in JOINTS:

        j = named(
            root,
            "joint",
            name,
        )

        print(
            f"  {name}: "
            f"{j.get('stiffness', '0')} -> 0"
        )

        j.set(
            "stiffness",
            "0",
        )

        j.set(
            "springref",
            "0",
        )


    # ============================================================
    # 2. Remove any prior M2 sites/tendons.
    # ============================================================

    for p in PAIRS:

        remove_named(
            root,
            "site",
            p["p_new"],
        )

        remove_named(
            root,
            "site",
            p["c_new"],
        )


    tendon_block = root.find(
        "tendon"
    )

    if tendon_block is None:
        raise RuntimeError(
            "Missing <tendon>"
        )


    for p in PAIRS:

        for t in list(
            tendon_block
        ):

            if t.get(
                "name"
            ) == p["tendon"]:

                tendon_block.remove(
                    t
                )


    # ============================================================
    # 3. Create LOCAL mirrored sites and one spring per joint.
    # ============================================================

    for p in PAIRS:

        add_mirrored_site(
            root,
            p["parent"],
            p["p_new"],
        )

        add_mirrored_site(
            root,
            p["child"],
            p["c_new"],
        )


        t = ET.SubElement(
            tendon_block,
            "spatial",
            {
                "name":
                    p["tendon"],

                "rgba":
                    "0 1 0 1",

                "width":
                    "0.0015",

                "stiffness":
                    f"{K_RETURN:.12g}",

                "damping":
                    f"{D_RETURN:.12g}",

                "frictionloss":
                    "0",
            },
        )


        ET.SubElement(
            t,
            "site",
            {
                "site":
                    p["p_new"],
            },
        )

        ET.SubElement(
            t,
            "site",
            {
                "site":
                    p["c_new"],
            },
        )


    # ============================================================
    # 4. Compile temporary model and obtain exact open lengths.
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


    open_lengths = {}


    for p in PAIRS:

        tid = tendon_id(
            model,
            p["tendon"],
        )

        open_lengths[
            p["tendon"]
        ] = float(
            data.ten_length[
                tid
            ]
        )


    # ============================================================
    # 5. Write each open length as spring rest length.
    # ============================================================

    for p in PAIRS:

        t = named(
            root,
            "spatial",
            p["tendon"],
        )

        L0 = open_lengths[
            p["tendon"]
        ]

        t.set(
            "springlength",
            f"{L0:.15g}",
        )


    write(
        tree,
        OUTPUT,
    )


    # ============================================================
    # 6. Compile final model and verify signs.
    # ============================================================

    model = mujoco.MjModel.from_xml_path(
        str(OUTPUT)
    )

    data = mujoco.MjData(
        model
    )


    poses = [
        (
            "open",
            [0.0, 0.0, 0.0],
        ),

        (
            "hardware",
            [14.5355, 23.8110, 9.3270],
        ),
    ]


    print()
    print("=" * 112)
    print("LOCAL RETURN GEOMETRY CHECK")
    print("=" * 112)

    all_good = True


    for label, pose in poses:

        set_pose(
            model,
            data,
            pose,
        )

        print()
        print(
            f"{label}: "
            f"{pose[0]:.3f}/"
            f"{pose[1]:.3f}/"
            f"{pose[2]:.3f} deg"
        )


        for i, p in enumerate(
            PAIRS
        ):

            tid = tendon_id(
                model,
                p["tendon"],
            )

            r = moment_arm(
                model,
                data,
                tid,
                JOINTS[i],
            )

            print(
                f"  {p['label']}: "
                f"dL/dq = "
                f"{r * 1000:+.4f} mm/rad"
            )


            if r <= 0:

                all_good = False


    print()
    print("=" * 112)

    if not all_good:

        print(
            "ERROR:"
        )

        print(
            "At least one local return tendon has "
            "the wrong sign."
        )

        print(
            "Do NOT run contact replay with this XML."
        )

        print(
            "Send this output back."
        )

        raise SystemExit(
            2
        )


    print(
        "All local return tendons stretch under flexion."
    )

    print()

    print(
        f"return stiffness = "
        f"{K_RETURN:.1f} N/m EACH"
    )

    print(
        f"return damping   = "
        f"{D_RETURN:.2f}"
    )


    for p in PAIRS:

        print(
            f"{p['label']} rest length = "
            f"{open_lengths[p['tendon']] * 1000:.4f} mm"
        )


    print()
    print(
        "FINAL MODEL COMPILED SUCCESSFULLY"
    )

    print(
        "V35M2 ready."
    )


    try:
        TEMP.unlink()
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
