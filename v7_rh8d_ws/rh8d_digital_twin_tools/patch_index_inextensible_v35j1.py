#!/usr/bin/env python3

from pathlib import Path
import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = WS / "src/ur5e_mujoco_description/mjcf"


# Prefer the actual latest result.
SOURCE_CANDIDATES = [
    MJCF / "ur5e_rh8d_index_cad_spatial_v35i1_best.xml",
    MJCF / "ur5e_rh8d_index_cad_spatial_v35h1_best.xml",
    MJCF / "ur5e_rh8d_index_cad_spatial_v35b2_selected_v35e2a_zero_margin.xml",
]


OUTPUT = (
    MJCF /
    "ur5e_rh8d_index_cad_spatial_v35j1_inextensible.xml"
)

PROVISIONAL = (
    MJCF /
    "_v35j1_provisional.xml"
)


INDEX_JOINTS = [
    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",
]


def find_source():

    for p in SOURCE_CANDIDATES:

        if p.exists():
            return p.resolve()

    raise RuntimeError(
        "Could not find I1, H1 or E2A source XML."
    )


def find_named(root, tag, name):

    for e in root.iter(tag):

        if e.get("name") == name:
            return e

    raise RuntimeError(
        f"Could not find <{tag} name='{name}'>"
    )


def find_parent(root, target):

    for parent in root.iter():

        for child in list(parent):

            if child is target:
                return parent

    return None


def remove_named(root, tag, name):

    for e in list(root.iter(tag)):

        if e.get("name") == name:

            parent = find_parent(
                root,
                e,
            )

            if parent is not None:
                parent.remove(e)


def write_xml(tree, path):

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


def tendon_length(model_path):

    model = mujoco.MjModel.from_xml_path(
        str(model_path)
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
        "idx_flex_spatial",
    )

    if tid < 0:
        raise RuntimeError(
            "idx_flex_spatial missing"
        )

    return (
        model,
        data,
        tid,
        float(
            data.ten_length[tid]
        ),
    )


def moment_arms(
    model,
    data,
    tid,
):

    eps = 1.0e-5

    output = []

    for name in INDEX_JOINTS:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            raise RuntimeError(
                f"Missing joint {name}"
            )

        qadr = int(
            model.jnt_qposadr[jid]
        )

        q0 = float(
            data.qpos[qadr]
        )

        data.qpos[qadr] = (
            q0 + eps
        )

        mujoco.mj_forward(
            model,
            data,
        )

        lp = float(
            data.ten_length[tid]
        )

        data.qpos[qadr] = (
            q0 - eps
        )

        mujoco.mj_forward(
            model,
            data,
        )

        lm = float(
            data.ten_length[tid]
        )

        data.qpos[qadr] = q0

        output.append(
            (
                lp - lm
            )
            /
            (
                2.0 * eps
            )
        )

    mujoco.mj_forward(
        model,
        data,
    )

    return np.asarray(
        output,
        dtype=float,
    )


def main():

    source = find_source()

    print()
    print(
        "=" * 100
    )

    print(
        "V35J1 — NEAR-INEXTENSIBLE FLEXOR PATCH"
    )

    print(
        "=" * 100
    )

    print(
        f"SOURCE:\n{source}"
    )

    print(
        f"\nOUTPUT:\n{OUTPUT}"
    )

    print()


    tree = ET.parse(
        source
    )

    root = tree.getroot()


    # ============================================================
    # 1. Preserve the current CAD finger route but REMOVE the
    #    artificial moving spool site from the spatial path.
    #
    # The motor spool will instead change the required cable
    # length through an equality constraint.
    # ============================================================

    flex = find_named(
        root,
        "spatial",
        "idx_flex_spatial",
    )


    sites = list(
        flex.findall(
            "site"
        )
    )


    spool_sites = [
        s
        for s in sites
        if s.get(
            "site"
        )
        ==
        "v35_spool_site"
    ]


    if len(
        spool_sites
    ) != 1:

        raise RuntimeError(
            "Expected exactly one v35_spool_site "
            "inside idx_flex_spatial."
        )


    flex.remove(
        spool_sites[0]
    )


    # ============================================================
    # 2. The FLEXOR is no longer a large spring.
    #
    # Its length is imposed by the cable/spool constraint below.
    # Passive elasticity remains at the joints for this diagnostic.
    # ============================================================

    flex.set(
        "stiffness",
        "0",
    )

    flex.set(
        "damping",
        "0",
    )

    flex.set(
        "frictionloss",
        "0",
    )


    if "springlength" in flex.attrib:

        del flex.attrib[
            "springlength"
        ]


    # ============================================================
    # 3. Add a one-DOF generalized tendon representing how much
    #    cable the motor spool has taken up.
    # ============================================================

    tendon_block = root.find(
        "tendon"
    )

    if tendon_block is None:

        raise RuntimeError(
            "Missing <tendon> block"
        )


    remove_named(
        root,
        "fixed",
        "idx_spool_drive",
    )


    spool_tendon = ET.Element(
        "fixed",
        {
            "name":
                "idx_spool_drive",
        },
    )


    ET.SubElement(
        spool_tendon,
        "joint",
        {
            "joint":
                "index_motor_spool",

            "coef":
                "1",
        },
    )


    tendon_block.insert(
        0,
        spool_tendon,
    )


    # ============================================================
    # 4. Compile a temporary model WITHOUT the equality.
    #
    # This lets MuJoCo tell us the EXACT current open length of
    # the CAD tendon path. We do NOT guess this number.
    # ============================================================

    write_xml(
        tree,
        PROVISIONAL,
    )


    model, data, tid, L0 = (
        tendon_length(
            PROVISIONAL
        )
    )


    arms = moment_arms(
        model,
        data,
        tid,
    )


    print(
        f"Open finger cable path = "
        f"{L0 * 1000.0:.6f} mm"
    )

    print()

    print(
        "Geometric dL/dq at open pose:"
    )

    print(
        f"  MCP = "
        f"{arms[0] * 1000.0:+.4f} mm/rad"
    )

    print(
        f"  PIP = "
        f"{arms[1] * 1000.0:+.4f} mm/rad"
    )

    print(
        f"  DIP = "
        f"{arms[2] * 1000.0:+.4f} mm/rad"
    )


    # ============================================================
    # 5. Add inextensible cable relationship:
    #
    #          L_finger = L_open - q_spool
    #
    # Positive spool displacement therefore TAKES UP cable.
    #
    # This is the key architectural experiment.
    # ============================================================

    equality = root.find(
        "equality"
    )


    if equality is None:

        equality = ET.Element(
            "equality"
        )

        actuator = root.find(
            "actuator"
        )

        children = list(
            root
        )

        if actuator is not None:

            root.insert(
                children.index(
                    actuator
                ),
                equality,
            )

        else:

            root.append(
                equality
            )


    for e in list(
        equality.findall(
            "tendon"
        )
    ):

        if e.get(
            "name"
        ) == (
            "idx_inextensible_coupling"
        ):

            equality.remove(
                e
            )


    ET.SubElement(
        equality,
        "tendon",
        {
            "name":
                "idx_inextensible_coupling",

            "tendon1":
                "idx_flex_spatial",

            "tendon2":
                "idx_spool_drive",

            # tendon1 =
            # L0 - tendon2
            "polycoef":
                f"{L0:.15g} -1 0 0 0",

            # stiff but not numerically singular.
            "solref":
                "0.003 1",

            "solimp":
                "0.95 0.99 0.001",
        },
    )


    write_xml(
        tree,
        OUTPUT,
    )


    # ============================================================
    # 6. Final compile check.
    # ============================================================

    final_model = (
        mujoco.MjModel.from_xml_path(
            str(
                OUTPUT
            )
        )
    )


    final_data = (
        mujoco.MjData(
            final_model
        )
    )


    mujoco.mj_forward(
        final_model,
        final_data,
    )


    try:

        PROVISIONAL.unlink()

    except FileNotFoundError:

        pass


    print()
    print(
        "FINAL MODEL COMPILED SUCCESSFULLY"
    )

    print(
        f"nq={final_model.nq} "
        f"nv={final_model.nv} "
        f"nu={final_model.nu} "
        f"ntendon={final_model.ntendon} "
        f"neq={final_model.neq}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  * CAD flexor route unchanged"
    )

    print(
        "  * encoder/spool actuator unchanged"
    )

    print(
        "  * 2100 N/m flexor spring REMOVED"
    )

    print(
        "  * spool displacement now imposes cable take-up"
    )

    print(
        "  * joint return springs still present"
    )

    print()
    print(
        "V35J1 ready."
    )


if __name__ == "__main__":

    main()
