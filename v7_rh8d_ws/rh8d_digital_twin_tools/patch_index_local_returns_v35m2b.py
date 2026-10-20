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
    / "ur5e_rh8d_index_cad_spatial_v35m2b_local_fixed_returns.xml"
)


JOINTS = [
    (
        "MCP",
        "Index_Proximal--palm_assembly:1",
        "idx_return_mcp",
    ),

    (
        "PIP",
        "Index_Middle--Index_Proximal",
        "idx_return_pip",
    ),

    (
        "DIP",
        "Index_Distal--Index_Middle",
        "idx_return_dip",
    ),
]


# Local elastic element parameters
RETURN_R = 0.005     # m = 5 mm effective pulley/moment arm
RETURN_K = 180.0     # N/m
RETURN_D = 0.30      # N s/m


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


def main():

    source = find_source()

    print()
    print("=" * 112)
    print("V35M2B — THREE LOCAL FIXED-TENDON RETURN SPRINGS")
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
    # 1. Remove old independent rotational joint springs.
    # ============================================================

    print()
    print("Removing torsional joint springs:")


    for label, joint_name, _ in JOINTS:

        j = named(
            root,
            "joint",
            joint_name,
        )

        print(
            f"  {label}: "
            f"stiffness "
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
    # 2. Add one LOCAL elastic tendon per joint.
    #
    # tendon length:
    #
    #     L = r * q
    #
    # q > 0 flexion therefore stretches the return spring.
    #
    # Tendon tension:
    #
    #     F = K * L
    #
    # restoring joint torque:
    #
    #     tau = -r * F
    #         = -K*r^2*q
    #
    # Thus each behaves like a physical local spring wrapped
    # around a ~5 mm pulley.
    # ============================================================

    tendon_block = root.find(
        "tendon"
    )

    if tendon_block is None:

        raise RuntimeError(
            "Missing <tendon> block"
        )


    # Remove old copies if rerun.
    for _, _, tendon_name in JOINTS:

        for t in list(
            tendon_block
        ):

            if t.get(
                "name"
            ) == tendon_name:

                tendon_block.remove(
                    t
                )


    for label, joint_name, tendon_name in JOINTS:

        fixed = ET.SubElement(
            tendon_block,
            "fixed",
            {
                "name":
                    tendon_name,

                "stiffness":
                    f"{RETURN_K:.12g}",

                "damping":
                    f"{RETURN_D:.12g}",

                "frictionloss":
                    "0",

                # q=0 corresponds to unstretched return spring
                "springlength":
                    "0",
            },
        )

        ET.SubElement(
            fixed,
            "joint",
            {
                "joint":
                    joint_name,

                "coef":
                    f"{RETURN_R:.12g}",
            },
        )


    try:
        ET.indent(
            tree,
            space="  ",
        )
    except AttributeError:
        pass


    tree.write(
        OUTPUT,
        encoding="utf-8",
        xml_declaration=True,
    )


    # ============================================================
    # 3. Compile and numerically verify.
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


    ktheta = (
        RETURN_K
        *
        RETURN_R
        *
        RETURN_R
    )


    print()
    print("=" * 112)
    print("M2B PHYSICAL PARAMETERS")
    print("=" * 112)

    print(
        f"return radius        = "
        f"{RETURN_R * 1000:.3f} mm"
    )

    print(
        f"linear stiffness     = "
        f"{RETURN_K:.3f} N/m"
    )

    print(
        f"equiv joint stiffness= "
        f"{ktheta:.6f} Nm/rad"
    )


    hardware_deg = [
        14.5355,
        23.8110,
        9.3270,
    ]


    print()
    print(
        "Expected return torque at hardware target:"
    )


    for (
        (label, joint_name, tendon_name),
        deg
    ) in zip(
        JOINTS,
        hardware_deg,
    ):

        q = math.radians(
            deg
        )

        extension = (
            RETURN_R * q
        )

        force = (
            RETURN_K
            * extension
        )

        tau = (
            -RETURN_R
            * force
        )


        tid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            tendon_name,
        )

        if tid < 0:

            raise RuntimeError(
                f"Missing tendon {tendon_name}"
            )


        print(
            f"  {label}: "
            f"q={deg:6.3f} deg   "
            f"extension={extension*1000:6.3f} mm   "
            f"F={force:6.3f} N   "
            f"tau={tau:+.6f} Nm"
        )


    print()
    print("=" * 112)
    print("FINAL MODEL COMPILED SUCCESSFULLY")
    print("=" * 112)

    print()
    print(
        "Architecture:"
    )

    print(
        "  motor flexor        : unchanged"
    )

    print(
        "  flexor CAD routing  : unchanged"
    )

    print(
        "  MCP return          : local elastic tendon"
    )

    print(
        "  PIP return          : local elastic tendon"
    )

    print(
        "  DIP return          : local elastic tendon"
    )

    print(
        "  joint stiffness=    : 0 / 0 / 0"
    )

    print()
    print(
        "V35M2B ready."
    )


if __name__ == "__main__":
    main()
