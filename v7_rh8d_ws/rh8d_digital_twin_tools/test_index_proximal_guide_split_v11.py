#!/usr/bin/env python3

from pathlib import Path
import csv
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as base
import compare_index_configdep_replay_v6 as ab


# ============================================================
# RH8D INDEX — PROXIMAL GUIDE SPLIT V11
#
# PURPOSE
#
# Introduce ONE internal tendon-slip state that can later
# represent guide friction / local tendon sliding.
#
# V11 itself has ZERO slip friction.
#
# Therefore it is only an architectural equivalence test:
#
# trusted constant tendon
#
#       vs
#
# motor segment -> free slip DOF -> finger segment
#
# If V10 approximately preserves the trusted free-space model,
# we can then identify friction/hysteresis on this internal DOF.
# ============================================================


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()

MJCF_DIR = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

SOURCE = (
    MJCF_DIR
    / "ur5e_rh8d_index_configdep_geom_v5.xml"
)

CONTROL_CONTACT = (
    MJCF_DIR
    / "tmp_v11_constant_contact.xml"
)

CONTROL_FREE = (
    MJCF_DIR
    / "tmp_v11_constant_free.xml"
)

SPLIT_CONTACT = (
    MJCF_DIR
    / "ur5e_rh8d_index_proximal_guide_split_v11.xml"
)

SPLIT_FREE = (
    MJCF_DIR
    / "tmp_v11_split_free.xml"
)

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "proximal_guide_split_v11"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY = (
    RESULT_DIR
    / "summary.csv"
)


# ============================================================
# NAMES
# ============================================================

SPOOL = "index_motor_spool"

SLIP_BODY = "IndexTendonSlipDummy"
SLIP_JOINT = "index_tendon_slip"

MOTOR_TENDON = "idx_motor_segment"
FINGER_TENDON = "idx_flex_spatial"

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"


# ============================================================
# TRUSTED INTERFACE
# ============================================================

SCALE = 5.10e-6

OLD_EFFECTIVE_K = 2100.0


# ============================================================
# SPLIT TENDON PARAMETERS
#
# Motor-side segment is deliberately much stiffer than the
# finger-side segment.
#
# Kfinger is slightly above 2100 so the two series springs
# approximately reproduce the original 2100-N/m equivalent:
#
#   1/Keq = 1/Kmotor + 1/Kfinger
#
# Kmotor = 50000
# Kfinger ~= 2192
# Keq ~= 2100
# ============================================================

K_MOTOR = 50000.0
K_FINGER = 2192.0

D_MOTOR = 20.0
D_FINGER = 2.0


# V10 MUST have zero hysteretic friction.
SLIP_FRICTION = 0.0

SLIP_DAMPING = 0.001

SLIP_ARMATURE = 0.10


# ============================================================
# HARDWARE TARGET — comparison only
# ============================================================

HW = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


# ============================================================
# XML HELPERS
# ============================================================

def remove_existing_index_tendon(
    root,
):

    section = root.find(
        "tendon"
    )

    if section is None:

        raise RuntimeError(
            "<tendon> section missing."
        )


    for element in list(
        section
    ):

        if (
            element.attrib.get("name")
            == FINGER_TENDON
        ):

            section.remove(
                element
            )


    return section


def add_slip_body(
    root,
):

    worldbody = root.find(
        "worldbody"
    )

    if worldbody is None:

        raise RuntimeError(
            "<worldbody> missing."
        )


    # Remove stale copy if script is rerun.
    for body in list(
        worldbody.findall(
            "body"
        )
    ):

        if (
            body.attrib.get("name")
            == SLIP_BODY
        ):

            worldbody.remove(
                body
            )


    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name":
                SLIP_BODY,

            "pos":
                "0 0 0",
        },
    )


    ET.SubElement(
        body,
        "inertial",
        {
            "pos":
                "0 0 0",

            "mass":
                "0.001",

            "diaginertia":
                "1e-7 1e-7 1e-7",
        },
    )


    ET.SubElement(
        body,
        "joint",
        {
            "name":
                SLIP_JOINT,

            "type":
                "slide",

            "axis":
                "1 0 0",

            "limited":
                "true",

            "range":
                "-0.010 0.030",

            "damping":
                f"{SLIP_DAMPING:.10g}",

            "frictionloss":
                f"{SLIP_FRICTION:.10g}",

            "armature":
                f"{SLIP_ARMATURE:.10g}",
        },
    )


def add_split_tendons(
    section,
):

    # ========================================================
    # MOTOR / PROXIMAL SEGMENT
    #
    # L1 =
    #
    #   + spool
    #   - 0.005 MCP
    #   - slip
    #
    # This represents the tendon section from the motor,
    # through the proximal routing / MCP region, up to the
    # lumped guide-friction state.
    # ========================================================

    motor = ET.SubElement(
        section,
        "fixed",
        {
            "name":
                MOTOR_TENDON,

            "stiffness":
                f"{K_MOTOR:.10g}",

            "damping":
                f"{D_MOTOR:.10g}",

            "springlength":
                "-0.050 0.0",
        },
    )


    ET.SubElement(
        motor,
        "joint",
        {
            "joint":
                SPOOL,

            "coef":
                "1.0",
        },
    )


    ET.SubElement(
        motor,
        "joint",
        {
            "joint":
                MCP,

            "coef":
                "-0.005",
        },
    )


    ET.SubElement(
        motor,
        "joint",
        {
            "joint":
                SLIP_JOINT,

            "coef":
                "-1.0",
        },
    )


    # ========================================================
    # DISTAL FINGER SEGMENT
    #
    # L2 =
    #
    #   + slip
    #   - 0.004 PIP
    #   - 0.001 DIP
    #
    # Friction on the slip DOF can later allow:
    #
    #       T_proximal != T_distal
    #
    # which is exactly the capability missing from a single
    # ideal tendon.
    # ========================================================

    finger = ET.SubElement(
        section,
        "fixed",
        {
            "name":
                FINGER_TENDON,

            "stiffness":
                f"{K_FINGER:.10g}",

            "damping":
                f"{D_FINGER:.10g}",

            "springlength":
                "-0.050 0.0",
        },
    )


    ET.SubElement(
        finger,
        "joint",
        {
            "joint":
                SLIP_JOINT,

            "coef":
                "1.0",
        },
    )


    ET.SubElement(
        finger,
        "joint",
        {
            "joint":
                PIP,

            "coef":
                "-0.004",
        },
    )


    ET.SubElement(
        finger,
        "joint",
        {
            "joint":
                DIP,

            "coef":
                "-0.001",
        },
    )


def build_split_model():

    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()


    section = (
        remove_existing_index_tendon(
            root
        )
    )


    add_slip_body(
        root
    )


    add_split_tendons(
        section
    )


    tree.write(
        SPLIT_CONTACT,
        encoding="unicode",
        xml_declaration=True,
    )


    # Compile immediately.
    mujoco.MjModel.from_xml_path(
        str(
            SPLIT_CONTACT
        )
    )


def get_joint_id(
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
            f"Missing joint: {name}"
        )

    return jid


def structural_audit():

    model = mujoco.MjModel.from_xml_path(
        str(
            SPLIT_CONTACT
        )
    )

    slip_id = get_joint_id(
        model,
        SLIP_JOINT,
    )


    slip_dof = int(
        model.jnt_dofadr[
            slip_id
        ]
    )


    motor_tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        MOTOR_TENDON,
    )

    finger_tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        FINGER_TENDON,
    )


    if (
        motor_tid < 0
        or
        finger_tid < 0
    ):

        raise RuntimeError(
            "Split tendons were not created."
        )


    equivalent_k = (
        K_MOTOR
        * K_FINGER
        /
        (
            K_MOTOR
            + K_FINGER
        )
    )


    print()
    print("=" * 110)
    print(
        "SPLIT-TENDON STRUCTURAL AUDIT"
    )
    print("=" * 110)


    print()
    print(
        f"Motor segment stiffness  = "
        f"{K_MOTOR:.1f} N/m"
    )

    print(
        f"Finger segment stiffness = "
        f"{K_FINGER:.1f} N/m"
    )

    print(
        f"Series-equivalent k      = "
        f"{equivalent_k:.2f} N/m"
    )

    print(
        f"Trusted old k            = "
        f"{OLD_EFFECTIVE_K:.2f} N/m"
    )


    print()
    print(
        "Internal slip DOF:"
    )

    print(
        f"  frictionloss = "
        f"{model.dof_frictionloss[slip_dof]:.6f} N"
    )

    print(
        f"  damping      = "
        f"{model.dof_damping[slip_dof]:.6f}"
    )

    print(
        f"  armature     = "
        f"{model.dof_armature[slip_dof]:.6f}"
    )


    print()
    print(
        "V11 friction must equal ZERO:"
    )

    print(
        f"  PASS = "
        f"{abs(model.dof_frictionloss[slip_dof]) < 1e-12}"
    )


# ============================================================
# REPLAY
# ============================================================

def run_model(
    label,
    free_model,
    contact_model,
    free_bag,
    contact_bag,
    finger_k,
):

    old_scale = (
        base.METERS_PER_TICK
    )

    old_k = (
        base.TENDON_STIFFNESS
    )


    try:

        base.METERS_PER_TICK = (
            SCALE
        )

        # The replay helper can still identify idx_flex_spatial.
        base.TENDON_STIFFNESS = (
            finger_k
        )


        free_metrics, free_log = (
            base.run_free(
                free_model,
                free_bag,
            )
        )


        contact_metrics, contact_log = (
            base.run_contact(
                contact_model,
                contact_bag,
            )
        )


    finally:

        base.METERS_PER_TICK = (
            old_scale
        )

        base.TENDON_STIFFNESS = (
            old_k
        )


    (
        RESULT_DIR
        / f"{label.lower()}_free.txt"
    ).write_text(
        free_log
    )


    (
        RESULT_DIR
        / f"{label.lower()}_contact.txt"
    ).write_text(
        contact_log
    )


    return {
        "label":
            label,

        "free":
            free_metrics,

        "contact":
            contact_metrics,
    }


def main():

    import sys


    if len(
        sys.argv
    ) != 3:

        print(
            "Usage:\n"
            " python3 test_index_split_tendon_v10.py "
            "/path/to/free_bag "
            "/path/to/contact_bag"
        )

        raise SystemExit(
            1
        )


    free_bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    contact_bag = Path(
        sys.argv[2]
    ).expanduser().resolve()


    # Make old replay scripts use frozen V5 bottle registration.
    ab.patch_replay_modules()


    # --------------------------------------------------------
    # TRUSTED CONSTANT CONTROL
    # --------------------------------------------------------

    ab.replace_index_with_constant_tendon(
        SOURCE,
        CONTROL_CONTACT,
    )


    ab.make_free_model(
        CONTROL_CONTACT,
        CONTROL_FREE,
    )


    # --------------------------------------------------------
    # SPLIT TENDON
    # --------------------------------------------------------

    build_split_model()


    ab.make_free_model(
        SPLIT_CONTACT,
        SPLIT_FREE,
    )


    structural_audit()


    print()
    print("=" * 110)
    print(
        "RUNNING TRUSTED CONSTANT CONTROL"
    )
    print("=" * 110)


    control = run_model(
        "CONSTANT",
        CONTROL_FREE,
        CONTROL_CONTACT,
        free_bag,
        contact_bag,
        OLD_EFFECTIVE_K,
    )


    print()
    print("=" * 110)
    print(
        "RUNNING ZERO-FRICTION SPLIT TENDON"
    )
    print("=" * 110)


    split = run_model(
        "SPLIT_ZERO_FRICTION",
        SPLIT_FREE,
        SPLIT_CONTACT,
        free_bag,
        contact_bag,
        K_FINGER,
    )


    # ========================================================
    # REPORT
    # ========================================================

    rows = []


    print()
    print("=" * 110)
    print(
        "V11 PROXIMAL-GUIDE EQUIVALENCE RESULTS"
    )
    print("=" * 110)


    for result in [
        control,
        split,
    ]:

        pose = np.asarray(
            result[
                "contact"
            ][
                "pose"
            ],
            dtype=float,
        )


        print()
        print(
            result[
                "label"
            ]
        )

        print(
            "-" * 70
        )


        print(
            "Free MCP/PIP/combined RMSE:"
        )

        print(
            f"  "
            f"{result['free']['mcp_rmse']:.3f} / "
            f"{result['free']['pip_rmse']:.3f} / "
            f"{result['free']['combined']:.3f} deg"
        )


        print(
            "Contact pose:"
        )

        print(
            f"  "
            f"({pose[0]:.3f}, "
            f"{pose[1]:.3f}, "
            f"{pose[2]:.3f}) deg"
        )


        print(
            f"Contact RMSE = "
            f"{result['contact']['rmse']:.3f} deg"
        )


        rows.append(
            {
                "model":
                    result[
                        "label"
                    ],

                "free_mcp":
                    result[
                        "free"
                    ][
                        "mcp_rmse"
                    ],

                "free_pip":
                    result[
                        "free"
                    ][
                        "pip_rmse"
                    ],

                "free_combined":
                    result[
                        "free"
                    ][
                        "combined"
                    ],

                "contact_mcp":
                    pose[0],

                "contact_pip":
                    pose[1],

                "contact_dip":
                    pose[2],

                "contact_rmse":
                    result[
                        "contact"
                    ][
                        "rmse"
                    ],
            }
        )


    free_delta = (
        split[
            "free"
        ][
            "combined"
        ]
        -
        control[
            "free"
        ][
            "combined"
        ]
    )


    split_pose = np.asarray(
        split[
            "contact"
        ][
            "pose"
        ]
    )


    control_pose = np.asarray(
        control[
            "contact"
        ][
            "pose"
        ]
    )


    contact_pose_delta = (
        split_pose
        -
        control_pose
    )


    print()
    print("=" * 110)
    print(
        "ZERO-FRICTION ARCHITECTURE DIFFERENCE"
    )
    print("=" * 110)


    print()
    print(
        f"Free combined RMSE delta "
        f"(split - constant) = "
        f"{free_delta:+.3f} deg"
    )


    print()
    print(
        "Contact pose delta "
        "(split - constant):"
    )

    print(
        f"  MCP = "
        f"{contact_pose_delta[0]:+.3f} deg"
    )

    print(
        f"  PIP = "
        f"{contact_pose_delta[1]:+.3f} deg"
    )

    print(
        f"  DIP = "
        f"{contact_pose_delta[2]:+.3f} deg"
    )


    # --------------------------------------------------------
    # Equivalence gate
    # --------------------------------------------------------

    free_ok = (
        abs(
            free_delta
        )
        <= 1.5
    )


    contact_ok = (
        np.max(
            np.abs(
                contact_pose_delta
            )
        )
        <= 3.0
    )


    passed = (
        free_ok
        and
        contact_ok
    )


    print()
    print(
        f"Free equivalence <= 1.5 deg : "
        f"{free_ok}"
    )

    print(
        f"Contact pose equivalence <= 3 deg/joint : "
        f"{contact_ok}"
    )


    print()
    print("=" * 110)


    if passed:

        print(
            "RESULT: PASS — the zero-friction split-tendon "
            "architecture preserves the trusted model closely "
            "enough to identify guide friction next."
        )

    else:

        print(
            "RESULT: DO NOT ADD FRICTION YET."
        )

        print(
            "First tune ONLY the numerical series-equivalence "
            "of the two tendon segments; do not use hardware "
            "contact data for that correction."
        )


    print()
    print("=" * 110)


    with open(
        SUMMARY,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    print()
    print(
        "Split model:"
    )

    print(
        SPLIT_CONTACT
    )

    print()
    print(
        "Summary:"
    )

    print(
        SUMMARY
    )


    # Keep SPLIT_CONTACT because it is our V10 model.
    # Remove only temporary models.

    for path in [
        CONTROL_CONTACT,
        CONTROL_FREE,
        SPLIT_FREE,
    ]:

        try:
            path.unlink()

        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
