#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as base
import compare_index_configdep_replay_v6 as ab
import scan_and_view_index_contact_friction_v18 as viewer_tools


# ============================================================
# RH8D INDEX — PIP PASSIVE ABLATION V22
#
# Diagnostic / causal experiment.
#
# V21 showed:
#
#   tendon PIP torque       ~ +0.012 Nm
#   return spring           ~ -0.005 Nm
#   dry friction bound      ~  0.006 Nm
#
# Thus the current passive model can consume nearly all of the
# available PIP flexion torque.
#
# V22 asks:
#
#   Is the existing tendon transmission capable of wrapping
#   the PIP if those passive resistances are removed?
#
# NOT parameter identification.
#
# FROZEN:
#   - V5 object registration
#   - bottle position
#   - constant 5/4/1 tendon transmission
#   - motor dynamics
#   - encoder/spool scale
#   - contact mu = 0.05
#   - contact stiffness = 1000 N/m
#   - collision geometry
#   - PIP damping
#
# Only PIP:
#   stiffness
#   frictionloss
#
# are ablated.
# ============================================================


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()

MJCF_DIR = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "pip_passive_ablation_v22"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY = (
    RESULT_DIR
    / "summary.csv"
)

SELECTED_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_pip_passive_v22_diagnostic.xml"
)


# ============================================================
# FROZEN PHYSICS
# ============================================================

SCALE = 5.10e-6
TENDON_K = 2100.0

CONTACT_MU = 0.05
CONTACT_K = 1000.0


PIP = "Index_Middle--Index_Proximal"


FREE_LIMIT_EXTRA = 1.0


# ============================================================
# COUNTERFACTUALS
# ============================================================

CASES = [

    {
        "name":
            "BASELINE",

        "spring_scale":
            1.0,

        "friction_scale":
            1.0,
    },

    {
        "name":
            "NO_PIP_FRICTION",

        "spring_scale":
            1.0,

        "friction_scale":
            0.0,
    },

    {
        "name":
            "NO_PIP_SPRING",

        "spring_scale":
            0.0,

        "friction_scale":
            1.0,
    },

    {
        "name":
            "NO_PIP_SPRING_OR_FRICTION",

        "spring_scale":
            0.0,

        "friction_scale":
            0.0,
    },
]


# ============================================================
# TEMPLATE
# ============================================================

def create_template(
    output,
):

    ab.replace_index_with_constant_tendon(
        ab.V5_SOURCE,
        output,
    )


def pip_properties(
    model_path,
):

    model = mujoco.MjModel.from_xml_path(
        str(
            model_path
        )
    )


    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        PIP,
    )


    if jid < 0:

        raise RuntimeError(
            f"Missing PIP joint: {PIP}"
        )


    dof = int(
        model.jnt_dofadr[
            jid
        ]
    )


    return {

        "stiffness":
            float(
                model.jnt_stiffness[
                    jid
                ]
            ),

        "frictionloss":
            float(
                model.dof_frictionloss[
                    dof
                ]
            ),

        "damping":
            float(
                model.dof_damping[
                    dof
                ]
            ),
    }


# ============================================================
# BUILD COUNTERFACTUAL
# ============================================================

def build_case(
    template,
    output,
    case,
    baseline_properties,
):

    tree = ET.parse(
        template
    )

    root = tree.getroot()


    found = False


    for joint in root.iter(
        "joint"
    ):

        if (
            joint.attrib.get(
                "name"
            )
            == PIP
        ):

            joint.set(
                "stiffness",
                f"{baseline_properties['stiffness'] * case['spring_scale']:.12g}",
            )


            joint.set(
                "frictionloss",
                f"{baseline_properties['frictionloss'] * case['friction_scale']:.12g}",
            )


            found = True

            break


    if not found:

        raise RuntimeError(
            "PIP joint not found in candidate XML."
        )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


    # Compile immediately.
    mujoco.MjModel.from_xml_path(
        str(
            output
        )
    )


# ============================================================
# RUN ONE CASE
# ============================================================

def run_case(
    case,
    template,
    baseline_properties,
    free_bag,
    contact_bag,
):

    label = (
        case[
            "name"
        ].lower()
    )


    contact_xml = (
        MJCF_DIR
        /
        f"tmp_v22_{label}_contact.xml"
    )


    free_xml = (
        MJCF_DIR
        /
        f"tmp_v22_{label}_free.xml"
    )


    build_case(
        template,
        contact_xml,
        case,
        baseline_properties,
    )


    ab.make_free_model(
        contact_xml,
        free_xml,
    )


    compiled = pip_properties(
        contact_xml
    )


    old_scale = (
        base.METERS_PER_TICK
    )

    old_tendon = (
        base.TENDON_STIFFNESS
    )

    old_mu = (
        base.CONTACT_MU
    )

    old_contact_k = (
        base.CONTACT_STIFFNESS
    )


    try:

        base.METERS_PER_TICK = (
            SCALE
        )

        base.TENDON_STIFFNESS = (
            TENDON_K
        )

        base.CONTACT_MU = (
            CONTACT_MU
        )

        base.CONTACT_STIFFNESS = (
            CONTACT_K
        )


        free_metrics, free_log = (
            base.run_free(
                free_xml,
                free_bag,
            )
        )


        contact_metrics, contact_log = (
            base.run_contact(
                contact_xml,
                contact_bag,
            )
        )


    finally:

        base.METERS_PER_TICK = (
            old_scale
        )

        base.TENDON_STIFFNESS = (
            old_tendon
        )

        base.CONTACT_MU = (
            old_mu
        )

        base.CONTACT_STIFFNESS = (
            old_contact_k
        )


    (
        RESULT_DIR
        /
        f"{label}_free.txt"
    ).write_text(
        free_log
    )


    (
        RESULT_DIR
        /
        f"{label}_contact.txt"
    ).write_text(
        contact_log
    )


    pose = np.asarray(
        contact_metrics[
            "pose"
        ],
        dtype=float,
    )


    result = {

        "name":
            case[
                "name"
            ],

        "pip_stiffness":
            compiled[
                "stiffness"
            ],

        "pip_frictionloss":
            compiled[
                "frictionloss"
            ],

        "pip_damping":
            compiled[
                "damping"
            ],

        "free_mcp":
            free_metrics[
                "mcp_rmse"
            ],

        "free_pip":
            free_metrics[
                "pip_rmse"
            ],

        "free_combined":
            free_metrics[
                "combined"
            ],

        "contact_mcp":
            pose[
                0
            ],

        "contact_pip":
            pose[
                1
            ],

        "contact_dip":
            pose[
                2
            ],

        "contact_rmse":
            contact_metrics[
                "rmse"
            ],
    }


    try:
        contact_xml.unlink()
    except FileNotFoundError:
        pass


    try:
        free_xml.unlink()
    except FileNotFoundError:
        pass


    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "free_bag",
        type=Path,
    )


    parser.add_argument(
        "contact_bag",
        type=Path,
    )


    parser.add_argument(
        "--no-viewer",
        action="store_true",
    )


    args = parser.parse_args()


    free_bag = (
        args.free_bag
        .expanduser()
        .resolve()
    )


    contact_bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )


    if not free_bag.exists():

        raise RuntimeError(
            f"Free bag missing:\n"
            f"{free_bag}"
        )


    if not contact_bag.exists():

        raise RuntimeError(
            f"Contact bag missing:\n"
            f"{contact_bag}"
        )


    # ========================================================
    # FREEZE FIXTURE
    # ========================================================

    ab.patch_replay_modules()


    # ========================================================
    # TEMPLATE + TRUE BASELINE PARAMETERS
    # ========================================================

    template = (
        MJCF_DIR
        /
        "tmp_v22_template.xml"
    )


    create_template(
        template
    )


    baseline_properties = (
        pip_properties(
            template
        )
    )


    print()
    print("=" * 118)
    print(
        "RH8D INDEX — PIP PASSIVE "
        "COUNTERFACTUAL ABLATION V22"
    )
    print("=" * 118)


    print()
    print(
        "Bottle/object registration = FROZEN V5"
    )


    print(
        f"Contact mu = "
        f"{CONTACT_MU:.3f}"
    )

    print(
        f"Contact stiffness = "
        f"{CONTACT_K:.1f} N/m"
    )


    print()
    print(
        "Compiled baseline PIP mechanics:"
    )

    print(
        f"  stiffness    = "
        f"{baseline_properties['stiffness']:.6f} Nm/rad"
    )

    print(
        f"  frictionloss = "
        f"{baseline_properties['frictionloss']:.6f} Nm"
    )

    print(
        f"  damping      = "
        f"{baseline_properties['damping']:.6f} Nms/rad"
    )


    print()
    print(
        "Constant tendon remains active:"
    )

    print(
        "  MCP = 5 mm/rad"
    )

    print(
        "  PIP = 4 mm/rad"
    )

    print(
        "  DIP = 1 mm/rad"
    )


    results = []


    for index, case in enumerate(
        CASES,
        start=1,
    ):

        print()
        print("=" * 118)

        print(
            f"[{index:02d}/{len(CASES):02d}] "
            f"{case['name']}"
        )

        print("=" * 118)


        result = run_case(
            case,
            template,
            baseline_properties,
            free_bag,
            contact_bag,
        )


        results.append(
            result
        )


        print()
        print(
            "Compiled PIP:"
        )

        print(
            f"  stiffness = "
            f"{result['pip_stiffness']:.6f}"
        )

        print(
            f"  frictionloss = "
            f"{result['pip_frictionloss']:.6f}"
        )

        print(
            f"  damping = "
            f"{result['pip_damping']:.6f}"
        )


        print()
        print(
            "Free-space MCP/PIP/combined RMSE:"
        )

        print(
            f"  "
            f"{result['free_mcp']:.3f} / "
            f"{result['free_pip']:.3f} / "
            f"{result['free_combined']:.3f} deg"
        )


        print()
        print(
            "Contact MCP/PIP/DIP:"
        )

        print(
            f"  "
            f"{result['contact_mcp']:.3f} / "
            f"{result['contact_pip']:.3f} / "
            f"{result['contact_dip']:.3f} deg"
        )


        print(
            f"Contact RMSE = "
            f"{result['contact_rmse']:.3f} deg"
        )


    # ========================================================
    # SAVE
    # ========================================================

    with open(
        SUMMARY,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # ========================================================
    # CAUSAL EFFECT
    # ========================================================

    baseline = next(
        r
        for r in results
        if (
            r[
                "name"
            ]
            == "BASELINE"
        )
    )


    zero_both = next(
        r
        for r in results
        if (
            r[
                "name"
            ]
            == "NO_PIP_SPRING_OR_FRICTION"
        )
    )


    print()
    print("=" * 118)
    print(
        "V22 MAXIMUM PASSIVE-REMOVAL EFFECT"
    )
    print("=" * 118)


    print()
    print(
        "BASELINE contact:"
    )

    print(
        "  "
        f"{baseline['contact_mcp']:.3f} / "
        f"{baseline['contact_pip']:.3f} / "
        f"{baseline['contact_dip']:.3f} deg"
    )


    print()
    print(
        "NO PIP SPRING + NO PIP FRICTION:"
    )

    print(
        "  "
        f"{zero_both['contact_mcp']:.3f} / "
        f"{zero_both['contact_pip']:.3f} / "
        f"{zero_both['contact_dip']:.3f} deg"
    )


    pip_gain = (
        zero_both[
            "contact_pip"
        ]
        -
        baseline[
            "contact_pip"
        ]
    )


    print()
    print(
        f"Maximum PIP gain from removing "
        f"both passive terms = "
        f"{pip_gain:+.3f} deg"
    )


    print(
        f"Free-space penalty = "
        f"{zero_both['free_combined'] - baseline['free_combined']:+.3f} deg"
    )


    # ========================================================
    # FREE-SPACE-PRESERVING BEST
    # ========================================================

    free_limit = (
        baseline[
            "free_combined"
        ]
        +
        FREE_LIMIT_EXTRA
    )


    preserving = [
        r
        for r in results
        if (
            r[
                "free_combined"
            ]
            <= free_limit
        )
    ]


    selected = min(
        preserving,
        key=lambda r:
            r[
                "contact_rmse"
            ],
    )


    print()
    print("=" * 118)
    print(
        "V22 BEST FREE-SPACE-PRESERVING COUNTERFACTUAL"
    )
    print("=" * 118)


    print()
    print(
        f"Free-space gate = "
        f"{free_limit:.3f} deg"
    )


    print(
        f"Selected = "
        f"{selected['name']}"
    )


    print(
        f"Free combined RMSE = "
        f"{selected['free_combined']:.3f} deg"
    )


    print(
        "Contact MCP/PIP/DIP = "
        f"{selected['contact_mcp']:.3f} / "
        f"{selected['contact_pip']:.3f} / "
        f"{selected['contact_dip']:.3f} deg"
    )


    print(
        f"Contact RMSE = "
        f"{selected['contact_rmse']:.3f} deg"
    )


    # ========================================================
    # STRUCTURAL DECISION
    # ========================================================

    print()
    print("=" * 118)
    print(
        "V22 STRUCTURAL DECISION"
    )
    print("=" * 118)


    if pip_gain >= 10.0:

        print()
        print(
            "PIP passive resistance is a MAJOR structural "
            "bottleneck."
        )

        print(
            "Removing the current return spring + Coulomb "
            "friction releases substantial PIP wrapping."
        )

        print()
        print(
            "Do NOT keep zero resistance as the final model."
        )

        print(
            "NEXT: replace the constant PIP passive model "
            "with a physically motivated hysteretic/"
            "breakaway return-element model identified from "
            "the closing/opening hardware experiment."
        )


    elif pip_gain >= 4.0:

        print()
        print(
            "PIP passive resistance contributes materially, "
            "but does not fully explain the hardware wrapping."
        )

        print()
        print(
            "NEXT: retain state-dependent passive mechanics "
            "and simultaneously revisit tendon load "
            "redistribution."
        )


    else:

        print()
        print(
            "Even completely removing the PIP return spring "
            "and dry friction does NOT generate substantial "
            "PIP wrapping."
        )

        print()
        print(
            "The constant 5/4/1 transmission is therefore "
            "the dominant structural limitation."
        )

        print(
            "NEXT: stop tuning PIP passives and construct a "
            "load/history-dependent underactuated tendon "
            "transmission."
        )


    # ========================================================
    # REBUILD SELECTED MODEL FOR VIEWER
    # ========================================================

    selected_case = next(
        c
        for c in CASES
        if (
            c[
                "name"
            ]
            == selected[
                "name"
            ]
        )
    )


    build_case(
        template,
        SELECTED_MODEL,
        selected_case,
        baseline_properties,
    )


    print()
    print(
        "Diagnostic viewer model:"
    )

    print(
        SELECTED_MODEL
    )


    print()
    print(
        "Summary:"
    )

    print(
        SUMMARY
    )


    # ========================================================
    # AUTOMATIC VIEWER
    # ========================================================

    if not args.no_viewer:

        print()
        print("=" * 118)
        print(
            "V22 AUTOMATIC BEST "
            "FREE-SPACE-PRESERVING REPLAY"
        )
        print("=" * 118)


        print()
        print(
            f"Viewing: "
            f"{selected['name']}"
        )

        print(
            "Bottle position remains unchanged."
        )

        print(
            "No custom viewer width/height is requested."
        )


        old_mu = (
            viewer_tools.base.CONTACT_MU
        )

        old_k = (
            viewer_tools.base.CONTACT_STIFFNESS
        )


        try:

            viewer_tools.base.CONTACT_MU = (
                CONTACT_MU
            )

            viewer_tools.base.CONTACT_STIFFNESS = (
                CONTACT_K
            )


            viewer_tools.visual_replay(
                SELECTED_MODEL,
                contact_bag,
            )


        finally:

            viewer_tools.base.CONTACT_MU = (
                old_mu
            )

            viewer_tools.base.CONTACT_STIFFNESS = (
                old_k
            )


    # ========================================================
    # CLEANUP TEMPLATE ONLY
    # ========================================================

    try:
        template.unlink()
    except FileNotFoundError:
        pass


    print()
    print("=" * 118)
    print(
        "V22 COMPLETE"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
