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
# RH8D INDEX — PROXIMAL GUIDE FRICTION V12
#
# V11 proved that the following zero-friction topology is
# numerically equivalent to the trusted constant tendon:
#
# spool -> MCP -> [slip/guide] -> PIP -> DIP
#
# V12 changes ONE parameter only:
#
#     frictionloss(index_tendon_slip)
#
# Units: N because index_tendon_slip is a slide DOF.
#
# Frozen:
#   motor model
#   encoder/spool scale
#   segment stiffnesses
#   joint passive mechanics
#   MCP/PIP/DIP transmission coefficients
#   V5 fixture registration
#   collision geometry
#
# No contact-dependent switching is used.
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
    / "ur5e_rh8d_index_proximal_guide_split_v11.xml"
)


BEST_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_proximal_guide_friction_v12_candidate.xml"
)


RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "proximal_guide_friction_v12"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUMMARY = (
    RESULT_DIR
    / "summary.csv"
)


SLIP_JOINT = "index_tendon_slip"

FINGER_TENDON = "idx_flex_spatial"


# ============================================================
# TRUSTED PARAMETERS
# ============================================================

SCALE = 5.10e-6

K_FINGER = 2192.0


# ============================================================
# GUIDE FRICTION CANDIDATES
#
# Loads previously observed in the trusted model are roughly
# 2–5 N, so this brackets the mechanically relevant range.
# ============================================================

FRICTIONS_N = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    2.50,
    3.00,
    4.00,
]


# ============================================================
# VALIDATION
# ============================================================

TRUSTED_FREE = 5.085

FREE_LIMIT = (
    TRUSTED_FREE
    + 1.0
)


HW_POSE = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


# ============================================================
# XML
# ============================================================

def build_candidate(
    friction,
    output,
):

    tree = ET.parse(
        SOURCE
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
            == SLIP_JOINT
        ):

            joint.set(
                "frictionloss",
                f"{friction:.10g}",
            )

            found = True
            break


    if not found:

        raise RuntimeError(
            f"Missing joint: {SLIP_JOINT}"
        )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


    # Immediate structural compile.
    mujoco.MjModel.from_xml_path(
        str(
            output
        )
    )


def slip_friction_from_model(
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
        SLIP_JOINT,
    )


    if jid < 0:

        raise RuntimeError(
            "Slip joint missing after compile."
        )


    dof = int(
        model.jnt_dofadr[
            jid
        ]
    )


    return float(
        model.dof_frictionloss[
            dof
        ]
    )


# ============================================================
# RUN ONE CANDIDATE
# ============================================================

def run_candidate(
    friction,
    free_bag,
    contact_bag,
):

    label = (
        f"f{friction:.2f}"
        .replace(
            ".",
            "p",
        )
    )


    contact_xml = (
        MJCF_DIR
        /
        f"tmp_v12_contact_{label}.xml"
    )


    free_xml = (
        MJCF_DIR
        /
        f"tmp_v12_free_{label}.xml"
    )


    build_candidate(
        friction,
        contact_xml,
    )


    ab.make_free_model(
        contact_xml,
        free_xml,
    )


    compiled_friction = (
        slip_friction_from_model(
            contact_xml
        )
    )


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

        # run_free/run_contact may patch the named downstream
        # tendon stiffness. Keep it at the successful V11 value.
        base.TENDON_STIFFNESS = (
            K_FINGER
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
            old_k
        )


    (
        RESULT_DIR
        / f"{label}_free.txt"
    ).write_text(
        free_log
    )


    (
        RESULT_DIR
        / f"{label}_contact.txt"
    ).write_text(
        contact_log
    )


    try:
        contact_xml.unlink()
    except FileNotFoundError:
        pass


    try:
        free_xml.unlink()
    except FileNotFoundError:
        pass


    pose = np.asarray(
        contact_metrics[
            "pose"
        ],
        dtype=float,
    )


    return {

        "friction_N":
            compiled_friction,

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
            pose[0],

        "contact_pip":
            pose[1],

        "contact_dip":
            pose[2],

        "contact_rmse":
            contact_metrics[
                "rmse"
            ],

        "middle_duty":
            contact_metrics.get(
                "middle_duty",
                None,
            ),

        "dmcp":
            contact_metrics.get(
                "dmcp",
                None,
            ),

        "dpip":
            contact_metrics.get(
                "dpip",
                None,
            ),

        "ddip":
            contact_metrics.get(
                "ddip",
                None,
            ),
    }


# ============================================================
# RENDER TWO COMPARISON IMAGES
# ============================================================

def render_comparison(
    control,
    selected,
):

    old_result_dir = (
        ab.RESULT_DIR
    )

    old_scale = (
        ab.METERS_PER_TICK
    )


    control_model = (
        MJCF_DIR
        / "tmp_v12_render_control.xml"
    )


    try:

        ab.RESULT_DIR = (
            RESULT_DIR
        )

        ab.METERS_PER_TICK = (
            SCALE
        )


        # ----------------------------------------------------
        # ZERO-FRICTION V11
        # ----------------------------------------------------

        build_candidate(
            0.0,
            control_model,
        )


        ab.render_hold(
            control_model,
            np.array(
                [
                    control[
                        "contact_mcp"
                    ],
                    control[
                        "contact_pip"
                    ],
                    control[
                        "contact_dip"
                    ],
                ]
            ),
            "V11_ZERO_FRICTION",
            control[
                "contact_rmse"
            ],
        )


        # ----------------------------------------------------
        # BEST V12 FRICTION CANDIDATE
        # ----------------------------------------------------

        ab.render_hold(
            BEST_MODEL,
            np.array(
                [
                    selected[
                        "contact_mcp"
                    ],
                    selected[
                        "contact_pip"
                    ],
                    selected[
                        "contact_dip"
                    ],
                ]
            ),
            "V12_BEST_GUIDE_FRICTION",
            selected[
                "contact_rmse"
            ],
        )


    finally:

        ab.RESULT_DIR = (
            old_result_dir
        )

        ab.METERS_PER_TICK = (
            old_scale
        )


        try:
            control_model.unlink()
        except FileNotFoundError:
            pass


# ============================================================
# MAIN
# ============================================================

def main():

    import sys


    if len(
        sys.argv
    ) != 3:

        print(
            "Usage:\n"
            " python3 "
            "scan_index_proximal_guide_friction_v12.py "
            "/path/to/free_bag "
            "/path/to/contact_bag"
        )

        raise SystemExit(
            1
        )


    free_bag = Path(
        sys.argv[
            1
        ]
    ).expanduser().resolve()


    contact_bag = Path(
        sys.argv[
            2
        ]
    ).expanduser().resolve()


    if not SOURCE.exists():

        raise RuntimeError(
            f"Missing V11 model:\n"
            f"{SOURCE}"
        )


    # Existing contact replay helpers contain the previous
    # fixture constant. Force them to the frozen V5 position.
    ab.patch_replay_modules()


    print()
    print("=" * 118)
    print(
        "RH8D INDEX — PROXIMAL GUIDE "
        "FRICTION IDENTIFICATION V12"
    )
    print("=" * 118)


    print()
    print(
        "Frozen architecture:"
    )

    print(
        "  spool -> MCP -> guide/slip -> PIP -> DIP"
    )

    print(
        f"  encoder scale = "
        f"{SCALE*1e6:.2f} um/tick"
    )

    print(
        "  segment stiffnesses unchanged from V11"
    )

    print(
        "  passive joints unchanged"
    )

    print(
        "  V5 fixture unchanged"
    )


    print()
    print(
        "Guide-friction candidates [N]:"
    )

    print(
        FRICTIONS_N
    )


    results = []


    for i, friction in enumerate(
        FRICTIONS_N,
        start=1,
    ):

        print()
        print(
            f"[{i:02d}/"
            f"{len(FRICTIONS_N):02d}] "
            f"guide friction = "
            f"{friction:.2f} N"
        )


        result = run_candidate(
            friction,
            free_bag,
            contact_bag,
        )


        results.append(
            result
        )


        print(
            f"  free = "
            f"{result['free_combined']:.3f} deg"
        )

        print(
            "  contact pose = "
            f"("
            f"{result['contact_mcp']:.3f}, "
            f"{result['contact_pip']:.3f}, "
            f"{result['contact_dip']:.3f}"
            f") deg"
        )

        print(
            f"  contact RMSE = "
            f"{result['contact_rmse']:.3f} deg"
        )


        if (
            result[
                "middle_duty"
            ]
            is not None
        ):

            print(
                f"  middle duty = "
                f"{result['middle_duty']:.1f}%"
            )


        if (
            result[
                "dpip"
            ]
            is not None
        ):

            print(
                "  middle->tip redistribution = "
                f"("
                f"{result['dmcp']:+.3f}, "
                f"{result['dpip']:+.3f}, "
                f"{result['ddip']:+.3f}"
                f") deg"
            )


    # ========================================================
    # SAVE ALL RESULTS
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
    # ZERO-FRICTION CONTROL
    # ========================================================

    control = min(
        results,
        key=lambda r:
            abs(
                r[
                    "friction_N"
                ]
            ),
    )


    print()
    print("=" * 118)
    print(
        "V11 ZERO-FRICTION CONTROL"
    )
    print("=" * 118)


    print()
    print(
        f"free MCP/PIP/combined = "
        f"{control['free_mcp']:.3f} / "
        f"{control['free_pip']:.3f} / "
        f"{control['free_combined']:.3f} deg"
    )


    print(
        "contact pose = "
        f"("
        f"{control['contact_mcp']:.3f}, "
        f"{control['contact_pip']:.3f}, "
        f"{control['contact_dip']:.3f}"
        f") deg"
    )


    print(
        f"contact RMSE = "
        f"{control['contact_rmse']:.3f} deg"
    )


    # ========================================================
    # FREE-SPACE-PRESERVING RESULTS
    # ========================================================

    preserving = [
        r
        for r in results
        if (
            r[
                "free_combined"
            ]
            <= FREE_LIMIT
        )
    ]


    preserving.sort(
        key=lambda r:
            (
                r[
                    "contact_rmse"
                ],
                r[
                    "free_combined"
                ],
            )
    )


    print()
    print("=" * 118)
    print(
        "FREE-SPACE-PRESERVING GUIDE-FRICTION CANDIDATES"
    )
    print(
        f"(free combined <= "
        f"{FREE_LIMIT:.3f} deg)"
    )
    print("=" * 118)


    if not preserving:

        print()
        print(
            "NONE."
        )


    else:

        for rank, r in enumerate(
            preserving,
            start=1,
        ):

            print()
            print(
                f"#{rank} "
                f"friction={r['friction_N']:.2f} N"
            )

            print(
                f"  free MCP/PIP/combined = "
                f"{r['free_mcp']:.3f} / "
                f"{r['free_pip']:.3f} / "
                f"{r['free_combined']:.3f} deg"
            )

            print(
                "  contact pose = "
                f"("
                f"{r['contact_mcp']:.3f}, "
                f"{r['contact_pip']:.3f}, "
                f"{r['contact_dip']:.3f}"
                f") deg"
            )

            print(
                f"  contact RMSE = "
                f"{r['contact_rmse']:.3f} deg"
            )


    # ========================================================
    # BEST CONTACT
    # ========================================================

    best_contact = min(
        results,
        key=lambda r:
            r[
                "contact_rmse"
            ],
    )


    print()
    print("=" * 118)
    print(
        "BEST CONTACT RESULT REGARDLESS OF FREE-SPACE"
    )
    print("=" * 118)


    print()
    print(
        f"friction = "
        f"{best_contact['friction_N']:.2f} N"
    )

    print(
        f"free = "
        f"{best_contact['free_combined']:.3f} deg"
    )

    print(
        "contact pose = "
        f"("
        f"{best_contact['contact_mcp']:.3f}, "
        f"{best_contact['contact_pip']:.3f}, "
        f"{best_contact['contact_dip']:.3f}"
        f") deg"
    )

    print(
        f"contact RMSE = "
        f"{best_contact['contact_rmse']:.3f} deg"
    )


    # ========================================================
    # SELECT FAIR CANDIDATE
    # ========================================================

    selected = (
        preserving[
            0
        ]
        if preserving
        else best_contact
    )


    build_candidate(
        selected[
            "friction_N"
        ],
        BEST_MODEL,
    )


    # ========================================================
    # DIRECT EFFECT
    # ========================================================

    control_pose = np.array(
        [
            control[
                "contact_mcp"
            ],
            control[
                "contact_pip"
            ],
            control[
                "contact_dip"
            ],
        ]
    )


    selected_pose = np.array(
        [
            selected[
                "contact_mcp"
            ],
            selected[
                "contact_pip"
            ],
            selected[
                "contact_dip"
            ],
        ]
    )


    pose_delta = (
        selected_pose
        - control_pose
    )


    pip_error_control = abs(
        control_pose[
            1
        ]
        -
        HW_POSE[
            1
        ]
    )


    pip_error_selected = abs(
        selected_pose[
            1
        ]
        -
        HW_POSE[
            1
        ]
    )


    pip_error_reduction = (
        pip_error_control
        -
        pip_error_selected
    )


    print()
    print("=" * 118)
    print(
        "SELECTED V12 GUIDE-FRICTION CANDIDATE"
    )
    print("=" * 118)


    print()
    print(
        f"friction = "
        f"{selected['friction_N']:.3f} N"
    )


    print(
        f"free combined RMSE = "
        f"{selected['free_combined']:.3f} deg"
    )


    print()
    print(
        "contact prediction:"
    )

    print(
        f"  MCP = "
        f"{selected['contact_mcp']:.3f} deg"
    )

    print(
        f"  PIP = "
        f"{selected['contact_pip']:.3f} deg"
    )

    print(
        f"  DIP = "
        f"{selected['contact_dip']:.3f} deg"
    )


    print()
    print(
        "Change from zero-friction V11:"
    )

    print(
        f"  dMCP = "
        f"{pose_delta[0]:+.3f} deg"
    )

    print(
        f"  dPIP = "
        f"{pose_delta[1]:+.3f} deg"
    )

    print(
        f"  dDIP = "
        f"{pose_delta[2]:+.3f} deg"
    )


    print()
    print(
        f"PIP absolute-error reduction = "
        f"{pip_error_reduction:+.3f} deg"
    )


    print()
    print(
        f"contact RMSE = "
        f"{selected['contact_rmse']:.3f} deg"
    )


    print()
    print(
        "Candidate XML:"
    )

    print(
        BEST_MODEL
    )


    # ========================================================
    # TWO CLOSE-UPS
    # ========================================================

    print()
    print("=" * 118)
    print(
        "TWO CLOSE-UP COMPARISON IMAGES"
    )
    print("=" * 118)


    render_comparison(
        control,
        selected,
    )


    print()
    print(
        RESULT_DIR
        /
        "v11_zero_friction_hold_closeup.png"
    )

    print(
        RESULT_DIR
        /
        "v12_best_guide_friction_hold_closeup.png"
    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    print()
    print("=" * 118)
    print(
        "INTERPRETATION"
    )
    print("=" * 118)


    if (
        selected[
            "free_combined"
        ]
        <= FREE_LIMIT
        and
        pip_error_reduction
        > 2.0
    ):

        print()
        print(
            "Guide friction produced a meaningful increase "
            "in contact PIP motion while preserving the "
            "validated free-space regime."
        )

        print()
        print(
            "NEXT: instrument the selected candidate's "
            "slip displacement and proximal/distal segment "
            "tensions through the contact sequence."
        )


    elif preserving:

        print()
        print(
            "Some guide-friction candidates preserve "
            "free-space, but the contact PIP improvement "
            "is small."
        )

        print()
        print(
            "Do not increase friction blindly. "
            "Next inspect internal slip/tension history "
            "and determine whether static-to-kinetic "
            "friction or another stateful mechanism is "
            "required."
        )


    else:

        print()
        print(
            "No friction value preserved the validated "
            "free-space regime."
        )

        print()
        print(
            "A single Coulomb-friction guide is therefore "
            "insufficient in this architecture. "
            "Do not continue increasing friction."
        )


    print()
    print(
        "Summary CSV:"
    )

    print(
        SUMMARY
    )


    print()
    print("=" * 118)


if __name__ == "__main__":
    main()
