#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv

import numpy as np

import scan_and_view_index_contact_friction_v18 as v18


# ============================================================
# RH8D INDEX — RUNTIME CONTACT FRICTION V19
#
# V18 XML friction scan was invalid because:
#
#   base.run_contact()
#
# explicitly performs:
#
#   contact_replay.MU = base.CONTACT_MU
#
# before every hardware-contact replay.
#
# Therefore XML changes were overwritten at runtime.
#
# V19 changes the ACTUAL wrapper global:
#
#   base.CONTACT_MU
#
# before every candidate.
#
# NORMAL CONTACT STIFFNESS IS FROZEN.
#
# No tendon/passive-joint/fixture/motor parameters change.
#
# The script also launches the selected candidate through
# V18's native MuJoCo visual replay.
#
# No custom viewer width/height is requested.
# ============================================================


MU_VALUES = [
    0.05,
    0.10,
    0.20,
    0.40,
    0.80,
    1.20,
]


RESULT_DIR = (
    v18.WS
    / "rh8d_digital_twin_tools/results/"
      "runtime_contact_friction_v19"
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
    v18.MJCF_DIR
    / "ur5e_rh8d_index_runtime_contact_friction_v19_candidate.xml"
)


# ============================================================
# RUN ONE CANDIDATE
# ============================================================

def run_runtime_candidate(
    mu,
    contact_bag,
    frozen_contact_stiffness,
):

    # --------------------------------------------------------
    # THIS is the critical V19 change.
    #
    # run_contact() reads these globals and then copies them
    # into replay_index_tabletop_hardware_contact_v1.
    # --------------------------------------------------------

    v18.base.CONTACT_MU = (
        float(mu)
    )

    v18.base.CONTACT_STIFFNESS = (
        frozen_contact_stiffness
    )


    result = v18.run_candidate(
        mu,
        contact_bag,
    )


    # --------------------------------------------------------
    # Hard validation gate.
    #
    # V18 failed exactly here:
    #
    # requested 0.1 ... 1.2
    # effective always 0.05.
    #
    # V19 is not accepted unless the runtime contact itself
    # reports the requested coefficient.
    # --------------------------------------------------------

    effective = float(
        result[
            "effective_mu"
        ]
    )


    error = abs(
        effective
        -
        mu
    )


    result[
        "runtime_mu_error"
    ] = error


    if (
        not np.isfinite(
            effective
        )
        or
        error
        > 0.005
    ):

        raise RuntimeError(
            "\n"
            "V19 RUNTIME FRICTION VALIDATION FAILED\n"
            f"  requested mu = {mu:.4f}\n"
            f"  effective mu = {effective:.4f}\n"
            "\n"
            "Do not interpret the candidate results."
        )


    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "contact_bag",
        type=Path,
    )


    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help=(
            "Run numerical V19 scan without opening "
            "the MuJoCo GUI."
        ),
    )


    args = parser.parse_args()


    contact_bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )


    if not contact_bag.exists():

        raise RuntimeError(
            f"Contact bag not found:\n"
            f"{contact_bag}"
        )


    # Ensure frozen V5 fixture registration.
    v18.ab.patch_replay_modules()


    # ========================================================
    # SAVE ORIGINAL WRAPPER GLOBALS
    # ========================================================

    original_mu = float(
        v18.base.CONTACT_MU
    )


    original_contact_stiffness = float(
        v18.base.CONTACT_STIFFNESS
    )


    print()
    print("=" * 118)
    print(
        "RH8D INDEX — VALIDATED RUNTIME "
        "CONTACT FRICTION SCAN V19"
    )
    print("=" * 118)


    print()
    print(
        "Wrapper defaults discovered at runtime:"
    )

    print(
        f"  base.CONTACT_MU = "
        f"{original_mu:.6f}"
    )

    print(
        f"  base.CONTACT_STIFFNESS = "
        f"{original_contact_stiffness:.3f}"
    )


    print()
    print(
        "V19 changes ONLY base.CONTACT_MU."
    )

    print(
        "Contact stiffness remains frozen at the "
        "value printed above."
    )


    print()
    print(
        "Candidates:"
    )

    print(
        MU_VALUES
    )


    results = []


    try:

        for i, mu in enumerate(
            MU_VALUES,
            start=1,
        ):

            print()
            print("=" * 118)

            print(
                f"[{i:02d}/{len(MU_VALUES):02d}] "
                f"RUNTIME CONTACT MU = "
                f"{mu:.3f}"
            )

            print("=" * 118)


            result = (
                run_runtime_candidate(
                    mu,
                    contact_bag,
                    original_contact_stiffness,
                )
            )


            results.append(
                result
            )


            print()
            print(
                "RUNTIME VALIDATION:"
            )

            print(
                f"  requested mu = "
                f"{mu:.3f}"
            )

            print(
                f"  effective MuJoCo mu = "
                f"{result['effective_mu']:.3f}"
            )

            print(
                f"  absolute error = "
                f"{result['runtime_mu_error']:.6f}"
            )


            print()
            print(
                "Established replay evaluator:"
            )

            print(
                "  MCP/PIP/DIP = "
                f"{result['contact_mcp']:.3f} / "
                f"{result['contact_pip']:.3f} / "
                f"{result['contact_dip']:.3f} deg"
            )

            print(
                f"  contact RMSE = "
                f"{result['contact_rmse']:.3f} deg"
            )


            print()
            print(
                "Contact mechanics:"
            )

            print(
                f"  middle duration = "
                f"{result['middle_duration_s']:.3f} s"
            )

            print(
                f"  tip duration = "
                f"{result['tip_duration_s']:.3f} s"
            )

            print(
                f"  middle utilization p95 = "
                f"{result['middle_util_p95']:.3f}"
            )

            print(
                f"  max middle Fn = "
                f"{result['max_middle_force_N']:.3f} N"
            )

            print(
                f"  max tip Fn = "
                f"{result['max_tip_force_N']:.3f} N"
            )


            if (
                result[
                    "first_tip_elapsed_s"
                ]
                is None
            ):

                print(
                    "  first tip contact = NEVER"
                )

            else:

                print(
                    f"  first tip contact = "
                    f"{result['first_tip_elapsed_s']:.3f} s"
                )


            print()
            print(
                "Corrected recorder state "
                f"at {v18.HOLD_ELAPSED:.3f} s:"
            )

            print(
                "  MCP/PIP/DIP = "
                f"{result['hold_mcp']:.3f} / "
                f"{result['hold_pip']:.3f} / "
                f"{result['hold_dip']:.3f} deg"
            )

            print(
                f"  middle gap/Fn = "
                f"{result['hold_mid_gap_mm']:+.3f} mm / "
                f"{result['hold_mid_force_N']:.3f} N"
            )

            print(
                f"  tip gap/Fn = "
                f"{result['hold_tip_gap_mm']:+.3f} mm / "
                f"{result['hold_tip_force_N']:.3f} N"
            )


    finally:

        # Restore original values after numerical scan.
        v18.base.CONTACT_MU = (
            original_mu
        )

        v18.base.CONTACT_STIFFNESS = (
            original_contact_stiffness
        )


    # ========================================================
    # SAVE SUMMARY
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
    # BASELINE
    # ========================================================

    baseline = min(
        results,
        key=lambda r:
            abs(
                r[
                    "requested_mu"
                ]
                -
                0.05
            ),
    )


    # ========================================================
    # SELECTION
    #
    # Sequential wrapping is more important than blindly
    # minimizing final RMSE.
    #
    # Prefer candidates which actually produce tip contact.
    # Among them choose minimum established contact RMSE.
    #
    # If none reaches tip, choose minimum contact RMSE but
    # explicitly report that the mechanical sequence failed.
    # ========================================================

    with_tip = [
        r
        for r in results
        if (
            r[
                "tip_duration_s"
            ]
            > 0.0
        )
    ]


    if with_tip:

        selected = min(
            with_tip,
            key=lambda r:
                r[
                    "contact_rmse"
                ],
        )

        selection_mode = (
            "TIP-CONTACT CANDIDATE"
        )


    else:

        selected = min(
            results,
            key=lambda r:
                r[
                    "contact_rmse"
                ],
        )

        selection_mode = (
            "NO TIP CONTACT IN SCAN"
        )


    # ========================================================
    # DIRECT EFFECT
    # ========================================================

    d_mcp = (
        selected[
            "contact_mcp"
        ]
        -
        baseline[
            "contact_mcp"
        ]
    )


    d_pip = (
        selected[
            "contact_pip"
        ]
        -
        baseline[
            "contact_pip"
        ]
    )


    d_dip = (
        selected[
            "contact_dip"
        ]
        -
        baseline[
            "contact_dip"
        ]
    )


    print()
    print("=" * 118)
    print(
        "V19 VALIDATED FRICTION RESULT"
    )
    print("=" * 118)


    print()
    print(
        f"Selection mode: "
        f"{selection_mode}"
    )


    print()
    print(
        f"selected requested/effective mu = "
        f"{selected['requested_mu']:.3f} / "
        f"{selected['effective_mu']:.3f}"
    )


    print()
    print(
        "Selected contact prediction:"
    )

    print(
        "  MCP/PIP/DIP = "
        f"{selected['contact_mcp']:.3f} / "
        f"{selected['contact_pip']:.3f} / "
        f"{selected['contact_dip']:.3f} deg"
    )

    print(
        f"  RMSE = "
        f"{selected['contact_rmse']:.3f} deg"
    )


    print()
    print(
        "Change relative to valid mu=0.05 baseline:"
    )

    print(
        f"  dMCP = "
        f"{d_mcp:+.3f} deg"
    )

    print(
        f"  dPIP = "
        f"{d_pip:+.3f} deg"
    )

    print(
        f"  dDIP = "
        f"{d_dip:+.3f} deg"
    )


    print()
    print(
        f"middle contact duration = "
        f"{selected['middle_duration_s']:.3f} s"
    )

    print(
        f"tip contact duration = "
        f"{selected['tip_duration_s']:.3f} s"
    )


    if (
        selected[
            "first_tip_elapsed_s"
        ]
        is None
    ):

        print(
            "first tip contact = NEVER"
        )

    else:

        print(
            f"first tip contact = "
            f"{selected['first_tip_elapsed_s']:.3f} s"
        )


    print()
    print(
        "Hardware target:"
    )

    print(
        "  14.536 / 23.811 / 9.327 deg"
    )


    # ========================================================
    # MECHANICAL INTERPRETATION
    # ========================================================

    print()
    print("=" * 118)
    print(
        "V19 MECHANICAL INTERPRETATION"
    )
    print("=" * 118)


    if (
        d_pip > 2.0
        and
        d_mcp < 0.0
    ):

        print()
        print(
            "VALID runtime friction increase produces "
            "the desired redistribution direction:"
        )

        print(
            "  MCP decreases"
        )

        print(
            "  PIP increases"
        )


    elif (
        d_pip > 2.0
    ):

        print()
        print(
            "Runtime friction significantly increases PIP, "
            "but MCP does not decrease."
        )

        print(
            "Contact friction contributes but is not the "
            "complete missing mechanism."
        )


    else:

        print()
        print(
            "Even with VERIFIED runtime friction changes, "
            "PIP redistribution remains small."
        )

        print(
            "Then friction alone is genuinely insufficient."
        )


    if not with_tip:

        print()
        print(
            "No tested friction value produced distal-tip "
            "contact."
        )

        print(
            "If the visual replay still shows a hard bounce, "
            "V20 should vary ONLY normal contact compliance "
            "using the same runtime-global override method."
        )


    # ========================================================
    # BUILD VISUAL MODEL
    # ========================================================

    v18.build_candidate(
        selected[
            "requested_mu"
        ],
        SELECTED_MODEL,
    )


    print()
    print(
        "Selected model:"
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
    #
    # IMPORTANT:
    # set runtime global again before visual replay.
    #
    # V18's viewer itself does not request custom dimensions.
    # ========================================================

    if not args.no_viewer:

        print()
        print("=" * 118)
        print(
            "V19 AUTOMATIC MUJOCO VISUAL REPLAY"
        )
        print("=" * 118)


        print()
        print(
            f"Runtime mu = "
            f"{selected['requested_mu']:.3f}"
        )

        print(
            f"Frozen contact stiffness = "
            f"{original_contact_stiffness:.3f}"
        )


        old_mu = float(
            v18.base.CONTACT_MU
        )

        old_k = float(
            v18.base.CONTACT_STIFFNESS
        )


        try:

            v18.base.CONTACT_MU = (
                selected[
                    "requested_mu"
                ]
            )

            v18.base.CONTACT_STIFFNESS = (
                original_contact_stiffness
            )


            v18.visual_replay(
                SELECTED_MODEL,
                contact_bag,
            )


        finally:

            v18.base.CONTACT_MU = (
                old_mu
            )

            v18.base.CONTACT_STIFFNESS = (
                old_k
            )


    print()
    print("=" * 118)
    print(
        "V19 COMPLETE"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
