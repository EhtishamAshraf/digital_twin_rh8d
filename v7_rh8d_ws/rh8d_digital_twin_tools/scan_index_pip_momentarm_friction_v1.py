#!/usr/bin/env python3

import sys
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import scan_index_pip_passive_dynamic_v2 as base


# ============================================================
# PURPOSE
#
# Test whether ONE CONSTANT PIP tendon coefficient can explain
# both:
#
#   validated dynamic free-space behavior
#   corrected tabletop contact behavior
#
# PIP stiffness stays fixed because the previous passive scan
# showed that the free-space-preserving region remained at
# k_PIP = 0.031 Nm/rad.
# ============================================================


PIP_NAME = "Index_Middle--Index_Proximal"
TENDON_NAME = "idx_flex_spatial"

PIP_STIFFNESS = 0.031


# Effective coefficients only -- NOT claimed pulley radii.
PIP_ARMS_MM = [
     2.0,
     3.0,
     4.0,
     5.0,
     6.0,
     7.0,
     8.0,
    10.0,
    12.0,
]


# Previous scan showed:
#   0.006 = validated baseline
#   0.003 = best free-space-preserving passive alternative
#
# Include an intermediate value.
PIP_FRICTIONS = [
    0.0030,
    0.0045,
    0.0060,
]


RESULT_DIR = (
    base.free_replay.WS
    / "rh8d_digital_twin_tools/results/"
      "pip_momentarm_friction_v1"
)

RESULT_CSV = (
    RESULT_DIR
    / "summary.csv"
)


def label_for(
    arm_mm,
    friction,
):

    return (
        f"arm{arm_mm:.1f}_f{friction:.4f}"
        .replace(".", "p")
    )


def build_candidate(
    source,
    output,
    arm_mm,
    friction,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()


    found_named_pip = False
    found_tendon = False
    found_pip_coef = False


    # --------------------------------------------------------
    # Named physical PIP joint:
    # freeze stiffness, vary friction.
    # --------------------------------------------------------

    for element in root.iter("joint"):

        if (
            element.attrib.get("name")
            == PIP_NAME
        ):

            element.set(
                "stiffness",
                f"{PIP_STIFFNESS:.10g}",
            )

            element.set(
                "frictionloss",
                f"{friction:.10g}",
            )

            found_named_pip = True


    # --------------------------------------------------------
    # Fixed index tendon:
    # freeze tendon stiffness = Candidate A
    # vary ONLY PIP coefficient.
    # --------------------------------------------------------

    for fixed in root.iter("fixed"):

        if (
            fixed.attrib.get("name")
            != TENDON_NAME
        ):
            continue


        fixed.set(
            "stiffness",
            f"{base.TENDON_STIFFNESS:.10g}",
        )

        found_tendon = True


        for joint in fixed.findall("joint"):

            if (
                joint.attrib.get("joint")
                == PIP_NAME
            ):

                coef = (
                    -arm_mm
                    / 1000.0
                )

                joint.set(
                    "coef",
                    f"{coef:.10g}",
                )

                found_pip_coef = True


    if not found_named_pip:

        raise RuntimeError(
            f"Could not find named PIP joint in:\n"
            f"{source}"
        )


    if not found_tendon:

        raise RuntimeError(
            f"Could not find tendon {TENDON_NAME} in:\n"
            f"{source}"
        )


    if not found_pip_coef:

        raise RuntimeError(
            f"Could not find PIP coefficient inside "
            f"{TENDON_NAME} in:\n{source}"
        )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "  python3 scan_index_pip_momentarm_friction_v1.py "
            "/path/to/free_space_bag "
            "/path/to/contact_bag"
        )

        sys.exit(1)


    free_bag = Path(
        sys.argv[1]
    ).expanduser().resolve()

    contact_bag = Path(
        sys.argv[2]
    ).expanduser().resolve()


    if not free_bag.exists():
        raise RuntimeError(
            f"Free-space bag missing:\n{free_bag}"
        )


    if not contact_bag.exists():
        raise RuntimeError(
            f"Contact bag missing:\n{contact_bag}"
        )


    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print("=" * 140)
    print(
        "RH8D INDEX — CONSTANT PIP MOMENT-ARM "
        "+ FRICTION SCAN V1"
    )
    print("=" * 140)

    print()
    print(
        f"PIP stiffness frozen = "
        f"{PIP_STIFFNESS:.4f} Nm/rad"
    )

    print(
        f"encoder/spool scale   = "
        f"{base.METERS_PER_TICK*1e6:.3f} um/tick"
    )

    print(
        f"tendon stiffness      = "
        f"{base.TENDON_STIFFNESS:.1f} N/m"
    )

    print()
    print(
        "MCP coefficient frozen = -5 mm"
    )

    print(
        "DIP coefficient frozen = -1 mm"
    )

    print()
    print(
        "NOTE: scanned PIP values are EFFECTIVE "
        "transmission coefficients, not claimed "
        "physical pulley radii."
    )


    total = (
        len(PIP_ARMS_MM)
        *
        len(PIP_FRICTIONS)
    )


    print()
    print(
        f"Candidates = {total}"
    )


    results = []


    count = 0


    for arm_mm in PIP_ARMS_MM:

        for friction in PIP_FRICTIONS:

            count += 1

            label = label_for(
                arm_mm,
                friction,
            )


            # Must remain beside source XML because the MJCF
            # uses relative mesh paths.
            free_xml = (
                base.FREE_SOURCE.parent
                /
                f"tmp_piparm_free_{label}.xml"
            )

            contact_xml = (
                base.CONTACT_SOURCE.parent
                /
                f"tmp_piparm_contact_{label}.xml"
            )


            build_candidate(
                base.FREE_SOURCE,
                free_xml,
                arm_mm,
                friction,
            )

            build_candidate(
                base.CONTACT_SOURCE,
                contact_xml,
                arm_mm,
                friction,
            )


            print()
            print(
                f"[{count:02d}/{total}] "
                f"arm={arm_mm:4.1f} mm  "
                f"friction={friction:.4f}"
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


            pose = contact_metrics[
                "pose"
            ]


            result = {

                "arm_mm":
                    arm_mm,

                "friction":
                    friction,

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

                "contact_rmse":
                    contact_metrics[
                        "rmse"
                    ],

                "contact_mcp":
                    pose[0],

                "contact_pip":
                    pose[1],

                "contact_dip":
                    pose[2],

                "dmcp":
                    contact_metrics[
                        "dmcp"
                    ],

                "dpip":
                    contact_metrics[
                        "dpip"
                    ],

                "ddip":
                    contact_metrics[
                        "ddip"
                    ],

                "middle_duty":
                    contact_metrics[
                        "middle_duty"
                    ],
            }


            results.append(
                result
            )


            print(
                f"  free = "
                f"{result['free_combined']:.3f} deg"
            )

            print(
                f"  contact = "
                f"{result['contact_rmse']:.3f} deg"
            )

            print(
                "  hold = "
                f"({result['contact_mcp']:.2f}, "
                f"{result['contact_pip']:.2f}, "
                f"{result['contact_dip']:.2f})"
            )


            # Remove only our generated temporary models.
            for path in [
                free_xml,
                contact_xml,
            ]:

                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


    # ========================================================
    # Save
    # ========================================================

    with open(
        RESULT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # ========================================================
    # Baseline
    # ========================================================

    baseline = next(
        r
        for r in results
        if (
            abs(
                r["arm_mm"]
                - 4.0
            ) < 1e-12
            and
            abs(
                r["friction"]
                - 0.006
            ) < 1e-12
        )
    )


    baseline_free = (
        baseline[
            "free_combined"
        ]
    )


    print()
    print("=" * 140)
    print(
        "BASELINE CHECK"
    )
    print("=" * 140)

    print()
    print(
        f"arm = "
        f"{baseline['arm_mm']:.1f} mm"
    )

    print(
        f"friction = "
        f"{baseline['friction']:.4f}"
    )

    print(
        f"dynamic free RMSE = "
        f"{baseline['free_combined']:.3f} deg"
    )

    print(
        "contact pose = "
        f"({baseline['contact_mcp']:.3f}, "
        f"{baseline['contact_pip']:.3f}, "
        f"{baseline['contact_dip']:.3f}) deg"
    )

    print(
        f"contact RMSE = "
        f"{baseline['contact_rmse']:.3f} deg"
    )


    # ========================================================
    # Preserve validated free-space behavior
    # ========================================================

    FREE_ALLOWANCE = 1.0


    eligible = [
        r
        for r in results
        if (
            r["free_combined"]
            <=
            baseline_free
            +
            FREE_ALLOWANCE
        )
    ]


    eligible.sort(
        key=lambda r:
            r["contact_rmse"]
    )


    print()
    print("=" * 140)
    print(
        "FREE-SPACE-PRESERVING CONSTANT-TRANSMISSION CANDIDATES"
    )
    print(
        f"(dynamic free <= baseline + "
        f"{FREE_ALLOWANCE:.1f} deg)"
    )
    print("=" * 140)


    if not eligible:

        print()
        print("NONE")

    else:

        for rank, r in enumerate(
            eligible[:15],
            start=1,
        ):

            print()
            print(
                f"#{rank} "
                f"arm={r['arm_mm']:.1f} mm  "
                f"friction={r['friction']:.4f}"
            )

            print(
                f"  free MCP/PIP/combined = "
                f"{r['free_mcp']:.3f} / "
                f"{r['free_pip']:.3f} / "
                f"{r['free_combined']:.3f} deg"
            )

            print(
                "  contact pose = "
                f"({r['contact_mcp']:.3f}, "
                f"{r['contact_pip']:.3f}, "
                f"{r['contact_dip']:.3f}) deg"
            )

            print(
                f"  contact RMSE = "
                f"{r['contact_rmse']:.3f} deg"
            )

            print(
                "  redistribution dMCP/dPIP/dDIP = "
                f"{r['dmcp']} / "
                f"{r['dpip']} / "
                f"{r['ddip']}"
            )

            print(
                f"  middle duty = "
                f"{r['middle_duty']} %"
            )


    # ========================================================
    # Best contact result regardless of free-space
    # ========================================================

    best_contact = min(
        results,
        key=lambda r:
            r["contact_rmse"],
    )


    print()
    print("=" * 140)
    print(
        "BEST CONTACT RESULT REGARDLESS OF FREE-SPACE"
    )
    print("=" * 140)

    print()
    print(
        f"arm = "
        f"{best_contact['arm_mm']:.1f} mm"
    )

    print(
        f"friction = "
        f"{best_contact['friction']:.4f}"
    )

    print(
        f"dynamic free RMSE = "
        f"{best_contact['free_combined']:.3f} deg"
    )

    print(
        "contact pose = "
        f"({best_contact['contact_mcp']:.3f}, "
        f"{best_contact['contact_pip']:.3f}, "
        f"{best_contact['contact_dip']:.3f}) deg"
    )

    print(
        f"contact RMSE = "
        f"{best_contact['contact_rmse']:.3f} deg"
    )


    # ========================================================
    # Interpretation
    # ========================================================

    print()
    print("=" * 140)
    print(
        "INTERPRETATION"
    )
    print("=" * 140)


    good = [
        r
        for r in eligible
        if (
            r["contact_pip"]
            >= 18.0
            and
            r["contact_rmse"]
            <= 6.0
        )
    ]


    if good:

        best = good[0]

        print()
        print(
            "A CONSTANT PIP transmission coefficient "
            "can produce substantial contact redistribution "
            "while preserving the validated free-space fit."
        )

        print()
        print(
            "Best qualifying candidate:"
        )

        print(
            f"  arm = "
            f"{best['arm_mm']:.1f} mm"
        )

        print(
            f"  friction = "
            f"{best['friction']:.4f}"
        )

        print(
            f"  free RMSE = "
            f"{best['free_combined']:.3f} deg"
        )

        print(
            f"  contact PIP = "
            f"{best['contact_pip']:.3f} deg"
        )

        print(
            f"  contact RMSE = "
            f"{best['contact_rmse']:.3f} deg"
        )

        print()
        print(
            "Do NOT introduce configuration-dependent "
            "transmission yet. Refine this constant region."
        )


    elif (
        best_contact[
            "contact_pip"
        ]
        >= 18.0
    ):

        print()
        print(
            "A constant coefficient CAN create the required "
            "contact PIP flexion, but doing so destroys the "
            "validated free-space behavior."
        )

        print()
        print(
            "This is strong evidence that one constant "
            "effective PIP coefficient cannot explain "
            "both regimes."
        )

        print()
        print(
            "The next structural test should be a "
            "configuration/load-dependent effective "
            "transmission."
        )


    else:

        print()
        print(
            "No tested constant PIP coefficient generated "
            "hardware-like PIP contact flexion while the "
            "rest of the model remained frozen."
        )

        print()
        print(
            "Before introducing configuration dependence, "
            "we should next test coupled MCP/PIP/DIP "
            "transmission ratios."
        )


    print()
    print(
        "Summary CSV:"
    )

    print(
        RESULT_CSV
    )

    print()
    print("=" * 140)


if __name__ == "__main__":
    main()
