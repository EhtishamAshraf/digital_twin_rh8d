#!/usr/bin/env python3

import sys
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import scan_index_pip_passive_dynamic_v2 as base


# ============================================================
# COUPLED CONSTANT TRANSMISSION TEST
#
# Question:
#
# Can ONE constant effective transmission vector
#
#   [MCP coefficient, PIP coefficient, DIP coefficient]
#
# reproduce BOTH:
#
#   - validated dynamic free-space trajectory
#   - corrected tabletop contact redistribution
#
# before we introduce configuration/load dependence?
# ============================================================


MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

TENDON_NAME = "idx_flex_spatial"


# ============================================================
# Passive PIP mechanics
#
# Freeze at the best contact-friendly setting that still
# stayed close to the validated free-space model.
# ============================================================

PIP_STIFFNESS = 0.031
PIP_FRICTION = 0.003


# ============================================================
# Effective tendon coefficients.
#
# These are NOT claimed physical pulley radii.
#
# Baseline = [5, 4, 1] mm
# ============================================================

MCP_ARMS_MM = [
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
]

PIP_ARMS_MM = [
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
]

DIP_ARMS_MM = [
    0.5,
    1.0,
    1.5,
]


# Trusted Candidate-A free-space baseline.
REFERENCE_FREE_RMSE = 5.085

FREE_ALLOWANCE = 1.0
FREE_LIMIT = (
    REFERENCE_FREE_RMSE
    + FREE_ALLOWANCE
)


# Diagnostic gates only, not final validation criteria.
MIN_MIDDLE_DUTY = 70.0
MIN_CONTACT_PIP = 18.0
MAX_CONTACT_RMSE = 6.0


RESULT_DIR = (
    base.free_replay.WS
    / "rh8d_digital_twin_tools/results/"
      "coupled_transmission_v1"
)

RESULT_CSV = (
    RESULT_DIR
    / "summary.csv"
)


def label_for(
    mcp,
    pip,
    dip,
):

    return (
        f"m{mcp:.1f}_p{pip:.1f}_d{dip:.1f}"
        .replace(".", "p")
    )


def build_candidate(
    source,
    output,
    mcp_mm,
    pip_mm,
    dip_mm,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()


    found_pip_joint = False
    found_tendon = False

    found_mcp = False
    found_pip = False
    found_dip = False


    # ========================================================
    # Passive PIP joint
    # ========================================================

    for joint in root.iter("joint"):

        if (
            joint.attrib.get("name")
            == PIP_NAME
        ):

            joint.set(
                "stiffness",
                f"{PIP_STIFFNESS:.10g}",
            )

            joint.set(
                "frictionloss",
                f"{PIP_FRICTION:.10g}",
            )

            found_pip_joint = True


    # ========================================================
    # Fixed index tendon
    # ========================================================

    for fixed in root.iter("fixed"):

        if (
            fixed.attrib.get("name")
            != TENDON_NAME
        ):
            continue


        found_tendon = True


        # Candidate-A tendon stiffness remains frozen.
        fixed.set(
            "stiffness",
            f"{base.TENDON_STIFFNESS:.10g}",
        )


        for joint in fixed.findall("joint"):

            name = joint.attrib.get(
                "joint"
            )


            if name == MCP_NAME:

                joint.set(
                    "coef",
                    f"{-mcp_mm/1000.0:.10g}",
                )

                found_mcp = True


            elif name == PIP_NAME:

                joint.set(
                    "coef",
                    f"{-pip_mm/1000.0:.10g}",
                )

                found_pip = True


            elif name == DIP_NAME:

                joint.set(
                    "coef",
                    f"{-dip_mm/1000.0:.10g}",
                )

                found_dip = True


    if not found_pip_joint:
        raise RuntimeError(
            f"PIP physical joint not found:\n{source}"
        )

    if not found_tendon:
        raise RuntimeError(
            f"Index tendon not found:\n{source}"
        )

    if not (
        found_mcp
        and found_pip
        and found_dip
    ):
        raise RuntimeError(
            "Could not find all three index tendon "
            "coefficients."
        )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


def valid_sequence(
    result,
):

    return (
        result["dmcp"] is not None
        and
        result["dpip"] is not None
        and
        result["ddip"] is not None
        and
        result["middle_duty"] is not None
        and
        result["middle_duty"]
        >= MIN_MIDDLE_DUTY
    )


def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            " python3 "
            "scan_index_coupled_transmission_v1.py "
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
            f"Missing free-space bag:\n{free_bag}"
        )

    if not contact_bag.exists():
        raise RuntimeError(
            f"Missing contact bag:\n{contact_bag}"
        )


    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    total = (
        len(MCP_ARMS_MM)
        *
        len(PIP_ARMS_MM)
        *
        len(DIP_ARMS_MM)
    )


    print()
    print("=" * 145)
    print(
        "RH8D INDEX — COUPLED CONSTANT "
        "MCP/PIP/DIP TRANSMISSION SCAN V1"
    )
    print("=" * 145)

    print()
    print(
        "Validated motor/transmission layer frozen:"
    )

    print(
        f"  scale = "
        f"{base.METERS_PER_TICK*1e6:.3f} um/tick"
    )

    print(
        f"  tendon stiffness = "
        f"{base.TENDON_STIFFNESS:.1f} N/m"
    )

    print(
        f"  motor delay = "
        f"{base.free_replay.MOTOR_DELAY*1000:.1f} ms"
    )

    print(
        f"  closing rate = "
        f"{base.free_replay.CLOSING_RATE:.1f} ticks/s"
    )

    print(
        f"  opening rate = "
        f"{base.free_replay.OPENING_RATE:.1f} ticks/s"
    )


    print()
    print(
        "PIP passive mechanics frozen:"
    )

    print(
        f"  stiffness = "
        f"{PIP_STIFFNESS:.4f} Nm/rad"
    )

    print(
        f"  frictionloss = "
        f"{PIP_FRICTION:.4f} Nm"
    )


    print()
    print(
        "Transmission baseline = "
        "[5, 4, 1] mm"
    )

    print(
        "Scanned values remain EFFECTIVE "
        "transmission coefficients."
    )


    print()
    print(
        f"Candidates = {total}"
    )

    print(
        f"Free-space preservation gate = "
        f"{FREE_LIMIT:.3f} deg"
    )


    results = []

    count = 0


    for mcp_mm in MCP_ARMS_MM:

        for pip_mm in PIP_ARMS_MM:

            for dip_mm in DIP_ARMS_MM:

                count += 1


                label = label_for(
                    mcp_mm,
                    pip_mm,
                    dip_mm,
                )


                free_xml = (
                    base.FREE_SOURCE.parent
                    /
                    f"tmp_coupled_free_{label}.xml"
                )

                contact_xml = (
                    base.CONTACT_SOURCE.parent
                    /
                    f"tmp_coupled_contact_{label}.xml"
                )


                build_candidate(
                    base.FREE_SOURCE,
                    free_xml,
                    mcp_mm,
                    pip_mm,
                    dip_mm,
                )

                build_candidate(
                    base.CONTACT_SOURCE,
                    contact_xml,
                    mcp_mm,
                    pip_mm,
                    dip_mm,
                )


                print()
                print(
                    f"[{count:02d}/{total}] "
                    f"MCP={mcp_mm:.1f}  "
                    f"PIP={pip_mm:.1f}  "
                    f"DIP={dip_mm:.1f} mm"
                )


                try:

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


                    pose = (
                        contact_metrics[
                            "pose"
                        ]
                    )


                    result = {

                        "mcp_mm":
                            mcp_mm,

                        "pip_mm":
                            pip_mm,

                        "dip_mm":
                            dip_mm,

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


                finally:

                    for p in [
                        free_xml,
                        contact_xml,
                    ]:

                        try:
                            p.unlink()
                        except FileNotFoundError:
                            pass


    # ========================================================
    # CSV
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
    # Control candidate
    #
    # [5,4,1] with PIP friction 0.003.
    # Should reproduce approximately the previously observed
    # free=5.840/contact=10.457 result.
    # ========================================================

    control = next(
        r
        for r in results
        if (
            r["mcp_mm"] == 5.0
            and
            r["pip_mm"] == 4.0
            and
            r["dip_mm"] == 1.0
        )
    )


    print()
    print("=" * 145)
    print(
        "CONTROL CHECK"
    )
    print("=" * 145)

    print()
    print(
        "Transmission = [5, 4, 1] mm"
    )

    print(
        f"free RMSE = "
        f"{control['free_combined']:.3f} deg"
    )

    print(
        "contact pose = "
        f"({control['contact_mcp']:.3f}, "
        f"{control['contact_pip']:.3f}, "
        f"{control['contact_dip']:.3f}) deg"
    )

    print(
        f"contact RMSE = "
        f"{control['contact_rmse']:.3f} deg"
    )


    # ========================================================
    # Free-space preserving + valid sequence
    # ========================================================

    eligible = [
        r
        for r in results
        if (
            r["free_combined"]
            <= FREE_LIMIT
            and
            valid_sequence(r)
        )
    ]


    eligible.sort(
        key=lambda r:
            r[
                "contact_rmse"
            ]
    )


    print()
    print("=" * 145)
    print(
        "FREE-SPACE-PRESERVING COUPLED CANDIDATES"
    )

    print(
        f"(free <= {FREE_LIMIT:.3f} deg "
        f"and middle duty >= "
        f"{MIN_MIDDLE_DUTY:.0f}%)"
    )

    print("=" * 145)


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
                f"[MCP,PIP,DIP] = "
                f"[{r['mcp_mm']:.1f}, "
                f"{r['pip_mm']:.1f}, "
                f"{r['dip_mm']:.1f}] mm"
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
                "  redistribution "
                "dMCP/dPIP/dDIP = "
                f"{r['dmcp']} / "
                f"{r['dpip']} / "
                f"{r['ddip']}"
            )

            print(
                f"  middle duty = "
                f"{r['middle_duty']} %"
            )


    # ========================================================
    # Best contact candidate regardless of free-space
    # ========================================================

    best_contact = min(
        results,
        key=lambda r:
            r[
                "contact_rmse"
            ],
    )


    print()
    print("=" * 145)
    print(
        "BEST CONTACT RESULT REGARDLESS OF FREE-SPACE"
    )
    print("=" * 145)

    print()
    print(
        "[MCP,PIP,DIP] = "
        f"[{best_contact['mcp_mm']:.1f}, "
        f"{best_contact['pip_mm']:.1f}, "
        f"{best_contact['dip_mm']:.1f}] mm"
    )

    print(
        f"free RMSE = "
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
    # Gate
    # ========================================================

    qualifying = [
        r
        for r in eligible
        if (
            r["contact_pip"]
            >= MIN_CONTACT_PIP
            and
            r["contact_rmse"]
            <= MAX_CONTACT_RMSE
        )
    ]


    print()
    print("=" * 145)
    print(
        "INTERPRETATION"
    )
    print("=" * 145)


    if qualifying:

        best = qualifying[0]

        print()
        print(
            "A coupled CONSTANT transmission vector "
            "can explain substantially more of both regimes."
        )

        print()
        print(
            "Best qualifying vector:"
        )

        print(
            f"  [MCP,PIP,DIP] = "
            f"[{best['mcp_mm']:.1f}, "
            f"{best['pip_mm']:.1f}, "
            f"{best['dip_mm']:.1f}] mm"
        )

        print(
            f"  dynamic free = "
            f"{best['free_combined']:.3f} deg"
        )

        print(
            f"  contact pose = "
            f"({best['contact_mcp']:.3f}, "
            f"{best['contact_pip']:.3f}, "
            f"{best['contact_dip']:.3f}) deg"
        )

        print(
            f"  contact RMSE = "
            f"{best['contact_rmse']:.3f} deg"
        )

        print()
        print(
            "Do NOT introduce configuration dependence yet."
        )


    else:

        print()
        print(
            "No coarse constant MCP/PIP/DIP transmission "
            "vector passed both the dynamic free-space "
            "and contact gates."
        )

        print()
        print(
            "This strengthens the evidence against the "
            "current constant-transmission model."
        )

        print()
        print(
            "However, this is NOT yet proof that "
            "configuration dependence is required."
        )

        print(
            "Before making that structural change, the next "
            "step should allow a LOCAL re-identification of "
            "encoder/spool scale and tendon stiffness together "
            "with the best coupled transmission ratios."
        )


    print()
    print(
        "Summary:"
    )

    print(
        RESULT_CSV
    )

    print()
    print("=" * 145)


if __name__ == "__main__":
    main()
