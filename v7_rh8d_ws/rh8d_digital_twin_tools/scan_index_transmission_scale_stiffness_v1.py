#!/usr/bin/env python3

import sys
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import scan_index_pip_passive_dynamic_v2 as base


# ============================================================
# LOCAL CONSTANT-MODEL RE-IDENTIFICATION
#
# For several Pareto-knee constant transmission vectors,
# jointly re-identify:
#
#   encoder -> spool scale
#   tendon stiffness
#
# Every candidate is evaluated with:
#
#   1. exact validated dynamic free-space replay
#   2. actual hardware encoder contact replay
#
# This is the final fair test of a CONSTANT effective
# transmission before considering configuration dependence.
# ============================================================


MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

TENDON_NAME = "idx_flex_spatial"


# ============================================================
# PIP passive mechanics
#
# Frozen at best contact-friendly value that remained close
# to the validated free-space region.
# ============================================================

PIP_STIFFNESS = 0.031
PIP_FRICTION = 0.003


# ============================================================
# Representative constant transmission vectors
#
# effective coefficients in mm/rad equivalent
# NOT claimed physical pulley radii.
# ============================================================

TRANSMISSIONS = [

    # original/control
    (5.0, 4.0, 1.0),

    # free-space side of Pareto knee
    (6.0, 5.0, 1.0),

    # central Pareto knee
    (5.0, 5.0, 1.0),

    # contact side of Pareto knee
    (5.0, 6.0, 1.0),
]


# ============================================================
# Local scale/stiffness search around Candidate A
#
# Candidate A:
#   scale = 5.1 um/tick
#   k     = 2100 N/m
# ============================================================

SCALES_UM = [
    4.60,
    4.85,
    5.10,
    5.35,
    5.60,
]


TENDON_STIFFNESSES = [
    1600.0,
    1850.0,
    2100.0,
    2350.0,
    2600.0,
]


# ============================================================
# Diagnostic gates
# ============================================================

ORIGINAL_VALIDATED_FREE = 5.085

FREE_LIMIT = 6.085

CONTACT_TARGET = (
    14.5355,
    23.8110,
    9.3270,
)

MIN_CONTACT_PIP = 18.0
MAX_CONTACT_RMSE = 6.0
MIN_MIDDLE_DUTY = 70.0


RESULT_DIR = (
    base.free_replay.WS
    / "rh8d_digital_twin_tools/results/"
      "transmission_scale_stiffness_v1"
)

RESULT_CSV = (
    RESULT_DIR
    / "summary.csv"
)


def make_label(
    mcp,
    pip,
    dip,
    scale,
    stiffness,
):

    text = (
        f"m{mcp:.1f}_"
        f"p{pip:.1f}_"
        f"d{dip:.1f}_"
        f"s{scale:.2f}_"
        f"k{stiffness:.0f}"
    )

    return text.replace(
        ".",
        "p",
    )


def build_candidate(
    source,
    output,
    mcp_mm,
    pip_mm,
    dip_mm,
    tendon_stiffness,
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


    # --------------------------------------------------------
    # Physical PIP passive mechanics
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Fixed effective tendon
    # --------------------------------------------------------

    for fixed in root.iter("fixed"):

        if (
            fixed.attrib.get("name")
            != TENDON_NAME
        ):
            continue


        found_tendon = True


        fixed.set(
            "stiffness",
            f"{tendon_stiffness:.10g}",
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
            f"PIP physical joint missing:\n{source}"
        )


    if not found_tendon:

        raise RuntimeError(
            f"Index tendon missing:\n{source}"
        )


    if not (
        found_mcp
        and found_pip
        and found_dip
    ):

        raise RuntimeError(
            "Could not find all three index "
            "transmission coefficients."
        )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


def run_candidate(
    free_xml,
    contact_xml,
    free_bag,
    contact_bag,
    scale_um,
    tendon_stiffness,
):

    # base.run_free() and base.run_contact() use these globals.
    # Save/restore so every candidate is isolated.

    old_scale = (
        base.METERS_PER_TICK
    )

    old_stiffness = (
        base.TENDON_STIFFNESS
    )


    try:

        base.METERS_PER_TICK = (
            scale_um
            * 1.0e-6
        )

        base.TENDON_STIFFNESS = (
            tendon_stiffness
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
            old_stiffness
        )


    return (
        free_metrics,
        free_log,
        contact_metrics,
        contact_log,
    )


def sequence_valid(
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


def dominates(
    a,
    b,
):

    return (
        a["free_combined"]
        <= b["free_combined"]
        and
        a["contact_rmse"]
        <= b["contact_rmse"]
        and
        (
            a["free_combined"]
            < b["free_combined"]
            or
            a["contact_rmse"]
            < b["contact_rmse"]
        )
    )


def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            " python3 "
            "scan_index_transmission_scale_stiffness_v1.py "
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
            f"Missing free-space bag:\n"
            f"{free_bag}"
        )


    if not contact_bag.exists():

        raise RuntimeError(
            f"Missing contact bag:\n"
            f"{contact_bag}"
        )


    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    total = (
        len(TRANSMISSIONS)
        *
        len(SCALES_UM)
        *
        len(TENDON_STIFFNESSES)
    )


    print()
    print("=" * 150)
    print(
        "RH8D INDEX — CONSTANT TRANSMISSION "
        "LOCAL SCALE + STIFFNESS RE-IDENTIFICATION V1"
    )
    print("=" * 150)


    print()
    print(
        "Motor model remains frozen:"
    )

    print(
        f"  delay = "
        f"{base.free_replay.MOTOR_DELAY*1000:.1f} ms"
    )

    print(
        f"  closing = "
        f"{base.free_replay.CLOSING_RATE:.1f} ticks/s"
    )

    print(
        f"  opening = "
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
        "Transmission vectors:"
    )

    for vector in TRANSMISSIONS:

        print(
            f"  {vector}"
        )


    print()
    print(
        f"Scales = {SCALES_UM}"
    )

    print(
        f"Tendon stiffnesses = "
        f"{TENDON_STIFFNESSES}"
    )

    print()
    print(
        f"Total candidates = {total}"
    )


    results = []

    count = 0


    for (
        mcp_mm,
        pip_mm,
        dip_mm,
    ) in TRANSMISSIONS:


        for scale_um in SCALES_UM:


            for tendon_k in TENDON_STIFFNESSES:

                count += 1


                label = make_label(
                    mcp_mm,
                    pip_mm,
                    dip_mm,
                    scale_um,
                    tendon_k,
                )


                # Keep temporary MJCF beside source MJCF because
                # source models use relative mesh paths.

                free_xml = (
                    base.FREE_SOURCE.parent
                    /
                    f"tmp_local_free_{label}.xml"
                )


                contact_xml = (
                    base.CONTACT_SOURCE.parent
                    /
                    f"tmp_local_contact_{label}.xml"
                )


                build_candidate(
                    base.FREE_SOURCE,
                    free_xml,
                    mcp_mm,
                    pip_mm,
                    dip_mm,
                    tendon_k,
                )


                build_candidate(
                    base.CONTACT_SOURCE,
                    contact_xml,
                    mcp_mm,
                    pip_mm,
                    dip_mm,
                    tendon_k,
                )


                print()
                print(
                    f"[{count:03d}/{total}] "
                    f"[{mcp_mm:.1f},"
                    f"{pip_mm:.1f},"
                    f"{dip_mm:.1f}] "
                    f"scale={scale_um:.2f} "
                    f"k={tendon_k:.0f}"
                )


                try:

                    (
                        free_metrics,
                        free_log,
                        contact_metrics,
                        contact_log,
                    ) = run_candidate(
                        free_xml,
                        contact_xml,
                        free_bag,
                        contact_bag,
                        scale_um,
                        tendon_k,
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


                    pose = (
                        contact_metrics[
                            "pose"
                        ]
                    )


                    r = {

                        "mcp_mm":
                            mcp_mm,

                        "pip_mm":
                            pip_mm,

                        "dip_mm":
                            dip_mm,

                        "scale_um":
                            scale_um,

                        "tendon_k":
                            tendon_k,

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
                        r
                    )


                    print(
                        f"  free    = "
                        f"{r['free_combined']:.3f} deg"
                    )

                    print(
                        f"  contact = "
                        f"{r['contact_rmse']:.3f} deg"
                    )

                    print(
                        f"  hold    = "
                        f"({r['contact_mcp']:.2f}, "
                        f"{r['contact_pip']:.2f}, "
                        f"{r['contact_dip']:.2f})"
                    )


                finally:

                    for path in [
                        free_xml,
                        contact_xml,
                    ]:

                        try:
                            path.unlink()
                        except FileNotFoundError:
                            pass


    # ========================================================
    # Save summary
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
    # Control
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
            and
            abs(
                r["scale_um"]
                - 5.10
            ) < 1e-9
            and
            abs(
                r["tendon_k"]
                - 2100.0
            ) < 1e-9
        )
    )


    print()
    print("=" * 150)
    print(
        "CONTROL CHECK"
    )
    print("=" * 150)

    print()
    print(
        "[MCP,PIP,DIP] = [5,4,1] mm"
    )

    print(
        "scale = 5.10 um/tick"
    )

    print(
        "tendon stiffness = 2100 N/m"
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
    # Best candidate for each transmission under free gate
    # ========================================================

    print()
    print("=" * 150)
    print(
        "BEST FREE-SPACE-PRESERVING RESULT "
        "FOR EACH TRANSMISSION"
    )
    print(
        f"(free RMSE <= {FREE_LIMIT:.3f} deg)"
    )
    print("=" * 150)


    free_preserving = []


    for vector in TRANSMISSIONS:

        mcp, pip, dip = vector


        group = [
            r
            for r in results
            if (
                r["mcp_mm"] == mcp
                and
                r["pip_mm"] == pip
                and
                r["dip_mm"] == dip
                and
                r["free_combined"]
                <= FREE_LIMIT
            )
        ]


        group.sort(
            key=lambda r:
                r["contact_rmse"]
        )


        print()
        print(
            f"Transmission "
            f"[{mcp:.1f},"
            f"{pip:.1f},"
            f"{dip:.1f}] mm"
        )


        if not group:

            print(
                "  NO candidate preserved "
                "the free-space gate."
            )

            continue


        best = group[0]

        free_preserving.append(
            best
        )


        print(
            f"  scale = "
            f"{best['scale_um']:.2f} um/tick"
        )

        print(
            f"  tendon k = "
            f"{best['tendon_k']:.0f} N/m"
        )

        print(
            f"  free MCP/PIP/combined = "
            f"{best['free_mcp']:.3f} / "
            f"{best['free_pip']:.3f} / "
            f"{best['free_combined']:.3f} deg"
        )

        print(
            "  contact pose = "
            f"({best['contact_mcp']:.3f}, "
            f"{best['contact_pip']:.3f}, "
            f"{best['contact_dip']:.3f}) deg"
        )

        print(
            f"  contact RMSE = "
            f"{best['contact_rmse']:.3f} deg"
        )

        print(
            f"  middle duty = "
            f"{best['middle_duty']} %"
        )


    # ========================================================
    # Global Pareto frontier
    # ========================================================

    pareto = []


    for r in results:

        if not any(
            dominates(
                other,
                r,
            )
            for other in results
            if other is not r
        ):

            pareto.append(
                r
            )


    pareto.sort(
        key=lambda r:
            r["free_combined"]
    )


    print()
    print("=" * 150)
    print(
        "GLOBAL PARETO FRONTIER"
    )
    print("=" * 150)


    for r in pareto:

        print(
            f"[{r['mcp_mm']:.1f},"
            f"{r['pip_mm']:.1f},"
            f"{r['dip_mm']:.1f}] "
            f"s={r['scale_um']:.2f} "
            f"k={r['tendon_k']:.0f}   "
            f"free={r['free_combined']:.3f}   "
            f"contact={r['contact_rmse']:.3f}   "
            f"pose=("
            f"{r['contact_mcp']:.2f},"
            f"{r['contact_pip']:.2f},"
            f"{r['contact_dip']:.2f})"
        )


    # ========================================================
    # Strict constant-model gate
    # ========================================================

    qualifying = [
        r
        for r in results
        if (
            r["free_combined"]
            <= FREE_LIMIT
            and
            r["contact_pip"]
            >= MIN_CONTACT_PIP
            and
            r["contact_rmse"]
            <= MAX_CONTACT_RMSE
            and
            sequence_valid(
                r
            )
        )
    ]


    qualifying.sort(
        key=lambda r:
            (
                r["contact_rmse"],
                r["free_combined"],
            )
    )


    print()
    print("=" * 150)
    print(
        "STRICT CONSTANT-MODEL GATE"
    )
    print("=" * 150)


    if qualifying:

        print()
        print(
            f"PASSING CANDIDATES = "
            f"{len(qualifying)}"
        )


        for rank, r in enumerate(
            qualifying[:10],
            start=1,
        ):

            print()
            print(
                f"#{rank}"
            )

            print(
                f"  transmission = "
                f"[{r['mcp_mm']:.1f}, "
                f"{r['pip_mm']:.1f}, "
                f"{r['dip_mm']:.1f}] mm"
            )

            print(
                f"  scale = "
                f"{r['scale_um']:.2f} um/tick"
            )

            print(
                f"  tendon k = "
                f"{r['tendon_k']:.0f} N/m"
            )

            print(
                f"  free = "
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


    else:

        print()
        print(
            "NO candidate passed:"
        )

        print(
            f"  free <= {FREE_LIMIT:.3f} deg"
        )

        print(
            f"  contact PIP >= "
            f"{MIN_CONTACT_PIP:.1f} deg"
        )

        print(
            f"  contact RMSE <= "
            f"{MAX_CONTACT_RMSE:.1f} deg"
        )

        print(
            f"  middle contact duty >= "
            f"{MIN_MIDDLE_DUTY:.0f}%"
        )


    # ========================================================
    # Interpretation
    # ========================================================

    print()
    print("=" * 150)
    print(
        "INTERPRETATION"
    )
    print("=" * 150)


    if qualifying:

        print()
        print(
            "A re-identified CONSTANT effective "
            "transmission model can satisfy both "
            "the free-space and contact diagnostic gates."
        )

        print()
        print(
            "Do NOT introduce configuration-dependent "
            "transmission yet."
        )

        print(
            "Next: locally refine the passing region and "
            "validate against held-out trajectories."
        )


    else:

        best_free_contact = None


        if free_preserving:

            best_free_contact = min(
                free_preserving,
                key=lambda r:
                    r["contact_rmse"],
            )


        print()
        print(
            "No locally re-identified constant model "
            "satisfied both regimes."
        )


        if best_free_contact is not None:

            print()
            print(
                "Best contact result that still preserves "
                "the free-space gate:"
            )

            print(
                f"  transmission = "
                f"[{best_free_contact['mcp_mm']:.1f}, "
                f"{best_free_contact['pip_mm']:.1f}, "
                f"{best_free_contact['dip_mm']:.1f}] mm"
            )

            print(
                f"  scale = "
                f"{best_free_contact['scale_um']:.2f} um/tick"
            )

            print(
                f"  tendon k = "
                f"{best_free_contact['tendon_k']:.0f} N/m"
            )

            print(
                f"  free = "
                f"{best_free_contact['free_combined']:.3f} deg"
            )

            print(
                "  contact pose = "
                f"({best_free_contact['contact_mcp']:.3f}, "
                f"{best_free_contact['contact_pip']:.3f}, "
                f"{best_free_contact['contact_dip']:.3f}) deg"
            )

            print(
                f"  contact RMSE = "
                f"{best_free_contact['contact_rmse']:.3f} deg"
            )


        print()
        print(
            "If the Pareto frontier still shows the same "
            "systematic free/contact trade-off, the next "
            "structural hypothesis is a configuration-"
            "dependent effective transmission."
        )

        print()
        print(
            "That would be introduced because the constant "
            "model was falsified across passive mechanics, "
            "constant ratios, and local transmission "
            "re-identification—not simply because one "
            "parameter set fit poorly."
        )


    print()
    print(
        "Summary CSV:"
    )

    print(
        RESULT_CSV
    )

    print()
    print("=" * 150)


if __name__ == "__main__":
    main()
