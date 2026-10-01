#!/usr/bin/env python3

import sys
import re
import csv
import io
import contextlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import replay_index_dynamic_v1 as free_replay
import replay_index_tabletop_hardware_contact_v1 as contact_replay


# ============================================================
# PURPOSE
#
# SAME PIP candidate is evaluated with:
#
#   1. validated full dynamic free-space replay
#      command -> delay -> motor rate -> spool -> tendon -> finger
#
#   2. actual hardware encoder contact replay
#
# No identified/final XML is overwritten.
# ============================================================


FREE_SOURCE = free_replay.MODEL_PATH
CONTACT_SOURCE = contact_replay.MODEL_PATH


PIP_NAME = "Index_Middle--Index_Proximal"
TENDON_NAME = "idx_flex_spatial"


# ============================================================
# Freeze Candidate A
# ============================================================

METERS_PER_TICK = 5.1e-6
TENDON_STIFFNESS = 2100.0


# ============================================================
# Freeze tabletop/contact calibration
# ============================================================

BOTTLE_POS = np.array(
    [
         0.503422950,
        -0.372517762,
         0.799000000,
    ],
    dtype=float,
)

CONTACT_MU = 0.05
CONTACT_STIFFNESS = 1000.0


# ============================================================
# PIP passive grid
#
# Current:
#   stiffness    = 0.031 Nm/rad
#   frictionloss = 0.006 Nm
# ============================================================

PIP_STIFFNESSES = [
    0.0000,
    0.0030,
    0.0075,
    0.0150,
    0.0230,
    0.0310,
]

PIP_FRICTIONS = [
    0.0000,
    0.0005,
    0.0015,
    0.0030,
    0.0060,
]


RESULT_DIR = (
    free_replay.WS
    / "rh8d_digital_twin_tools/results/"
      "pip_passive_dynamic_v2"
)

RESULT_CSV = (
    RESULT_DIR
    / "summary.csv"
)


NUM = (
    r"[-+]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)


def label_for(k, f):

    text = (
        f"k{k:.4f}_f{f:.4f}"
    )

    return text.replace(
        ".",
        "p",
    )


def modify_candidate_xml(
    source,
    output,
    pip_stiffness,
    pip_friction,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()


    found_pip = False
    found_tendon = False


    for element in root.iter():

        # ----------------------------------------------------
        # Actual PIP joint definition.
        #
        # Do NOT accidentally modify the <fixed><joint joint=...>
        # transmission element.
        # ----------------------------------------------------

        if (
            element.tag == "joint"
            and
            element.attrib.get("name")
            == PIP_NAME
        ):

            element.set(
                "stiffness",
                f"{pip_stiffness:.10g}",
            )

            element.set(
                "frictionloss",
                f"{pip_friction:.10g}",
            )

            found_pip = True


        # ----------------------------------------------------
        # Candidate-A tendon stiffness.
        # ----------------------------------------------------

        if (
            element.attrib.get("name")
            == TENDON_NAME
        ):

            element.set(
                "stiffness",
                f"{TENDON_STIFFNESS:.10g}",
            )

            found_tendon = True


    if not found_pip:

        raise RuntimeError(
            f"Could not find PIP joint in:\n"
            f"{source}"
        )


    if not found_tendon:

        raise RuntimeError(
            f"Could not find tendon "
            f"{TENDON_NAME} in:\n"
            f"{source}"
        )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


def regex_float(
    text,
    pattern,
):

    match = re.search(
        pattern,
        text,
        re.MULTILINE,
    )

    if match is None:

        raise RuntimeError(
            "Could not parse expected replay output:\n"
            + pattern
        )


    return float(
        match.group(1)
    )


def parse_free_output(
    text,
):

    return {

        "motor_rmse":
            regex_float(
                text,
                rf"encoder RMSE\s*=\s*({NUM})\s*ticks",
            ),

        "mcp_rmse":
            regex_float(
                text,
                rf"MCP RMSE\s*=\s*({NUM})\s*deg",
            ),

        "pip_rmse":
            regex_float(
                text,
                rf"PIP RMSE\s*=\s*({NUM})\s*deg",
            ),

        "combined":
            regex_float(
                text,
                rf"combined MCP/PIP\s*=\s*({NUM})\s*deg",
            ),
    }


def parse_contact_output(
    text,
):

    pose_match = re.search(
        rf"SIM pose\s*=\s*"
        rf"\(\s*({NUM})\s*,\s*"
        rf"({NUM})\s*,\s*"
        rf"({NUM})\s*\)\s*deg",
        text,
        re.MULTILINE,
    )


    if pose_match is None:

        raise RuntimeError(
            "Could not parse contact SIM pose."
        )


    pose = np.array(
        [
            float(
                pose_match.group(1)
            ),
            float(
                pose_match.group(2)
            ),
            float(
                pose_match.group(3)
            ),
        ],
        dtype=float,
    )


    rmse = regex_float(
        text,
        rf"3-joint pose RMSE\s*=\s*({NUM})\s*deg",
    )


    def optional_float(
        pattern,
    ):

        m = re.search(
            pattern,
            text,
            re.MULTILINE,
        )

        if m is None:
            return None

        return float(
            m.group(1)
        )


    return {

        "pose":
            pose,

        "rmse":
            rmse,

        "dmcp":
            optional_float(
                rf"delta MCP\s*=\s*({NUM})\s*deg"
            ),

        "dpip":
            optional_float(
                rf"delta PIP\s*=\s*({NUM})\s*deg"
            ),

        "ddip":
            optional_float(
                rf"delta DIP\s*=\s*({NUM})\s*deg"
            ),

        "middle_duty":
            optional_float(
                rf"Middle contact duty\s*=\s*({NUM})%"
            ),
    }


def run_free(
    candidate_xml,
    free_bag,
):

    old_model = (
        free_replay.MODEL_PATH
    )

    old_scale = (
        free_replay.METERS_PER_TICK
    )

    old_argv = list(
        sys.argv
    )


    stream = io.StringIO()


    try:

        free_replay.MODEL_PATH = (
            candidate_xml
        )

        free_replay.METERS_PER_TICK = (
            METERS_PER_TICK
        )

        sys.argv = [
            "replay_index_dynamic_v1.py",
            str(
                free_bag
            ),
        ]


        with contextlib.redirect_stdout(
            stream
        ):

            free_replay.main()


    finally:

        free_replay.MODEL_PATH = (
            old_model
        )

        free_replay.METERS_PER_TICK = (
            old_scale
        )

        sys.argv = (
            old_argv
        )


    text = stream.getvalue()

    return (
        parse_free_output(
            text
        ),
        text,
    )


def run_contact(
    candidate_xml,
    contact_bag,
):

    old_model = (
        contact_replay.MODEL_PATH
    )

    old_argv = list(
        sys.argv
    )


    # Save globals too, even though we explicitly freeze them.
    old_scale = (
        contact_replay.SCALE
    )

    old_tendon = (
        contact_replay.TENDON_STIFFNESS
    )

    old_mu = (
        contact_replay.MU
    )

    old_contact_k = (
        contact_replay.CONTACT_STIFFNESS
    )

    old_bottle = (
        contact_replay.BOTTLE_POS.copy()
    )


    stream = io.StringIO()


    try:

        contact_replay.MODEL_PATH = (
            candidate_xml
        )

        contact_replay.SCALE = (
            METERS_PER_TICK
        )

        contact_replay.TENDON_STIFFNESS = (
            TENDON_STIFFNESS
        )

        contact_replay.MU = (
            CONTACT_MU
        )

        contact_replay.CONTACT_STIFFNESS = (
            CONTACT_STIFFNESS
        )

        contact_replay.BOTTLE_POS = (
            BOTTLE_POS.copy()
        )


        sys.argv = [
            "replay_index_tabletop_hardware_contact_v1.py",
            str(
                contact_bag
            ),
        ]


        with contextlib.redirect_stdout(
            stream
        ):

            contact_replay.main()


    finally:

        contact_replay.MODEL_PATH = (
            old_model
        )

        contact_replay.SCALE = (
            old_scale
        )

        contact_replay.TENDON_STIFFNESS = (
            old_tendon
        )

        contact_replay.MU = (
            old_mu
        )

        contact_replay.CONTACT_STIFFNESS = (
            old_contact_k
        )

        contact_replay.BOTTLE_POS = (
            old_bottle
        )

        sys.argv = (
            old_argv
        )


    text = stream.getvalue()

    return (
        parse_contact_output(
            text
        ),
        text,
    )


def main():

    if len(
        sys.argv
    ) != 3:

        print()
        print(
            "Usage:"
        )

        print(
            " python3 "
            "scan_index_pip_passive_dynamic_v2.py "
            "/path/to/free_space_rosbag "
            "/path/to/contact_rosbag"
        )

        sys.exit(
            1
        )


    free_bag = Path(
        sys.argv[1]
    ).expanduser().resolve()

    contact_bag = Path(
        sys.argv[2]
    ).expanduser().resolve()


    if not free_bag.exists():

        raise RuntimeError(
            f"Free-space bag does not exist:\n"
            f"{free_bag}"
        )


    if not contact_bag.exists():

        raise RuntimeError(
            f"Contact bag does not exist:\n"
            f"{contact_bag}"
        )


    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Temporary MJCF files MUST remain beside their source
    # MJCF files because the models use relative mesh paths
    # such as ../meshes/...
    #
    # Putting candidate XMLs inside the results directory
    # breaks MuJoCo asset resolution.


    print()
    print("=" * 140)
    print(
        "RH8D INDEX — COMBINED DYNAMIC FREE-SPACE "
        "+ CONTACT PIP PASSIVE SCAN V2"
    )
    print("=" * 140)

    print()
    print(
        "Free-space evaluator:"
    )

    print(
        "  EXACT replay_index_dynamic_v1 pipeline"
    )

    print(
        f"  motor delay  = "
        f"{free_replay.MOTOR_DELAY*1000:.1f} ms"
    )

    print(
        f"  closing rate = "
        f"{free_replay.CLOSING_RATE:.1f} ticks/s"
    )

    print(
        f"  opening rate = "
        f"{free_replay.OPENING_RATE:.1f} ticks/s"
    )


    print()
    print(
        "Frozen transmission:"
    )

    print(
        f"  scale            = "
        f"{METERS_PER_TICK*1e6:.3f} um/tick"
    )

    print(
        f"  tendon stiffness = "
        f"{TENDON_STIFFNESS:.1f} N/m"
    )

    print(
        "  PIP moment arm   = 4.0 mm"
    )


    print()
    print(
        "Frozen contact setup:"
    )

    print(
        f"  bottle = "
        f"{BOTTLE_POS}"
    )

    print(
        f"  friction = "
        f"{CONTACT_MU:.3f}"
    )

    print(
        f"  diagnostic contact stiffness = "
        f"{CONTACT_STIFFNESS:.1f} N/m"
    )


    total = (
        len(
            PIP_STIFFNESSES
        )
        *
        len(
            PIP_FRICTIONS
        )
    )


    print()
    print(
        f"Candidates = {total}"
    )

    print()


    results = []


    count = 0


    for pip_k in PIP_STIFFNESSES:

        for pip_f in PIP_FRICTIONS:

            count += 1

            label = label_for(
                pip_k,
                pip_f,
            )


            # Keep temporary models in the SAME directory
            # as their source MJCF so relative mesh paths
            # continue to resolve correctly.
            free_xml = (
                FREE_SOURCE.parent
                /
                (
                    "tmp_pip_free_"
                    + label
                    + ".xml"
                )
            )

            contact_xml = (
                CONTACT_SOURCE.parent
                /
                (
                    "tmp_pip_contact_"
                    + label
                    + ".xml"
                )
            )


            modify_candidate_xml(
                FREE_SOURCE,
                free_xml,
                pip_k,
                pip_f,
            )

            modify_candidate_xml(
                CONTACT_SOURCE,
                contact_xml,
                pip_k,
                pip_f,
            )


            print(
                f"[{count:02d}/{total}] "
                f"k={pip_k:.4f} "
                f"f={pip_f:.4f} ...",
                flush=True,
            )


            free_metrics, free_log = (
                run_free(
                    free_xml,
                    free_bag,
                )
            )


            contact_metrics, contact_log = (
                run_contact(
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

                "k":
                    pip_k,

                "f":
                    pip_f,

                "free_motor":
                    free_metrics[
                        "motor_rmse"
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
                f"       free={result['free_combined']:.3f} deg   "
                f"contact={result['contact_rmse']:.3f} deg   "
                f"hold=("
                f"{result['contact_mcp']:.2f}, "
                f"{result['contact_pip']:.2f}, "
                f"{result['contact_dip']:.2f})"
            )


    # ========================================================
    # Save machine-readable summary
    # ========================================================

    fields = list(
        results[0].keys()
    )


    with open(
        RESULT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # ========================================================
    # Exact current baseline
    # ========================================================

    baseline = next(
        r
        for r in results
        if (
            abs(
                r["k"]
                - 0.031
            ) < 1e-12
            and
            abs(
                r["f"]
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
        "VALIDATED CURRENT BASELINE"
    )
    print("=" * 140)

    print()
    print(
        f"PIP stiffness    = "
        f"{baseline['k']:.4f}"
    )

    print(
        f"PIP frictionloss = "
        f"{baseline['f']:.4f}"
    )

    print()
    print(
        f"Dynamic free-space:"
    )

    print(
        f"  MCP RMSE      = "
        f"{baseline['free_mcp']:.3f} deg"
    )

    print(
        f"  PIP RMSE      = "
        f"{baseline['free_pip']:.3f} deg"
    )

    print(
        f"  combined      = "
        f"{baseline['free_combined']:.3f} deg"
    )


    print()
    print(
        f"Contact:"
    )

    print(
        f"  pose = "
        f"({baseline['contact_mcp']:.3f}, "
        f"{baseline['contact_pip']:.3f}, "
        f"{baseline['contact_dip']:.3f}) deg"
    )

    print(
        f"  RMSE = "
        f"{baseline['contact_rmse']:.3f} deg"
    )


    # ========================================================
    # Candidates that preserve trusted free-space model
    # ========================================================

    FREE_ALLOWANCE = 1.0


    eligible = [
        r
        for r in results
        if (
            r[
                "free_combined"
            ]
            <=
            baseline_free
            +
            FREE_ALLOWANCE
        )
    ]


    eligible = sorted(
        eligible,
        key=lambda r:
            r[
                "contact_rmse"
            ],
    )


    print()
    print("=" * 140)
    print(
        "FREE-SPACE-PRESERVING CANDIDATES"
    )

    print(
        f"(dynamic free RMSE <= baseline + "
        f"{FREE_ALLOWANCE:.1f} deg)"
    )

    print("=" * 140)


    for rank, r in enumerate(
        eligible[:15],
        start=1,
    ):

        print()
        print(
            f"#{rank}  "
            f"k={r['k']:.4f}  "
            f"friction={r['f']:.4f}"
        )

        print(
            f"  free MCP/PIP/combined = "
            f"{r['free_mcp']:.3f} / "
            f"{r['free_pip']:.3f} / "
            f"{r['free_combined']:.3f} deg"
        )

        print(
            f"  contact pose = "
            f"({r['contact_mcp']:.3f}, "
            f"{r['contact_pip']:.3f}, "
            f"{r['contact_dip']:.3f}) deg"
        )

        print(
            f"  contact RMSE = "
            f"{r['contact_rmse']:.3f} deg"
        )

        print(
            f"  redistribution "
            f"dMCP/dPIP/dDIP = "
            f"{r['dmcp']} / "
            f"{r['dpip']} / "
            f"{r['ddip']}"
        )

        print(
            f"  middle duty = "
            f"{r['middle_duty']} %"
        )


    # ========================================================
    # Zero-resistance diagnostic
    # ========================================================

    zero = next(
        r
        for r in results
        if (
            abs(
                r["k"]
            ) < 1e-12
            and
            abs(
                r["f"]
            ) < 1e-12
        )
    )


    print()
    print("=" * 140)
    print(
        "ZERO-PIP-RESISTANCE DIAGNOSTIC"
    )
    print("=" * 140)

    print()
    print(
        f"Dynamic free RMSE = "
        f"{zero['free_combined']:.3f} deg"
    )

    print(
        f"Contact pose = "
        f"({zero['contact_mcp']:.3f}, "
        f"{zero['contact_pip']:.3f}, "
        f"{zero['contact_dip']:.3f}) deg"
    )

    print(
        f"Contact RMSE = "
        f"{zero['contact_rmse']:.3f} deg"
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


    if not eligible:

        print()
        print(
            "No passive-PIP candidate preserved "
            "the validated free-space fit."
        )

    else:

        best = eligible[0]


        print()
        print(
            "Best free-space-preserving candidate:"
        )

        print(
            f"  stiffness    = "
            f"{best['k']:.4f}"
        )

        print(
            f"  frictionloss = "
            f"{best['f']:.4f}"
        )

        print(
            f"  dynamic free = "
            f"{best['free_combined']:.3f} deg"
        )

        print(
            f"  contact RMSE = "
            f"{best['contact_rmse']:.3f} deg"
        )

        print(
            f"  contact PIP  = "
            f"{best['contact_pip']:.3f} deg"
        )


        if (
            best[
                "contact_pip"
            ]
            >= 18.0
        ):

            print()
            print(
                "Passive PIP mechanics alone can produce "
                "a large hardware-like contact redistribution "
                "while retaining the dynamic free-space fit."
            )

            print(
                "Refine this passive region before changing "
                "the tendon transmission."
            )

        elif (
            zero[
                "contact_pip"
            ]
            >= 18.0
        ):

            print()
            print(
                "Low PIP resistance CAN generate the required "
                "contact flexion, but the candidates that do so "
                "do not preserve the validated free-space behavior."
            )

            print(
                "This is evidence of a free-space/contact tradeoff "
                "that constant passive PIP parameters alone cannot "
                "resolve."
            )

            print(
                "The next hypothesis should be the effective "
                "PIP tendon transmission / configuration-dependent "
                "underactuation."
            )

        else:

            print()
            print(
                "Even near-zero PIP resistance does not generate "
                "the required contact PIP motion."
            )

            print(
                "The fixed tendon transmission becomes the primary "
                "structural hypothesis."
            )


    print()
    print(
        "Summary CSV:"
    )

    print(
        RESULT_CSV
    )

    print()
    print(
        "Individual replay logs:"
    )

    print(
        RESULT_DIR
    )

    print()
    print("=" * 140)


if __name__ == "__main__":
    main()
