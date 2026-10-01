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
# PURPOSE
#
# Fair local re-identification of the V6 spatial tendon.
#
# Frozen:
#   - V2/V6 routing geometry
#   - V5 fixture registration
#   - contact mask
#   - passive joint mechanics
#   - motor delay/rates
#   - collision geometry
#
# Re-identify ONLY:
#   - encoder -> spool scale
#   - tendon stiffness
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
    / "ur5e_rh8d_index_configdep_spool_v6.xml"
)

BEST_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_configdep_spool_v7_candidate.xml"
)

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "configdep_scale_stiffness_v7"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY = (
    RESULT_DIR
    / "summary.csv"
)


TENDON_NAME = "idx_flex_spatial"


# ============================================================
# SEARCH
#
# 5.1/2100 is retained as the old control.
#
# 6.2-6.35 um/tick is motivated directly by the change in
# open MCP/PIP moment arms:
#
#   5.1 * 5.0 / 4.015 ~= 6.35
#   5.1 * 4.0 / 3.275 ~= 6.23
#
# 6.4 remains below the ~22 mm spool travel at the maximum
# encoder range used in the free-space experiment.
# ============================================================

SCALES_UM = [
    5.10,
    5.80,
    6.00,
    6.20,
    6.30,
    6.40,
]

TENDON_K = [
    1600.0,
    1900.0,
    2100.0,
    2400.0,
    2700.0,
]


# Trusted constant-model dynamic baseline.
TRUSTED_FREE_RMSE = 5.085

FREE_LIMIT = (
    TRUSTED_FREE_RMSE
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


# V6 un-refitted result from the previous A/B run.
V6_UNREFITTED_PIP = 3.378
V6_UNREFITTED_CONTACT_RMSE = 13.249


def set_tendon_stiffness(
    source,
    output,
    stiffness,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()


    found = False


    tendon_section = root.find(
        "tendon"
    )


    if tendon_section is None:

        raise RuntimeError(
            "<tendon> section missing."
        )


    for tendon in list(
        tendon_section
    ):

        if (
            tendon.attrib.get("name")
            == TENDON_NAME
        ):

            tendon.set(
                "stiffness",
                f"{stiffness:.10g}",
            )

            found = True
            break


    if not found:

        raise RuntimeError(
            f"Missing tendon: "
            f"{TENDON_NAME}"
        )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


def run_candidate(
    contact_xml,
    free_xml,
    free_bag,
    contact_bag,
    scale_um,
    tendon_k,
):

    old_scale = (
        base.METERS_PER_TICK
    )

    old_k = (
        base.TENDON_STIFFNESS
    )


    try:

        base.METERS_PER_TICK = (
            scale_um
            * 1.0e-6
        )

        base.TENDON_STIFFNESS = (
            tendon_k
        )


        (
            free_metrics,
            free_log,
        ) = base.run_free(
            free_xml,
            free_bag,
        )


        (
            contact_metrics,
            contact_log,
        ) = base.run_contact(
            contact_xml,
            contact_bag,
        )


    finally:

        base.METERS_PER_TICK = (
            old_scale
        )

        base.TENDON_STIFFNESS = (
            old_k
        )


    return (
        free_metrics,
        free_log,
        contact_metrics,
        contact_log,
    )


def label_for(
    scale,
    k,
):

    return (
        f"s{scale:.2f}_k{k:.0f}"
        .replace(
            ".",
            "p",
        )
    )


def save_best_model(
    scale_um,
    tendon_k,
):

    set_tendon_stiffness(
        SOURCE,
        BEST_MODEL,
        tendon_k,
    )


    # Scale is intentionally NOT encoded in the XML because it
    # belongs to the hardware encoder -> spool interface.
    sidecar = (
        BEST_MODEL.with_suffix(
            ".txt"
        )
    )


    sidecar.write_text(
        (
            "V7 provisional configuration-dependent "
            "candidate\n"
            f"encoder_spool_scale_um_per_tick="
            f"{scale_um:.6f}\n"
            f"tendon_stiffness_N_per_m="
            f"{tendon_k:.6f}\n"
            "NOT FINAL IDENTIFIED MODEL\n"
        )
    )


def read_constant_reference():

    path = (
        WS
        / "rh8d_digital_twin_tools/results/"
          "configdep_replay_v6/summary.csv"
    )


    if not path.exists():

        return None


    with open(
        path
    ) as f:

        for row in csv.DictReader(
            f
        ):

            if (
                row[
                    "model"
                ].upper()
                == "CONSTANT"
            ):

                return {
                    key:
                        float(value)
                    if (
                        value
                        not in (
                            "",
                            "None",
                        )
                        and
                        key
                        != "model"
                    )
                    else value

                    for key, value
                    in row.items()
                }


    return None


def main():

    import sys


    if len(
        sys.argv
    ) != 3:

        print(
            "Usage:\n"
            " python3 "
            "scan_index_configdep_scale_stiffness_v7.py "
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


    if not SOURCE.exists():

        raise RuntimeError(
            f"Missing V6 source:\n"
            f"{SOURCE}"
        )


    if not free_bag.exists():

        raise RuntimeError(
            f"Missing free bag:\n"
            f"{free_bag}"
        )


    if not contact_bag.exists():

        raise RuntimeError(
            f"Missing contact bag:\n"
            f"{contact_bag}"
        )


    # Ensure old contact replay helpers use the frozen V5
    # fixture registration, rather than the older V4 value.
    patched = ab.patch_replay_modules()


    print()
    print("=" * 125)
    print(
        "RH8D INDEX — CONFIGURATION-DEPENDENT "
        "SCALE + STIFFNESS RE-ID V7"
    )
    print("=" * 125)


    print()
    print(
        "Frozen model structure:"
    )

    print(
        "  V6 spatial tendon geometry"
    )

    print(
        "  V5 bottle registration/contact mask"
    )

    print(
        "  existing passive joint mechanics"
    )

    print(
        "  existing motor delay/rates"
    )


    print()
    print(
        f"Scale candidates = "
        f"{SCALES_UM}"
    )

    print(
        f"Tendon-k candidates = "
        f"{TENDON_K}"
    )


    print()
    print(
        f"Free-space preservation gate = "
        f"{FREE_LIMIT:.3f} deg"
    )


    if patched:

        print()
        print(
            "Fixture constants patched to V5:"
        )

        for (
            module_name,
            variable,
            old,
            new,
        ) in patched:

            print(
                f"  {module_name}.{variable}: "
                f"{old} -> {new}"
            )


    results = []

    total = (
        len(
            SCALES_UM
        )
        *
        len(
            TENDON_K
        )
    )

    count = 0


    for scale_um in SCALES_UM:

        for tendon_k in TENDON_K:

            count += 1


            label = label_for(
                scale_um,
                tendon_k,
            )


            contact_xml = (
                MJCF_DIR
                /
                (
                    "tmp_v7_contact_"
                    f"{label}.xml"
                )
            )


            free_xml = (
                MJCF_DIR
                /
                (
                    "tmp_v7_free_"
                    f"{label}.xml"
                )
            )


            set_tendon_stiffness(
                SOURCE,
                contact_xml,
                tendon_k,
            )


            ab.make_free_model(
                contact_xml,
                free_xml,
            )


            # Compile before expensive replay.
            mujoco.MjModel.from_xml_path(
                str(
                    contact_xml
                )
            )

            mujoco.MjModel.from_xml_path(
                str(
                    free_xml
                )
            )


            print()
            print(
                f"[{count:02d}/{total}] "
                f"scale={scale_um:.2f} um/tick  "
                f"k={tendon_k:.0f} N/m"
            )


            try:

                (
                    free_metrics,
                    free_log,
                    contact_metrics,
                    contact_log,
                ) = run_candidate(
                    contact_xml,
                    free_xml,
                    free_bag,
                    contact_bag,
                    scale_um,
                    tendon_k,
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
                    f"{pose[0]:.2f}, "
                    f"{pose[1]:.2f}, "
                    f"{pose[2]:.2f}"
                    f") deg"
                )

                print(
                    f"  contact RMSE = "
                    f"{result['contact_rmse']:.3f} deg"
                )


            finally:

                for path in [
                    contact_xml,
                    free_xml,
                ]:

                    try:
                        path.unlink()

                    except FileNotFoundError:
                        pass


    # ========================================================
    # SAVE CSV
    # ========================================================

    if not results:

        raise RuntimeError(
            "No candidate completed."
        )


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
    # CONTROL
    # ========================================================

    control = min(
        results,
        key=lambda r:
            (
                abs(
                    r[
                        "scale_um"
                    ]
                    - 5.10
                )
                +
                abs(
                    r[
                        "tendon_k"
                    ]
                    - 2100.0
                )
                / 1000.0
            ),
    )


    print()
    print("=" * 125)
    print(
        "V6 CONTROL CHECK"
    )
    print("=" * 125)


    print()
    print(
        "scale = 5.10 um/tick"
    )

    print(
        "k = 2100 N/m"
    )

    print(
        f"free = "
        f"{control['free_combined']:.3f} deg"
    )

    print(
        "contact = "
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
    # FREE-PRESERVING CANDIDATES
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
    print("=" * 125)
    print(
        "FREE-SPACE-PRESERVING "
        "CONFIGURATION-DEPENDENT CANDIDATES"
    )
    print(
        f"(free <= {FREE_LIMIT:.3f} deg)"
    )
    print("=" * 125)


    if not preserving:

        print()
        print(
            "NONE."
        )


    else:

        for rank, r in enumerate(
            preserving[
                :10
            ],
            start=1,
        ):

            print()
            print(
                f"#{rank} "
                f"scale={r['scale_um']:.2f} "
                f"k={r['tendon_k']:.0f}"
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
    # BEST OVERALL CONTACT
    # ========================================================

    best_contact = min(
        results,
        key=lambda r:
            r[
                "contact_rmse"
            ],
    )


    print()
    print("=" * 125)
    print(
        "BEST CONTACT RESULT "
        "REGARDLESS OF FREE-SPACE"
    )
    print("=" * 125)


    print()
    print(
        f"scale = "
        f"{best_contact['scale_um']:.2f} um/tick"
    )

    print(
        f"k = "
        f"{best_contact['tendon_k']:.0f} N/m"
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
    # PICK BEST FAIR CANDIDATE
    # ========================================================

    selected = (
        preserving[
            0
        ]
        if preserving
        else best_contact
    )


    save_best_model(
        selected[
            "scale_um"
        ],
        selected[
            "tendon_k"
        ],
    )


    # ========================================================
    # TWO CLOSE-UP IMAGES
    # ========================================================

    print()
    print("=" * 125)
    print(
        "TWO CLOSE-UP COMPARISON IMAGES"
    )
    print("=" * 125)


    # Reuse A/B renderer but put output here.
    old_ab_result_dir = (
        ab.RESULT_DIR
    )

    old_ab_scale = (
        ab.METERS_PER_TICK
    )


    try:

        ab.RESULT_DIR = (
            RESULT_DIR
        )


        # ----------------------------------------------------
        # CONSTANT REFERENCE IMAGE FROM PREVIOUS FAIR A/B
        # ----------------------------------------------------

        constant_reference = (
            read_constant_reference()
        )


        if (
            constant_reference
            is not None
        ):

            constant_model = (
                MJCF_DIR
                /
                "tmp_v7_constant_reference.xml"
            )


            ab.replace_index_with_constant_tendon(
                ab.V5_SOURCE,
                constant_model,
            )


            ab.METERS_PER_TICK = (
                5.10e-6
            )


            ab.render_hold(
                constant_model,
                np.array(
                    [
                        constant_reference[
                            "contact_mcp"
                        ],
                        constant_reference[
                            "contact_pip"
                        ],
                        constant_reference[
                            "contact_dip"
                        ],
                    ]
                ),
                "CONSTANT_REFERENCE",
                constant_reference[
                    "contact_rmse"
                ],
            )


            try:
                constant_model.unlink()
            except FileNotFoundError:
                pass


        # ----------------------------------------------------
        # SELECTED CONFIGDEP CANDIDATE
        # ----------------------------------------------------

        ab.METERS_PER_TICK = (
            selected[
                "scale_um"
            ]
            * 1.0e-6
        )


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
            "V7_BEST_CONFIGDEP",
            selected[
                "contact_rmse"
            ],
        )


    finally:

        ab.RESULT_DIR = (
            old_ab_result_dir
        )

        ab.METERS_PER_TICK = (
            old_ab_scale
        )


    # ========================================================
    # INTERPRETATION NUMBERS
    # ========================================================

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


    selected_error = (
        selected_pose
        - HW_POSE
    )


    pip_gain_vs_v6 = (
        abs(
            V6_UNREFITTED_PIP
            - HW_POSE[
                1
            ]
        )
        -
        abs(
            selected[
                "contact_pip"
            ]
            - HW_POSE[
                1
            ]
        )
    )


    print()
    print("=" * 125)
    print(
        "SELECTED V7 CANDIDATE"
    )
    print("=" * 125)


    print()
    print(
        f"scale = "
        f"{selected['scale_um']:.2f} um/tick"
    )

    print(
        f"tendon k = "
        f"{selected['tendon_k']:.0f} N/m"
    )


    print()
    print(
        f"free combined RMSE = "
        f"{selected['free_combined']:.3f} deg"
    )


    print()
    print(
        "contact predicted pose:"
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
        "contact error SIM-HW:"
    )

    print(
        f"  MCP = "
        f"{selected_error[0]:+.3f} deg"
    )

    print(
        f"  PIP = "
        f"{selected_error[1]:+.3f} deg"
    )

    print(
        f"  DIP = "
        f"{selected_error[2]:+.3f} deg"
    )


    print()
    print(
        f"contact RMSE = "
        f"{selected['contact_rmse']:.3f} deg"
    )


    print()
    print(
        f"PIP absolute-error reduction "
        f"relative to un-refitted V6 = "
        f"{pip_gain_vs_v6:+.3f} deg"
    )


    print()
    print(
        "Candidate model:"
    )

    print(
        BEST_MODEL
    )


    print()
    print(
        "Summary CSV:"
    )

    print(
        SUMMARY
    )


    print()
    print(
        "Close-ups:"
    )

    print(
        RESULT_DIR
        /
        "constant_reference_hold_closeup.png"
    )

    print(
        RESULT_DIR
        /
        "v7_best_configdep_hold_closeup.png"
    )


    print()
    print("=" * 125)


if __name__ == "__main__":
    main()
