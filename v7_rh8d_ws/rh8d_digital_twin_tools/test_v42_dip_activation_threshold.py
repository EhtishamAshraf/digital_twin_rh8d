#!/usr/bin/env python3

from pathlib import Path
import csv
import math
import xml.etree.ElementTree as ET

import mujoco

import fit_spatial_free_and_test_contact_v35p7 as p7
import fit_spatial_guide_geometry_v37 as v37


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = WS / "src/ur5e_mujoco_description/mjcf"

SOURCE = (
    MJCF
    / "ur5e_rh8d_index_spatial_v38_final.xml"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v42_dip_activation_threshold"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTCSV = (
    OUTDIR
    / "v42_summary.csv"
)

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

FLEXOR_K = 2831.47


# ============================================================
# READ CURRENT MECHANICS
# ============================================================

def read_joint(
    root,
    name,
):

    j = root.find(
        f".//joint[@name='{name}']"
    )

    if j is None:
        raise RuntimeError(
            f"Missing joint {name}"
        )

    def f(attr, default=0.0):

        value = j.get(attr)

        if value is None:
            return default

        return float(
            value.split()[0]
        )

    return {
        "k":
            f("stiffness"),

        "friction":
            f("frictionloss"),

        "springref":
            f("springref"),
    }


ROOT = ET.parse(
    SOURCE
).getroot()

JM = read_joint(
    ROOT,
    MCP,
)

JP = read_joint(
    ROOT,
    PIP,
)

JD = read_joint(
    ROOT,
    DIP,
)


def threshold(j):

    return (
        j["friction"]
        +
        j["k"]
        *
        abs(
            j["springref"]
        )
    )


MCP_THRESHOLD = threshold(
    JM
)

PIP_THRESHOLD = threshold(
    JP
)

DIP_BASE = JD[
    "friction"
]


# ============================================================
# TEST VALUES
#
# Values are deliberately derived from the V41 experiment.
#
# DIP springref stays ZERO.
#
# We reproduce the initial resistance that V41 created using
# preload, but now as a pure motion threshold.
# ============================================================

CANDIDATES = sorted(
    set([
        DIP_BASE,

        DIP_BASE
        +
        JD["k"] * 0.03,

        DIP_BASE
        +
        JD["k"] * 0.06,

        PIP_THRESHOLD,

        DIP_BASE
        +
        JD["k"] * 0.09,

        1.15
        *
        PIP_THRESHOLD,

        DIP_BASE
        +
        JD["k"] * 0.12,
    ])
)


# ============================================================
# CONTACT PIPELINE
# ============================================================

v37.p3.b2.v30.v24.ab.patch_replay_modules()

CONTACT_REPLAY = (
    v37.p3.b2.v30.v24.base.contact_replay
)

CONTACT_REPLAY.CONTACT_DAMPING = (
    v37.p3.CONTACT_DAMPING
)

v37.p3.b2.v35.CONTACT_MU = (
    v37.p3.CONTACT_MU
)

v37.p3.b2.v35.CONTACT_K = (
    v37.p3.CONTACT_K
)


# ============================================================
# MODEL
# ============================================================

def build_candidate(
    friction,
    output,
):

    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()

    j = root.find(
        f".//joint[@name='{DIP}']"
    )

    if j is None:

        raise RuntimeError(
            "DIP joint not found."
        )


    # Critical:
    # no DIP elastic preload in this experiment.
    j.set(
        "springref",
        "0"
    )

    j.set(
        "frictionloss",
        f"{float(friction):.12g}",
    )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


    mujoco.MjModel.from_xml_path(
        str(output)
    )


def run_contact(
    xml,
    label,
):

    old_xml = (
        v37.p3.XML
    )

    try:

        v37.p3.XML = (
            xml
        )

        result = (
            v37.p3.run_case(
                label,
                FLEXOR_K,
                True,
                CONTACT_REPLAY,
            )
        )

    finally:

        v37.p3.XML = (
            old_xml
        )


    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 165)
    print(
        "V42 — DIP ACTIVATION-THRESHOLD TEST"
    )
    print("=" * 165)

    print()
    print(
        "This distinguishes activation threshold "
        "from continuous DIP preload."
    )


    print()
    print("V38 approximate initial resistance:")

    print(
        f"  MCP = "
        f"{MCP_THRESHOLD:.6f} Nm"
    )

    print(
        f"  PIP = "
        f"{PIP_THRESHOLD:.6f} Nm"
    )

    print(
        f"  DIP = "
        f"{DIP_BASE:.6f} Nm"
    )


    print()
    print(
        "Current ordering:"
    )

    print(
        "  DIP < MCP < PIP"
    )

    print()
    print(
        "Hardware/video hypothesis:"
    )

    print(
        "  MCP -> PIP -> DIP"
    )


    rows = []


    for i, friction in enumerate(
        CANDIDATES,
        1,
    ):

        xml = (
            MJCF
            /
            f"tmp_v42_dip_friction_{friction:.6f}.xml"
        )


        build_candidate(
            friction,
            xml,
        )


        print()
        print("-" * 165)

        print(
            f"[{i}/{len(CANDIDATES)}] "
            f"DIP friction threshold = "
            f"{friction:.6f} Nm"
        )


        fm, fp, fc, _ = (
            p7.run_free(
                xml
            )
        )


        cr = run_contact(
            xml,
            f"V42_{i}",
        )


        row = {
            "dip_friction":
                friction,

            "free_mcp":
                fm,

            "free_pip":
                fp,

            "free_combined":
                fc,

            "mcp":
                cr["mcp_deg"],

            "pip":
                cr["pip_deg"],

            "dip":
                cr["dip_deg"],

            "contact_rmse":
                cr["rmse_deg"],

            "mid_on":
                cr["mid_on"],

            "tip_on":
                cr["tip_on"],

            "dpip":
                cr[
                    "middle_to_tip_dPIP_deg"
                ],

            "ddip":
                cr[
                    "middle_to_tip_dDIP_deg"
                ],

            "mid_gap":
                cr[
                    "target_mid_gap_mm"
                ],

            "tip_gap":
                cr[
                    "target_tip_gap_mm"
                ],
        }


        rows.append(
            row
        )


        print(
            "FREE MCP/PIP/combined = "
            f"{fm:.3f} / "
            f"{fp:.3f} / "
            f"{fc:.3f} deg"
        )

        print(
            "CONTACT pose = "
            f"{cr['mcp_deg']:.3f} / "
            f"{cr['pip_deg']:.3f} / "
            f"{cr['dip_deg']:.3f} deg"
        )

        print(
            f"CONTACT RMSE = "
            f"{cr['rmse_deg']:.3f} deg"
        )

        print(
            f"M/T = "
            f"{cr['mid_on']} / "
            f"{cr['tip_on']}"
        )

        print(
            "dPIP/dDIP = "
            f"{cr['middle_to_tip_dPIP_deg']:+.3f} / "
            f"{cr['middle_to_tip_dDIP_deg']:+.3f} deg"
        )


    with OUTCSV.open(
        "w",
        newline="",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    print()
    print("=" * 175)
    print("V42 SUMMARY")
    print("=" * 175)

    print(
        "DIPthreshold  freeRMSE     "
        "MCP / PIP / DIP                "
        "contactRMSE  M/T      dPIP      dDIP"
    )

    print("-" * 175)


    for r in rows:

        print(
            f"{r['dip_friction']:11.6f}  "
            f"{r['free_combined']:8.3f}    "
            f"{r['mcp']:6.2f} / "
            f"{r['pip']:6.2f} / "
            f"{r['dip']:6.2f}             "
            f"{r['contact_rmse']:8.3f}   "
            f"{r['mid_on']}/{r['tip_on']}   "
            f"{r['dpip']:+8.3f} "
            f"{r['ddip']:+8.3f}"
        )


    valid = [
        r
        for r in rows
        if (
            r["mid_on"] >= 1
            and
            r["tip_on"] >= 1
            and
            math.isfinite(
                r["ddip"]
            )
            and
            r["ddip"] > 0.0
        )
    ]


    if valid:

        best_contact = min(
            valid,
            key=lambda r:
                r[
                    "contact_rmse"
                ],
        )


        best_balance = min(
            valid,
            key=lambda r:
                (
                    r[
                        "free_combined"
                    ]
                    +
                    r[
                        "contact_rmse"
                    ]
                ),
        )


        print()
        print("=" * 175)
        print("BEST CONTACT")
        print("=" * 175)

        print(
            f"DIP threshold = "
            f"{best_contact['dip_friction']:.6f} Nm"
        )

        print(
            f"free/contact = "
            f"{best_contact['free_combined']:.3f} / "
            f"{best_contact['contact_rmse']:.3f} deg"
        )

        print(
            "pose = "
            f"{best_contact['mcp']:.3f} / "
            f"{best_contact['pip']:.3f} / "
            f"{best_contact['dip']:.3f}"
        )


        print()
        print("=" * 175)
        print("BEST FREE + CONTACT BALANCE")
        print("=" * 175)

        print(
            f"DIP threshold = "
            f"{best_balance['dip_friction']:.6f} Nm"
        )

        print(
            f"free/contact = "
            f"{best_balance['free_combined']:.3f} / "
            f"{best_balance['contact_rmse']:.3f} deg"
        )

        print(
            "pose = "
            f"{best_balance['mcp']:.3f} / "
            f"{best_balance['pip']:.3f} / "
            f"{best_balance['dip']:.3f}"
        )


    print()
    print(
        f"CSV:\n{OUTCSV}"
    )

    print("=" * 175)


if __name__ == "__main__":
    main()
