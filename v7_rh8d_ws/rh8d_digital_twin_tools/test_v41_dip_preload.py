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
    / "v41_dip_preload"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTCSV = OUTDIR / "v41_dip_preload_summary.csv"

DIP = "Index_Distal--Index_Middle"

FLEXOR_K = 2831.47


# ------------------------------------------------------------
# This is NOT a broad optimization.
#
# Negative springref = opening preload at q=0.
#
# V38 currently has DIP springref ~= 0.
# ------------------------------------------------------------

DIP_SPRINGREFS = [
     0.00,
    -0.03,
    -0.06,
    -0.09,
    -0.12,
    -0.15,
    -0.20,
    -0.25,
]


# ------------------------------------------------------------
# Contact pipeline: exactly the same one used for V37/V38.
# ------------------------------------------------------------

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


def build_candidate(
    springref,
    output,
):

    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()

    joint = root.find(
        f".//joint[@name='{DIP}']"
    )

    if joint is None:

        raise RuntimeError(
            f"Could not find DIP joint: {DIP}"
        )

    joint.set(
        "springref",
        f"{float(springref):.12g}",
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

    old_xml = v37.p3.XML

    try:

        v37.p3.XML = xml

        result = v37.p3.run_case(
            label,
            FLEXOR_K,
            True,
            CONTACT_REPLAY,
        )

    finally:

        v37.p3.XML = old_xml

    return result


def main():

    print()
    print("=" * 150)
    print(
        "V41 — AERO-INSPIRED DELAYED-DIP PRELOAD TEST"
    )
    print("=" * 150)

    print()
    print(
        "Only DIP springref changes."
    )

    print(
        "All flexor/contact/geometry parameters remain V38."
    )


    rows = []


    for i, springref in enumerate(
        DIP_SPRINGREFS
    ):

        # IMPORTANT:
        # Keep candidate MJCF beside the source MJCF so that
        # relative mesh/include paths such as ../meshes/... resolve
        # exactly as they do in V38.
        xml = (
            MJCF
            /
            f"tmp_v41_dip_springref_{springref:+.3f}.xml"
        )


        build_candidate(
            springref,
            xml,
        )


        print()
        print("-" * 150)

        print(
            f"[{i+1}/{len(DIP_SPRINGREFS)}] "
            f"DIP springref = "
            f"{springref:+.3f} rad"
        )


        fm, fp, fc, _ = p7.run_free(
            xml
        )


        cr = run_contact(
            xml,
            f"V41_{springref:+.3f}",
        )


        dpip = float(
            cr[
                "middle_to_tip_dPIP_deg"
            ]
        )

        ddip = float(
            cr[
                "middle_to_tip_dDIP_deg"
            ]
        )


        row = {
            "springref_rad":
                springref,

            "free_mcp_rmse":
                fm,

            "free_pip_rmse":
                fp,

            "free_combined_rmse":
                fc,

            "contact_mcp_deg":
                cr["mcp_deg"],

            "contact_pip_deg":
                cr["pip_deg"],

            "contact_dip_deg":
                cr["dip_deg"],

            "contact_rmse_deg":
                cr["rmse_deg"],

            "mid_on":
                cr["mid_on"],

            "tip_on":
                cr["tip_on"],

            "dPIP_deg":
                dpip,

            "dDIP_deg":
                ddip,

            "mid_gap_mm":
                cr["target_mid_gap_mm"],

            "tip_gap_mm":
                cr["target_tip_gap_mm"],
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
            f"middle->tip dPIP/dDIP = "
            f"{dpip:+.3f} / "
            f"{ddip:+.3f} deg"
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
    print("=" * 165)
    print("V41 SUMMARY")
    print("=" * 165)

    print(
        "springref   freeRMSE   "
        "contact pose MCP/PIP/DIP       "
        "contactRMSE   M/T    dPIP    dDIP"
    )

    print("-" * 165)


    for r in rows:

        print(
            f"{r['springref_rad']:+8.3f}   "
            f"{r['free_combined_rmse']:8.3f}   "
            f"{r['contact_mcp_deg']:6.2f} / "
            f"{r['contact_pip_deg']:6.2f} / "
            f"{r['contact_dip_deg']:6.2f}       "
            f"{r['contact_rmse_deg']:8.3f}   "
            f"{r['mid_on']}/{r['tip_on']}   "
            f"{r['dPIP_deg']:+7.3f} "
            f"{r['dDIP_deg']:+7.3f}"
        )


    # --------------------------------------------------------
    # Select only physically valid contact candidates.
    # --------------------------------------------------------

    valid = [
        r
        for r in rows
        if (
            r["mid_on"] >= 1
            and
            r["tip_on"] >= 1
            and
            math.isfinite(
                r["dDIP_deg"]
            )
            and
            r["dDIP_deg"] > 0.0
        )
    ]


    if valid:

        best = min(
            valid,
            key=lambda r:
                (
                    r["free_combined_rmse"]
                    +
                    r["contact_rmse_deg"]
                ),
        )


        print()
        print("=" * 165)
        print("BEST VALID V41")
        print("=" * 165)

        print(
            f"DIP springref = "
            f"{best['springref_rad']:+.3f} rad"
        )

        print(
            f"free RMSE = "
            f"{best['free_combined_rmse']:.3f} deg"
        )

        print(
            f"contact RMSE = "
            f"{best['contact_rmse_deg']:.3f} deg"
        )

        print(
            "contact pose = "
            f"{best['contact_mcp_deg']:.3f} / "
            f"{best['contact_pip_deg']:.3f} / "
            f"{best['contact_dip_deg']:.3f}"
        )


    print()
    print(
        f"CSV:\n{OUTCSV}"
    )

    print("=" * 165)


if __name__ == "__main__":
    main()
