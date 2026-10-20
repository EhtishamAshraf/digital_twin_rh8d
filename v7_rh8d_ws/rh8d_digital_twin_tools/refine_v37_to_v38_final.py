#!/usr/bin/env python3

from pathlib import Path
import csv
import math
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import fit_spatial_guide_geometry_v37 as v37
import fit_spatial_free_and_test_contact_v35p7 as p7


# ============================================================
# PATHS
# ============================================================

WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = WS / "src/ur5e_mujoco_description/mjcf"

V37_XML = (
    MJCF
    / "ur5e_rh8d_index_spatial_v37_routefit.xml"
)

TEMP_XML = (
    MJCF
    / "tmp_v38_local_refine.xml"
)

BEST_XML = (
    MJCF
    / "ur5e_rh8d_index_spatial_v38_final.xml"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v38_final_local_refine"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_PATH = (
    OUTDIR
    / "v38_all_evaluations.csv"
)


# ============================================================
# NAMES
# ============================================================

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"


# ============================================================
# V37 ROUTE — START POINT
#
# These are the actual winning V37 route parameters.
# ============================================================

V37_SM = 0.9980
V37_SP = 1.2516
V37_SD = 0.7022
V37_KT = 2731.47


def xml_joint_value(
    xml,
    name,
    attribute,
):

    root = ET.parse(xml).getroot()

    j = root.find(
        f".//joint[@name='{name}']"
    )

    if j is None:
        raise RuntimeError(
            f"Missing joint {name}"
        )

    value = j.get(
        attribute
    )

    if value is None:
        return 0.0

    return float(
        value.split()[0]
    )


# Read the exact V37 passive starting point.
K_MCP_0 = xml_joint_value(
    V37_XML,
    MCP,
    "stiffness",
)

K_PIP_0 = xml_joint_value(
    V37_XML,
    PIP,
    "stiffness",
)

F_PIP_0 = xml_joint_value(
    V37_XML,
    PIP,
    "frictionloss",
)

K_DIP_0 = xml_joint_value(
    V37_XML,
    DIP,
    "stiffness",
)

F_DIP_0 = xml_joint_value(
    V37_XML,
    DIP,
    "frictionloss",
)


# ============================================================
# PARAMETER VECTOR
#
# x =
# [
#   MCP stiffness,
#   PIP stiffness,
#   PIP friction,
#   DIP stiffness,
#   DIP friction,
#   MCP guide scale,
#   PIP guide scale,
#   DIP guide scale,
#   tendon stiffness
# ]
#
# Search is deliberately LOCAL around the successful V37 route.
# ============================================================

X0 = np.array([
    K_MCP_0,
    K_PIP_0,
    F_PIP_0,
    K_DIP_0,
    F_DIP_0,
    V37_SM,
    V37_SP,
    V37_SD,
    V37_KT,
], dtype=float)


LOW = np.array([
    0.0040,     # MCP k
    0.0060,     # PIP k
    0.0000,     # PIP friction
    0.0150,     # DIP k
    0.0000,     # DIP friction
    0.88,       # MCP guide scale
    1.10,       # PIP guide scale
    0.55,       # DIP guide scale
    2000.0,     # tendon K
], dtype=float)


HIGH = np.array([
    0.0200,
    0.0350,
    0.0080,
    0.0700,
    0.0030,
    1.08,
    1.40,
    0.85,
    3800.0,
], dtype=float)


# Initial coordinate-search step.
STEPS = np.array([
    0.0030,
    0.0050,
    0.0012,
    0.0080,
    0.0005,
    0.040,
    0.060,
    0.050,
    400.0,
], dtype=float)


# ============================================================
# BUILD MODEL
# ============================================================

def patch_passives(
    xml,
    x,
):

    tree = ET.parse(
        xml
    )

    root = tree.getroot()


    values = {
        MCP: (
            x[0],
            0.002,
        ),

        PIP: (
            x[1],
            x[2],
        ),

        DIP: (
            x[3],
            x[4],
        ),
    }


    for name, (
        stiffness,
        friction,
    ) in values.items():

        j = root.find(
            f".//joint[@name='{name}']"
        )

        if j is None:
            raise RuntimeError(
                f"Missing joint {name}"
            )

        j.set(
            "stiffness",
            f"{float(stiffness):.12g}",
        )

        j.set(
            "frictionloss",
            f"{float(friction):.12g}",
        )


    tree.write(
        xml,
        encoding="unicode",
        xml_declaration=True,
    )


def build(
    x,
    output,
):

    x = np.asarray(
        x,
        dtype=float,
    )


    # Use V37's already-tested geometric builder.
    route_x = [
        float(x[5]),  # MCP scale
        float(x[6]),  # PIP scale
        float(x[7]),  # DIP scale
        float(x[8]),  # tendon K
        float(x[2]),  # PIP friction
    ]


    old_temp = v37.TEMP

    try:

        v37.TEMP = output

        v37.build_candidate(
            route_x,
            output,
        )

    finally:

        v37.TEMP = old_temp


    patch_passives(
        output,
        x,
    )


    # Compile before simulation.
    mujoco.MjModel.from_xml_path(
        str(output)
    )


# ============================================================
# SCORE
# ============================================================

def score_result(
    free_rmse,
    contact_rmse,
    cr,
):

    # Basic objective.
    score = (
        free_rmse
        +
        contact_rmse
    )


    # --------------------------------------------------------
    # HARD PRIORITY:
    # restore free-space <= 6 degrees.
    # --------------------------------------------------------

    if free_rmse > 6.0:

        score += (
            6.0
            *
            (
                free_rmse
                -
                6.0
            )**2
        )


    # --------------------------------------------------------
    # Contact must remain good.
    # --------------------------------------------------------

    if contact_rmse > 2.5:

        score += (
            4.0
            *
            (
                contact_rmse
                -
                2.5
            )**2
        )


    # --------------------------------------------------------
    # Correct contact topology is mandatory.
    # --------------------------------------------------------

    if cr["mid_on"] < 1:

        score += 30.0


    if cr["tip_on"] < 1:

        score += 30.0


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


    if not math.isfinite(
        dpip
    ):

        score += 20.0

    elif dpip > 4.0:

        score += (
            2.0
            *
            (
                dpip
                -
                4.0
            )**2
        )


    if not math.isfinite(
        ddip
    ):

        score += 30.0

    elif ddip <= 0.0:

        score += (
            15.0
            +
            5.0
            *
            abs(
                ddip
            )
        )


    return float(
        score
    )


# ============================================================
# EVALUATION
# ============================================================

CACHE = {}

HISTORY = []

COUNT = 0


def cache_key(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    return tuple(
        np.round(
            x,
            decimals=9,
        )
    )


def evaluate(
    x,
    save_best=False,
):

    global COUNT

    x = np.clip(
        np.asarray(
            x,
            dtype=float,
        ),
        LOW,
        HIGH,
    )


    key = cache_key(
        x
    )


    if key in CACHE:

        return CACHE[
            key
        ]


    COUNT += 1


    try:

        build(
            x,
            TEMP_XML,
        )


        # ---------------- FREE SPACE ----------------

        fm, fp, free_rmse, _ = p7.run_free(
            TEMP_XML
        )


        # ---------------- CONTACT ----------------

        old_xml = v37.p3.XML

        try:

            v37.p3.XML = (
                TEMP_XML
            )

            cr = v37.p3.run_case(
                f"V38_EVAL_{COUNT}",
                float(
                    x[8]
                ),
                True,
                v37.CONTACT_REPLAY,
            )

        finally:

            v37.p3.XML = (
                old_xml
            )


        contact_rmse = float(
            cr[
                "rmse_deg"
            ]
        )


        score = score_result(
            free_rmse,
            contact_rmse,
            cr,
        )


    except Exception as exc:

        print(
            f"[{COUNT:03d}] FAILED: {exc}"
        )

        result = {
            "score":
                1e6,

            "x":
                x.copy(),

            "free_mcp":
                math.nan,

            "free_pip":
                math.nan,

            "free":
                math.nan,

            "contact":
                math.nan,

            "cr":
                None,
        }

        CACHE[
            key
        ] = result

        return result


    result = {
        "score":
            score,

        "x":
            x.copy(),

        "free_mcp":
            float(
                fm
            ),

        "free_pip":
            float(
                fp
            ),

        "free":
            float(
                free_rmse
            ),

        "contact":
            float(
                contact_rmse
            ),

        "cr":
            dict(
                cr
            ),
    }


    CACHE[
        key
    ] = result


    HISTORY.append({
        "eval":
            COUNT,

        "score":
            score,

        "free_mcp":
            fm,

        "free_pip":
            fp,

        "free_combined":
            free_rmse,

        "contact_rmse":
            contact_rmse,

        "contact_mcp":
            cr["mcp_deg"],

        "contact_pip":
            cr["pip_deg"],

        "contact_dip":
            cr["dip_deg"],

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

        "k_mcp":
            x[0],

        "k_pip":
            x[1],

        "f_pip":
            x[2],

        "k_dip":
            x[3],

        "f_dip":
            x[4],

        "scale_mcp":
            x[5],

        "scale_pip":
            x[6],

        "scale_dip":
            x[7],

        "tendon_k":
            x[8],
    })


    print(
        f"[{COUNT:03d}] "
        f"score={score:7.3f}  "
        f"free={free_rmse:6.3f}  "
        f"contact={contact_rmse:6.3f}  "
        f"M/T={cr['mid_on']}/{cr['tip_on']}  "
        f"dPIP={cr['middle_to_tip_dPIP_deg']:+6.3f}  "
        f"dDIP={cr['middle_to_tip_dDIP_deg']:+6.3f}"
    )


    return result


# ============================================================
# SAVE
# ============================================================

def save_history():

    if not HISTORY:

        return


    with CSV_PATH.open(
        "w",
        newline="",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                HISTORY[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            HISTORY
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 160)
    print(
        "V38 — FINAL LOCAL JOINT IDENTIFICATION AROUND "
        "SUCCESSFUL V37 SPATIAL ROUTE"
    )
    print("=" * 160)

    print()
    print(
        "Goal:"
    )

    print(
        "  free <= 6.0 deg"
    )

    print(
        "  contact <= 2.5 deg"
    )

    print(
        "  middle + tip contact"
    )

    print(
        "  distal flexion after middle contact"
    )


    # --------------------------------------------------------
    # Baseline V37
    # --------------------------------------------------------

    current = X0.copy()

    best = evaluate(
        current
    )


    print()
    print("=" * 160)
    print("STARTING V37")
    print("=" * 160)

    print(
        f"score   = "
        f"{best['score']:.3f}"
    )

    print(
        f"free    = "
        f"{best['free']:.3f} deg"
    )

    print(
        f"contact = "
        f"{best['contact']:.3f} deg"
    )


    # --------------------------------------------------------
    # Deterministic coordinate refinement.
    #
    # Four rounds:
    # each round tests +/- each mechanical variable.
    # Step size is halved after every complete pass.
    # --------------------------------------------------------

    steps = (
        STEPS.copy()
    )


    names = [
        "MCP stiffness",
        "PIP stiffness",
        "PIP friction",
        "DIP stiffness",
        "DIP friction",
        "MCP guide scale",
        "PIP guide scale",
        "DIP guide scale",
        "tendon stiffness",
    ]


    for round_index in range(
        4
    ):

        print()
        print("=" * 160)

        print(
            f"ROUND {round_index + 1}/4"
        )

        print("=" * 160)


        improved_round = False


        for i, name in enumerate(
            names
        ):

            candidates = []


            for direction in [
                -1.0,
                +1.0,
            ]:

                trial = (
                    current.copy()
                )

                trial[i] += (
                    direction
                    *
                    steps[i]
                )

                trial = np.clip(
                    trial,
                    LOW,
                    HIGH,
                )


                r = evaluate(
                    trial
                )

                candidates.append(
                    r
                )


            local_best = min(
                candidates,
                key=lambda r:
                    r["score"],
            )


            if (
                local_best["score"]
                +
                1e-9
                <
                best["score"]
            ):

                best = (
                    local_best
                )

                current = (
                    best[
                        "x"
                    ].copy()
                )

                improved_round = True


                print()
                print(
                    "  ACCEPT "
                    f"{name}"
                )

                print(
                    f"    new best score = "
                    f"{best['score']:.3f}"
                )

                print(
                    f"    free/contact = "
                    f"{best['free']:.3f} / "
                    f"{best['contact']:.3f} deg"
                )


        steps *= 0.5


        print()
        print(
            "Round result:"
        )

        print(
            f"  free/contact = "
            f"{best['free']:.3f} / "
            f"{best['contact']:.3f} deg"
        )

        print(
            f"  score = "
            f"{best['score']:.3f}"
        )


    # --------------------------------------------------------
    # Build permanent best model.
    # --------------------------------------------------------

    build(
        best[
            "x"
        ],
        BEST_XML,
    )


    shutil.copy2(
        BEST_XML,
        OUTDIR
        /
        BEST_XML.name,
    )


    save_history()


    # --------------------------------------------------------
    # FINAL independent repeat of same evaluators.
    # --------------------------------------------------------

    fm, fp, free_final, _ = p7.run_free(
        BEST_XML,
        OUTDIR
        /
        "v38_final_free.log",
    )


    old_xml = (
        v37.p3.XML
    )

    try:

        v37.p3.XML = (
            BEST_XML
        )

        cr = v37.p3.run_case(
            "V38_FINAL",
            float(
                best[
                    "x"
                ][8]
            ),
            True,
            v37.CONTACT_REPLAY,
        )

    finally:

        v37.p3.XML = (
            old_xml
        )


    contact_final = float(
        cr[
            "rmse_deg"
        ]
    )


    x = best[
        "x"
    ]


    print()
    print("=" * 160)
    print("V38 FINAL RESULT")
    print("=" * 160)

    print()
    print("PARAMETERS")

    print(
        f"  guide scales M/P/D = "
        f"{x[5]:.4f} / "
        f"{x[6]:.4f} / "
        f"{x[7]:.4f}"
    )

    print(
        f"  tendon K = "
        f"{x[8]:.2f} N/m"
    )

    print(
        f"  MCP stiffness = "
        f"{x[0]:.6f}"
    )

    print(
        f"  PIP stiffness = "
        f"{x[1]:.6f}"
    )

    print(
        f"  PIP friction = "
        f"{x[2]:.6f}"
    )

    print(
        f"  DIP stiffness = "
        f"{x[3]:.6f}"
    )

    print(
        f"  DIP friction = "
        f"{x[4]:.6f}"
    )


    print()
    print("FREE SPACE")

    print(
        f"  MCP RMSE = "
        f"{fm:.3f} deg"
    )

    print(
        f"  PIP RMSE = "
        f"{fp:.3f} deg"
    )

    print(
        f"  combined = "
        f"{free_final:.3f} deg"
    )


    print()
    print("CONTACT")

    print(
        f"  pose = "
        f"{cr['mcp_deg']:.3f} / "
        f"{cr['pip_deg']:.3f} / "
        f"{cr['dip_deg']:.3f} deg"
    )

    print(
        f"  RMSE = "
        f"{contact_final:.3f} deg"
    )

    print(
        f"  middle/tip events = "
        f"{cr['mid_on']} / "
        f"{cr['tip_on']}"
    )

    print(
        f"  dPIP = "
        f"{cr['middle_to_tip_dPIP_deg']:.3f} deg"
    )

    print(
        f"  dDIP = "
        f"{cr['middle_to_tip_dDIP_deg']:.3f} deg"
    )

    print(
        f"  target gaps M/T = "
        f"{cr['target_mid_gap_mm']:.3f} / "
        f"{cr['target_tip_gap_mm']:.3f} mm"
    )


    # --------------------------------------------------------
    # STRICT FINAL DECISION
    # --------------------------------------------------------

    passed = (
        free_final <= 6.0
        and
        contact_final <= 2.5
        and
        cr["mid_on"] >= 1
        and
        cr["tip_on"] >= 1
        and
        math.isfinite(
            cr[
                "middle_to_tip_dDIP_deg"
            ]
        )
        and
        cr[
            "middle_to_tip_dDIP_deg"
        ] > 0.0
    )


    print()
    print("=" * 160)

    if passed:

        print(
            "V38 STATUS: PASS"
        )

        print()
        print(
            "KEEP THIS ARCHITECTURE."
        )

        print(
            "Next step is held-out hardware validation, "
            "not more fitting."
        )

    else:

        print(
            "V38 STATUS: FAIL"
        )

        print()
        print(
            "STOP PARAMETER TUNING OF THIS SINGLE-FLEXOR MODEL."
        )

        print(
            "Next model must add the physical coupled "
            "elastic return mechanism."
        )


    print()
    print(
        f"BEST XML:\n{BEST_XML}"
    )

    print(
        f"\nALL RESULTS:\n{OUTDIR}"
    )

    print("=" * 160)


if __name__ == "__main__":
    main()
