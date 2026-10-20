#!/usr/bin/env python3

from pathlib import Path
import math
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

import fit_spatial_free_and_test_contact_v35p7 as p7
import fit_spatial_guide_geometry_v37 as v37


# ============================================================
# PATHS
# ============================================================

WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

SOURCE = (
    MJCF
    / "ur5e_rh8d_index_spatial_v38_final.xml"
)

TEMP = (
    MJCF
    / "tmp_v40_pd_differential.xml"
)

BEST = (
    MJCF
    / "ur5e_rh8d_index_spatial_v40_pd_differential.xml"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v40_pd_differential"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

DIFF_TENDON = (
    "idx_pip_dip_differential_return"
)


# ============================================================
# FIXED V38 FLEXOR
# ============================================================

FLEXOR_K = 2831.47

# Effective radius of differential elastic element.
R_PIP = 0.003       # 3 mm

# Small damping only; this experiment is about elastic
# redistribution, not dynamic damping tuning.
DIFF_DAMPING = 0.15


# ============================================================
# SEARCH VARIABLES
#
# x[0] = differential stiffness [N/m]
#
# x[1] = DIP/PIP effective arm ratio
#
# x[2] = neutral relative angle [deg]
#
# Tendon coordinate:
#
#     L = rP*qP - rD*qD
#
# with
#
#     rD = ratio * rP
#
# If PIP becomes too small and DIP too large,
# L falls below its rest value and the spring produces:
#
#     + torque at PIP
#     - torque at DIP
#
# exactly the direction needed by V38 contact error.
# ============================================================

BOUNDS = [
    (0.0, 3000.0),     # differential K
    (0.35, 1.65),      # DIP/PIP arm ratio
    (-35.0, 35.0),     # neutral relative angle deg
]


# ============================================================
# XML
# ============================================================

def build(
    x,
    output,
):

    k_rel = float(
        x[0]
    )

    ratio = float(
        x[1]
    )

    qrel_ref_deg = float(
        x[2]
    )

    r_dip = (
        R_PIP
        *
        ratio
    )

    # Fixed-tendon coordinate:
    #
    # L = R_PIP*qPIP - r_dip*qDIP
    #
    # Define rest length from equivalent relative angle.
    rest_length = (
        R_PIP
        *
        math.radians(
            qrel_ref_deg
        )
    )


    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()


    tendon_root = root.find(
        "tendon"
    )

    if tendon_root is None:

        tendon_root = ET.SubElement(
            root,
            "tendon",
        )


    # Safe if script is run repeatedly.
    for element in list(
        tendon_root
    ):

        if (
            element.get(
                "name"
            )
            ==
            DIFF_TENDON
        ):

            tendon_root.remove(
                element
            )


    fixed = ET.SubElement(
        tendon_root,
        "fixed",
        {
            "name":
                DIFF_TENDON,

            "stiffness":
                f"{k_rel:.12g}",

            "damping":
                f"{DIFF_DAMPING:.12g}",

            "springlength":
                f"{rest_length:.12g}",
        },
    )


    ET.SubElement(
        fixed,
        "joint",
        {
            "joint":
                PIP,

            "coef":
                f"{R_PIP:.12g}",
        },
    )


    ET.SubElement(
        fixed,
        "joint",
        {
            "joint":
                DIP,

            # IMPORTANT:
            # opposite sign creates differential coupling.
            "coef":
                f"{-r_dip:.12g}",
        },
    )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


    # Reject malformed models immediately.
    mujoco.MjModel.from_xml_path(
        str(output)
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
# RUN ONE MODEL
# ============================================================

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


def evaluate_model(
    x,
):

    build(
        x,
        TEMP,
    )


    fm, fp, fc, _ = (
        p7.run_free(
            TEMP
        )
    )


    cr = run_contact(
        TEMP,
        "V40_EVAL",
    )


    contact_rmse = float(
        cr[
            "rmse_deg"
        ]
    )


    # ========================================================
    # OBJECTIVE
    # ========================================================

    score = (
        fc
        +
        contact_rmse
    )


    # Free-space gate.
    if fc > 6.0:

        score += (
            10.0
            *
            (
                fc
                -
                6.0
            )**2
        )


    # Contact gate.
    if contact_rmse > 2.5:

        score += (
            8.0
            *
            (
                contact_rmse
                -
                2.5
            )**2
        )


    # Both contacts mandatory.
    if cr["mid_on"] < 1:

        score += 40.0


    if cr["tip_on"] < 1:

        score += 40.0


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

        score += 30.0


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

        score += 40.0


    elif ddip <= 0.0:

        score += (
            25.0
            +
            10.0
            *
            abs(
                ddip
            )
        )


    # Very mild regularization:
    # prefer smaller extra elastic element when two candidates
    # perform essentially the same.
    score += (
        0.01
        *
        float(
            x[0]
        )
        /
        1000.0
    )


    return (
        float(score),
        float(fm),
        float(fp),
        float(fc),
        cr,
    )


# ============================================================
# SEARCH
# ============================================================

COUNT = 0

BEST_SEEN = {
    "score":
        float("inf"),

    "x":
        None,

    "free":
        None,

    "contact":
        None,
}


def objective(
    x,
):

    global COUNT

    COUNT += 1


    try:

        score, fm, fp, fc, cr = (
            evaluate_model(
                x
            )
        )


    except Exception as exc:

        print(
            f"[{COUNT:03d}] FAILED: {exc}"
        )

        return 1e6


    if score < BEST_SEEN[
        "score"
    ]:

        BEST_SEEN[
            "score"
        ] = (
            score
        )

        BEST_SEEN[
            "x"
        ] = np.asarray(
            x,
            dtype=float,
        ).copy()

        BEST_SEEN[
            "free"
        ] = (
            fm,
            fp,
            fc,
        )

        BEST_SEEN[
            "contact"
        ] = dict(
            cr
        )


        print()
        print(
            f"NEW V40 BEST [eval {COUNT}]"
        )

        print(
            f"  K differential = "
            f"{x[0]:.2f} N/m"
        )

        print(
            f"  DIP/PIP arm ratio = "
            f"{x[1]:.4f}"
        )

        print(
            f"  relative rest angle = "
            f"{x[2]:.3f} deg"
        )

        print(
            "  FREE MCP/PIP/combined = "
            f"{fm:.3f} / "
            f"{fp:.3f} / "
            f"{fc:.3f} deg"
        )

        print(
            "  CONTACT pose = "
            f"{cr['mcp_deg']:.3f} / "
            f"{cr['pip_deg']:.3f} / "
            f"{cr['dip_deg']:.3f} deg"
        )

        print(
            f"  CONTACT RMSE = "
            f"{cr['rmse_deg']:.3f} deg"
        )

        print(
            f"  M/T events = "
            f"{cr['mid_on']} / "
            f"{cr['tip_on']}"
        )

        print(
            "  dPIP/dDIP = "
            f"{cr['middle_to_tip_dPIP_deg']:.3f} / "
            f"{cr['middle_to_tip_dDIP_deg']:.3f} deg"
        )

        print(
            f"  SCORE = "
            f"{score:.3f}"
        )


    elif (
        COUNT
        %
        10
        ==
        0
    ):

        print(
            f"[{COUNT:03d}] "
            f"score={score:.3f} "
            f"best={BEST_SEEN['score']:.3f}"
        )


    return score


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 160)

    print(
        "V40 — PIP/DIP DIFFERENTIAL ELASTIC COUPLING"
    )

    print("=" * 160)


    print()
    print(
        "The V38 contact error is approximately:"
    )

    print(
        "  MCP  +0.43 deg"
    )

    print(
        "  PIP  -3.16 deg"
    )

    print(
        "  DIP  +3.02 deg"
    )

    print()
    print(
        "Therefore this experiment adds ONLY an "
        "opposite-sign PIP/DIP elastic coupling."
    )


    # --------------------------------------------------------
    # Reproduce V38
    # --------------------------------------------------------

    print()
    print("=" * 160)
    print(
        "V38 BASELINE CHECK"
    )
    print("=" * 160)


    fm0, fp0, fc0, _ = (
        p7.run_free(
            SOURCE
        )
    )


    cr0 = run_contact(
        SOURCE,
        "V38_BASELINE",
    )


    print(
        f"FREE MCP/PIP/combined = "
        f"{fm0:.3f} / "
        f"{fp0:.3f} / "
        f"{fc0:.3f}"
    )

    print(
        "CONTACT pose = "
        f"{cr0['mcp_deg']:.3f} / "
        f"{cr0['pip_deg']:.3f} / "
        f"{cr0['dip_deg']:.3f}"
    )

    print(
        f"CONTACT RMSE = "
        f"{cr0['rmse_deg']:.3f}"
    )


    # --------------------------------------------------------
    # Include exact baseline in best set.
    #
    # K=0 means differential tendon has no force, regardless
    # of other two coordinates.
    # --------------------------------------------------------

    baseline_x = np.array(
        [
            0.0,
            1.0,
            0.0,
        ],
        dtype=float,
    )


    bscore, bfm, bfp, bfc, bcr = (
        evaluate_model(
            baseline_x
        )
    )


    BEST_SEEN[
        "score"
    ] = (
        bscore
    )

    BEST_SEEN[
        "x"
    ] = (
        baseline_x.copy()
    )

    BEST_SEEN[
        "free"
    ] = (
        bfm,
        bfp,
        bfc,
    )

    BEST_SEEN[
        "contact"
    ] = dict(
        bcr
    )


    print()
    print(
        f"Baseline objective score = "
        f"{bscore:.3f}"
    )


    # --------------------------------------------------------
    # Search only 3 variables.
    # --------------------------------------------------------

    print()
    print("=" * 160)
    print(
        "AUTOMATED DIFFERENTIAL-COUPLING SEARCH"
    )
    print("=" * 160)


    differential_evolution(
        objective,
        BOUNDS,

        seed=40,

        popsize=5,

        maxiter=5,

        tol=0.01,

        mutation=(
            0.5,
            1.0,
        ),

        recombination=0.7,

        polish=False,

        workers=1,

        updating="immediate",
    )


    x = (
        BEST_SEEN[
            "x"
        ]
    )


    build(
        x,
        BEST,
    )


    shutil.copy2(
        BEST,
        OUTDIR
        /
        BEST.name,
    )


    # --------------------------------------------------------
    # Final repeat
    # --------------------------------------------------------

    fm, fp, fc, _ = (
        p7.run_free(
            BEST,
            OUTDIR
            /
            "v40_final_free.log",
        )
    )


    cr = run_contact(
        BEST,
        "V40_FINAL",
    )


    contact_rmse = float(
        cr[
            "rmse_deg"
        ]
    )


    print()
    print("=" * 160)
    print(
        "V40 FINAL RESULT"
    )
    print("=" * 160)


    print()
    print("DIFFERENTIAL ELEMENT")

    print(
        f"  K = "
        f"{x[0]:.3f} N/m"
    )

    print(
        f"  DIP/PIP arm ratio = "
        f"{x[1]:.4f}"
    )

    print(
        f"  relative rest angle = "
        f"{x[2]:.4f} deg"
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
        f"{fc:.3f} deg"
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
        f"{contact_rmse:.3f} deg"
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


    passed = (
        fc <= 6.0
        and
        contact_rmse <= 2.5
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
            "V40 STATUS: PASS"
        )

        print()
        print(
            "FREEZE THE MODEL."
        )

        print(
            "Do not fit this experiment further."
        )

        print(
            "Next step = held-out repetitions."
        )


    else:

        print(
            "V40 STATUS: FAIL"
        )

        print()

        if (
            x[0]
            <
            50.0
        ):

            print(
                "Optimizer rejected the differential "
                "elastic mechanism itself."
            )

        else:

            print(
                "Differential coupling helps but is "
                "not sufficient."
            )

        print(
            "Do NOT return to passive stiffness/"
            "contact-friction scans."
        )

        print(
            "Next structural candidate will be "
            "load-dependent guide/tension redistribution."
        )


    print()
    print(
        f"BEST XML:\n{BEST}"
    )

    print(
        f"\nRESULTS:\n{OUTDIR}"
    )

    print("=" * 160)


if __name__ == "__main__":
    main()
