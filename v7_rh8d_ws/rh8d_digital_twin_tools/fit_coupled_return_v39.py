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


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = WS / "src/ur5e_mujoco_description/mjcf"

SOURCE = (
    MJCF
    / "ur5e_rh8d_index_spatial_v38_final.xml"
)

TEMP = (
    MJCF
    / "tmp_v39_coupled_return.xml"
)

BEST = (
    MJCF
    / "ur5e_rh8d_index_spatial_v39_coupled_return.xml"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v39_coupled_return"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

RET_PROX = "idx_return_prox_coupled"
RET_DIST = "idx_return_dist_coupled"


# ============================================================
# READ V38 BASELINE PASSIVES
# ============================================================

def joint_attr(
    root,
    name,
    attr,
):

    j = root.find(
        f".//joint[@name='{name}']"
    )

    if j is None:
        raise RuntimeError(
            f"Missing joint: {name}"
        )

    value = j.get(
        attr
    )

    if value is None:
        return 0.0

    return float(
        value.split()[0]
    )


ROOT0 = ET.parse(
    SOURCE
).getroot()

K0 = {
    MCP:
        joint_attr(
            ROOT0,
            MCP,
            "stiffness",
        ),

    PIP:
        joint_attr(
            ROOT0,
            PIP,
            "stiffness",
        ),

    DIP:
        joint_attr(
            ROOT0,
            DIP,
            "stiffness",
        ),
}


# ============================================================
# VARIABLES
#
# x[0] = residual independent-joint spring scale
#
# x[1] = proximal return stiffness N/m
# x[2] = distal return stiffness N/m
#
# x[3] = proximal preload mm
# x[4] = distal preload mm
#
# x[5] = proximal PIP/MCP moment-arm ratio
# x[6] = distal DIP/PIP moment-arm ratio
#
# Return tendon reference moment arm = 3 mm.
# Positive coefficients generate extension-restoring torque
# as flexion increases tendon length.
# ============================================================

BOUNDS = [
    (0.00, 1.10),       # residual joint spring scale

    (0.0, 4500.0),      # proximal return stiffness
    (0.0, 4500.0),      # distal return stiffness

    (0.0, 1.5),         # proximal preload mm
    (0.0, 1.5),         # distal preload mm

    (0.40, 1.80),       # prox PIP / MCP arm ratio
    (0.30, 1.80),       # distal DIP / PIP arm ratio
]


RBASE = 0.003   # 3 mm effective return-tendon moment arm
RETURN_DAMPING = 0.20


# ============================================================
# XML BUILDER
# ============================================================

def set_joint_stiffness(
    root,
    name,
    value,
):

    j = root.find(
        f".//joint[@name='{name}']"
    )

    if j is None:
        raise RuntimeError(
            f"Missing joint {name}"
        )

    j.set(
        "stiffness",
        f"{float(value):.12g}",
    )


def add_fixed_return(
    tendon_root,
    name,
    stiffness,
    preload_m,
    terms,
):

    fixed = ET.SubElement(
        tendon_root,
        "fixed",
        {
            "name":
                name,

            "stiffness":
                f"{float(stiffness):.12g}",

            "damping":
                f"{RETURN_DAMPING:.12g}",

            # Negative rest coordinate means a small preload
            # already exists at q = 0.
            "springlength":
                f"{-float(preload_m):.12g}",
        },
    )


    for joint_name, coef in terms:

        ET.SubElement(
            fixed,
            "joint",
            {
                "joint":
                    joint_name,

                "coef":
                    f"{float(coef):.12g}",
            },
        )


def build(
    x,
    output,
):

    x = np.asarray(
        x,
        dtype=float,
    )


    (
        residual_scale,
        k_prox,
        k_dist,
        preload_prox_mm,
        preload_dist_mm,
        ratio_prox,
        ratio_dist,
    ) = x


    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()


    # --------------------------------------------------------
    # Residual independent joint elasticity.
    #
    # If residual_scale -> 0:
    # elasticity is almost entirely carried by the coupled
    # return elements.
    #
    # If residual_scale -> 1:
    # V38 torsional springs remain almost unchanged.
    # --------------------------------------------------------

    for name in [
        MCP,
        PIP,
        DIP,
    ]:

        set_joint_stiffness(
            root,
            name,
            K0[name]
            *
            residual_scale,
        )


    tendon_root = root.find(
        "tendon"
    )

    if tendon_root is None:

        tendon_root = ET.SubElement(
            root,
            "tendon",
        )


    # Safe re-run behavior.
    for element in list(
        tendon_root
    ):

        if element.get(
            "name"
        ) in [
            RET_PROX,
            RET_DIST,
        ]:

            tendon_root.remove(
                element
            )


    # --------------------------------------------------------
    # PROXIMAL COUPLED RETURN
    #
    # Lp =
    # rM*qM + rP*qP
    #
    # Flexion of either MCP or PIP stretches the same element
    # and changes restoring torque at BOTH joints.
    # --------------------------------------------------------

    add_fixed_return(
        tendon_root,
        RET_PROX,
        k_prox,
        preload_prox_mm
        *
        1e-3,
        [
            (
                MCP,
                RBASE,
            ),
            (
                PIP,
                RBASE
                *
                ratio_prox,
            ),
        ],
    )


    # --------------------------------------------------------
    # DISTAL COUPLED RETURN
    #
    # Ld =
    # rP*qP + rD*qD
    #
    # PIP/DIP therefore share one elastic state.
    # --------------------------------------------------------

    add_fixed_return(
        tendon_root,
        RET_DIST,
        k_dist,
        preload_dist_mm
        *
        1e-3,
        [
            (
                PIP,
                RBASE,
            ),
            (
                DIP,
                RBASE
                *
                ratio_dist,
            ),
        ],
    )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


    # Must compile before it is ever evaluated.
    mujoco.MjModel.from_xml_path(
        str(output)
    )


# ============================================================
# CONTACT INITIALIZATION
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
# OBJECTIVE
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


def calculate_score(
    free_rmse,
    contact_rmse,
    cr,
    x,
):

    score = (
        free_rmse
        +
        contact_rmse
    )


    # Strongly discourage moving away from the desired region.

    if free_rmse > 6.0:

        score += (
            8.0
            *
            (
                free_rmse
                -
                6.0
            )**2
        )


    if contact_rmse > 2.5:

        score += (
            6.0
            *
            (
                contact_rmse
                -
                2.5
            )**2
        )


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
            3.0
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


    # Mild complexity regularization.
    #
    # Do not allow the optimizer to create absurdly stiff returns
    # merely to shave a tiny fraction of a degree.
    score += (
        0.02
        *
        (
            x[1]
            +
            x[2]
        )
        /
        1000.0
    )


    return float(
        score
    )


def evaluate(
    x,
):

    build(
        x,
        TEMP,
    )


    # ---------------- FREE SPACE ----------------

    fm, fp, fc, _ = p7.run_free(
        TEMP
    )


    # ---------------- CONTACT ----------------

    old_xml = (
        v37.p3.XML
    )


    try:

        v37.p3.XML = (
            TEMP
        )

        cr = v37.p3.run_case(
            "V39",
            # Flexor tendon stiffness remains whatever is
            # already in frozen V38.
            2831.47,
            True,
            CONTACT_REPLAY,
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


    score = calculate_score(
        fc,
        contact_rmse,
        cr,
        x,
    )


    return (
        score,
        fm,
        fp,
        fc,
        cr,
    )


def objective(
    x,
):

    global COUNT

    COUNT += 1


    try:

        (
            score,
            fm,
            fp,
            fc,
            cr,
        ) = evaluate(
            x
        )


    except Exception as exc:

        print(
            f"[{COUNT:03d}] FAILED: {exc}"
        )

        return 1000.0


    if score < BEST_SEEN[
        "score"
    ]:

        BEST_SEEN[
            "score"
        ] = score

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
            f"NEW V39 BEST [eval {COUNT}]"
        )

        print(
            f"  residual joint spring scale = "
            f"{x[0]:.3f}"
        )

        print(
            "  return K prox/dist = "
            f"{x[1]:.1f} / "
            f"{x[2]:.1f} N/m"
        )

        print(
            "  preload prox/dist = "
            f"{x[3]:.3f} / "
            f"{x[4]:.3f} mm"
        )

        print(
            "  return arm ratios prox/dist = "
            f"{x[5]:.3f} / "
            f"{x[6]:.3f}"
        )

        print(
            "  FREE MCP/PIP/combined = "
            f"{fm:.3f} / "
            f"{fp:.3f} / "
            f"{fc:.3f} deg"
        )

        print(
            "  CONTACT pose = "
            f"{cr['mcp_deg']:.2f} / "
            f"{cr['pip_deg']:.2f} / "
            f"{cr['dip_deg']:.2f} deg"
        )

        print(
            f"  CONTACT RMSE = "
            f"{cr['rmse_deg']:.3f} deg"
        )

        print(
            "  M/T events = "
            f"{cr['mid_on']} / "
            f"{cr['tip_on']}"
        )

        print(
            "  middle->tip dPIP/dDIP = "
            f"{cr['middle_to_tip_dPIP_deg']:.3f} / "
            f"{cr['middle_to_tip_dDIP_deg']:.3f}"
        )

        print(
            f"  SCORE = "
            f"{score:.3f}"
        )


    elif COUNT % 10 == 0:

        print(
            f"[{COUNT:03d}] "
            f"score={score:.3f} "
            f"best={BEST_SEEN['score']:.3f}"
        )


    return score


# ============================================================
# BASELINE SANITY
# ============================================================

def run_v38_baseline():

    fm, fp, fc, _ = p7.run_free(
        SOURCE
    )


    old_xml = (
        v37.p3.XML
    )


    try:

        v37.p3.XML = (
            SOURCE
        )

        cr = v37.p3.run_case(
            "V38_BASELINE",
            2831.47,
            True,
            CONTACT_REPLAY,
        )

    finally:

        v37.p3.XML = (
            old_xml
        )


    return (
        fm,
        fp,
        fc,
        cr,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 160)

    print(
        "V39 — COUPLED ELASTIC RETURN MODEL"
    )

    print("=" * 160)

    print()
    print(
        "Frozen:"
    )

    print(
        "  V38 flexor spatial geometry"
    )

    print(
        "  flexor K = 2831.47 N/m"
    )

    print(
        "  encoder scale/boundary"
    )

    print(
        "  object/contact model"
    )

    print(
        "  joint damping + dry friction"
    )

    print()
    print(
        "New physics:"
    )

    print(
        "  shared MCP-PIP elastic return"
    )

    print(
        "  shared PIP-DIP elastic return"
    )


    print()
    print("=" * 160)
    print(
        "V38 REPRODUCTION CHECK"
    )
    print("=" * 160)


    fm0, fp0, fc0, cr0 = (
        run_v38_baseline()
    )


    print(
        f"free = "
        f"{fm0:.3f} / "
        f"{fp0:.3f} / "
        f"{fc0:.3f} deg"
    )

    print(
        f"contact = "
        f"{cr0['rmse_deg']:.3f} deg"
    )

    print(
        "contact pose = "
        f"{cr0['mcp_deg']:.3f} / "
        f"{cr0['pip_deg']:.3f} / "
        f"{cr0['dip_deg']:.3f}"
    )


    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    print()
    print("=" * 160)
    print(
        "AUTOMATED V39 IDENTIFICATION"
    )
    print("=" * 160)


    result = differential_evolution(
        objective,
        BOUNDS,

        seed=39,

        popsize=3,

        maxiter=4,

        tol=0.02,

        mutation=(
            0.5,
            1.0,
        ),

        recombination=0.7,

        polish=False,

        workers=1,

        updating="immediate",
    )


    if BEST_SEEN[
        "x"
    ] is None:

        raise RuntimeError(
            "No valid V39 candidate."
        )


    x = BEST_SEEN[
        "x"
    ]


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
    # FINAL REPEAT
    # --------------------------------------------------------

    fm, fp, fc, _ = p7.run_free(
        BEST,
        OUTDIR
        /
        "v39_final_free.log",
    )


    old_xml = (
        v37.p3.XML
    )


    try:

        v37.p3.XML = (
            BEST
        )

        cr = v37.p3.run_case(
            "V39_FINAL",
            2831.47,
            True,
            CONTACT_REPLAY,
        )

    finally:

        v37.p3.XML = (
            old_xml
        )


    print()
    print("=" * 160)
    print(
        "V39 FINAL RESULT"
    )
    print("=" * 160)

    print()
    print(
        f"residual independent spring scale = "
        f"{x[0]:.4f}"
    )

    print(
        "return stiffness prox/dist = "
        f"{x[1]:.2f} / "
        f"{x[2]:.2f} N/m"
    )

    print(
        "return preload prox/dist = "
        f"{x[3]:.4f} / "
        f"{x[4]:.4f} mm"
    )

    print(
        "return arm ratios prox/dist = "
        f"{x[5]:.4f} / "
        f"{x[6]:.4f}"
    )


    print()
    print(
        "FREE SPACE"
    )

    print(
        f"  MCP = "
        f"{fm:.3f} deg"
    )

    print(
        f"  PIP = "
        f"{fp:.3f} deg"
    )

    print(
        f"  combined = "
        f"{fc:.3f} deg"
    )


    print()
    print(
        "CONTACT"
    )

    print(
        f"  pose = "
        f"{cr['mcp_deg']:.3f} / "
        f"{cr['pip_deg']:.3f} / "
        f"{cr['dip_deg']:.3f} deg"
    )

    print(
        f"  RMSE = "
        f"{cr['rmse_deg']:.3f} deg"
    )

    print(
        f"  M/T events = "
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
        f"  gaps M/T = "
        f"{cr['target_mid_gap_mm']:.3f} / "
        f"{cr['target_tip_gap_mm']:.3f} mm"
    )


    passed = (
        fc <= 6.0
        and
        cr["rmse_deg"] <= 2.5
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
        ] > 0
    )


    print()
    print("=" * 160)

    if passed:

        print(
            "V39 STATUS: PASS"
        )

        print()
        print(
            "FREEZE THE MODEL."
        )

        print(
            "Next step: held-out hardware validation, "
            "not more fitting."
        )

    else:

        print(
            "V39 STATUS: FAIL"
        )

        print()
        print(
            "The two coupled linear return elements "
            "are still insufficient."
        )

        print(
            "Next architecture would require spatial/nonlinear "
            "elastic return geometry rather than another "
            "joint-parameter scan."
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
