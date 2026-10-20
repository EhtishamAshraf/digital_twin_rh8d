#!/usr/bin/env python3

from pathlib import Path
import math
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

import fit_spatial_free_and_test_contact_v35p7 as p7
import test_prescribed_spool_v35p3 as p3


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = WS / "src/ur5e_mujoco_description/mjcf"

SOURCE = (
    MJCF
    / "ur5e_rh8d_index_cad_spatial_v35p7_freefit.xml"
)

TEMP = MJCF / "tmp_v37_route_fit.xml"

BEST = (
    MJCF
    / "ur5e_rh8d_index_spatial_v37_routefit.xml"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v37_route_identification"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# JOINT / GUIDE DEFINITIONS
# ============================================================

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

GUIDES = {
    MCP: [
        "v35_p02_mcp_parent",
        "v35_p03_mcp_child",
    ],

    PIP: [
        "v35_p04_pip_parent",
        "v35_p05_pip_child",
    ],

    DIP: [
        "v35_p06_dip_parent",
        "v35_p07_dip_child",
    ],
}

TENDON = "idx_flex_spatial"


# ============================================================
# SEARCH VARIABLES
#
# sm, sp, sd:
#   radial guide-distance multiplier relative to present route.
#
# Current route is 1.0.
#
# DIP is deliberately allowed to become much smaller because
# the present contact result badly over-flexes DIP.
# ============================================================

BOUNDS = [
    (0.40, 1.15),       # MCP radial scale
    (0.60, 1.50),       # PIP radial scale
    (0.15, 1.00),       # DIP radial scale
    (1800.0, 6500.0),   # tendon stiffness N/m
    (0.0000, 0.0080),   # PIP frictionloss Nm
]


# ============================================================
# REFERENCE GEOMETRY
# ============================================================

def obj_id(model, objtype, name):

    i = mujoco.mj_name2id(
        model,
        objtype,
        name,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing object: {name}"
        )

    return i


def get_site_element(root, name):

    e = root.find(
        f".//site[@name='{name}']"
    )

    if e is None:
        raise RuntimeError(
            f"Missing XML site: {name}"
        )

    return e


def reference_geometry():

    model = mujoco.MjModel.from_xml_path(
        str(SOURCE)
    )

    data = mujoco.MjData(model)

    mujoco.mj_forward(
        model,
        data,
    )

    ref = {}


    for joint_name, site_names in GUIDES.items():

        jid = obj_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        anchor = np.asarray(
            data.xanchor[jid],
            dtype=float,
        ).copy()

        axis = np.asarray(
            data.xaxis[jid],
            dtype=float,
        ).copy()

        axis /= np.linalg.norm(axis)


        ref[joint_name] = {
            "anchor":
                anchor,

            "axis":
                axis,

            "sites":
                {},
        }


        for site_name in site_names:

            sid = obj_id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                site_name,
            )

            bid = int(
                model.site_bodyid[sid]
            )

            world = np.asarray(
                data.site_xpos[sid],
                dtype=float,
            ).copy()

            body_pos = np.asarray(
                data.xpos[bid],
                dtype=float,
            ).copy()

            body_R = np.asarray(
                data.xmat[bid],
                dtype=float,
            ).reshape(
                3,
                3,
            ).copy()


            rel = world - anchor

            axial = (
                axis
                *
                np.dot(
                    rel,
                    axis,
                )
            )

            radial = (
                rel
                -
                axial
            )


            ref[joint_name][
                "sites"
            ][site_name] = {
                "axial":
                    axial,

                "radial":
                    radial,

                "body_pos":
                    body_pos,

                "body_R":
                    body_R,

                "r0":
                    float(
                        np.linalg.norm(
                            radial
                        )
                    ),
            }


    return ref


REF = reference_geometry()


# ============================================================
# BUILD CANDIDATE
# ============================================================

def build_candidate(
    x,
    output,
):

    sm, sp, sd, tendon_k, pip_f = [
        float(v)
        for v in x
    ]


    scales = {
        MCP: sm,
        PIP: sp,
        DIP: sd,
    }


    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()


    for joint_name, site_names in GUIDES.items():

        scale = scales[
            joint_name
        ]

        anchor = REF[
            joint_name
        ]["anchor"]


        for site_name in site_names:

            r = REF[
                joint_name
            ]["sites"][site_name]


            desired_world = (
                anchor
                +
                r["axial"]
                +
                scale
                *
                r["radial"]
            )


            local = (
                r["body_R"].T
                @
                (
                    desired_world
                    -
                    r["body_pos"]
                )
            )


            e = get_site_element(
                root,
                site_name,
            )

            e.set(
                "pos",
                " ".join(
                    f"{v:.12g}"
                    for v in local
                ),
            )


    # Tendon stiffness.
    t = root.find(
        f".//spatial[@name='{TENDON}']"
    )

    if t is None:
        raise RuntimeError(
            "Missing spatial flexor tendon."
        )

    t.set(
        "stiffness",
        f"{tendon_k:.12g}",
    )


    # PIP friction is allowed to re-identify because P7 pushed it
    # against the old bound.
    jp = root.find(
        f".//joint[@name='{PIP}']"
    )

    if jp is None:
        raise RuntimeError(
            "Missing PIP joint."
        )

    jp.set(
        "frictionloss",
        f"{pip_f:.12g}",
    )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


    mujoco.MjModel.from_xml_path(
        str(output)
    )


# ============================================================
# CONTACT INITIALIZATION
# ============================================================

p3.b2.v30.v24.ab.patch_replay_modules()

CONTACT_REPLAY = (
    p3.b2.v30.v24.base.contact_replay
)

CONTACT_REPLAY.CONTACT_DAMPING = (
    p3.CONTACT_DAMPING
)

p3.b2.v35.CONTACT_MU = (
    p3.CONTACT_MU
)

p3.b2.v35.CONTACT_K = (
    p3.CONTACT_K
)


# ============================================================
# EVALUATION
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


def evaluate(
    x,
    keep=False,
):

    build_candidate(
        x,
        TEMP,
    )


    # ---------------- FREE SPACE ----------------

    fm, fp, fc, _ = p7.run_free(
        TEMP
    )


    # ---------------- CONTACT ----------------

    old_xml = p3.XML

    try:

        p3.XML = TEMP

        cr = p3.run_case(
            "V37",
            float(x[3]),
            True,
            CONTACT_REPLAY,
        )

    finally:

        p3.XML = old_xml


    contact_rmse = float(
        cr["rmse_deg"]
    )


    # ========================================================
    # SCORE
    #
    # Free and contact errors both matter.
    #
    # A candidate that never reaches the tip cannot masquerade
    # as a good solution merely because one snapshot is close.
    # ========================================================

    score = (
        fc
        +
        contact_rmse
    )


    # Require the physical contact sequence.
    if cr["mid_on"] < 1:

        score += 15.0


    if cr["tip_on"] < 1:

        score += 15.0


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


    if math.isfinite(ddip):

        # Distal joint must continue FLEXING after middle touch.
        if ddip <= 0.0:

            score += (
                4.0
                +
                2.0
                *
                abs(ddip)
            )


    if math.isfinite(dpip):

        # Middle link should become substantially constrained
        # rather than continuing another large 5+ degree sweep.
        if dpip > 4.0:

            score += (
                dpip
                -
                4.0
            )


    return (
        score,
        fm,
        fp,
        fc,
        cr,
    )


def objective(x):

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


    if score < BEST_SEEN["score"]:

        BEST_SEEN["score"] = (
            score
        )

        BEST_SEEN["x"] = (
            np.asarray(
                x,
                dtype=float,
            ).copy()
        )

        BEST_SEEN["free"] = (
            fm,
            fp,
            fc,
        )

        BEST_SEEN["contact"] = (
            dict(cr)
        )


        print()
        print(
            f"NEW V37 BEST [eval {COUNT}]"
        )

        print(
            "  guide scales M/P/D = "
            f"{x[0]:.3f} / "
            f"{x[1]:.3f} / "
            f"{x[2]:.3f}"
        )

        print(
            f"  tendon K = "
            f"{x[3]:.1f} N/m"
        )

        print(
            f"  PIP friction = "
            f"{x[4]:.6f} Nm"
        )

        print(
            "  free MCP/PIP/combined = "
            f"{fm:.3f} / "
            f"{fp:.3f} / "
            f"{fc:.3f} deg"
        )

        print(
            "  contact pose = "
            f"{cr['mcp_deg']:.2f} / "
            f"{cr['pip_deg']:.2f} / "
            f"{cr['dip_deg']:.2f} deg"
        )

        print(
            f"  contact RMSE = "
            f"{cr['rmse_deg']:.3f} deg"
        )

        print(
            "  contact events M/T = "
            f"{cr['mid_on']} / "
            f"{cr['tip_on']}"
        )

        print(
            "  middle->tip dPIP/dDIP = "
            f"{cr['middle_to_tip_dPIP_deg']:.3f} / "
            f"{cr['middle_to_tip_dDIP_deg']:.3f} deg"
        )

        print(
            f"  TOTAL SCORE = "
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
# MAIN
# ============================================================

def main():

    print()
    print("=" * 155)
    print(
        "V37 — CONFIGURATION-DEPENDENT SPATIAL GUIDE "
        "GEOMETRY IDENTIFICATION"
    )
    print("=" * 155)

    print(
        "Starting model:"
    )

    print(
        SOURCE
    )

    print()
    print(
        "Only real spatial-guide geometry is changed."
    )

    print(
        "No contact-triggered logic."
    )

    print(
        "No MCP/PIP/DIP trajectory prescription."
    )


    print()
    print("Original guide radial distances:")

    for j in [
        MCP,
        PIP,
        DIP,
    ]:

        print()
        print(j)

        for s, r in REF[j][
            "sites"
        ].items():

            print(
                f"  {s:26s} "
                f"{1000*r['r0']:.3f} mm"
            )


    result = differential_evolution(
        objective,
        BOUNDS,

        seed=37,

        popsize=4,

        maxiter=5,

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


    if BEST_SEEN["x"] is None:

        raise RuntimeError(
            "No valid V37 candidate."
        )


    x = BEST_SEEN["x"]


    build_candidate(
        x,
        BEST,
    )


    shutil.copy2(
        BEST,
        OUTDIR
        /
        BEST.name,
    )


    # Final deterministic evaluation.
    old_xml = p3.XML

    try:

        fm, fp, fc, _ = p7.run_free(
            BEST,
            OUTDIR
            /
            "v37_best_free.log",
        )


        p3.XML = BEST

        cr = p3.run_case(
            "V37_FINAL",
            float(x[3]),
            True,
            CONTACT_REPLAY,
        )

    finally:

        p3.XML = old_xml


    print()
    print("=" * 155)
    print("V37 FINAL")
    print("=" * 155)

    print(
        "guide radial scales M/P/D = "
        f"{x[0]:.4f} / "
        f"{x[1]:.4f} / "
        f"{x[2]:.4f}"
    )

    print(
        f"tendon K = "
        f"{x[3]:.2f} N/m"
    )

    print(
        f"PIP friction = "
        f"{x[4]:.6f} Nm"
    )

    print()
    print(
        "FREE:"
    )

    print(
        f"  MCP/PIP/combined = "
        f"{fm:.3f} / "
        f"{fp:.3f} / "
        f"{fc:.3f} deg"
    )


    print()
    print(
        "CONTACT:"
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
        "  middle -> first tip:"
    )

    print(
        f"    dPIP = "
        f"{cr['middle_to_tip_dPIP_deg']:.3f} deg"
    )

    print(
        f"    dDIP = "
        f"{cr['middle_to_tip_dDIP_deg']:.3f} deg"
    )

    print(
        f"  target gaps M/T = "
        f"{cr['target_mid_gap_mm']:.3f} / "
        f"{cr['target_tip_gap_mm']:.3f} mm"
    )


    print()
    print(
        f"BEST XML:\n{BEST}"
    )

    print(
        f"\nOUTPUT:\n{OUTDIR}"
    )

    print("=" * 155)


if __name__ == "__main__":
    main()
