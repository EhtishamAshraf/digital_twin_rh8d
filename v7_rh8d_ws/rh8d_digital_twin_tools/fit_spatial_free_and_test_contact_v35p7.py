#!/usr/bin/env python3

from pathlib import Path
import contextlib
import io
import math
import re
import sys
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

try:
    from scipy.optimize import differential_evolution
except Exception as e:
    raise RuntimeError(
        "scipy is required for P7.\n"
        f"Import error: {e}"
    )

import replay_index_dynamic_measured_encoder_v2 as free_replay
import test_prescribed_spool_v35p3 as contact_p3


# =====================================================================
# PATHS
# =====================================================================

WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

TOOLS = (
    WS
    / "rh8d_digital_twin_tools"
)

SOURCE = (
    MJCF
    / "ur5e_rh8d_index_cad_spatial_v35i1_best.xml"
)

TEMP_XML = (
    MJCF
    / "tmp_v35p7_spatial_fit.xml"
)

BEST_XML = (
    MJCF
    / "ur5e_rh8d_index_cad_spatial_v35p7_freefit.xml"
)

FREE_BAG = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P2_rh8d_hardware/"
    "rh8d_robot_control/v1_rh8dR_ws/"
    "rh8d_dashboard_data/rosbags/"
    "rh8d_bag_20260910_103316"
).expanduser().resolve()

OUTDIR = (
    TOOLS
    / "results"
    / "v35p7_spatial_freefit_and_contact"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# FROZEN QUANTITIES
# =====================================================================

METERS_PER_TICK = 5.10e-6

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

TENDON = "idx_flex_spatial"


# Joint damping stays frozen.
#
# P6B proved changing tendon damping 2 -> 12 changed total RMSE
# by only ~0.002 deg, therefore it is NOT included as a fit variable.
#
# Current spatial tendon geometry and spring references are also frozen.


# =====================================================================
# PARAMETERS TO IDENTIFY
# =====================================================================
#
# x = [
#     tendon stiffness [N/m],
#     MCP stiffness [Nm/rad],
#     PIP stiffness [Nm/rad],
#     PIP frictionloss [Nm],
#     DIP stiffness [Nm/rad],
#     DIP frictionloss [Nm],
# ]
#
# Bounds cover the historical and current values without allowing
# absurdly large compensation.
# =====================================================================

BOUNDS = [
    (1000.0, 6000.0),   # tendon K
    (0.0020, 0.0150),   # MCP K
    (0.0020, 0.0400),   # PIP K
    (0.0000, 0.0080),   # PIP friction
    (0.0050, 0.0800),   # DIP K
    (0.0000, 0.0020),   # DIP friction
]


# =====================================================================
# XML MODIFICATION
# =====================================================================

def joint(root, name):

    e = root.find(
        f".//joint[@name='{name}']"
    )

    if e is None:
        raise RuntimeError(
            f"Could not find joint: {name}"
        )

    return e


def tendon(root):

    for tag in [
        "spatial",
        "fixed",
    ]:

        e = root.find(
            f".//{tag}[@name='{TENDON}']"
        )

        if e is not None:
            return e

    raise RuntimeError(
        f"Could not find tendon: {TENDON}"
    )


def build_candidate(
    x,
    output,
):

    (
        tendon_k,
        mcp_k,
        pip_k,
        pip_f,
        dip_k,
        dip_f,
    ) = [
        float(v)
        for v in x
    ]


    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()


    jm = joint(
        root,
        MCP,
    )

    jp = joint(
        root,
        PIP,
    )

    jd = joint(
        root,
        DIP,
    )


    jm.set(
        "stiffness",
        f"{mcp_k:.12g}",
    )

    # MCP friction remains the measured/current common value.
    # It was identical in OLD and V35I1 and is not fitted.
    jm.set(
        "frictionloss",
        "0.002",
    )


    jp.set(
        "stiffness",
        f"{pip_k:.12g}",
    )

    jp.set(
        "frictionloss",
        f"{pip_f:.12g}",
    )


    jd.set(
        "stiffness",
        f"{dip_k:.12g}",
    )

    jd.set(
        "frictionloss",
        f"{dip_f:.12g}",
    )


    t = tendon(
        root
    )

    t.set(
        "stiffness",
        f"{tendon_k:.12g}",
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


# =====================================================================
# FREE-SPACE EVALUATOR
# =====================================================================

NUM = r"[-+0-9.eE]+"


def regex_value(
    text,
    pattern,
):

    m = re.search(
        pattern,
        text,
        re.MULTILINE,
    )

    if m is None:
        raise RuntimeError(
            "Could not parse:\n"
            + pattern
        )

    return float(
        m.group(1)
    )


def run_free(
    xml,
    keep_log=None,
):

    old_argv = list(
        sys.argv
    )

    old_model = (
        free_replay.MODEL_PATH
    )

    old_scale = (
        free_replay.METERS_PER_TICK
    )

    old_tendon_global = getattr(
        free_replay,
        "TENDON_STIFFNESS",
        None,
    )


    try:

        free_replay.MODEL_PATH = (
            xml
        )

        free_replay.METERS_PER_TICK = (
            METERS_PER_TICK
        )


        # If this historical module exposes a tendon global,
        # use the value already stored in the candidate XML.
        if hasattr(
            free_replay,
            "TENDON_STIFFNESS",
        ):

            model = mujoco.MjModel.from_xml_path(
                str(xml)
            )

            tid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_TENDON,
                TENDON,
            )

            free_replay.TENDON_STIFFNESS = float(
                model.tendon_stiffness[
                    tid
                ]
            )


        sys.argv = [
            free_replay.__file__,
            str(
                FREE_BAG
            ),
        ]


        stream = io.StringIO()


        with contextlib.redirect_stdout(
            stream
        ):

            free_replay.main()


        text = stream.getvalue()


    finally:

        sys.argv = old_argv

        free_replay.MODEL_PATH = (
            old_model
        )

        free_replay.METERS_PER_TICK = (
            old_scale
        )


        if (
            old_tendon_global is not None
            and
            hasattr(
                free_replay,
                "TENDON_STIFFNESS",
            )
        ):

            free_replay.TENDON_STIFFNESS = (
                old_tendon_global
            )


    mcp_rmse = regex_value(
        text,
        rf"MCP RMSE\s*=\s*({NUM})\s*deg",
    )

    pip_rmse = regex_value(
        text,
        rf"PIP RMSE\s*=\s*({NUM})\s*deg",
    )


    combined = math.sqrt(
        (
            mcp_rmse**2
            +
            pip_rmse**2
        )
        /
        2.0
    )


    if keep_log is not None:

        Path(
            keep_log
        ).write_text(
            text
        )


    return (
        mcp_rmse,
        pip_rmse,
        combined,
        text,
    )


# =====================================================================
# OPTIMIZATION
# =====================================================================

EVAL_COUNT = 0

BEST_SEEN = {
    "rmse":
        float("inf"),

    "x":
        None,
}


def objective(x):

    global EVAL_COUNT

    EVAL_COUNT += 1


    try:

        build_candidate(
            x,
            TEMP_XML,
        )


        m, p, c, _ = run_free(
            TEMP_XML
        )


    except Exception as exc:

        print(
            f"[{EVAL_COUNT:03d}] FAILED: {exc}"
        )

        return 1000.0


    if c < BEST_SEEN["rmse"]:

        BEST_SEEN["rmse"] = (
            c
        )

        BEST_SEEN["x"] = (
            np.asarray(
                x,
                dtype=float,
            ).copy()
        )


        print()
        print(
            "NEW BEST "
            f"[eval {EVAL_COUNT}]"
        )

        print(
            f"  free MCP/PIP/combined = "
            f"{m:.3f} / "
            f"{p:.3f} / "
            f"{c:.3f} deg"
        )

        print(
            "  "
            f"Kt={x[0]:.1f}  "
            f"kM={x[1]:.5f}  "
            f"kP={x[2]:.5f}  "
            f"fP={x[3]:.5f}  "
            f"kD={x[4]:.5f}  "
            f"fD={x[5]:.5f}"
        )


    elif (
        EVAL_COUNT
        %
        10
        ==
        0
    ):

        print(
            f"[{EVAL_COUNT:03d}] "
            f"current={c:.3f} deg   "
            f"best={BEST_SEEN['rmse']:.3f} deg"
        )


    return c


# =====================================================================
# MAIN
# =====================================================================

def main():

    if not SOURCE.exists():

        raise RuntimeError(
            f"Missing source:\n{SOURCE}"
        )


    if not (
        FREE_BAG
        / "metadata.yaml"
    ).exists():

        raise RuntimeError(
            f"Missing free-space bag:\n{FREE_BAG}"
        )


    print()
    print("=" * 150)
    print(
        "V35P7 — DIRECT SPATIAL FREE-SPACE IDENTIFICATION "
        "+ IMMEDIATE CONTACT TEST"
    )
    print("=" * 150)

    print()
    print(
        "No more manual parameter-by-parameter loop."
    )

    print()
    print(
        "Frozen:"
    )

    print(
        "  CAD spatial tendon route"
    )

    print(
        "  encoder -> spool scale = "
        "5.10 um/tick"
    )

    print(
        "  MEASURED encoder drives finger mechanics"
    )

    print(
        "  joint damping = current values"
    )

    print(
        "  spring references = current values"
    )

    print(
        "  MCP dry friction = 0.002 Nm"
    )

    print()
    print(
        "Identified simultaneously:"
    )

    print(
        "  tendon K"
    )

    print(
        "  MCP K"
    )

    print(
        "  PIP K + friction"
    )

    print(
        "  DIP K + friction"
    )


    # ---------------------------------------------------------
    # Current baseline
    # ---------------------------------------------------------

    print()
    print("=" * 150)
    print("CURRENT SPATIAL BASELINE")
    print("=" * 150)


    bm, bp, bc, _ = run_free(
        SOURCE,
        OUTDIR / "baseline_free.log",
    )


    print(
        f"MCP/PIP/combined = "
        f"{bm:.3f} / "
        f"{bp:.3f} / "
        f"{bc:.3f} deg"
    )


    # ---------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------

    print()
    print("=" * 150)
    print("AUTOMATED IDENTIFICATION")
    print("=" * 150)

    print(
        "Approximately 100-150 free-space simulations."
    )

    print(
        "This may take several minutes."
    )


    result = differential_evolution(
        objective,
        BOUNDS,

        seed=17,

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


    if (
        BEST_SEEN["x"]
        is None
    ):

        raise RuntimeError(
            "Optimizer produced no valid candidate."
        )


    best_x = (
        BEST_SEEN["x"]
    )


    build_candidate(
        best_x,
        BEST_XML,
    )


    shutil.copy2(
        BEST_XML,
        OUTDIR
        /
        BEST_XML.name,
    )


    fm, fp, fc, _ = run_free(
        BEST_XML,
        OUTDIR / "best_free.log",
    )


    print()
    print("=" * 150)
    print("P7 BEST FREE-SPACE SPATIAL MODEL")
    print("=" * 150)

    print(
        f"Tendon stiffness = "
        f"{best_x[0]:.2f} N/m"
    )

    print(
        f"MCP stiffness    = "
        f"{best_x[1]:.6f} Nm/rad"
    )

    print(
        f"PIP stiffness    = "
        f"{best_x[2]:.6f} Nm/rad"
    )

    print(
        f"PIP friction     = "
        f"{best_x[3]:.6f} Nm"
    )

    print(
        f"DIP stiffness    = "
        f"{best_x[4]:.6f} Nm/rad"
    )

    print(
        f"DIP friction     = "
        f"{best_x[5]:.6f} Nm"
    )

    print()

    print(
        f"FREE MCP RMSE      = "
        f"{fm:.3f} deg"
    )

    print(
        f"FREE PIP RMSE      = "
        f"{fp:.3f} deg"
    )

    print(
        f"FREE combined RMSE = "
        f"{fc:.3f} deg"
    )

    print(
        f"improvement        = "
        f"{bc-fc:+.3f} deg"
    )

    print(
        f"\nBest XML:\n{BEST_XML}"
    )


    # ---------------------------------------------------------
    # Gate before contact
    # ---------------------------------------------------------

    print()
    print("=" * 150)
    print("FREE-SPACE GATE")
    print("=" * 150)


    if fc > 7.0:

        print(
            "FAIL: even automated passive/compliance "
            "identification cannot recover acceptable "
            "free-space behavior with this CAD spatial route."
        )

        print()
        print(
            "DECISION:"
        )

        print(
            "Stop tuning passives."
        )

        print(
            "The spatial transmission geometry/effective "
            "routing itself must be changed."
        )

        print()
        print(
            f"Outputs:\n{OUTDIR}"
        )

        return


    print(
        "PASS: spatial model can reproduce free-space "
        "well enough to test contact."
    )


    # ---------------------------------------------------------
    # IMMEDIATE CONTACT TEST
    # ---------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "CONTACT TEST — SAME IDENTIFIED MODEL, "
        "MEASURED ENCODER PRESCRIBED"
    )
    print("=" * 150)


    contact_dir = (
        OUTDIR
        /
        "contact_test"
    )

    contact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    old_xml = (
        contact_p3.XML
    )

    old_outdir = (
        contact_p3.OUTDIR
    )

    old_outcsv = (
        contact_p3.OUTCSV
    )

    old_cases = list(
        contact_p3.CASES
    )


    try:

        contact_p3.XML = (
            BEST_XML
        )

        contact_p3.OUTDIR = (
            contact_dir
        )

        contact_p3.OUTCSV = (
            contact_dir
            /
            "p7_best_contact_summary.csv"
        )

        contact_p3.CASES = [
            (
                "P7_FREEFIT_CONTACT",
                float(
                    best_x[0]
                ),
                True,
            )
        ]


        contact_p3.main()


    finally:

        contact_p3.XML = (
            old_xml
        )

        contact_p3.OUTDIR = (
            old_outdir
        )

        contact_p3.OUTCSV = (
            old_outcsv
        )

        contact_p3.CASES = (
            old_cases
        )


    print()
    print("=" * 150)
    print("P7 COMPLETE")
    print("=" * 150)

    print(
        "One model was identified ONLY from free-space "
        "MCP/PIP motion."
    )

    print(
        "The exact same model was then tested on contact "
        "without fitting contact parameters."
    )

    print(
        f"\nEverything is under:\n{OUTDIR}"
    )

    print("=" * 150)


if __name__ == "__main__":
    main()
