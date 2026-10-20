#!/usr/bin/env python3

from pathlib import Path
import contextlib
import csv
import io
import math
import re
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as oldscan
import replay_index_dynamic_measured_encoder_v2 as replay


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = WS / "src/ur5e_mujoco_description/mjcf"

FREE_BAG = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P2_rh8d_hardware/"
    "rh8d_robot_control/v1_rh8dR_ws/"
    "rh8d_dashboard_data/rosbags/"
    "rh8d_bag_20260910_103316"
).expanduser().resolve()

OLD_SOURCE = Path(
    oldscan.FREE_SOURCE
).expanduser().resolve()

CURRENT = (
    MJCF
    / "ur5e_rh8d_index_cad_spatial_v35i1_best.xml"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v35p6b_free_loss_decomposition"
)

OUTCSV = (
    OUTDIR
    / "v35p6b_summary.csv"
)

SCALE = 5.10e-6
KT = 2100.0


JOINTS = [
    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",
]

TENDON = "idx_flex_spatial"


def mj_id(model, obj, name):

    i = mujoco.mj_name2id(
        model,
        obj,
        name,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing MuJoCo object: {name}"
        )

    return i


def compiled_joint_params(path):

    model = mujoco.MjModel.from_xml_path(
        str(path)
    )

    out = {}

    for name in JOINTS:

        ji = mj_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        da = int(
            model.jnt_dofadr[ji]
        )

        out[name] = {
            "stiffness":
                float(
                    model.jnt_stiffness[ji]
                ),

            "damping":
                float(
                    model.dof_damping[da]
                ),

            "frictionloss":
                float(
                    model.dof_frictionloss[da]
                ),

            "springref":
                float(
                    model.qpos_spring[
                        int(
                            model.jnt_qposadr[ji]
                        )
                    ]
                ),
        }

    return out


def exact_old_candidate():

    path = (
        MJCF
        / "tmp_p6b_exact_old_baseline.xml"
    )

    oldscan.modify_candidate_xml(
        OLD_SOURCE,
        path,
        0.0310,
        0.0060,
    )

    # Make absolutely explicit what the historical
    # candidate evaluator intended.
    tree = ET.parse(path)
    root = tree.getroot()

    t = root.find(
        f".//fixed[@name='{TENDON}']"
    )

    if t is None:
        raise RuntimeError(
            "Historical fixed tendon not found."
        )

    t.set(
        "stiffness",
        str(KT),
    )

    tree.write(
        path,
        encoding="unicode",
        xml_declaration=True,
    )

    # Compile now so malformed historical attributes
    # cannot silently enter the experiment.
    mujoco.MjModel.from_xml_path(
        str(path)
    )

    return path


def patch_joints(
    source,
    output,
    params,
):

    tree = ET.parse(source)
    root = tree.getroot()

    for name in JOINTS:

        j = root.find(
            f".//joint[@name='{name}']"
        )

        if j is None:
            raise RuntimeError(
                f"Missing joint {name}"
            )

        p = params[name]

        j.set(
            "stiffness",
            f"{p['stiffness']:.12g}",
        )

        j.set(
            "damping",
            f"{p['damping']:.12g}",
        )

        j.set(
            "frictionloss",
            f"{p['frictionloss']:.12g}",
        )

        # Preserve source springref semantics.
        # We intentionally do not transplant qpos_spring here
        # because the old and new XMLs already use the relevant
        # nominal references.

    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


def set_tendon_damping(
    path,
    damping,
):

    tree = ET.parse(path)
    root = tree.getroot()

    found = None

    for tag in [
        "fixed",
        "spatial",
    ]:

        e = root.find(
            f".//{tag}[@name='{TENDON}']"
        )

        if e is not None:
            found = e
            break

    if found is None:
        raise RuntimeError(
            f"Missing tendon {TENDON}"
        )

    found.set(
        "stiffness",
        str(KT),
    )

    found.set(
        "damping",
        str(damping),
    )

    tree.write(
        path,
        encoding="unicode",
        xml_declaration=True,
    )


def scalar(
    text,
    pattern,
):

    m = re.search(
        pattern,
        text,
        re.MULTILINE | re.IGNORECASE,
    )

    if m is None:
        return math.nan

    return float(
        m.group(1)
    )


def parse_reference_rows(text):

    pat = re.compile(
        r"^\s*"
        r"(closing|opening)\s+"
        r"([0-9.]+)\s+"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s*$",
        re.MULTILINE,
    )

    rows = []

    for m in pat.finditer(text):

        rows.append({
            "phase":
                m.group(1),

            "target":
                float(
                    m.group(2)
                ),

            "motor":
                float(
                    m.group(3)
                ),

            "hw_mcp":
                float(
                    m.group(4)
                ),

            "sim_mcp":
                float(
                    m.group(5)
                ),

            "hw_pip":
                float(
                    m.group(6)
                ),

            "sim_pip":
                float(
                    m.group(7)
                ),
        })

    return rows


def rmse(rows, phase=None):

    use = [
        r
        for r in rows
        if (
            phase is None
            or
            r["phase"] == phase
        )
    ]

    em = np.asarray(
        [
            r["sim_mcp"]
            -
            r["hw_mcp"]
            for r in use
        ],
        dtype=float,
    )

    ep = np.asarray(
        [
            r["sim_pip"]
            -
            r["hw_pip"]
            for r in use
        ],
        dtype=float,
    )

    m = float(
        np.sqrt(
            np.mean(
                em * em
            )
        )
    )

    p = float(
        np.sqrt(
            np.mean(
                ep * ep
            )
        )
    )

    c = float(
        np.sqrt(
            (
                m*m
                +
                p*p
            )
            /
            2.0
        )
    )

    return m, p, c


def run_model(
    label,
    xml,
):

    old_argv = list(
        sys.argv
    )

    old_model = replay.MODEL_PATH
    old_scale = replay.METERS_PER_TICK

    old_k = getattr(
        replay,
        "TENDON_STIFFNESS",
        None,
    )


    try:

        replay.MODEL_PATH = xml
        replay.METERS_PER_TICK = SCALE

        if hasattr(
            replay,
            "TENDON_STIFFNESS",
        ):
            replay.TENDON_STIFFNESS = KT


        sys.argv = [
            replay.__file__,
            str(
                FREE_BAG
            ),
        ]


        stream = io.StringIO()

        with contextlib.redirect_stdout(
            stream
        ):
            replay.main()


        text = stream.getvalue()


    finally:

        replay.MODEL_PATH = old_model
        replay.METERS_PER_TICK = old_scale

        if (
            old_k is not None
            and
            hasattr(
                replay,
                "TENDON_STIFFNESS",
            )
        ):
            replay.TENDON_STIFFNESS = old_k

        sys.argv = old_argv


    (
        OUTDIR
        /
        f"{label}.log"
    ).write_text(
        text
    )


    rows = parse_reference_rows(
        text
    )

    mc, pc, cc = rmse(
        rows,
        "closing",
    )

    mo, po, co = rmse(
        rows,
        "opening",
    )

    ma, pa, ca = rmse(
        rows,
        None,
    )


    return {
        "case":
            label,

        "xml":
            str(xml),

        "motor_rmse_ticks":
            scalar(
                text,
                r"encoder RMSE\s*=\s*"
                r"([-+0-9.eE]+)\s*ticks",
            ),

        "closing_mcp":
            mc,

        "closing_pip":
            pc,

        "closing_combined":
            cc,

        "opening_mcp":
            mo,

        "opening_pip":
            po,

        "opening_combined":
            co,

        "all_mcp":
            ma,

        "all_pip":
            pa,

        "all_combined":
            ca,
    }


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    old_exact = exact_old_candidate()

    print()
    print("=" * 150)
    print("P6B — FREE-SPACE LOSS CAUSAL DECOMPOSITION")
    print("=" * 150)

    print(
        "\nA exact historical candidate:\n"
        f"{old_exact}"
    )

    print(
        "\nE current V35I1:\n"
        f"{CURRENT}"
    )


    old_params = compiled_joint_params(
        old_exact
    )

    current_params = compiled_joint_params(
        CURRENT
    )


    print()
    print("=" * 150)
    print("COMPILED JOINT PARAMETERS")
    print("=" * 150)

    for name in JOINTS:

        print()
        print(name)

        print(
            "  OLD     "
            f"k={old_params[name]['stiffness']:.6f}  "
            f"d={old_params[name]['damping']:.6f}  "
            f"f={old_params[name]['frictionloss']:.6f}"
        )

        print(
            "  CURRENT "
            f"k={current_params[name]['stiffness']:.6f}  "
            f"d={current_params[name]['damping']:.6f}  "
            f"f={current_params[name]['frictionloss']:.6f}"
        )


    # ---------------------------------------------------------
    # A: old transmission + old passives + damping 2
    # ---------------------------------------------------------

    A = (
        MJCF
        / "tmp_p6b_A_oldTrans_oldPassive_d2.xml"
    )

    ET.parse(
        old_exact
    ).write(
        A,
        encoding="unicode",
        xml_declaration=True,
    )

    set_tendon_damping(
        A,
        2.0,
    )


    # ---------------------------------------------------------
    # B: old transmission + current passives + damping 2
    # ---------------------------------------------------------

    B = (
        MJCF
        / "tmp_p6b_B_oldTrans_currentPassive_d2.xml"
    )

    patch_joints(
        old_exact,
        B,
        current_params,
    )

    set_tendon_damping(
        B,
        2.0,
    )


    # ---------------------------------------------------------
    # C: spatial transmission + old passives + damping 2
    # ---------------------------------------------------------

    C = (
        MJCF
        / "tmp_p6b_C_spatial_oldPassive_d2.xml"
    )

    patch_joints(
        CURRENT,
        C,
        old_params,
    )

    set_tendon_damping(
        C,
        2.0,
    )


    # ---------------------------------------------------------
    # D: spatial transmission + current passives + damping 2
    # ---------------------------------------------------------

    D = (
        MJCF
        / "tmp_p6b_D_spatial_currentPassive_d2.xml"
    )

    ET.parse(
        CURRENT
    ).write(
        D,
        encoding="unicode",
        xml_declaration=True,
    )

    set_tendon_damping(
        D,
        2.0,
    )


    # ---------------------------------------------------------
    # E: current V35I1 exactly
    # ---------------------------------------------------------

    E = CURRENT


    cases = [
        (
            "A_oldTrans_oldPassive_d2",
            A,
        ),
        (
            "B_oldTrans_currentPassive_d2",
            B,
        ),
        (
            "C_spatial_oldPassive_d2",
            C,
        ),
        (
            "D_spatial_currentPassive_d2",
            D,
        ),
        (
            "E_spatial_currentPassive_d12",
            E,
        ),
    ]


    # Compile every model before running.
    for label, path in cases:

        mujoco.MjModel.from_xml_path(
            str(path)
        )

        print(
            f"\nCOMPILED: {label}"
        )


    results = []


    for i, (
        label,
        path,
    ) in enumerate(
        cases,
        1,
    ):

        print()
        print("-" * 150)

        print(
            f"[{i}/{len(cases)}] "
            f"{label}"
        )


        result = run_model(
            label,
            path,
        )

        results.append(
            result
        )


        print(
            "  closing combined = "
            f"{result['closing_combined']:.3f} deg"
        )

        print(
            "  opening combined = "
            f"{result['opening_combined']:.3f} deg"
        )

        print(
            "  ALL MCP/PIP/combined = "
            f"{result['all_mcp']:.3f} / "
            f"{result['all_pip']:.3f} / "
            f"{result['all_combined']:.3f} deg"
        )


    with OUTCSV.open(
        "w",
        newline="",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )


    lookup = {
        r["case"]:
            r
        for r in results
    }


    Ares = lookup[
        "A_oldTrans_oldPassive_d2"
    ]

    Bres = lookup[
        "B_oldTrans_currentPassive_d2"
    ]

    Cres = lookup[
        "C_spatial_oldPassive_d2"
    ]

    Dres = lookup[
        "D_spatial_currentPassive_d2"
    ]

    Eres = lookup[
        "E_spatial_currentPassive_d12"
    ]


    print()
    print("=" * 165)
    print("P6B SUMMARY")
    print("=" * 165)

    print(
        "case                              "
        "closing     opening      "
        "MCP        PIP       ALL combined"
    )

    print("-" * 165)

    for r in results:

        print(
            f"{r['case']:34s} "
            f"{r['closing_combined']:8.3f}    "
            f"{r['opening_combined']:8.3f}    "
            f"{r['all_mcp']:8.3f}   "
            f"{r['all_pip']:8.3f}   "
            f"{r['all_combined']:8.3f}"
        )


    print()
    print("=" * 165)
    print("CAUSAL DELTAS — ALL COMBINED RMSE")
    print("=" * 165)

    print(
        "PASSIVE CHANGE under OLD transmission:"
    )

    print(
        f"  B - A = "
        f"{Bres['all_combined'] - Ares['all_combined']:+.3f} deg"
    )


    print()
    print(
        "TRANSMISSION change with OLD passives:"
    )

    print(
        f"  C - A = "
        f"{Cres['all_combined'] - Ares['all_combined']:+.3f} deg"
    )


    print()
    print(
        "PASSIVE CHANGE under SPATIAL transmission:"
    )

    print(
        f"  D - C = "
        f"{Dres['all_combined'] - Cres['all_combined']:+.3f} deg"
    )


    print()
    print(
        "TENDON DAMPING 2 -> 12 under CURRENT spatial/passives:"
    )

    print(
        f"  E - D = "
        f"{Eres['all_combined'] - Dres['all_combined']:+.3f} deg"
    )


    print()
    print("=" * 165)
    print("DECISION")
    print("=" * 165)

    delta_passive = abs(
        Bres["all_combined"]
        -
        Ares["all_combined"]
    )

    delta_trans = abs(
        Cres["all_combined"]
        -
        Ares["all_combined"]
    )

    delta_damp = abs(
        Eres["all_combined"]
        -
        Dres["all_combined"]
    )


    if (
        delta_trans
        >
        delta_passive
        and
        delta_trans
        >
        delta_damp
    ):

        print(
            "PRIMARY SOURCE OF FREE-SPACE LOSS: "
            "TRANSMISSION / SPATIAL ROUTING."
        )

    elif (
        delta_passive
        >
        delta_trans
        and
        delta_passive
        >
        delta_damp
    ):

        print(
            "PRIMARY SOURCE OF FREE-SPACE LOSS: "
            "PASSIVE JOINT MECHANICS."
        )

    else:

        print(
            "PRIMARY SOURCE OF FREE-SPACE LOSS: "
            "TENDON DAMPING OR STRONG INTERACTION."
        )


    print()
    print(
        f"CSV:\n{OUTCSV}"
    )

    print(
        f"\nLogs:\n{OUTDIR}"
    )

    print("=" * 165)


if __name__ == "__main__":
    main()
