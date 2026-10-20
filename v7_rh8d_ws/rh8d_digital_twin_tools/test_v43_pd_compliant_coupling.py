#!/usr/bin/env python3

from pathlib import Path
import csv
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import fit_spatial_free_and_test_contact_v35p7 as p7
import fit_spatial_guide_geometry_v37 as v37


# ============================================================
# PATHS
# ============================================================

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
    / "v43_pd_compliant_coupling"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTCSV = OUTDIR / "v43_summary.csv"


# ============================================================
# NAMES
# ============================================================

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

COUPLING = "idx_pip_dip_compliant_coupling"

FLEXOR_K = 2831.47


# ============================================================
# HARDWARE TARGET
# ============================================================

HW_MCP = 14.5355
HW_PIP = 23.8110
HW_DIP = 9.3270


# ============================================================
# COUPLING
#
# L = r * (qPIP - qDIP)
#
# Preferred relative angle:
#
# qPIP - qDIP = qrel_ref
#
# If current relative angle is too small:
#
#   PIP receives + torque
#   DIP receives - torque
#
# MCP is not directly part of this elastic element.
# ============================================================

R = 0.003  # 3 mm effective radius


# Small diagnostic grid only.
#
# Not another huge optimizer.
#
# 1 baseline + 15 coupling cases.
# ============================================================

K_VALUES = [
    250.0,
    500.0,
    750.0,
    1000.0,
    1500.0,
]

RELATIVE_REST_DEG = [
    10.0,
    15.0,
    20.0,
]


CASES = [
    (0.0, 0.0)
]

for k in K_VALUES:
    for qref in RELATIVE_REST_DEG:
        CASES.append(
            (k, qref)
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
# XML BUILD
# ============================================================

def build_candidate(
    stiffness,
    relative_rest_deg,
    output,
):

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


    # Safe rerun.
    for e in list(
        tendon_root
    ):

        if (
            e.get("name")
            ==
            COUPLING
        ):

            tendon_root.remove(
                e
            )


    rest_length = (
        R
        *
        math.radians(
            relative_rest_deg
        )
    )


    fixed = ET.SubElement(
        tendon_root,
        "fixed",
        {
            "name":
                COUPLING,

            "stiffness":
                f"{float(stiffness):.12g}",

            "damping":
                "0",

            "springlength":
                f"{rest_length:.12g}",
        },
    )


    # Positive PIP coefficient.
    ET.SubElement(
        fixed,
        "joint",
        {
            "joint":
                PIP,

            "coef":
                f"{R:.12g}",
        },
    )


    # Negative DIP coefficient:
    # this is the critical differential coupling.
    ET.SubElement(
        fixed,
        "joint",
        {
            "joint":
                DIP,

            "coef":
                f"{-R:.12g}",
        },
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
# TORQUE-DIRECTION SANITY CHECK
# ============================================================

def dof_index(
    model,
    joint_name,
):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if jid < 0:

        raise RuntimeError(
            f"Joint not found: {joint_name}"
        )

    return int(
        model.jnt_dofadr[
            jid
        ]
    )


def spring_qfrc(
    xml,
):

    model = mujoco.MjModel.from_xml_path(
        str(xml)
    )

    data = mujoco.MjData(
        model
    )

    # Evaluate at fully open configuration.
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    return (
        model,
        data.qfrc_spring.copy(),
    )


def sanity_check():

    candidate = (
        MJCF
        /
        "tmp_v43_direction_check.xml"
    )


    build_candidate(
        750.0,
        15.0,
        candidate,
    )


    m0, q0 = spring_qfrc(
        SOURCE
    )

    m1, q1 = spring_qfrc(
        candidate
    )


    dm = (
        q1[
            dof_index(
                m1,
                MCP,
            )
        ]
        -
        q0[
            dof_index(
                m0,
                MCP,
            )
        ]
    )


    dp = (
        q1[
            dof_index(
                m1,
                PIP,
            )
        ]
        -
        q0[
            dof_index(
                m0,
                PIP,
            )
        ]
    )


    dd = (
        q1[
            dof_index(
                m1,
                DIP,
            )
        ]
        -
        q0[
            dof_index(
                m0,
                DIP,
            )
        ]
    )


    print()
    print("=" * 150)
    print("COUPLING DIRECTION SANITY")
    print("=" * 150)

    print(
        f"Added MCP spring torque = "
        f"{dm:+.8f} Nm"
    )

    print(
        f"Added PIP spring torque = "
        f"{dp:+.8f} Nm"
    )

    print(
        f"Added DIP spring torque = "
        f"{dd:+.8f} Nm"
    )


    # We explicitly require:
    #
    # MCP approximately unaffected
    # PIP flexion-positive
    # DIP flexion-negative

    if abs(dm) > 1e-7:

        raise RuntimeError(
            "Coupling unexpectedly acts directly on MCP."
        )


    if not (
        dp > 0.0
        and
        dd < 0.0
    ):

        raise RuntimeError(
            "Wrong PIP/DIP coupling direction."
        )


    print()
    print(
        "PASS: coupling acts +PIP / -DIP "
        "with no direct MCP torque."
    )


# ============================================================
# CONTACT
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

        cr = v37.p3.run_case(
            label,
            FLEXOR_K,
            True,
            CONTACT_REPLAY,
        )

    finally:

        v37.p3.XML = (
            old_xml
        )


    return cr


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 170)
    print(
        "V43 — COMPLIANT PIP/DIP DIFFERENTIAL COUPLING"
    )
    print("=" * 170)

    print()
    print(
        "V41/V42 proved that independent DIP resistance "
        "moves PIP/DIP correctly but also steals MCP motion."
    )

    print()
    print(
        "V43 therefore applies opposite internal torques "
        "only to PIP and DIP."
    )


    # --------------------------------------------------------
    # Mandatory sign check before any experiment.
    # --------------------------------------------------------

    sanity_check()


    rows = []


    for i, (
        stiffness,
        qref,
    ) in enumerate(
        CASES,
        1,
    ):

        xml = (
            MJCF
            /
            f"tmp_v43_k{stiffness:.0f}_r{qref:.0f}.xml"
        )


        build_candidate(
            stiffness,
            qref,
            xml,
        )


        print()
        print("-" * 170)

        print(
            f"[{i}/{len(CASES)}] "
            f"K={stiffness:.1f} N/m  "
            f"preferred(PIP-DIP)={qref:.1f} deg"
        )


        fm, fp, fc, _ = (
            p7.run_free(
                xml
            )
        )


        cr = run_contact(
            xml,
            f"V43_{i}",
        )


        em = (
            cr["mcp_deg"]
            -
            HW_MCP
        )

        ep = (
            cr["pip_deg"]
            -
            HW_PIP
        )

        ed = (
            cr["dip_deg"]
            -
            HW_DIP
        )


        max_joint_error = max(
            abs(em),
            abs(ep),
            abs(ed),
        )


        row = {
            "K_Npm":
                stiffness,

            "relative_rest_deg":
                qref,

            "free_mcp_rmse":
                fm,

            "free_pip_rmse":
                fp,

            "free_combined":
                fc,

            "contact_mcp":
                cr["mcp_deg"],

            "contact_pip":
                cr["pip_deg"],

            "contact_dip":
                cr["dip_deg"],

            "mcp_error":
                em,

            "pip_error":
                ep,

            "dip_error":
                ed,

            "max_joint_error":
                max_joint_error,

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
            f"{cr['dip_deg']:.3f}"
        )

        print(
            "CONTACT errors = "
            f"{em:+.3f} / "
            f"{ep:+.3f} / "
            f"{ed:+.3f} deg"
        )

        print(
            f"contact RMSE = "
            f"{cr['rmse_deg']:.3f} deg"
        )

        print(
            f"max joint error = "
            f"{max_joint_error:.3f} deg"
        )

        print(
            f"M/T = "
            f"{cr['mid_on']} / "
            f"{cr['tip_on']}"
        )

        print(
            "dPIP/dDIP = "
            f"{cr['middle_to_tip_dPIP_deg']:+.3f} / "
            f"{cr['middle_to_tip_dDIP_deg']:+.3f}"
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


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 190)
    print("V43 SUMMARY")
    print("=" * 190)

    print(
        "K      rest   freeRMSE    "
        "MCP / PIP / DIP            "
        "errors M/P/D              "
        "contactRMSE  maxErr   M/T"
    )

    print("-" * 190)


    for r in rows:

        print(
            f"{r['K_Npm']:6.0f} "
            f"{r['relative_rest_deg']:6.1f} "
            f"{r['free_combined']:9.3f}   "
            f"{r['contact_mcp']:6.2f} / "
            f"{r['contact_pip']:6.2f} / "
            f"{r['contact_dip']:6.2f}   "
            f"{r['mcp_error']:+6.2f} / "
            f"{r['pip_error']:+6.2f} / "
            f"{r['dip_error']:+6.2f}   "
            f"{r['contact_rmse']:8.3f} "
            f"{r['max_joint_error']:7.3f} "
            f"{r['mid_on']}/{r['tip_on']}"
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

        best = min(
            valid,
            key=lambda r:
                (
                    r["max_joint_error"]
                    +
                    0.25
                    *
                    r["free_combined"]
                ),
        )


        print()
        print("=" * 190)
        print("BEST V43")
        print("=" * 190)

        print(
            f"K = "
            f"{best['K_Npm']:.1f} N/m"
        )

        print(
            f"preferred PIP-DIP difference = "
            f"{best['relative_rest_deg']:.1f} deg"
        )

        print(
            f"free RMSE = "
            f"{best['free_combined']:.3f} deg"
        )

        print(
            f"contact RMSE = "
            f"{best['contact_rmse']:.3f} deg"
        )

        print(
            f"max individual joint error = "
            f"{best['max_joint_error']:.3f} deg"
        )

        print(
            "pose = "
            f"{best['contact_mcp']:.3f} / "
            f"{best['contact_pip']:.3f} / "
            f"{best['contact_dip']:.3f}"
        )


    print()
    print(
        f"CSV:\n{OUTCSV}"
    )

    print("=" * 190)


if __name__ == "__main__":
    main()
