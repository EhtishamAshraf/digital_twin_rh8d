#!/usr/bin/env python3

from pathlib import Path
import copy
import math
import random
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


# ============================================================
# RH8D INDEX — CONFIGURATION-DEPENDENT TENDON GEOMETRY V1
#
# PURPOSE
#
# Build the first geometry-based spatial flexor tendon after
# the constant-transmission family failed to explain both:
#
#   - dynamic free-space behaviour
#   - contact-induced PIP redistribution
#
# IMPORTANT:
#
# This script does NOT yet fit the dynamic model.
# It only searches for a physically sensible tendon geometry
# whose moment arms:
#
#   1. have the correct flexion sign,
#   2. resemble [5,4,1] mm near open,
#   3. naturally increase PIP leverage as the finger flexes,
#   4. keep MCP/DIP leverage within reasonable bounds.
#
# No force/contact result is used as a hard-coded switch.
# Configuration dependence emerges only from tendon geometry.
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
    / "ur5e_rh8d_index_tabletop_sysid_v1.xml"
)


OUTPUT = (
    MJCF_DIR
    / "ur5e_rh8d_index_configdep_geom_v1.xml"
)


RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "index_configdep_v1"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CSV_OUT = (
    RESULT_DIR
    / "moment_arm_candidates_v1.csv"
)


# ============================================================
# Joint / tendon names
# ============================================================

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

TENDON = "idx_flex_spatial"


# ============================================================
# Existing RH8D site coordinates from the original model.
#
# These are only our geometric starting reference.
# They are NOT accepted as identified physical truth.
# ============================================================

PALM_ORIGIN = np.array(
    [-0.0520, -0.0090, 0.0685]
)

GUIDE_P_Z = 0.0120
GUIDE_M_Z = 0.0100
GUIDE_D_Z = 0.0080
ANCHOR_Z  = -0.0120


# Existing lateral x locations.
GUIDE_X = -0.0100
ANCHOR_X = -0.0055


# ============================================================
# Data-informed but NOT final moment-arm design targets.
#
# Tension torque:
#
#     tau = -T * dL/dq
#
# Positive q is flexion in this model, so dL/dq must be
# NEGATIVE for the tendon to generate positive flexion torque.
#
# Near-open [5,4,1] mm comes from the best validated constant
# free-space family.
#
# The flexed/contact target asks for increased PIP leverage,
# motivated by the observed hardware redistribution and by the
# failed constant-ratio scans.
#
# These are routing-design seeds, not final identified values.
# ============================================================

TARGET_OPEN_MM = np.array(
    [-5.0, -4.0, -1.0]
)

TARGET_CONTACT_MM = np.array(
    [-5.0, -7.0, -1.5]
)


# Hardware loaded pose measured from video.
Q_CONTACT_DEG = np.array(
    [14.5355, 23.8110, 9.3270]
)


# Generic intermediate configuration used only to check that
# the moment-arm curve changes smoothly.
Q_MID_DEG = np.array(
    [30.0, 15.0, 5.0]
)


# More-flexed diagnostic configuration.
Q_FLEX_DEG = np.array(
    [50.0, 45.0, 15.0]
)


RNG_SEED = 17

# Random geometric candidates.
N_SAMPLES = 12000


# Search bounds for the y-offset of each tendon guide.
#
# Joint axes are approximately along local x, therefore the
# y/z offsets determine the effective lever arms.
#
# +/- 12 mm keeps the search around the finger volume rather
# than allowing absurd remote pulley locations.
Y_MIN = -0.012
Y_MAX =  0.012


# ============================================================
# XML helpers
# ============================================================

def find_body(
    root,
    name,
):

    for body in root.iter("body"):

        if body.attrib.get("name") == name:
            return body

    raise RuntimeError(
        f"Body not found: {name}"
    )


def ensure_site(
    body,
    name,
    pos,
    rgba,
):

    site = None

    for s in body.findall("site"):

        if s.attrib.get("name") == name:
            site = s
            break


    if site is None:

        site = ET.SubElement(
            body,
            "site",
        )


    site.set(
        "name",
        name,
    )

    site.set(
        "pos",
        " ".join(
            f"{x:.9g}"
            for x in pos
        ),
    )

    site.set(
        "size",
        "0.0018",
    )

    site.set(
        "rgba",
        rgba,
    )


    return site


def prepare_spatial_xml():

    if not SOURCE.exists():

        raise RuntimeError(
            f"Source model not found:\n{SOURCE}"
        )


    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()


    palm = find_body(
        root,
        "palm_assembly:1",
    )

    prox = find_body(
        root,
        "Index_Proximal",
    )

    middle = find_body(
        root,
        "Index_Middle",
    )

    distal = find_body(
        root,
        "Index_Distal",
    )


    # --------------------------------------------------------
    # Add/overwrite routing sites.
    #
    # y coordinates are placeholders here. They will be
    # modified in memory during the search.
    # --------------------------------------------------------

    ensure_site(
        palm,
        "cfg_idx_origin",
        PALM_ORIGIN,
        "0 1 1 1",
    )

    ensure_site(
        prox,
        "cfg_idx_guide_p",
        [GUIDE_X, -0.006, GUIDE_P_Z],
        "1 1 0 1",
    )

    ensure_site(
        middle,
        "cfg_idx_guide_m",
        [GUIDE_X, -0.006, GUIDE_M_Z],
        "1 0.5 0 1",
    )

    ensure_site(
        distal,
        "cfg_idx_guide_d",
        [GUIDE_X, -0.006, GUIDE_D_Z],
        "1 0 1 1",
    )

    ensure_site(
        distal,
        "cfg_idx_anchor",
        [ANCHOR_X, -0.004, ANCHOR_Z],
        "0 1 0 1",
    )


    # --------------------------------------------------------
    # Replace only the INDEX fixed tendon with a spatial
    # tendon. Other tendons are untouched.
    # --------------------------------------------------------

    tendon_parent = root.find("tendon")

    if tendon_parent is None:

        raise RuntimeError(
            "<tendon> section not found."
        )


    old = None

    old_index = None


    for i, element in enumerate(
        list(tendon_parent)
    ):

        if (
            element.attrib.get("name")
            == TENDON
        ):

            old = element
            old_index = i
            break


    if old is None:

        raise RuntimeError(
            f"Tendon not found: {TENDON}"
        )


    tendon_parent.remove(
        old
    )


    spatial = ET.Element(
        "spatial",
        {
            "name": TENDON,
            "width": "0.0015",
            "rgba": "1 0 1 1",
        },
    )


    for site_name in [
        "cfg_idx_origin",
        "cfg_idx_guide_p",
        "cfg_idx_guide_m",
        "cfg_idx_guide_d",
        "cfg_idx_anchor",
    ]:

        ET.SubElement(
            spatial,
            "site",
            {
                "site": site_name
            },
        )


    tendon_parent.insert(
        old_index,
        spatial,
    )


    tree.write(
        OUTPUT,
        encoding="unicode",
        xml_declaration=True,
    )


# ============================================================
# MuJoCo helpers
# ============================================================

def joint_qpos_address(
    model,
    name,
):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:

        raise RuntimeError(
            f"Joint missing: {name}"
        )


    return int(
        model.jnt_qposadr[jid]
    )


def site_id(
    model,
    name,
):

    sid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        name,
    )

    if sid < 0:

        raise RuntimeError(
            f"Site missing: {name}"
        )

    return sid


def tendon_id(
    model,
    name,
):

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        name,
    )

    if tid < 0:

        raise RuntimeError(
            f"Tendon missing: {name}"
        )

    return tid


# ============================================================
# Moment-arm computation
# ============================================================

def configure_q(
    model,
    data,
    qpos0,
    addresses,
    q_deg,
):

    data.qpos[:] = qpos0


    for adr, deg in zip(
        addresses,
        q_deg,
    ):

        data.qpos[adr] = math.radians(
            float(deg)
        )


    mujoco.mj_forward(
        model,
        data,
    )


def tendon_length(
    model,
    data,
    tid,
):

    mujoco.mj_forward(
        model,
        data,
    )

    return float(
        data.ten_length[tid]
    )


def moment_arms_mm(
    model,
    data,
    qpos0,
    addresses,
    tid,
    q_deg,
):

    q_deg = np.asarray(
        q_deg,
        dtype=float,
    )


    configure_q(
        model,
        data,
        qpos0,
        addresses,
        q_deg,
    )


    base_qpos = data.qpos.copy()

    arms = []


    eps = 1.0e-5


    for adr in addresses:

        q_here = float(
            base_qpos[adr]
        )


        # At the zero boundary use a forward derivative.
        if abs(q_here) < 2.0 * eps:

            data.qpos[:] = base_qpos

            mujoco.mj_forward(
                model,
                data,
            )

            L0 = float(
                data.ten_length[tid]
            )


            data.qpos[adr] = (
                q_here
                + eps
            )

            mujoco.mj_forward(
                model,
                data,
            )

            L1 = float(
                data.ten_length[tid]
            )


            deriv = (
                L1 - L0
            ) / eps


        else:

            data.qpos[:] = base_qpos

            data.qpos[adr] = (
                q_here
                + eps
            )

            mujoco.mj_forward(
                model,
                data,
            )

            Lp = float(
                data.ten_length[tid]
            )


            data.qpos[:] = base_qpos

            data.qpos[adr] = (
                q_here
                - eps
            )

            mujoco.mj_forward(
                model,
                data,
            )

            Lm = float(
                data.ten_length[tid]
            )


            deriv = (
                Lp - Lm
            ) / (
                2.0 * eps
            )


        arms.append(
            deriv * 1000.0
        )


    data.qpos[:] = base_qpos

    mujoco.mj_forward(
        model,
        data,
    )


    return np.array(
        arms,
        dtype=float,
    )


# ============================================================
# Candidate score
# ============================================================

def candidate_score(
    open_arm,
    mid_arm,
    contact_arm,
    flex_arm,
):

    # Correct flexion sign throughout.
    all_arms = np.vstack(
        [
            open_arm,
            mid_arm,
            contact_arm,
            flex_arm,
        ]
    )


    sign_penalty = (
        np.maximum(
            all_arms,
            0.0,
        ) ** 2
    ).sum() * 300.0


    # Near-open behaviour should remain close to the
    # validated constant model.
    open_error = np.sum(
        (
            open_arm
            - TARGET_OPEN_MM
        ) ** 2
    )


    # Hardware contact suggests PIP leverage should increase,
    # without destroying MCP and DIP.
    contact_error = np.sum(
        (
            contact_arm
            - TARGET_CONTACT_MM
        ) ** 2
    )


    # Encourage PIP leverage magnitude to grow with flexion.
    p_open = abs(
        open_arm[1]
    )

    p_contact = abs(
        contact_arm[1]
    )


    desired_ratio = 1.55


    ratio = (
        p_contact
        / max(
            p_open,
            0.25,
        )
    )


    ratio_penalty = (
        max(
            0.0,
            desired_ratio
            - ratio,
        )
        ** 2
        * 35.0
    )


    # MCP should not collapse during flexion.
    mcp_change_penalty = (
        max(
            0.0,
            abs(
                contact_arm[0]
                - open_arm[0]
            )
            - 1.5,
        )
        ** 2
        * 5.0
    )


    # DIP should not become a huge lever arm.
    dip_penalty = (
        max(
            0.0,
            abs(
                contact_arm[2]
            )
            - 3.0,
        )
        ** 2
        * 12.0
    )


    # Avoid extreme routing behaviour at the flexed state.
    extreme_penalty = (
        np.maximum(
            np.abs(
                flex_arm
            )
            - 12.0,
            0.0,
        )
        ** 2
    ).sum() * 20.0


    return (
        1.8 * open_error
        +
        contact_error
        +
        ratio_penalty
        +
        mcp_change_penalty
        +
        dip_penalty
        +
        sign_penalty
        +
        extreme_penalty
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 112)
    print(
        "RH8D INDEX — CONFIGURATION-DEPENDENT "
        "SPATIAL TENDON GEOMETRY SEARCH V1"
    )
    print("=" * 112)


    prepare_spatial_xml()


    print()
    print(
        "Temporary geometry model:"
    )

    print(
        OUTPUT
    )


    # Compile once.
    model = mujoco.MjModel.from_xml_path(
        str(
            OUTPUT
        )
    )

    data = mujoco.MjData(
        model
    )


    qpos0 = model.qpos0.copy()


    addresses = [
        joint_qpos_address(
            model,
            MCP,
        ),
        joint_qpos_address(
            model,
            PIP,
        ),
        joint_qpos_address(
            model,
            DIP,
        ),
    ]


    tid = tendon_id(
        model,
        TENDON,
    )


    sid_p = site_id(
        model,
        "cfg_idx_guide_p",
    )

    sid_m = site_id(
        model,
        "cfg_idx_guide_m",
    )

    sid_d = site_id(
        model,
        "cfg_idx_guide_d",
    )

    sid_a = site_id(
        model,
        "cfg_idx_anchor",
    )


    print()
    print(
        "Searching effective guide geometry..."
    )

    print(
        f"Candidates = {N_SAMPLES}"
    )


    rng = random.Random(
        RNG_SEED
    )


    best = []


    for i in range(
        N_SAMPLES
    ):

        yp = rng.uniform(
            Y_MIN,
            Y_MAX,
        )

        ym = rng.uniform(
            Y_MIN,
            Y_MAX,
        )

        yd = rng.uniform(
            Y_MIN,
            Y_MAX,
        )

        ya = rng.uniform(
            Y_MIN,
            Y_MAX,
        )


        model.site_pos[
            sid_p,
            1
        ] = yp

        model.site_pos[
            sid_m,
            1
        ] = ym

        model.site_pos[
            sid_d,
            1
        ] = yd

        model.site_pos[
            sid_a,
            1
        ] = ya


        open_arm = moment_arms_mm(
            model,
            data,
            qpos0,
            addresses,
            tid,
            [0.0, 0.0, 0.0],
        )


        mid_arm = moment_arms_mm(
            model,
            data,
            qpos0,
            addresses,
            tid,
            Q_MID_DEG,
        )


        contact_arm = moment_arms_mm(
            model,
            data,
            qpos0,
            addresses,
            tid,
            Q_CONTACT_DEG,
        )


        flex_arm = moment_arms_mm(
            model,
            data,
            qpos0,
            addresses,
            tid,
            Q_FLEX_DEG,
        )


        score = candidate_score(
            open_arm,
            mid_arm,
            contact_arm,
            flex_arm,
        )


        candidate = (
            score,
            yp,
            ym,
            yd,
            ya,
            open_arm.copy(),
            mid_arm.copy(),
            contact_arm.copy(),
            flex_arm.copy(),
        )


        best.append(
            candidate
        )


    best.sort(
        key=lambda x:
            x[0]
    )


    best = best[:25]


    # ========================================================
    # Save search results
    # ========================================================

    import csv


    with open(
        CSV_OUT,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )


        writer.writerow(
            [
                "rank",
                "score",
                "guide_p_y_mm",
                "guide_m_y_mm",
                "guide_d_y_mm",
                "anchor_y_mm",
                "open_mcp_mm",
                "open_pip_mm",
                "open_dip_mm",
                "mid_mcp_mm",
                "mid_pip_mm",
                "mid_dip_mm",
                "contact_mcp_mm",
                "contact_pip_mm",
                "contact_dip_mm",
                "flex_mcp_mm",
                "flex_pip_mm",
                "flex_dip_mm",
            ]
        )


        for rank, c in enumerate(
            best,
            start=1,
        ):

            (
                score,
                yp,
                ym,
                yd,
                ya,
                oa,
                ma,
                ca,
                fa,
            ) = c


            writer.writerow(
                [
                    rank,
                    score,
                    yp * 1000.0,
                    ym * 1000.0,
                    yd * 1000.0,
                    ya * 1000.0,
                    *oa,
                    *ma,
                    *ca,
                    *fa,
                ]
            )


    # ========================================================
    # Write best routing back into XML
    # ========================================================

    (
        score,
        yp,
        ym,
        yd,
        ya,
        open_arm,
        mid_arm,
        contact_arm,
        flex_arm,
    ) = best[0]


    tree = ET.parse(
        OUTPUT
    )

    root = tree.getroot()


    site_values = {
        "cfg_idx_guide_p":
            [
                GUIDE_X,
                yp,
                GUIDE_P_Z,
            ],

        "cfg_idx_guide_m":
            [
                GUIDE_X,
                ym,
                GUIDE_M_Z,
            ],

        "cfg_idx_guide_d":
            [
                GUIDE_X,
                yd,
                GUIDE_D_Z,
            ],

        "cfg_idx_anchor":
            [
                ANCHOR_X,
                ya,
                ANCHOR_Z,
            ],
    }


    for site in root.iter("site"):

        name = site.attrib.get(
            "name"
        )


        if name in site_values:

            site.set(
                "pos",
                " ".join(
                    f"{x:.9g}"
                    for x in site_values[
                        name
                    ]
                ),
            )


    tree.write(
        OUTPUT,
        encoding="unicode",
        xml_declaration=True,
    )


    # ========================================================
    # Recompile final selected route and verify.
    # ========================================================

    model_final = (
        mujoco.MjModel.from_xml_path(
            str(
                OUTPUT
            )
        )
    )

    data_final = mujoco.MjData(
        model_final
    )

    q0_final = (
        model_final.qpos0.copy()
    )


    addresses_final = [
        joint_qpos_address(
            model_final,
            MCP,
        ),
        joint_qpos_address(
            model_final,
            PIP,
        ),
        joint_qpos_address(
            model_final,
            DIP,
        ),
    ]


    tid_final = tendon_id(
        model_final,
        TENDON,
    )


    states = [
        (
            "OPEN",
            [0.0, 0.0, 0.0],
        ),
        (
            "MID",
            Q_MID_DEG,
        ),
        (
            "HW_CONTACT",
            Q_CONTACT_DEG,
        ),
        (
            "FLEXED",
            Q_FLEX_DEG,
        ),
    ]


    print()
    print("=" * 112)
    print(
        "BEST CONFIGURATION-DEPENDENT ROUTING V1"
    )
    print("=" * 112)


    print()
    print(
        "Guide y offsets:"
    )

    print(
        f"  proximal = "
        f"{yp*1000.0:+.3f} mm"
    )

    print(
        f"  middle   = "
        f"{ym*1000.0:+.3f} mm"
    )

    print(
        f"  distal   = "
        f"{yd*1000.0:+.3f} mm"
    )

    print(
        f"  anchor   = "
        f"{ya*1000.0:+.3f} mm"
    )


    print()
    print(
        "Reminder: negative dL/dq gives positive "
        "flexion torque for this model."
    )


    print()
    print(
        f"{'STATE':<15}"
        f"{'MCP mm/rad':>14}"
        f"{'PIP mm/rad':>14}"
        f"{'DIP mm/rad':>14}"
        f"{'|PIP|/open':>14}"
    )

    print(
        "-" * 72
    )


    open_pip_abs = None


    final_results = {}


    for label, q_deg in states:

        arms = moment_arms_mm(
            model_final,
            data_final,
            q0_final,
            addresses_final,
            tid_final,
            q_deg,
        )


        final_results[
            label
        ] = arms


        if label == "OPEN":

            open_pip_abs = abs(
                arms[1]
            )


        ratio = (
            abs(
                arms[1]
            )
            /
            max(
                open_pip_abs,
                1.0e-9,
            )
        )


        print(
            f"{label:<15}"
            f"{arms[0]:>14.3f}"
            f"{arms[1]:>14.3f}"
            f"{arms[2]:>14.3f}"
            f"{ratio:>14.3f}"
        )


    # ========================================================
    # Basic structural checks
    # ========================================================

    all_final = np.vstack(
        list(
            final_results.values()
        )
    )


    correct_sign = bool(
        np.all(
            all_final < 0.0
        )
    )


    open_ok = bool(
        (
            3.0
            <= abs(
                final_results[
                    "OPEN"
                ][0]
            )
            <= 7.0
        )
        and
        (
            2.5
            <= abs(
                final_results[
                    "OPEN"
                ][1]
            )
            <= 5.5
        )
        and
        (
            0.3
            <= abs(
                final_results[
                    "OPEN"
                ][2]
            )
            <= 3.0
        )
    )


    pip_growth = (
        abs(
            final_results[
                "HW_CONTACT"
            ][1]
        )
        /
        max(
            abs(
                final_results[
                    "OPEN"
                ][1]
            ),
            1.0e-9,
        )
    )


    growth_ok = bool(
        pip_growth >= 1.35
    )


    print()
    print("=" * 112)
    print(
        "STRUCTURAL CHECK"
    )
    print("=" * 112)

    print()
    print(
        f"Correct flexion sign everywhere : "
        f"{correct_sign}"
    )

    print(
        f"Open moment arms reasonable     : "
        f"{open_ok}"
    )

    print(
        f"PIP configuration growth        : "
        f"{pip_growth:.3f} x"
    )

    print(
        f"PIP growth >= 1.35              : "
        f"{growth_ok}"
    )


    print()
    print(
        "Selected geometry XML:"
    )

    print(
        OUTPUT
    )


    print()
    print(
        "Candidate table:"
    )

    print(
        CSV_OUT
    )


    print()
    print("=" * 112)


    if (
        correct_sign
        and
        open_ok
        and
        growth_ok
    ):

        print(
            "RESULT: PASS — geometry is suitable for the "
            "next dynamic/contact replay stage."
        )

    else:

        print(
            "RESULT: DO NOT FIT YET — routing geometry "
            "needs another search."
        )


    print()
    print("=" * 112)


if __name__ == "__main__":
    main()
