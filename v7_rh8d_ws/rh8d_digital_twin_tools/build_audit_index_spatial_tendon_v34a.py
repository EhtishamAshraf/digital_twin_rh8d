#!/usr/bin/env python3
"""
V34A — RH8D index finger spatial-tendon architecture audit

Purpose
-------
Replace the old constant fixed tendon with a genuine MuJoCo spatial tendon,
while preserving the identified motor/spool coordinate.

This is NOT a contact fit and does NOT use any contact-triggered switching.

The script:
  1) starts from ur5e_rh8d_index_sysid_v5.xml;
  2) creates a moving spool cable site on index_motor_spool;
  3) routes one spatial flexor tendon through the existing RH8D index guide sites;
  4) searches only physically bounded guide-side offsets to obtain the correct
     flexion torque sign and an initial near-open moment-arm scale consistent with
     the previously identified effective transmission;
  5) audits configuration-dependent moment arms at several finger poses;
  6) writes ur5e_rh8d_index_spatial_v34a.xml beside the source MJCF;
  7) performs a FREE-SPACE motor-spool pull with the entire support chain fixed;
  8) automatically opens the native MuJoCo viewer and keeps it open for inspection.

No joint angles are commanded.
No contact-triggered logic is used.
No object-specific parameter is used.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# =============================================================================
# PATHS
# =============================================================================

WS = Path(__file__).resolve().parents[1]

MJCF_DIR = (
    WS
    / "src"
    / "ur5e_mujoco_description"
    / "mjcf"
)

BASE_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_sysid_v5.xml"
)

OUT_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_spatial_v34a.xml"
)

TMP_MODEL = (
    MJCF_DIR
    / ".ur5e_rh8d_index_spatial_v34a_tmp.xml"
)


# =============================================================================
# IDENTIFIED / FROZEN STARTING VALUES
# =============================================================================

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

SPOOL = "index_motor_spool"
SPOOL_BODY = "IndexSpoolDummy"

TENDON = "idx_flex_spatial"
ACTUATOR = "pos_index_tendon"

PALM_ORIGIN = "palm_flexor_origin_idx"
GUIDE_P = "idx_guide_p"
GUIDE_M = "idx_guide_m"
GUIDE_D = "idx_guide_d"
ANCHOR = "idx_anchor"
SPOOL_SITE = "idx_spool_pull_site"

# Previous free-space identified effective moment-arm scale.
# V34A uses this ONLY to initialize a physically routed tendon.
TARGET_ARMS_M = np.array(
    [0.0050, 0.0040, 0.0010],
    dtype=float,
)

TENDON_STIFFNESS = 2100.0
TENDON_DAMPING = 2.0

METERS_PER_TICK = 5.1e-6
ENCODER_ZERO = 100.0

# Structural free-space visualization only.
VIEW_ENCODER = 1000.0
VIEW_SPOOL_TARGET = (
    METERS_PER_TICK
    * (VIEW_ENCODER - ENCODER_ZERO)
)


# =============================================================================
# XML HELPERS
# =============================================================================

def vec3(text: str) -> np.ndarray:
    vals = [float(x) for x in text.split()]
    if len(vals) != 3:
        raise RuntimeError(f"Expected 3-vector, got: {text}")
    return np.asarray(vals, dtype=float)


def fmt_vec(x: np.ndarray) -> str:
    return " ".join(f"{v:.10g}" for v in x)


def find_named(root, tag: str, name: str):
    x = root.find(f".//{tag}[@name='{name}']")
    if x is None:
        raise RuntimeError(
            f"Could not find <{tag} name='{name}'>"
        )
    return x


def body_named(root, name: str):
    return find_named(root, "body", name)


def site_named(root, name: str):
    return find_named(root, "site", name)


def joint_named(root, name: str):
    return find_named(root, "joint", name)


def tendon_parent(root):
    x = root.find(".//tendon")
    if x is None:
        raise RuntimeError("MJCF has no <tendon> block.")
    return x


def make_index_visuals_solid(root):
    for body_name in (
        "Index_Proximal",
        "Index_Middle",
        "Index_Distal",
    ):
        body = body_named(root, body_name)

        for geom in body.iter("geom"):
            rgba = geom.attrib.get("rgba")
            if not rgba:
                continue

            vals = rgba.split()
            if len(vals) != 4:
                continue

            vals[3] = "1"
            geom.set("rgba", " ".join(vals))


def disable_v5_fixture_for_free_audit(root):
    """
    V34A is a transmission audit, not a contact trial.
    If the old calibrated bottle fixture is present, make it non-colliding
    and faint so it cannot influence the result.
    """
    for name in (
        "index_contact_fixture",
        "index_contact_bottle",
    ):
        geom = root.find(f".//geom[@name='{name}']")
        if geom is None:
            continue

        geom.set("contype", "0")
        geom.set("conaffinity", "0")

        raw = geom.attrib.get(
            "rgba",
            "0.2 0.8 0.2 0.15",
        ).split()

        if len(raw) == 4:
            raw[3] = "0.08"
            geom.set("rgba", " ".join(raw))


def replace_index_tendon_with_spatial(root):
    """
    Build the spatial route:

      moving spool site
        -> palm cable lead
        -> proximal guide
        -> middle guide
        -> distal guide
        -> distal anchor

    The moving spool site is attached to index_motor_spool, therefore its
    displacement contributes directly to tendon length.
    """

    spool_body = body_named(
        root,
        SPOOL_BODY,
    )

    # Ensure exactly one moving spool site.
    old = spool_body.find(
        f".//site[@name='{SPOOL_SITE}']"
    )

    if old is None:
        ET.SubElement(
            spool_body,
            "site",
            {
                "name": SPOOL_SITE,
                "pos": "0 0 0",
                "size": "0.0015",
                "rgba": "1 0 1 1",
            },
        )

    # Align hidden slide coordinate with the cable lead.
    #
    # palm_flexor_origin_idx and IndexSpoolDummy are siblings under the
    # palm assembly in the current RH8D model. Positive spool displacement
    # is chosen to move AWAY from the palm origin, increasing cable length.
    palm_site = site_named(
        root,
        PALM_ORIGIN,
    )

    spool_body_pos = vec3(
        spool_body.attrib.get(
            "pos",
            "0 0 0",
        )
    )

    palm_pos = vec3(
        palm_site.attrib["pos"]
    )

    away = (
        spool_body_pos
        -
        palm_pos
    )

    n = np.linalg.norm(away)

    if n < 1e-9:
        raise RuntimeError(
            "Cannot construct spool axis: "
            "spool and palm tendon origin coincide."
        )

    away /= n

    spool_joint = joint_named(
        root,
        SPOOL,
    )

    spool_joint.set(
        "axis",
        fmt_vec(away),
    )

    # Replace either the old <fixed> or any previous <spatial>
    # tendon carrying this name.
    parent = tendon_parent(root)

    for child in list(parent):
        if child.attrib.get("name") == TENDON:
            parent.remove(child)

    spatial = ET.SubElement(
        parent,
        "spatial",
        {
            "name": TENDON,
            "rgba": "1 0 1 1",
            "width": "0.0015",
            # stiffness is set after geometric audit / L0 measurement
            "damping": "0",
            "frictionloss": "0",
        },
    )

    for site in (
        SPOOL_SITE,
        PALM_ORIGIN,
        GUIDE_P,
        GUIDE_M,
        GUIDE_D,
        ANCHOR,
    ):
        ET.SubElement(
            spatial,
            "site",
            {"site": site},
        )

    return spatial, away


def write_tree(tree, path: Path):
    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )


# =============================================================================
# MUJOCO HELPERS
# =============================================================================

def id_joint(model, name):
    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )
    if i < 0:
        raise RuntimeError(
            f"Missing joint in compiled model: {name}"
        )
    return i


def id_site(model, name):
    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        name,
    )
    if i < 0:
        raise RuntimeError(
            f"Missing site in compiled model: {name}"
        )
    return i


def id_tendon(model, name):
    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        name,
    )
    if i < 0:
        raise RuntimeError(
            f"Missing tendon in compiled model: {name}"
        )
    return i


def id_actuator(model, name):
    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )
    if i < 0:
        raise RuntimeError(
            f"Missing actuator in compiled model: {name}"
        )
    return i


def qadr(model, joint_name):
    jid = id_joint(
        model,
        joint_name,
    )
    return int(
        model.jnt_qposadr[jid]
    )


def dadr(model, joint_name):
    jid = id_joint(
        model,
        joint_name,
    )
    return int(
        model.jnt_dofadr[jid]
    )


def tendon_length(
    model,
    data,
    tendon_id,
):
    mujoco.mj_forward(
        model,
        data,
    )
    return float(
        data.ten_length[
            tendon_id
        ]
    )


def set_finger_pose(
    model,
    data,
    q_deg,
    spool_m=0.0,
):
    names = (
        MCP,
        PIP,
        DIP,
    )

    for name, angle_deg in zip(
        names,
        q_deg,
    ):
        data.qpos[
            qadr(model, name)
        ] = math.radians(
            float(angle_deg)
        )

    data.qpos[
        qadr(model, SPOOL)
    ] = float(
        spool_m
    )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


def moment_arms(
    model,
    data,
    tendon_id,
    q_deg,
):
    """
    Return positive flexion moment arms:

        r_i = - dL / dq_i

    because tendon torque is -J_tendon^T * tension.
    """

    set_finger_pose(
        model,
        data,
        q_deg,
        spool_m=0.002,
    )

    eps = 1e-4

    out = []

    for joint_name in (
        MCP,
        PIP,
        DIP,
    ):
        a = qadr(
            model,
            joint_name,
        )

        q0 = float(
            data.qpos[a]
        )

        data.qpos[a] = q0 + eps
        lp = tendon_length(
            model,
            data,
            tendon_id,
        )

        data.qpos[a] = q0 - eps
        lm = tendon_length(
            model,
            data,
            tendon_id,
        )

        data.qpos[a] = q0

        dldq = (
            lp - lm
        ) / (
            2.0 * eps
        )

        out.append(
            -dldq
        )

    # Spool transmission derivative dL/dx.
    a = qadr(
        model,
        SPOOL,
    )

    x0 = float(
        data.qpos[a]
    )

    epsx = 1e-5

    data.qpos[a] = x0 + epsx
    lp = tendon_length(
        model,
        data,
        tendon_id,
    )

    data.qpos[a] = x0 - epsx
    lm = tendon_length(
        model,
        data,
        tendon_id,
    )

    data.qpos[a] = x0

    dldx = (
        lp - lm
    ) / (
        2.0 * epsx
    )

    mujoco.mj_forward(
        model,
        data,
    )

    return (
        np.asarray(
            out,
            dtype=float,
        ),
        float(
            dldx
        ),
    )


# =============================================================================
# ROUTE SEARCH
# =============================================================================

def search_route(
    model,
    data,
):
    """
    Search ONLY the flexor-side distance from the joint axes.

    x/z values, body hierarchy, link lengths, joint axes and distal
    attachment topology remain unchanged.

    This is a bounded initialization search, not a contact fit.
    """

    tendon_id = id_tendon(
        model,
        TENDON,
    )

    ids = {
        GUIDE_P: id_site(
            model,
            GUIDE_P,
        ),
        GUIDE_M: id_site(
            model,
            GUIDE_M,
        ),
        GUIDE_D: id_site(
            model,
            GUIDE_D,
        ),
        ANCHOR: id_site(
            model,
            ANCHOR,
        ),
    }

    # Preserve x/z exactly.
    initial = {
        name:
            model.site_pos[
                sid
            ].copy()
        for name, sid in ids.items()
    }

    # Negative-y is the flexor side for the current index joint convention.
    p_values = np.asarray(
        [-0.0045, -0.0055, -0.0065, -0.0075, -0.0085]
    )

    m_values = np.asarray(
        [-0.0045, -0.0055, -0.0065, -0.0075, -0.0085]
    )

    d_values = np.asarray(
        [-0.0055, -0.0070, -0.0085, -0.0100]
    )

    a_values = np.asarray(
        [-0.0005, -0.0015, -0.0025, -0.0035]
    )

    rows = []

    near_open = (
        5.0,
        5.0,
        2.0,
    )

    for yp, ym, yd, ya in itertools.product(
        p_values,
        m_values,
        d_values,
        a_values,
    ):
        model.site_pos[
            ids[GUIDE_P],
            1,
        ] = yp

        model.site_pos[
            ids[GUIDE_M],
            1,
        ] = ym

        model.site_pos[
            ids[GUIDE_D],
            1,
        ] = yd

        model.site_pos[
            ids[ANCHOR],
            1,
        ] = ya

        r, spool_gain = moment_arms(
            model,
            data,
            tendon_id,
            near_open,
        )

        # Physically unacceptable sign / magnitude.
        if (
            np.any(
                ~np.isfinite(r)
            )
            or
            np.any(
                r <= 0.0
            )
            or
            np.any(
                r > 0.020
            )
            or
            not np.isfinite(
                spool_gain
            )
        ):
            continue

        # Scale-normalized fit to the prior free-space effective arms.
        denom = np.asarray(
            [0.0025, 0.0020, 0.0015]
        )

        arm_score = float(
            np.sum(
                (
                    (
                        r
                        -
                        TARGET_ARMS_M
                    )
                    /
                    denom
                )
                ** 2
            )
        )

        # Prefer a one-to-one hidden spool pull.
        spool_score = (
            25.0
            *
            (
                spool_gain
                -
                1.0
            )
            ** 2
        )

        # Weak geometry prior around the original RH8D guide magnitudes.
        prior = np.asarray(
            [0.0065, 0.0065, 0.0085, 0.0015]
        )

        actual = np.abs(
            np.asarray(
                [yp, ym, yd, ya]
            )
        )

        prior_score = (
            0.10
            *
            float(
                np.sum(
                    (
                        (
                            actual
                            -
                            prior
                        )
                        /
                        0.003
                    )
                    ** 2
                )
            )
        )

        score = (
            arm_score
            +
            spool_score
            +
            prior_score
        )

        rows.append(
            {
                "score": score,
                "yp": yp,
                "ym": ym,
                "yd": yd,
                "ya": ya,
                "r": r.copy(),
                "spool_gain": spool_gain,
            }
        )

    if not rows:
        raise RuntimeError(
            "No physically valid spatial-tendon route "
            "was found in the bounded V34A search."
        )

    rows.sort(
        key=lambda x: x["score"]
    )

    print()
    print("=" * 110)
    print(
        "V34A BOUNDED SPATIAL-ROUTE INITIALIZATION"
    )
    print("=" * 110)

    print()
    print(
        "Target near-open effective scale "
        "(initialization only):"
    )

    print(
        "  MCP/PIP/DIP = "
        f"{TARGET_ARMS_M[0]*1000:.3f} / "
        f"{TARGET_ARMS_M[1]*1000:.3f} / "
        f"{TARGET_ARMS_M[2]*1000:.3f} mm"
    )

    print()
    print(
        "Top 10 physically valid routes:"
    )

    for i, row in enumerate(
        rows[:10],
        start=1,
    ):
        r = row["r"] * 1000.0

        print(
            f" {i:02d}. "
            f"score={row['score']:.4f}  "
            f"y[p/m/d/a]="
            f"{row['yp']*1000:+.2f}/"
            f"{row['ym']*1000:+.2f}/"
            f"{row['yd']*1000:+.2f}/"
            f"{row['ya']*1000:+.2f} mm  "
            f"arms={r[0]:.3f}/"
            f"{r[1]:.3f}/"
            f"{r[2]:.3f} mm  "
            f"dL/dx={row['spool_gain']:.5f}"
        )

    best = rows[0]

    # Install selected route in the compiled audit model.
    model.site_pos[
        ids[GUIDE_P],
        1,
    ] = best["yp"]

    model.site_pos[
        ids[GUIDE_M],
        1,
    ] = best["ym"]

    model.site_pos[
        ids[GUIDE_D],
        1,
    ] = best["yd"]

    model.site_pos[
        ids[ANCHOR],
        1,
    ] = best["ya"]

    mujoco.mj_forward(
        model,
        data,
    )

    return best, initial


# =============================================================================
# FIX SUPPORT FOR STRUCTURAL VIEW
# =============================================================================

def create_fixed_state(
    model,
    data,
):
    """
    Freeze every joint except:
      index spool
      index MCP/PIP/DIP

    This is the correct boundary condition for the hardware contact setup,
    and is also useful here to prevent unrelated UR/hand motion.
    """

    free_names = {
        SPOOL,
        MCP,
        PIP,
        DIP,
    }

    frozen = []

    for jid in range(
        model.njnt
    ):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )

        if name in free_names:
            continue

        jtype = int(
            model.jnt_type[
                jid
            ]
        )

        if jtype not in (
            int(
                mujoco.mjtJoint.mjJNT_HINGE
            ),
            int(
                mujoco.mjtJoint.mjJNT_SLIDE
            ),
        ):
            continue

        qa = int(
            model.jnt_qposadr[
                jid
            ]
        )

        da = int(
            model.jnt_dofadr[
                jid
            ]
        )

        frozen.append(
            (
                qa,
                da,
                float(
                    data.qpos[
                        qa
                    ]
                ),
            )
        )

    return frozen


def enforce_fixed_state(
    model,
    data,
    frozen,
):
    for qa, da, q0 in frozen:
        data.qpos[
            qa
        ] = q0

        data.qvel[
            da
        ] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 110)
    print(
        "V34A — PHYSICAL SPATIAL TENDON ARCHITECTURE AUDIT"
    )
    print("=" * 110)

    print()
    print(
        "This experiment removes the old constant index tendon "
        "and uses one spatial flexor route."
    )

    print(
        "No contact-triggered moment-arm switching."
    )

    print(
        "No PIP breakaway logic."
    )

    print(
        "No object-specific parameters."
    )

    if not BASE_MODEL.exists():
        raise RuntimeError(
            "Base V5 model not found:\n"
            f"{BASE_MODEL}"
        )

    tree = ET.parse(
        BASE_MODEL
    )

    root = tree.getroot()

    make_index_visuals_solid(
        root
    )

    disable_v5_fixture_for_free_audit(
        root
    )

    spatial_xml, spool_axis = (
        replace_index_tendon_with_spatial(
            root
        )
    )

    # Ensure initial guide side is flexor side before compile.
    site_named(
        root,
        GUIDE_P,
    ).set(
        "pos",
        "-0.010 -0.0065 0.0120",
    )

    site_named(
        root,
        GUIDE_M,
    ).set(
        "pos",
        "-0.010 -0.0065 0.0100",
    )

    site_named(
        root,
        GUIDE_D,
    ).set(
        "pos",
        "-0.010 -0.0085 0.0080",
    )

    site_named(
        root,
        ANCHOR,
    ).set(
        "pos",
        "-0.0055 -0.0015 -0.012",
    )

    write_tree(
        tree,
        TMP_MODEL,
    )

    try:
        model = mujoco.MjModel.from_xml_path(
            str(
                TMP_MODEL
            )
        )

        data = mujoco.MjData(
            model
        )

        best, _ = search_route(
            model,
            data,
        )

        # Put selected site coordinates back into XML.
        selected_y = {
            GUIDE_P: best["yp"],
            GUIDE_M: best["ym"],
            GUIDE_D: best["yd"],
            ANCHOR: best["ya"],
        }

        for name, y in selected_y.items():
            site = site_named(
                root,
                name,
            )

            pos = vec3(
                site.attrib["pos"]
            )

            pos[1] = y

            site.set(
                "pos",
                fmt_vec(
                    pos
                ),
            )

        # Compile selected zero-stiffness geometry to measure natural L0.
        write_tree(
            tree,
            TMP_MODEL,
        )

        model0 = mujoco.MjModel.from_xml_path(
            str(
                TMP_MODEL
            )
        )

        data0 = mujoco.MjData(
            model0
        )

        tid0 = id_tendon(
            model0,
            TENDON,
        )

        set_finger_pose(
            model0,
            data0,
            (0.0, 0.0, 0.0),
            spool_m=0.0,
        )

        L0 = tendon_length(
            model0,
            data0,
            tid0,
        )

        # Final passive elastic spatial tendon.
        spatial_xml.set(
            "stiffness",
            f"{TENDON_STIFFNESS:.10g}",
        )

        spatial_xml.set(
            "damping",
            f"{TENDON_DAMPING:.10g}",
        )

        spatial_xml.set(
            "springlength",
            f"{L0:.10g}",
        )

        spatial_xml.set(
            "frictionloss",
            "0",
        )

        write_tree(
            tree,
            OUT_MODEL,
        )

        print()
        print("=" * 110)
        print(
            "V34A SELECTED PHYSICAL ROUTE"
        )
        print("=" * 110)

        print()
        print(
            "Hidden spool axis = "
            f"[{spool_axis[0]:+.6f}, "
            f"{spool_axis[1]:+.6f}, "
            f"{spool_axis[2]:+.6f}]"
        )

        print(
            "Selected local flexor-side y offsets:"
        )

        print(
            f"  proximal guide = "
            f"{best['yp']*1000:+.3f} mm"
        )

        print(
            f"  middle guide   = "
            f"{best['ym']*1000:+.3f} mm"
        )

        print(
            f"  distal guide   = "
            f"{best['yd']*1000:+.3f} mm"
        )

        print(
            f"  distal anchor  = "
            f"{best['ya']*1000:+.3f} mm"
        )

        print(
            f"Open spatial tendon length L0 = "
            f"{L0*1000:.3f} mm"
        )

        print(
            f"Tendon stiffness = "
            f"{TENDON_STIFFNESS:.1f} N/m"
        )

        print(
            f"Tendon damping = "
            f"{TENDON_DAMPING:.2f}"
        )

        print()
        print(
            f"Saved model:\n  {OUT_MODEL}"
        )

        # ---------------------------------------------------------------------
        # Final compile + moment-arm audit
        # ---------------------------------------------------------------------

        model = mujoco.MjModel.from_xml_path(
            str(
                OUT_MODEL
            )
        )

        data = mujoco.MjData(
            model
        )

        tendon_id = id_tendon(
            model,
            TENDON,
        )

        poses = [
            (5.0, 5.0, 2.0),
            (30.0, 10.0, 5.0),
            (45.0, 25.0, 10.0),
            (60.0, 50.0, 20.0),
        ]

        print()
        print("=" * 110)
        print(
            "CONFIGURATION-DEPENDENT MOMENT-ARM AUDIT"
        )
        print("=" * 110)

        all_positive = True

        for pose in poses:
            r, gain = moment_arms(
                model,
                data,
                tendon_id,
                pose,
            )

            all_positive = (
                all_positive
                and
                bool(
                    np.all(
                        r > 0.0
                    )
                )
            )

            print(
                " q="
                f"{pose[0]:5.1f}/"
                f"{pose[1]:5.1f}/"
                f"{pose[2]:5.1f} deg"
                "  arms="
                f"{r[0]*1000:7.3f}/"
                f"{r[1]*1000:7.3f}/"
                f"{r[2]*1000:7.3f} mm"
                f"  dL/dx={gain:.5f}"
            )

        if not all_positive:
            print()
            print(
                "FAIL: at least one spatial-tendon moment arm "
                "changes to the extension sign."
            )

            print(
                "Do NOT proceed to contact fitting."
            )

            return

        print()
        print(
            "PASS: all audited tendon moment arms create "
            "positive flexion torque."
        )

        # ---------------------------------------------------------------------
        # Structural free-space pull + GUI
        # ---------------------------------------------------------------------

        data = mujoco.MjData(
            model
        )

        frozen = create_fixed_state(
            model,
            data,
        )

        actuator_id = id_actuator(
            model,
            ACTUATOR,
        )

        print()
        print("=" * 110)
        print(
            "V34A NATIVE MUJOCO VISUALIZATION"
        )
        print("=" * 110)

        print()
        print(
            "The magenta line is now the ACTIVE spatial tendon."
        )

        print(
            "The bottle fixture is non-colliding for this "
            "free-space architecture audit."
        )

        print(
            f"Spool will ramp from 0 to "
            f"{VIEW_SPOOL_TARGET*1000:.3f} mm "
            f"(hardware-equivalent encoder {VIEW_ENCODER:.0f})."
        )

        print()
        print(
            "CHECK VISUALLY:"
        )

        print(
            "  1. index flexes in the hardware closing direction;"
        )

        print(
            "  2. magenta tendon remains on the flexor side;"
        )

        print(
            "  3. tendon passes smoothly from palm -> proximal "
            "-> middle -> distal;"
        )

        print(
            "  4. no artificial joint-angle command is present."
        )

        viewer = mujoco.viewer.launch_passive(
            model,
            data,
        )

        try:
            # Aim camera around the index middle link without changing
            # native viewer dimensions.
            bid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                "Index_Middle",
            )

            if bid >= 0:
                mujoco.mj_forward(
                    model,
                    data,
                )

                viewer.cam.lookat[:] = (
                    data.xpos[
                        bid
                    ]
                )

                viewer.cam.distance = 0.22

            total = 3.0

            steps = max(
                1,
                int(
                    total
                    /
                    model.opt.timestep
                ),
            )

            for k in range(
                steps
            ):
                if not viewer.is_running():
                    break

                t = (
                    k
                    *
                    model.opt.timestep
                )

                # Smooth 2 s ramp, then hold.
                u = min(
                    1.0,
                    t / 2.0,
                )

                target = (
                    VIEW_SPOOL_TARGET
                    *
                    (
                        3*u*u
                        -
                        2*u*u*u
                    )
                )

                data.ctrl[
                    actuator_id
                ] = target

                mujoco.mj_step(
                    model,
                    data,
                )

                enforce_fixed_state(
                    model,
                    data,
                    frozen,
                )

                viewer.sync()

                time.sleep(
                    model.opt.timestep
                )

            q_final = np.degrees(
                [
                    data.qpos[
                        qadr(
                            model,
                            MCP,
                        )
                    ],
                    data.qpos[
                        qadr(
                            model,
                            PIP,
                        )
                    ],
                    data.qpos[
                        qadr(
                            model,
                            DIP,
                        )
                    ],
                ]
            )

            print()
            print(
                "Free-space structural final pose "
                "(NOT yet a fitted V34 result):"
            )

            print(
                "  MCP/PIP/DIP = "
                f"{q_final[0]:.3f} / "
                f"{q_final[1]:.3f} / "
                f"{q_final[2]:.3f} deg"
            )

            print()
            print(
                "Viewer remains open for your inspection / "
                "screen recording."
            )

            print(
                "Close the MuJoCo window to finish V34A."
            )

            while viewer.is_running():
                viewer.sync()
                time.sleep(
                    0.05
                )

        finally:
            viewer.close()

        print()
        print("=" * 110)
        print(
            "V34A COMPLETE"
        )
        print("=" * 110)

        print()
        print(
            "Next gate:"
        )

        print(
            "If the spatial tendon visibly routes correctly and "
            "the moment-arm audit stays positive, V34B will fit "
            "this SAME architecture to the trusted free-space "
            "hardware trajectory before any contact tuning."
        )

    finally:
        try:
            TMP_MODEL.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
