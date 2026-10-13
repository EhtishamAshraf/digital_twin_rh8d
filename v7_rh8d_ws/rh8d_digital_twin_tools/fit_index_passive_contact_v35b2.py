#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import test_index_cad_spatial_contact_v35a as v35
import test_index_middle_mesh_collision_v28 as v28
import scan_index_fixed_support_v30 as v30


# ============================================================
# V35B
#
# CAD tendon is FROZEN.
# V33 environment is FROZEN.
# Cylinder is FROZEN.
# Contact parameters are FROZEN.
#
# ONLY passive joint mechanics are varied.
#
# This tests the hypothesis indicated strongly by V35A:
#
#   PIP is too resistant
#   DIP is too compliant
#
# No contact-triggered switching.
# No artificial moment arms.
# No prescribed joint angles.
# ============================================================


WS = v35.WS
MJCF_DIR = v35.MJCF_DIR

BASE_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_cad_spatial_v35b_base.xml"
)

CANDIDATE_MODEL = (
    MJCF_DIR
    / ".ur5e_rh8d_index_cad_spatial_v35b_candidate.xml"
)

SELECTED_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_cad_spatial_v35b2_selected.xml"
)


# Hardware H1 reference.
HW = np.array(
    [14.5355, 23.8110, 9.3270],
    dtype=float,
)


# ============================================================
# COARSE PHYSICAL SEARCH
#
# Current approximate V33 values:
#
# MCP stiffness ~0.00725
# PIP stiffness ~0.031
# DIP stiffness ~0.020
# PIP friction ~0.006
#
# V35A strongly suggests lowering PIP resistance and increasing
# DIP resistance.
# ============================================================

MCP_K = [
    0.0045,
    0.0060,
    0.0075,
]

PIP_K = [
    0.0015,
    0.0025,
    0.0035,
    0.0045,
]

PIP_FRICTION = [
    0.00075,
    0.00125,
    0.00175,
    0.00225,
]

DIP_K = [
    0.0375,
    0.0425,
    0.0475,
    0.0525,
]


# Preserve current MCP dry friction.
MCP_FRICTION = 0.0020

# Keep DIP friction zero in this first structural scan.
DIP_FRICTION = 0.0


def find_joint(root, name):

    joint = root.find(
        f".//joint[@name='{name}']"
    )

    if joint is None:
        raise RuntimeError(
            f"Missing joint: {name}"
        )

    return joint


def patch_passives(
    source,
    destination,
    mcp_k,
    pip_k,
    pip_friction,
    dip_k,
):

    tree = ET.parse(source)

    root = tree.getroot()

    mcp = find_joint(
        root,
        v35.MCP,
    )

    pip = find_joint(
        root,
        v35.PIP,
    )

    dip = find_joint(
        root,
        v35.DIP,
    )

    mcp.set(
        "stiffness",
        f"{mcp_k:.10g}",
    )

    mcp.set(
        "frictionloss",
        f"{MCP_FRICTION:.10g}",
    )

    pip.set(
        "stiffness",
        f"{pip_k:.10g}",
    )

    pip.set(
        "frictionloss",
        f"{pip_friction:.10g}",
    )

    dip.set(
        "stiffness",
        f"{dip_k:.10g}",
    )

    dip.set(
        "frictionloss",
        f"{DIP_FRICTION:.10g}",
    )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )


def run_candidate(
    model_path,
    contact_bag,
):

    clamp = v30.FixedSupport()
    clamp.install()

    recorder = v28.Recorder(
        v28.MID_MESH_COLLISION
    )

    recorder.install()

    old_scale = (
        v30.v24.base.METERS_PER_TICK
    )

    old_tendon = (
        v30.v24.base.TENDON_STIFFNESS
    )

    old_mu = (
        v30.v24.base.CONTACT_MU
    )

    old_contact_k = (
        v30.v24.base.CONTACT_STIFFNESS
    )

    try:

        v30.v24.base.METERS_PER_TICK = (
            v35.METERS_PER_TICK
        )

        v30.v24.base.TENDON_STIFFNESS = (
            v35.TENDON_K
        )

        v30.v24.base.CONTACT_MU = (
            v35.CONTACT_MU
        )

        v30.v24.base.CONTACT_STIFFNESS = (
            v35.CONTACT_K
        )

        metrics, _ = (
            v30.v24.base.run_contact(
                model_path,
                contact_bag,
            )
        )

    finally:

        recorder.restore()
        clamp.restore()

        v30.v24.base.METERS_PER_TICK = (
            old_scale
        )

        v30.v24.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v30.v24.base.CONTACT_MU = (
            old_mu
        )

        v30.v24.base.CONTACT_STIFFNESS = (
            old_contact_k
        )

    retention = v28.analyse(
        recorder.rows
    )

    pose = np.asarray(
        metrics["pose"],
        dtype=float,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (pose - HW) ** 2
            )
        )
    )

    # --------------------------------------------------------
    # CONTACT / WRAP PENALTY
    #
    # We explicitly do NOT allow an angle-only winner.
    # --------------------------------------------------------

    penalty = 0.0

    tip = float(
        retention["tip_duration_s"]
    )

    middle = float(
        retention["middle_duration_s"]
    )

    overlap = float(
        retention["overlap_s"]
    )

    after_tip = float(
        retention["middle_after_tip_s"]
    )

    # Hardware H1 has distal contact.
    if tip < 0.03:
        penalty += 8.0

    # Middle must not simply disappear once distal contact starts.
    if after_tip < 0.03:
        penalty += 5.0

    # Reward actual simultaneous wrapping.
    if overlap < 0.02:
        penalty += 4.0

    # Need meaningful middle-object interaction.
    if middle < 0.30:
        penalty += 3.0

    score = (
        rmse
        +
        penalty
    )

    return {
        "pose":
            pose,

        "rmse":
            rmse,

        "score":
            score,

        "middle":
            middle,

        "tip":
            tip,

        "overlap":
            overlap,

        "after_tip":
            after_tip,

        "mu":
            float(
                retention[
                    "effective_mu"
                ]
            ),
    }


def make_viewer_transparent(
    path,
):

    tree = ET.parse(path)

    root = tree.getroot()

    for body_name in (
        "Index_Proximal",
        "Index_Middle",
        "Index_Distal",
    ):

        body = root.find(
            f".//body[@name='{body_name}']"
        )

        if body is None:
            continue

        for geom in body.findall(
            "geom"
        ):

            rgba = geom.attrib.get(
                "rgba"
            )

            if rgba is None:
                continue

            values = rgba.split()

            if len(values) != 4:
                continue

            values[3] = "0.35"

            geom.set(
                "rgba",
                " ".join(values),
            )

    anchor = root.find(
        ".//site[@name='v35_p09_anchor']"
    )

    if anchor is not None:

        anchor.set(
            "rgba",
            "0 1 1 1",
        )

        anchor.set(
            "size",
            "0.005",
        )

    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )


def visual_replay(
    model_path,
    contact_bag,
):

    clamp = v30.FixedSupport()
    clamp.install()

    old_scale = (
        v30.v24.viewer_tools.base.METERS_PER_TICK
    )

    old_tendon = (
        v30.v24.viewer_tools.base.TENDON_STIFFNESS
    )

    old_mu = (
        v30.v24.viewer_tools.base.CONTACT_MU
    )

    old_contact_k = (
        v30.v24.viewer_tools.base.CONTACT_STIFFNESS
    )

    try:

        v30.v24.viewer_tools.base.METERS_PER_TICK = (
            v35.METERS_PER_TICK
        )

        v30.v24.viewer_tools.base.TENDON_STIFFNESS = (
            v35.TENDON_K
        )

        v30.v24.viewer_tools.base.CONTACT_MU = (
            v35.CONTACT_MU
        )

        v30.v24.viewer_tools.base.CONTACT_STIFFNESS = (
            v35.CONTACT_K
        )

        v30.v24.viewer_tools.visual_replay(
            model_path,
            contact_bag,
        )

    finally:

        clamp.restore()

        v30.v24.viewer_tools.base.METERS_PER_TICK = (
            old_scale
        )

        v30.v24.viewer_tools.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v30.v24.viewer_tools.base.CONTACT_MU = (
            old_mu
        )

        v30.v24.viewer_tools.base.CONTACT_STIFFNESS = (
            old_contact_k
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    args = parser.parse_args()

    contact_bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )

    if not contact_bag.exists():
        raise RuntimeError(
            f"Missing bag:\n{contact_bag}"
        )

    # Exact same replay initialization used by V33/V35A.
    v30.v24.ab.patch_replay_modules()

    print()
    print("=" * 120)
    print(
        "V35B2 — LOCAL PASSIVE-JOINT REFINEMENT"
    )
    print("=" * 120)

    print()
    print(
        "Frozen:"
    )

    print(
        "  V33 environment"
    )

    print(
        "  V5 H1 cylinder"
    )

    print(
        "  CAD spatial tendon"
    )

    print(
        "  tendon stiffness"
    )

    print(
        "  contact mu / contact stiffness"
    )

    print()
    print(
        "Scanning ONLY:"
    )

    print(
        "  MCP spring stiffness"
    )

    print(
        "  PIP spring stiffness"
    )

    print(
        "  PIP frictionloss"
    )

    print(
        "  DIP spring stiffness"
    )

    # Build the CAD model once.
    v35.build_model(
        BASE_MODEL
    )

    combinations = list(
        itertools.product(
            MCP_K,
            PIP_K,
            PIP_FRICTION,
            DIP_K,
        )
    )

    print()
    print(
        f"Candidates = {len(combinations)}"
    )

    rows = []

    for index, (
        mcp_k,
        pip_k,
        pip_f,
        dip_k,
    ) in enumerate(
        combinations,
        start=1,
    ):

        patch_passives(
            BASE_MODEL,
            CANDIDATE_MODEL,
            mcp_k,
            pip_k,
            pip_f,
            dip_k,
        )

        result = run_candidate(
            CANDIDATE_MODEL,
            contact_bag,
        )

        result.update(
            {
                "mcp_k":
                    mcp_k,

                "pip_k":
                    pip_k,

                "pip_f":
                    pip_f,

                "dip_k":
                    dip_k,
            }
        )

        rows.append(
            result
        )

        if (
            index % 12 == 0
            or
            index == len(
                combinations
            )
        ):

            print(
                f"  completed "
                f"{index}/{len(combinations)}"
            )

    rows.sort(
        key=lambda x:
            x["score"]
    )

    print()
    print("=" * 120)
    print(
        "TOP V35B CANDIDATES"
    )
    print("=" * 120)

    for rank, row in enumerate(
        rows[:12],
        start=1,
    ):

        q = row["pose"]

        print(
            f"{rank:02d}. "
            f"score={row['score']:6.3f} "
            f"RMSE={row['rmse']:6.3f}  "
            f"kM={row['mcp_k']:.4f} "
            f"kP={row['pip_k']:.4f} "
            f"fP={row['pip_f']:.4f} "
            f"kD={row['dip_k']:.4f}  "
            f"q="
            f"{q[0]:6.2f}/"
            f"{q[1]:6.2f}/"
            f"{q[2]:6.2f}  "
            f"mid={row['middle']:.3f}s "
            f"tip={row['tip']:.3f}s "
            f"overlap={row['overlap']:.3f}s "
            f"mid_after_tip="
            f"{row['after_tip']:.3f}s"
        )

    best = rows[0]

    patch_passives(
        BASE_MODEL,
        SELECTED_MODEL,
        best["mcp_k"],
        best["pip_k"],
        best["pip_f"],
        best["dip_k"],
    )

    make_viewer_transparent(
        SELECTED_MODEL
    )

    print()
    print("=" * 120)
    print(
        "V35B2 SELECTED"
    )
    print("=" * 120)

    print(
        f"MCP stiffness = "
        f"{best['mcp_k']}"
    )

    print(
        f"PIP stiffness = "
        f"{best['pip_k']}"
    )

    print(
        f"PIP frictionloss = "
        f"{best['pip_f']}"
    )

    print(
        f"DIP stiffness = "
        f"{best['dip_k']}"
    )

    print(
        "Pose = "
        f"{best['pose'][0]:.3f} / "
        f"{best['pose'][1]:.3f} / "
        f"{best['pose'][2]:.3f} deg"
    )

    print(
        f"RMSE = "
        f"{best['rmse']:.3f} deg"
    )

    print(
        "Contact = "
        f"middle {best['middle']:.3f}s, "
        f"tip {best['tip']:.3f}s, "
        f"overlap {best['overlap']:.3f}s, "
        f"middle-after-tip "
        f"{best['after_tip']:.3f}s"
    )

    print()
    print(
        "Opening MuJoCo ONLY for selected candidate."
    )

    visual_replay(
        SELECTED_MODEL,
        contact_bag,
    )

    print()
    print("=" * 120)
    print(
        "V35B2 COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
