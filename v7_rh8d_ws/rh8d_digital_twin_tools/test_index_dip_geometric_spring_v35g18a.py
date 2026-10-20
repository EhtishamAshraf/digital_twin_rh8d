#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import scan_index_contact_friction_v35g7 as g7
import scan_index_dip_stiffness_v35g15b as g15b


# ================================================================
# G18A — GEOMETRIC DIP RETURN-SPRING TEST
#
# Replace the simple rotational DIP spring with a physical spatial
# spring spanning the DIP joint.
#
# Motor flexor tendon is NOT changed.
#
# Spring attachment positions are derived from the already-existing
# DIP flexor routing sites and mirrored to the opposite side of the
# joint.
#
# NO:
#   contact-state switching
#   joint locking
#   prescribed motion
#   extra actuator
#   object-specific geometry
# ================================================================


SCALE = 5.70e-6
TENDON_K = 2100.0

CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

DIP_JOINT = "Index_Distal--Index_Middle"

DIP_PARENT_SITE = "v35_p06_dip_parent"
DIP_CHILD_SITE = "v35_p07_dip_child"

RETURN_TENDON = "idx_dip_return_spring"

TARGET_DIP_DEG = 9.3270
TARGET_DIP_RAD = math.radians(
    TARGET_DIP_DEG
)

# G16B -0.15 rad candidate:
# physically useful because it delayed premature DIP activation.
LINEAR_REF = -0.15

BASE_TARGET_TORQUE = (
    0.0475
    *
    TARGET_DIP_RAD
)

LINEAR_K = (
    BASE_TARGET_TORQUE
    /
    (
        TARGET_DIP_RAD
        -
        LINEAR_REF
    )
)

# Torque magnitudes we preserve.
TORQUE_AT_ZERO = (
    LINEAR_K
    *
    abs(
        LINEAR_REF
    )
)

TORQUE_AT_TARGET = (
    LINEAR_K
    *
    (
        TARGET_DIP_RAD
        -
        LINEAR_REF
    )
)


GEOMETRY_SCALES = [
    0.75,
    1.00,
    1.25,
]


def find_site_and_body(
    root,
    site_name,
):

    for body in root.iter(
        "body"
    ):

        for site in body.findall(
            "site"
        ):

            if (
                site.get("name")
                ==
                site_name
            ):
                return body, site

    raise RuntimeError(
        f"Site not found: {site_name}"
    )


def find_joint(
    root,
    name,
):

    for joint in root.iter(
        "joint"
    ):

        if (
            joint.get("name")
            ==
            name
        ):
            return joint

    raise RuntimeError(
        f"Joint not found: {name}"
    )


def vec3(text):

    x = [
        float(v)
        for v in text.split()
    ]

    if len(x) != 3:
        raise RuntimeError(
            f"Expected vec3, got {text}"
        )

    return np.asarray(
        x,
        dtype=float,
    )


def vec3_string(x):

    return " ".join(
        f"{v:.12g}"
        for v in x
    )


def add_geometric_spring(
    source_xml,
    destination,
    lateral_scale,
):

    tree = ET.parse(
        source_xml
    )

    root = tree.getroot()

    dip_joint = find_joint(
        root,
        DIP_JOINT,
    )

    # Disable only the old rotational elastic term.
    # Keep damping/friction exactly as before.
    dip_joint.set(
        "stiffness",
        "0"
    )

    dip_joint.set(
        "springref",
        "0"
    )

    (
        parent_body,
        parent_site,
    ) = find_site_and_body(
        root,
        DIP_PARENT_SITE,
    )

    (
        child_body,
        child_site,
    ) = find_site_and_body(
        root,
        DIP_CHILD_SITE,
    )

    p_parent = vec3(
        parent_site.get(
            "pos"
        )
    )

    p_child = vec3(
        child_site.get(
            "pos"
        )
    )

    # ------------------------------------------------------------
    # Motor tendon runs on negative-y side in the current CAD route.
    #
    # The return spring is placed on the opposite side.
    #
    # x/z positions are preserved from the actual CAD-derived sites;
    # only the transverse offset is mirrored.
    # ------------------------------------------------------------

    spring_parent = (
        p_parent.copy()
    )

    spring_child = (
        p_child.copy()
    )

    spring_parent[1] = (
        abs(
            p_parent[1]
        )
        *
        lateral_scale
    )

    spring_child[1] = (
        abs(
            p_child[1]
        )
        *
        lateral_scale
    )

    parent_name = (
        "g18_dip_spring_parent"
    )

    child_name = (
        "g18_dip_spring_child"
    )

    ET.SubElement(
        parent_body,
        "site",
        {
            "name":
                parent_name,

            "pos":
                vec3_string(
                    spring_parent
                ),

            "size":
                "0.0015",

            "rgba":
                "0 1 0 1",
        },
    )

    ET.SubElement(
        child_body,
        "site",
        {
            "name":
                child_name,

            "pos":
                vec3_string(
                    spring_child
                ),

            "size":
                "0.0015",

            "rgba":
                "0 1 0 1",
        },
    )

    tendon_root = root.find(
        "tendon"
    )

    if tendon_root is None:

        raise RuntimeError(
            "No <tendon> section found."
        )

    spatial = ET.SubElement(
        tendon_root,
        "spatial",
        {
            "name":
                RETURN_TENDON,

            # Placeholder.
            # Automatically calibrated below.
            "stiffness":
                "1",

            "damping":
                "0",

            "springlength":
                "0",

            "width":
                "0.001",

            "rgba":
                "0 1 0 1",
        },
    )

    ET.SubElement(
        spatial,
        "site",
        {
            "site":
                parent_name
        },
    )

    ET.SubElement(
        spatial,
        "site",
        {
            "site":
                child_name
        },
    )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )

    return (
        spring_parent,
        spring_child,
    )


def tendon_geometry(
    xml_path,
    q_value,
):

    model = (
        mujoco.MjModel
        .from_xml_path(
            str(
                xml_path
            )
        )
    )

    data = mujoco.MjData(
        model
    )

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        DIP_JOINT,
    )

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        RETURN_TENDON,
    )

    if jid < 0 or tid < 0:

        raise RuntimeError(
            "Could not locate G18 DIP joint/tendon."
        )

    qadr = int(
        model.jnt_qposadr[
            jid
        ]
    )

    def length_at(q):

        data.qpos[:] = (
            model.qpos0
        )

        data.qpos[
            qadr
        ] = q

        mujoco.mj_forward(
            model,
            data,
        )

        return float(
            data.ten_length[
                tid
            ]
        )

    L = length_at(
        q_value
    )

    h = 1e-5

    if q_value <= h:

        Lp = length_at(
            q_value
            +
            h
        )

        J = (
            Lp
            -
            L
        ) / h

    else:

        Lp = length_at(
            q_value
            +
            h
        )

        Lm = length_at(
            q_value
            -
            h
        )

        J = (
            Lp
            -
            Lm
        ) / (
            2.0
            *
            h
        )

    return (
        L,
        J,
    )


def calibrate_spring(
    xml_path,
):

    (
        L0,
        J0,
    ) = tendon_geometry(
        xml_path,
        0.0,
    )

    (
        L1,
        J1,
    ) = tendon_geometry(
        xml_path,
        TARGET_DIP_RAD,
    )

    print()
    print(
        "  raw geometric spring:"
    )

    print(
        f"    L(0)      = "
        f"{L0*1000:.6f} mm"
    )

    print(
        f"    L(target) = "
        f"{L1*1000:.6f} mm"
    )

    print(
        f"    dL/dq(0)  = "
        f"{J0*1000:.6f} mm/rad"
    )

    print(
        f"    dL/dq(T)  = "
        f"{J1*1000:.6f} mm/rad"
    )

    if (
        J0 <= 0
        or
        J1 <= 0
    ):

        raise RuntimeError(
            "Return spring geometry does not lengthen "
            "with DIP flexion. Do not continue."
        )

    if (
        L1
        <=
        L0
    ):

        raise RuntimeError(
            "Return spring does not become longer "
            "with DIP flexion."
        )

    # Required tendon tension at the two calibration states.
    F0 = (
        TORQUE_AT_ZERO
        /
        J0
    )

    F1 = (
        TORQUE_AT_TARGET
        /
        J1
    )

    stiffness = (
        F1
        -
        F0
    ) / (
        L1
        -
        L0
    )

    if (
        not np.isfinite(
            stiffness
        )
        or
        stiffness
        <= 0
    ):

        raise RuntimeError(
            "This geometry cannot reproduce the desired "
            "two-point passive torque with a positive "
            "linear extension spring."
        )

    springlength = (
        L0
        -
        F0
        /
        stiffness
    )

    if springlength <= 0:

        raise RuntimeError(
            "Computed non-positive spring rest length."
        )

    return {
        "L0":
            L0,

        "L1":
            L1,

        "J0":
            J0,

        "J1":
            J1,

        "F0":
            F0,

        "F1":
            F1,

        "stiffness":
            stiffness,

        "springlength":
            springlength,
    }


def apply_calibration(
    xml_path,
    calibration,
):

    tree = ET.parse(
        xml_path
    )

    root = tree.getroot()

    found = None

    tendon_root = root.find(
        "tendon"
    )

    for tendon in tendon_root.findall(
        "spatial"
    ):

        if (
            tendon.get("name")
            ==
            RETURN_TENDON
        ):

            found = tendon
            break

    if found is None:

        raise RuntimeError(
            "Return tendon not found."
        )

    found.set(
        "stiffness",
        f"{calibration['stiffness']:.12g}",
    )

    found.set(
        "springlength",
        f"{calibration['springlength']:.12g}",
    )

    tree.write(
        xml_path,
        encoding="utf-8",
        xml_declaration=True,
    )


def print_torque_curve(
    xml_path,
    calibration,
):

    k = calibration[
        "stiffness"
    ]

    lrest = calibration[
        "springlength"
    ]

    print()
    print(
        "  static geometric DIP spring torque curve:"
    )

    print(
        "    q_deg    length_mm   "
        "moment_arm_mm  tension_N  torque_Nm"
    )

    for q_deg in [
        0,
        5,
        9.327,
        15,
        20,
        30,
        40,
    ]:

        q = math.radians(
            q_deg
        )

        L, J = tendon_geometry(
            xml_path,
            q,
        )

        tension = (
            k
            *
            (
                L
                -
                lrest
            )
        )

        torque = (
            tension
            *
            J
        )

        print(
            f"    {q_deg:7.3f} "
            f"{L*1000:10.4f} "
            f"{J*1000:13.4f} "
            f"{tension:10.4f} "
            f"{torque:10.6f}"
        )


def make_linear_preload(
    source_xml,
    destination,
):

    tree = ET.parse(
        source_xml
    )

    root = tree.getroot()

    dip = find_joint(
        root,
        DIP_JOINT,
    )

    dip.set(
        "springref",
        f"{LINEAR_REF:.12g}",
    )

    dip.set(
        "stiffness",
        f"{LINEAR_K:.12g}",
    )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g18a_dip_geometric_spring"
        ),
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
    )

    args = parser.parse_args()

    bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )

    out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        out_dir
        /
        Path(__file__).name,
    )

    g7.b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    baseline_xml = (
        g7.make_zero_margin_model(
            out_dir
        )
    )

    variants = [
        (
            "baseline_linear",
            baseline_xml,
            None,
        )
    ]

    linear_preload_xml = (
        out_dir
        /
        "linear_preload_m015.xml"
    )

    make_linear_preload(
        baseline_xml,
        linear_preload_xml,
    )

    variants.append(
        (
            "linear_preload_m015",
            linear_preload_xml,
            None,
        )
    )

    calibrations = {}

    for scale in (
        GEOMETRY_SCALES
    ):

        tag = (
            f"{scale:.2f}"
            .replace(".", "p")
        )

        path = (
            out_dir
            /
            f"geometric_spring_{tag}.xml"
        )

        (
            p_parent,
            p_child,
        ) = add_geometric_spring(
            baseline_xml,
            path,
            scale,
        )

        print()
        print("=" * 120)

        print(
            f"GEOMETRIC SPRING "
            f"LATERAL SCALE = {scale:.2f}"
        )

        print("=" * 120)

        print(
            "  parent site =",
            p_parent,
        )

        print(
            "  child site  =",
            p_child,
        )

        calibration = (
            calibrate_spring(
                path
            )
        )

        apply_calibration(
            path,
            calibration,
        )

        calibrations[
            scale
        ] = calibration

        print(
            f"  calibrated stiffness = "
            f"{calibration['stiffness']:.3f} N/m"
        )

        print(
            f"  spring rest length   = "
            f"{calibration['springlength']*1000:.4f} mm"
        )

        print(
            f"  initial tension      = "
            f"{calibration['F0']:.4f} N"
        )

        print_torque_curve(
            path,
            calibration,
        )

        variants.append(
            (
                f"geometric_{tag}",
                path,
                scale,
            )
        )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    old_tendon_k = float(
        g7.b2.v35.TENDON_K
    )

    old_mu = float(
        g7.b2.v35.CONTACT_MU
    )

    old_contact_k = float(
        g7.b2.v35.CONTACT_K
    )

    old_contact_damping = float(
        contact_replay.CONTACT_DAMPING
    )

    rows = []
    trajectories = {}

    try:

        g7.b2.v35.METERS_PER_TICK = (
            SCALE
        )

        g7.b2.v35.TENDON_K = (
            TENDON_K
        )

        g7.b2.v35.CONTACT_MU = (
            CONTACT_MU
        )

        g7.b2.v35.CONTACT_K = (
            CONTACT_K
        )

        contact_replay.CONTACT_DAMPING = (
            CONTACT_DAMPING
        )

        print()
        print("=" * 150)

        print(
            "G18A — DYNAMIC H1 TEST"
        )

        print("=" * 150)

        for (
            name,
            xml,
            geometry_scale,
        ) in variants:

            print()
            print("-" * 150)

            print(
                f"CANDIDATE: {name}"
            )

            print("-" * 150)

            (
                result,
                trace,
                qpos,
                times,
            ) = g7.run_scale(
                SCALE,
                CONTACT_MU,
                xml,
                bag,
                contact_replay,
            )

            row = dict(
                result
            )

            row.update(
                g15b.mechanics(
                    trace
                )
            )

            row[
                "candidate"
            ] = name

            row[
                "geometry_scale"
            ] = (
                np.nan
                if geometry_scale is None
                else geometry_scale
            )

            rows.append(
                row
            )

            trajectories[
                name
            ] = (
                xml,
                qpos,
                times,
            )

            g7.save_trace_csv(
                out_dir
                /
                f"{name}_timeseries.csv",
                trace,
            )

            np.savez_compressed(
                out_dir
                /
                f"{name}_trajectory.npz",
                times=times,
                qpos=qpos,
            )

            print(
                "target pose = "
                f"{row['target_mcp_deg']:.3f} / "
                f"{row['target_pip_deg']:.3f} / "
                f"{row['target_dip_deg']:.3f}"
            )

            print(
                f"RMSE = "
                f"{row['rmse_deg']:.3f} deg"
            )

            print(
                "episodes M/T = "
                f"{row['middle_episodes']} / "
                f"{row['tip_episodes']}"
            )

            print(
                "target contacts M/T = "
                f"{row['target_mid_contact']} / "
                f"{row['target_tip_contact']}"
            )

            print(
                "first-middle pose = "
                f"{row['mcp_first_middle_deg']:.3f} / "
                f"{row['pip_first_middle_deg']:.3f} / "
                f"{row['dip_first_middle_deg']:.3f}"
            )

            print(
                f"middle->tip delay = "
                f"{row['middle_to_tip_delay_s']:.4f} s"
            )

            print(
                "middle->tip dPIP/dDIP = "
                f"{row['delta_pip_middle_to_tip_deg']:+.3f} / "
                f"{row['delta_dip_middle_to_tip_deg']:+.3f}"
            )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

        g7.b2.v35.TENDON_K = (
            old_tendon_k
        )

        g7.b2.v35.CONTACT_MU = (
            old_mu
        )

        g7.b2.v35.CONTACT_K = (
            old_contact_k
        )

        contact_replay.CONTACT_DAMPING = (
            old_contact_damping
        )

    with (
        out_dir
        /
        "g18a_summary.csv"
    ).open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print("=" * 160)

    print(
        "G18A SUMMARY"
    )

    print("=" * 160)

    print(
        "candidate              RMSE "
        "epM epT targetM targetT "
        "DIP@M  delay   dPIP   dDIP"
    )

    for r in rows:

        print(
            f"{r['candidate']:<22s} "
            f"{r['rmse_deg']:5.3f} "
            f"{r['middle_episodes']:3d} "
            f"{r['tip_episodes']:3d} "
            f"{str(r['target_mid_contact']):>7s} "
            f"{str(r['target_tip_contact']):>7s} "
            f"{r['dip_first_middle_deg']:6.2f} "
            f"{r['middle_to_tip_delay_s']:7.4f} "
            f"{r['delta_pip_middle_to_tip_deg']:+6.2f} "
            f"{r['delta_dip_middle_to_tip_deg']:+6.2f}"
        )

    if not args.no_gui:

        # Baseline + linear preload + all geometric variants.
        for i, row in enumerate(
            rows,
            start=1,
        ):

            name = row[
                "candidate"
            ]

            (
                xml,
                qpos,
                times,
            ) = trajectories[
                name
            ]

            print()
            print("#" * 130)

            print(
                f"GUI {i}/{len(rows)} — "
                f"{name}"
            )

            print(
                "episodes M/T = "
                f"{row['middle_episodes']} / "
                f"{row['tip_episodes']}"
            )

            print(
                "dPIP/dDIP = "
                f"{row['delta_pip_middle_to_tip_deg']:+.3f} / "
                f"{row['delta_dip_middle_to_tip_deg']:+.3f}"
            )

            print(
                f"RMSE = "
                f"{row['rmse_deg']:.3f}"
            )

            print("#" * 130)

            g7.exact_gui_replay(
                xml,
                qpos,
                times,
                (
                    f"G18A {name}"
                ),
            )

    with (
        out_dir
        /
        "manifest.txt"
    ).open(
        "w"
    ) as f:

        f.write(
            "V35G18A — geometric DIP return spring\n\n"
        )

        f.write(
            "Motor flexor tendon unchanged.\n"
        )

        f.write(
            "Old DIP rotational spring removed only "
            "for geometric-spring candidates.\n"
        )

        f.write(
            "Return-spring attachment geometry derived by "
            "mirroring existing DIP tendon sites to the "
            "opposite side of the hinge.\n"
        )

        f.write(
            "Geometric spring calibrated to preserve "
            "G16B -0.15 passive torque at q=0 and "
            "hardware target q=9.327 deg.\n"
        )

        f.write(
            "No contact-state logic or object-specific rule.\n"
        )

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
