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
import audit_index_tendon_transmission_v35g12 as g12
import test_transverse_spatial_springs_v35g31a as g31a
import test_direct_spatial_tendon_drive_v35g32a as g32a
import test_spatial_prescribed_spool_v35g29b as g29b


# =====================================================================
# G33A — WRAPPED TRANSVERSE RETURN SPRINGS
#
# Starting architecture:
#
# measured encoder
#      -> direct position control of idx_flex_spatial
#      -> independent MCP / PIP / DIP
#
# Passive mechanics:
#
# parent anchor
#      -> cylindrical guide centred on joint axis
#      -> child anchor
#
# This replaces G31A's straight extension-spring path.
#
# Main flexor stays spatial.
# No PIP/DIP equality.
# No friction scan.
# No K/preload scan.
# K and preload are re-identified after changing spring geometry.
# =====================================================================


MAIN_TENDON = "idx_flex_spatial"

JOINTS = {
    "mcp": "Index_Proximal--palm_assembly:1",
    "pip": "Index_Middle--Index_Proximal",
    "dip": "Index_Distal--Index_Middle",
}

SPRINGS = {
    "mcp": "idx_mcp_return_spring",
    "pip": "idx_pip_return_spring",
    "dip": "idx_dip_return_spring",
}

CAD_CROSSINGS = {
    "mcp": (
        "v35_p02_mcp_parent",
        "v35_p03_mcp_child",
    ),
    "pip": (
        "v35_p04_pip_parent",
        "v35_p05_pip_child",
    ),
    "dip": (
        "v35_p06_dip_parent",
        "v35_p07_dip_child",
    ),
}


# Preserve approximately the 4 mm/rad low-angle leverage
# of the G31A straight spring.
GUIDE_RADIUS = 0.0040

# Sites stay just outside the guide.
ANCHOR_RADIAL = 0.0060

# Distance along each adjacent link from hinge.
ANCHOR_AXIAL = 0.0040

GUIDE_HALF_LENGTH = 0.0060


SCALE = 5.70e-6

ID_POINTS = [
    (416.1,   10.620, 3.994, 1.911),
    (611.7,   18.094, 4.526, 3.724),
    (673.573, 18.543, 6.161, 5.276),
]


HW_CONTACT_ENCODER = 673.573

HW_CONTACT_POSE = np.array(
    [18.543, 6.161, 5.276],
    dtype=float,
)

TRUSTED_FINAL = np.array(
    [14.5355, 23.8110, 9.3270],
    dtype=float,
)

TARGET_TIME = 38.970486


def unit(v):
    v = np.asarray(v, dtype=float)

    n = float(
        np.linalg.norm(v)
    )

    if n < 1e-12:
        raise RuntimeError(
            "Zero vector in geometry construction."
        )

    return v / n


def project_perpendicular(v, axis):
    return (
        np.asarray(v, dtype=float)
        -
        axis
        *
        np.dot(v, axis)
    )


def local_from_world(
    body_pos,
    body_R,
    point,
):
    return (
        body_R.T
        @
        (
            point
            -
            body_pos
        )
    )


def find_body_xml(root, name):
    for body in root.iter("body"):
        if body.get("name") == name:
            return body

    raise RuntimeError(
        f"Body not found: {name}"
    )


def find_joint_xml(root, name):
    for joint in root.iter("joint"):
        if joint.get("name") == name:
            return joint

    raise RuntimeError(
        f"Joint not found: {name}"
    )


def build_wrapped_geometry(
    source,
    destination,
):

    # Flatten first, just as G31A did.
    source_model = mujoco.MjModel.from_xml_path(
        str(source)
    )

    flat = (
        destination.parent
        /
        "g33a_flat_source.xml"
    )

    mujoco.mj_saveLastXML(
        str(flat),
        source_model,
    )

    model = mujoco.MjModel.from_xml_path(
        str(flat)
    )

    data = mujoco.MjData(
        model
    )

    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    tree = ET.parse(
        flat
    )

    root = tree.getroot()

    tendon_root = root.find(
        "./tendon"
    )

    if tendon_root is None:
        raise RuntimeError(
            "Top-level <tendon> missing."
        )

    print()
    print("=" * 150)
    print(
        "G33A STEP 1 — BUILD WRAPPED RETURN-SPRING GEOMETRY"
    )
    print("=" * 150)

    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        joint_name = JOINTS[
            key
        ]

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if jid < 0:
            raise RuntimeError(
                f"Joint missing: {joint_name}"
            )

        child_bid = int(
            model.jnt_bodyid[
                jid
            ]
        )

        parent_bid = int(
            model.body_parentid[
                child_bid
            ]
        )

        child_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            child_bid,
        )

        parent_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            parent_bid,
        )

        Rc = np.asarray(
            data.xmat[
                child_bid
            ]
        ).reshape(
            3,
            3,
        )

        Rp = np.asarray(
            data.xmat[
                parent_bid
            ]
        ).reshape(
            3,
            3,
        )

        pc = np.asarray(
            data.xpos[
                child_bid
            ],
            dtype=float,
        )

        pp = np.asarray(
            data.xpos[
                parent_bid
            ],
            dtype=float,
        )

        joint_local = np.asarray(
            model.jnt_pos[
                jid
            ],
            dtype=float,
        )

        axis_local = unit(
            np.asarray(
                model.jnt_axis[
                    jid
                ],
                dtype=float,
            )
        )

        center = (
            pc
            +
            Rc
            @
            joint_local
        )

        axis_world = unit(
            Rc
            @
            axis_local
        )

        # Child-link longitudinal direction.
        distal_world = project_perpendicular(
            Rc
            @
            np.array(
                [
                    0.0,
                    0.0,
                    -1.0,
                ]
            ),
            axis_world,
        )

        distal_world = unit(
            distal_world
        )

        # ---------------------------------------------------------
        # Find which side the CAD FLEXOR passes.
        #
        # Return spring is routed on the OPPOSITE side.
        # ---------------------------------------------------------

        parent_site_name, child_site_name = (
            CAD_CROSSINGS[
                key
            ]
        )

        parent_sid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            parent_site_name,
        )

        child_sid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            child_site_name,
        )

        if (
            parent_sid < 0
            or
            child_sid < 0
        ):
            raise RuntimeError(
                f"CAD crossing sites missing for {key}"
            )

        crossing_mid = (
            0.5
            *
            (
                np.asarray(
                    data.site_xpos[
                        parent_sid
                    ]
                )
                +
                np.asarray(
                    data.site_xpos[
                        child_sid
                    ]
                )
            )
        )

        flexor_radial = project_perpendicular(
            crossing_mid
            -
            center,
            axis_world,
        )

        flexor_radial = unit(
            flexor_radial
        )

        dorsal_world = (
            -
            flexor_radial
        )

        # Parent/child anchors.
        parent_world = (
            center
            -
            ANCHOR_AXIAL
            *
            distal_world
            +
            ANCHOR_RADIAL
            *
            dorsal_world
        )

        child_world = (
            center
            +
            ANCHOR_AXIAL
            *
            distal_world
            +
            ANCHOR_RADIAL
            *
            dorsal_world
        )

        side_world = (
            center
            +
            (
                GUIDE_RADIUS
                +
                0.002
            )
            *
            dorsal_world
        )

        parent_local = local_from_world(
            pp,
            Rp,
            parent_world,
        )

        child_local = local_from_world(
            pc,
            Rc,
            child_world,
        )

        side_local = local_from_world(
            pc,
            Rc,
            side_world,
        )

        parent_body = find_body_xml(
            root,
            parent_name,
        )

        child_body = find_body_xml(
            root,
            child_name,
        )

        psite = (
            f"g33a_{key}_spring_parent"
        )

        csite = (
            f"g33a_{key}_spring_child"
        )

        sidesite = (
            f"g33a_{key}_spring_side"
        )

        guide = (
            f"g33a_{key}_spring_guide"
        )

        ET.SubElement(
            parent_body,
            "site",
            {
                "name": psite,
                "pos": " ".join(
                    f"{x:.12g}"
                    for x in parent_local
                ),
                "size": "0.001",
                "rgba": "0 1 0 1",
                "group": "4",
            },
        )

        ET.SubElement(
            child_body,
            "site",
            {
                "name": csite,
                "pos": " ".join(
                    f"{x:.12g}"
                    for x in child_local
                ),
                "size": "0.001",
                "rgba": "0 1 0 1",
                "group": "4",
            },
        )

        ET.SubElement(
            child_body,
            "site",
            {
                "name": sidesite,
                "pos": " ".join(
                    f"{x:.12g}"
                    for x in side_local
                ),
                "size": "0.0015",
                "rgba": "1 1 0 1",
                "group": "4",
            },
        )

        # Cylinder lies exactly on the hinge axis.
        p1 = (
            joint_local
            -
            GUIDE_HALF_LENGTH
            *
            axis_local
        )

        p2 = (
            joint_local
            +
            GUIDE_HALF_LENGTH
            *
            axis_local
        )

        ET.SubElement(
            child_body,
            "geom",
            {
                "name": guide,
                "type": "cylinder",
                "fromto":
                    " ".join(
                        f"{x:.12g}"
                        for x in np.concatenate(
                            [p1, p2]
                        )
                    ),
                "size":
                    f"{GUIDE_RADIUS:.12g}",
                "rgba":
                    "0.15 0.8 0.15 0.35",
                "contype":
                    "0",
                "conaffinity":
                    "0",
                "group":
                    "2",
            },
        )

        spring = ET.SubElement(
            tendon_root,
            "spatial",
            {
                "name":
                    SPRINGS[
                        key
                    ],
                "stiffness":
                    "0",
                "springlength":
                    "0",
                "width":
                    "0.0012",
                "rgba":
                    "0 1 0 1",
                "group":
                    "2",
            },
        )

        ET.SubElement(
            spring,
            "site",
            {
                "site": psite,
            },
        )

        ET.SubElement(
            spring,
            "geom",
            {
                "geom": guide,
                "sidesite": sidesite,
            },
        )

        ET.SubElement(
            spring,
            "site",
            {
                "site": csite,
            },
        )

        # Remove the old scalar spring/friction representation.
        # Damping remains.
        joint_xml = find_joint_xml(
            root,
            joint_name,
        )

        joint_xml.set(
            "stiffness",
            "0",
        )

        joint_xml.set(
            "springref",
            "0",
        )

        joint_xml.set(
            "frictionloss",
            "0",
        )

        print()
        print(
            key.upper()
        )

        print(
            f"  joint         = {joint_name}"
        )

        print(
            f"  guide radius  = "
            f"{1000*GUIDE_RADIUS:.3f} mm"
        )

        print(
            "  return side   = opposite the CAD flexor route"
        )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )

    # Compile gate.
    mujoco.MjModel.from_xml_path(
        str(destination)
    )


def nearest_index(
    values,
    target,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    return int(
        np.argmin(
            np.abs(
                values
                -
                target
            )
        )
    )


def contact_metrics(
    rows,
):

    mid_starts, mid_ends = (
        g32a.episodes(
            rows,
            "mid_contact",
        )
    )

    tip_starts, tip_ends = (
        g32a.episodes(
            rows,
            "tip_contact",
        )
    )

    if not mid_starts:
        raise RuntimeError(
            "No middle contact."
        )

    first = int(
        mid_starts[
            0
        ]
    )

    first_end = int(
        mid_ends[
            0
        ]
    )

    first_row = rows[
        first
    ]

    pose = np.array(
        [
            first_row[
                "mcp_deg"
            ],
            first_row[
                "pip_deg"
            ],
            first_row[
                "dip_deg"
            ],
        ],
        dtype=float,
    )

    first_rmse = float(
        np.sqrt(
            np.mean(
                (
                    pose
                    -
                    HW_CONTACT_POSE
                )
                **
                2
            )
        )
    )

    if len(
        mid_starts
    ) >= 2:

        gap = (
            rows[
                mid_starts[
                    1
                ]
            ][
                "time_s"
            ]
            -
            rows[
                first_end
            ][
                "time_s"
            ]
        )

    else:

        gap = 0.0

    return {
        "mid_starts":
            mid_starts,

        "tip_starts":
            tip_starts,

        "first_row":
            first_row,

        "first_pose":
            pose,

        "first_rmse":
            first_rmse,

        "gap":
            gap,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
        help="G25B spatial-tendon model",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g33a_wrapped_return_springs"
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

    source = (
        args.xml
        .expanduser()
        .resolve()
    )

    out = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        out
        /
        Path(__file__).name,
    )

    geometry_xml = (
        out
        /
        "g33a_wrapped_spring_geometry.xml"
    )

    fitted_xml = (
        out
        /
        "g33a_wrapped_springs_fitted.xml"
    )

    direct_xml = (
        out
        /
        "g33a_wrapped_springs_direct_tendon.xml"
    )

    free_xml = (
        out
        /
        "g33a_free.xml"
    )

    print()
    print("=" * 150)
    print(
        "G33A — WRAPPED TRANSVERSE RETURN SPRINGS"
    )
    print("=" * 150)

    g29b.verify_spatial_tendon(
        source
    )

    # ============================================================
    # 1. Build actual wrapped spring topology.
    # ============================================================

    build_wrapped_geometry(
        source,
        geometry_xml,
    )

    # Reuse G31A's mechanical geometry audit.
    geometry_info = (
        g31a.geometry_audit(
            geometry_xml
        )
    )

    # ============================================================
    # 2. Re-identify spring K + preload for NEW geometry.
    # ============================================================

    fitted = (
        g31a.identify_springs(
            geometry_xml,
            bag,
            geometry_info,
        )
    )

    g31a.apply_fitted_springs(
        geometry_xml,
        fitted_xml,
        fitted,
    )

    # ============================================================
    # 3. Apply G32A direct spatial-tendon drive.
    # ============================================================

    g32a.build_direct_model(
        fitted_xml,
        direct_xml,
    )

    g32a.verify_architecture(
        direct_xml
    )

    (
        open_length,
        ratio,
        ratio_rmse,
    ) = (
        g32a.fit_encoder_to_tendon_shortening(
            direct_xml
        )
    )

    g32a.disable_fixture(
        direct_xml,
        free_xml,
    )

    # ============================================================
    # 4. Dynamics
    # ============================================================

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    g7.b2.v35.METERS_PER_TICK = (
        SCALE
    )

    try:

        print()
        print("=" * 150)
        print(
            "G33A STEP 2 — H1 APPROACH"
        )
        print("=" * 150)

        (
            (
                free_rows,
                free_qpos,
                free_times,
                free_summary,
            ),
            free_history,
        ) = g32a.run_direct(
            free_xml,
            bag,
            replay,
            open_length,
            ratio,
        )

        enc = np.asarray(
            [
                r[
                    "encoder_tick"
                ]
                for r in free_rows
            ],
            dtype=float,
        )

        errors = []

        print()
        print(
            "encoder       HW MCP/PIP/DIP          "
            "SIM MCP/PIP/DIP          error"
        )

        print("-" * 140)

        for (
            tick,
            hm,
            hp,
            hd,
        ) in ID_POINTS:

            i = nearest_index(
                enc,
                tick,
            )

            row = free_rows[
                i
            ]

            sim = np.array(
                [
                    row["mcp_deg"],
                    row["pip_deg"],
                    row["dip_deg"],
                ],
                dtype=float,
            )

            hw = np.array(
                [
                    hm,
                    hp,
                    hd,
                ],
                dtype=float,
            )

            error = (
                sim
                -
                hw
            )

            errors.extend(
                error.tolist()
            )

            print(
                f"{tick:8.3f}     "
                f"{hm:7.3f}/{hp:7.3f}/{hd:7.3f}     "
                f"{sim[0]:7.3f}/{sim[1]:7.3f}/{sim[2]:7.3f}     "
                f"{error[0]:+7.3f}/"
                f"{error[1]:+7.3f}/"
                f"{error[2]:+7.3f}"
            )

        approach_rmse = float(
            np.sqrt(
                np.mean(
                    np.asarray(
                        errors
                    )
                    **
                    2
                )
            )
        )

        print()
        print(
            f"APPROACH RMSE = "
            f"{approach_rmse:.3f} deg"
        )

        print()
        print("=" * 150)
        print(
            "G33A STEP 3 — ONE H1 CONTACT RUN"
        )
        print("=" * 150)

        (
            (
                rows,
                qpos,
                times,
                summary,
            ),
            contact_history,
        ) = g32a.run_direct(
            direct_xml,
            bag,
            replay,
            open_length,
            ratio,
        )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

    cm = contact_metrics(
        rows
    )

    first_row = cm[
        "first_row"
    ]

    first_pose = cm[
        "first_pose"
    ]

    print()
    print(
        "FIRST MIDDLE CONTACT"
    )

    print(
        f"  encoder = "
        f"{first_row['encoder_tick']:.3f}"
    )

    print(
        f"  hardware= "
        f"{HW_CONTACT_ENCODER:.3f}"
    )

    print(
        f"  error   = "
        f"{first_row['encoder_tick'] - HW_CONTACT_ENCODER:+.3f}"
    )

    print()

    print(
        "  simulation = "
        f"{first_pose[0]:.3f} / "
        f"{first_pose[1]:.3f} / "
        f"{first_pose[2]:.3f}"
    )

    print(
        "  hardware   = "
        f"{HW_CONTACT_POSE[0]:.3f} / "
        f"{HW_CONTACT_POSE[1]:.3f} / "
        f"{HW_CONTACT_POSE[2]:.3f}"
    )

    print(
        f"  pose RMSE  = "
        f"{cm['first_rmse']:.3f} deg"
    )

    print()
    print(
        "CONTACT CONTINUITY"
    )

    print(
        f"  middle episodes = "
        f"{len(cm['mid_starts'])}"
    )

    print(
        f"  tip episodes    = "
        f"{len(cm['tip_starts'])}"
    )

    print(
        f"  detached gap    = "
        f"{cm['gap']:.3f} s"
    )

    # ============================================================
    # Final H1
    # ============================================================

    row_times = np.asarray(
        [
            r[
                "time_s"
            ]
            for r in rows
        ]
    )

    ti = nearest_index(
        row_times,
        TARGET_TIME,
    )

    final = np.array(
        [
            rows[ti]["mcp_deg"],
            rows[ti]["pip_deg"],
            rows[ti]["dip_deg"],
        ]
    )

    final_error = (
        final
        -
        TRUSTED_FINAL
    )

    final_rmse = float(
        np.sqrt(
            np.mean(
                final_error
                **
                2
            )
        )
    )

    print()
    print(
        "TRUSTED FINAL H1"
    )

    print(
        "  simulation = "
        f"{final[0]:.3f} / "
        f"{final[1]:.3f} / "
        f"{final[2]:.3f}"
    )

    print(
        "  hardware   = "
        f"{TRUSTED_FINAL[0]:.3f} / "
        f"{TRUSTED_FINAL[1]:.3f} / "
        f"{TRUSTED_FINAL[2]:.3f}"
    )

    print(
        "  error      = "
        f"{final_error[0]:+.3f} / "
        f"{final_error[1]:+.3f} / "
        f"{final_error[2]:+.3f}"
    )

    print(
        f"  RMSE       = "
        f"{final_rmse:.3f} deg"
    )

    # ============================================================
    # Save
    # ============================================================

    summary_csv = (
        out
        /
        "g33a_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shortening_ratio",
                "approach_rmse_deg",
                "first_contact_encoder",
                "first_contact_pose_rmse_deg",
                "middle_episodes",
                "tip_episodes",
                "middle_gap_s",
                "final_mcp_deg",
                "final_pip_deg",
                "final_dip_deg",
                "final_rmse_deg",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "shortening_ratio":
                ratio,

            "approach_rmse_deg":
                approach_rmse,

            "first_contact_encoder":
                first_row[
                    "encoder_tick"
                ],

            "first_contact_pose_rmse_deg":
                cm[
                    "first_rmse"
                ],

            "middle_episodes":
                len(
                    cm[
                        "mid_starts"
                    ]
                ),

            "tip_episodes":
                len(
                    cm[
                        "tip_starts"
                    ]
                ),

            "middle_gap_s":
                cm[
                    "gap"
                ],

            "final_mcp_deg":
                final[
                    0
                ],

            "final_pip_deg":
                final[
                    1
                ],

            "final_dip_deg":
                final[
                    2
                ],

            "final_rmse_deg":
                final_rmse,
        })

    np.savez_compressed(
        out
        /
        "g33a_contact_trajectory.npz",
        times=np.asarray(
            times
        ),
        qpos=np.asarray(
            qpos
        ),
    )

    # ============================================================
    # Decision
    # ============================================================

    print()
    print("=" * 150)
    print(
        "G33A DECISION"
    )
    print("=" * 150)

    print(
        f"approach RMSE       = "
        f"{approach_rmse:.3f} deg"
    )

    print(
        f"first-contact RMSE  = "
        f"{cm['first_rmse']:.3f} deg"
    )

    print(
        f"middle gap          = "
        f"{cm['gap']:.3f} s"
    )

    print(
        f"final PIP           = "
        f"{final[1]:.3f} deg "
        f"(HW {TRUSTED_FINAL[1]:.3f})"
    )

    print(
        f"final H1 RMSE       = "
        f"{final_rmse:.3f} deg"
    )

    print()

    if (
        approach_rmse <= 3.5
        and
        cm[
            "first_rmse"
        ] <= 3.0
        and
        cm[
            "gap"
        ] <= 0.25
        and
        final_rmse <= 3.0
    ):

        print(
            "PASS: wrapped return-spring architecture reproduces H1."
        )

        print(
            "FREEZE H1 and move to H2."
        )

    elif (
        cm[
            "gap"
        ] <= 0.25
        and
        final[
            1
        ] > 12.0
    ):

        print(
            "MECHANICALLY PROMISING:"
        )

        print(
            "contact retention remains solved and PIP recruitment "
            "improved strongly."
        )

        print(
            "Do NOT scan spring parameters."
        )

        print(
            "Next = main flexor persistent-wrap geometry from CAD."
        )

    else:

        print(
            "WRAPPED RETURN SPRINGS ARE NOT SUFFICIENT."
        )

        print(
            "No spring K / preload / radius scan."
        )

        print(
            "Next = main flexor persistent-wrap geometry from CAD."
        )

    print()
    print(
        f"Candidate XML:\n  "
        f"{direct_xml}"
    )

    print()
    print(
        f"Summary:\n  "
        f"{summary_csv}"
    )

    print()
    print(
        f"All outputs:\n  "
        f"{out}"
    )

    if not args.no_gui:

        view_xml = (
            out
            /
            "g33a_view.xml"
        )

        g29b.make_view_xml(
            direct_xml,
            view_xml,
        )

        print()
        print("=" * 150)
        print(
            "G33A GUI — WRAPPED RETURN SPRINGS"
        )
        print("=" * 150)

        print(
            "Look specifically at the GREEN return springs."
        )

        print(
            "They should now wrap around a green cylindrical guide "
            "at each joint rather than lying straight across it."
        )

        g7.exact_gui_replay(
            view_xml,
            qpos,
            times,
            "G33A wrapped transverse return springs",
        )


if __name__ == "__main__":
    main()
