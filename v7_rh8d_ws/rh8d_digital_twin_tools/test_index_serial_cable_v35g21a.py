#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import fit_index_contact_v1 as bagbase
import scan_index_contact_friction_v35g7 as g7
import scan_index_dip_stiffness_v35g15b as g15b
import test_index_breakaway_hierarchy_v35g20c as g20c
import test_index_mcp_backdrive_v35g20d as g20d


# ================================================================
# G21A — SERIAL COMPLIANT CABLE / GUIDE-FRICTION MODEL
#
# STRUCTURAL HYPOTHESIS:
#
# The real RH8D tendon is physically continuous, but friction/sliding
# through internal guides means local tension need not be uniform:
#
#       T0 != T1 != T2
#
# Current MuJoCo model:
#       one spatial tendon -> one scalar tendon tension
#
# G21A:
#
# spool -- compliant segment -- s1 -- compliant segment -- s2
#       -- compliant distal segment
#
# s1, s2 are INTERNAL cable-slip coordinates, not finger joints.
#
# Their dry friction represents tension drop / resistance to cable
# sliding through internal guides.
#
# IMPORTANT:
#   - original CAD spatial path is preserved for visualization/sensor
#   - its passive force is disabled in G21 candidates
#   - motor spool trajectory is unchanged
#   - MCP/PIP/DIP springs return to ORIGINAL baseline
#   - MCP/PIP/DIP frictionloss return to ORIGINAL baseline
#   - contact geometry unchanged
#   - no contact-state switching
#   - no finger-joint locks
#   - no prescribed MCP/PIP/DIP trajectory
#
# This is a LUMPED structural test. If successful, later replace the
# linearized cable segments with a more detailed spatial/guide model.
# ================================================================


SCALE = 5.70e-6

ORIGINAL_TENDON = "idx_flex_spatial"

SPOOL = "index_motor_spool"

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

SLIP1 = "g21_cable_slip_mcp_pip"
SLIP2 = "g21_cable_slip_pip_dip"

SEG0 = "g21_segment_motor_mcp"
SEG1 = "g21_segment_pip"
SEG2 = "g21_segment_dip"

FIXTURE = "index_contact_fixture"


# Original effective whole-cable axial stiffness.
TOTAL_K = 2100.0


# ----------------------------------------------------------------
# Guide friction is a FORCE [N], because hidden coordinates are
# translational cable-slip coordinates.
#
# These are not H1 contact parameters.
# They represent allowable tension difference across cable guides.
# ----------------------------------------------------------------

CANDIDATES = [
    ("serial_no_guide_friction", 0.00, 0.00),

    ("serial_mild",              0.20, 0.30),

    ("serial_medium",            0.35, 0.55),

    ("serial_strong",            0.50, 0.80),

    ("serial_distal_strong",     0.35, 0.90),
]


CHECKPOINTS = [
    300.0,
    500.0,
    700.0,
    850.0,
    1000.0,
    1027.0,
]


SEGMENT_SITE_PATHS = {
    "proximal": [
        "v35_spool_site",
        "v35_p01_palm_entry",
        "v35_p02_mcp_parent",
        "v35_p03_mcp_child",
        "v35_p04_pip_parent",
    ],

    "middle": [
        "v35_p04_pip_parent",
        "v35_p05_pip_child",
        "v35_p06_dip_parent",
    ],

    "distal": [
        "v35_p06_dip_parent",
        "v35_p07_dip_child",
        "v35_p08_distal_turn",
        "v35_p09_anchor",
    ],
}


def q_at_time(times, qpos, t):

    times = np.asarray(
        times,
        dtype=float,
    )

    qpos = np.asarray(
        qpos,
        dtype=float,
    )

    out = np.empty(
        qpos.shape[1],
        dtype=float,
    )

    for j in range(
        qpos.shape[1]
    ):

        out[j] = np.interp(
            t,
            times,
            qpos[:, j],
        )

    return out


def joint_id(model, name):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    return int(jid)


def joint_qadr(model, name):

    jid = joint_id(
        model,
        name,
    )

    return int(
        model.jnt_qposadr[
            jid
        ]
    )


def tendon_id(model, name):

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        name,
    )

    if tid < 0:
        raise RuntimeError(
            f"Tendon not found: {name}"
        )

    return int(tid)


def site_id(model, name):

    sid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        name,
    )

    if sid < 0:
        raise RuntimeError(
            f"Site not found: {name}"
        )

    return int(sid)


def find_joint_xml(root, name):

    for joint in root.iter("joint"):

        if joint.get("name") == name:
            return joint

    raise RuntimeError(
        f"XML joint not found: {name}"
    )


def find_geom_xml(root, name):

    for geom in root.iter("geom"):

        if geom.get("name") == name:
            return geom

    raise RuntimeError(
        f"XML geom not found: {name}"
    )


def find_spatial_xml(root, name):

    for tendon in root.iter("spatial"):

        if tendon.get("name") == name:
            return tendon

    raise RuntimeError(
        f"XML spatial tendon not found: {name}"
    )


def path_length(
    model,
    data,
    names,
):

    pts = []

    for name in names:

        sid = site_id(
            model,
            name,
        )

        pts.append(
            np.asarray(
                data.site_xpos[
                    sid
                ],
                dtype=float,
            ).copy()
        )

    length = 0.0

    for a, b in zip(
        pts[:-1],
        pts[1:],
    ):

        length += float(
            np.linalg.norm(
                b - a
            )
        )

    return length


def tendon_length(
    model,
    q,
    tendon_name,
):

    data = mujoco.MjData(
        model
    )

    data.qpos[:] = q
    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    tid = tendon_id(
        model,
        tendon_name,
    )

    return float(
        data.ten_length[
            tid
        ]
    )


def finite_difference_coef(
    model,
    q,
    tendon_name,
    joint_name,
):

    adr = joint_qadr(
        model,
        joint_name,
    )

    q_plus = np.asarray(
        q,
        dtype=float,
    ).copy()

    q_minus = q_plus.copy()

    jid = joint_id(
        model,
        joint_name,
    )

    jtype = int(
        model.jnt_type[
            jid
        ]
    )

    # Hinge perturbation [rad], slide perturbation [m].
    if jtype == int(
        mujoco.mjtJoint.mjJNT_HINGE
    ):
        h = 1.0e-5
    else:
        h = 1.0e-6

    q_plus[
        adr
    ] += h

    q_minus[
        adr
    ] -= h

    lp = tendon_length(
        model,
        q_plus,
        tendon_name,
    )

    lm = tendon_length(
        model,
        q_minus,
        tendon_name,
    )

    return (
        lp - lm
    ) / (
        2.0 * h
    )


def derive_transmission(
    source_xml,
    q_ref,
):

    model = mujoco.MjModel.from_xml_path(
        str(
            source_xml
        )
    )

    data = mujoco.MjData(
        model
    )

    data.qpos[:] = q_ref
    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    raw = {
        "spool":
            finite_difference_coef(
                model,
                q_ref,
                ORIGINAL_TENDON,
                SPOOL,
            ),

        "mcp":
            finite_difference_coef(
                model,
                q_ref,
                ORIGINAL_TENDON,
                MCP,
            ),

        "pip":
            finite_difference_coef(
                model,
                q_ref,
                ORIGINAL_TENDON,
                PIP,
            ),

        "dip":
            finite_difference_coef(
                model,
                q_ref,
                ORIGINAL_TENDON,
                DIP,
            ),
    }

    # Orient the abstract cable coordinate such that increasing
    # motor spool motion LOADS the proximal segment.
    orient = (
        1.0
        if raw[
            "spool"
        ] > 0.0
        else -1.0
    )

    coef = {
        key:
            orient * value
        for key, value
        in raw.items()
    }

    if coef[
        "spool"
    ] <= 0.0:

        raise RuntimeError(
            "Could not orient spool coefficient."
        )

    print()
    print(
        "CAD tendon Jacobian at APPROACH reference:"
    )

    print(
        "  raw dL/dq:"
    )

    for key in [
        "spool",
        "mcp",
        "pip",
        "dip",
    ]:

        print(
            f"    {key:<6s} = "
            f"{raw[key]:+.9f}"
        )

    print()
    print(
        "  oriented serial coefficients:"
    )

    for key in [
        "spool",
        "mcp",
        "pip",
        "dip",
    ]:

        print(
            f"    {key:<6s} = "
            f"{coef[key]:+.9f}"
        )

    # Flexion should relieve cable stretch.
    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        if coef[
            key
        ] >= 0.0:

            raise RuntimeError(
                f"Unexpected tendon sign for {key}: "
                f"{coef[key]:+.9f}. "
                "Do not continue automatically."
            )

    lengths = {}

    for key, names in (
        SEGMENT_SITE_PATHS.items()
    ):

        lengths[
            key
        ] = path_length(
            model,
            data,
            names,
        )

    total_length = sum(
        lengths.values()
    )

    # Uniform cable material:
    #
    #   k_i = EA / L_i
    #
    # and preserve equivalent full-cable stiffness:
    #
    #   1/Ktotal = sum(1/k_i)
    #
    # -> EA = Ktotal * Ltotal
    #
    EA = (
        TOTAL_K
        *
        total_length
    )

    stiffness = {
        key:
            EA
            /
            length
        for key, length
        in lengths.items()
    }

    q_spool = q_ref[
        joint_qadr(
            model,
            SPOOL,
        )
    ]

    q_mcp = q_ref[
        joint_qadr(
            model,
            MCP,
        )
    ]

    q_pip = q_ref[
        joint_qadr(
            model,
            PIP,
        )
    ]

    q_dip = q_ref[
        joint_qadr(
            model,
            DIP,
        )
    ]

    # Internal slip coordinates are zero at reference.
    rest = {
        "proximal":
            (
                coef["spool"]
                *
                q_spool
                +
                coef["mcp"]
                *
                q_mcp
            ),

        "middle":
            (
                coef["pip"]
                *
                q_pip
            ),

        "distal":
            (
                coef["dip"]
                *
                q_dip
            ),
    }

    print()
    print(
        "Physical CAD path lengths:"
    )

    for key in [
        "proximal",
        "middle",
        "distal",
    ]:

        print(
            f"  {key:<8s} = "
            f"{lengths[key]*1000:.3f} mm"
        )

    print(
        f"  total    = "
        f"{total_length*1000:.3f} mm"
    )

    print()
    print(
        "Distributed cable stiffness preserving "
        "2100 N/m equivalent stiffness:"
    )

    for key in [
        "proximal",
        "middle",
        "distal",
    ]:

        print(
            f"  {key:<8s} = "
            f"{stiffness[key]:.2f} N/m"
        )

    return {
        "coef":
            coef,

        "lengths":
            lengths,

        "stiffness":
            stiffness,

        "rest":
            rest,
    }


def add_fixed_tendon(
    tendon_root,
    name,
    stiffness,
    rest,
    joints,
):

    # One-sided cable spring:
    #
    # length <= rest:
    #     slack, zero spring force
    #
    # length > rest:
    #     tension
    #
    # -50 mm is simply a large slack-side deadband.
    lower = (
        rest
        -
        0.050
    )

    tendon = ET.SubElement(
        tendon_root,
        "fixed",
        {
            "name":
                name,

            "stiffness":
                f"{stiffness:.12g}",

            # Deliberately zero in G21A.
            # Built-in tendon damping acts even while the cable
            # coordinate is inside the spring deadband, which would
            # transmit force through a nominally slack segment.
            "damping":
                "0",

            "frictionloss":
                "0",

            "springlength":
                (
                    f"{lower:.12g} "
                    f"{rest:.12g}"
                ),
        },
    )

    for joint_name, coefficient in joints:

        ET.SubElement(
            tendon,
            "joint",
            {
                "joint":
                    joint_name,

                "coef":
                    f"{coefficient:.12g}",
            },
        )


def make_serial_model(
    source_xml,
    destination,
    transmission,
    guide1_friction,
    guide2_friction,
    free_space,
):

    tree = ET.parse(
        source_xml
    )

    root = tree.getroot()

    # ------------------------------------------------------------
    # Restore ORIGINAL physical finger passive parameters.
    #
    # G20C hierarchy was useful diagnostically but is NOT retained.
    # ------------------------------------------------------------

    find_joint_xml(
        root,
        MCP,
    ).set(
        "frictionloss",
        "0.002"
    )

    find_joint_xml(
        root,
        PIP,
    ).set(
        "frictionloss",
        "0.00125"
    )

    find_joint_xml(
        root,
        DIP,
    ).set(
        "frictionloss",
        "0"
    )

    # ------------------------------------------------------------
    # Keep original CAD spatial tendon path for visualization and
    # len_index sensing, but disable its passive force.
    # ------------------------------------------------------------

    original = find_spatial_xml(
        root,
        ORIGINAL_TENDON,
    )

    original.set(
        "stiffness",
        "0"
    )

    original.set(
        "damping",
        "0"
    )

    original.set(
        "frictionloss",
        "0"
    )

    # ------------------------------------------------------------
    # Hidden internal cable-slip coordinates.
    #
    # They do NOT move any finger geometry.
    # They are lumped cable transmission states.
    # ------------------------------------------------------------

    worldbody = root.find(
        "worldbody"
    )

    if worldbody is None:

        raise RuntimeError(
            "No <worldbody> found."
        )

    for (
        body_name,
        joint_name,
        friction,
    ) in [
        (
            "g21_cable_state_1",
            SLIP1,
            guide1_friction,
        ),
        (
            "g21_cable_state_2",
            SLIP2,
            guide2_friction,
        ),
    ]:

        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name":
                    body_name,

                # Arbitrary location because the body is invisible
                # and only its scalar slide coordinate is used.
                "pos":
                    "0 0 0",
            },
        )

        ET.SubElement(
            body,
            "inertial",
            {
                "pos":
                    "0 0 0",

                "mass":
                    "0.0001",

                "diaginertia":
                    "1e-8 1e-8 1e-8",
            },
        )

        ET.SubElement(
            body,
            "joint",
            {
                "name":
                    joint_name,

                "type":
                    "slide",

                "axis":
                    "1 0 0",

                "range":
                    "-0.020 0.020",

                "damping":
                    "0",

                "frictionloss":
                    f"{friction:.12g}",

                # Small effective cable/guide inertia improves
                # numerical conditioning without imposing motion.
                "armature":
                    "0.0001",
            },
        )

    tendon_root = root.find(
        "tendon"
    )

    if tendon_root is None:

        raise RuntimeError(
            "No <tendon> section found."
        )

    c = transmission[
        "coef"
    ]

    k = transmission[
        "stiffness"
    ]

    r = transmission[
        "rest"
    ]

    # ------------------------------------------------------------
    # Segment 0
    #
    # x0 =
    #   + spool displacement
    #   - MCP cable consumption
    #   - cable passed through guide 1
    #
    # Force balance on slip1:
    #
    #       T0 - T1 - guide friction = 0
    # ------------------------------------------------------------

    add_fixed_tendon(
        tendon_root,
        SEG0,
        k[
            "proximal"
        ],
        r[
            "proximal"
        ],
        [
            (
                SPOOL,
                c[
                    "spool"
                ],
            ),
            (
                MCP,
                c[
                    "mcp"
                ],
            ),
            (
                SLIP1,
                -1.0,
            ),
        ],
    )

    # ------------------------------------------------------------
    # Segment 1
    #
    # x1 =
    #   cable entering from guide 1
    #   - PIP cable consumption
    #   - cable passed through guide 2
    # ------------------------------------------------------------

    add_fixed_tendon(
        tendon_root,
        SEG1,
        k[
            "middle"
        ],
        r[
            "middle"
        ],
        [
            (
                SLIP1,
                +1.0,
            ),
            (
                PIP,
                c[
                    "pip"
                ],
            ),
            (
                SLIP2,
                -1.0,
            ),
        ],
    )

    # ------------------------------------------------------------
    # Segment 2
    #
    # x2 =
    #   cable entering from guide 2
    #   - DIP cable consumption
    # ------------------------------------------------------------

    add_fixed_tendon(
        tendon_root,
        SEG2,
        k[
            "distal"
        ],
        r[
            "distal"
        ],
        [
            (
                SLIP2,
                +1.0,
            ),
            (
                DIP,
                c[
                    "dip"
                ],
            ),
        ],
    )

    if free_space:

        fixture = find_geom_xml(
            root,
            FIXTURE,
        )

        fixture.set(
            "contype",
            "0"
        )

        fixture.set(
            "conaffinity",
            "0"
        )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )


def scalar_first(x):

    a = np.asarray(
        x,
        dtype=float,
    ).reshape(-1)

    return float(
        a[0]
    )


def serial_state(
    model,
    q,
):

    data = mujoco.MjData(
        model
    )

    data.qpos[:] = q
    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    output = {
        "slip1_mm":
            (
                q[
                    joint_qadr(
                        model,
                        SLIP1,
                    )
                ]
                *
                1000.0
            ),

        "slip2_mm":
            (
                q[
                    joint_qadr(
                        model,
                        SLIP2,
                    )
                ]
                *
                1000.0
            ),
    }

    for label, tendon_name in [
        (
            "T0",
            SEG0,
        ),
        (
            "T1",
            SEG1,
        ),
        (
            "T2",
            SEG2,
        ),
    ]:

        tid = tendon_id(
            model,
            tendon_name,
        )

        length = float(
            data.ten_length[
                tid
            ]
        )

        springrange = np.asarray(
            model.tendon_lengthspring[
                tid
            ],
            dtype=float,
        ).reshape(-1)

        rest_high = float(
            springrange[
                -1
            ]
        )

        k = scalar_first(
            model.tendon_stiffness[
                tid
            ]
        )

        extension = max(
            0.0,
            length
            -
            rest_high,
        )

        tension = (
            k
            *
            extension
        )

        output[
            f"{label}_N"
        ] = float(
            tension
        )

        output[
            f"{label}_extension_mm"
        ] = float(
            extension
            *
            1000.0
        )

    return output


def checkpoint_state(
    xml,
    times,
    qpos,
    encoder,
    approach_start,
    regulation_start,
):

    model = mujoco.MjModel.from_xml_path(
        str(
            xml
        )
    )

    out = {}

    for tick in CHECKPOINTS:

        t = g20c.time_for_encoder(
            encoder,
            tick,
            approach_start,
            regulation_start
            +
            0.5,
        )

        if not np.isfinite(
            t
        ):
            continue

        q = q_at_time(
            times,
            qpos,
            t,
        )

        state = serial_state(
            model,
            q,
        )

        for key, value in state.items():

            out[
                f"{key}_at_{int(tick)}"
            ] = value

    return out


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
            "v35g21a_serial_cable"
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

    (
        states,
        force_data,
        encoder_data,
    ) = bagbase.read_bag(
        bag
    )

    encoder_data = np.asarray(
        encoder_data,
        dtype=float,
    ).copy()

    encoder_data[:, 1] = (
        bagbase.median_filter_5(
            encoder_data[:, 1]
        )
    )

    transitions = (
        bagbase.compress_states(
            states
        )
    )

    (
        approach_start,
        regulation_start,
        hold_start,
        hold_end,
    ) = bagbase.find_intervals(
        transitions
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

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    old_k = float(
        g7.b2.v35.TENDON_K
    )

    old_damping = getattr(
        g7.b2.v35,
        "TENDON_DAMPING",
        None,
    )

    # ------------------------------------------------------------
    # Run ORIGINAL model once.
    #
    # This gives:
    #   - exact approach reference qpos
    #   - original baseline GUI trajectory
    # ------------------------------------------------------------

    g7.b2.v35.METERS_PER_TICK = (
        SCALE
    )

    g7.b2.v35.TENDON_K = (
        TOTAL_K
    )

    print()
    print("=" * 160)
    print(
        "G21A — ORIGINAL BASELINE REFERENCE RUN"
    )
    print("=" * 160)

    (
        baseline_result,
        baseline_trace,
        baseline_qpos,
        baseline_times,
    ) = g7.run_scale(
        SCALE,
        0.05,
        baseline_xml,
        bag,
        contact_replay,
    )

    q_ref = q_at_time(
        baseline_times,
        baseline_qpos,
        approach_start,
    )

    transmission = derive_transmission(
        baseline_xml,
        q_ref,
    )

    rows = []
    trajectories = {}
    free_trajectories = {}

    try:

        # --------------------------------------------------------
        # Disable force from the ORIGINAL continuous spatial tendon
        # in runtime code as well as in candidate XML.
        #
        # The original tendon remains only as CAD path / sensor.
        # --------------------------------------------------------

        g7.b2.v35.TENDON_K = 0.0

        if old_damping is not None:

            g7.b2.v35.TENDON_DAMPING = 0.0

        print()
        print("=" * 160)
        print(
            "G21A — SERIAL CABLE CANDIDATES"
        )
        print("=" * 160)

        for (
            name,
            guide1,
            guide2,
        ) in CANDIDATES:

            print()
            print("=" * 160)
            print(
                f"CANDIDATE: {name}"
            )
            print(
                f"guide friction F1/F2 = "
                f"{guide1:.3f} / "
                f"{guide2:.3f} N"
            )
            print("=" * 160)

            # ====================================================
            # FREE SPACE
            # ====================================================

            free_xml = (
                out_dir
                /
                f"{name}_free.xml"
            )

            make_serial_model(
                baseline_xml,
                free_xml,
                transmission,
                guide1,
                guide2,
                free_space=True,
            )

            # Compile immediately.
            free_model = (
                mujoco.MjModel
                .from_xml_path(
                    str(
                        free_xml
                    )
                )
            )

            print(
                f"compiled free model: "
                f"nq={free_model.nq}, "
                f"ntendon={free_model.ntendon}"
            )

            (
                free_result,
                free_trace,
                free_qpos,
                free_times,
            ) = g7.run_scale(
                SCALE,
                0.05,
                free_xml,
                bag,
                contact_replay,
            )

            fm = g20c.free_space_metrics(
                free_xml,
                free_times,
                free_qpos,
                encoder_data,
                approach_start,
                regulation_start,
            )

            cable_free = checkpoint_state(
                free_xml,
                free_times,
                free_qpos,
                encoder_data,
                approach_start,
                regulation_start,
            )

            print()
            print(
                "FREE SPACE:"
            )

            print(
                "  activation 1deg MCP/PIP/DIP = "
                f"{fm['mcp_activation_1deg_tick']:.1f} / "
                f"{fm['pip_activation_1deg_tick']:.1f} / "
                f"{fm['dip_activation_1deg_tick']:.1f}"
            )

            print(
                "  activation 3deg MCP/PIP/DIP = "
                f"{fm['mcp_activation_3deg_tick']:.1f} / "
                f"{fm['pip_activation_3deg_tick']:.1f} / "
                f"{fm['dip_activation_3deg_tick']:.1f}"
            )

            print(
                "  staging pass 1deg/3deg = "
                f"{fm['order_1deg_pass']} / "
                f"{fm['order_3deg_pass']}"
            )

            for tick in [
                300,
                500,
                700,
                850,
            ]:

                print(
                    f"  {tick} ticks: "
                    f"MCP={fm.get(f'mcp_at_{tick}_deg', np.nan):+.3f} "
                    f"PIP={fm.get(f'pip_at_{tick}_deg', np.nan):+.3f} "
                    f"DIP={fm.get(f'dip_at_{tick}_deg', np.nan):+.3f}"
                )

                print(
                    "      cable "
                    f"T0/T1/T2="
                    f"{cable_free.get(f'T0_N_at_{tick}', np.nan):.3f}/"
                    f"{cable_free.get(f'T1_N_at_{tick}', np.nan):.3f}/"
                    f"{cable_free.get(f'T2_N_at_{tick}', np.nan):.3f} N "
                    f"s1/s2="
                    f"{cable_free.get(f'slip1_mm_at_{tick}', np.nan):.3f}/"
                    f"{cable_free.get(f'slip2_mm_at_{tick}', np.nan):.3f} mm"
                )

            free_trajectories[
                name
            ] = (
                free_xml,
                free_qpos,
                free_times,
            )

            # ====================================================
            # H1 CONTACT
            # ====================================================

            contact_xml = (
                out_dir
                /
                f"{name}_contact.xml"
            )

            make_serial_model(
                baseline_xml,
                contact_xml,
                transmission,
                guide1,
                guide2,
                free_space=False,
            )

            contact_model = (
                mujoco.MjModel
                .from_xml_path(
                    str(
                        contact_xml
                    )
                )
            )

            print(
                f"compiled contact model: "
                f"nq={contact_model.nq}, "
                f"ntendon={contact_model.ntendon}"
            )

            (
                result,
                trace,
                qpos,
                times,
            ) = g7.run_scale(
                SCALE,
                0.05,
                contact_xml,
                bag,
                contact_replay,
            )

            mechanics = g15b.mechanics(
                trace
            )

            contact_episode = (
                g20d.first_middle_episode_metrics(
                    contact_xml,
                    trace,
                    qpos,
                    times,
                )
            )

            cable_contact = checkpoint_state(
                contact_xml,
                times,
                qpos,
                encoder_data,
                approach_start,
                regulation_start,
            )

            row = dict(
                result
            )

            row.update(
                mechanics
            )

            row.update(
                contact_episode
            )

            row.update(
                cable_contact
            )

            row.update(
                {
                    "candidate":
                        name,

                    "guide1_friction_N":
                        guide1,

                    "guide2_friction_N":
                        guide2,

                    "free_order_1deg":
                        fm[
                            "order_1deg_pass"
                        ],

                    "free_order_3deg":
                        fm[
                            "order_3deg_pass"
                        ],

                    "free_mcp_E1":
                        fm[
                            "mcp_activation_1deg_tick"
                        ],

                    "free_pip_E1":
                        fm[
                            "pip_activation_1deg_tick"
                        ],

                    "free_dip_E1":
                        fm[
                            "dip_activation_1deg_tick"
                        ],
                }
            )

            rows.append(
                row
            )

            trajectories[
                name
            ] = (
                contact_xml,
                qpos,
                times,
            )

            g7.save_trace_csv(
                out_dir
                /
                f"{name}_contact_timeseries.csv",
                trace,
            )

            np.savez_compressed(
                out_dir
                /
                f"{name}_contact_trajectory.npz",
                times=np.asarray(
                    times
                ),
                qpos=np.asarray(
                    qpos
                ),
            )

            print()
            print(
                "H1 CONTACT:"
            )

            print(
                f"  pose = "
                f"{row['target_mcp_deg']:.3f} / "
                f"{row['target_pip_deg']:.3f} / "
                f"{row['target_dip_deg']:.3f}"
            )

            print(
                f"  RMSE = "
                f"{row['rmse_deg']:.3f} deg"
            )

            print(
                "  episodes M/T = "
                f"{row['middle_episodes']} / "
                f"{row['tip_episodes']}"
            )

            print(
                "  target contacts M/T = "
                f"{row['target_mid_contact']} / "
                f"{row['target_tip_contact']}"
            )

            print(
                "  first middle pose = "
                f"{row['mcp_first_middle_deg']:.3f} / "
                f"{row['pip_first_middle_deg']:.3f} / "
                f"{row['dip_first_middle_deg']:.3f}"
            )

            print(
                f"  first middle duration = "
                f"{row['first_mid_duration_s']:.3f} s"
            )

            print(
                f"  gap to second middle = "
                f"{row['gap_to_second_middle_s']:.3f} s"
            )

            print(
                "  middle->tip dPIP/dDIP = "
                f"{row['delta_pip_middle_to_tip_deg']:+.3f} / "
                f"{row['delta_dip_middle_to_tip_deg']:+.3f}"
            )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

        g7.b2.v35.TENDON_K = (
            old_k
        )

        if old_damping is not None:

            g7.b2.v35.TENDON_DAMPING = (
                old_damping
            )

    # ============================================================
    # SAVE SUMMARY
    # ============================================================

    csv_path = (
        out_dir
        /
        "g21a_summary.csv"
    )

    with csv_path.open(
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

    # ============================================================
    # MECHANICS-FIRST RANK
    # ============================================================

    ranked = sorted(
        rows,
        key=lambda r: (
            not r[
                "free_order_1deg"
            ],

            not r[
                "free_order_3deg"
            ],

            not (
                r[
                    "middle_episodes"
                ]
                ==
                1
            ),

            not (
                r[
                    "tip_episodes"
                ]
                ==
                1
            ),

            not (
                r[
                    "target_mid_contact"
                ]
                and
                r[
                    "target_tip_contact"
                ]
            ),

            r[
                "rmse_deg"
            ],
        ),
    )

    print()
    print("=" * 190)
    print(
        "G21A FINAL SUMMARY"
    )
    print("=" * 190)

    print(
        "candidate                 "
        "Fg1  Fg2  "
        "ord1 ord3 "
        "E_M   E_P   E_D   "
        "epM epT targetM targetT "
        "gapMid RMSE"
    )

    print("-" * 190)

    for r in ranked:

        print(
            f"{r['candidate']:<25s} "
            f"{r['guide1_friction_N']:4.2f} "
            f"{r['guide2_friction_N']:4.2f} "
            f"{str(r['free_order_1deg']):>5s} "
            f"{str(r['free_order_3deg']):>5s} "
            f"{r['free_mcp_E1']:6.1f} "
            f"{r['free_pip_E1']:6.1f} "
            f"{r['free_dip_E1']:6.1f} "
            f"{r['middle_episodes']:3d} "
            f"{r['tip_episodes']:3d} "
            f"{str(r['target_mid_contact']):>7s} "
            f"{str(r['target_tip_contact']):>7s} "
            f"{r['gap_to_second_middle_s']:6.3f} "
            f"{r['rmse_deg']:6.3f}"
        )

    # ============================================================
    # GUI
    # ============================================================

    if not args.no_gui:

        best = ranked[0]

        best_name = best[
            "candidate"
        ]

        print()
        print("=" * 140)
        print(
            "MANDATORY G21A GUI VALIDATION"
        )
        print("=" * 140)

        print()
        print(
            "GUI 1/3 — ORIGINAL continuous-tension baseline"
        )

        g7.exact_gui_replay(
            baseline_xml,
            baseline_qpos,
            baseline_times,
            "G21A original baseline",
        )

        print()
        print(
            f"GUI 2/3 — {best_name} FREE SPACE"
        )

        print(
            "Check MCP -> PIP -> DIP staging."
        )

        (
            xml,
            qpos,
            times,
        ) = free_trajectories[
            best_name
        ]

        g7.exact_gui_replay(
            xml,
            qpos,
            times,
            (
                f"G21A {best_name} FREE SPACE"
            ),
        )

        print()
        print(
            f"GUI 3/3 — {best_name} H1 CONTACT"
        )

        print(
            "Critical acceptance sequence:"
        )

        print(
            "  middle touches"
        )

        print(
            "  -> middle REMAINS touching"
        )

        print(
            "  -> PIP wraps"
        )

        print(
            "  -> distal touches"
        )

        print(
            "  -> DIP wraps"
        )

        print(
            "  -> no visible gap / rebound"
        )

        (
            xml,
            qpos,
            times,
        ) = trajectories[
            best_name
        ]

        g7.exact_gui_replay(
            xml,
            qpos,
            times,
            (
                f"G21A {best_name} H1 CONTACT"
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
            "V35G21A — serial compliant cable / "
            "guide-friction structural test\n\n"
        )

        f.write(
            "Reason:\n"
        )

        f.write(
            "G20A proved original tendon can redistribute.\n"
        )

        f.write(
            "G20C proved staged load hierarchy is needed.\n"
        )

        f.write(
            "G20D falsified joint-level MCP friction as "
            "the cause of middle-contact loss.\n\n"
        )

        f.write(
            "Original continuous CAD tendon retained as "
            "zero-force visualization/sensor path.\n"
        )

        f.write(
            "Force transmission replaced experimentally by "
            "three compliant serial cable coordinates and "
            "two frictional internal slip DOFs.\n"
        )

        f.write(
            "No contact-state switching, no finger locks, "
            "no prescribed phalanx trajectories.\n"
        )

    print()
    print(
        f"Saved:\n  {csv_path}"
    )

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
