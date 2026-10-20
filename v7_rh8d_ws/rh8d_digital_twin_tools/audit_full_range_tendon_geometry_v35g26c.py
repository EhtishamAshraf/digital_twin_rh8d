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


# =====================================================================
# G26C — FULL-RANGE TENDON GEOMETRY AUDIT
#
# NO IDENTIFICATION.
# NO PARAMETER FITTING.
# NO CONTACT TUNING.
#
# Question:
#
# Does the CURRENT CAD spatial tendon remain mechanically valid over
# the real hardware free-space range?
#
# We inspect:
#
#   - signed tendon Jacobian / moment arm at MCP, PIP, DIP
#   - joint-only tendon path-length change
#   - spool contribution
#   - total tendon length
#   - spring extension
#
# If a moment arm changes sign or collapses strongly, the current
# straight-segment spatial routing is mechanically wrong and no spring
# identification should be trusted until tendon wrapping is fixed.
# =====================================================================


SCALE = 5.70e-6

TENDON_NAME = "idx_flex_spatial"
FIXTURE_NAME = "index_contact_fixture"
SPOOL_NAME = "index_motor_spool"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"


FREE_SPACE = [
    # tick,   MCP,      PIP,      DIP
    (100.0,     0.000,    0.000,    0.000),
    (500.0,    16.952,    0.968,   -0.428),
    (1000.0,   39.472,   -0.907,   -0.343),
    (1500.0,   57.291,    3.654,   -4.019),
    (2000.0,   73.632,    4.667,    2.012),
    (2500.0,   73.632,   27.119,    3.190),
    (3000.0,   77.347,   48.581,    7.810),
    (3500.0,   80.074,   70.872,   -7.199),
]


# Useful low-angle H1 measurements too.
H1_POINTS = [
    ("H1_416",   416.1, 10.620, 3.994, 1.911),
    ("H1_612",   611.7, 18.094, 4.526, 3.724),
    ("H1_touch", 673.573, 18.543, 6.161, 5.276),
]


def nearest_index(values, target):

    values = np.asarray(
        values,
        dtype=float,
    )

    return int(
        np.argmin(
            np.abs(
                values - target
            )
        )
    )


def find_geom_xml(root, name):

    for geom in root.iter("geom"):

        if geom.get("name") == name:
            return geom

    raise RuntimeError(
        f"Geom not found: {name}"
    )


def make_free_xml(source, destination):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    fixture = find_geom_xml(
        root,
        FIXTURE_NAME,
    )

    fixture.set(
        "contype",
        "0",
    )

    fixture.set(
        "conaffinity",
        "0",
    )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )


def tendon_jacobian_row(model, data, tid):

    raw = np.asarray(
        data.ten_J,
        dtype=float,
    )

    if (
        raw.ndim == 2
        and
        raw.shape == (
            model.ntendon,
            model.nv,
        )
    ):

        return raw[
            tid,
            :
        ].copy()

    if (
        raw.ndim == 1
        and
        raw.size
        ==
        model.ntendon
        *
        model.nv
    ):

        return raw.reshape(
            model.ntendon,
            model.nv,
        )[
            tid,
            :
        ].copy()

    J = np.zeros(
        model.nv,
        dtype=float,
    )

    rownnz = np.asarray(
        model.ten_J_rownnz,
        dtype=int,
    )

    rowadr = np.asarray(
        model.ten_J_rowadr,
        dtype=int,
    )

    colind = np.asarray(
        model.ten_J_colind,
        dtype=int,
    )

    nnz = int(
        rownnz[
            tid
        ]
    )

    adr = int(
        rowadr[
            tid
        ]
    )

    for k in range(nnz):

        index = (
            adr
            +
            k
        )

        J[
            int(
                colind[
                    index
                ]
            )
        ] = float(
            raw[
                index
            ]
        )

    return J


def joint_refs(model):

    result = {}

    names = {
        "mcp": MCP_NAME,
        "pip": PIP_NAME,
        "dip": DIP_NAME,
    }

    for key, name in names.items():

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:

            raise RuntimeError(
                f"Joint missing: {name}"
            )

        result[
            key
        ] = {
            "jid":
                int(jid),

            "qadr":
                int(
                    model.jnt_qposadr[
                        jid
                    ]
                ),

            "dadr":
                int(
                    model.jnt_dofadr[
                        jid
                    ]
                ),
        }

    return result


def evaluate(
    model,
    data,
    refs,
    tendon_id,
    spool_qadr,
    q_reference,
    open_spool,
    open_encoder,
    spring_reference,
    label,
    tick,
    mcp_deg,
    pip_deg,
    dip_deg,
):

    # --------------------------------------------------------------
    # A. JOINT GEOMETRY ONLY:
    #
    # Hold the motor spool at OPEN position.
    # This isolates the tendon path-length change caused by
    # MCP/PIP/DIP geometry itself.
    # --------------------------------------------------------------

    data.qpos[:] = (
        q_reference
    )

    data.qvel[:] = 0.0

    data.qpos[
        spool_qadr
    ] = open_spool

    data.qpos[
        refs[
            "mcp"
        ][
            "qadr"
        ]
    ] = math.radians(
        mcp_deg
    )

    data.qpos[
        refs[
            "pip"
        ][
            "qadr"
        ]
    ] = math.radians(
        pip_deg
    )

    data.qpos[
        refs[
            "dip"
        ][
            "qadr"
        ]
    ] = math.radians(
        dip_deg
    )

    mujoco.mj_forward(
        model,
        data,
    )

    joint_only_length = float(
        data.ten_length[
            tendon_id
        ]
    )

    # --------------------------------------------------------------
    # B. ADD MOTOR SPOOL MOTION USING THE IDENTIFIED EFFECTIVE SCALE.
    # --------------------------------------------------------------

    spool_q = (
        open_spool
        +
        SCALE
        *
        (
            tick
            -
            open_encoder
        )
    )

    data.qpos[
        spool_qadr
    ] = spool_q

    mujoco.mj_forward(
        model,
        data,
    )

    full_length = float(
        data.ten_length[
            tendon_id
        ]
    )

    J = tendon_jacobian_row(
        model,
        data,
        tendon_id,
    )

    extension = g12.signed_extension(
        full_length,
        spring_reference,
    )

    result = {
        "label":
            label,

        "encoder_tick":
            tick,

        "mcp_deg":
            mcp_deg,

        "pip_deg":
            pip_deg,

        "dip_deg":
            dip_deg,

        "spool_q_m":
            spool_q,

        "motor_delta_mm":
            1000.0
            *
            (
                spool_q
                -
                open_spool
            ),

        "joint_only_length_mm":
            1000.0
            *
            joint_only_length,

        "full_length_mm":
            1000.0
            *
            full_length,

        "extension_mm":
            1000.0
            *
            extension,

        "spool_J":
            float(
                J[
                    int(
                        model.jnt_dofadr[
                            mujoco.mj_name2id(
                                model,
                                mujoco.mjtObj.mjOBJ_JOINT,
                                SPOOL_NAME,
                            )
                        ]
                    )
                ]
            ),

        "mcp_J_mm_rad":
            1000.0
            *
            float(
                J[
                    refs[
                        "mcp"
                    ][
                        "dadr"
                    ]
                ]
            ),

        "pip_J_mm_rad":
            1000.0
            *
            float(
                J[
                    refs[
                        "pip"
                    ][
                        "dadr"
                    ]
                ]
            ),

        "dip_J_mm_rad":
            1000.0
            *
            float(
                J[
                    refs[
                        "dip"
                    ][
                        "dadr"
                    ]
                ]
            ),
    }

    return result


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
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g26c_full_range_tendon_geometry"
        ),
    )

    args = parser.parse_args()

    bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )

    source_xml = (
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

    free_xml = (
        out
        /
        "g26c_free_geometry.xml"
    )

    make_free_xml(
        source_xml,
        free_xml,
    )

    # --------------------------------------------------------------
    # Obtain a real replay OPEN configuration for all irrelevant DOFs.
    # --------------------------------------------------------------

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

        (
            replay_rows,
            replay_qpos,
            replay_times,
            replay_summary,
        ) = g12.run_audit(
            free_xml,
            bag,
            replay,
        )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

    encoder = np.asarray(
        [
            row[
                "encoder_tick"
            ]
            for row in replay_rows
        ],
        dtype=float,
    )

    open_index = nearest_index(
        encoder,
        100.0,
    )

    open_encoder = float(
        encoder[
            open_index
        ]
    )

    q_reference = np.asarray(
        replay_qpos[
            open_index
        ],
        dtype=float,
    ).copy()

    # --------------------------------------------------------------
    # Static model.
    # --------------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(
            free_xml
        )
    )

    data = mujoco.MjData(
        model
    )

    refs = joint_refs(
        model
    )

    tendon_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    spool_jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_NAME,
    )

    if tendon_id < 0:

        raise RuntimeError(
            f"Tendon not found: {TENDON_NAME}"
        )

    if spool_jid < 0:

        raise RuntimeError(
            f"Spool joint not found: {SPOOL_NAME}"
        )

    spool_qadr = int(
        model.jnt_qposadr[
            spool_jid
        ]
    )

    open_spool = float(
        q_reference[
            spool_qadr
        ]
    )

    spring_reference = (
        g12.tendon_spring_reference(
            model,
            tendon_id,
        )
    )

    # Exact OPEN tendon length.
    data.qpos[:] = (
        q_reference
    )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    open_length = float(
        data.ten_length[
            tendon_id
        ]
    )

    print()
    print("=" * 150)
    print(
        "G26C — FULL-RANGE CAD TENDON GEOMETRY AUDIT"
    )
    print("=" * 150)

    print()
    print(
        f"open encoder          = "
        f"{open_encoder:.3f} ticks"
    )

    print(
        f"effective scale       = "
        f"{SCALE:.9e} m/tick"
    )

    print(
        f"open spool qpos       = "
        f"{open_spool:+.9f} m"
    )

    print(
        f"open tendon length    = "
        f"{1000.0 * open_length:.6f} mm"
    )

    print(
        f"tendon spring ref     = "
        f"{1000.0 * spring_reference:.6f} mm"
    )

    # --------------------------------------------------------------
    # Full free-space hardware trajectory.
    # --------------------------------------------------------------

    results = []

    for (
        tick,
        mcp,
        pip,
        dip,
    ) in FREE_SPACE:

        row = evaluate(
            model,
            data,
            refs,
            tendon_id,
            spool_qadr,
            q_reference,
            open_spool,
            open_encoder,
            spring_reference,
            "FREE",
            tick,
            mcp,
            pip,
            dip,
        )

        row[
            "joint_path_change_mm"
        ] = (
            row[
                "joint_only_length_mm"
            ]
            -
            1000.0
            *
            open_length
        )

        row[
            "spool_path_change_mm"
        ] = (
            row[
                "full_length_mm"
            ]
            -
            row[
                "joint_only_length_mm"
            ]
        )

        row[
            "net_path_change_mm"
        ] = (
            row[
                "full_length_mm"
            ]
            -
            1000.0
            *
            open_length
        )

        results.append(
            row
        )

    # Low-angle H1 range.
    for (
        label,
        tick,
        mcp,
        pip,
        dip,
    ) in H1_POINTS:

        row = evaluate(
            model,
            data,
            refs,
            tendon_id,
            spool_qadr,
            q_reference,
            open_spool,
            open_encoder,
            spring_reference,
            label,
            tick,
            mcp,
            pip,
            dip,
        )

        row[
            "joint_path_change_mm"
        ] = (
            row[
                "joint_only_length_mm"
            ]
            -
            1000.0
            *
            open_length
        )

        row[
            "spool_path_change_mm"
        ] = (
            row[
                "full_length_mm"
            ]
            -
            row[
                "joint_only_length_mm"
            ]
        )

        row[
            "net_path_change_mm"
        ] = (
            row[
                "full_length_mm"
            ]
            -
            1000.0
            *
            open_length
        )

        results.append(
            row
        )

    # --------------------------------------------------------------
    # Establish expected sign from low-angle motion.
    # --------------------------------------------------------------

    free_nonzero = [
        row
        for row in results
        if (
            row[
                "label"
            ]
            ==
            "FREE"
            and
            row[
                "encoder_tick"
            ]
            >=
            500.0
        )
    ]

    sign_reference = {}

    for joint in [
        "mcp",
        "pip",
        "dip",
    ]:

        key = (
            f"{joint}_J_mm_rad"
        )

        candidate = None

        for row in free_nonzero:

            value = float(
                row[
                    key
                ]
            )

            if abs(
                value
            ) > 1.0:

                candidate = value
                break

        if candidate is None:

            candidate = float(
                free_nonzero[
                    0
                ][
                    key
                ]
            )

        sign_reference[
            joint
        ] = np.sign(
            candidate
        )

    # --------------------------------------------------------------
    # Print.
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "REAL FREE-SPACE POSES — TENDON GEOMETRY"
    )
    print("=" * 150)

    print(
        "tick    MCP     PIP     DIP   "
        "dL_joint  dL_spool   extension   "
        "J_MCP    J_PIP    J_DIP"
    )

    print("-" * 150)

    for row in results:

        if row[
            "label"
        ] != "FREE":
            continue

        print(
            f"{row['encoder_tick']:4.0f} "
            f"{row['mcp_deg']:7.2f} "
            f"{row['pip_deg']:7.2f} "
            f"{row['dip_deg']:7.2f} "
            f"{row['joint_path_change_mm']:+9.3f} "
            f"{row['spool_path_change_mm']:+9.3f} "
            f"{row['extension_mm']:+10.3f} "
            f"{row['mcp_J_mm_rad']:+8.3f} "
            f"{row['pip_J_mm_rad']:+8.3f} "
            f"{row['dip_J_mm_rad']:+8.3f}"
        )

    print()
    print("=" * 150)
    print(
        "LOW-ANGLE H1 REGION"
    )
    print("=" * 150)

    print(
        "label       tick    MCP     PIP     DIP   "
        "extension    J_MCP    J_PIP    J_DIP"
    )

    print("-" * 150)

    for row in results:

        if row[
            "label"
        ] == "FREE":
            continue

        print(
            f"{row['label']:<10s} "
            f"{row['encoder_tick']:7.1f} "
            f"{row['mcp_deg']:7.2f} "
            f"{row['pip_deg']:7.2f} "
            f"{row['dip_deg']:7.2f} "
            f"{row['extension_mm']:+10.3f} "
            f"{row['mcp_J_mm_rad']:+8.3f} "
            f"{row['pip_J_mm_rad']:+8.3f} "
            f"{row['dip_J_mm_rad']:+8.3f}"
        )

    # --------------------------------------------------------------
    # Detect moment-arm sign reversals / collapse.
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "MOMENT-ARM VALIDITY CHECK"
    )
    print("=" * 150)

    any_failure = False

    for joint in [
        "mcp",
        "pip",
        "dip",
    ]:

        key = (
            f"{joint}_J_mm_rad"
        )

        reference_sign = (
            sign_reference[
                joint
            ]
        )

        sign_flip_rows = []

        collapse_rows = []

        for row in results:

            if row[
                "label"
            ] != "FREE":
                continue

            if (
                row["encoder_tick"]
                <=
                100.0
            ):
                continue

            value = float(
                row[
                    key
                ]
            )

            if (
                reference_sign != 0.0
                and
                np.sign(
                    value
                )
                != 0.0
                and
                np.sign(
                    value
                )
                !=
                reference_sign
            ):

                sign_flip_rows.append(
                    row
                )

            if abs(
                value
            ) < 1.0:

                collapse_rows.append(
                    row
                )

        print()
        print(
            joint.upper()
        )

        print(
            f"  low-angle reference sign = "
            f"{reference_sign:+.0f}"
        )

        if sign_flip_rows:

            any_failure = True

            print(
                "  SIGN REVERSAL:"
            )

            for row in sign_flip_rows:

                print(
                    f"    tick={row['encoder_tick']:.0f} "
                    f"q="
                    f"{row['mcp_deg']:.1f}/"
                    f"{row['pip_deg']:.1f}/"
                    f"{row['dip_deg']:.1f} "
                    f"J={row[key]:+.4f} mm/rad"
                )

        else:

            print(
                "  no sign reversal"
            )

        if collapse_rows:

            any_failure = True

            print(
                "  MOMENT-ARM COLLAPSE (<1 mm):"
            )

            for row in collapse_rows:

                print(
                    f"    tick={row['encoder_tick']:.0f} "
                    f"J={row[key]:+.4f} mm/rad"
                )

        else:

            print(
                "  no <1 mm collapse"
            )

    # --------------------------------------------------------------
    # Additional path consistency.
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "PATH-LENGTH / STRETCH CHECK"
    )
    print("=" * 150)

    for row in results:

        if row[
            "label"
        ] != "FREE":
            continue

        print(
            f"{row['encoder_tick']:4.0f} ticks: "
            f"motor={row['motor_delta_mm']:+7.3f} mm, "
            f"joint-path={row['joint_path_change_mm']:+7.3f} mm, "
            f"net={row['net_path_change_mm']:+7.3f} mm, "
            f"extension={row['extension_mm']:+7.3f} mm"
        )

    # --------------------------------------------------------------
    # Save.
    # --------------------------------------------------------------

    csv_path = (
        out
        /
        "g26c_full_range_tendon_geometry.csv"
    )

    fields = list(
        results[
            0
        ].keys()
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print()
    print("=" * 150)
    print(
        "G26C DECISION"
    )
    print("=" * 150)

    if any_failure:

        print(
            "RESULT: CURRENT SPATIAL TENDON ROUTING FAILS "
            "THE FULL-RANGE GEOMETRY TEST."
        )

        print()
        print(
            "Do NOT fit nonlinear joint springs to this tendon."
        )

        print(
            "Next step: repair tendon wrapping/moment-arm geometry "
            "using physical pulley/wrap routing."
        )

    else:

        print(
            "RESULT: CURRENT TENDON ROUTING PRESERVES MOMENT-ARM "
            "DIRECTION THROUGH THE MEASURED RANGE."
        )

        print()
        print(
            "Then the next model class can legitimately be "
            "nonlinear transverse-joint spring mechanics."
        )

    print()
    print(
        f"CSV:\n  {csv_path}"
    )

    print()
    print(
        f"All outputs:\n  {out}"
    )


if __name__ == "__main__":
    main()
