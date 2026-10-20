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
# G26B — DATA-DRIVEN PASSIVE JOINT SPRING IDENTIFICATION
#
# PURPOSE
# -------
# Stop guessing PIP/DIP resistance.
#
# Use measured hardware FREE-SPACE closing configurations and solve
# the quasi-static torque balance:
#
#      tendon torque
#    + joint spring torque
#    - bias torque
#    ~= 0
#
# Model each physical transverse joint spring as:
#
#      tau_spring(q) = -(tau0 + k*q)
#
# which is exactly equivalent to a MuJoCo torsional spring:
#
#      tau = -k * (q - qref)
#
# with:
#
#      qref = -tau0/k
#
#
# IMPORTANT
# ---------
# - K_tendon = 6900 N/m is NOT declared final.
# - It is used because G23A empirically reduced the middle gap
#   from 1.577 s to 0.094 s.
# - We fit joint springs from hardware rather than hand-selecting
#   resistance.
# - No contact parameters are tuned.
# - No friction hierarchy.
# - No contact switching.
#
# After fitting, ONE dynamic H1 run is performed.
# =====================================================================


TENDON_K = 6900.0
SCALE = 5.70e-6

TENDON_NAME = "idx_flex_spatial"
FIXTURE_NAME = "index_contact_fixture"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

SPOOL_EXACT = "index_motor_spool"


# =====================================================================
# Previously measured REAL free-space closing data.
#
# DIP marker measurement is less reliable, therefore obviously
# negative DIP measurements are not used in the spring fit.
# =====================================================================

FREE_SPACE = [
    # tick,  MCP,     PIP,      DIP
    (100.0,    0.000,   0.000,   0.000),
    (500.0,   16.952,   0.968,  -0.428),
    (1000.0,  39.472,  -0.907,  -0.343),
    (1500.0,  57.291,   3.654,  -4.019),
    (2000.0,  73.632,   4.667,   2.012),
    (2500.0,  73.632,  27.119,   3.190),
    (3000.0,  77.347,  48.581,   7.810),
    (3500.0,  80.074,  70.872,  -7.199),
]


# H1 hardware pre-contact measurements for independent comparison.
H1_PRECONTACT = [
    # tick, MCP, PIP, DIP
    (416.1, 10.620, 3.994, 1.911),
    (611.7, 18.094, 4.526, 3.724),
]


HW_FIRST_CONTACT = np.array([
    18.543,
    6.161,
    5.276,
])

HW_FIRST_CONTACT_ENCODER = 673.573

TRUSTED_FINAL = np.array([
    14.5355,
    23.8110,
    9.3270,
])

TARGET_TIME = 38.970486


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


def find_joint_xml(root, name):

    for joint in root.iter("joint"):

        if joint.get("name") == name:
            return joint

    raise RuntimeError(
        f"Joint not found: {name}"
    )


def find_tendon_xml(root, name):

    for tendon in root.iter("spatial"):

        if tendon.get("name") == name:
            return tendon

    raise RuntimeError(
        f"Tendon not found: {name}"
    )


def find_geom_xml(root, name):

    for geom in root.iter("geom"):

        if geom.get("name") == name:
            return geom

    raise RuntimeError(
        f"Geom not found: {name}"
    )


def make_analysis_model(
    source,
    destination,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    tendon = find_tendon_xml(
        root,
        TENDON_NAME,
    )

    tendon.set(
        "stiffness",
        f"{TENDON_K:.12g}",
    )

    # Disable bottle while identifying FREE-SPACE passives.
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


def make_candidate_model(
    source,
    destination,
    fitted,
    free_space=False,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    tendon = find_tendon_xml(
        root,
        TENDON_NAME,
    )

    tendon.set(
        "stiffness",
        f"{TENDON_K:.12g}",
    )

    names = {
        "mcp": MCP_NAME,
        "pip": PIP_NAME,
        "dip": DIP_NAME,
    }

    for key, name in names.items():

        joint = find_joint_xml(
            root,
            name,
        )

        k = fitted[key]["k_Nm_rad"]
        qref = fitted[key]["qref_rad"]

        joint.set(
            "stiffness",
            f"{k:.12g}",
        )

        joint.set(
            "springref",
            f"{qref:.12g}",
        )

        # The closing-data torque has already been absorbed into
        # the fitted effective spring law.
        #
        # Remove dry-friction proxy to avoid double counting.
        joint.set(
            "frictionloss",
            "0",
        )

    if free_space:

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


def get_spool_joint(model):

    exact = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_EXACT,
    )

    if exact >= 0:
        return int(exact)

    candidates = []

    for jid in range(
        model.njnt
    ):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )

        if name is None:
            continue

        lower = name.lower()

        if (
            "spool" in lower
            and
            "index" in lower
        ):

            candidates.append(
                jid
            )

    if len(candidates) == 1:
        return candidates[0]

    print()
    print(
        "Available spool-like joint names:"
    )

    for jid in range(
        model.njnt
    ):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )

        if (
            name
            and
            (
                "spool" in name.lower()
                or
                "motor" in name.lower()
            )
        ):

            print(
                f"  {jid}: {name}"
            )

    raise RuntimeError(
        "Could not uniquely identify index spool joint."
    )


def tendon_jacobian_row(
    model,
    data,
    tid,
):

    raw = np.asarray(
        data.ten_J,
        dtype=float,
    )

    if (
        raw.ndim == 2
        and
        raw.shape
        ==
        (
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

        return (
            raw.reshape(
                model.ntendon,
                model.nv,
            )[
                tid,
                :
            ].copy()
        )

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

    for k in range(
        nnz
    ):

        idx = (
            adr
            +
            k
        )

        J[
            int(
                colind[
                    idx
                ]
            )
        ] = float(
            raw[
                idx
            ]
        )

    return J


def episode_indices(rows, key):

    flags = np.asarray(
        [
            bool(
                row[
                    key
                ]
            )
            for row in rows
        ],
        dtype=bool,
    )

    starts = g12.contact_episode_starts(
        flags
    )

    ends = g12.contact_episode_ends(
        flags
    )

    return starts, ends


def fit_spring(
    joint,
    samples,
):

    # Fit:
    #
    # -tau_required = tau0 + k*q
    #
    # with:
    #
    # tau0 >= 0
    # k    >= 0

    q = []
    y = []

    for sample in samples:

        q_deg = sample[
            f"{joint}_deg"
        ]

        tau_required = sample[
            f"{joint}_required_spring_Nm"
        ]

        # Remove negative-angle marker artifacts for PIP/DIP.
        if (
            joint in (
                "pip",
                "dip",
            )
            and
            q_deg < 0.0
        ):

            continue

        q.append(
            math.radians(
                q_deg
            )
        )

        y.append(
            -
            tau_required
        )

    q = np.asarray(
        q,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    X = np.column_stack([
        np.ones_like(
            q
        ),
        q,
    ])

    # ------------------------------------------------------------
    # Non-negative least squares for only TWO unknowns:
    #
    #     y ~= tau0 + k*q
    #
    # subject to:
    #
    #     tau0 >= 0
    #     k    >= 0
    #
    # For two variables the constrained optimum is either:
    #
    #   - unconstrained interior solution
    #   - tau0-only boundary
    #   - k-only boundary
    #   - origin
    #
    # Therefore no SciPy dependency is required.
    # ------------------------------------------------------------

    candidates = []

    unconstrained, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    if np.all(
        unconstrained >= 0.0
    ):

        candidates.append(
            unconstrained
        )

    # tau0 only
    tau_only = np.array(
        [
            max(
                float(
                    np.mean(
                        y
                    )
                ),
                0.0,
            ),
            0.0,
        ],
        dtype=float,
    )

    candidates.append(
        tau_only
    )

    # k only
    denom = float(
        np.dot(
            q,
            q,
        )
    )

    if denom > 1e-15:

        k_only_value = max(
            float(
                np.dot(
                    q,
                    y,
                )
                /
                denom
            ),
            0.0,
        )

    else:

        k_only_value = 0.0

    k_only = np.array(
        [
            0.0,
            k_only_value,
        ],
        dtype=float,
    )

    candidates.append(
        k_only
    )

    candidates.append(
        np.zeros(
            2,
            dtype=float,
        )
    )

    beta = min(
        candidates,
        key=lambda b:
            float(
                np.sum(
                    (
                        X @ b
                        -
                        y
                    )
                    **
                    2
                )
            ),
    )

    tau0 = float(
        beta[
            0
        ]
    )

    k = float(
        beta[
            1
        ]
    )

    prediction = (
        X
        @
        beta
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    prediction
                    -
                    y
                )
                **
                2
            )
        )
    )

    if k > 1e-10:

        qref = (
            -
            tau0
            /
            k
        )

    else:

        qref = 0.0

    return {
        "tau0_Nm":
            tau0,

        "k_Nm_rad":
            k,

        "qref_rad":
            qref,

        "qref_deg":
            math.degrees(
                qref
            ),

        "fit_rmse_Nm":
            rmse,

        "n":
            len(
                q
            ),
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
        help=(
            "G25B calibrated contact XML"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g26b_identified_joint_springs"
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

    analysis_xml = (
        out
        /
        "g26b_analysis_free.xml"
    )

    candidate_contact_xml = (
        out
        /
        "g26b_identified_contact.xml"
    )

    candidate_free_xml = (
        out
        /
        "g26b_identified_free.xml"
    )

    make_analysis_model(
        source_xml,
        analysis_xml,
    )

    # -------------------------------------------------------------
    # Runtime constants.
    # -------------------------------------------------------------

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    old_k = float(
        g7.b2.v35.TENDON_K
    )

    g7.b2.v35.METERS_PER_TICK = (
        SCALE
    )

    g7.b2.v35.TENDON_K = (
        TENDON_K
    )

    try:

        # =========================================================
        # 1. Recover spool motion versus encoder from real H1 input.
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G26B STEP 1 — RECOVER MOTOR SPOOL / ENCODER MAPPING"
        )
        print("=" * 150)

        (
            replay_rows,
            replay_qpos,
            replay_times,
            replay_summary,
        ) = g12.run_audit(
            analysis_xml,
            bag,
            replay,
        )

        model = mujoco.MjModel.from_xml_path(
            str(
                analysis_xml
            )
        )

        spool_jid = get_spool_joint(
            model
        )

        spool_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            spool_jid,
        )

        spool_qadr = int(
            model.jnt_qposadr[
                spool_jid
            ]
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

        qpos_array = np.asarray(
            replay_qpos,
            dtype=float,
        )

        spool = qpos_array[
            :,
            spool_qadr
        ]

        valid = np.isfinite(
            encoder
        ) & np.isfinite(
            spool
        )

        # Remove long duplicated hold plateaus from regression.
        enc_v = encoder[
            valid
        ]

        spool_v = spool[
            valid
        ]

        order = np.argsort(
            enc_v
        )

        enc_v = enc_v[
            order
        ]

        spool_v = spool_v[
            order
        ]

        unique_enc, unique_index = np.unique(
            np.round(
                enc_v,
                3
            ),
            return_index=True,
        )

        enc_fit = enc_v[
            unique_index
        ]

        spool_fit = spool_v[
            unique_index
        ]

        slope, intercept = np.polyfit(
            enc_fit,
            spool_fit,
            1,
        )

        prediction = (
            intercept
            +
            slope
            *
            enc_fit
        )

        ss_res = float(
            np.sum(
                (
                    spool_fit
                    -
                    prediction
                )
                **
                2
            )
        )

        ss_tot = float(
            np.sum(
                (
                    spool_fit
                    -
                    np.mean(
                        spool_fit
                    )
                )
                **
                2
            )
        )

        r2 = (
            1.0
            -
            ss_res
            /
            max(
                ss_tot,
                1e-15,
            )
        )

        print(
            f"spool joint = "
            f"{spool_name}"
        )

        print(
            "spool qpos ~= "
            f"{intercept:+.9e} "
            f"+ ({slope:+.9e}) * encoder"
        )

        print(
            f"linear mapping R^2 = "
            f"{r2:.9f}"
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "  This H1 spool/encoder regression is diagnostic only."
        )

        print(
            "  It will NOT be extrapolated to the 3500-tick "
            "free-space experiment."
        )

        print(
            "  Free-space spool displacement uses the already "
            "identified effective scale:"
        )

        print(
            f"      {SCALE:.8e} m/tick"
        )

        # Reference complete qpos from open state.
        open_index = nearest_index(
            encoder,
            100.0,
        )

        q_reference = (
            qpos_array[
                open_index
            ].copy()
        )

        open_encoder = float(
            encoder[
                open_index
            ]
        )

        open_spool_qpos = float(
            q_reference[
                spool_qadr
            ]
        )

        print()
        print(
            "DIRECT FREE-SPACE SPOOL MAPPING:"
        )

        print(
            f"  open encoder      = "
            f"{open_encoder:.3f} ticks"
        )

        print(
            f"  open spool qpos   = "
            f"{open_spool_qpos:+.9f} m"
        )

        print(
            f"  displacement/tick = "
            f"{SCALE:.9e} m/tick"
        )

        # =========================================================
        # 2. Inverse static torque requirements at measured
        #    free-space poses.
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G26B STEP 2 — INVERSE PASSIVE TORQUE FROM REAL FREE SPACE"
        )
        print("=" * 150)

        data = mujoco.MjData(
            model
        )

        refs = g12.get_joint_refs(
            model
        )

        tendon_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            TENDON_NAME,
        )

        spring_reference = (
            g12.tendon_spring_reference(
                model,
                tendon_id,
            )
        )

        samples = []

        print()
        print(
            "tick      MCP      PIP      DIP    "
            "F_tendon   reqMCP     reqPIP     reqDIP"
        )

        print("-" * 150)

        for (
            tick,
            mcp_deg,
            pip_deg,
            dip_deg,
        ) in FREE_SPACE:

            data.qpos[:] = (
                q_reference
            )

            data.qvel[:] = 0.0

            # ----------------------------------------------------
            # Effective encoder -> tendon/spool displacement.
            #
            # This is the already identified scale used throughout
            # the digital-twin experiments:
            #
            #     delta_spool =
            #         SCALE * (encoder - open_encoder)
            #
            # Do NOT extrapolate the H1 dynamic actuator regression.
            # ----------------------------------------------------

            data.qpos[
                spool_qadr
            ] = (
                open_spool_qpos
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

            tendon_J = tendon_jacobian_row(
                model,
                data,
                tendon_id,
            )

            extension = (
                g12.signed_extension(
                    float(
                        data.ten_length[
                            tendon_id
                        ]
                    ),
                    spring_reference,
                )
            )

            tendon_force = (
                TENDON_K
                *
                extension
            )

            sample = {
                "encoder_tick":
                    tick,

                "mcp_deg":
                    mcp_deg,

                "pip_deg":
                    pip_deg,

                "dip_deg":
                    dip_deg,

                "tendon_extension_mm":
                    1000.0
                    *
                    extension,

                "tendon_force_N":
                    tendon_force,
            }

            for joint in [
                "mcp",
                "pip",
                "dip",
            ]:

                dadr = refs[
                    joint
                ][
                    "dadr"
                ]

                moment = float(
                    tendon_J[
                        dadr
                    ]
                )

                tendon_tau = (
                    -
                    tendon_force
                    *
                    moment
                )

                bias = float(
                    data.qfrc_bias[
                        dadr
                    ]
                )

                # Static:
                #
                # tendon_tau
                # + spring_tau
                # - bias
                # = 0
                #
                spring_required = (
                    bias
                    -
                    tendon_tau
                )

                sample[
                    f"{joint}_moment_arm_mm"
                ] = (
                    1000.0
                    *
                    abs(
                        moment
                    )
                )

                sample[
                    f"{joint}_tendon_tau_Nm"
                ] = tendon_tau

                sample[
                    f"{joint}_bias_Nm"
                ] = bias

                sample[
                    f"{joint}_required_spring_Nm"
                ] = spring_required

            samples.append(
                sample
            )

            print(
                f"{tick:5.0f} "
                f"{mcp_deg:+8.3f} "
                f"{pip_deg:+8.3f} "
                f"{dip_deg:+8.3f} "
                f"{tendon_force:+9.4f} "
                f"{sample['mcp_required_spring_Nm']:+10.5f} "
                f"{sample['pip_required_spring_Nm']:+10.5f} "
                f"{sample['dip_required_spring_Nm']:+10.5f}"
            )

        # Save inverse table.
        inverse_csv = (
            out
            /
            "g26b_inverse_free_space.csv"
        )

        with inverse_csv.open(
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    samples[
                        0
                    ].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                samples
            )

        # =========================================================
        # 3. Fit preload + stiffness.
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G26B STEP 3 — IDENTIFIED EFFECTIVE TRANSVERSE SPRINGS"
        )
        print("=" * 150)

        fitted = {}

        for joint in [
            "mcp",
            "pip",
            "dip",
        ]:

            fitted[
                joint
            ] = fit_spring(
                joint,
                samples,
            )

            p = fitted[
                joint
            ]

            print()
            print(
                joint.upper()
            )

            print(
                f"  preload tau0 = "
                f"{p['tau0_Nm']:.6f} Nm"
            )

            print(
                f"  stiffness    = "
                f"{p['k_Nm_rad']:.6f} Nm/rad"
            )

            print(
                f"  springref    = "
                f"{p['qref_deg']:.3f} deg"
            )

            print(
                f"  fit RMSE     = "
                f"{p['fit_rmse_Nm']:.6f} Nm"
            )

            print(
                f"  samples      = "
                f"{p['n']}"
            )

        # Sanity stop against absurd identification.
        for joint, p in fitted.items():

            if (
                p[
                    "k_Nm_rad"
                ]
                >
                2.0
            ):

                raise RuntimeError(
                    f"{joint} fitted stiffness exceeds "
                    f"2 Nm/rad; refusing dynamic test."
                )

            if (
                abs(
                    p[
                        "qref_deg"
                    ]
                )
                >
                180.0
            ):

                raise RuntimeError(
                    f"{joint} fitted springref is physically "
                    f"implausible; refusing dynamic test."
                )

        # =========================================================
        # 4. Build one candidate.
        # =========================================================

        make_candidate_model(
            source_xml,
            candidate_contact_xml,
            fitted,
            free_space=False,
        )

        make_candidate_model(
            source_xml,
            candidate_free_xml,
            fitted,
            free_space=True,
        )

        mujoco.MjModel.from_xml_path(
            str(
                candidate_contact_xml
            )
        )

        mujoco.MjModel.from_xml_path(
            str(
                candidate_free_xml
            )
        )

        # =========================================================
        # 5. Dynamic PRE-CONTACT validation.
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G26B STEP 4 — DYNAMIC PRE-CONTACT VALIDATION"
        )
        print("=" * 150)

        (
            free_rows,
            free_qpos,
            free_times,
            free_summary,
        ) = g12.run_audit(
            candidate_free_xml,
            bag,
            replay,
        )

        free_encoder = np.asarray(
            [
                row[
                    "encoder_tick"
                ]
                for row in free_rows
            ],
            dtype=float,
        )

        pre_errors = []

        print()
        print(
            "encoder      HARDWARE MCP/PIP/DIP        "
            "SIM MCP/PIP/DIP             error"
        )

        print("-" * 150)

        for (
            tick,
            hm,
            hp,
            hd,
        ) in H1_PRECONTACT:

            i = nearest_index(
                free_encoder,
                tick,
            )

            row = free_rows[
                i
            ]

            sim = np.array([
                row[
                    "mcp_deg"
                ],
                row[
                    "pip_deg"
                ],
                row[
                    "dip_deg"
                ],
            ])

            hw = np.array([
                hm,
                hp,
                hd,
            ])

            error = (
                sim
                -
                hw
            )

            pre_errors.extend(
                error.tolist()
            )

            print(
                f"{tick:7.1f}   "
                f"{hm:7.2f}/{hp:7.2f}/{hd:7.2f}      "
                f"{sim[0]:7.2f}/{sim[1]:7.2f}/{sim[2]:7.2f}      "
                f"{error[0]:+6.2f}/{error[1]:+6.2f}/{error[2]:+6.2f}"
            )

        pre_rmse = float(
            np.sqrt(
                np.mean(
                    np.asarray(
                        pre_errors
                    )
                    **
                    2
                )
            )
        )

        print()
        print(
            f"pre-contact RMSE = "
            f"{pre_rmse:.3f} deg"
        )

        # Candidate state near hardware first-contact encoder,
        # still with bottle disabled.
        fc_i = nearest_index(
            free_encoder,
            HW_FIRST_CONTACT_ENCODER,
        )

        fc_free = free_rows[
            fc_i
        ]

        print()
        print(
            "candidate FREE finger at hardware "
            "first-contact encoder:"
        )

        print(
            f"  encoder = "
            f"{fc_free['encoder_tick']:.3f}"
        )

        print(
            "  simulation = "
            f"{fc_free['mcp_deg']:.3f} / "
            f"{fc_free['pip_deg']:.3f} / "
            f"{fc_free['dip_deg']:.3f}"
        )

        print(
            "  hardware   = "
            f"{HW_FIRST_CONTACT[0]:.3f} / "
            f"{HW_FIRST_CONTACT[1]:.3f} / "
            f"{HW_FIRST_CONTACT[2]:.3f}"
        )

        # =========================================================
        # 6. ONE H1 contact run.
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G26B STEP 5 — ONE H1 CONTACT RUN"
        )
        print("=" * 150)

        (
            rows,
            qpos,
            times,
            summary,
        ) = g12.run_audit(
            candidate_contact_xml,
            bag,
            replay,
        )

        mid_starts, mid_ends = (
            episode_indices(
                rows,
                "mid_contact",
            )
        )

        tip_starts, tip_ends = (
            episode_indices(
                rows,
                "tip_contact",
            )
        )

        if not mid_starts:

            raise RuntimeError(
                "Candidate produced no middle contact."
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

        first_duration = (
            rows[
                first_end
            ][
                "time_s"
            ]
            -
            first_row[
                "time_s"
            ]
        )

        if len(
            mid_starts
        ) >= 2:

            detached_gap = (
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

            detached_gap = 0.0

        print()
        print(
            "FIRST MIDDLE CONTACT:"
        )

        print(
            f"  encoder = "
            f"{first_row['encoder_tick']:.3f}"
        )

        print(
            "  simulation pose = "
            f"{first_row['mcp_deg']:.3f} / "
            f"{first_row['pip_deg']:.3f} / "
            f"{first_row['dip_deg']:.3f}"
        )

        print(
            "  hardware pose   = "
            f"{HW_FIRST_CONTACT[0]:.3f} / "
            f"{HW_FIRST_CONTACT[1]:.3f} / "
            f"{HW_FIRST_CONTACT[2]:.3f}"
        )

        print(
            f"  encoder error = "
            f"{first_row['encoder_tick'] - HW_FIRST_CONTACT_ENCODER:+.3f}"
        )

        print()

        print(
            "CONTACT CONTINUITY:"
        )

        print(
            f"  middle episodes = "
            f"{len(mid_starts)}"
        )

        print(
            f"  tip episodes    = "
            f"{len(tip_starts)}"
        )

        print(
            f"  first duration  = "
            f"{first_duration:.3f} s"
        )

        print(
            f"  detached gap    = "
            f"{detached_gap:.3f} s"
        )

        # ---------------------------------------------------------
        # Trusted final pose.
        # ---------------------------------------------------------

        row_times = np.asarray(
            [
                row[
                    "time_s"
                ]
                for row in rows
            ],
            dtype=float,
        )

        target_index = nearest_index(
            row_times,
            TARGET_TIME,
        )

        target = rows[
            target_index
        ]

        sim_final = np.array([
            target[
                "mcp_deg"
            ],
            target[
                "pip_deg"
            ],
            target[
                "dip_deg"
            ],
        ])

        final_error = (
            sim_final
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
            "TRUSTED FINAL H1:"
        )

        print(
            "  simulation = "
            f"{sim_final[0]:.3f} / "
            f"{sim_final[1]:.3f} / "
            f"{sim_final[2]:.3f}"
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
            f"  RMSE = "
            f"{final_rmse:.3f} deg"
        )

        # =========================================================
        # DECISION.
        # =========================================================

        first_pose = np.array([
            first_row[
                "mcp_deg"
            ],
            first_row[
                "pip_deg"
            ],
            first_row[
                "dip_deg"
            ],
        ])

        contact_pose_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        first_pose
                        -
                        HW_FIRST_CONTACT
                    )
                    **
                    2
                )
            )
        )

        encoder_error = abs(
            first_row[
                "encoder_tick"
            ]
            -
            HW_FIRST_CONTACT_ENCODER
        )

        print()
        print("=" * 150)
        print(
            "G26B DECISION"
        )
        print("=" * 150)

        print(
            f"pre-contact RMSE       = "
            f"{pre_rmse:.3f} deg"
        )

        print(
            f"first-contact pose RMSE= "
            f"{contact_pose_rmse:.3f} deg"
        )

        print(
            f"first-contact enc error= "
            f"{encoder_error:.1f} ticks"
        )

        print(
            f"middle detached gap    = "
            f"{detached_gap:.3f} s"
        )

        print(
            f"final H1 RMSE           = "
            f"{final_rmse:.3f} deg"
        )

        print()

        if (
            pre_rmse <= 3.0
            and
            contact_pose_rmse <= 4.0
            and
            encoder_error <= 60.0
            and
            len(
                mid_starts
            )
            ==
            1
        ):

            print(
                "RESULT: IDENTIFIED SPRING MODEL PASSES "
                "THE MECHANICAL TEST."
            )

            print(
                "Freeze index architecture and proceed "
                "to held-out H2."
            )

        elif (
            pre_rmse <= 4.0
            and
            contact_pose_rmse <= 5.0
        ):

            print(
                "RESULT: PRE-CONTACT MECHANICS ARE NOW "
                "SUBSTANTIALLY CORRECT."
            )

            print(
                "If contact still breaks, next step is "
                "tendon load/compliance identification only."
            )

            print(
                "Do NOT alter passive springs again."
            )

        else:

            print(
                "RESULT: A SINGLE LINEAR TORSIONAL-SPRING "
                "MODEL IS INSUFFICIENT."
            )

            print(
                "Next model class must be nonlinear spring "
                "geometry, not another parameter scan."
            )

        # =========================================================
        # Save compact summary.
        # =========================================================

        summary_csv = (
            out
            /
            "g26b_summary.csv"
        )

        with summary_csv.open(
            "w",
            newline="",
        ) as f:

            fields = [
                "mcp_k",
                "mcp_qref_deg",
                "pip_k",
                "pip_qref_deg",
                "dip_k",
                "dip_qref_deg",
                "spool_encoder_r2",
                "precontact_rmse_deg",
                "first_contact_encoder",
                "first_contact_encoder_error",
                "first_contact_pose_rmse_deg",
                "middle_episodes",
                "tip_episodes",
                "middle_gap_s",
                "final_rmse_deg",
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()

            writer.writerow({
                "mcp_k":
                    fitted[
                        "mcp"
                    ][
                        "k_Nm_rad"
                    ],

                "mcp_qref_deg":
                    fitted[
                        "mcp"
                    ][
                        "qref_deg"
                    ],

                "pip_k":
                    fitted[
                        "pip"
                    ][
                        "k_Nm_rad"
                    ],

                "pip_qref_deg":
                    fitted[
                        "pip"
                    ][
                        "qref_deg"
                    ],

                "dip_k":
                    fitted[
                        "dip"
                    ][
                        "k_Nm_rad"
                    ],

                "dip_qref_deg":
                    fitted[
                        "dip"
                    ][
                        "qref_deg"
                    ],

                "spool_encoder_r2":
                    r2,

                "precontact_rmse_deg":
                    pre_rmse,

                "first_contact_encoder":
                    first_row[
                        "encoder_tick"
                    ],

                "first_contact_encoder_error":
                    first_row[
                        "encoder_tick"
                    ]
                    -
                    HW_FIRST_CONTACT_ENCODER,

                "first_contact_pose_rmse_deg":
                    contact_pose_rmse,

                "middle_episodes":
                    len(
                        mid_starts
                    ),

                "tip_episodes":
                    len(
                        tip_starts
                    ),

                "middle_gap_s":
                    detached_gap,

                "final_rmse_deg":
                    final_rmse,
            })

        np.savez_compressed(
            out
            /
            "g26b_candidate_trajectory.npz",
            times=np.asarray(
                times
            ),
            qpos=np.asarray(
                qpos
            ),
        )

        print()
        print(
            f"Inverse table:\n  {inverse_csv}"
        )

        print()
        print(
            f"Candidate XML:\n  {candidate_contact_xml}"
        )

        print()
        print(
            f"Summary:\n  {summary_csv}"
        )

        print()
        print(
            f"All outputs:\n  {out}"
        )

        # =========================================================
        # GUI.
        # =========================================================

        if not args.no_gui:

            print()
            print("=" * 150)
            print(
                "G26B GUI — IDENTIFIED SPRING CANDIDATE"
            )
            print("=" * 150)

            print(
                "Look for the REAL sequence:"
            )

            print(
                "  MCP dominates first"
            )

            print(
                "  -> middle contacts with PIP/DIP still modest"
            )

            print(
                "  -> middle stays constrained"
            )

            print(
                "  -> PIP continues wrapping"
            )

            print(
                "  -> distal contacts"
            )

            print(
                "  -> DIP participates"
            )

            g7.exact_gui_replay(
                candidate_contact_xml,
                qpos,
                times,
                "G26B identified transverse-spring candidate",
            )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

        g7.b2.v35.TENDON_K = (
            old_k
        )


if __name__ == "__main__":
    main()
