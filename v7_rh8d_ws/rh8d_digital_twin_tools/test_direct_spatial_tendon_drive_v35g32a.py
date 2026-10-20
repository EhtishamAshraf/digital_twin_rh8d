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
import test_spatial_prescribed_spool_v35g29b as g29b


# =====================================================================
# G32A — AERO-STYLE DIRECT SPATIAL-TENDON POSITION DRIVE
#
# PRESERVED:
#   - idx_flex_spatial remains a REAL <spatial> tendon
#   - G31A transverse spatial return springs remain
#   - independent MCP/PIP/DIP
#   - contact model unchanged
#   - bottle unchanged
#
# REMOVED FROM THE MECHANICAL CHAIN:
#   measured encoder -> simulated spool servo -> elastic tendon stretch
#
# NEW:
#   measured encoder -> identified tendon shortening
#                    -> position actuator on idx_flex_spatial
#
# Exactly the actuator topology used by Aero:
#
#       <position tendon="..." kp="10000"/>
#
# Main tendon stiffness is disabled because otherwise the spatial
# tendon would simultaneously be:
#
#   (1) a position-controlled cable
#   (2) a spring cable
#
# which double-counts its axial mechanics.
#
# The PASSIVE return springs remain spatial tendons.
# =====================================================================


MAIN_TENDON = "idx_flex_spatial"

OLD_SPOOL_ACTUATOR = "pos_index_tendon"
SPOOL_JOINT = "index_motor_spool"

DIRECT_ACTUATOR = "g32_index_tendon_position"

SCALE = 5.70e-6

OPEN_ENCODER = 100.0


ID_POINTS = [
    # encoder, MCP, PIP, DIP
    (416.1,   10.620, 3.994, 1.911),
    (611.7,   18.094, 4.526, 3.724),
    (673.573, 18.543, 6.161, 5.276),
]


HW_CONTACT_ENCODER = 673.573

HW_CONTACT_POSE = np.array(
    [
        18.543,
        6.161,
        5.276,
    ],
    dtype=float,
)


TRUSTED_FINAL = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


TARGET_TIME = 38.970486


RETURN_SPRINGS = [
    "idx_mcp_return_spring",
    "idx_pip_return_spring",
    "idx_dip_return_spring",
]


def nearest(values, target):

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


def top_level_tendon(root):

    element = root.find("./tendon")

    if element is None:

        raise RuntimeError(
            "Top-level <tendon> missing."
        )

    return element


def top_level_actuator(root):

    element = root.find("./actuator")

    if element is None:

        raise RuntimeError(
            "Top-level <actuator> missing."
        )

    return element


def find_spatial(
    root,
    name,
):

    tendon_root = top_level_tendon(
        root
    )

    for spatial in tendon_root.findall(
        "spatial"
    ):

        if spatial.get(
            "name"
        ) == name:

            return spatial

    raise RuntimeError(
        f"Spatial tendon not found: {name}"
    )


def find_geom(
    root,
    name,
):

    for geom in root.iter(
        "geom"
    ):

        if geom.get(
            "name"
        ) == name:

            return geom

    raise RuntimeError(
        f"Geom not found: {name}"
    )


def tendon_jacobian_row(
    model,
    data,
    tendon_id,
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
            tendon_id,
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
                tendon_id,
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
            tendon_id
        ]
    )

    adr = int(
        rowadr[
            tendon_id
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


def build_direct_model(
    source,
    destination,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    spatial = find_spatial(
        root,
        MAIN_TENDON,
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    #
    # Remove ONLY the artificial spool endpoint from the tendon.
    #
    # The exposed RH8D finger-side tendon remains spatial:
    #
    # palm entry -> MCP -> PIP -> DIP -> anchor
    #
    # Hidden palm/motor routing is now represented by the measured
    # tendon-length command rather than invented geometry.
    # --------------------------------------------------------------

    removed_spool_site = False

    for child in list(
        spatial
    ):

        if (
            child.tag == "site"
            and
            child.get(
                "site"
            )
            ==
            "v35_spool_site"
        ):

            spatial.remove(
                child
            )

            removed_spool_site = True

            break

    if not removed_spool_site:

        raise RuntimeError(
            "v35_spool_site not found in idx_flex_spatial."
        )

    # --------------------------------------------------------------
    # Main cable becomes position-controlled, Aero-style.
    #
    # Passive spring tendons are untouched.
    # --------------------------------------------------------------

    spatial.attrib.pop(
        "stiffness",
        None,
    )

    spatial.attrib.pop(
        "damping",
        None,
    )

    spatial.attrib.pop(
        "frictionloss",
        None,
    )

    spatial.attrib.pop(
        "springlength",
        None,
    )

    actuator_root = top_level_actuator(
        root
    )

    # Avoid accidental duplicate.
    for actuator in list(
        actuator_root
    ):

        if actuator.get(
            "name"
        ) == DIRECT_ACTUATOR:

            actuator_root.remove(
                actuator
            )

    ET.SubElement(
        actuator_root,
        "position",
        {
            "name":
                DIRECT_ACTUATOR,

            "tendon":
                MAIN_TENDON,

            # Aero hand uses kp=10000 for its directly
            # actuated spatial finger tendons.
            "kp":
                "10000",
        },
    )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )

    model = mujoco.MjModel.from_xml_path(
        str(
            destination
        )
    )

    return model


def verify_architecture(
    xml,
):

    tree = ET.parse(
        xml
    )

    root = tree.getroot()

    main = find_spatial(
        root,
        MAIN_TENDON,
    )

    sites = [
        child.get(
            "site"
        )
        for child in list(
            main
        )
        if child.tag == "site"
    ]

    if "v35_spool_site" in sites:

        raise RuntimeError(
            "Spool site still present in directly driven tendon."
        )

    model = mujoco.MjModel.from_xml_path(
        str(
            xml
        )
    )

    main_tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        MAIN_TENDON,
    )

    direct_aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        DIRECT_ACTUATOR,
    )

    if (
        main_tid < 0
        or
        direct_aid < 0
    ):

        raise RuntimeError(
            "Direct spatial-tendon architecture did not compile."
        )

    for spring_name in RETURN_SPRINGS:

        tid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            spring_name,
        )

        if tid < 0:

            raise RuntimeError(
                f"Return spring lost: {spring_name}"
            )

    print()
    print(
        "ARCHITECTURE CHECK: PASS"
    )

    print(
        "  idx_flex_spatial remains a REAL spatial tendon"
    )

    print(
        "  hidden spool endpoint removed from tendon path"
    )

    print(
        "  direct position actuator attached to idx_flex_spatial"
    )

    print(
        "  all 3 G31A spatial return springs remain"
    )


def fit_encoder_to_tendon_shortening(
    xml,
):

    model = mujoco.MjModel.from_xml_path(
        str(
            xml
        )
    )

    data = mujoco.MjData(
        model
    )

    refs = g12.get_joint_refs(
        model
    )

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        MAIN_TENDON,
    )

    data.qpos[:] = (
        model.qpos0
    )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    open_length = float(
        data.ten_length[
            tid
        ]
    )

    pulls = []
    shortenings = []

    print()
    print("=" * 150)
    print(
        "G32A STEP 1 — IDENTIFY ENCODER -> SPATIAL-TENDON SHORTENING"
    )
    print("=" * 150)

    print()
    print(
        "encoder      spool take-up      hardware tendon shortening"
    )

    print("-" * 90)

    for (
        encoder,
        mcp_deg,
        pip_deg,
        dip_deg,
    ) in ID_POINTS:

        data.qpos[:] = (
            model.qpos0
        )

        data.qvel[:] = 0.0

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

        length = float(
            data.ten_length[
                tid
            ]
        )

        shortening = (
            open_length
            -
            length
        )

        pull = (
            SCALE
            *
            (
                encoder
                -
                OPEN_ENCODER
            )
        )

        pulls.append(
            pull
        )

        shortenings.append(
            shortening
        )

        print(
            f"{encoder:8.3f}      "
            f"{1000.0*pull:9.4f} mm      "
            f"{1000.0*shortening:9.4f} mm"
        )

    pulls = np.asarray(
        pulls,
        dtype=float,
    )

    shortenings = np.asarray(
        shortenings,
        dtype=float,
    )

    denominator = float(
        np.dot(
            pulls,
            pulls,
        )
    )

    ratio = float(
        np.dot(
            pulls,
            shortenings,
        )
        /
        denominator
    )

    predicted = (
        ratio
        *
        pulls
    )

    residual_mm = (
        1000.0
        *
        (
            predicted
            -
            shortenings
        )
    )

    rmse_mm = float(
        np.sqrt(
            np.mean(
                residual_mm
                **
                2
            )
        )
    )

    print()
    print(
        f"identified shortening ratio = "
        f"{ratio:.6f}"
    )

    print(
        f"mapping residual RMSE       = "
        f"{rmse_mm:.4f} mm"
    )

    print(
        f"open finger tendon length   = "
        f"{1000.0*open_length:.4f} mm"
    )

    if not (
        0.25
        <=
        ratio
        <=
        1.25
    ):

        raise RuntimeError(
            "Encoder-to-tendon ratio is implausible; "
            "refusing dynamic run."
        )

    return (
        open_length,
        ratio,
        rmse_mm,
    )


class DirectSpatialTendonDrive:

    def __init__(
        self,
        open_length,
        shortening_ratio,
    ):

        self.original_step = (
            mujoco.mj_step
        )

        self.open_length = float(
            open_length
        )

        self.ratio = float(
            shortening_ratio
        )

        self.cache = {}

        self.history = []

    def setup(
        self,
        model,
    ):

        key = id(
            model
        )

        if key in self.cache:

            return self.cache[
                key
            ]

        old_aid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            OLD_SPOOL_ACTUATOR,
        )

        new_aid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            DIRECT_ACTUATOR,
        )

        spool_jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            SPOOL_JOINT,
        )

        tid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            MAIN_TENDON,
        )

        if (
            old_aid < 0
            or
            new_aid < 0
            or
            spool_jid < 0
            or
            tid < 0
        ):

            self.cache[
                key
            ] = None

            return None

        spool_qadr = int(
            model.jnt_qposadr[
                spool_jid
            ]
        )

        spool_dadr = int(
            model.jnt_dofadr[
                spool_jid
            ]
        )

        state = {
            "model":
                model,

            "old_aid":
                old_aid,

            "new_aid":
                new_aid,

            "tid":
                tid,

            "spool_qadr":
                spool_qadr,

            "spool_dadr":
                spool_dadr,

            "old_gain":
                model.actuator_gainprm[
                    old_aid
                ].copy(),

            "old_bias":
                model.actuator_biasprm[
                    old_aid
                ].copy(),

            "open_spool_target":
                None,

            "fixed_spool_q":
                float(
                    model.qpos0[
                        spool_qadr
                    ]
                ),
        }

        # Neutralize only the old dummy-spool servo.
        model.actuator_gainprm[
            old_aid,
            :
        ] = 0.0

        model.actuator_biasprm[
            old_aid,
            :
        ] = 0.0

        self.cache[
            key
        ] = state

        return state

    def step(
        self,
        model,
        data,
    ):

        state = self.setup(
            model
        )

        if state is None:

            return self.original_step(
                model,
                data,
            )

        old_aid = state[
            "old_aid"
        ]

        new_aid = state[
            "new_aid"
        ]

        measured_spool_target = float(
            data.ctrl[
                old_aid
            ]
        )

        if state[
            "open_spool_target"
        ] is None:

            state[
                "open_spool_target"
            ] = measured_spool_target

        pull = (
            measured_spool_target
            -
            state[
                "open_spool_target"
            ]
        )

        # H1 is a closing experiment.
        pull = max(
            0.0,
            pull,
        )

        tendon_target = (
            self.open_length
            -
            self.ratio
            *
            pull
        )

        data.ctrl[
            new_aid
        ] = tendon_target

        # Spool no longer belongs to the spatial tendon path.
        # Keep it mechanically neutral/fixed.
        data.qpos[
            state[
                "spool_qadr"
            ]
        ] = state[
            "fixed_spool_q"
        ]

        data.qvel[
            state[
                "spool_dadr"
            ]
        ] = 0.0

        self.original_step(
            model,
            data,
        )

        actual_length = float(
            data.ten_length[
                state[
                    "tid"
                ]
            ]
        )

        self.history.append({
            "time_s":
                float(
                    data.time
                ),

            "measured_spool_target_m":
                measured_spool_target,

            "pull_mm":
                1000.0
                *
                pull,

            "target_tendon_length_mm":
                1000.0
                *
                tendon_target,

            "actual_tendon_length_mm":
                1000.0
                *
                actual_length,

            "tracking_error_mm":
                1000.0
                *
                (
                    actual_length
                    -
                    tendon_target
                ),

            "direct_actuator_force":
                float(
                    data.actuator_force[
                        new_aid
                    ]
                ),
        })

    def install(
        self,
    ):

        mujoco.mj_step = (
            self.step
        )

    def restore(
        self,
    ):

        mujoco.mj_step = (
            self.original_step
        )

        for state in self.cache.values():

            if state is None:
                continue

            model = state[
                "model"
            ]

            old_aid = state[
                "old_aid"
            ]

            model.actuator_gainprm[
                old_aid,
                :
            ] = state[
                "old_gain"
            ]

            model.actuator_biasprm[
                old_aid,
                :
            ] = state[
                "old_bias"
            ]


def disable_fixture(
    source,
    destination,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    fixture = find_geom(
        root,
        g12.FIXTURE_NAME,
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

    mujoco.MjModel.from_xml_path(
        str(
            destination
        )
    )


def run_direct(
    xml,
    bag,
    replay,
    open_length,
    ratio,
):

    drive = DirectSpatialTendonDrive(
        open_length,
        ratio,
    )

    drive.install()

    try:

        result = g12.run_audit(
            xml,
            bag,
            replay,
        )

        history = list(
            drive.history
        )

    finally:

        drive.restore()

    return (
        result,
        history,
    )


def episodes(
    rows,
    key,
):

    flags = np.asarray(
        [
            bool(
                r[
                    key
                ]
            )
            for r in rows
        ],
        dtype=bool,
    )

    return (
        g12.contact_episode_starts(
            flags
        ),
        g12.contact_episode_ends(
            flags
        ),
    )


def save_history(
    path,
    history,
):

    if not history:
        return

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                history[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            history
        )


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
            "G31A identified transverse-spatial-spring model"
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
            "v35g32a_direct_spatial_tendon_drive"
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

    candidate = (
        out
        /
        "g32a_direct_spatial_tendon.xml"
    )

    free_xml = (
        out
        /
        "g32a_direct_spatial_tendon_free.xml"
    )

    print()
    print("=" * 150)
    print(
        "G32A — AERO-STYLE DIRECT RH8D SPATIAL-TENDON DRIVE"
    )
    print("=" * 150)

    build_direct_model(
        source,
        candidate,
    )

    verify_architecture(
        candidate
    )

    (
        open_length,
        ratio,
        ratio_rmse_mm,
    ) = fit_encoder_to_tendon_shortening(
        candidate
    )

    disable_fixture(
        candidate,
        free_xml,
    )

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

        # =========================================================
        # Approach validation
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G32A STEP 2 — H1 APPROACH, NO BOTTLE"
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
        ) = run_direct(
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

            i = nearest(
                enc,
                tick,
            )

            row = free_rows[
                i
            ]

            sim = np.array(
                [
                    row[
                        "mcp_deg"
                    ],
                    row[
                        "pip_deg"
                    ],
                    row[
                        "dip_deg"
                    ],
                ]
            )

            hw = np.array(
                [
                    hm,
                    hp,
                    hd,
                ]
            )

            error = sim - hw

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

        # =========================================================
        # Contact run
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G32A STEP 3 — ONE H1 CONTACT RUN"
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
        ) = run_direct(
            candidate,
            bag,
            replay,
            open_length,
            ratio,
        )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

    mid_starts, mid_ends = episodes(
        rows,
        "mid_contact",
    )

    tip_starts, tip_ends = episodes(
        rows,
        "tip_contact",
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

    first_pose = np.array(
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

    first_error = (
        first_pose
        -
        HW_CONTACT_POSE
    )

    first_rmse = float(
        np.sqrt(
            np.mean(
                first_error
                **
                2
            )
        )
    )

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
        "  error      = "
        f"{first_error[0]:+.3f} / "
        f"{first_error[1]:+.3f} / "
        f"{first_error[2]:+.3f}"
    )

    print(
        f"  pose RMSE  = "
        f"{first_rmse:.3f} deg"
    )

    print()
    print(
        "CONTACT CONTINUITY"
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

    # =============================================================
    # Trusted final pose
    # =============================================================

    row_times = np.asarray(
        [
            r[
                "time_s"
            ]
            for r in rows
        ],
        dtype=float,
    )

    ti = nearest(
        row_times,
        TARGET_TIME,
    )

    final = np.array(
        [
            rows[
                ti
            ][
                "mcp_deg"
            ],
            rows[
                ti
            ][
                "pip_deg"
            ],
            rows[
                ti
            ][
                "dip_deg"
            ],
        ],
        dtype=float,
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

    # =============================================================
    # Tendon actuator tracking
    # =============================================================

    track = np.asarray(
        [
            abs(
                x[
                    "tracking_error_mm"
                ]
            )
            for x in contact_history
        ],
        dtype=float,
    )

    mean_track = float(
        np.mean(
            track
        )
    )

    max_track = float(
        np.max(
            track
        )
    )

    force_abs = np.asarray(
        [
            abs(
                x[
                    "direct_actuator_force"
                ]
            )
            for x in contact_history
        ],
        dtype=float,
    )

    print()
    print(
        "DIRECT SPATIAL-TENDON ACTUATOR"
    )

    print(
        f"  mean tendon-length tracking error = "
        f"{mean_track:.4f} mm"
    )

    print(
        f"  max tendon-length tracking error  = "
        f"{max_track:.4f} mm"
    )

    print(
        f"  peak |actuator force|             = "
        f"{np.max(force_abs):.4f}"
    )

    # =============================================================
    # Save
    # =============================================================

    save_history(
        out
        /
        "g32a_free_tendon_drive.csv",
        free_history,
    )

    save_history(
        out
        /
        "g32a_contact_tendon_drive.csv",
        contact_history,
    )

    summary_csv = (
        out
        /
        "g32a_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shortening_ratio",
                "shortening_fit_rmse_mm",
                "approach_rmse_deg",
                "first_contact_encoder",
                "first_contact_encoder_error",
                "first_contact_pose_rmse_deg",
                "middle_episodes",
                "tip_episodes",
                "middle_gap_s",
                "final_rmse_deg",
                "mean_tendon_tracking_error_mm",
                "max_tendon_tracking_error_mm",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "shortening_ratio":
                ratio,

            "shortening_fit_rmse_mm":
                ratio_rmse_mm,

            "approach_rmse_deg":
                approach_rmse,

            "first_contact_encoder":
                first_row[
                    "encoder_tick"
                ],

            "first_contact_encoder_error":
                first_row[
                    "encoder_tick"
                ]
                -
                HW_CONTACT_ENCODER,

            "first_contact_pose_rmse_deg":
                first_rmse,

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

            "mean_tendon_tracking_error_mm":
                mean_track,

            "max_tendon_tracking_error_mm":
                max_track,
        })

    np.savez_compressed(
        out
        /
        "g32a_contact_trajectory.npz",
        times=np.asarray(
            times
        ),
        qpos=np.asarray(
            qpos
        ),
    )

    # =============================================================
    # Decision
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G32A DECISION"
    )
    print("=" * 150)

    print(
        f"approach RMSE         = "
        f"{approach_rmse:.3f} deg"
    )

    print(
        f"first-contact RMSE    = "
        f"{first_rmse:.3f} deg"
    )

    print(
        f"first-contact enc err = "
        f"{first_row['encoder_tick'] - HW_CONTACT_ENCODER:+.1f} ticks"
    )

    print(
        f"middle detached gap   = "
        f"{detached_gap:.3f} s"
    )

    print(
        f"final H1 RMSE         = "
        f"{final_rmse:.3f} deg"
    )

    print()

    if (
        approach_rmse <= 3.0
        and
        first_rmse <= 4.0
        and
        detached_gap <= 0.25
    ):

        print(
            "PASS: direct spatial-tendon actuation solves "
            "the major H1 mechanics."
        )

        print(
            "Freeze actuation architecture."
        )

    elif (
        approach_rmse <= 3.5
        and
        first_rmse <= 4.0
        and
        detached_gap > 0.25
    ):

        print(
            "PARTIAL PASS:"
        )

        print(
            "Aero-style tendon actuation reproduces the approach,"
        )

        print(
            "but contact retention is still wrong."
        )

        print(
            "Next and ONLY branch = compliant contact pad."
        )

    else:

        print(
            "REJECT direct-tendon actuation as the missing solution."
        )

        print(
            "Do not tune kp."
        )

        print(
            "Next branch = rebuild the RH8D flexor wrapping geometry "
            "from CAD before further H1 fitting."
        )

    print()
    print(
        f"Candidate XML:\n  "
        f"{candidate}"
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
            "g32a_view.xml"
        )

        g29b.make_view_xml(
            candidate,
            view_xml,
        )

        print()
        print("=" * 150)
        print(
            "G32A GUI — DIRECTLY ACTUATED SPATIAL TENDON"
        )
        print("=" * 150)

        print(
            "The index flexor must still be visible as a "
            "spatial tendon."
        )

        print(
            "G31A return springs remain visible."
        )

        g7.exact_gui_replay(
            view_xml,
            qpos,
            times,
            "G32A Aero-style direct spatial tendon drive",
        )


if __name__ == "__main__":
    main()
