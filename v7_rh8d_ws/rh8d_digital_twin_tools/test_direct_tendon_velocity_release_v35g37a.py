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
import test_direct_spatial_tendon_drive_v35g32a as g32
import test_spatial_prescribed_spool_v35g29b as g29b


# =====================================================================
# G37A
#
# Combine two PREVIOUSLY SUPPORTED effects:
#
#   G36:
#       high slow-speed PIP/DIP resistance gives good staging.
#
#   G30:
#       releasing that resistance during sliding restores later motion.
#
#   G32/G35:
#       direct spatial-tendon actuation eliminates middle-contact bounce.
#
# No parameter scan.
# =====================================================================


JOINTS = {
    "mcp":
        "Index_Proximal--palm_assembly:1",

    "pip":
        "Index_Middle--Index_Proximal",

    "dip":
        "Index_Distal--Index_Middle",
}


# MCP was already the useful G20C/G36 value.
MCP_FRICTION = 0.00035


# PIP/DIP:
#
# Fs = slow / breakaway capacity
# Fc = sliding capacity
#
PIP_FS = 0.00300
PIP_FC = 0.00125

DIP_FS = 0.00600
DIP_FC = 0.00000


# Reuse G30 diagnostic transition scale.
VS_DEG_S = 2.0
VS_RAD_S = math.radians(
    VS_DEG_S
)


SCALE = 5.70e-6


H1_POINTS = [
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


def nearest(
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


def find_joint_xml(
    root,
    name,
):

    for joint in root.iter(
        "joint"
    ):

        if joint.get(
            "name"
        ) == name:

            return joint

    raise RuntimeError(
        f"Joint not found: {name}"
    )


def prepare_candidate(
    source,
    destination,
):

    # -------------------------------------------------------------
    # First create clean G32 direct spatial-tendon architecture.
    # -------------------------------------------------------------

    temp = (
        destination.parent
        /
        "g37a_direct_temp.xml"
    )

    g32.build_direct_model(
        source,
        temp,
    )

    tree = ET.parse(
        temp
    )

    root = tree.getroot()

    # -------------------------------------------------------------
    # MCP constant low resistance.
    #
    # PIP/DIP initial XML values do not define the final model;
    # they are overwritten continuously by the velocity-dependent
    # passive law during dynamic simulation.
    # -------------------------------------------------------------

    find_joint_xml(
        root,
        JOINTS[
            "mcp"
        ],
    ).set(
        "frictionloss",
        f"{MCP_FRICTION:.12g}",
    )

    find_joint_xml(
        root,
        JOINTS[
            "pip"
        ],
    ).set(
        "frictionloss",
        f"{PIP_FS:.12g}",
    )

    find_joint_xml(
        root,
        JOINTS[
            "dip"
        ],
    ).set(
        "frictionloss",
        f"{DIP_FS:.12g}",
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

    # -------------------------------------------------------------
    # Architecture gates.
    # -------------------------------------------------------------

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        "idx_flex_spatial",
    )

    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "g32_index_tendon_position",
    )

    if (
        tid < 0
        or
        aid < 0
    ):

        raise RuntimeError(
            "Direct spatial-tendon architecture missing."
        )

    for name in RETURN_SPRINGS:

        rid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            name,
        )

        if rid >= 0:

            raise RuntimeError(
                f"STOP: unwanted G31/G33 spring exists: {name}"
            )

    print()
    print(
        "G37A ARCHITECTURE CHECK: PASS"
    )

    print(
        "  idx_flex_spatial preserved"
    )

    print(
        "  direct tendon drive preserved"
    )

    print(
        "  no G31/G33 added springs"
    )

    print(
        f"  MCP friction = "
        f"{MCP_FRICTION:.6f} Nm"
    )

    print(
        f"  PIP Fs/Fc    = "
        f"{PIP_FS:.6f} / "
        f"{PIP_FC:.6f} Nm"
    )

    print(
        f"  DIP Fs/Fc    = "
        f"{DIP_FS:.6f} / "
        f"{DIP_FC:.6f} Nm"
    )

    print(
        f"  velocity scale = "
        f"{VS_DEG_S:.3f} deg/s"
    )


class VelocityReleasePassive:

    """
    Passive velocity-dependent dry-friction capacity.

    High capacity close to zero speed.
    Low capacity during sliding.

        F(v) =
            Fc
            +
            (Fs-Fc)
            exp(-(abs(v)/vs)^2)

    No contact signal is used.
    No joint position signal is used.
    """

    def __init__(self):

        self.original_step = (
            mujoco.mj_step
        )

        self.cache = {}

        self.history = []

    @staticmethod
    def capacity(
        velocity,
        Fs,
        Fc,
    ):

        return (
            Fc
            +
            (
                Fs
                -
                Fc
            )
            *
            math.exp(
                -(
                    abs(
                        velocity
                    )
                    /
                    VS_RAD_S
                )
                **
                2
            )
        )

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

        result = {}

        for (
            label,
            Fs,
            Fc,
        ) in [
            (
                "pip",
                PIP_FS,
                PIP_FC,
            ),
            (
                "dip",
                DIP_FS,
                DIP_FC,
            ),
        ]:

            jid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                JOINTS[
                    label
                ],
            )

            if jid < 0:

                self.cache[
                    key
                ] = None

                return None

            dadr = int(
                model.jnt_dofadr[
                    jid
                ]
            )

            result[
                label
            ] = {
                "dadr":
                    dadr,

                "Fs":
                    Fs,

                "Fc":
                    Fc,

                "original":
                    float(
                        model.dof_frictionloss[
                            dadr
                        ]
                    ),
            }

        self.cache[
            key
        ] = {
            "model":
                model,

            "joints":
                result,
        }

        return self.cache[
            key
        ]

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

        row = {
            "time_s":
                float(
                    data.time
                ),
        }

        for (
            label,
            joint,
        ) in state[
            "joints"
        ].items():

            dadr = joint[
                "dadr"
            ]

            velocity = float(
                data.qvel[
                    dadr
                ]
            )

            friction = (
                self.capacity(
                    velocity,
                    joint[
                        "Fs"
                    ],
                    joint[
                        "Fc"
                    ],
                )
            )

            model.dof_frictionloss[
                dadr
            ] = friction

            row[
                f"{label}_velocity_deg_s"
            ] = math.degrees(
                velocity
            )

            row[
                f"{label}_friction_Nm"
            ] = friction

        self.history.append(
            row
        )

        return self.original_step(
            model,
            data,
        )

    def install(
        self,
    ):

        mujoco.mj_step = (
            self.step
        )

    def restore(
        self,
    ):

        for state in self.cache.values():

            if state is None:
                continue

            model = state[
                "model"
            ]

            for joint in state[
                "joints"
            ].values():

                model.dof_frictionloss[
                    joint[
                        "dadr"
                    ]
                ] = joint[
                    "original"
                ]

        mujoco.mj_step = (
            self.original_step
        )


def run_case(
    xml,
    bag,
    replay,
    open_length,
    ratio,
):

    # -------------------------------------------------------------
    # Installation order:
    #
    # velocity passive first
    # direct tendon wrapper second
    #
    # Therefore direct drive sets its control and calls the
    # velocity-dependent passive wrapper before the real mj_step.
    # -------------------------------------------------------------

    passive = (
        VelocityReleasePassive()
    )

    passive.install()

    drive = (
        g32.DirectSpatialTendonDrive(
            open_length,
            ratio,
        )
    )

    drive.install()

    try:

        result = g12.run_audit(
            xml,
            bag,
            replay,
        )

        passive_history = list(
            passive.history
        )

        drive_history = list(
            drive.history
        )

    finally:

        drive.restore()
        passive.restore()

    return (
        result,
        passive_history,
        drive_history,
    )


def episodes(
    rows,
    key,
):

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

    return (
        g12.contact_episode_starts(
            flags
        ),
        g12.contact_episode_ends(
            flags
        ),
    )


def save_rows(
    path,
    rows,
):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
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
        help="G25B baseline spatial-tendon model",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g37a_direct_velocity_release"
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
        "g37a_direct_velocity_release.xml"
    )

    free_xml = (
        out
        /
        "g37a_free.xml"
    )

    print()
    print("=" * 150)
    print(
        "G37A — DIRECT SPATIAL TENDON + VELOCITY-RELEASE PASSIVE"
    )
    print("=" * 150)

    prepare_candidate(
        source,
        candidate,
    )

    (
        open_length,
        ratio,
        mapping_rmse,
    ) = (
        g32.fit_encoder_to_tendon_shortening(
            candidate
        )
    )

    print()
    print(
        f"shortening ratio = "
        f"{ratio:.6f}"
    )

    print(
        f"mapping RMSE     = "
        f"{mapping_rmse:.4f} mm"
    )

    g32.disable_fixture(
        candidate,
        free_xml,
    )

    # -------------------------------------------------------------
    # Replay.
    # -------------------------------------------------------------

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
        # STEP 1 — approach
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G37A STEP 1 — H1 APPROACH"
        )
        print("=" * 150)

        (
            (
                free_rows,
                free_qpos,
                free_times,
                free_summary,
            ),
            free_passive,
            free_drive,
        ) = run_case(
            free_xml,
            bag,
            replay,
            open_length,
            ratio,
        )

        encoder = np.asarray(
            [
                row[
                    "encoder_tick"
                ]
                for row in free_rows
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
        ) in H1_POINTS:

            i = nearest(
                encoder,
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

        # =========================================================
        # STEP 2 — contact
        # =========================================================

        print()
        print("=" * 150)
        print(
            "G37A STEP 2 — ONE H1 CONTACT RUN"
        )
        print("=" * 150)

        (
            (
                rows,
                qpos,
                times,
                summary,
            ),
            contact_passive,
            contact_drive,
        ) = run_case(
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
        f"  detached gap    = "
        f"{gap:.3f} s"
    )

    # =============================================================
    # Final H1
    # =============================================================

    row_times = np.asarray(
        [
            row[
                "time_s"
            ]
            for row in rows
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

    recruitment = (
        final
        -
        first_pose
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

    print()
    print(
        "POST-CONTACT RECRUITMENT"
    )

    print(
        f"  dMCP = "
        f"{recruitment[0]:+.3f} deg"
    )

    print(
        f"  dPIP = "
        f"{recruitment[1]:+.3f} deg"
    )

    print(
        f"  dDIP = "
        f"{recruitment[2]:+.3f} deg"
    )

    # =============================================================
    # What friction actually did around final H1.
    # =============================================================

    if contact_passive:

        # Histories are MuJoCo-clock based, therefore use the
        # final recorded passive sample as a simple diagnostic only.
        pf = contact_passive[
            -1
        ]

        print()
        print(
            "FINAL VELOCITY-RELEASE STATE"
        )

        print(
            f"  PIP velocity = "
            f"{pf['pip_velocity_deg_s']:+.3f} deg/s"
        )

        print(
            f"  PIP friction = "
            f"{pf['pip_friction_Nm']:.6f} Nm"
        )

        print(
            f"  DIP velocity = "
            f"{pf['dip_velocity_deg_s']:+.3f} deg/s"
        )

        print(
            f"  DIP friction = "
            f"{pf['dip_friction_Nm']:.6f} Nm"
        )

    # =============================================================
    # Direct actuator tracking diagnostic.
    # =============================================================

    tracking = np.asarray(
        [
            abs(
                row[
                    "tracking_error_mm"
                ]
            )
            for row in contact_drive
        ],
        dtype=float,
    )

    mean_tracking = float(
        np.mean(
            tracking
        )
    )

    max_tracking = float(
        np.max(
            tracking
        )
    )

    print()
    print(
        "DIRECT TENDON TRACKING"
    )

    print(
        f"  mean error = "
        f"{mean_tracking:.4f} mm"
    )

    print(
        f"  max error  = "
        f"{max_tracking:.4f} mm"
    )

    # =============================================================
    # Hard decision.
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G37A DECISION"
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
        f"middle episodes       = "
        f"{len(mid_starts)}"
    )

    print(
        f"middle gap            = "
        f"{gap:.3f} s"
    )

    print(
        f"final RMSE            = "
        f"{final_rmse:.3f} deg"
    )

    print(
        f"post-contact dPIP     = "
        f"{recruitment[1]:+.3f} deg"
    )

    print()

    if (
        approach_rmse <= 3.5
        and
        first_rmse <= 4.0
        and
        len(
            mid_starts
        ) == 1
        and
        gap <= 0.25
        and
        final_rmse <= 3.0
    ):

        print(
            "PASS:"
        )

        print(
            "velocity-release passive + direct spatial tendon "
            "reproduces the important H1 mechanics."
        )

        print(
            "FREEZE this architecture."
        )

        print(
            "Next = full free-space validation and H2 "
            "without retuning."
        )

    elif (
        len(
            mid_starts
        ) == 1
        and
        recruitment[
            1
        ] >= 8.0
        and
        final_rmse <= 4.0
    ):

        print(
            "STRONG PARTIAL PASS:"
        )

        print(
            "PIP release works and contact remains continuous."
        )

        print(
            "Do NOT tune friction."
        )

        print(
            "Remaining timing error should be treated at the "
            "actuator/transmission input layer."
        )

    else:

        print(
            "REJECT THE FRICTION / HYSTERESIS BRANCH."
        )

        print(
            "No Fs/Fc/vs tuning."
        )

        print(
            "Next = free-space-derived nonlinear joint-stop/"
            "transverse-spring mechanics."
        )

    # =============================================================
    # Save all outputs
    # =============================================================

    save_rows(
        out
        /
        "g37a_free_passive_history.csv",
        free_passive,
    )

    save_rows(
        out
        /
        "g37a_contact_passive_history.csv",
        contact_passive,
    )

    save_rows(
        out
        /
        "g37a_contact_drive_history.csv",
        contact_drive,
    )

    summary_csv = (
        out
        /
        "g37a_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "approach_rmse_deg",
                "first_contact_encoder",
                "first_contact_encoder_error",
                "first_contact_pose_rmse_deg",
                "middle_episodes",
                "tip_episodes",
                "middle_gap_s",
                "final_mcp_deg",
                "final_pip_deg",
                "final_dip_deg",
                "final_rmse_deg",
                "post_contact_dmcp_deg",
                "post_contact_dpip_deg",
                "post_contact_ddip_deg",
                "mean_tendon_tracking_error_mm",
                "max_tendon_tracking_error_mm",
            ],
        )

        writer.writeheader()

        writer.writerow({
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
                len(mid_starts),

            "tip_episodes":
                len(tip_starts),

            "middle_gap_s":
                gap,

            "final_mcp_deg":
                final[0],

            "final_pip_deg":
                final[1],

            "final_dip_deg":
                final[2],

            "final_rmse_deg":
                final_rmse,

            "post_contact_dmcp_deg":
                recruitment[0],

            "post_contact_dpip_deg":
                recruitment[1],

            "post_contact_ddip_deg":
                recruitment[2],

            "mean_tendon_tracking_error_mm":
                mean_tracking,

            "max_tendon_tracking_error_mm":
                max_tracking,
        })

    np.savez_compressed(
        out
        /
        "g37a_contact_trajectory.npz",
        times=np.asarray(
            times
        ),
        qpos=np.asarray(
            qpos
        ),
    )

    print()
    print(
        f"Candidate XML:\n  {candidate}"
    )

    print()

    print(
        f"Summary:\n  {summary_csv}"
    )

    print()

    print(
        f"All outputs:\n  {out}"
    )

    if not args.no_gui:

        view_xml = (
            out
            /
            "g37a_view.xml"
        )

        g29b.make_view_xml(
            candidate,
            view_xml,
        )

        print()
        print("=" * 150)
        print(
            "G37A GUI"
        )
        print("=" * 150)

        print(
            "Watch only:"
        )

        print(
            "  1. MCP dominates approach"
        )

        print(
            "  2. middle contacts"
        )

        print(
            "  3. PIP breaks away and wraps"
        )

        print(
            "  4. middle remains attached"
        )

        print(
            "  5. DIP participates later"
        )

        g7.exact_gui_replay(
            view_xml,
            qpos,
            times,
            "G37A direct spatial tendon + velocity release",
        )


if __name__ == "__main__":
    main()
