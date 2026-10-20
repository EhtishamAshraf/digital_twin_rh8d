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
# G29B — SPATIAL TENDON + PRESCRIBED MEASURED SPOOL
#
# CRITICAL ARCHITECTURE:
#
#     index_motor_spool
#             ↓
#       idx_flex_spatial
#             ↓
#       MCP / PIP / DIP
#
# The spatial tendon is NEVER replaced.
#
# Only the already-measured hardware motor coordinate is imposed on
# index_motor_spool during H1 mechanics replay.
#
# MCP/PIP/DIP remain fully dynamic MuJoCo joints.
#
# NO:
#   fixed tendon
#   joint qpos replay
#   spring tuning
#   tendon K tuning
#   contact tuning
#   friction tuning
# =====================================================================


SPATIAL_TENDON = "idx_flex_spatial"
SPOOL_JOINT = "index_motor_spool"
SPOOL_ACTUATOR = "pos_index_tendon"

OPEN_ENCODER = 100.0
SCALE = 5.70e-6

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


def verify_spatial_tendon(xml_path):

    tree = ET.parse(
        xml_path
    )

    root = tree.getroot()

    spatial_found = False
    fixed_found = False

    for tendon_root in root.iter(
        "tendon"
    ):

        for element in list(
            tendon_root
        ):

            if (
                element.get(
                    "name"
                )
                ==
                SPATIAL_TENDON
            ):

                if element.tag == "spatial":

                    spatial_found = True

                if element.tag == "fixed":

                    fixed_found = True

    if not spatial_found:

        raise RuntimeError(
            "STOP: idx_flex_spatial is NOT a <spatial> tendon."
        )

    if fixed_found:

        raise RuntimeError(
            "STOP: fixed tendon named idx_flex_spatial found."
        )

    model = mujoco.MjModel.from_xml_path(
        str(
            xml_path
        )
    )

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        SPATIAL_TENDON,
    )

    if tid < 0:

        raise RuntimeError(
            "MuJoCo could not resolve idx_flex_spatial."
        )

    print()
    print(
        "SPATIAL-TENDON ARCHITECTURE CHECK: PASS"
    )

    print(
        "  idx_flex_spatial is a real MuJoCo <spatial> tendon."
    )

    print(
        "  It will remain in the simulation and in the GUI."
    )


def make_view_xml(
    source,
    destination,
):

    """
    Visualization only.

    Palm + index transparent.
    Every other finger opaque.
    Tendon unchanged and visible.
    """

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    transparent_index = {
        "Index_Proximal",
        "Index_Middle",
        "Index_Distal",
    }

    for body in root.iter(
        "body"
    ):

        name = body.get(
            "name",
            "",
        )

        if name == "palm_assembly:1":

            alpha = 0.25

        elif name in transparent_index:

            alpha = 0.35

        else:

            alpha = 1.0

        # Only this body's own visual geometry.
        # Do not propagate palm transparency into child fingers.
        for geom in body.findall(
            "geom"
        ):

            if (
                geom.get(
                    "class"
                )
                !=
                "visual_mesh"
            ):

                continue

            rgba = geom.get(
                "rgba"
            )

            if rgba is None:

                continue

            values = [
                float(x)
                for x in rgba.split()
            ]

            if len(
                values
            ) != 4:

                continue

            values[
                3
            ] = alpha

            geom.set(
                "rgba",
                " ".join(
                    f"{x:.6g}"
                    for x in values
                ),
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


class PrescribedSpool:

    """
    Intercepts mj_step for the H1 model.

    The ordinary contact replay already calculates

        ctrl[pos_index_tendon]

    from the measured hardware encoder trajectory.

    Instead of asking another MuJoCo position servo to chase that
    measured target, we impose that target directly on the spool DOF.

    Finger joints are NEVER overwritten.
    """

    def __init__(self):

        self.original_step = (
            mujoco.mj_step
        )

        self.cache = {}

        self.post_step_drifts = []

        self.targets = []

    def _setup(
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

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            SPOOL_JOINT,
        )

        aid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            SPOOL_ACTUATOR,
        )

        tid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            SPATIAL_TENDON,
        )

        if (
            jid < 0
            or
            aid < 0
            or
            tid < 0
        ):

            # Not our H1 model.
            self.cache[
                key
            ] = None

            return None

        qadr = int(
            model.jnt_qposadr[
                jid
            ]
        )

        dadr = int(
            model.jnt_dofadr[
                jid
            ]
        )

        # ---------------------------------------------------------
        # IMPORTANT:
        #
        # Disable ONLY the artificial MuJoCo servo responsible for
        # making the spool chase an already measured motor state.
        #
        # The actuator still exists and data.ctrl[aid] is still used
        # by the replay code as the measured-spool target.
        #
        # XML on disk is NOT changed.
        # ---------------------------------------------------------

        gain = (
            model.actuator_gainprm[
                aid
            ].copy()
        )

        bias = (
            model.actuator_biasprm[
                aid
            ].copy()
        )

        model.actuator_gainprm[
            aid,
            :
        ] = 0.0

        model.actuator_biasprm[
            aid,
            :
        ] = 0.0

        state = {
            "model":
                model,

            "jid":
                jid,

            "aid":
                aid,

            "tid":
                tid,

            "qadr":
                qadr,

            "dadr":
                dadr,

            "gain":
                gain,

            "bias":
                bias,

            "last_target":
                None,
        }

        self.cache[
            key
        ] = state

        return state

    def step(
        self,
        model,
        data,
    ):

        state = self._setup(
            model
        )

        if state is None:

            return self.original_step(
                model,
                data,
            )

        aid = state[
            "aid"
        ]

        qadr = state[
            "qadr"
        ]

        dadr = state[
            "dadr"
        ]

        dt = float(
            model.opt.timestep
        )

        # Position actuator control is the spool target in metres.
        target = float(
            data.ctrl[
                aid
            ]
        )

        # Respect the physical spool joint range.
        jid = state[
            "jid"
        ]

        if bool(
            model.jnt_limited[
                jid
            ]
        ):

            lower = float(
                model.jnt_range[
                    jid,
                    0
                ]
            )

            upper = float(
                model.jnt_range[
                    jid,
                    1
                ]
            )

            target = float(
                np.clip(
                    target,
                    lower,
                    upper,
                )
            )

        previous = state[
            "last_target"
        ]

        if previous is None:

            velocity = 0.0

        else:

            velocity = (
                target
                -
                previous
            ) / dt

        # ---------------------------------------------------------
        # PRESCRIBE ONLY SPOOL.
        # ---------------------------------------------------------

        data.qpos[
            qadr
        ] = target

        data.qvel[
            dadr
        ] = velocity

        state[
            "last_target"
        ] = target

        self.targets.append(
            (
                float(
                    data.time
                ),
                target,
            )
        )

        # Finger mechanics evolve dynamically.
        self.original_step(
            model,
            data,
        )

        drift = abs(
            float(
                data.qpos[
                    qadr
                ]
            )
            -
            target
        )

        self.post_step_drifts.append(
            drift
        )

    def install(self):

        mujoco.mj_step = (
            self.step
        )

    def restore(self):

        mujoco.mj_step = (
            self.original_step
        )

        # Restore actuator parameters on any model objects that
        # still exist.
        for state in self.cache.values():

            if state is None:
                continue

            model = state[
                "model"
            ]

            aid = state[
                "aid"
            ]

            model.actuator_gainprm[
                aid,
                :
            ] = state[
                "gain"
            ]

            model.actuator_biasprm[
                aid,
                :
            ] = state[
                "bias"
            ]


def episode_indices(
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
            "v35g29b_spatial_prescribed_spool"
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

    xml = (
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

    # =============================================================
    # Mandatory topology check.
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G29B — SPATIAL TENDON + PRESCRIBED MEASURED SPOOL"
    )
    print("=" * 150)

    verify_spatial_tendon(
        xml
    )

    model_check = mujoco.MjModel.from_xml_path(
        str(
            xml
        )
    )

    tid = mujoco.mj_name2id(
        model_check,
        mujoco.mjtObj.mjOBJ_TENDON,
        SPATIAL_TENDON,
    )

    print()
    print(
        f"spatial tendon stiffness = "
        f"{model_check.tendon_stiffness[tid]:.3f} N/m"
    )

    print()
    print(
        "ONLY experimental change:"
    )

    print(
        "  measured spool target is imposed directly on "
        "index_motor_spool."
    )

    print()
    print(
        "MCP/PIP/DIP remain completely dynamic."
    )

    # =============================================================
    # Dynamic run.
    # =============================================================

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

    clamp = PrescribedSpool()

    clamp.install()

    try:

        (
            rows,
            qpos,
            times,
            summary,
        ) = g12.run_audit(
            xml,
            bag,
            replay,
        )

    finally:

        clamp.restore()

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

    # =============================================================
    # Metrics.
    # =============================================================

    mid_starts, mid_ends = episode_indices(
        rows,
        "mid_contact",
    )

    tip_starts, tip_ends = episode_indices(
        rows,
        "tip_contact",
    )

    if not mid_starts:

        raise RuntimeError(
            "No middle contact occurred."
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

    first_pose_error = (
        first_pose
        -
        HW_CONTACT_POSE
    )

    first_pose_rmse = float(
        np.sqrt(
            np.mean(
                first_pose_error
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

    row_times = np.asarray(
        [
            row[
                "time_s"
            ]
            for row in rows
        ],
        dtype=float,
    )

    ti = nearest_index(
        row_times,
        TARGET_TIME,
    )

    target_row = rows[
        ti
    ]

    final_pose = np.array(
        [
            target_row[
                "mcp_deg"
            ],
            target_row[
                "pip_deg"
            ],
            target_row[
                "dip_deg"
            ],
        ],
        dtype=float,
    )

    final_error = (
        final_pose
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

    max_step_drift_mm = (
        1000.0
        *
        max(
            clamp.post_step_drifts,
            default=0.0,
        )
    )

    print()
    print("=" * 150)
    print(
        "G29B RESULTS"
    )
    print("=" * 150)

    print()
    print(
        "FIRST MIDDLE CONTACT"
    )

    print(
        f"  encoder = "
        f"{first_row['encoder_tick']:.3f} ticks"
    )

    print(
        f"  hardware= "
        f"{HW_CONTACT_ENCODER:.3f} ticks"
    )

    print(
        f"  error   = "
        f"{first_row['encoder_tick'] - HW_CONTACT_ENCODER:+.3f} ticks"
    )

    print()
    print(
        "  simulation pose = "
        f"{first_pose[0]:.3f} / "
        f"{first_pose[1]:.3f} / "
        f"{first_pose[2]:.3f}"
    )

    print(
        "  hardware pose   = "
        f"{HW_CONTACT_POSE[0]:.3f} / "
        f"{HW_CONTACT_POSE[1]:.3f} / "
        f"{HW_CONTACT_POSE[2]:.3f}"
    )

    print(
        f"  pose RMSE       = "
        f"{first_pose_rmse:.3f} deg"
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

    print()
    print(
        "TRUSTED FINAL H1"
    )

    print(
        "  simulation = "
        f"{final_pose[0]:.3f} / "
        f"{final_pose[1]:.3f} / "
        f"{final_pose[2]:.3f}"
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
        "PRESCRIBED-SPOOL ENFORCEMENT"
    )

    print(
        f"  maximum within-step drift = "
        f"{max_step_drift_mm:.6f} mm"
    )

    # =============================================================
    # Save.
    # =============================================================

    summary_csv = (
        out
        /
        "g29b_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "spatial_tendon",
                "first_contact_encoder",
                "first_contact_encoder_error",
                "first_contact_mcp_deg",
                "first_contact_pip_deg",
                "first_contact_dip_deg",
                "first_contact_pose_rmse_deg",
                "middle_episodes",
                "tip_episodes",
                "middle_detached_gap_s",
                "final_mcp_deg",
                "final_pip_deg",
                "final_dip_deg",
                "final_rmse_deg",
                "max_spool_step_drift_mm",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "spatial_tendon":
                True,

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

            "first_contact_mcp_deg":
                first_pose[
                    0
                ],

            "first_contact_pip_deg":
                first_pose[
                    1
                ],

            "first_contact_dip_deg":
                first_pose[
                    2
                ],

            "first_contact_pose_rmse_deg":
                first_pose_rmse,

            "middle_episodes":
                len(
                    mid_starts
                ),

            "tip_episodes":
                len(
                    tip_starts
                ),

            "middle_detached_gap_s":
                detached_gap,

            "final_mcp_deg":
                final_pose[
                    0
                ],

            "final_pip_deg":
                final_pose[
                    1
                ],

            "final_dip_deg":
                final_pose[
                    2
                ],

            "final_rmse_deg":
                final_rmse,

            "max_spool_step_drift_mm":
                max_step_drift_mm,
        })

    np.savez_compressed(
        out
        /
        "g29b_dynamic_trajectory.npz",
        times=np.asarray(
            times
        ),
        qpos=np.asarray(
            qpos
        ),
    )

    # =============================================================
    # Decision.
    # =============================================================

    print()
    print("=" * 150)
    print(
        "G29B DECISION"
    )
    print("=" * 150)

    if (
        abs(
            first_row[
                "encoder_tick"
            ]
            -
            HW_CONTACT_ENCODER
        )
        <=
        60.0
        and
        first_pose_rmse
        <=
        4.0
    ):

        print(
            "Direct measured-spool input explains a substantial "
            "part of the H1 approach error."
        )

    else:

        print(
            "Removing the extra spool servo lag is NOT sufficient "
            "to reproduce hardware first contact."
        )

    if len(
        mid_starts
    ) > 1:

        print(
            "Middle-contact bounce remains."
        )

    else:

        print(
            "Middle contact remains continuous."
        )

    print()
    print(
        "Regardless of result, the spatial tendon remains mandatory."
    )

    print()
    print(
        f"Summary:\n  {summary_csv}"
    )

    print(
        f"All outputs:\n  {out}"
    )

    # =============================================================
    # GUI — transparency only, no physics modification.
    # =============================================================

    if not args.no_gui:

        view_xml = (
            out
            /
            "g29b_spatial_view.xml"
        )

        make_view_xml(
            xml,
            view_xml,
        )

        print()
        print("=" * 150)
        print(
            "G29B GUI — SPATIAL TENDON MUST BE VISIBLE"
        )
        print("=" * 150)

        print(
            "Palm + index are transparent."
        )

        print(
            "Thumb/middle/ring/small remain opaque."
        )

        print(
            "The magenta index spatial tendon should be visible."
        )

        g7.exact_gui_replay(
            view_xml,
            qpos,
            times,
            "G29B spatial tendon + measured prescribed spool",
        )


if __name__ == "__main__":
    main()
