#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import mujoco
import numpy as np

import scan_index_contact_friction_v35g7 as g7
import audit_index_tendon_transmission_v35g12 as g12
import test_direct_spatial_tendon_drive_v35g32a as g32a


MAIN_TENDON = "idx_flex_spatial"
DIRECT_ACTUATOR = "g32_index_tendon_position"

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

TARGET_TIME = 38.970486


def tendon_jacobian_row(model, data, tid):

    raw = np.asarray(
        data.ten_J,
        dtype=float,
    )

    if (
        raw.ndim == 2
        and
        raw.shape == (model.ntendon, model.nv)
    ):
        return raw[tid, :].copy()

    if (
        raw.ndim == 1
        and
        raw.size == model.ntendon * model.nv
    ):
        return raw.reshape(
            model.ntendon,
            model.nv,
        )[tid, :].copy()

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
        rownnz[tid]
    )

    adr = int(
        rowadr[tid]
    )

    for k in range(nnz):

        idx = adr + k

        J[
            int(
                colind[idx]
            )
        ] = float(
            raw[idx]
        )

    return J


class ForceLogger:

    def __init__(self):

        self.original_step = mujoco.mj_step

        self.cache = {}

        self.rows = []

    def setup(self, model):

        key = id(model)

        if key in self.cache:
            return self.cache[key]

        aid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            DIRECT_ACTUATOR,
        )

        main_tid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            MAIN_TENDON,
        )

        if (
            aid < 0
            or
            main_tid < 0
        ):
            self.cache[key] = None
            return None

        refs = g12.get_joint_refs(
            model
        )

        spring_tids = {}

        for key_name, name in SPRINGS.items():

            tid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_TENDON,
                name,
            )

            if tid < 0:
                raise RuntimeError(
                    f"Missing spring tendon: {name}"
                )

            spring_tids[
                key_name
            ] = tid

        state = {
            "aid":
                aid,

            "main_tid":
                main_tid,

            "refs":
                refs,

            "spring_tids":
                spring_tids,
        }

        self.cache[key] = state

        return state

    def step(self, model, data):

        state = self.setup(
            model
        )

        # This calls the ORIGINAL MuJoCo step because the
        # DirectSpatialTendonDrive wrapper is installed outside us.
        self.original_step(
            model,
            data,
        )

        if state is None:
            return

        main_J = tendon_jacobian_row(
            model,
            data,
            state[
                "main_tid"
            ],
        )

        row = {
            "time_s":
                float(
                    data.time
                ),

            "ncon":
                int(
                    data.ncon
                ),

            "direct_actuator_force":
                float(
                    data.actuator_force[
                        state[
                            "aid"
                        ]
                    ]
                ),
        }

        for joint in [
            "mcp",
            "pip",
            "dip",
        ]:

            ref = state[
                "refs"
            ][
                joint
            ]

            qadr = ref[
                "qadr"
            ]

            dadr = ref[
                "dadr"
            ]

            spring_tid = state[
                "spring_tids"
            ][
                joint
            ]

            spring_J = tendon_jacobian_row(
                model,
                data,
                spring_tid,
            )

            # --------------------------------------------------
            # This MuJoCo Python build does not expose
            # data.ten_force.
            #
            # G33 return tendons are pure elastic springs with no
            # tendon damping, therefore recover their tension from
            #
            #     F = K * extension
            #
            # using the same MuJoCo tendon length/reference.
            # --------------------------------------------------

            spring_length = float(
                data.ten_length[
                    spring_tid
                ]
            )

            spring_reference = (
                g12.tendon_spring_reference(
                    model,
                    spring_tid,
                )
            )

            spring_extension = (
                g12.signed_extension(
                    spring_length,
                    spring_reference,
                )
            )

            spring_K = float(
                model.tendon_stiffness[
                    spring_tid
                ]
            )

            spring_force = max(
                0.0,
                spring_K
                *
                spring_extension,
            )

            spring_tau = (
                -
                spring_J[
                    dadr
                ]
                *
                spring_force
            )

            row.update({
                f"{joint}_q_deg":
                    math.degrees(
                        float(
                            data.qpos[
                                qadr
                            ]
                        )
                    ),

                f"{joint}_qvel_deg_s":
                    math.degrees(
                        float(
                            data.qvel[
                                dadr
                            ]
                        )
                    ),

                f"{joint}_qacc_deg_s2":
                    math.degrees(
                        float(
                            data.qacc[
                                dadr
                            ]
                        )
                    ),

                # Complete generalized actuator contribution.
                f"{joint}_tau_actuator":
                    float(
                        data.qfrc_actuator[
                            dadr
                        ]
                    ),

                # All passive forces reported by MuJoCo.
                f"{joint}_tau_passive":
                    float(
                        data.qfrc_passive[
                            dadr
                        ]
                    ),

                # Contact + joint-limit constraints.
                f"{joint}_tau_constraint":
                    float(
                        data.qfrc_constraint[
                            dadr
                        ]
                    ),

                f"{joint}_bias":
                    float(
                        data.qfrc_bias[
                            dadr
                        ]
                    ),

                f"{joint}_main_J_mm_rad":
                    1000.0
                    *
                    float(
                        main_J[
                            dadr
                        ]
                    ),

                f"{joint}_spring_force_N":
                    spring_force,

                f"{joint}_spring_J_mm_rad":
                    1000.0
                    *
                    float(
                        spring_J[
                            dadr
                        ]
                    ),

                f"{joint}_spring_tau":
                    float(
                        spring_tau
                    ),
            })

        self.rows.append(
            row
        )

    def install(self):

        mujoco.mj_step = (
            self.step
        )

    def restore(self):

        mujoco.mj_step = (
            self.original_step
        )


def nearest_row(rows, target):

    return min(
        rows,
        key=lambda r:
            abs(
                r[
                    "time_s"
                ]
                -
                target
            ),
    )


def print_state(
    title,
    row,
):

    print()
    print("=" * 150)
    print(title)
    print("=" * 150)

    print(
        f"time = {row['time_s']:.6f} s"
    )

    print(
        f"active MuJoCo contacts = {row['ncon']}"
    )

    print(
        f"direct tendon actuator force = "
        f"{row['direct_actuator_force']:+.5f}"
    )

    print()
    print(
        "joint      q       qvel       main J    "
        "actuator      passive    constraint       bias      spring-only"
    )

    print("-" * 150)

    for joint in [
        "mcp",
        "pip",
        "dip",
    ]:

        print(
            f"{joint.upper():4s} "
            f"{row[f'{joint}_q_deg']:8.3f} "
            f"{row[f'{joint}_qvel_deg_s']:10.3f} "
            f"{row[f'{joint}_main_J_mm_rad']:10.4f} "
            f"{row[f'{joint}_tau_actuator']:+11.6f} "
            f"{row[f'{joint}_tau_passive']:+11.6f} "
            f"{row[f'{joint}_tau_constraint']:+11.6f} "
            f"{row[f'{joint}_bias']:+10.6f} "
            f"{row[f'{joint}_spring_tau']:+12.6f}"
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
            "v35g34a_g33_force_balance"
        ),
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

    # --------------------------------------------------------------
    # Recover EXACT same direct-tendon mapping as G33A.
    # --------------------------------------------------------------

    (
        open_length,
        ratio,
        mapping_rmse,
    ) = (
        g32a.fit_encoder_to_tendon_shortening(
            xml
        )
    )

    print()
    print("=" * 150)
    print(
        "G34A — G33 PIP FORCE-BALANCE AUTOPSY"
    )
    print("=" * 150)

    print()
    print(
        f"direct tendon shortening ratio = "
        f"{ratio:.6f}"
    )

    print(
        f"mapping RMSE                  = "
        f"{mapping_rmse:.4f} mm"
    )

    print()
    print(
        "NO PARAMETERS ARE CHANGED."
    )

    # --------------------------------------------------------------
    # Normal replay setup.
    # --------------------------------------------------------------

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    g7.b2.v35.METERS_PER_TICK = (
        5.70e-6
    )

    # --------------------------------------------------------------
    # Installation order is deliberate:
    #
    # ForceLogger first
    # Direct drive second
    #
    # Direct drive therefore calls ForceLogger after setting
    # the measured tendon-length command.
    # --------------------------------------------------------------

    logger = ForceLogger()
    logger.install()

    drive = g32a.DirectSpatialTendonDrive(
        open_length,
        ratio,
    )

    drive.install()

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

        drive.restore()
        logger.restore()

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

    if not logger.rows:

        raise RuntimeError(
            "No force-balance samples recorded."
        )

    # --------------------------------------------------------------
    # Recover first middle-contact event.
    # --------------------------------------------------------------

    flags = np.asarray(
        [
            bool(
                r[
                    "mid_contact"
                ]
            )
            for r in rows
        ],
        dtype=bool,
    )

    starts = g12.contact_episode_starts(
        flags
    )

    if not starts:

        raise RuntimeError(
            "No middle contact."
        )

    first_index = int(
        starts[
            0
        ]
    )

    first_time = float(
        rows[
            first_index
        ][
            "time_s"
        ]
    )

    # --------------------------------------------------------------
    # Inspect four physically meaningful states.
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # IMPORTANT CLOCK ALIGNMENT
    #
    # rows[*]["time_s"] uses the rosbag / experiment time base.
    #
    # logger.rows[*]["time_s"] uses MuJoCo data.time, which starts
    # close to zero for the simulation.
    #
    # They have the same elapsed-time progression but DIFFERENT
    # origins.  The old G34A code compared them directly, causing
    # every requested stage to clamp to the last MuJoCo sample.
    # --------------------------------------------------------------

    audit_t0 = float(
        rows[
            0
        ][
            "time_s"
        ]
    )

    logger_t0 = float(
        logger.rows[
            0
        ][
            "time_s"
        ]
    )

    clock_offset = (
        audit_t0
        -
        logger_t0
    )

    def audit_to_logger_time(
        audit_time,
    ):

        return (
            logger_t0
            +
            (
                float(
                    audit_time
                )
                -
                audit_t0
            )
        )

    first_logger_time = (
        audit_to_logger_time(
            first_time
        )
    )

    target_logger_time = (
        audit_to_logger_time(
            TARGET_TIME
        )
    )

    print()
    print("=" * 150)
    print(
        "G34A CLOCK ALIGNMENT"
    )
    print("=" * 150)

    print(
        f"audit/rosbag first time = "
        f"{audit_t0:.6f} s"
    )

    print(
        f"MuJoCo logger first time = "
        f"{logger_t0:.6f} s"
    )

    print(
        f"clock offset             = "
        f"{clock_offset:+.6f} s"
    )

    print(
        f"first-contact audit time = "
        f"{first_time:.6f} s"
    )

    print(
        f"first-contact MuJoCo time= "
        f"{first_logger_time:.6f} s"
    )

    print(
        f"H1 target audit time     = "
        f"{TARGET_TIME:.6f} s"
    )

    print(
        f"H1 target MuJoCo time    = "
        f"{target_logger_time:.6f} s"
    )

    # Sanity gate: refuse another misleading diagnosis.
    logger_min = float(
        logger.rows[
            0
        ][
            "time_s"
        ]
    )

    logger_max = float(
        logger.rows[
            -1
        ][
            "time_s"
        ]
    )

    if not (
        logger_min
        <=
        first_logger_time
        <=
        logger_max
    ):

        raise RuntimeError(
            "Clock-aligned first-contact time lies outside "
            "the MuJoCo logger range."
        )

    if not (
        logger_min
        <=
        target_logger_time
        <=
        logger_max
    ):

        raise RuntimeError(
            "Clock-aligned H1 target time lies outside "
            "the MuJoCo logger range."
        )

    stages = [
        (
            "FIRST MIDDLE CONTACT",
            first_logger_time,
        ),

        (
            "200 ms AFTER FIRST CONTACT",
            first_logger_time
            +
            0.200,
        ),

        (
            "500 ms AFTER FIRST CONTACT",
            first_logger_time
            +
            0.500,
        ),

        (
            "TRUSTED H1 TARGET TIME",
            target_logger_time,
        ),
    ]

    selected = {}

    previous_selected_time = None

    for name, time_value in stages:

        r = nearest_row(
            logger.rows,
            time_value,
        )

        selected[
            name
        ] = r

        actual_selected_time = float(
            r[
                "time_s"
            ]
        )

        # The first 3 stages MUST NOT all collapse to one sample.
        if (
            previous_selected_time is not None
            and
            name
            !=
            "TRUSTED H1 TARGET TIME"
            and
            abs(
                actual_selected_time
                -
                previous_selected_time
            )
            <
            0.050
        ):

            raise RuntimeError(
                "Force-balance stages collapsed onto nearly "
                "the same MuJoCo timestamp. Refusing diagnosis."
            )

        previous_selected_time = (
            actual_selected_time
        )

        print_state(
            name,
            r,
        )

    final = selected[
        "TRUSTED H1 TARGET TIME"
    ]

    # --------------------------------------------------------------
    # PIP-specific diagnosis.
    # --------------------------------------------------------------

    act = float(
        final[
            "pip_tau_actuator"
        ]
    )

    passive = float(
        final[
            "pip_tau_passive"
        ]
    )

    constraint = float(
        final[
            "pip_tau_constraint"
        ]
    )

    bias = float(
        final[
            "pip_bias"
        ]
    )

    spring_only = float(
        final[
            "pip_spring_tau"
        ]
    )

    # For positive flexion, negative quantities oppose flexion.
    passive_opp = max(
        0.0,
        -passive,
    )

    constraint_opp = max(
        0.0,
        -constraint,
    )

    bias_opp = max(
        0.0,
        bias,
    )

    net_generalized = (
        act
        +
        passive
        +
        constraint
        -
        bias
    )

    print()
    print("=" * 150)
    print(
        "G34A PIP DIAGNOSIS AT H1 TARGET"
    )
    print("=" * 150)

    print()
    print(
        f"PIP angle                  = "
        f"{final['pip_q_deg']:.3f} deg"
    )

    print(
        f"PIP angular velocity       = "
        f"{final['pip_qvel_deg_s']:+.3f} deg/s"
    )

    print(
        f"main tendon moment arm     = "
        f"{final['pip_main_J_mm_rad']:+.4f} mm/rad"
    )

    print(
        f"flexor actuator torque     = "
        f"{act:+.6f} Nm"
    )

    print(
        f"passive torque             = "
        f"{passive:+.6f} Nm"
    )

    print(
        f"  wrapped spring component = "
        f"{spring_only:+.6f} Nm"
    )

    print(
        f"constraint/contact torque  = "
        f"{constraint:+.6f} Nm"
    )

    print(
        f"bias                       = "
        f"{bias:+.6f} Nm"
    )

    print(
        f"net generalized drive      = "
        f"{net_generalized:+.6f} Nm"
    )

    print()
    print(
        "Opposition magnitudes:"
    )

    print(
        f"  passive      = "
        f"{passive_opp:.6f} Nm"
    )

    print(
        f"  contact/constraint = "
        f"{constraint_opp:.6f} Nm"
    )

    print(
        f"  bias         = "
        f"{bias_opp:.6f} Nm"
    )

    # --------------------------------------------------------------
    # Decision.
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "G34A DECISION"
    )
    print("=" * 150)

    abs_act = abs(
        act
    )

    if (
        abs_act < 0.005
        and
        abs(
            final[
                "direct_actuator_force"
            ]
        )
        > 2.0
    ):

        print(
            "MAIN FLEXOR TRANSMISSION IS THE BOTTLENECK."
        )

        print(
            "The actuator is pulling strongly but little PIP "
            "generalized torque reaches the joint."
        )

        print(
            "NEXT: rebuild persistent PIP/MCP/DIP flexor wraps "
            "from the RH8D CAD route."
        )

    elif (
        constraint_opp
        >
        1.5
        *
        max(
            passive_opp,
            1e-9,
        )
        and
        constraint_opp
        >
        0.5
        *
        max(
            abs_act,
            1e-9,
        )
    ):

        print(
            "CONTACT/CONSTRAINT MECHANICS ARE JAMMING PIP."
        )

        print(
            "The flexor reaches PIP, but the bottle/contact reaction "
            "prevents the hardware-like sliding/wrapping motion."
        )

        print(
            "NEXT: compliant/distributed middle-phalanx contact pad. "
            "Do NOT change tendon or springs."
        )

    elif (
        passive_opp
        >
        1.5
        *
        max(
            constraint_opp,
            1e-9,
        )
        and
        passive_opp
        >
        0.5
        *
        max(
            abs_act,
            1e-9,
        )
    ):

        print(
            "RETURN-SPRING/PASSIVE MECHANICS ARE BLOCKING PIP."
        )

        print(
            "NEXT: reject the identified G33 PIP spring model. "
            "Do NOT touch the flexor/contact yet."
        )

    else:

        print(
            "NO SINGLE TORQUE SOURCE DOMINATES."
        )

        print(
            "Use the printed force balance to choose the largest "
            "physically opposing term; do not parameter-scan."
        )

    # --------------------------------------------------------------
    # Save all logged samples.
    # --------------------------------------------------------------

    csv_path = (
        out
        /
        "g34a_force_balance_timeseries.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                logger.rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            logger.rows
        )

    print()
    print(
        f"Force-balance CSV:\n  {csv_path}"
    )

    print()
    print(
        f"All outputs:\n  {out}"
    )


if __name__ == "__main__":
    main()
