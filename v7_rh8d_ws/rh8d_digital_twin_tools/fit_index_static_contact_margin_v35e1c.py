#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

import fit_index_passive_contact_v35b2 as b2
import audit_index_torque_budget_v35d2 as d2
import audit_index_contact_force_v35d5 as d5
import fit_index_static_contact_impedance_v35e1b as e1b


# ================================================================
# V35E1C — STATIC CONTACT-MARGIN IDENTIFICATION
#
# VARIABLE ONLY:
#     total cylinder/finger contact margin
#
# FROZEN:
#     direct solref = [-1000, -20]
#     solimp        = [0.95, 0.95, 0.001, 0.5, 2]
#     mu            = 0.05
#     CAD tendon
#     V35B2 passives
#     H1 cylinder registration
#     exact TARGET_BAG_TIME spool/controller state
#
# Measured H1 joint angles are prescribed ONLY for this
# static diagnostic.
#
# IMPORTANT:
# margin is NOT geometric mesh inflation in the visual model.
# It is the MuJoCo contact activation/constraint margin.
# ================================================================


HW_POSE_DEG = np.array(
    [14.5355, 23.8110, 9.3270],
    dtype=float,
)

HW_TIP_FORCE = 0.14563

DIRECT_K = 1000.0
DIRECT_DAMPING = 20.0

SOLIMP = np.array(
    [0.95, 0.95, 0.001, 0.5, 2.0],
    dtype=float,
)


# Total contact margin candidates, in millimetres.
MARGIN_MM_VALUES = [
    0.000,
    0.005,
    0.010,
    0.020,
    0.030,
    0.040,
    0.050,
    0.075,
    0.100,
    0.150,
    0.200,
    0.300,
    0.500,
    0.750,
    1.000,
]


FIXTURE_NAME = "index_contact_fixture"

INDEX_CONTACT_GEOMS = [
    "collision_idx_prox",
    "collision_idx_mid_mesh",
    "collision_idx_tip",
]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    args = parser.parse_args()

    bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )

    if not bag.exists():
        raise RuntimeError(
            f"Missing bag:\n{bag}"
        )

    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
    )

    target_time = float(
        contact_replay.TARGET_BAG_TIME
    )

    old_damping = float(
        contact_replay.CONTACT_DAMPING
    )

    old_mu = b2.v35.CONTACT_MU
    old_k = b2.v35.CONTACT_K

    contact_replay.CONTACT_DAMPING = (
        DIRECT_DAMPING
    )

    b2.v35.CONTACT_MU = 0.05
    b2.v35.CONTACT_K = DIRECT_K

    # ============================================================
    # Capture exact TARGET_BAG_TIME state.
    # ============================================================

    states = []

    captured = {
        "model": None,
        "data": None,
    }

    original_step = mujoco.mj_step

    def capture_step(
        model,
        data,
        *args,
        **kwargs,
    ):

        result = original_step(
            model,
            data,
            *args,
            **kwargs,
        )

        t = d2.find_replay_frame_time()

        if t is not None:

            states.append(
                d5.snapshot(
                    t,
                    data,
                )
            )

            captured["model"] = model
            captured["data"] = data

        return result

    mujoco.mj_step = capture_step

    try:

        print()
        print("=" * 120)
        print(
            "V35E1C — ACQUIRING SYNCHRONIZED H1 STATE"
        )
        print("=" * 120)

        b2.run_candidate(
            b2.SELECTED_MODEL,
            bag,
        )

    finally:

        mujoco.mj_step = original_step

    if not states:
        raise RuntimeError(
            "No replay states captured."
        )

    model = captured["model"]
    data = captured["data"]

    times = np.asarray(
        [
            row["time"]
            for row in states
        ],
        dtype=float,
    )

    target_index = int(
        np.argmin(
            np.abs(
                times
                -
                target_time
            )
        )
    )

    target_state = states[
        target_index
    ]

    print()
    print(
        f"TARGET_BAG_TIME = "
        f"{target_time:.6f}"
    )

    print(
        f"Captured time   = "
        f"{target_state['time']:.6f}"
    )

    print(
        f"Timing error    = "
        f"{1000*abs(target_state['time']-target_time):.3f} ms"
    )

    joint_info = d2.joint_addresses(
        model
    )

    fixture_gid = d2.oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FIXTURE_NAME,
    )

    if fixture_gid is None:
        raise RuntimeError(
            "Missing cylinder fixture."
        )

    index_gids = []

    for name in INDEX_CONTACT_GEOMS:

        gid = d2.oid(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )

        if gid is None:
            raise RuntimeError(
                f"Missing geom: {name}"
            )

        index_gids.append(
            gid
        )

    all_gids = [
        fixture_gid,
        *index_gids,
    ]

    # ============================================================
    # Preserve original contact parameters.
    # ============================================================

    old_margin = {
        gid:
            float(
                model.geom_margin[gid]
            )
        for gid in all_gids
    }

    old_gap = {
        gid:
            float(
                model.geom_gap[gid]
            )
        for gid in all_gids
    }

    old_solref = {
        gid:
            model.geom_solref[
                gid
            ].copy()
        for gid in all_gids
    }

    old_solimp = {
        gid:
            model.geom_solimp[
                gid
            ].copy()
        for gid in all_gids
    }

    print()
    print(
        "Original contact margins:"
    )

    for gid in all_gids:

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            gid,
        )

        print(
            f"  {name:<30s} "
            f"margin="
            f"{1000*old_margin[gid]:.4f} mm  "
            f"gap="
            f"{1000*old_gap[gid]:.4f} mm"
        )

    rows = []

    try:

        print()
        print("=" * 120)
        print(
            "V35E1C — TOTAL CONTACT-MARGIN SCAN"
        )
        print("=" * 120)

        print()
        print(
            f"Hardware tip-force target = "
            f"{HW_TIP_FORCE:.5f} N"
        )

        print(
            f"Frozen solref = "
            f"[-{DIRECT_K:.1f}, "
            f"-{DIRECT_DAMPING:.1f}]"
        )

        print(
            "Frozen solimp =",
            SOLIMP,
        )

        print()
        print(
            " margin    midDist   tipDist   "
            "midFn[N]  tipFn[N]  tipErr[N]  "
            "max|qacc|[deg/s2]  midPts tipPts"
        )

        print(
            "  [mm]       [mm]      [mm]"
        )

        print(
            "-" * 126
        )

        for margin_mm in MARGIN_MM_VALUES:

            d5.restore(
                model,
                data,
                target_state,
            )

            # ------------------------------------------------
            # Measured H1 pose, diagnostic only.
            # ------------------------------------------------

            for (
                angle,
                (
                    _name,
                    _jid,
                    qadr,
                    _dadr,
                ),
            ) in zip(
                HW_POSE_DEG,
                joint_info,
            ):

                data.qpos[
                    qadr
                ] = np.deg2rad(
                    angle
                )

            data.qvel[:] = 0.0

            # ------------------------------------------------
            # Freeze solver law exactly.
            # ------------------------------------------------

            for gid in all_gids:

                model.geom_solref[
                    gid
                ] = [
                    -DIRECT_K,
                    -DIRECT_DAMPING,
                ]

                model.geom_solimp[
                    gid
                ] = SOLIMP

                model.geom_gap[
                    gid
                ] = 0.0

            # ------------------------------------------------
            # Make TOTAL pair margin unambiguous:
            #
            # finger margins = 0
            # fixture margin = candidate
            #
            # MuJoCo sums the two geom margins.
            # ------------------------------------------------

            for gid in index_gids:

                model.geom_margin[
                    gid
                ] = 0.0

            model.geom_margin[
                fixture_gid
            ] = (
                margin_mm
                * 1e-3
            )

            mujoco.mj_forward(
                model,
                data,
            )

            mid_dist = d2.physical_gap(
                model,
                data,
                "collision_idx_mid_mesh",
                fixture_gid,
            )

            tip_dist = d2.physical_gap(
                model,
                data,
                "collision_idx_tip",
                fixture_gid,
            )

            (
                force,
                points,
                actual_solref,
                actual_solimp,
            ) = e1b.grouped_forces(
                model,
                data,
                fixture_gid,
            )

            mid_force = force[
                "collision_idx_mid_mesh"
            ]

            tip_force = force[
                "collision_idx_tip"
            ]

            force_error = abs(
                tip_force
                -
                HW_TIP_FORCE
            )

            qacc_deg = np.degrees(
                np.asarray(
                    [
                        data.qacc[dadr]
                        for (
                            _name,
                            _jid,
                            _qadr,
                            dadr,
                        )
                        in joint_info
                    ],
                    dtype=float,
                )
            )

            max_qacc = float(
                np.max(
                    np.abs(
                        qacc_deg
                    )
                )
            )

            print(
                f"{margin_mm:7.3f} "
                f"{1000*mid_dist:+9.4f} "
                f"{1000*tip_dist:+9.4f} "
                f"{mid_force:9.5f} "
                f"{tip_force:9.5f} "
                f"{force_error:10.5f} "
                f"{max_qacc:18.3f} "
                f"{points['collision_idx_mid_mesh']:7d} "
                f"{points['collision_idx_tip']:6d}"
            )

            rows.append(
                {
                    "margin_mm":
                        float(
                            margin_mm
                        ),

                    "mid_force":
                        float(
                            mid_force
                        ),

                    "tip_force":
                        float(
                            tip_force
                        ),

                    "error":
                        float(
                            force_error
                        ),

                    "max_qacc":
                        float(
                            max_qacc
                        ),

                    "solref":
                        actual_solref,

                    "solimp":
                        actual_solimp,
                }
            )

        selected = min(
            rows,
            key=lambda row:
                row["error"],
        )

        print()
        print("=" * 120)
        print(
            "V35E1C NEAREST STATIC FORCE MATCH"
        )
        print("=" * 120)

        print()
        print(
            f"total contact margin = "
            f"{selected['margin_mm']:.4f} mm"
        )

        print(
            f"sim tip force        = "
            f"{selected['tip_force']:.5f} N"
        )

        print(
            f"hardware target      = "
            f"{HW_TIP_FORCE:.5f} N"
        )

        print(
            f"absolute force error = "
            f"{selected['error']:.5f} N"
        )

        print(
            f"sim middle force     = "
            f"{selected['mid_force']:.5f} N"
        )

        print(
            "max instantaneous |joint acceleration| = "
            f"{selected['max_qacc']:.3f} deg/s^2"
        )

        if selected["solref"] is not None:

            print(
                "actual contact solref =",
                selected["solref"],
            )

        if selected["solimp"] is not None:

            print(
                "actual contact solimp =",
                selected["solimp"],
            )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "This is still only a static diagnostic."
        )

        print(
            "The selected margin is NOT accepted until "
            "full dynamic H1 replay validates contact onset, "
            "force, pose, and sustained wrapping."
        )

    finally:

        for gid in all_gids:

            model.geom_margin[
                gid
            ] = old_margin[
                gid
            ]

            model.geom_gap[
                gid
            ] = old_gap[
                gid
            ]

            model.geom_solref[
                gid
            ] = old_solref[
                gid
            ]

            model.geom_solimp[
                gid
            ] = old_solimp[
                gid
            ]

        contact_replay.CONTACT_DAMPING = (
            old_damping
        )

        b2.v35.CONTACT_MU = old_mu
        b2.v35.CONTACT_K = old_k

    print()
    print("=" * 120)
    print(
        "V35E1C COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
