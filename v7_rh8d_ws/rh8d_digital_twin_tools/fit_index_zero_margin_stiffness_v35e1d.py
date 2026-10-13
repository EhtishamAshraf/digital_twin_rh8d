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
# V35E1D
#
# DIRECT SOLREF STIFFNESS SCAN WITH ZERO CONTACT MARGIN
#
# VARIABLE:
#     direct-format contact stiffness only
#
# FROZEN:
#     total margin = 0
#     damping = 20
#     solimp = [0.95, 0.95, 0.001, 0.5, 2]
#     friction = 0.05
#     CAD tendon
#     V35B2 passive mechanics
#     cylinder geometry/registration
#     target-time spool/controller state
#
# Hardware joint pose is prescribed ONLY for this static audit.
# ================================================================


HW_POSE_DEG = np.array(
    [14.5355, 23.8110, 9.3270],
    dtype=float,
)

HW_TIP_FORCE = 0.14563

DIRECT_DAMPING = 20.0

SOLIMP = np.array(
    [0.95, 0.95, 0.001, 0.5, 2.0],
    dtype=float,
)


K_VALUES = [
    0.001,
    0.005,
    0.010,
    0.020,
    0.050,
    0.100,
    0.200,
    0.500,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    300.0,
    500.0,
    700.0,
    1000.0,
]


FIXTURE_NAME = "index_contact_fixture"

CONTACT_GEOMS = [
    "index_contact_fixture",
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
    b2.v35.CONTACT_K = 1000.0


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
            "V35E1D — ACQUIRING SYNCHRONIZED H1 STATE"
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
            s["time"]
            for s in states
        ],
        dtype=float,
    )

    index = int(
        np.argmin(
            np.abs(
                times
                -
                target_time
            )
        )
    )

    target_state = states[index]

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


    gids = []

    for name in CONTACT_GEOMS:

        gid = d2.oid(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )

        if gid is None:

            raise RuntimeError(
                f"Missing contact geom: {name}"
            )

        gids.append(gid)


    old_margin = {
        gid:
            float(
                model.geom_margin[gid]
            )
        for gid in gids
    }

    old_gap = {
        gid:
            float(
                model.geom_gap[gid]
            )
        for gid in gids
    }

    old_solref = {
        gid:
            model.geom_solref[gid].copy()
        for gid in gids
    }

    old_solimp = {
        gid:
            model.geom_solimp[gid].copy()
        for gid in gids
    }


    rows = []

    try:

        print()
        print("=" * 120)
        print(
            "V35E1D — ZERO-MARGIN DIRECT-STIFFNESS SCAN"
        )
        print("=" * 120)

        print()
        print(
            f"Hardware tip-force target = "
            f"{HW_TIP_FORCE:.5f} N"
        )

        print(
            "Total contact margin       = 0.000 mm"
        )

        print(
            f"Direct damping frozen      = "
            f"{DIRECT_DAMPING:.3f}"
        )

        print(
            "solimp frozen              =",
            SOLIMP,
        )

        print()
        print(
            "      k       midFn[N]   tipFn[N]   "
            "tipErr[N]   max|qacc|[deg/s2]   "
            "midPts tipPts"
        )

        print("-" * 105)


        for k_direct in K_VALUES:

            d5.restore(
                model,
                data,
                target_state,
            )

            # ------------------------------------------------
            # Prescribe measured H1 pose for this STATIC
            # diagnostic only.
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

                data.qpos[qadr] = (
                    np.deg2rad(angle)
                )

            data.qvel[:] = 0.0


            # ------------------------------------------------
            # Zero total margin + frozen impedance.
            # ------------------------------------------------

            for gid in gids:

                model.geom_margin[
                    gid
                ] = 0.0

                model.geom_gap[
                    gid
                ] = 0.0

                model.geom_solref[
                    gid
                ] = [
                    -float(k_direct),
                    -DIRECT_DAMPING,
                ]

                model.geom_solimp[
                    gid
                ] = SOLIMP


            mujoco.mj_forward(
                model,
                data,
            )


            (
                forces,
                points,
                actual_solref,
                actual_solimp,
            ) = e1b.grouped_forces(
                model,
                data,
                fixture_gid,
            )


            mid_force = forces[
                "collision_idx_mid_mesh"
            ]

            tip_force = forces[
                "collision_idx_tip"
            ]


            error = abs(
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
                f"{k_direct:9.3f} "
                f"{mid_force:11.5f} "
                f"{tip_force:11.5f} "
                f"{error:11.5f} "
                f"{max_qacc:18.3f} "
                f"{points['collision_idx_mid_mesh']:7d} "
                f"{points['collision_idx_tip']:6d}"
            )


            rows.append(
                {
                    "k":
                        float(k_direct),

                    "mid_force":
                        float(mid_force),

                    "tip_force":
                        float(tip_force),

                    "error":
                        float(error),

                    "max_qacc":
                        float(max_qacc),

                    "solref":
                        actual_solref,

                    "solimp":
                        actual_solimp,
                }
            )


        selected = min(
            rows,
            key=lambda r:
                r["error"],
        )


        print()
        print("=" * 120)
        print(
            "V35E1D NEAREST STATIC FORCE MATCH"
        )
        print("=" * 120)

        print()
        print(
            f"direct stiffness = "
            f"{selected['k']:.6f}"
        )

        print(
            f"sim tip force    = "
            f"{selected['tip_force']:.5f} N"
        )

        print(
            f"hardware target  = "
            f"{HW_TIP_FORCE:.5f} N"
        )

        print(
            f"absolute error   = "
            f"{selected['error']:.5f} N"
        )

        print(
            f"middle force     = "
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
            "No candidate is accepted here merely "
            "because its static force matches."
        )

        print(
            "A candidate must next survive the complete "
            "dynamic H1 replay."
        )


    finally:

        for gid in gids:

            model.geom_margin[
                gid
            ] = old_margin[gid]

            model.geom_gap[
                gid
            ] = old_gap[gid]

            model.geom_solref[
                gid
            ] = old_solref[gid]

            model.geom_solimp[
                gid
            ] = old_solimp[gid]


        contact_replay.CONTACT_DAMPING = (
            old_damping
        )

        b2.v35.CONTACT_MU = old_mu
        b2.v35.CONTACT_K = old_k


    print()
    print("=" * 120)
    print(
        "V35E1D COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
