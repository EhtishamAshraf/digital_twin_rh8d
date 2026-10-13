#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

import fit_index_passive_contact_v35b2 as b2
import audit_index_torque_budget_v35d2 as d2
import audit_index_contact_force_v35d5 as d5


# ================================================================
# V35E1 — STATIC CONTACT-FORCE IDENTIFICATION
#
# Identify ONE parameter only:
#
#   direct-format MuJoCo contact stiffness
#
# FROZEN:
#   CAD tendon
#   V35B2 passives
#   exact target-time spool/controller state
#   measured H1 MCP/PIP/DIP pose
#   current solimp
#   damping = 20
#   friction = 0.05
#   cylinder geometry/registration
#
# IMPORTANT:
#   k below is MuJoCo direct-solref stiffness [1/s^2],
#   NOT a physical N/m material stiffness.
#
# The H1 fingertip force is used as an effective calibration target.
# ================================================================


HW_POSE_DEG = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)

HW_TIP_FORCE = 0.14563

DIRECT_DAMPING = 20.0


K_VALUES = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
    1.50,
    2.00,
]


FIXTURE_NAME = "index_contact_fixture"

PAIR_GEOMS = [
    "index_contact_fixture",
    "collision_idx_prox",
    "collision_idx_mid_mesh",
    "collision_idx_tip",
]


def tip_force(
    model,
    data,
    fixture_gid,
):

    total_fn = 0.0
    npoints = 0

    actual_solref = None
    actual_solimp = None

    for ci in range(
        data.ncon
    ):

        contact = data.contact[
            ci
        ]

        g1 = int(
            contact.geom1
        )

        g2 = int(
            contact.geom2
        )

        if (
            g1 != fixture_gid
            and
            g2 != fixture_gid
        ):
            continue

        other = (
            g2
            if g1 == fixture_gid
            else g1
        )

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            other,
        )

        if name != "collision_idx_tip":
            continue

        wrench = np.zeros(
            6,
            dtype=float,
        )

        mujoco.mj_contactForce(
            model,
            data,
            ci,
            wrench,
        )

        total_fn += abs(
            float(
                wrench[0]
            )
        )

        npoints += 1

        if actual_solref is None:

            if hasattr(
                contact,
                "solref",
            ):

                actual_solref = np.asarray(
                    contact.solref,
                    dtype=float,
                ).copy()

            if hasattr(
                contact,
                "solimp",
            ):

                actual_solimp = np.asarray(
                    contact.solimp,
                    dtype=float,
                ).copy()

    return (
        total_fn,
        npoints,
        actual_solref,
        actual_solimp,
    )


def put_hardware_pose(
    model,
    data,
    target_state,
    joint_info,
):

    d5.restore(
        model,
        data,
        target_state,
    )

    for (
        angle_deg,
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
            angle_deg
        )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


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

    # ------------------------------------------------------------
    # Acquire exact synchronized target-time state using current
    # V35C2 selected model.
    # ------------------------------------------------------------

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

    states = []

    captured = {
        "model": None,
        "data": None,
    }

    original_step = mujoco.mj_step

    def capture_step(
        model,
        data,
        *step_args,
        **step_kwargs,
    ):

        result = original_step(
            model,
            data,
            *step_args,
            **step_kwargs,
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
            "V35E1 — ACQUIRING SYNCHRONIZED H1 STATE"
        )
        print("=" * 120)

        b2.run_candidate(
            b2.SELECTED_MODEL,
            bag,
        )

    finally:

        mujoco.mj_step = (
            original_step
        )

    if not states:

        raise RuntimeError(
            "No replay states captured."
        )

    model = captured["model"]
    data = captured["data"]

    times = np.asarray(
        [
            state["time"]
            for state in states
        ]
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

    pair_gids = []

    print()
    print(
        "Contact-pair geoms modified ONLY during "
        "this static diagnostic:"
    )

    for name in PAIR_GEOMS:

        gid = d2.oid(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )

        if gid is None:

            print(
                f"  {name}: NOT FOUND"
            )

            continue

        pair_gids.append(
            gid
        )

        print(
            f"  {name}: gid={gid}"
        )

    if fixture_gid not in pair_gids:

        pair_gids.append(
            fixture_gid
        )

    original_solref = {
        gid:
            model.geom_solref[
                gid
            ].copy()
        for gid in pair_gids
    }

    rows = []

    try:

        print()
        print("=" * 120)
        print(
            "V35E1 — STATIC FORCE SCAN"
        )
        print("=" * 120)

        print()
        print(
            f"Hardware tip-force target = "
            f"{HW_TIP_FORCE:.5f} N"
        )

        print(
            f"Direct damping frozen      = "
            f"{DIRECT_DAMPING:.3f} 1/s"
        )

        print(
            "solimp                      = FROZEN"
        )

        print()
        print(
            " k[1/s^2]   tip Fn[N]    error[N] "
            " points      actual contact solref"
        )

        print(
            "-" * 92
        )

        for k_direct in K_VALUES:

            put_hardware_pose(
                model,
                data,
                target_state,
                joint_info,
            )

            # ------------------------------------------------
            # Give both surfaces the same effective pair law.
            # This avoids ambiguity from geom parameter mixing
            # during this identification diagnostic.
            # ------------------------------------------------

            for gid in pair_gids:

                model.geom_solref[
                    gid
                ] = [
                    -float(
                        k_direct
                    ),
                    -DIRECT_DAMPING,
                ]

            mujoco.mj_forward(
                model,
                data,
            )

            (
                force,
                points,
                actual_solref,
                actual_solimp,
            ) = tip_force(
                model,
                data,
                fixture_gid,
            )

            error = abs(
                force
                -
                HW_TIP_FORCE
            )

            if actual_solref is None:

                solref_text = (
                    "NONE"
                )

            else:

                solref_text = (
                    f"["
                    f"{actual_solref[0]:.3f}, "
                    f"{actual_solref[1]:.3f}"
                    f"]"
                )

            print(
                f"{k_direct:9.2f} "
                f"{force:11.5f} "
                f"{error:11.5f} "
                f"{points:6d}      "
                f"{solref_text}"
            )

            rows.append(
                {
                    "k":
                        float(
                            k_direct
                        ),

                    "force":
                        float(
                            force
                        ),

                    "error":
                        float(
                            error
                        ),

                    "points":
                        int(
                            points
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
                row[
                    "error"
                ],
        )

        print()
        print("=" * 120)
        print(
            "V35E1 SELECTED STATIC FORCE MATCH"
        )
        print("=" * 120)

        print()
        print(
            f"direct stiffness = "
            f"{selected['k']:.3f} 1/s^2"
        )

        print(
            f"direct damping   = "
            f"{DIRECT_DAMPING:.3f} 1/s"
        )

        print(
            f"sim tip force    = "
            f"{selected['force']:.5f} N"
        )

        print(
            f"hardware target  = "
            f"{HW_TIP_FORCE:.5f} N"
        )

        print(
            f"absolute error   = "
            f"{selected['error']:.5f} N"
        )

        if selected["solref"] is not None:

            print()
            print(
                "Actual contact solref:"
            )

            print(
                selected[
                    "solref"
                ]
            )

        if selected["solimp"] is not None:

            print()
            print(
                "Actual contact solimp "
                "(NOT fitted in E1):"
            )

            print(
                selected[
                    "solimp"
                ]
            )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "This is ONLY a static force identification."
        )

        print(
            "It is NOT yet accepted as the final "
            "dynamic contact law."
        )

        print()
        print(
            "Next experiment must replay the full H1 "
            "trajectory with this value and test:"
        )

        print(
            "  - bounce"
        )

        print(
            "  - sustained middle+tip contact"
        )

        print(
            "  - pose RMSE"
        )

        print(
            "  - contact onset"
        )

        print(
            "  - force trajectory"
        )

    finally:

        for gid, value in (
            original_solref.items()
        ):

            model.geom_solref[
                gid
            ] = value

        contact_replay.CONTACT_DAMPING = (
            old_damping
        )

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_k
        )

    print()
    print("=" * 120)
    print(
        "V35E1 COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
