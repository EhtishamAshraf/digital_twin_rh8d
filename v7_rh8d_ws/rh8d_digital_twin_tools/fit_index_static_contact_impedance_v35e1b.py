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
# V35E1B — STATIC CONTACT IMPEDANCE AUDIT
#
# VARIABLE ONLY:
#     constant solimp impedance level d
#
# FROZEN:
#     direct solref stiffness = 1000
#     direct solref damping   = 20
#     solimp width            = 0.001
#     solimp midpoint         = 0.5
#     solimp power            = 2
#     mu                      = 0.05
#     CAD tendon
#     V35B2 passives
#     cylinder pose/geometry
#     exact TARGET_BAG_TIME spool/controller state
#
# Measured MCP/PIP/DIP are prescribed ONLY for this static
# diagnostic. They are not controlled in any dynamic replay.
# ================================================================


HW_POSE_DEG = np.array(
    [14.5355, 23.8110, 9.3270],
    dtype=float,
)

HW_TIP_FORCE = 0.14563

DIRECT_K = 1000.0
DIRECT_DAMPING = 20.0

SOLIMP_DWIDTH = 0.95
SOLIMP_WIDTH = 0.001
SOLIMP_MIDPOINT = 0.5
SOLIMP_POWER = 2.0


D_VALUES = [
    0.001,
    0.002,
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
    0.700,
    0.900,
    0.950,
]


FIXTURE_NAME = "index_contact_fixture"

PAIR_GEOMS = [
    "index_contact_fixture",
    "collision_idx_prox",
    "collision_idx_mid_mesh",
    "collision_idx_tip",
]


def grouped_forces(
    model,
    data,
    fixture_gid,
):

    result = {
        "collision_idx_mid_mesh": 0.0,
        "collision_idx_tip": 0.0,
    }

    points = {
        "collision_idx_mid_mesh": 0,
        "collision_idx_tip": 0,
    }

    actual_solref = None
    actual_solimp = None

    for ci in range(data.ncon):

        contact = data.contact[ci]

        g1 = int(contact.geom1)
        g2 = int(contact.geom2)

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

        if name not in result:
            continue

        wrench = np.zeros(6)

        mujoco.mj_contactForce(
            model,
            data,
            ci,
            wrench,
        )

        result[name] += abs(
            float(wrench[0])
        )

        points[name] += 1

        if (
            name == "collision_idx_tip"
            and
            actual_solref is None
        ):

            if hasattr(contact, "solref"):
                actual_solref = np.asarray(
                    contact.solref,
                    dtype=float,
                ).copy()

            if hasattr(contact, "solimp"):
                actual_solimp = np.asarray(
                    contact.solimp,
                    dtype=float,
                ).copy()

    return (
        result,
        points,
        actual_solref,
        actual_solimp,
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
    # Capture exact target-time dynamic state.
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

        out = original_step(
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

        return out

    mujoco.mj_step = capture_step

    try:

        print()
        print("=" * 120)
        print(
            "V35E1B — ACQUIRING SYNCHRONIZED H1 STATE"
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
            "No target-time states captured."
        )

    model = captured["model"]
    data = captured["data"]

    times = np.asarray(
        [
            x["time"]
            for x in states
        ]
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
            "Missing fixture geom."
        )

    gids = []

    for name in PAIR_GEOMS:

        gid = d2.oid(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )

        if gid is not None:
            gids.append(gid)

    original_solref = {
        gid:
            model.geom_solref[gid].copy()
        for gid in gids
    }

    original_solimp = {
        gid:
            model.geom_solimp[gid].copy()
        for gid in gids
    }

    rows = []

    try:

        print()
        print("=" * 120)
        print(
            "V35E1B — CONSTANT SOLIMP IMPEDANCE SCAN"
        )
        print("=" * 120)

        print()
        print(
            f"Hardware tip-force target = "
            f"{HW_TIP_FORCE:.5f} N"
        )

        print(
            f"Frozen direct solref       = "
            f"[-{DIRECT_K:.1f}, "
            f"-{DIRECT_DAMPING:.1f}]"
        )

        print(
            "Variable ONLY              = "
            "solimp d0 only"
        )

        print(
            f"Frozen solimp shape        = "
            f"d_width {SOLIMP_DWIDTH}, width {SOLIMP_WIDTH}, "
            f"midpoint {SOLIMP_MIDPOINT}, "
            f"power {SOLIMP_POWER}"
        )

        print()
        print(
            "      d     midFn[N]   tipFn[N]   "
            "tipErr[N]   max|qacc|[deg/s2]   "
            "midPts tipPts    actual solimp"
        )

        print("-" * 126)

        for d in D_VALUES:

            d5.restore(
                model,
                data,
                target_state,
            )

            # Prescribe hardware pose only for static audit.
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

            # Set identical contact law on both surfaces so
            # contact-parameter mixing is not ambiguous.
            for gid in gids:

                model.geom_solref[gid] = [
                    -DIRECT_K,
                    -DIRECT_DAMPING,
                ]

                model.geom_solimp[gid] = [
                    float(d),
                    SOLIMP_DWIDTH,
                    SOLIMP_WIDTH,
                    SOLIMP_MIDPOINT,
                    SOLIMP_POWER,
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
            ) = grouped_forces(
                model,
                data,
                fixture_gid,
            )

            mid_fn = force[
                "collision_idx_mid_mesh"
            ]

            tip_fn = force[
                "collision_idx_tip"
            ]

            error = abs(
                tip_fn
                -
                HW_TIP_FORCE
            )

            qacc = np.degrees(
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
                    ]
                )
            )

            max_qacc = float(
                np.max(
                    np.abs(qacc)
                )
            )

            if actual_solimp is None:

                solimp_text = "NONE"

            else:

                solimp_text = (
                    "["
                    + ", ".join(
                        f"{x:.4g}"
                        for x
                        in actual_solimp
                    )
                    + "]"
                )

            print(
                f"{d:7.3f} "
                f"{mid_fn:10.5f} "
                f"{tip_fn:10.5f} "
                f"{error:11.5f} "
                f"{max_qacc:18.3f} "
                f"{points['collision_idx_mid_mesh']:7d} "
                f"{points['collision_idx_tip']:6d}    "
                f"{solimp_text}"
            )

            rows.append(
                {
                    "d": float(d),
                    "mid_force": float(mid_fn),
                    "tip_force": float(tip_fn),
                    "error": float(error),
                    "max_qacc": max_qacc,
                    "solref": actual_solref,
                    "solimp": actual_solimp,
                }
            )

        nearest = min(
            rows,
            key=lambda x:
                x["error"],
        )

        print()
        print("=" * 120)
        print(
            "V35E1B NEAREST STATIC FORCE MATCH"
        )
        print("=" * 120)

        print()
        print(
            f"d = "
            f"{nearest['d']:.4f}"
        )

        print(
            f"sim tip force = "
            f"{nearest['tip_force']:.5f} N"
        )

        print(
            f"hardware target = "
            f"{HW_TIP_FORCE:.5f} N"
        )

        print(
            f"absolute error = "
            f"{nearest['error']:.5f} N"
        )

        print(
            f"middle force = "
            f"{nearest['mid_force']:.5f} N"
        )

        print(
            f"max instantaneous |joint acceleration| = "
            f"{nearest['max_qacc']:.3f} deg/s^2"
        )

        if nearest["solref"] is not None:

            print(
                "actual contact solref =",
                nearest["solref"],
            )

        if nearest["solimp"] is not None:

            print(
                "actual contact solimp =",
                nearest["solimp"],
            )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "This is still a STATIC diagnostic."
        )

        print(
            "The nearest force match is NOT accepted "
            "until a full dynamic H1 replay is run."
        )

    finally:

        for gid in gids:

            model.geom_solref[gid] = (
                original_solref[gid]
            )

            model.geom_solimp[gid] = (
                original_solimp[gid]
            )

        contact_replay.CONTACT_DAMPING = (
            old_damping
        )

        b2.v35.CONTACT_MU = old_mu
        b2.v35.CONTACT_K = old_k

    print()
    print("=" * 120)
    print(
        "V35E1B COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
