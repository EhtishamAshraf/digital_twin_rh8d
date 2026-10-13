#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

import fit_index_passive_contact_v35b2 as b2
import audit_index_torque_budget_v35d2 as d2


# ================================================================
# V35D5
#
# CONTACT-FORCE AUDIT AT EXACT SYNCHRONIZED H1 STATE
#
# NO fitting.
# NO parameter changes.
#
# 1. Replay exact current model.
# 2. Capture TARGET_BAG_TIME state.
# 3. Preserve its spool/controller state.
# 4. Replace ONLY MCP/PIP/DIP with measured hardware pose.
# 5. qdot = 0.
# 6. Evaluate MuJoCo contact forces.
#
# Hardware H1 fingertip force during hold ~0.144 N.
# Important: this corresponds most directly to distal/tip loading,
# not necessarily total middle+tip normal force.
# ================================================================


HW_DEG = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

FIXTURE_NAME = "index_contact_fixture"


def snapshot(
    t,
    data,
):

    return {
        "time": float(t),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "ctrl": data.ctrl.copy(),
        "act": (
            data.act.copy()
            if data.act.size
            else np.array(
                [],
                dtype=float,
            )
        ),
    }


def restore(
    model,
    data,
    state,
):

    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]

    if data.ctrl.size:
        data.ctrl[:] = state["ctrl"]

    if (
        data.act.size
        and
        state["act"].size == data.act.size
    ):
        data.act[:] = state["act"]

    mujoco.mj_forward(
        model,
        data,
    )


def contact_force_report(
    model,
    data,
    fixture_gid,
):

    grouped = {}

    print()
    print("INDIVIDUAL CYLINDER CONTACT POINTS")
    print("-" * 100)

    found = False

    for ci in range(
        data.ncon
    ):

        contact = data.contact[ci]

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

        found = True

        other_gid = (
            g2
            if g1 == fixture_gid
            else g1
        )

        other_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            other_gid,
        )

        if other_name is None:
            other_name = (
                f"geom_{other_gid}"
            )

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

        normal = abs(
            float(
                wrench[0]
            )
        )

        tangent = float(
            np.linalg.norm(
                wrench[
                    1:3
                ]
            )
        )

        grouped.setdefault(
            other_name,
            {
                "normal":
                    0.0,

                "tangent":
                    0.0,

                "points":
                    0,
            },
        )

        grouped[
            other_name
        ]["normal"] += normal

        grouped[
            other_name
        ]["tangent"] += tangent

        grouped[
            other_name
        ]["points"] += 1

        print(
            f"contact {ci:3d}  "
            f"{other_name:<30s} "
            f"dist={1000*contact.dist:+8.4f} mm  "
            f"Fn={normal:10.5f} N  "
            f"Ft={tangent:10.5f} N"
        )

    if not found:

        print(
            "NONE"
        )

    print()
    print(
        "GROUPED CYLINDER CONTACT FORCE"
    )

    print("-" * 100)

    total_normal = 0.0
    total_tangent = 0.0

    for name, values in grouped.items():

        total_normal += (
            values[
                "normal"
            ]
        )

        total_tangent += (
            values[
                "tangent"
            ]
        )

        print(
            f"{name:<30s} "
            f"points={values['points']:2d}  "
            f"Fn_sum={values['normal']:10.5f} N  "
            f"Ft_sum={values['tangent']:10.5f} N"
        )

    print("-" * 100)

    print(
        f"{'TOTAL':<30s} "
        f"Fn_sum={total_normal:10.5f} N  "
        f"Ft_sum={total_tangent:10.5f} N"
    )

    return grouped


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
        CONTACT_DAMPING
    )

    b2.v35.CONTACT_MU = (
        CONTACT_MU
    )

    b2.v35.CONTACT_K = (
        CONTACT_K
    )

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
                snapshot(
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
        print("=" * 110)
        print(
            "V35D5 — ACQUIRING EXACT TARGET-TIME STATE"
        )
        print("=" * 110)

        result = b2.run_candidate(
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

    target = states[index]

    print()
    print(
        f"TARGET_BAG_TIME = "
        f"{target_time:.6f}"
    )

    print(
        f"Captured time   = "
        f"{target['time']:.6f}"
    )

    print(
        f"Timing error    = "
        f"{1000*abs(target['time']-target_time):.3f} ms"
    )

    joint_info = (
        d2.joint_addresses(
            model
        )
    )

    fixture_gid = d2.oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FIXTURE_NAME,
    )

    if fixture_gid is None:

        raise RuntimeError(
            "Missing steel-cylinder fixture."
        )

    # ============================================================
    # A — actual dynamic simulation at target time.
    # ============================================================

    restore(
        model,
        data,
        target,
    )

    print()
    print("=" * 110)
    print(
        "V35D5-A — ACTUAL SIMULATION AT TARGET TIME"
    )
    print("=" * 110)

    q = d2.qpos_deg(
        data,
        joint_info,
    )

    print()
    print(
        "q = "
        f"{q[0]:.4f} / "
        f"{q[1]:.4f} / "
        f"{q[2]:.4f} deg"
    )

    print(
        "middle gap = "
        f"{1000*d2.physical_gap(model,data,'collision_idx_mid_mesh',fixture_gid):+.4f} mm"
    )

    print(
        "tip gap    = "
        f"{1000*d2.physical_gap(model,data,'collision_idx_tip',fixture_gid):+.4f} mm"
    )

    contact_force_report(
        model,
        data,
        fixture_gid,
    )

    # ============================================================
    # B — measured hardware pose, SAME target-time spool/controller.
    # ============================================================

    restore(
        model,
        data,
        target,
    )

    for (
        hw_deg,
        (
            name,
            jid,
            qadr,
            dadr,
        ),
    ) in zip(
        HW_DEG,
        joint_info,
    ):

        data.qpos[
            qadr
        ] = np.deg2rad(
            hw_deg
        )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    print()
    print("=" * 110)
    print(
        "V35D5-B — HARDWARE POSE AT SAME TARGET-TIME SPOOL STATE"
    )
    print("=" * 110)

    print()
    print(
        "MCP/PIP/DIP = "
        f"{HW_DEG[0]:.4f} / "
        f"{HW_DEG[1]:.4f} / "
        f"{HW_DEG[2]:.4f} deg"
    )

    print(
        "middle gap = "
        f"{1000*d2.physical_gap(model,data,'collision_idx_mid_mesh',fixture_gid):+.4f} mm"
    )

    print(
        "tip gap    = "
        f"{1000*d2.physical_gap(model,data,'collision_idx_tip',fixture_gid):+.4f} mm"
    )

    grouped = contact_force_report(
        model,
        data,
        fixture_gid,
    )

    print()
    print("=" * 110)
    print(
        "V35D5 INTERPRETATION"
    )
    print("=" * 110)

    print()
    print(
        "Hardware fingertip force during the H1 hold "
        "is approximately 0.144 N."
    )

    print()
    print(
        "Compare that primarily against the simulated "
        "collision_idx_tip normal-force sum."
    )

    print()
    print(
        "If simulated tip normal force is many times larger,"
    )

    print(
        "the present MuJoCo contact law is over-reactive "
        "and explains the bounce."
    )

    print()
    print(
        "Do NOT interpret CONTACT_STIFFNESS=1000 as a "
        "measured 1000 N/m material stiffness."
    )

    contact_replay.CONTACT_DAMPING = (
        old_damping
    )

    b2.v35.CONTACT_MU = old_mu
    b2.v35.CONTACT_K = old_k


if __name__ == "__main__":
    main()
