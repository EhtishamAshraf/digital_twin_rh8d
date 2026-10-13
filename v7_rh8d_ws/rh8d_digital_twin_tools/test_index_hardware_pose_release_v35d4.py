#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import fit_index_passive_contact_v35b2 as b2
import audit_index_torque_budget_v35d2 as d2


# ================================================================
# V35D4 — RELEASE FROM MEASURED HARDWARE CONFIGURATION
#
# Diagnostic only.
#
# 1. Run normal V35C2 replay to obtain the REAL final
#    encoder/spool/controller state.
#
# 2. Preserve that complete state.
#
# 3. Replace ONLY MCP/PIP/DIP by the measured H1 angles.
#
# 4. Set velocity = 0.
#
# 5. Release the finger with NO angle control for 5 s.
#
# If it leaves the measured configuration, the current
# static mechanics do not support the real hardware equilibrium.
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

RUN_TIME = 5.0

SAMPLE_TIMES = [
    0.0,
    0.10,
    0.25,
    0.50,
    1.00,
    2.00,
    3.00,
    5.00,
]

FIXTURE_NAME = "index_contact_fixture"


def metrics(
    model,
    data,
    joint_info,
    fixture_gid,
    elapsed,
):

    q = d2.qpos_deg(
        data,
        joint_info,
    )

    qdot = d2.qvel_deg(
        data,
        joint_info,
    )

    contacts = d2.contact_names(
        model,
        data,
        fixture_gid,
    )

    mid_gap = d2.physical_gap(
        model,
        data,
        "collision_idx_mid_mesh",
        fixture_gid,
    )

    tip_gap = d2.physical_gap(
        model,
        data,
        "collision_idx_tip",
        fixture_gid,
    )

    return {
        "t":
            float(elapsed),

        "q":
            q.copy(),

        "qdot":
            qdot.copy(),

        "mid_gap":
            float(mid_gap),

        "tip_gap":
            float(tip_gap),

        "mid":
            (
                "collision_idx_mid_mesh"
                in contacts
            ),

        "tip":
            (
                "collision_idx_tip"
                in contacts
            ),
    }


def print_row(
    row,
):

    q = row["q"]
    qd = row["qdot"]

    print(
        f"{row['t']:6.2f}   "
        f"{q[0]:8.3f} "
        f"{q[1]:8.3f} "
        f"{q[2]:8.3f}   "
        f"{qd[0]:+8.3f} "
        f"{qd[1]:+8.3f} "
        f"{qd[2]:+8.3f}   "
        f"{1000*row['mid_gap']:+8.3f} "
        f"{1000*row['tip_gap']:+8.3f}   "
        f"{'YES' if row['mid'] else ' no'}   "
        f"{'YES' if row['tip'] else ' no'}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    args = parser.parse_args()

    contact_bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )

    if not contact_bag.exists():

        raise RuntimeError(
            f"Missing bag:\n{contact_bag}"
        )

    if not b2.SELECTED_MODEL.exists():

        raise RuntimeError(
            "Missing V35B2 selected model."
        )

    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
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

        captured["model"] = model
        captured["data"] = data

        return result

    mujoco.mj_step = capture_step

    try:

        print()
        print("=" * 120)
        print(
            "V35D4 — ACQUIRING FINAL HARDWARE/SPool STATE"
        )
        print("=" * 120)

        result = b2.run_candidate(
            b2.SELECTED_MODEL,
            contact_bag,
        )

    finally:

        mujoco.mj_step = original_step

    model = captured["model"]
    data = captured["data"]

    if model is None or data is None:

        raise RuntimeError(
            "Could not capture replay state."
        )

    final_ctrl = (
        data.ctrl.copy()
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
            "Missing bottle fixture."
        )

    # ------------------------------------------------------------
    # Put ONLY MCP/PIP/DIP at measured hardware pose.
    #
    # All other qpos, including the spool position, remain exactly
    # as obtained from the real hardware replay.
    # ------------------------------------------------------------

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

    # Static-release test.
    data.qvel[:] = 0.0

    if data.ctrl.size:

        data.ctrl[:] = final_ctrl

    mujoco.mj_forward(
        model,
        data,
    )

    initial_snapshot = {
        "time":
            0.0,

        "qpos":
            data.qpos.copy(),

        "qvel":
            data.qvel.copy(),

        "ctrl":
            data.ctrl.copy(),

        "act":
            (
                data.act.copy()
                if data.act.size
                else np.array([])
            ),
    }

    tendon_tid, tendon_name = (
        d2.find_index_tendon(
            model
        )
    )

    print()
    print("=" * 120)
    print(
        "V35D4-A — TORQUE BUDGET AT HARDWARE POSE BEFORE RELEASE"
    )
    print("=" * 120)

    d2.audit_state(
        "MEASURED H1 HARDWARE POSE — t = 0 BEFORE RELEASE",
        model,
        data,
        initial_snapshot,
        joint_info,
        tendon_tid,
        tendon_name,
        fixture_gid,
    )

    # audit_state restores the snapshot.
    mujoco.mj_forward(
        model,
        data,
    )

    print()
    print("=" * 120)
    print(
        "V35D4-B — FREE RELEASE FROM HARDWARE POSE"
    )
    print("=" * 120)

    print()
    print(
        "MCP/PIP/DIP are NOT controlled after t=0."
    )

    print(
        "Final hardware spool/controller command remains frozen."
    )

    print()
    print(
        " time    MCP      PIP      DIP       "
        "dMCP     dPIP     dDIP      "
        "midGap   tipGap    MID   TIP"
    )

    print(
        "-" * 112
    )

    clamp = b2.v30.FixedSupport()
    clamp.install()

    dt = float(
        model.opt.timestep
    )

    steps = int(
        round(
            RUN_TIME / dt
        )
    )

    next_sample = 0

    middle_time = 0.0
    tip_time = 0.0
    overlap_time = 0.0

    try:

        # Native MuJoCo GUI.
        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:

            for step in range(
                steps + 1
            ):

                elapsed = (
                    step * dt
                )

                if (
                    next_sample
                    <
                    len(
                        SAMPLE_TIMES
                    )
                    and
                    elapsed
                    >=
                    SAMPLE_TIMES[
                        next_sample
                    ]
                    -
                    0.5 * dt
                ):

                    row = metrics(
                        model,
                        data,
                        joint_info,
                        fixture_gid,
                        elapsed,
                    )

                    print_row(
                        row
                    )

                    next_sample += 1

                if step == steps:

                    break

                if data.ctrl.size:

                    data.ctrl[:] = (
                        final_ctrl
                    )

                mujoco.mj_step(
                    model,
                    data,
                )

                current = metrics(
                    model,
                    data,
                    joint_info,
                    fixture_gid,
                    elapsed + dt,
                )

                if current["mid"]:

                    middle_time += dt

                if current["tip"]:

                    tip_time += dt

                if (
                    current["mid"]
                    and
                    current["tip"]
                ):

                    overlap_time += dt

                viewer.sync()

    finally:

        clamp.restore()

    final_row = metrics(
        model,
        data,
        joint_info,
        fixture_gid,
        RUN_TIME,
    )

    final_snapshot = {
        "time":
            RUN_TIME,

        "qpos":
            data.qpos.copy(),

        "qvel":
            data.qvel.copy(),

        "ctrl":
            data.ctrl.copy(),

        "act":
            (
                data.act.copy()
                if data.act.size
                else np.array([])
            ),
    }

    print()
    print("=" * 120)
    print(
        "V35D4-C — FINAL RELEASE STATE"
    )
    print("=" * 120)

    print()
    print(
        "Initial hardware pose:"
    )

    print(
        f"  {HW_DEG[0]:.3f} / "
        f"{HW_DEG[1]:.3f} / "
        f"{HW_DEG[2]:.3f} deg"
    )

    print(
        "After 5 s:"
    )

    print(
        f"  {final_row['q'][0]:.3f} / "
        f"{final_row['q'][1]:.3f} / "
        f"{final_row['q'][2]:.3f} deg"
    )

    delta = (
        final_row["q"]
        -
        HW_DEG
    )

    print(
        "Change:"
    )

    print(
        f"  {delta[0]:+.3f} / "
        f"{delta[1]:+.3f} / "
        f"{delta[2]:+.3f} deg"
    )

    print()
    print(
        f"Middle contact time = "
        f"{middle_time:.3f} s"
    )

    print(
        f"Tip contact time = "
        f"{tip_time:.3f} s"
    )

    print(
        f"Middle+tip overlap = "
        f"{overlap_time:.3f} s"
    )

    print(
        f"Final middle gap = "
        f"{1000*final_row['mid_gap']:+.3f} mm"
    )

    print(
        f"Final tip gap = "
        f"{1000*final_row['tip_gap']:+.3f} mm"
    )

    print()
    print("=" * 120)
    print(
        "V35D4-D — TORQUE BUDGET AFTER 5 s RELEASE"
    )
    print("=" * 120)

    d2.audit_state(
        "FINAL STATE AFTER RELEASE",
        model,
        data,
        final_snapshot,
        joint_info,
        tendon_tid,
        tendon_name,
        fixture_gid,
    )

    print()
    print("=" * 120)
    print(
        "V35D4 INTERPRETATION"
    )
    print("=" * 120)

    print()
    print(
        "If measured H1 pose rapidly evolves toward:"
    )

    print(
        "  lower MCP + larger PIP"
    )

    print(
        "then V35B2's passive distribution is not a "
        "physical static model, despite low one-frame RMSE."
    )

    print()
    print(
        "If the pose remains close and contact remains sustained,"
    )

    print(
        "then static mechanics are adequate and the remaining "
        "problem is the dynamic path/hysteresis."
    )

    print()

    contact_replay.CONTACT_DAMPING = (
        old_damping
    )

    b2.v35.CONTACT_MU = old_mu
    b2.v35.CONTACT_K = old_k


if __name__ == "__main__":
    main()
