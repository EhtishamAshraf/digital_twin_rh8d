#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

import fit_index_passive_contact_v35b2 as b2
import audit_index_torque_budget_v35d2 as d2


# ================================================================
# V35D3 — POST-HOLD SETTLING TEST
#
# Run exact V35C2 selected replay until hardware HOLD_END.
#
# THEN:
#   keep the final spool/controller command FIXED
#   keep upstream support FIXED
#   continue MuJoCo physics for another 5 seconds
#
# NO:
#   parameter fitting
#   angle commands
#   contact-triggered rules
#   tendon switching
#
# Purpose:
#   determine the true equilibrium of the current model.
# ================================================================


CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

EXTRA_TIME = 5.0

SAMPLE_TIMES = [
    0.0,
    0.25,
    0.50,
    1.00,
    2.00,
    3.00,
    5.00,
]

FIXTURE_NAME = "index_contact_fixture"


def snapshot_metrics(
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

    middle_gap = d2.physical_gap(
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

    middle_active = (
        "collision_idx_mid_mesh"
        in contacts
    )

    tip_active = (
        "collision_idx_tip"
        in contacts
    )

    return {
        "time": float(elapsed),
        "q": q.copy(),
        "qdot": qdot.copy(),
        "middle_gap": float(middle_gap),
        "tip_gap": float(tip_gap),
        "middle_contact": middle_active,
        "tip_contact": tip_active,
        "contacts": list(contacts),
    }


def print_row(
    row,
):

    q = row["q"]
    qd = row["qdot"]

    print(
        f"{row['time']:7.3f}  "
        f"{q[0]:8.3f} "
        f"{q[1]:8.3f} "
        f"{q[2]:8.3f}   "
        f"{qd[0]:+8.3f} "
        f"{qd[1]:+8.3f} "
        f"{qd[2]:+8.3f}   "
        f"{1000*row['middle_gap']:+8.3f} "
        f"{1000*row['tip_gap']:+8.3f}   "
        f"{'YES' if row['middle_contact'] else ' no'}   "
        f"{'YES' if row['tip_contact'] else ' no'}"
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
            f"Missing bag:\n"
            f"{contact_bag}"
        )

    if not b2.SELECTED_MODEL.exists():

        raise RuntimeError(
            "Missing V35B2 selected model:\n"
            f"{b2.SELECTED_MODEL}"
        )

    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
    )

    # ------------------------------------------------------------
    # Exact V35C2 selected contact law.
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Capture the model/data objects from the normal replay.
    # ------------------------------------------------------------

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
            "V35D3 — NORMAL HARDWARE REPLAY TO HOLD_END"
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
            "Could not capture final replay state."
        )

    # Keep the FINAL hardware/spool command exactly fixed.
    final_ctrl = data.ctrl.copy()

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
            "Missing cylinder fixture geom."
        )

    print()
    print(
        "Video-aligned evaluator result:"
    )

    print(
        f"  MCP/PIP/DIP = "
        f"{result['pose'][0]:.3f} / "
        f"{result['pose'][1]:.3f} / "
        f"{result['pose'][2]:.3f} deg"
    )

    print(
        f"  RMSE = "
        f"{result['rmse']:.3f} deg"
    )

    print()
    print("=" * 120)
    print(
        "V35D3 — EXTENDING PHYSICS AFTER HOLD_END"
    )
    print("=" * 120)

    print()
    print(
        "The hardware encoder/spool command is now frozen."
    )

    print(
        "No new commands are introduced."
    )

    print(
        f"Extra simulation time = "
        f"{EXTRA_TIME:.1f} s"
    )

    print()

    print(
        " time[s]    MCP      PIP      DIP       "
        "dMCP     dPIP     dDIP     "
        "midGap   tipGap    MID   TIP"
    )

    print(
        "-" * 112
    )

    # ------------------------------------------------------------
    # Upstream support must remain fixed exactly as in V30/V35.
    # ------------------------------------------------------------

    clamp = b2.v30.FixedSupport()

    clamp.install()

    rows = []

    try:

        # Initial state = exact replay HOLD_END.
        mujoco.mj_forward(
            model,
            data,
        )

        row0 = snapshot_metrics(
            model,
            data,
            joint_info,
            fixture_gid,
            0.0,
        )

        rows.append(row0)
        print_row(row0)

        dt = float(
            model.opt.timestep
        )

        steps = int(
            round(
                EXTRA_TIME
                /
                dt
            )
        )

        sample_index = 1

        middle_time = 0.0
        tip_time = 0.0
        overlap_time = 0.0

        middle_transitions = 0
        tip_transitions = 0

        previous_middle = (
            row0["middle_contact"]
        )

        previous_tip = (
            row0["tip_contact"]
        )

        for step in range(
            1,
            steps + 1,
        ):

            # Freeze controller/spool command.
            data.ctrl[:] = final_ctrl

            mujoco.mj_step(
                model,
                data,
            )

            elapsed = step * dt

            current = snapshot_metrics(
                model,
                data,
                joint_info,
                fixture_gid,
                elapsed,
            )

            if current["middle_contact"]:
                middle_time += dt

            if current["tip_contact"]:
                tip_time += dt

            if (
                current["middle_contact"]
                and
                current["tip_contact"]
            ):
                overlap_time += dt

            if (
                current["middle_contact"]
                !=
                previous_middle
            ):

                middle_transitions += 1

            if (
                current["tip_contact"]
                !=
                previous_tip
            ):

                tip_transitions += 1

            previous_middle = (
                current["middle_contact"]
            )

            previous_tip = (
                current["tip_contact"]
            )

            while (
                sample_index
                <
                len(SAMPLE_TIMES)
                and
                elapsed
                >=
                SAMPLE_TIMES[
                    sample_index
                ]
                -
                0.5 * dt
            ):

                row = snapshot_metrics(
                    model,
                    data,
                    joint_info,
                    fixture_gid,
                    elapsed,
                )

                rows.append(
                    row
                )

                print_row(
                    row
                )

                sample_index += 1

        final = snapshot_metrics(
            model,
            data,
            joint_info,
            fixture_gid,
            EXTRA_TIME,
        )

    finally:

        clamp.restore()

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
        "V35D3 POST-HOLD SUMMARY"
    )
    print("=" * 120)

    print()

    print(
        f"Middle contact during extra 5 s = "
        f"{middle_time:.3f} s"
    )

    print(
        f"Tip contact during extra 5 s    = "
        f"{tip_time:.3f} s"
    )

    print(
        f"Simultaneous middle+tip         = "
        f"{overlap_time:.3f} s"
    )

    print(
        f"Middle contact transitions      = "
        f"{middle_transitions}"
    )

    print(
        f"Tip contact transitions         = "
        f"{tip_transitions}"
    )

    print()

    print(
        "Final 5-s extended state:"
    )

    print(
        "  MCP/PIP/DIP = "
        f"{final['q'][0]:.3f} / "
        f"{final['q'][1]:.3f} / "
        f"{final['q'][2]:.3f} deg"
    )

    print(
        "  qdot = "
        f"{final['qdot'][0]:+.4f} / "
        f"{final['qdot'][1]:+.4f} / "
        f"{final['qdot'][2]:+.4f} deg/s"
    )

    print(
        f"  middle gap = "
        f"{1000*final['middle_gap']:+.3f} mm"
    )

    print(
        f"  tip gap    = "
        f"{1000*final['tip_gap']:+.3f} mm"
    )

    print(
        "  middle contact = "
        f"{final['middle_contact']}"
    )

    print(
        "  tip contact    = "
        f"{final['tip_contact']}"
    )

    max_speed = float(
        np.max(
            np.abs(
                final["qdot"]
            )
        )
    )

    print()

    print(
        f"Maximum final joint speed = "
        f"{max_speed:.4f} deg/s"
    )

    print()
    print("=" * 120)
    print(
        "V35D3 INTERPRETATION"
    )
    print("=" * 120)

    print()
    print(
        "A) If contact returns and becomes sustained:"
    )

    print(
        "   Current mechanics can wrap, but the dynamic "
        "settling time/damping is wrong."
    )

    print()
    print(
        "B) If it converges to a stable air gap:"
    )

    print(
        "   The model has the wrong static equilibrium; "
        "next audit tendon rest length/preload and "
        "passive hysteresis."
    )

    print()
    print(
        "C) If it repeatedly contacts/detaches for 5 s:"
    )

    print(
        "   Contact/passive damping dynamics remain wrong."
    )

    print()
    print(
        "D) If it is still moving significantly after 5 s:"
    )

    print(
        "   Joint/tendon dynamic damping is structurally "
        "too weak before we discuss equilibrium."
    )

    print()
    print("=" * 120)
    print(
        "V35D3 COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
