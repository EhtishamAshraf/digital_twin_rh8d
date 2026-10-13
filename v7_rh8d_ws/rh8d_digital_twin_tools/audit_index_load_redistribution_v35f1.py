#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

import fit_index_passive_contact_v35b2 as b2
import replay_index_zero_margin_v35e2a as e2a
import audit_index_torque_budget_v35d2 as d2


# ================================================================
# V35F1 — DYNAMIC LOAD-REDISTRIBUTION AUDIT
#
# NO fitting.
# NO angle control.
# NO contact-triggered mechanics.
#
# Frozen diagnostic baseline:
#   zero contact margin
#   mu = 0.05
#   direct stiffness = 1000
#   direct damping = 20
#   CAD spatial tendon
#   V35B2 passive mechanics
#   measured hardware encoder replay
#
# Purpose:
# Determine WHY middle contact does not naturally lead to
# sustained distal wrapping.
# ================================================================


CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

FIXTURE_NAME = "index_contact_fixture"


def make_snapshot(
    t,
    data,
):

    return {
        "time":
            float(t),

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

    data.qpos[:] = (
        state["qpos"]
    )

    data.qvel[:] = (
        state["qvel"]
    )

    if data.ctrl.size:

        data.ctrl[:] = (
            state["ctrl"]
        )

    if (
        data.act.size
        and
        state["act"].size
        ==
        data.act.size
    ):

        data.act[:] = (
            state["act"]
        )

    mujoco.mj_forward(
        model,
        data,
    )


def tendon_moment_arms_mm(
    model,
    data,
    state,
    joints,
    tendon_tid,
):

    # Central finite difference.
    h = 1e-5

    values = []

    for (
        name,
        _jid,
        qadr,
        _dadr,
    ) in joints:

        restore(
            model,
            data,
            state,
        )

        data.qpos[
            qadr
        ] += h

        mujoco.mj_forward(
            model,
            data,
        )

        l_plus = float(
            data.ten_length[
                tendon_tid
            ]
        )

        restore(
            model,
            data,
            state,
        )

        data.qpos[
            qadr
        ] -= h

        mujoco.mj_forward(
            model,
            data,
        )

        l_minus = float(
            data.ten_length[
                tendon_tid
            ]
        )

        dldq = (
            l_plus
            -
            l_minus
        ) / (
            2.0 * h
        )

        # Tendon flexion moment arm convention.
        r = (
            -dldq
            *
            1000.0
        )

        values.append(
            (
                name,
                float(r),
            )
        )

    restore(
        model,
        data,
        state,
    )

    return values


def print_state_header(
    title,
    row,
):

    print()
    print("=" * 138)
    print(title)
    print("=" * 138)

    print()
    print(
        f"bag time   = "
        f"{row['time']:.6f}"
    )

    print(
        f"encoder    = "
        f"{row['encoder']:.3f} ticks"
    )

    print(
        f"middle Fn  = "
        f"{row['mid_force']:.6f} N"
    )

    print(
        f"tip Fn     = "
        f"{row['tip_force']:.6f} N"
    )

    print(
        f"middle gap = "
        f"{1000*row['mid_gap']:+.4f} mm"
    )

    print(
        f"tip gap    = "
        f"{1000*row['tip_gap']:+.4f} mm"
    )

    print(
        f"middle contact = "
        f"{row['mid_contact']}"
    )

    print(
        f"tip contact    = "
        f"{row['tip_contact']}"
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

    zero_margin_xml = (
        e2a.make_zero_margin_xml()
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


    rows = []

    refs = {
        "model": None,
        "data": None,
        "joints": None,
        "fixture": None,
        "mid_gid": None,
        "tip_gid": None,
    }


    original_step = (
        mujoco.mj_step
    )


    def capture_step(
        model,
        data,
        *step_args,
        **step_kwargs,
    ):

        out = original_step(
            model,
            data,
            *step_args,
            **step_kwargs,
        )

        local = (
            e2a.replay_locals()
        )

        if local is None:

            return out


        if refs["model"] is None:

            refs["model"] = model
            refs["data"] = data

            refs["joints"] = (
                d2.joint_addresses(
                    model
                )
            )

            refs["fixture"] = (
                d2.oid(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    FIXTURE_NAME,
                )
            )

            refs["mid_gid"] = (
                d2.oid(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    "collision_idx_mid_mesh",
                )
            )

            refs["tip_gid"] = (
                d2.oid(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    "collision_idx_tip",
                )
            )


        forces, active = (
            e2a.contact_forces(
                model,
                data,
                refs["fixture"],
            )
        )


        rows.append(
            {
                "time":
                    float(
                        local["time"]
                    ),

                "encoder":
                    float(
                        local["encoder"]
                    ),

                "mid_force":
                    float(
                        forces[
                            "collision_idx_mid_mesh"
                        ]
                    ),

                "tip_force":
                    float(
                        forces[
                            "collision_idx_tip"
                        ]
                    ),

                "mid_contact":
                    bool(
                        active[
                            "collision_idx_mid_mesh"
                        ]
                    ),

                "tip_contact":
                    bool(
                        active[
                            "collision_idx_tip"
                        ]
                    ),

                "mid_gap":
                    e2a.physical_distance(
                        model,
                        data,
                        refs["mid_gid"],
                        refs["fixture"],
                    ),

                "tip_gap":
                    e2a.physical_distance(
                        model,
                        data,
                        refs["tip_gid"],
                        refs["fixture"],
                    ),

                "state":
                    make_snapshot(
                        local["time"],
                        data,
                    ),
            }
        )

        return out


    mujoco.mj_step = capture_step


    try:

        print()
        print("=" * 138)
        print(
            "V35F1 — RUNNING ZERO-MARGIN DYNAMIC REPLAY"
        )
        print("=" * 138)

        print()
        print(
            "Frozen physics:"
        )

        print(
            "  margin = 0 mm"
        )

        print(
            f"  mu = "
            f"{CONTACT_MU}"
        )

        print(
            f"  direct stiffness = "
            f"{CONTACT_K}"
        )

        print(
            f"  direct damping = "
            f"{CONTACT_DAMPING}"
        )

        result = b2.run_candidate(
            zero_margin_xml,
            bag,
        )

    finally:

        mujoco.mj_step = (
            original_step
        )

        contact_replay.CONTACT_DAMPING = (
            old_damping
        )

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_k
        )


    if not rows:

        raise RuntimeError(
            "No dynamic states captured."
        )


    model = refs["model"]
    data = refs["data"]

    joints = refs["joints"]
    fixture_gid = refs["fixture"]


    tendon_tid, tendon_name = (
        d2.find_index_tendon(
            model
        )
    )


    # ============================================================
    # Select physically meaningful states.
    # ============================================================

    mid_indices = [
        i
        for i, row
        in enumerate(rows)
        if row["mid_contact"]
    ]

    tip_indices = [
        i
        for i, row
        in enumerate(rows)
        if row["tip_contact"]
    ]


    if not mid_indices:

        raise RuntimeError(
            "No middle contact found."
        )

    if not tip_indices:

        raise RuntimeError(
            "No tip contact found."
        )


    first_mid_i = (
        mid_indices[0]
    )

    first_tip_i = (
        tip_indices[0]
    )


    pre_mid_i = max(
        0,
        first_mid_i - 1,
    )


    mid_only_before_tip = [
        i
        for i in range(
            first_mid_i,
            first_tip_i,
        )
        if (
            rows[i][
                "mid_contact"
            ]
            and
            not rows[i][
                "tip_contact"
            ]
        )
    ]


    last_mid_only_i = (
        mid_only_before_tip[-1]
        if mid_only_before_tip
        else first_mid_i
    )


    overlap_indices = [
        i
        for i, row
        in enumerate(rows)
        if (
            row["mid_contact"]
            and
            row["tip_contact"]
        )
    ]


    if overlap_indices:

        strongest_overlap_i = max(
            overlap_indices,
            key=lambda i:
                (
                    rows[i][
                        "mid_force"
                    ]
                    +
                    rows[i][
                        "tip_force"
                    ]
                ),
        )

    else:

        strongest_overlap_i = (
            first_tip_i
        )


    target_time = float(
        contact_replay.TARGET_BAG_TIME
    )

    target_i = int(
        np.argmin(
            np.abs(
                np.asarray(
                    [
                        row["time"]
                        for row in rows
                    ],
                    dtype=float,
                )
                -
                target_time
            )
        )
    )


    selected = [
        (
            "V35F1-A — IMMEDIATELY BEFORE FIRST MIDDLE CONTACT",
            pre_mid_i,
        ),

        (
            "V35F1-B — FIRST MIDDLE CONTACT",
            first_mid_i,
        ),

        (
            "V35F1-C — LAST MIDDLE-ONLY STATE BEFORE TIP CONTACT",
            last_mid_only_i,
        ),

        (
            "V35F1-D — FIRST TIP CONTACT",
            first_tip_i,
        ),

        (
            "V35F1-E — STRONGEST SIMULTANEOUS MIDDLE+TIP LOAD",
            strongest_overlap_i,
        ),

        (
            "V35F1-F — VIDEO-ALIGNED TARGET STATE",
            target_i,
        ),

        (
            "V35F1-G — REPLAY END STATE",
            len(rows) - 1,
        ),
    ]


    print()
    print("=" * 138)
    print(
        "V35F1 — EXISTING VIDEO-ALIGNED RESULT"
    )
    print("=" * 138)

    print()
    print(
        f"SIM pose = "
        f"{result['pose'][0]:.3f} / "
        f"{result['pose'][1]:.3f} / "
        f"{result['pose'][2]:.3f} deg"
    )

    print(
        f"RMSE = "
        f"{result['rmse']:.3f} deg"
    )


    # ============================================================
    # Audit selected states.
    # ============================================================

    for title, index in selected:

        row = rows[index]

        print_state_header(
            title,
            row,
        )

        state = row["state"]


        arms = tendon_moment_arms_mm(
            model,
            data,
            state,
            joints,
            tendon_tid,
        )


        print()
        print(
            "CAD spatial-tendon instantaneous "
            "moment arms:"
        )

        for name, arm in arms:

            print(
                f"  {name:<40s} "
                f"{arm:+.4f} mm"
            )


        d2.audit_state(
            title,
            model,
            data,
            state,
            joints,
            tendon_tid,
            tendon_name,
            fixture_gid,
        )


    print()
    print("=" * 138)
    print(
        "V35F1 INTERPRETATION GUIDE"
    )
    print("=" * 138)

    print()
    print(
        "1. If tendon torque/moment arms remain similar "
        "for MCP/PIP/DIP but DIP passive opposition is "
        "much larger:"
    )

    print(
        "   -> current passive distribution is blocking "
        "distal wrapping."
    )

    print()
    print(
        "2. If PIP has strong positive net drive after "
        "middle contact while DIP has weak or negative "
        "net drive:"
    )

    print(
        "   -> the model naturally sends additional "
        "tendon displacement into PIP instead of DIP."
    )

    print()
    print(
        "3. If the tendon moment arm itself collapses at DIP "
        "after middle contact:"
    )

    print(
        "   -> revisit CAD tendon routing/transmission."
    )

    print()
    print(
        "4. If bottle-contact torque dominates and ejects "
        "the finger even with zero margin:"
    )

    print(
        "   -> collision/contact Jacobian still contributes "
        "to the wrong redistribution."
    )

    print()
    print(
        "Do not tune anything until this audit is interpreted."
    )

    print()
    print("=" * 138)
    print(
        "V35F1 COMPLETE — NO PARAMETERS FITTED"
    )
    print("=" * 138)


if __name__ == "__main__":
    main()
