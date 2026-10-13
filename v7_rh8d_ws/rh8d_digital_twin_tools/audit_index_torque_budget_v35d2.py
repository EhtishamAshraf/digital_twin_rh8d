#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import math
from pathlib import Path

import mujoco
import numpy as np

import fit_index_passive_contact_v35b2 as b2


# =====================================================================
# V35D2
#
# TORQUE-BUDGET AUDIT
#
# NO identification.
# NO parameter scan.
# NO prescribed MCP/PIP/DIP.
#
# We replay the exact V35C2 selected model:
#
#   CAD tendon
#   V35B2 passives
#   mu = 0.05
#   contact stiffness = 1000 N/m
#   contact damping = 20
#
# and capture the ACTUAL dynamic state at:
#
#   1. video-aligned TARGET_BAG_TIME
#   2. HOLD_END
#
# Then we audit generalized joint forces at exactly those states.
# =====================================================================


CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0


FIXTURE_NAME = (
    "index_contact_fixture"
)


INDEX_TENDON_CANDIDATES = [
    "idx_flex_spatial",
    "v35_cad_spatial",
    "v35_index_cad_spatial",
]


def oid(
    model,
    objtype,
    name,
):

    value = mujoco.mj_name2id(
        model,
        objtype,
        name,
    )

    if value < 0:

        return None

    return int(
        value
    )


def find_index_tendon(
    model,
):

    # ---------------------------------------------------------
    # First try expected exact names.
    # ---------------------------------------------------------

    for name in INDEX_TENDON_CANDIDATES:

        tid = oid(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            name,
        )

        if tid is not None:

            return tid, name


    # ---------------------------------------------------------
    # Fallback: inspect all tendon names.
    # ---------------------------------------------------------

    possible = []


    for tid in range(
        model.ntendon
    ):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            tid,
        )

        if not name:

            continue


        lower = name.lower()


        if (
            "idx" in lower
            or
            "index" in lower
            or
            "v35" in lower
            or
            "cad" in lower
        ):

            possible.append(
                (
                    tid,
                    name,
                )
            )


    if len(
        possible
    ) == 1:

        return possible[
            0
        ]


    print()
    print(
        "Available tendons:"
    )


    for tid in range(
        model.ntendon
    ):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            tid,
        )

        print(
            f"  {tid:3d}: {name}"
        )


    raise RuntimeError(
        "Could not uniquely identify the "
        "index CAD tendon."
    )


def joint_addresses(
    model,
):

    names = [
        b2.v35.MCP,
        b2.v35.PIP,
        b2.v35.DIP,
    ]


    result = []


    for name in names:

        jid = oid(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )


        if jid is None:

            raise RuntimeError(
                f"Missing joint: {name}"
            )


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


        result.append(
            (
                name,
                jid,
                qadr,
                dadr,
            )
        )


    return result


def find_replay_frame_time():

    # ---------------------------------------------------------
    # Called from inside the mj_step hook.
    #
    # Walk upward through the Python call stack until we find
    # replay_index_tabletop_hardware_contact_v1.main(), where
    # the local variable `t` is the actual hardware bag time.
    # ---------------------------------------------------------

    frame = inspect.currentframe()


    if frame is None:

        return None


    frame = frame.f_back


    while frame is not None:

        module_name = frame.f_globals.get(
            "__name__",
            "",
        )


        function_name = (
            frame.f_code.co_name
        )


        if (
            module_name
            ==
            "replay_index_tabletop_hardware_contact_v1"
            and
            function_name
            ==
            "main"
            and
            "t"
            in frame.f_locals
        ):

            try:

                return float(
                    frame.f_locals[
                        "t"
                    ]
                )

            except Exception:

                return None


        frame = frame.f_back


    return None


def copy_snapshot(
    t,
    model,
    data,
):

    return {
        "time":
            float(
                t
            ),

        "qpos":
            data.qpos.copy(),

        "qvel":
            data.qvel.copy(),

        "ctrl":
            data.ctrl.copy(),

        "act":
            data.act.copy()
            if data.act.size
            else np.array(
                [],
                dtype=float,
            ),
    }


def restore_state(
    model,
    data,
    snapshot,
):

    data.qpos[:] = (
        snapshot[
            "qpos"
        ]
    )

    data.qvel[:] = (
        snapshot[
            "qvel"
        ]
    )


    if data.ctrl.size:

        data.ctrl[:] = (
            snapshot[
                "ctrl"
            ]
        )


    if (
        data.act.size
        and
        snapshot[
            "act"
        ].size
        ==
        data.act.size
    ):

        data.act[:] = (
            snapshot[
                "act"
            ]
        )


    mujoco.mj_forward(
        model,
        data,
    )


def vector_at_dofs(
    array,
    joint_info,
):

    return np.asarray(
        [
            float(
                array[
                    dadr
                ]
            )
            for (
                _,
                _,
                _,
                dadr,
            )
            in joint_info
        ],
        dtype=float,
    )


def qpos_deg(
    data,
    joint_info,
):

    return np.degrees(
        np.asarray(
            [
                data.qpos[
                    qadr
                ]
                for (
                    _,
                    _,
                    qadr,
                    _,
                )
                in joint_info
            ],
            dtype=float,
        )
    )


def qvel_deg(
    data,
    joint_info,
):

    return np.degrees(
        np.asarray(
            [
                data.qvel[
                    dadr
                ]
                for (
                    _,
                    _,
                    _,
                    dadr,
                )
                in joint_info
            ],
            dtype=float,
        )
    )


def contact_names(
    model,
    data,
    fixture_gid,
):

    names = []


    for index in range(
        data.ncon
    ):

        contact = data.contact[
            index
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


        if name is None:

            name = f"geom_{other}"


        if name not in names:

            names.append(
                name
            )


    return names


def physical_gap(
    model,
    data,
    geom_name,
    fixture_gid,
):

    gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )


    if gid is None:

        return math.nan


    fromto = np.zeros(
        6,
        dtype=float,
    )


    distance = mujoco.mj_geomDistance(
        model,
        data,
        gid,
        fixture_gid,
        0.100,
        fromto,
    )


    return float(
        distance
    )


def audit_state(
    label,
    model,
    data,
    snapshot,
    joint_info,
    tendon_tid,
    tendon_name,
    fixture_gid,
):

    # =========================================================
    # BASELINE AT THE EXACT RECORDED STATE
    # =========================================================

    restore_state(
        model,
        data,
        snapshot,
    )


    q_deg = qpos_deg(
        data,
        joint_info,
    )


    qd_deg = qvel_deg(
        data,
        joint_info,
    )


    baseline_passive = vector_at_dofs(
        data.qfrc_passive,
        joint_info,
    )


    baseline_constraint = vector_at_dofs(
        data.qfrc_constraint,
        joint_info,
    )


    baseline_actuator = vector_at_dofs(
        data.qfrc_actuator,
        joint_info,
    )


    baseline_bias = vector_at_dofs(
        data.qfrc_bias,
        joint_info,
    )


    baseline_applied = vector_at_dofs(
        data.qfrc_applied,
        joint_info,
    )


    tendon_length = float(
        data.ten_length[
            tendon_tid
        ]
    )


    tendon_velocity = float(
        data.ten_velocity[
            tendon_tid
        ]
    )


    # =========================================================
    # TENDON CONTRIBUTION
    #
    # At exactly the same q/qdot:
    #
    #   tendon contribution =
    #       qfrc_passive(normal)
    #       -
    #       qfrc_passive(tendon stiffness/damping = 0)
    #
    # This does NOT alter the replay. It is a post-replay
    # counterfactual force decomposition at a frozen state.
    # =========================================================

    old_tendon_k = float(
        model.tendon_stiffness[
            tendon_tid
        ]
    )


    old_tendon_d = float(
        model.tendon_damping[
            tendon_tid
        ]
    )


    model.tendon_stiffness[
        tendon_tid
    ] = 0.0


    model.tendon_damping[
        tendon_tid
    ] = 0.0


    restore_state(
        model,
        data,
        snapshot,
    )


    passive_no_tendon = vector_at_dofs(
        data.qfrc_passive,
        joint_info,
    )


    tendon_tau = (
        baseline_passive
        -
        passive_no_tendon
    )


    # What remains in qfrc_passive is principally
    # joint spring/damping at these three joints.
    joint_passive_tau = (
        passive_no_tendon.copy()
    )


    model.tendon_stiffness[
        tendon_tid
    ] = old_tendon_k


    model.tendon_damping[
        tendon_tid
    ] = old_tendon_d


    # =========================================================
    # FIXTURE CONTACT CONTRIBUTION
    #
    # Compare qfrc_constraint at the identical state with the
    # steel cylinder collision disabled.
    #
    # Difference = generalized contribution associated with
    # fixture contact constraints.
    #
    # Joint dry friction / limits remain in residual_constraint.
    # =========================================================

    restore_state(
        model,
        data,
        snapshot,
    )


    old_fixture_contype = int(
        model.geom_contype[
            fixture_gid
        ]
    )


    old_fixture_conaff = int(
        model.geom_conaffinity[
            fixture_gid
        ]
    )


    model.geom_contype[
        fixture_gid
    ] = 0


    model.geom_conaffinity[
        fixture_gid
    ] = 0


    restore_state(
        model,
        data,
        snapshot,
    )


    constraint_no_fixture = vector_at_dofs(
        data.qfrc_constraint,
        joint_info,
    )


    bottle_contact_tau = (
        baseline_constraint
        -
        constraint_no_fixture
    )


    residual_constraint_tau = (
        constraint_no_fixture.copy()
    )


    model.geom_contype[
        fixture_gid
    ] = (
        old_fixture_contype
    )


    model.geom_conaffinity[
        fixture_gid
    ] = (
        old_fixture_conaff
    )


    # Restore the true baseline one final time.
    restore_state(
        model,
        data,
        snapshot,
    )


    active = contact_names(
        model,
        data,
        fixture_gid,
    )


    mid_gap = physical_gap(
        model,
        data,
        "collision_idx_mid_mesh",
        fixture_gid,
    )


    tip_gap = physical_gap(
        model,
        data,
        "collision_idx_tip",
        fixture_gid,
    )


    # =========================================================
    # EQUATION CHECK
    #
    # M qdd + bias =
    #     actuator
    #   + passive
    #   + constraint
    #   + applied
    #
    # So this quantity is the instantaneous generalized
    # acceleration-driving force.
    # =========================================================

    net_tau = (
        baseline_actuator
        +
        baseline_passive
        +
        baseline_constraint
        +
        baseline_applied
        -
        baseline_bias
    )


    print()
    print("=" * 132)
    print(label)
    print("=" * 132)


    print()
    print(
        f"bag time = "
        f"{snapshot['time']:.6f}"
    )


    print()
    print(
        "Dynamic joint state:"
    )


    print(
        "  q [deg]     = "
        f"{q_deg[0]:+9.4f}  "
        f"{q_deg[1]:+9.4f}  "
        f"{q_deg[2]:+9.4f}"
    )


    print(
        "  qdot [deg/s]= "
        f"{qd_deg[0]:+9.4f}  "
        f"{qd_deg[1]:+9.4f}  "
        f"{qd_deg[2]:+9.4f}"
    )


    print()
    print(
        f"Index tendon = {tendon_name}"
    )


    print(
        f"  length   = "
        f"{1000.0*tendon_length:.4f} mm"
    )


    print(
        f"  velocity = "
        f"{1000.0*tendon_velocity:+.4f} mm/s"
    )


    print(
        f"  stiffness= "
        f"{old_tendon_k:.3f} N/m"
    )


    print(
        f"  damping  = "
        f"{old_tendon_d:.3f} N s/m"
    )


    print()
    print(
        "Cylinder contacts:",
        active
        if active
        else "NONE",
    )


    print(
        "  middle mesh physical distance = "
        f"{1000.0*mid_gap:+.4f} mm"
    )


    print(
        "  tip physical distance         = "
        f"{1000.0*tip_gap:+.4f} mm"
    )


    print()
    print(
        "GENERALIZED TORQUE BUDGET [N m]"
    )


    print(
        "Positive sign = positive MuJoCo joint "
        "coordinate (flexion in this model)."
    )


    print()
    print(
        f"{'TERM':<30s}"
        f"{'MCP':>14s}"
        f"{'PIP':>14s}"
        f"{'DIP':>14s}"
    )


    print(
        "-" * 72
    )


    def row(
        name,
        values,
    ):

        print(
            f"{name:<30s}"
            f"{values[0]:>+14.7f}"
            f"{values[1]:>+14.7f}"
            f"{values[2]:>+14.7f}"
        )


    row(
        "TENDON",
        tendon_tau,
    )


    row(
        "JOINT PASSIVE",
        joint_passive_tau,
    )


    row(
        "BOTTLE CONTACT",
        bottle_contact_tau,
    )


    row(
        "CONSTRAINT RESIDUAL",
        residual_constraint_tau,
    )


    row(
        "ACTUATOR",
        baseline_actuator,
    )


    row(
        "APPLIED",
        baseline_applied,
    )


    row(
        "-BIAS",
        -baseline_bias,
    )


    print(
        "-" * 72
    )


    row(
        "NET DRIVE",
        net_tau,
    )


    print()
    print(
        "Raw MuJoCo terms:"
    )


    row(
        "qfrc_passive",
        baseline_passive,
    )


    row(
        "qfrc_constraint",
        baseline_constraint,
    )


    print()


    return {
        "q":
            q_deg,

        "qdot":
            qd_deg,

        "tendon":
            tendon_tau,

        "joint_passive":
            joint_passive_tau,

        "contact":
            bottle_contact_tau,

        "constraint_residual":
            residual_constraint_tau,

        "actuator":
            baseline_actuator,

        "bias":
            baseline_bias,

        "net":
            net_tau,

        "mid_gap":
            mid_gap,

        "tip_gap":
            tip_gap,

        "active_contacts":
            active,
    }


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
            f"Missing contact bag:\n"
            f"{contact_bag}"
        )


    if not b2.SELECTED_MODEL.exists():

        raise RuntimeError(
            "Missing V35B2 selected model:\n"
            f"{b2.SELECTED_MODEL}"
        )


    # Same replay patching used by V35 experiments.
    b2.v30.v24.ab.patch_replay_modules()


    contact_replay = (
        b2.v30.v24.base.contact_replay
    )


    if not hasattr(
        contact_replay,
        "TARGET_BAG_TIME",
    ):

        raise RuntimeError(
            "contact_replay has no TARGET_BAG_TIME."
        )


    target_time = float(
        contact_replay.TARGET_BAG_TIME
    )


    # =========================================================
    # Freeze exact V35C2 selected contact law.
    # =========================================================

    old_damping = float(
        contact_replay.CONTACT_DAMPING
    )


    old_mu = (
        b2.v35.CONTACT_MU
    )


    old_k = (
        b2.v35.CONTACT_K
    )


    contact_replay.CONTACT_DAMPING = (
        CONTACT_DAMPING
    )


    b2.v35.CONTACT_MU = (
        CONTACT_MU
    )


    b2.v35.CONTACT_K = (
        CONTACT_K
    )


    # =========================================================
    # Capture exact states carrying hardware bag time `t`.
    # =========================================================

    captured = []


    model_ref = {
        "model": None,
        "data": None,
    }


    original_mj_step = (
        mujoco.mj_step
    )


    def capture_step(
        model,
        data,
        *step_args,
        **step_kwargs,
    ):

        result = original_mj_step(
            model,
            data,
            *step_args,
            **step_kwargs,
        )


        t = find_replay_frame_time()


        if t is not None:

            captured.append(
                copy_snapshot(
                    t,
                    model,
                    data,
                )
            )


            model_ref[
                "model"
            ] = model


            model_ref[
                "data"
            ] = data


        return result


    mujoco.mj_step = (
        capture_step
    )


    try:

        print()
        print("=" * 132)
        print(
            "V35D2 — EXACT DYNAMIC TORQUE-BUDGET AUDIT"
        )
        print("=" * 132)


        print()
        print(
            f"TARGET_BAG_TIME = "
            f"{target_time:.6f}"
        )


        print(
            f"Contact law = "
            f"mu {CONTACT_MU:.3f}, "
            f"k {CONTACT_K:.1f}, "
            f"damping {CONTACT_DAMPING:.1f}"
        )


        result = b2.run_candidate(
            b2.SELECTED_MODEL,
            contact_bag,
        )


    finally:

        mujoco.mj_step = (
            original_mj_step
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


    if not captured:

        raise RuntimeError(
            "No time-labelled replay states were captured."
        )


    model = model_ref[
        "model"
    ]


    data = model_ref[
        "data"
    ]


    if (
        model is None
        or
        data is None
    ):

        raise RuntimeError(
            "Replay model/data capture failed."
        )


    times = np.asarray(
        [
            row[
                "time"
            ]
            for row in captured
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


    end_index = int(
        np.argmax(
            times
        )
    )


    target_snapshot = (
        captured[
            target_index
        ]
    )


    end_snapshot = (
        captured[
            end_index
        ]
    )


    print()
    print(
        f"Captured dynamic states = "
        f"{len(captured)}"
    )


    print(
        f"Captured time interval  = "
        f"{times.min():.6f} -> "
        f"{times.max():.6f}"
    )


    print(
        f"Target snapshot         = "
        f"{target_snapshot['time']:.6f} "
        f"(error "
        f"{1000.0*abs(target_snapshot['time']-target_time):.3f} ms)"
    )


    print(
        f"Hold-end snapshot       = "
        f"{end_snapshot['time']:.6f}"
    )


    print()
    print(
        "Evaluator-reported video-aligned pose:"
    )


    print(
        f"  {result['pose'][0]:.3f} / "
        f"{result['pose'][1]:.3f} / "
        f"{result['pose'][2]:.3f} deg"
    )


    print(
        f"  RMSE = "
        f"{result['rmse']:.3f} deg"
    )


    joint_info = joint_addresses(
        model
    )


    tendon_tid, tendon_name = (
        find_index_tendon(
            model
        )
    )


    fixture_gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FIXTURE_NAME,
    )


    if fixture_gid is None:

        raise RuntimeError(
            f"Missing fixture geom: "
            f"{FIXTURE_NAME}"
        )


    target_audit = audit_state(
        "V35D2-A — VIDEO-ALIGNED TARGET STATE",
        model,
        data,
        target_snapshot,
        joint_info,
        tendon_tid,
        tendon_name,
        fixture_gid,
    )


    end_audit = audit_state(
        "V35D2-B — HOLD-END / DETACHED STATE",
        model,
        data,
        end_snapshot,
        joint_info,
        tendon_tid,
        tendon_name,
        fixture_gid,
    )


    # =========================================================
    # DELTA
    # =========================================================

    print()
    print("=" * 132)
    print(
        "V35D2-C — CHANGE FROM TARGET -> HOLD END"
    )
    print("=" * 132)


    print()
    print(
        f"{'TERM':<30s}"
        f"{'MCP':>14s}"
        f"{'PIP':>14s}"
        f"{'DIP':>14s}"
    )


    print(
        "-" * 72
    )


    def delta_row(
        name,
        key,
    ):

        values = (
            end_audit[
                key
            ]
            -
            target_audit[
                key
            ]
        )


        print(
            f"{name:<30s}"
            f"{values[0]:>+14.7f}"
            f"{values[1]:>+14.7f}"
            f"{values[2]:>+14.7f}"
        )


    delta_row(
        "TENDON",
        "tendon",
    )


    delta_row(
        "JOINT PASSIVE",
        "joint_passive",
    )


    delta_row(
        "BOTTLE CONTACT",
        "contact",
    )


    delta_row(
        "CONSTRAINT RESIDUAL",
        "constraint_residual",
    )


    delta_row(
        "NET DRIVE",
        "net",
    )


    print()
    print(
        "Gap change:"
    )


    print(
        f"  middle: "
        f"{1000.0*target_audit['mid_gap']:+.4f} "
        f"-> "
        f"{1000.0*end_audit['mid_gap']:+.4f} mm"
    )


    print(
        f"  tip:    "
        f"{1000.0*target_audit['tip_gap']:+.4f} "
        f"-> "
        f"{1000.0*end_audit['tip_gap']:+.4f} mm"
    )


    print()
    print("=" * 132)
    print(
        "V35D2 INTERPRETATION GUIDE"
    )
    print("=" * 132)


    print()
    print(
        "1. If bottle-contact torque exists at TARGET but "
        "vanishes at HOLD-END, while passive torque drives "
        "the joints away:"
    )

    print(
        "   -> investigate tendon preload/rest length or "
        "nonlinear passive/hysteretic return mechanics."
    )


    print()
    print(
        "2. If tendon torque itself falls strongly between "
        "TARGET and HOLD-END:"
    )

    print(
        "   -> the tendon rest length / preload model is "
        "a primary suspect."
    )


    print()
    print(
        "3. If tendon torque remains strong but produces the "
        "wrong MCP/PIP/DIP balance:"
    )

    print(
        "   -> investigate configuration-dependent passive "
        "mechanics / tendon transmission, not contact."
    )


    print()
    print(
        "4. If substantial positive NET DRIVE remains at "
        "hold-end despite nearly zero velocity:"
    )

    print(
        "   -> inspect solver/constraint or state-capture "
        "assumptions before changing physics."
    )


    print()
    print("=" * 132)
    print(
        "V35D2 COMPLETE — NO MODEL PARAMETERS WERE FITTED"
    )
    print("=" * 132)


if __name__ == "__main__":
    main()
