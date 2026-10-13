#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import fit_index_passive_contact_v35b2 as b2


# ================================================================
# V35E2A
#
# FULL DYNAMIC H1 REPLAY
#
# SINGLE CHANGE FROM V35C2:
#
#       contact margin: 1.0 mm -> 0.0 mm
#
# FROZEN:
#       CAD tendon
#       V35B2 passives
#       scale = existing identified value
#       tendon stiffness = existing value
#       mu = 0.05
#       direct contact stiffness = 1000
#       direct contact damping = 20
#       cylinder pose
#       measured hardware encoder replay
#
# NO prescribed MCP/PIP/DIP.
# NO contact-triggered logic.
# ================================================================


CONTACT_MU = 0.05
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

HW_POSE = np.array(
    [14.5355, 23.8110, 9.3270],
    dtype=float,
)

HW_FORCE_MEDIAN = 0.14563

HW_HOLD_START = 38.051338
HW_HOLD_END = 40.074176

CONTACT_GEOMS = {
    "collision_idx_prox",
    "collision_idx_mid_mesh",
    "collision_idx_tip",
}

FIXTURE_NAME = "index_contact_fixture"


def make_zero_margin_xml():

    source = Path(
        b2.SELECTED_MODEL
    ).resolve()

    target = source.with_name(
        source.stem
        + "_v35e2a_zero_margin.xml"
    )

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    wanted = (
        CONTACT_GEOMS
        |
        {FIXTURE_NAME}
    )

    found = set()

    for geom in root.iter(
        "geom"
    ):

        name = geom.get(
            "name"
        )

        if name in wanted:

            geom.set(
                "margin",
                "0"
            )

            geom.set(
                "gap",
                "0"
            )

            found.add(
                name
            )

    missing = (
        wanted
        -
        found
    )

    if missing:

        raise RuntimeError(
            "Could not find contact geoms "
            "inside selected XML:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    tree.write(
        target,
        encoding="unicode",
    )

    return target


def replay_locals():

    frame = inspect.currentframe()

    if frame is None:
        return None

    frame = frame.f_back

    while frame is not None:

        if (
            frame.f_globals.get(
                "__name__",
                "",
            )
            ==
            "replay_index_tabletop_hardware_contact_v1"
            and
            frame.f_code.co_name
            ==
            "main"
            and
            "t"
            in frame.f_locals
        ):

            def value(
                key,
            ):

                try:
                    return float(
                        frame.f_locals[
                            key
                        ]
                    )
                except Exception:
                    return math.nan

            return {
                "time":
                    value("t"),

                "encoder":
                    value("encoder"),

                "spool":
                    value("spool"),
            }

        frame = frame.f_back

    return None


def jid(
    model,
    name,
):

    out = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if out < 0:

        raise RuntimeError(
            f"Missing joint: {name}"
        )

    return int(out)


def gid(
    model,
    name,
):

    out = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        name,
    )

    if out < 0:

        raise RuntimeError(
            f"Missing geom: {name}"
        )

    return int(out)


def joint_info(
    model,
):

    names = [
        b2.v35.MCP,
        b2.v35.PIP,
        b2.v35.DIP,
    ]

    rows = []

    for name in names:

        joint_id = jid(
            model,
            name,
        )

        rows.append(
            (
                name,
                int(
                    model.jnt_qposadr[
                        joint_id
                    ]
                ),
                int(
                    model.jnt_dofadr[
                        joint_id
                    ]
                ),
            )
        )

    return rows


def q_deg(
    data,
    joints,
):

    return np.degrees(
        np.asarray(
            [
                data.qpos[
                    qadr
                ]
                for (
                    _name,
                    qadr,
                    _dadr,
                )
                in joints
            ],
            dtype=float,
        )
    )


def physical_distance(
    model,
    data,
    geom_id,
    fixture_id,
):

    fromto = np.zeros(
        6,
        dtype=float,
    )

    return float(
        mujoco.mj_geomDistance(
            model,
            data,
            geom_id,
            fixture_id,
            0.100,
            fromto,
        )
    )


def contact_forces(
    model,
    data,
    fixture_id,
):

    values = {
        "collision_idx_mid_mesh":
            0.0,

        "collision_idx_tip":
            0.0,
    }

    active = {
        "collision_idx_mid_mesh":
            False,

        "collision_idx_tip":
            False,
    }

    for ci in range(
        data.ncon
    ):

        contact = data.contact[
            ci
        ]

        # Ignore inactive contacts if any.
        if int(
            contact.efc_address
        ) < 0:
            continue

        g1 = int(
            contact.geom1
        )

        g2 = int(
            contact.geom2
        )

        if (
            g1 != fixture_id
            and
            g2 != fixture_id
        ):
            continue

        other = (
            g2
            if g1 == fixture_id
            else g1
        )

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            other,
        )

        if name not in values:
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

        values[name] += abs(
            float(
                wrench[0]
            )
        )

        active[name] = True

    return (
        values,
        active,
    )


def longest_true_duration(
    flags,
    dt,
):

    best = 0
    current = 0

    for value in flags:

        if value:

            current += 1
            best = max(
                best,
                current,
            )

        else:

            current = 0

    return (
        best
        *
        dt
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
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

    zero_margin_xml = (
        make_zero_margin_xml()
    )

    print()
    print("=" * 124)
    print(
        "V35E2A — FULL DYNAMIC H1 REPLAY, ZERO CONTACT MARGIN"
    )
    print("=" * 124)

    print()
    print(
        f"Model = "
        f"{zero_margin_xml}"
    )

    print()
    print(
        "ONLY physics change from V35C2:"
    )

    print(
        "  contact margin: 1.000 mm -> 0.000 mm"
    )

    print()
    print(
        f"mu              = "
        f"{CONTACT_MU}"
    )

    print(
        f"direct stiffness= "
        f"{CONTACT_K}"
    )

    print(
        f"direct damping  = "
        f"{CONTACT_DAMPING}"
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


    trace = []

    model_ref = {
        "model": None,
        "data": None,
        "joints": None,
        "fixture": None,
        "mid_gid": None,
        "tip_gid": None,
    }


    original_step = mujoco.mj_step


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

        local = replay_locals()

        if local is None:

            return out


        if model_ref["model"] is None:

            model_ref["model"] = model
            model_ref["data"] = data

            model_ref["joints"] = (
                joint_info(
                    model
                )
            )

            model_ref["fixture"] = (
                gid(
                    model,
                    FIXTURE_NAME,
                )
            )

            model_ref["mid_gid"] = (
                gid(
                    model,
                    "collision_idx_mid_mesh",
                )
            )

            model_ref["tip_gid"] = (
                gid(
                    model,
                    "collision_idx_tip",
                )
            )


        forces, active = contact_forces(
            model,
            data,
            model_ref[
                "fixture"
            ],
        )


        trace.append(
            {
                "time":
                    local["time"],

                "encoder":
                    local["encoder"],

                "spool":
                    local["spool"],

                "q":
                    q_deg(
                        data,
                        model_ref[
                            "joints"
                        ],
                    ),

                "mid_force":
                    forces[
                        "collision_idx_mid_mesh"
                    ],

                "tip_force":
                    forces[
                        "collision_idx_tip"
                    ],

                "mid_contact":
                    active[
                        "collision_idx_mid_mesh"
                    ],

                "tip_contact":
                    active[
                        "collision_idx_tip"
                    ],

                "mid_gap":
                    physical_distance(
                        model,
                        data,
                        model_ref[
                            "mid_gid"
                        ],
                        model_ref[
                            "fixture"
                        ],
                    ),

                "tip_gap":
                    physical_distance(
                        model,
                        data,
                        model_ref[
                            "tip_gid"
                        ],
                        model_ref[
                            "fixture"
                        ],
                    ),

                # Full dynamic state for exact visual replay.
                "qpos":
                    data.qpos.copy(),
            }
        )

        return out


    mujoco.mj_step = (
        capture_step
    )


    try:

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


    if not trace:

        raise RuntimeError(
            "No dynamic trace captured."
        )


    model = model_ref[
        "model"
    ]

    dt = float(
        model.opt.timestep
    )

    times = np.asarray(
        [
            r["time"]
            for r in trace
        ],
        dtype=float,
    )


    # ============================================================
    # VIDEO-ALIGNED RESULT
    # ============================================================

    target_time = float(
        contact_replay.TARGET_BAG_TIME
    )

    target_idx = int(
        np.argmin(
            np.abs(
                times
                -
                target_time
            )
        )
    )

    target = trace[
        target_idx
    ]


    print()
    print("=" * 124)
    print(
        "V35E2A — VIDEO-ALIGNED H1 RESULT"
    )
    print("=" * 124)

    print()
    print(
        f"target bag time = "
        f"{target_time:.6f}"
    )

    print(
        f"captured time   = "
        f"{target['time']:.6f}"
    )


    print()
    print(
        "Existing evaluator:"
    )

    print(
        f"  SIM pose = "
        f"{result['pose'][0]:.3f} / "
        f"{result['pose'][1]:.3f} / "
        f"{result['pose'][2]:.3f} deg"
    )

    print(
        f"  H1 pose  = "
        f"{HW_POSE[0]:.3f} / "
        f"{HW_POSE[1]:.3f} / "
        f"{HW_POSE[2]:.3f} deg"
    )

    print(
        f"  RMSE     = "
        f"{result['rmse']:.3f} deg"
    )


    print()
    print(
        "Physical contact state at video target:"
    )

    print(
        f"  middle contact = "
        f"{target['mid_contact']}"
    )

    print(
        f"  tip contact    = "
        f"{target['tip_contact']}"
    )

    print(
        f"  middle distance= "
        f"{1000*target['mid_gap']:+.4f} mm"
    )

    print(
        f"  tip distance   = "
        f"{1000*target['tip_gap']:+.4f} mm"
    )

    print(
        f"  middle Fn      = "
        f"{target['mid_force']:.5f} N"
    )

    print(
        f"  tip Fn         = "
        f"{target['tip_force']:.5f} N"
    )


    # ============================================================
    # CONTACT TOPOLOGY
    # ============================================================

    mid_flags = np.asarray(
        [
            r["mid_contact"]
            for r in trace
        ],
        dtype=bool,
    )

    tip_flags = np.asarray(
        [
            r["tip_contact"]
            for r in trace
        ],
        dtype=bool,
    )

    overlap_flags = (
        mid_flags
        &
        tip_flags
    )


    mid_indices = np.flatnonzero(
        mid_flags
    )

    tip_indices = np.flatnonzero(
        tip_flags
    )


    first_mid = (
        trace[
            int(
                mid_indices[0]
            )
        ]
        if mid_indices.size
        else None
    )

    first_tip = (
        trace[
            int(
                tip_indices[0]
            )
        ]
        if tip_indices.size
        else None
    )


    print()
    print("=" * 124)
    print(
        "V35E2A — CONTACT TOPOLOGY"
    )
    print("=" * 124)

    print()

    if first_mid is None:

        print(
            "First middle contact: NONE"
        )

    else:

        print(
            f"First middle contact: "
            f"t={first_mid['time']:.6f}, "
            f"encoder={first_mid['encoder']:.2f}"
        )


    if first_tip is None:

        print(
            "First tip contact: NONE"
        )

    else:

        print(
            f"First tip contact:    "
            f"t={first_tip['time']:.6f}, "
            f"encoder={first_tip['encoder']:.2f}"
        )


    print()
    print(
        f"middle contact total       = "
        f"{np.sum(mid_flags)*dt:.3f} s"
    )

    print(
        f"tip contact total          = "
        f"{np.sum(tip_flags)*dt:.3f} s"
    )

    print(
        f"simultaneous contact total = "
        f"{np.sum(overlap_flags)*dt:.3f} s"
    )

    print(
        f"longest continuous overlap = "
        f"{longest_true_duration(overlap_flags,dt):.3f} s"
    )


    # ============================================================
    # HARDWARE HOLD-WINDOW FORCE
    # ============================================================

    hold_rows = [
        r
        for r in trace
        if (
            HW_HOLD_START
            <=
            r["time"]
            <=
            HW_HOLD_END
        )
    ]


    print()
    print("=" * 124)
    print(
        "V35E2A — HARDWARE HOLD-WINDOW FORCE"
    )
    print("=" * 124)

    print()

    if not hold_rows:

        print(
            "No rows inside hardware hold window."
        )

    else:

        tip_force = np.asarray(
            [
                r["tip_force"]
                for r in hold_rows
            ],
            dtype=float,
        )

        mid_force = np.asarray(
            [
                r["mid_force"]
                for r in hold_rows
            ],
            dtype=float,
        )


        print(
            f"Hardware hold window = "
            f"{HW_HOLD_START:.6f} -> "
            f"{HW_HOLD_END:.6f}"
        )

        print()
        print(
            f"HW FTS median        = "
            f"{HW_FORCE_MEDIAN:.5f} N"
        )


        print()
        print(
            "SIM tip normal force "
            "(zeros included):"
        )

        print(
            f"  median = "
            f"{np.median(tip_force):.5f} N"
        )

        print(
            f"  mean   = "
            f"{np.mean(tip_force):.5f} N"
        )

        print(
            f"  max    = "
            f"{np.max(tip_force):.5f} N"
        )


        positive_tip = (
            tip_force[
                tip_force > 0
            ]
        )

        if positive_tip.size:

            print(
                "SIM tip force while actually "
                "in contact:"
            )

            print(
                f"  median = "
                f"{np.median(positive_tip):.5f} N"
            )

            print(
                f"  mean   = "
                f"{np.mean(positive_tip):.5f} N"
            )


        print()
        print(
            "SIM middle normal force "
            "(zeros included):"
        )

        print(
            f"  median = "
            f"{np.median(mid_force):.5f} N"
        )

        print(
            f"  max    = "
            f"{np.max(mid_force):.5f} N"
        )


    # ============================================================
    # END STATE
    # ============================================================

    final = trace[
        -1
    ]


    print()
    print("=" * 124)
    print(
        "V35E2A — REPLAY END STATE"
    )
    print("=" * 124)

    print()

    print(
        f"q = "
        f"{final['q'][0]:.3f} / "
        f"{final['q'][1]:.3f} / "
        f"{final['q'][2]:.3f} deg"
    )

    print(
        f"middle distance = "
        f"{1000*final['mid_gap']:+.4f} mm"
    )

    print(
        f"tip distance    = "
        f"{1000*final['tip_gap']:+.4f} mm"
    )

    print(
        f"middle contact  = "
        f"{final['mid_contact']}"
    )

    print(
        f"tip contact     = "
        f"{final['tip_contact']}"
    )


    print()
    print("=" * 124)
    print(
        "V35E2A INTERPRETATION"
    )
    print("=" * 124)

    print()
    print(
        "This run tests ONLY whether removing the old "
        "1 mm contact margin improves the actual dynamic "
        "H1 behavior."
    )

    print()
    print(
        "Do not accept or reject the model from pose RMSE alone."
    )

    print(
        "Use pose + contact topology + hold force + visual replay."
    )


    # ============================================================
    # EXACT VISUAL PLAYBACK OF THE RECORDED DYNAMIC TRAJECTORY
    # ============================================================

    if not args.no_gui:

        print()
        print("=" * 124)
        print(
            "V35E2A — EXACT RECORDED DYNAMIC TRAJECTORY GUI"
        )
        print("=" * 124)

        print()
        print(
            "This GUI replays the qpos states recorded "
            "from the actual dynamic simulation."
        )

        print(
            "It does NOT prescribe the joints during the "
            "numerical experiment above."
        )

        visual_model = (
            mujoco.MjModel.from_xml_path(
                str(
                    zero_margin_xml
                )
            )
        )

        visual_data = mujoco.MjData(
            visual_model
        )

        try:

            with mujoco.viewer.launch_passive(
                visual_model,
                visual_data,
            ) as viewer:

                previous_t = trace[
                    0
                ]["time"]

                for row in trace:

                    if not viewer.is_running():
                        break

                    visual_data.qpos[:] = (
                        row["qpos"]
                    )

                    visual_data.qvel[:] = 0.0

                    mujoco.mj_forward(
                        visual_model,
                        visual_data,
                    )

                    viewer.sync()

                    delay = max(
                        0.0,
                        row["time"]
                        -
                        previous_t,
                    )

                    if delay > 0:

                        time.sleep(
                            min(
                                delay,
                                0.01,
                            )
                        )

                    previous_t = row[
                        "time"
                    ]

                # Hold final image briefly.
                end_wait = time.time() + 2.0

                while (
                    viewer.is_running()
                    and
                    time.time()
                    <
                    end_wait
                ):

                    viewer.sync()
                    time.sleep(
                        0.02
                    )

        except Exception as exc:

            print()
            print(
                "Viewer ended with:"
            )

            print(
                repr(exc)
            )


    print()
    print("=" * 124)
    print(
        "V35E2A COMPLETE"
    )
    print("=" * 124)


if __name__ == "__main__":
    main()
