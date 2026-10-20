#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import fit_index_passive_contact_v35b2 as b2
import replay_index_zero_margin_v35e2a as e2a
import scan_index_zero_margin_damping_v35e2b as e2b


# ================================================================
# V35G7 — OBJECT-CONTACT FRICTION / WRAP-REDISTRIBUTION SCAN
#
# THIS IS NOT A PARAMETER OPTIMIZATION.
#
# We inspect three informative transmission scales:
#
# 5.60 um/tick : PIP nearly matches hardware
# 5.70 um/tick : balanced pose region
# 5.80 um/tick : minimum angle RMSE, but suspicious force/contact
#
# The real hardware mechanism to reproduce is:
#
# MCP motion
# -> middle phalanx contacts bottle
# -> PIP continues wrapping
# -> middle becomes geometrically constrained
# -> further tendon displacement redistributes
# -> DIP flexes / distal phalanx wraps
#
# ACCEPTANCE IS NOT BASED ON RMSE ALONE.
# ================================================================


SCALES = [
    5.70e-6,
]

CONTACT_MUS = [
    0.05,
    0.10,
    0.20,
    0.40,
    0.80,
]

TENDON_K = 2100.0

CONTACT_MU = 0.05  # baseline only; G7 scans this
CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

HW_POSE = np.asarray(
    e2a.HW_POSE,
    dtype=float,
)

HW_FORCE = float(
    e2a.HW_FORCE_MEDIAN
)

CONTACT_GEOMS = {
    "collision_idx_prox",
    "collision_idx_mid_mesh",
    "collision_idx_tip",
}

FIXTURE_NAME = "index_contact_fixture"


def tag(scale):
    return f"{scale * 1e6:.2f}".replace(".", "p")


def transitions(flags):
    flags = np.asarray(flags, dtype=bool)

    if len(flags) < 2:
        return 0

    return int(
        np.count_nonzero(
            flags[1:] != flags[:-1]
        )
    )


def longest_true(flags, dt):
    flags = np.asarray(flags, dtype=bool)

    best = 0
    current = 0

    for value in flags:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return float(best * dt)


def make_zero_margin_model(out_dir: Path) -> Path:

    source = (
        Path(b2.SELECTED_MODEL)
        .expanduser()
        .resolve()
    )

    target = (
        out_dir
        / "v35g7_zero_margin_source.xml"
    )

    tree = ET.parse(source)
    root = tree.getroot()

    # The copied XML lives outside mjcf/, therefore make meshdir
    # absolute so MuJoCo can still resolve all CAD assets.
    package_dir = source.parent.parent
    meshes_dir = (
        package_dir
        / "meshes"
    ).resolve()

    compiler = root.find("compiler")

    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)

    compiler.set(
        "meshdir",
        str(meshes_dir),
    )

    wanted = (
        CONTACT_GEOMS
        | {FIXTURE_NAME}
    )

    found = set()

    for geom in root.iter("geom"):

        name = geom.get("name")

        if name in wanted:

            geom.set(
                "margin",
                "0",
            )

            geom.set(
                "gap",
                "0",
            )

            found.add(name)

    missing = wanted - found

    if missing:
        raise RuntimeError(
            "Missing contact geoms:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    tree.write(
        target,
        encoding="utf-8",
        xml_declaration=True,
    )

    shutil.copy2(
        source,
        out_dir
        / "v35g7_original_selected_model.xml",
    )

    return target


def velocity_stats(
    times,
    values,
    mask,
):

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if np.count_nonzero(mask) < 3:
        return np.nan

    t = times[mask]
    q = values[mask]

    dq = np.gradient(
        q,
        t,
    )

    return float(
        np.median(
            np.abs(dq)
        )
    )


def continuous_true_from(
    flags,
    start_index,
    dt,
):

    if start_index is None:
        return 0.0

    count = 0

    for i in range(
        start_index,
        len(flags),
    ):
        if not flags[i]:
            break
        count += 1

    return float(
        count * dt
    )


def save_trace_csv(
    path,
    trace,
):

    fields = [
        "time_s",
        "encoder_tick",
        "mcp_deg",
        "pip_deg",
        "dip_deg",
        "mcp_vel_deg_s",
        "pip_vel_deg_s",
        "dip_vel_deg_s",
        "mid_contact",
        "tip_contact",
        "mid_force_N",
        "tip_force_N",
        "mid_gap_mm",
        "tip_gap_mm",
        "tendon_length_m",
        "spool_qpos_m",
        "spool_qvel_m_s",
        "spool_actuator_force_N",
    ]

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in trace:
            writer.writerow(
                {
                    k: row[k]
                    for k in fields
                }
            )


def run_scale(
    scale,
    mu,
    model_xml,
    bag,
    contact_replay,
):

    outer_trace = []

    refs = {
        "model": None,
        "joints": None,
        "fixture": None,
        "mid_gid": None,
        "tip_gid": None,
        "tendon_id": None,
        "spool_jid": None,
        "spool_qadr": None,
        "spool_dadr": None,
        "actuator_id": None,
    }

    true_mj_step = mujoco.mj_step


    def outer_capture_step(
        model,
        data,
        *args,
        **kwargs,
    ):

        out = true_mj_step(
            model,
            data,
            *args,
            **kwargs,
        )

        local = e2a.replay_locals()

        if local is None:
            return out

        if refs["model"] is None:

            refs["model"] = model

            refs["joints"] = (
                e2a.joint_info(
                    model
                )
            )

            refs["fixture"] = (
                e2a.gid(
                    model,
                    FIXTURE_NAME,
                )
            )

            refs["mid_gid"] = (
                e2a.gid(
                    model,
                    "collision_idx_mid_mesh",
                )
            )

            refs["tip_gid"] = (
                e2a.gid(
                    model,
                    "collision_idx_tip",
                )
            )

            refs["tendon_id"] = (
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_TENDON,
                    "idx_flex_spatial",
                )
            )

            refs["spool_jid"] = (
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    "index_motor_spool",
                )
            )

            refs["spool_qadr"] = int(
                model.jnt_qposadr[
                    refs["spool_jid"]
                ]
            )

            refs["spool_dadr"] = int(
                model.jnt_dofadr[
                    refs["spool_jid"]
                ]
            )

            refs["actuator_id"] = (
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    "pos_index_tendon",
                )
            )


        forces, active = (
            e2a.contact_forces(
                model,
                data,
                refs["fixture"],
            )
        )

        q = np.asarray(
            e2a.q_deg(
                data,
                refs["joints"],
            ),
            dtype=float,
        )

        qvel = []

        for name in (
            "Index_Proximal--palm_assembly:1",
            "Index_Middle--Index_Proximal",
            "Index_Distal--Index_Middle",
        ):

            jid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            dadr = int(
                model.jnt_dofadr[jid]
            )

            qvel.append(
                math.degrees(
                    float(
                        data.qvel[dadr]
                    )
                )
            )

        tendon_length = np.nan

        if refs["tendon_id"] >= 0:
            tendon_length = float(
                data.ten_length[
                    refs["tendon_id"]
                ]
            )

        actuator_force = np.nan

        if refs["actuator_id"] >= 0:
            actuator_force = float(
                data.actuator_force[
                    refs["actuator_id"]
                ]
            )

        outer_trace.append(
            {
                "time_s":
                    float(local["time"]),

                "encoder_tick":
                    float(local["encoder"]),

                "mcp_deg":
                    float(q[0]),

                "pip_deg":
                    float(q[1]),

                "dip_deg":
                    float(q[2]),

                "mcp_vel_deg_s":
                    float(qvel[0]),

                "pip_vel_deg_s":
                    float(qvel[1]),

                "dip_vel_deg_s":
                    float(qvel[2]),

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

                "mid_force_N":
                    float(
                        forces[
                            "collision_idx_mid_mesh"
                        ]
                    ),

                "tip_force_N":
                    float(
                        forces[
                            "collision_idx_tip"
                        ]
                    ),

                "mid_gap_mm":
                    1000.0
                    *
                    float(
                        e2a.physical_distance(
                            model,
                            data,
                            refs["mid_gid"],
                            refs["fixture"],
                        )
                    ),

                "tip_gap_mm":
                    1000.0
                    *
                    float(
                        e2a.physical_distance(
                            model,
                            data,
                            refs["tip_gid"],
                            refs["fixture"],
                        )
                    ),

                "tendon_length_m":
                    tendon_length,

                "spool_qpos_m":
                    float(
                        data.qpos[
                            refs["spool_qadr"]
                        ]
                    ),

                "spool_qvel_m_s":
                    float(
                        data.qvel[
                            refs["spool_dadr"]
                        ]
                    ),

                "spool_actuator_force_N":
                    actuator_force,

                # Save full dynamic position state for exact GUI replay.
                "_qpos":
                    np.asarray(
                        data.qpos,
                        dtype=float,
                    ).copy(),
            }
        )

        return out


    old_step = mujoco.mj_step

    mujoco.mj_step = (
        outer_capture_step
    )

    try:

        summary = e2b.run_one(
            CONTACT_DAMPING,
            model_xml,
            bag,
            contact_replay,
        )

    finally:

        mujoco.mj_step = old_step


    if not outer_trace:
        raise RuntimeError(
            f"No G6 trace captured for "
            f"{scale * 1e6:.2f} um/tick"
        )


    # ------------------------------------------------------------
    # Arrays.
    # ------------------------------------------------------------

    times = np.asarray(
        [
            r["time_s"]
            for r in outer_trace
        ],
        dtype=float,
    )

    enc = np.asarray(
        [
            r["encoder_tick"]
            for r in outer_trace
        ],
        dtype=float,
    )

    mcp = np.asarray(
        [
            r["mcp_deg"]
            for r in outer_trace
        ],
        dtype=float,
    )

    pip = np.asarray(
        [
            r["pip_deg"]
            for r in outer_trace
        ],
        dtype=float,
    )

    dip = np.asarray(
        [
            r["dip_deg"]
            for r in outer_trace
        ],
        dtype=float,
    )

    mid = np.asarray(
        [
            r["mid_contact"]
            for r in outer_trace
        ],
        dtype=bool,
    )

    tip = np.asarray(
        [
            r["tip_contact"]
            for r in outer_trace
        ],
        dtype=bool,
    )

    dt = float(
        np.median(
            np.diff(times)
        )
    )


    mid_indices = np.flatnonzero(mid)
    tip_indices = np.flatnonzero(tip)

    i_mid = (
        int(mid_indices[0])
        if mid_indices.size
        else None
    )

    i_tip = (
        int(tip_indices[0])
        if tip_indices.size
        else None
    )


    sequence_ok = (
        i_mid is not None
        and
        i_tip is not None
        and
        i_mid < i_tip
    )


    if sequence_ok:

        q_mid = np.array(
            [
                mcp[i_mid],
                pip[i_mid],
                dip[i_mid],
            ]
        )

        q_tip = np.array(
            [
                mcp[i_tip],
                pip[i_tip],
                dip[i_tip],
            ]
        )

        delta_mid_to_tip = (
            q_tip
            -
            q_mid
        )

        interval = np.arange(
            i_mid,
            i_tip + 1,
        )

        mid_duty_mid_to_tip = float(
            np.mean(
                mid[interval]
            )
        )

        mid_retained_to_tip = bool(
            np.all(
                mid[interval]
            )
        )

    else:

        q_mid = np.full(
            3,
            np.nan,
        )

        q_tip = np.full(
            3,
            np.nan,
        )

        delta_mid_to_tip = np.full(
            3,
            np.nan,
        )

        mid_duty_mid_to_tip = np.nan
        mid_retained_to_tip = False


    # ------------------------------------------------------------
    # Velocity windows around contact events.
    # ------------------------------------------------------------

    if i_mid is not None:

        t_mid = times[i_mid]

        pre_mid_mask = (
            (times >= t_mid - 0.10)
            &
            (times < t_mid)
        )

    else:

        pre_mid_mask = np.zeros(
            len(times),
            dtype=bool,
        )


    if sequence_ok:

        mid_to_tip_mask = (
            (times >= times[i_mid])
            &
            (times <= times[i_tip])
        )

    else:

        mid_to_tip_mask = np.zeros(
            len(times),
            dtype=bool,
        )


    if i_tip is not None:

        t_tip = times[i_tip]

        post_tip_mask = (
            (times >= t_tip)
            &
            (times <= t_tip + 0.10)
        )

    else:

        post_tip_mask = np.zeros(
            len(times),
            dtype=bool,
        )


    pip_pre = velocity_stats(
        times,
        pip,
        pre_mid_mask,
    )

    dip_pre = velocity_stats(
        times,
        dip,
        pre_mid_mask,
    )

    pip_mid_tip = velocity_stats(
        times,
        pip,
        mid_to_tip_mask,
    )

    dip_mid_tip = velocity_stats(
        times,
        dip,
        mid_to_tip_mask,
    )

    pip_post = velocity_stats(
        times,
        pip,
        post_tip_mask,
    )

    dip_post = velocity_stats(
        times,
        dip,
        post_tip_mask,
    )


    redistribution_mid_tip = (
        dip_mid_tip
        /
        max(
            pip_mid_tip,
            1e-9,
        )
        if np.isfinite(
            dip_mid_tip
        )
        and
        np.isfinite(
            pip_mid_tip
        )
        else np.nan
    )

    redistribution_post_tip = (
        dip_post
        /
        max(
            pip_post,
            1e-9,
        )
        if np.isfinite(
            dip_post
        )
        and
        np.isfinite(
            pip_post
        )
        else np.nan
    )


    continuous_mid_from_first = (
        continuous_true_from(
            mid,
            i_mid,
            dt,
        )
    )

    continuous_overlap_from_tip = 0.0

    if i_tip is not None:

        overlap = (
            mid
            &
            tip
        )

        continuous_overlap_from_tip = (
            continuous_true_from(
                overlap,
                i_tip,
                dt,
            )
        )


    target_time = float(
        contact_replay.TARGET_BAG_TIME
    )

    i_target = int(
        np.argmin(
            np.abs(
                times
                -
                target_time
            )
        )
    )

    target_pose = np.array(
        [
            mcp[i_target],
            pip[i_target],
            dip[i_target],
        ],
        dtype=float,
    )

    target_error = (
        target_pose
        -
        HW_POSE
    )

    rmse = float(
        np.sqrt(
            np.mean(
                target_error ** 2
            )
        )
    )


    # ------------------------------------------------------------
    # Mechanically meaningful audit summary.
    # ------------------------------------------------------------

    result = {
        "scale_m_per_tick":
            float(scale),

        "scale_um_per_tick":
            float(scale * 1e6),

        "contact_mu":
            float(mu),

        "rmse_deg":
            rmse,

        "target_mcp_deg":
            float(target_pose[0]),

        "target_pip_deg":
            float(target_pose[1]),

        "target_dip_deg":
            float(target_pose[2]),

        "first_mid_time_s":
            (
                float(times[i_mid])
                if i_mid is not None
                else np.nan
            ),

        "first_tip_time_s":
            (
                float(times[i_tip])
                if i_tip is not None
                else np.nan
            ),

        "first_mid_encoder":
            (
                float(enc[i_mid])
                if i_mid is not None
                else np.nan
            ),

        "first_tip_encoder":
            (
                float(enc[i_tip])
                if i_tip is not None
                else np.nan
            ),

        "sequence_ok":
            bool(sequence_ok),

        "mid_retained_to_tip":
            bool(mid_retained_to_tip),

        "mid_duty_mid_to_tip":
            float(mid_duty_mid_to_tip),

        "mcp_at_mid_deg":
            float(q_mid[0]),

        "pip_at_mid_deg":
            float(q_mid[1]),

        "dip_at_mid_deg":
            float(q_mid[2]),

        "mcp_at_tip_deg":
            float(q_tip[0]),

        "pip_at_tip_deg":
            float(q_tip[1]),

        "dip_at_tip_deg":
            float(q_tip[2]),

        "delta_mcp_mid_to_tip_deg":
            float(delta_mid_to_tip[0]),

        "delta_pip_mid_to_tip_deg":
            float(delta_mid_to_tip[1]),

        "delta_dip_mid_to_tip_deg":
            float(delta_mid_to_tip[2]),

        "pip_speed_pre_mid_deg_s":
            float(pip_pre),

        "dip_speed_pre_mid_deg_s":
            float(dip_pre),

        "pip_speed_mid_to_tip_deg_s":
            float(pip_mid_tip),

        "dip_speed_mid_to_tip_deg_s":
            float(dip_mid_tip),

        "pip_speed_post_tip_deg_s":
            float(pip_post),

        "dip_speed_post_tip_deg_s":
            float(dip_post),

        "dip_over_pip_mid_to_tip":
            float(
                redistribution_mid_tip
            ),

        "dip_over_pip_post_tip":
            float(
                redistribution_post_tip
            ),

        "continuous_mid_from_first_s":
            float(
                continuous_mid_from_first
            ),

        "continuous_overlap_from_tip_s":
            float(
                continuous_overlap_from_tip
            ),

        "mid_transitions_total":
            transitions(mid),

        "tip_transitions_total":
            transitions(tip),

        "target_mid_contact":
            bool(
                outer_trace[
                    i_target
                ]["mid_contact"]
            ),

        "target_tip_contact":
            bool(
                outer_trace[
                    i_target
                ]["tip_contact"]
            ),

        "target_mid_gap_mm":
            float(
                outer_trace[
                    i_target
                ]["mid_gap_mm"]
            ),

        "target_tip_gap_mm":
            float(
                outer_trace[
                    i_target
                ]["tip_gap_mm"]
            ),

        "max_mid_force_N":
            float(
                max(
                    r["mid_force_N"]
                    for r in outer_trace
                )
            ),

        "max_tip_force_N":
            float(
                max(
                    r["tip_force_N"]
                    for r in outer_trace
                )
            ),

        "e2b_hold_tip_force_median_N":
            float(
                summary[
                    "positive_tip_median"
                ]
            ),

        "e2b_hold_tip_force_max_N":
            float(
                summary[
                    "max_tip_force"
                ]
            ),

        "e2b_overlap_longest_s":
            float(
                summary[
                    "overlap_longest"
                ]
            ),

        "e2b_mid_transitions_hold":
            int(
                summary[
                    "mid_transitions"
                ]
            ),

        "e2b_tip_transitions_hold":
            int(
                summary[
                    "tip_transitions"
                ]
            ),
    }


    # Save full qpos separately for exact GUI playback.
    qpos_matrix = np.vstack(
        [
            r["_qpos"]
            for r in outer_trace
        ]
    )

    return (
        result,
        outer_trace,
        qpos_matrix,
        times,
    )


def mechanical_score(row):
    """
    Lower is better.

    RMSE is intentionally only a late/tie-breaking term.
    """

    score = 0.0

    if not row["sequence_ok"]:
        score += 1000.0

    if not row["mid_retained_to_tip"]:
        score += 300.0

    if not row["target_mid_contact"]:
        score += 150.0

    if not row["target_tip_contact"]:
        score += 150.0

    duty = row["mid_duty_mid_to_tip"]

    if np.isfinite(duty):
        score += (
            100.0
            *
            max(
                0.0,
                1.0 - duty,
            )
        )

    # Penalize visible interpenetration.
    penetration = (
        max(
            0.0,
            -row["target_mid_gap_mm"],
        )
        +
        max(
            0.0,
            -row["target_tip_gap_mm"],
        )
    )

    score += (
        80.0
        *
        penetration
    )

    # Contact losses / bouncing.
    score += (
        15.0
        *
        max(
            0,
            row[
                "e2b_mid_transitions_hold"
            ]
            - 1,
        )
    )

    score += (
        15.0
        *
        max(
            0,
            row[
                "e2b_tip_transitions_hold"
            ]
            - 1,
        )
    )

    # Very large force spikes are strongly undesirable.
    spike = row[
        "e2b_hold_tip_force_max_N"
    ]

    if np.isfinite(spike):
        score += (
            5.0
            *
            max(
                0.0,
                spike - 2.0,
            )
        )

    # RMSE only after mechanics.
    score += (
        2.0
        *
        row["rmse_deg"]
    )

    return float(score)


def exact_gui_replay(
    model_xml,
    qpos_matrix,
    times,
    label,
):

    print()
    print("=" * 130)
    print(
        "V35G7 — EXACT DYNAMIC TRAJECTORY GUI"
    )
    print("=" * 130)

    print()
    print(
        f"Opening mechanically selected candidate: {label}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The joints shown in this viewer are the states "
        "recorded from the actual dynamic MuJoCo run."
    )

    print(
        "The numerical simulation itself was NOT kinematically "
        "prescribed."
    )

    print()
    print(
        "VISUALLY CHECK:"
    )

    print(
        "  1. MCP begins closing normally."
    )

    print(
        "  2. Middle phalanx contacts bottle."
    )

    print(
        "  3. PIP continues wrapping after initial contact."
    )

    print(
        "  4. Middle phalanx becomes constrained rather than bouncing away."
    )

    print(
        "  5. Additional tendon motion produces distal/DIP wrapping."
    )

    print(
        "  6. There is no obvious unrealistic penetration or impact jump."
    )

    model = mujoco.MjModel.from_xml_path(
        str(model_xml)
    )

    data = mujoco.MjData(
        model
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        previous_t = float(
            times[0]
        )

        for qpos, current_t in zip(
            qpos_matrix,
            times,
        ):

            if not viewer.is_running():
                break

            data.qpos[:] = qpos
            data.qvel[:] = 0.0

            mujoco.mj_forward(
                model,
                data,
            )

            viewer.sync()

            delay = max(
                0.0,
                float(current_t)
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

            previous_t = float(
                current_t
            )

        print()
        print(
            "Replay finished. GUI will stay open until YOU close it."
        )

        while viewer.is_running():

            viewer.sync()

            time.sleep(
                0.02
            )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            / "rh8d_results"
            / "v35g7_contact_friction"
        ),
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

    out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Preserve the exact code used for this experiment.
    try:
        shutil.copy2(
            Path(__file__).resolve(),
            out_dir
            / Path(__file__).name,
        )
    except Exception:
        pass


    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
    )


    model_xml = make_zero_margin_model(
        out_dir
    )


    old_scale = float(
        b2.v35.METERS_PER_TICK
    )

    old_tendon_k = float(
        b2.v35.TENDON_K
    )

    old_mu = float(
        b2.v35.CONTACT_MU
    )

    old_contact_k = float(
        b2.v35.CONTACT_K
    )

    old_damping = float(
        contact_replay.CONTACT_DAMPING
    )


    rows = []

    replay_data = {}


    try:

        b2.v35.TENDON_K = (
            TENDON_K
        )

        b2.v35.CONTACT_MU = (
            CONTACT_MU
        )

        b2.v35.CONTACT_K = (
            CONTACT_K
        )

        contact_replay.CONTACT_DAMPING = (
            CONTACT_DAMPING
        )


        print()
        print("=" * 130)
        print(
            "V35G7 — OBJECT-CONTACT FRICTION / WRAP-REDISTRIBUTION SCAN"
        )
        print("=" * 130)

        print()
        print(
            f"H1 bag:\n  {bag}"
        )

        print(
            f"Output folder:\n  {out_dir}"
        )

        print(
            f"Model:\n  {model_xml}"
        )

        print()
        print(
            "NO PARAMETERS ARE FIT IN THIS EXPERIMENT."
        )

        print()
        print(
            "Scales audited = "
            + str(
                [
                    round(
                        s * 1e6,
                        2,
                    )
                    for s in SCALES
                ]
            )
            + " um/tick"
        )


        scale = 5.70e-6

        b2.v35.METERS_PER_TICK = (
            float(scale)
        )

        for mu in CONTACT_MUS:

            print()
            print("-" * 130)

            print(
                f"RUN: scale={scale * 1e6:.2f} um/tick, "
                f"contact mu={mu:.3f}"
            )

            print("-" * 130)

            b2.v35.CONTACT_MU = (
                float(mu)
            )

            (
                result,
                trace,
                qpos_matrix,
                times,
            ) = run_scale(
                scale,
                mu,
                model_xml,
                bag,
                contact_replay,
            )

            result[
                "mechanical_score"
            ] = mechanical_score(
                result
            )

            rows.append(
                result
            )

            name = (
                f"mu_{mu:.2f}"
                .replace(".", "p")
            )

            save_trace_csv(
                out_dir
                / f"{name}_timeseries.csv",
                trace,
            )

            np.savez_compressed(
                out_dir
                / f"{name}_trajectory.npz",
                times=times,
                qpos=qpos_matrix,
            )

            replay_data[
                float(mu)
            ] = (
                qpos_matrix,
                times,
            )

            print(
                "Target pose = "
                f"{result['target_mcp_deg']:.3f} / "
                f"{result['target_pip_deg']:.3f} / "
                f"{result['target_dip_deg']:.3f} deg"
            )

            print(
                f"RMSE = "
                f"{result['rmse_deg']:.3f} deg"
            )

            print()
            print(
                "FIRST MIDDLE CONTACT:"
            )

            print(
                "  encoder = "
                f"{result['first_mid_encoder']:.2f}"
            )

            print(
                "  pose    = "
                f"{result['mcp_at_mid_deg']:.3f} / "
                f"{result['pip_at_mid_deg']:.3f} / "
                f"{result['dip_at_mid_deg']:.3f}"
            )

            print()
            print(
                "FIRST TIP CONTACT:"
            )

            print(
                "  encoder = "
                f"{result['first_tip_encoder']:.2f}"
            )

            print(
                "  pose    = "
                f"{result['mcp_at_tip_deg']:.3f} / "
                f"{result['pip_at_tip_deg']:.3f} / "
                f"{result['dip_at_tip_deg']:.3f}"
            )

            print()
            print(
                "MIDDLE -> TIP JOINT MOTION:"
            )

            print(
                "  dPIP = "
                f"{result['delta_pip_mid_to_tip_deg']:+.3f} deg"
            )

            print(
                "  dDIP = "
                f"{result['delta_dip_mid_to_tip_deg']:+.3f} deg"
            )

            print(
                "  DIP/PIP = "
                f"{result['dip_over_pip_mid_to_tip']:.3f}"
            )

            print()
            print(
                "MIDDLE CONTACT RETENTION:"
            )

            print(
                "  retained to first tip = "
                f"{result['mid_retained_to_tip']}"
            )

            print(
                "  duty middle->tip = "
                f"{100*result['mid_duty_mid_to_tip']:.1f}%"
            )

            print()
            print(
                "POST-TIP REDISTRIBUTION:"
            )

            print(
                "  PIP speed = "
                f"{result['pip_speed_post_tip_deg_s']:.3f} deg/s"
            )

            print(
                "  DIP speed = "
                f"{result['dip_speed_post_tip_deg_s']:.3f} deg/s"
            )

            print(
                "  DIP/PIP = "
                f"{result['dip_over_pip_post_tip']:.3f}"
            )

            print()
            print(
                "CONTACT STABILITY:"
            )

            print(
                "  transitions mid/tip = "
                f"{result['e2b_mid_transitions_hold']} / "
                f"{result['e2b_tip_transitions_hold']}"
            )

            print(
                "  longest overlap = "
                f"{result['e2b_overlap_longest_s']:.3f} s"
            )

            print(
                "  target gaps = "
                f"{result['target_mid_gap_mm']:+.3f} / "
                f"{result['target_tip_gap_mm']:+.3f} mm"
            )

            print(
                "  tip force median/max = "
                f"{result['e2b_hold_tip_force_median_N']:.4f} / "
                f"{result['e2b_hold_tip_force_max_N']:.4f} N"
            )

            print(
                "  mechanics score = "
                f"{result['mechanical_score']:.3f}"
            )


    finally:

        b2.v35.METERS_PER_TICK = (
            old_scale
        )

        b2.v35.TENDON_K = (
            old_tendon_k
        )

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_contact_k
        )

        contact_replay.CONTACT_DAMPING = (
            old_damping
        )


    # ------------------------------------------------------------
    # Save summary.
    # ------------------------------------------------------------

    rows_sorted = sorted(
        rows,
        key=lambda r:
            r[
                "mechanical_score"
            ],
    )


    summary_csv = (
        out_dir
        / "event_summary.csv"
    )


    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    best = rows_sorted[0]

    best_scale = float(
        best[
            "scale_m_per_tick"
        ]
    )

    best_mu = float(
        best[
            "contact_mu"
        ]
    )


    with (
        out_dir
        / "selected_for_visual_check.txt"
    ).open(
        "w"
    ) as f:

        f.write(
            "V35G7 mechanically selected candidate\n"
        )

        f.write(
            "====================================\n\n"
        )

        f.write(
            f"scale = "
            f"{best_scale * 1e6:.3f} um/tick\n"
        )

        f.write(
            f"contact mu = "
            f"{best_mu:.6f}\n"
        )

        f.write(
            f"mechanical score = "
            f"{best['mechanical_score']:.6f}\n"
        )

        f.write(
            f"RMSE = "
            f"{best['rmse_deg']:.6f} deg\n\n"
        )

        f.write(
            "THIS IS NOT AUTOMATICALLY ACCEPTED.\n"
        )

        f.write(
            "Final acceptance requires visual MuJoCo comparison "
            "against the hardware video.\n"
        )


    with (
        out_dir
        / "manifest.txt"
    ).open(
        "w"
    ) as f:

        f.write(
            "V35G7 — object-contact friction / redistribution scan\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"H1 bag:\n{bag}\n\n"
        )

        f.write(
            f"model:\n{model_xml}\n\n"
        )

        f.write(
            "Fixed transmission scale:\n"
        )

        f.write(
            "  5.700 um/tick\n"
        )

        f.write(
            "\nScanned contact friction mu:\n"
        )

        for mu in CONTACT_MUS:
            f.write(
                f"  {mu:.6f}\n"
            )

        f.write(
            "\nFrozen:\n"
        )

        f.write(
            "  tendon stiffness = 2100 N/m\n"
        )

        f.write(
            "  contact mu = 0.05\n"
        )

        f.write(
            "  contact stiffness = 1000\n"
        )

        f.write(
            "  contact damping = 20\n"
        )

        f.write(
            "  zero contact margin/gap\n"
        )

        f.write(
            "  V35B2 passive mechanics\n"
        )

        f.write(
            "  CAD spatial tendon\n"
        )


    print()
    print("=" * 130)
    print(
        "V35G7 — MECHANICS-FIRST COMPARISON"
    )
    print("=" * 130)

    print()

    print(
        "mu     score   RMSE  retain  duty   "
        "dPIP    dDIP   DIP/PIP  postRatio  "
        "gapM   gapT   trM trT  Fmax"
    )

    print("-" * 130)

    for r in rows_sorted:

        print(
            f"{r['contact_mu']:5.2f} "
            f"{r['mechanical_score']:7.2f} "
            f"{r['rmse_deg']:6.3f} "
            f"{str(r['mid_retained_to_tip']):>6s} "
            f"{100*r['mid_duty_mid_to_tip']:5.1f}% "
            f"{r['delta_pip_mid_to_tip_deg']:+7.3f} "
            f"{r['delta_dip_mid_to_tip_deg']:+7.3f} "
            f"{r['dip_over_pip_mid_to_tip']:8.3f} "
            f"{r['dip_over_pip_post_tip']:9.3f} "
            f"{r['target_mid_gap_mm']:+6.3f} "
            f"{r['target_tip_gap_mm']:+6.3f} "
            f"{r['e2b_mid_transitions_hold']:3d} "
            f"{r['e2b_tip_transitions_hold']:3d} "
            f"{r['e2b_hold_tip_force_max_N']:6.2f}"
        )


    print()
    print(
        "MECHANICALLY SELECTED FOR VISUAL INSPECTION:"
    )

    print(
        f"  scale = "
        f"{best_scale * 1e6:.3f} um/tick"
    )

    print(
        f"  contact mu = "
        f"{best_mu:.3f}"
    )

    print(
        f"  RMSE  = "
        f"{best['rmse_deg']:.3f} deg"
    )

    print()
    print(
        "This candidate is NOT accepted until YOU verify "
        "the MuJoCo motion visually."
    )


    if not args.no_gui:

        qpos_matrix, times = (
            replay_data[
                best_mu
            ]
        )

        exact_gui_replay(
            model_xml,
            qpos_matrix,
            times,
            (
                f"{best_scale * 1e6:.3f} um/tick, "
                f"mu={best_mu:.3f}"
            ),
        )


    print()
    print("=" * 130)
    print(
        "V35G7 COMPLETE"
    )
    print("=" * 130)

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
