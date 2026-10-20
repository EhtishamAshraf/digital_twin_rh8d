#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import fit_index_contact_v1 as bagbase
import scan_index_contact_friction_v35g7 as g7


# =====================================================================
# G20A — UNDERACTUATED REDISTRIBUTION BENCH TEST
#
# PURPOSE:
#   Determine whether the CURRENT continuous spatial tendon is
#   mechanically capable of redistributing motor displacement when
#   individual finger DOFs are externally prevented from moving.
#
# THIS IS A DIAGNOSTIC ONLY.
#
# Temporary equality constraints represent a bench-test situation in
# which a joint is physically held at its open/reference position.
#
# These constraints are NOT proposed for the final digital twin.
#
# Bottle collision is disabled so contact cannot affect the result.
#
# Motor/tendon topology, tendon stiffness, damping, routing and passive
# joint properties are otherwise unchanged.
# =====================================================================


SCALE = 5.70e-6
CONTACT_MU = 0.05

FIXTURE = "index_contact_fixture"

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

TENDON = "idx_flex_spatial"

# Encoder locations at which we inspect the mechanism.
ENCODER_CHECKPOINTS = [
    300.0,
    500.0,
    700.0,
    850.0,
    1000.0,
    1027.0,
]


# ---------------------------------------------------------------------
# Diagnostic configurations
#
# "lock" means an artificial bench constraint ONLY for this diagnostic.
# ---------------------------------------------------------------------

CASES = [
    (
        "all_free",
        [],
    ),
    (
        "lock_mcp",
        [MCP],
    ),
    (
        "lock_pip",
        [PIP],
    ),
    (
        "lock_dip",
        [DIP],
    ),
    (
        "lock_mcp_pip",
        [MCP, PIP],
    ),
    (
        "lock_mcp_dip",
        [MCP, DIP],
    ),
    (
        "lock_pip_dip",
        [PIP, DIP],
    ),
]


def find_geom(root, name):

    for geom in root.iter("geom"):

        if geom.get("name") == name:
            return geom

    raise RuntimeError(
        f"Could not find geom: {name}"
    )


def find_joint(root, name):

    for joint in root.iter("joint"):

        if joint.get("name") == name:
            return joint

    raise RuntimeError(
        f"Could not find joint: {name}"
    )


def create_case(
    source_xml: Path,
    destination: Path,
    locked_joints,
):

    tree = ET.parse(
        source_xml
    )

    root = tree.getroot()

    # -------------------------------------------------------------
    # Remove object interaction, but KEEP the object and its name.
    # Existing replay/diagnostic software can still find it.
    # -------------------------------------------------------------

    fixture = find_geom(
        root,
        FIXTURE,
    )

    fixture.set(
        "contype",
        "0",
    )

    fixture.set(
        "conaffinity",
        "0",
    )

    # -------------------------------------------------------------
    # Verify all requested joint names exist.
    # -------------------------------------------------------------

    for name in locked_joints:
        find_joint(
            root,
            name,
        )

    # -------------------------------------------------------------
    # Add equality locks.
    #
    # No joint2 + polycoef all zero:
    #
    #   q_joint = q_reference
    #
    # These are test-fixture constraints, NOT model mechanics.
    # -------------------------------------------------------------

    equality = root.find(
        "equality"
    )

    if equality is None:

        equality = ET.SubElement(
            root,
            "equality",
        )

    for i, name in enumerate(
        locked_joints
    ):

        short = (
            name
            .replace(":", "_")
            .replace("--", "_")
            .replace("-", "_")
        )

        ET.SubElement(
            equality,
            "joint",
            {
                "name":
                    f"g20_lock_{i}_{short}",

                "joint1":
                    name,

                "polycoef":
                    "0 0 0 0 0",

                "active":
                    "true",

                # Firm diagnostic fixture.
                "solref":
                    "0.002 1",

                "solimp":
                    "0.99 0.999 0.0001",
            },
        )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )


def joint_qadr(
    model,
    name,
):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:

        raise RuntimeError(
            f"MuJoCo joint not found: {name}"
        )

    return int(
        model.jnt_qposadr[
            jid
        ]
    )


def tendon_id(
    model,
    name,
):

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        name,
    )

    if tid < 0:

        raise RuntimeError(
            f"MuJoCo tendon not found: {name}"
        )

    return int(tid)


def first_encoder_crossing(
    encoder_data,
    target_tick,
    start_t,
    end_t,
):

    enc = np.asarray(
        encoder_data,
        dtype=float,
    )

    mask = (
        (enc[:, 0] >= start_t)
        &
        (enc[:, 0] <= end_t)
        &
        (enc[:, 1] >= target_tick)
    )

    ids = np.flatnonzero(
        mask
    )

    if not len(ids):
        return np.nan

    i = int(
        ids[0]
    )

    # Linear interpolation between adjacent encoder samples when
    # possible, rather than quantizing to a ROS sample.
    if i > 0:

        t0 = float(
            enc[i - 1, 0]
        )

        t1 = float(
            enc[i, 0]
        )

        e0 = float(
            enc[i - 1, 1]
        )

        e1 = float(
            enc[i, 1]
        )

        if (
            t0 >= start_t
            and
            e1 != e0
            and
            e0 <= target_tick <= e1
        ):

            a = (
                target_tick
                -
                e0
            ) / (
                e1
                -
                e0
            )

            return (
                t0
                +
                a
                *
                (
                    t1
                    -
                    t0
                )
            )

    return float(
        enc[i, 0]
    )


def interpolate_qpos(
    times,
    qpos,
    t,
):

    times = np.asarray(
        times,
        dtype=float,
    )

    qpos = np.asarray(
        qpos,
        dtype=float,
    )

    if (
        t < times[0]
        or
        t > times[-1]
    ):

        raise RuntimeError(
            f"Requested t={t:.6f} outside simulation "
            f"[{times[0]:.6f}, {times[-1]:.6f}]"
        )

    out = np.empty(
        qpos.shape[1],
        dtype=float,
    )

    for j in range(
        qpos.shape[1]
    ):

        out[j] = np.interp(
            t,
            times,
            qpos[:, j],
        )

    return out


def static_tendon_state(
    model,
    q,
    tid,
):

    data = mujoco.MjData(
        model
    )

    data.qpos[:] = q
    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    length = float(
        data.ten_length[
            tid
        ]
    )

    springlength = float(
        model.tendon_lengthspring[
            tid,
            0,
        ]
    )

    extension = (
        length
        -
        springlength
    )

    stiffness = float(
        model.tendon_stiffness[
            tid
        ]
    )

    # Static elastic contribution only.
    # Dynamic damping is deliberately excluded here.
    elastic_force = (
        stiffness
        *
        extension
    )

    return (
        length,
        extension,
        elastic_force,
    )


def angle_deg(
    q,
    qadr,
):

    return float(
        np.rad2deg(
            q[qadr]
        )
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
            /
            "rh8d_results"
            /
            "v35g20a_underactuation_audit"
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

    shutil.copy2(
        Path(__file__).resolve(),
        out_dir
        /
        Path(__file__).name,
    )

    # -------------------------------------------------------------
    # Existing validated H1 data loader.
    # -------------------------------------------------------------

    (
        states,
        force_data,
        encoder_data,
    ) = bagbase.read_bag(
        bag
    )

    encoder_data = np.asarray(
        encoder_data,
        dtype=float,
    )

    encoder_data = (
        encoder_data.copy()
    )

    encoder_data[:, 1] = (
        bagbase.median_filter_5(
            encoder_data[:, 1]
        )
    )

    transitions = (
        bagbase.compress_states(
            states
        )
    )

    (
        approach_start,
        regulation_start,
        hold_start,
        hold_end,
    ) = bagbase.find_intervals(
        transitions
    )

    # -------------------------------------------------------------
    # Encoder -> time checkpoints.
    # Use only first H1 closing phase.
    # -------------------------------------------------------------

    checkpoints = []

    for tick in ENCODER_CHECKPOINTS:

        t = first_encoder_crossing(
            encoder_data,
            tick,
            approach_start,
            hold_start + 0.1,
        )

        if np.isfinite(t):

            checkpoints.append(
                (
                    tick,
                    float(t),
                )
            )

    if not checkpoints:

        raise RuntimeError(
            "Could not find encoder checkpoints."
        )

    print()
    print("=" * 140)
    print(
        "G20A — UNDERACTUATED REDISTRIBUTION BENCH TEST"
    )
    print("=" * 140)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  Equality locks in this experiment are ONLY diagnostic "
        "bench constraints."
    )

    print(
        "  They are NOT part of the proposed RH8D digital twin."
    )

    print(
        "  Bottle collision is disabled."
    )

    print(
        "  The measured H1 motor encoder trajectory remains "
        "the input."
    )

    print()
    print(
        "H1 encoder checkpoints:"
    )

    for tick, t in checkpoints:

        print(
            f"  {tick:7.1f} ticks "
            f"-> {t:.6f} s"
        )

    # -------------------------------------------------------------
    # Obtain current zero-margin source using exactly the same
    # mechanism as our validated G7 experiments.
    # -------------------------------------------------------------

    g7.b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    baseline_xml = (
        g7.make_zero_margin_model(
            out_dir
        )
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    old_mu = float(
        g7.b2.v35.CONTACT_MU
    )

    rows = []
    trajectories = {}

    try:

        g7.b2.v35.METERS_PER_TICK = (
            SCALE
        )

        g7.b2.v35.CONTACT_MU = (
            CONTACT_MU
        )

        for (
            case_name,
            locked_joints,
        ) in CASES:

            print()
            print("=" * 140)

            print(
                f"CASE: {case_name}"
            )

            print(
                "locked joints:"
            )

            if locked_joints:

                for joint in locked_joints:
                    print(
                        f"  {joint}"
                    )

            else:

                print(
                    "  NONE"
                )

            print("=" * 140)

            xml = (
                out_dir
                /
                f"{case_name}.xml"
            )

            create_case(
                baseline_xml,
                xml,
                locked_joints,
            )

            # Compile before doing any expensive simulation.
            model_check = (
                mujoco.MjModel
                .from_xml_path(
                    str(xml)
                )
            )

            print(
                f"compiled OK: "
                f"nq={model_check.nq}, "
                f"neq={model_check.neq}"
            )

            (
                result,
                trace,
                qpos,
                times,
            ) = g7.run_scale(
                SCALE,
                CONTACT_MU,
                xml,
                bag,
                contact_replay,
            )

            model = (
                mujoco.MjModel
                .from_xml_path(
                    str(xml)
                )
            )

            mcp_adr = joint_qadr(
                model,
                MCP,
            )

            pip_adr = joint_qadr(
                model,
                PIP,
            )

            dip_adr = joint_qadr(
                model,
                DIP,
            )

            tid = tendon_id(
                model,
                TENDON,
            )

            q_ref = interpolate_qpos(
                times,
                qpos,
                approach_start,
            )

            ref_angles = np.asarray(
                [
                    angle_deg(
                        q_ref,
                        mcp_adr,
                    ),
                    angle_deg(
                        q_ref,
                        pip_adr,
                    ),
                    angle_deg(
                        q_ref,
                        dip_adr,
                    ),
                ],
                dtype=float,
            )

            print()
            print(
                "Reference at APPROACH start:"
            )

            print(
                "  MCP/PIP/DIP = "
                f"{ref_angles[0]:.4f} / "
                f"{ref_angles[1]:.4f} / "
                f"{ref_angles[2]:.4f} deg"
            )

            print()
            print(
                " tick     dMCP      dPIP      dDIP   "
                "tendon_ext   elastic_F"
            )

            print(
                "-" * 78
            )

            for tick, t in checkpoints:

                q = interpolate_qpos(
                    times,
                    qpos,
                    t,
                )

                angles = np.asarray(
                    [
                        angle_deg(
                            q,
                            mcp_adr,
                        ),
                        angle_deg(
                            q,
                            pip_adr,
                        ),
                        angle_deg(
                            q,
                            dip_adr,
                        ),
                    ],
                    dtype=float,
                )

                delta = (
                    angles
                    -
                    ref_angles
                )

                (
                    tendon_length,
                    extension,
                    elastic_force,
                ) = static_tendon_state(
                    model,
                    q,
                    tid,
                )

                rows.append(
                    {
                        "case":
                            case_name,

                        "encoder_tick":
                            tick,

                        "time_s":
                            t,

                        "mcp_deg":
                            angles[0],

                        "pip_deg":
                            angles[1],

                        "dip_deg":
                            angles[2],

                        "delta_mcp_deg":
                            delta[0],

                        "delta_pip_deg":
                            delta[1],

                        "delta_dip_deg":
                            delta[2],

                        "tendon_length_m":
                            tendon_length,

                        "tendon_extension_m":
                            extension,

                        "static_elastic_force_N":
                            elastic_force,

                        "locked_mcp":
                            MCP in locked_joints,

                        "locked_pip":
                            PIP in locked_joints,

                        "locked_dip":
                            DIP in locked_joints,
                    }
                )

                print(
                    f"{tick:6.0f} "
                    f"{delta[0]:+9.3f} "
                    f"{delta[1]:+9.3f} "
                    f"{delta[2]:+9.3f} "
                    f"{extension*1000:+10.3f} mm "
                    f"{elastic_force:+9.3f} N"
                )

            # -----------------------------------------------------
            # Verify diagnostic locks actually held.
            # -----------------------------------------------------

            final_tick, final_t = (
                checkpoints[-1]
            )

            q_final = interpolate_qpos(
                times,
                qpos,
                final_t,
            )

            final_angles = np.asarray(
                [
                    angle_deg(
                        q_final,
                        mcp_adr,
                    ),
                    angle_deg(
                        q_final,
                        pip_adr,
                    ),
                    angle_deg(
                        q_final,
                        dip_adr,
                    ),
                ],
                dtype=float,
            )

            final_delta = (
                final_angles
                -
                ref_angles
            )

            lock_map = {
                MCP:
                    final_delta[0],

                PIP:
                    final_delta[1],

                DIP:
                    final_delta[2],
            }

            for joint in locked_joints:

                err = abs(
                    lock_map[joint]
                )

                if err > 0.25:

                    print()
                    print(
                        "WARNING: diagnostic lock did "
                        "not hold tightly:"
                    )

                    print(
                        f"  {joint}: "
                        f"{lock_map[joint]:+.4f} deg"
                    )

            trajectories[
                case_name
            ] = (
                xml,
                qpos,
                times,
            )

            np.savez_compressed(
                out_dir
                /
                f"{case_name}_trajectory.npz",
                times=np.asarray(
                    times
                ),
                qpos=np.asarray(
                    qpos
                ),
            )

    finally:

        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

        g7.b2.v35.CONTACT_MU = (
            old_mu
        )

    # -------------------------------------------------------------
    # Save full quantitative table.
    # -------------------------------------------------------------

    summary_csv = (
        out_dir
        /
        "g20a_redistribution_summary.csv"
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

    # -------------------------------------------------------------
    # Compact final comparison at last common encoder checkpoint.
    # -------------------------------------------------------------

    last_tick = (
        checkpoints[-1][0]
    )

    final_rows = [
        r
        for r in rows
        if abs(
            r[
                "encoder_tick"
            ]
            -
            last_tick
        ) < 1e-9
    ]

    print()
    print("=" * 140)

    print(
        f"G20A REDISTRIBUTION SUMMARY "
        f"AT {last_tick:.0f} TICKS"
    )

    print("=" * 140)

    print(
        "case                 "
        "dMCP      dPIP      dDIP     "
        "ext_mm      Felastic"
    )

    print("-" * 140)

    for r in final_rows:

        print(
            f"{r['case']:<20s} "
            f"{r['delta_mcp_deg']:+9.3f} "
            f"{r['delta_pip_deg']:+9.3f} "
            f"{r['delta_dip_deg']:+9.3f} "
            f"{r['tendon_extension_m']*1000:+10.3f} "
            f"{r['static_elastic_force_N']:+10.3f}"
        )

    # -------------------------------------------------------------
    # Mechanism interpretation hints.
    # -------------------------------------------------------------

    lookup = {
        r["case"]:
            r
        for r in final_rows
    }

    print()
    print("=" * 140)

    print(
        "AUTOMATIC DIAGNOSTIC INTERPRETATION"
    )

    print("=" * 140)

    if (
        "lock_mcp"
        in lookup
    ):

        r = lookup[
            "lock_mcp"
        ]

        distal_motion = (
            abs(
                r[
                    "delta_pip_deg"
                ]
            )
            +
            abs(
                r[
                    "delta_dip_deg"
                ]
            )
        )

        print()

        if distal_motion > 2.0:

            print(
                "LOCK MCP: PASS — the current spatial tendon "
                "CAN redistribute motion into downstream joints."
            )

        else:

            print(
                "LOCK MCP: FAIL/WEAK — very little downstream "
                "motion occurs when MCP is unavailable."
            )

    if (
        "lock_mcp_pip"
        in lookup
    ):

        r = lookup[
            "lock_mcp_pip"
        ]

        print()

        if abs(
            r[
                "delta_dip_deg"
            ]
        ) > 1.0:

            print(
                "LOCK MCP+PIP: PASS — the existing tendon "
                "CAN mechanically drive DIP by itself."
            )

        else:

            print(
                "LOCK MCP+PIP: FAIL/WEAK — even when MCP/PIP "
                "are unavailable, DIP barely consumes tendon motion."
            )

    if (
        "lock_pip_dip"
        in lookup
    ):

        r = lookup[
            "lock_pip_dip"
        ]

        print()

        if abs(
            r[
                "delta_mcp_deg"
            ]
        ) > 1.0:

            print(
                "LOCK PIP+DIP: PASS — the existing tendon "
                "CAN mechanically drive MCP by itself."
            )

        else:

            print(
                "LOCK PIP+DIP: FAIL/WEAK — MCP itself is "
                "not being driven properly."
            )

    # -------------------------------------------------------------
    # Manifest.
    # -------------------------------------------------------------

    with (
        out_dir
        /
        "manifest.txt"
    ).open(
        "w"
    ) as f:

        f.write(
            "V35G20A — underactuated redistribution bench test\n\n"
        )

        f.write(
            "Diagnostic only; equality joint locks are NOT "
            "part of the proposed digital twin.\n"
        )

        f.write(
            "Bottle collision disabled.\n"
        )

        f.write(
            "Continuous idx_flex_spatial tendon unchanged.\n"
        )

        f.write(
            "H1 measured encoder trajectory retained.\n"
        )

        f.write(
            "Purpose: test whether existing spatial tendon "
            "can transfer displacement to remaining free DOFs "
            "when selected joints are externally constrained.\n"
        )

    # -------------------------------------------------------------
    # GUI — only the three most informative cases.
    # -------------------------------------------------------------

    if not args.no_gui:

        GUI_CASES = [
            "all_free",
            "lock_mcp",
            "lock_mcp_pip",
        ]

        print()
        print("=" * 140)

        print(
            "MANDATORY DIAGNOSTIC GUI"
        )

        print("=" * 140)

        print(
            "These are diagnostic constrained bench tests, "
            "not proposed model behavior."
        )

        print(
            "Close each window manually."
        )

        for i, case_name in enumerate(
            GUI_CASES,
            start=1,
        ):

            (
                xml,
                qpos,
                times,
            ) = trajectories[
                case_name
            ]

            print()
            print("#" * 120)

            print(
                f"GUI {i}/{len(GUI_CASES)} — "
                f"{case_name}"
            )

            if case_name == "all_free":

                print(
                    "No finger joint is externally held."
                )

            elif case_name == "lock_mcp":

                print(
                    "DIAGNOSTIC ONLY: MCP is externally held."
                )

                print(
                    "Watch whether PIP/DIP consume tendon motion."
                )

            elif case_name == "lock_mcp_pip":

                print(
                    "DIAGNOSTIC ONLY: MCP and PIP are externally held."
                )

                print(
                    "Watch whether DIP alone flexes under the same tendon."
                )

            print("#" * 120)

            g7.exact_gui_replay(
                xml,
                qpos,
                times,
                (
                    f"G20A DIAGNOSTIC — "
                    f"{case_name}"
                ),
            )

    print()
    print(
        f"Saved:\n  {summary_csv}"
    )

    print()
    print(
        f"All outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
