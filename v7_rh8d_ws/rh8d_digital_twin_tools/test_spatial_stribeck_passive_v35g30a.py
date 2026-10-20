#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import scan_index_contact_friction_v35g7 as g7
import audit_index_tendon_transmission_v35g12 as g12
import test_spatial_prescribed_spool_v35g29b as g29b


# ================================================================
# G30A
#
# FROZEN:
#   - continuous idx_flex_spatial tendon
#   - measured encoder -> prescribed index_motor_spool
#   - tendon K = XML value (2100 N/m)
#   - springs unchanged
#   - contact unchanged
#   - geometry unchanged
#
# ONLY NEW MECHANISM:
#
# velocity-dependent dry-friction capacity
#
#     F(v) = Fc + (Fs-Fc) exp(-(abs(v)/vs)^2)
#
# This approximates static -> kinetic/Stribeck resistance.
#
# It is:
#   - passive
#   - joint-velocity dependent
#   - independent of contact state
#
# NO contact-conditioned switching.
# NO qpos prescription for finger joints.
# NO fixed tendon.
# NO parameter scan.
# ================================================================


PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

# G20C mechanically useful near-static capacities.
PIP_FS = 0.00300
DIP_FS = 0.00600

# Existing model sliding resistance.
PIP_FC = 0.00125
DIP_FC = 0.00000

# Hardware marker motion:
# slow approach ~ sub-few-deg/s,
# faster wrapping ~ several deg/s.
VS_DEG_S = 2.0
VS_RAD_S = math.radians(VS_DEG_S)


H1_POINTS = [
    # encoder, MCP, PIP, DIP
    (416.1,   10.620, 3.994, 1.911),
    (611.7,   18.094, 4.526, 3.724),
    (673.573, 18.543, 6.161, 5.276),
]

HW_CONTACT_ENCODER = 673.573

HW_CONTACT_POSE = np.array(
    [18.543, 6.161, 5.276],
    dtype=float,
)

TRUSTED_FINAL = np.array(
    [14.5355, 23.8110, 9.3270],
    dtype=float,
)

TARGET_TIME = 38.970486


def nearest(values, target):
    values = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(values - target)))


def make_free_xml(source, destination):
    tree = ET.parse(source)
    root = tree.getroot()

    found = False

    for geom in root.iter("geom"):
        if geom.get("name") == g12.FIXTURE_NAME:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            found = True
            break

    if not found:
        raise RuntimeError(
            f"Fixture not found: {g12.FIXTURE_NAME}"
        )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )

    mujoco.MjModel.from_xml_path(
        str(destination)
    )


class StribeckPassive:
    """
    Modify only MuJoCo dry-friction capacity as a function
    of joint speed.

    MuJoCo's frictionloss constraint still handles the actual
    sticking/sliding generalized friction force.

    We change only the capacity:
        Fs near zero speed
        -> Fc during faster sliding.

    No contact information is used.
    """

    def __init__(self):
        # IMPORTANT:
        # Install this AFTER PrescribedSpool.
        # Then original_step == prescribed-spool step.
        self.original_step = mujoco.mj_step

        self.cache = {}

        self.history = []

    def setup(self, model):
        key = id(model)

        if key in self.cache:
            return self.cache[key]

        result = {}

        for label, joint_name, fs, fc in [
            ("PIP", PIP, PIP_FS, PIP_FC),
            ("DIP", DIP, DIP_FS, DIP_FC),
        ]:

            jid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if jid < 0:
                self.cache[key] = None
                return None

            dadr = int(
                model.jnt_dofadr[jid]
            )

            result[label] = {
                "jid": jid,
                "dadr": dadr,
                "fs": fs,
                "fc": fc,
                "original":
                    float(
                        model.dof_frictionloss[dadr]
                    ),
            }

        self.cache[key] = {
            "model": model,
            "joints": result,
        }

        return self.cache[key]

    @staticmethod
    def friction_capacity(v, fs, fc):
        return (
            fc
            +
            (fs - fc)
            *
            math.exp(
                -(
                    abs(v)
                    /
                    VS_RAD_S
                ) ** 2
            )
        )

    def step(self, model, data):
        state = self.setup(model)

        if state is None:
            return self.original_step(
                model,
                data,
            )

        record = {
            "time_s": float(data.time),
        }

        for label, j in state["joints"].items():
            dadr = j["dadr"]

            v = float(
                data.qvel[dadr]
            )

            capacity = (
                self.friction_capacity(
                    v,
                    j["fs"],
                    j["fc"],
                )
            )

            model.dof_frictionloss[
                dadr
            ] = capacity

            record[
                f"{label.lower()}_vel_deg_s"
            ] = math.degrees(v)

            record[
                f"{label.lower()}_friction_capacity_Nm"
            ] = capacity

        self.history.append(record)

        return self.original_step(
            model,
            data,
        )

    def install(self):
        mujoco.mj_step = self.step

    def restore(self):
        # Restore dynamic model friction values.
        for state in self.cache.values():
            if state is None:
                continue

            model = state["model"]

            for j in state["joints"].values():
                model.dof_frictionloss[
                    j["dadr"]
                ] = j["original"]

        mujoco.mj_step = self.original_step


def run_case(xml, bag, replay):
    # 1. measured encoder -> prescribed spool
    spool = g29b.PrescribedSpool()
    spool.install()

    # 2. speed-dependent passive friction
    # Captures spool.step as its underlying step.
    friction = StribeckPassive()
    friction.install()

    try:
        result = g12.run_audit(
            xml,
            bag,
            replay,
        )

        history = list(
            friction.history
        )

        drift = (
            1000.0
            *
            max(
                spool.post_step_drifts,
                default=0.0,
            )
        )

    finally:
        # Reverse installation order.
        friction.restore()
        spool.restore()

    return result, history, drift


def episodes(rows, key):
    flags = np.asarray(
        [bool(r[key]) for r in rows],
        dtype=bool,
    )

    return (
        g12.contact_episode_starts(flags),
        g12.contact_episode_ends(flags),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g30a_spatial_stribeck"
        ),
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
    )

    args = parser.parse_args()

    bag = args.contact_bag.expanduser().resolve()
    xml = args.xml.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        out / Path(__file__).name,
    )

    # ============================================================
    # HARD TOPOLOGY GATE
    # ============================================================

    print()
    print("=" * 150)
    print("G30A — SPATIAL TENDON + PASSIVE STRIBECK TEST")
    print("=" * 150)

    g29b.verify_spatial_tendon(
        xml
    )

    print()
    print("FROZEN:")
    print("  idx_flex_spatial        = unchanged")
    print("  tendon stiffness        = unchanged")
    print("  joint springs           = unchanged")
    print("  bottle/contact geometry = unchanged")
    print("  measured spool replay   = prescribed")
    print()

    print("ONLY NEW MECHANISM:")
    print(
        f"  PIP friction: "
        f"Fs={PIP_FS:.5f}, Fc={PIP_FC:.5f} Nm"
    )
    print(
        f"  DIP friction: "
        f"Fs={DIP_FS:.5f}, Fc={DIP_FC:.5f} Nm"
    )
    print(
        f"  Stribeck velocity scale = "
        f"{VS_DEG_S:.2f} deg/s"
    )

    # ============================================================
    # Replay configuration
    # ============================================================

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    g7.b2.v35.METERS_PER_TICK = 5.70e-6

    free_xml = (
        out
        /
        "g30a_free_no_fixture.xml"
    )

    make_free_xml(
        xml,
        free_xml,
    )

    # ============================================================
    # STEP 1 — PRE-CONTACT / FIXTURE DISABLED
    # ============================================================

    print()
    print("=" * 150)
    print("G30A STEP 1 — H1 APPROACH, FIXTURE DISABLED")
    print("=" * 150)

    try:
        (
            (
                free_rows,
                free_qpos,
                free_times,
                free_summary,
            ),
            free_friction_history,
            free_spool_drift,
        ) = run_case(
            free_xml,
            bag,
            replay,
        )

        enc = np.asarray(
            [
                r["encoder_tick"]
                for r in free_rows
            ],
            dtype=float,
        )

        all_errors = []

        print()
        print(
            "encoder       HW MCP/PIP/DIP            "
            "SIM MCP/PIP/DIP           error"
        )
        print("-" * 145)

        for tick, hm, hp, hd in H1_POINTS:
            i = nearest(
                enc,
                tick,
            )

            row = free_rows[i]

            sim = np.array(
                [
                    row["mcp_deg"],
                    row["pip_deg"],
                    row["dip_deg"],
                ]
            )

            hw = np.array(
                [hm, hp, hd]
            )

            error = sim - hw

            all_errors.extend(
                error.tolist()
            )

            print(
                f"{tick:8.3f}     "
                f"{hm:7.3f}/{hp:7.3f}/{hd:7.3f}     "
                f"{sim[0]:7.3f}/{sim[1]:7.3f}/{sim[2]:7.3f}     "
                f"{error[0]:+7.3f}/"
                f"{error[1]:+7.3f}/"
                f"{error[2]:+7.3f}"
            )

        approach_rmse = float(
            np.sqrt(
                np.mean(
                    np.asarray(
                        all_errors
                    ) ** 2
                )
            )
        )

        print()
        print(
            f"APPROACH RMSE = "
            f"{approach_rmse:.3f} deg"
        )

        # ========================================================
        # STEP 2 — H1 CONTACT
        # ========================================================

        print()
        print("=" * 150)
        print("G30A STEP 2 — ONE H1 CONTACT RUN")
        print("=" * 150)

        (
            (
                rows,
                qpos,
                times,
                summary,
            ),
            contact_friction_history,
            contact_spool_drift,
        ) = run_case(
            xml,
            bag,
            replay,
        )

    finally:
        g7.b2.v35.METERS_PER_TICK = (
            old_scale
        )

    mid_starts, mid_ends = episodes(
        rows,
        "mid_contact",
    )

    tip_starts, tip_ends = episodes(
        rows,
        "tip_contact",
    )

    if not mid_starts:
        raise RuntimeError(
            "No middle contact occurred."
        )

    first = int(
        mid_starts[0]
    )

    first_end = int(
        mid_ends[0]
    )

    first_row = rows[first]

    first_pose = np.array(
        [
            first_row["mcp_deg"],
            first_row["pip_deg"],
            first_row["dip_deg"],
        ],
        dtype=float,
    )

    first_error = (
        first_pose
        -
        HW_CONTACT_POSE
    )

    first_rmse = float(
        np.sqrt(
            np.mean(
                first_error ** 2
            )
        )
    )

    first_duration = (
        rows[first_end]["time_s"]
        -
        first_row["time_s"]
    )

    if len(mid_starts) >= 2:
        detached_gap = (
            rows[
                mid_starts[1]
            ]["time_s"]
            -
            rows[
                first_end
            ]["time_s"]
        )
    else:
        detached_gap = 0.0

    print()
    print("FIRST MIDDLE CONTACT")
    print(
        f"  encoder = "
        f"{first_row['encoder_tick']:.3f}"
    )
    print(
        f"  HW      = "
        f"{HW_CONTACT_ENCODER:.3f}"
    )
    print(
        f"  error   = "
        f"{first_row['encoder_tick'] - HW_CONTACT_ENCODER:+.3f}"
    )
    print()

    print(
        "  simulation = "
        f"{first_pose[0]:.3f} / "
        f"{first_pose[1]:.3f} / "
        f"{first_pose[2]:.3f}"
    )

    print(
        "  hardware   = "
        f"{HW_CONTACT_POSE[0]:.3f} / "
        f"{HW_CONTACT_POSE[1]:.3f} / "
        f"{HW_CONTACT_POSE[2]:.3f}"
    )

    print(
        "  error      = "
        f"{first_error[0]:+.3f} / "
        f"{first_error[1]:+.3f} / "
        f"{first_error[2]:+.3f}"
    )

    print(
        f"  pose RMSE  = "
        f"{first_rmse:.3f} deg"
    )

    print()
    print("CONTACT CONTINUITY")
    print(
        f"  middle episodes = "
        f"{len(mid_starts)}"
    )
    print(
        f"  tip episodes    = "
        f"{len(tip_starts)}"
    )
    print(
        f"  first duration  = "
        f"{first_duration:.3f} s"
    )
    print(
        f"  detached gap    = "
        f"{detached_gap:.3f} s"
    )

    # ============================================================
    # Trusted final pose
    # ============================================================

    row_times = np.asarray(
        [
            r["time_s"]
            for r in rows
        ]
    )

    ti = nearest(
        row_times,
        TARGET_TIME,
    )

    final = np.array(
        [
            rows[ti]["mcp_deg"],
            rows[ti]["pip_deg"],
            rows[ti]["dip_deg"],
        ]
    )

    final_error = (
        final
        -
        TRUSTED_FINAL
    )

    final_rmse = float(
        np.sqrt(
            np.mean(
                final_error ** 2
            )
        )
    )

    print()
    print("TRUSTED FINAL H1")

    print(
        "  simulation = "
        f"{final[0]:.3f} / "
        f"{final[1]:.3f} / "
        f"{final[2]:.3f}"
    )

    print(
        "  hardware   = "
        f"{TRUSTED_FINAL[0]:.3f} / "
        f"{TRUSTED_FINAL[1]:.3f} / "
        f"{TRUSTED_FINAL[2]:.3f}"
    )

    print(
        f"  RMSE       = "
        f"{final_rmse:.3f} deg"
    )

    # ============================================================
    # Save friction histories
    # ============================================================

    def save_history(path, history):
        if not history:
            return

        with path.open(
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    history[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                history
            )

    save_history(
        out / "g30a_free_friction_history.csv",
        free_friction_history,
    )

    save_history(
        out / "g30a_contact_friction_history.csv",
        contact_friction_history,
    )

    summary_csv = (
        out
        /
        "g30a_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "approach_rmse_deg",
                "first_contact_encoder",
                "first_contact_encoder_error",
                "first_contact_pose_rmse_deg",
                "middle_episodes",
                "tip_episodes",
                "middle_gap_s",
                "final_rmse_deg",
                "free_spool_drift_mm",
                "contact_spool_drift_mm",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "approach_rmse_deg":
                approach_rmse,

            "first_contact_encoder":
                first_row[
                    "encoder_tick"
                ],

            "first_contact_encoder_error":
                first_row[
                    "encoder_tick"
                ]
                -
                HW_CONTACT_ENCODER,

            "first_contact_pose_rmse_deg":
                first_rmse,

            "middle_episodes":
                len(mid_starts),

            "tip_episodes":
                len(tip_starts),

            "middle_gap_s":
                detached_gap,

            "final_rmse_deg":
                final_rmse,

            "free_spool_drift_mm":
                free_spool_drift,

            "contact_spool_drift_mm":
                contact_spool_drift,
        })

    np.savez_compressed(
        out
        /
        "g30a_contact_trajectory.npz",
        times=np.asarray(times),
        qpos=np.asarray(qpos),
    )

    # ============================================================
    # HARD STOPPING RULE
    # ============================================================

    print()
    print("=" * 150)
    print("G30A DECISION")
    print("=" * 150)

    print(
        f"approach RMSE         = "
        f"{approach_rmse:.3f} deg"
    )

    print(
        f"first-contact RMSE    = "
        f"{first_rmse:.3f} deg"
    )

    print(
        f"first-contact enc err = "
        f"{first_row['encoder_tick'] - HW_CONTACT_ENCODER:+.1f} ticks"
    )

    print(
        f"middle detached gap   = "
        f"{detached_gap:.3f} s"
    )

    print(
        f"final RMSE            = "
        f"{final_rmse:.3f} deg"
    )

    print()

    if (
        approach_rmse <= 3.0
        and
        first_rmse <= 4.0
    ):

        print(
            "PASS A: velocity-dependent passive resistance "
            "solves the approach/load-sharing problem."
        )

        if detached_gap <= 0.25:
            print(
                "PASS B: contact retention is also acceptable."
            )
            print(
                "Promote this mechanism to parameter identification."
            )
        else:
            print(
                "Contact bounce remains."
            )
            print(
                "FREEZE this pre-contact model and move ONLY "
                "to compliant contact/pad mechanics."
            )

    else:
        print(
            "REJECT STRIBECK PASSIVE CLASS FOR THIS MODEL."
        )
        print(
            "NO parameter scan of Fs/Fc/vs."
        )
        print(
            "Next and only remaining passive candidate is "
            "the real geometric transverse spring mechanism."
        )

    print()
    print(
        f"Summary:\n  {summary_csv}"
    )

    print(
        f"All outputs:\n  {out}"
    )

    # ============================================================
    # GUI
    # ============================================================

    if not args.no_gui:

        view_xml = (
            out
            /
            "g30a_spatial_view.xml"
        )

        g29b.make_view_xml(
            xml,
            view_xml,
        )

        print()
        print("=" * 150)
        print(
            "G30A GUI — SPATIAL TENDON PRESERVED"
        )
        print("=" * 150)

        print(
            "Check ONLY:"
        )
        print(
            "  MCP-dominant approach"
        )
        print(
            "  -> middle contact with low PIP/DIP"
        )
        print(
            "  -> PIP wrapping"
        )
        print(
            "  -> middle retained"
        )
        print(
            "  -> distal/DIP wrapping"
        )

        g7.exact_gui_replay(
            view_xml,
            qpos,
            times,
            "G30A spatial tendon + passive Stribeck",
        )


if __name__ == "__main__":
    main()
