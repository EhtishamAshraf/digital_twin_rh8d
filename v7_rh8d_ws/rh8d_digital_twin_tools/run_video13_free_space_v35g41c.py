#!/usr/bin/env python3
"""
G41C: Standalone RH8D free-space dynamic replay.

Input: measured index motor encoder from the complete ROS bag.
Model: G39A spatial flexor + original passives + direct tendon drive.

No H1 contact pipeline.
No finger-joint trajectory prescription.
No parameter optimization.
The UR5e/mounting joints are held at the experimental tabletop pose.
"""

import argparse
import csv
import math
import sqlite3
import shutil
import time
from pathlib import Path

import numpy as np
import mujoco

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import test_direct_spatial_tendon_drive_v35g32a as g32
import audit_index_tendon_transmission_v35g12 as g12

from rh8d_tabletop_stage_v1 import (
    TABLETOP_UR5,
    HAND_STAGE,
    apply_tabletop_stage,
)

ENCODER_ZERO = 100.0
METERS_PER_TICK = 5.70e-6

# Frozen preliminary VIDEO-13 geometric input identification.
SHORTENING_RATIO = 0.746379

JOINTS = ("mcp", "pip", "dip")
MAIN_TENDON = "idx_flex_spatial"
ACTUATOR = "g32_index_tendon_position"

# Encoder ticks, MCP, PIP, DIP from video-13 markers.
# Negative near-zero DIP/PIP observations are retained as missing
# measurements, not interpreted as physical negative flexion.
VIDEO13 = [
    (100,    0.000,  0.000,  0.000),
    (500,   16.952,  0.968, -0.428),
    (1000,  39.472, -0.907, -0.343),
    (1500,  57.291,  3.654, -4.019),
    (2000,  73.632,  4.667,  2.012),
    (2500,  73.632, 27.119,  3.190),
    (3000,  77.347, 48.581,  7.810),
    (3500,  80.074, 70.872, -7.199),
]

def read_encoder(bag):
    dbs = sorted(bag.glob("*.db3"))
    if not dbs:
        raise RuntimeError(f"No ROS bag .db3 files in {bag}")

    all_samples = []
    msg_type = get_message("control_msgs/msg/DynamicJointState")

    for db in dbs:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            found = con.execute(
                "SELECT id FROM topics WHERE name=?",
                ("/dynamic_joint_states",)
            ).fetchone()

            if found is None:
                continue

            cursor = con.execute(
                "SELECT timestamp,data FROM messages "
                "WHERE topic_id=? ORDER BY timestamp",
                (found[0],)
            )

            for timestamp, raw in cursor:
                msg = deserialize_message(raw, msg_type)

                if "index_flexion_motor" not in msg.joint_names:
                    continue

                index = list(msg.joint_names).index(
                    "index_flexion_motor"
                )

                interfaces = msg.interface_values[index]

                if "position" not in interfaces.interface_names:
                    continue

                j = list(interfaces.interface_names).index("position")
                encoder = float(interfaces.values[j])

                if np.isfinite(encoder):
                    all_samples.append((int(timestamp), encoder))

    if not all_samples:
        raise RuntimeError("Index encoder feedback not found in bag")

    all_samples = sorted(set(all_samples))
    ns = np.array([s[0] for s in all_samples], dtype=np.int64)
    encoder = np.array([s[1] for s in all_samples], dtype=float)
    t = (ns - ns[0]) * 1e-9

    if encoder.max() < 3000:
        raise RuntimeError(
            "Wrong bag: index motor never reached 3000 ticks"
        )

    if abs(encoder[0] - 100) > 30:
        raise RuntimeError(
            f"Unexpected initial encoder: {encoder[0]:.1f}"
        )

    if abs(encoder[-1] - 100) > 30:
        raise RuntimeError(
            f"Unexpected final encoder: {encoder[-1]:.1f}"
        )

    return t, encoder


def freeze_arm_indices(model, data):
    """Record only fixed UR5e and mounting joint states."""
    targets = []

    for joint_name in list(TABLETOP_UR5) + list(HAND_STAGE):
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if jid < 0:
            raise RuntimeError(f"Staging joint missing: {joint_name}")

        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])
        targets.append((qadr, dadr, float(data.qpos[qadr])))

    return targets


def enforce_static_arm(data, targets):
    # Only UR5e/mounting joints are fixed.
    # MCP/PIP/DIP remain fully dynamic.
    for qadr, dadr, value in targets:
        data.qpos[qadr] = value
        data.qvel[dadr] = 0.0


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_video13(rows, t_input, enc_input, out):
    first_peak = int(np.argmax(enc_input))
    peak_time = float(t_input[first_peak])

    # Use only rising/closing trajectory.
    closing = [r for r in rows if r["time_s"] <= peak_time]

    comparison = []
    errors = {name: [] for name in JOINTS}

    print("\n" + "=" * 115)
    print("G41C — VIDEO-13 CLOSING BRANCH COMPARISON")
    print("=" * 115)
    print(
        "Tick      HW MCP/PIP/DIP             "
        "SIM MCP/PIP/DIP          Recorded encoder"
    )
    print("-" * 115)

    for tick, hm, hp, hd in VIDEO13:
        # Select the last nearby encoder reading on the closing
        # branch: avoids taking the first transient of a plateau.
        candidates = [
            r for r in closing
            if abs(r["encoder_tick"] - tick) <= 6.0
        ]

        if not candidates:
            row = min(
                closing,
                key=lambda r: abs(r["encoder_tick"] - tick)
            )
        else:
            row = candidates[-1]

        hw = {"mcp": hm, "pip": hp, "dip": hd}
        sim = {
            key: float(row[f"{key}_deg"])
            for key in JOINTS
        }

        record = {
            "target_encoder_tick": tick,
            "sample_encoder_tick": row["encoder_tick"],
            "time_s": row["time_s"],
        }

        for key in JOINTS:
            observed = hw[key]
            # Negative marker estimates are not reliable measurements.
            reliable = observed >= 0

            # At high closing, distal marker tracking was unreliable.
            if key == "dip" and tick == 3500:
                reliable = False

            record[f"hw_{key}_deg"] = (
                observed if reliable else float("nan")
            )
            record[f"sim_{key}_deg"] = sim[key]
            record[f"err_{key}_deg"] = (
                sim[key] - observed
                if reliable else float("nan")
            )

            if reliable and tick > 100:
                errors[key].append(sim[key] - observed)

        comparison.append(record)

        def show(x):
            return f"{x:7.2f}" if np.isfinite(x) else "    n/a"

        print(
            f"{tick:4.0f}   "
            f"{show(record['hw_mcp_deg'])}/"
            f"{show(record['hw_pip_deg'])}/"
            f"{show(record['hw_dip_deg'])}       "
            f"{sim['mcp']:7.2f}/{sim['pip']:7.2f}/{sim['dip']:7.2f}"
            f"          {row['encoder_tick']:8.1f}"
        )

    print("\nVIDEO-13 RMSE (available marker measurements)")
    for key in JOINTS:
        arr = np.asarray(errors[key], dtype=float)
        rmse = math.sqrt(np.mean(arr**2)) if len(arr) else float("nan")
        print(f"  {key.upper()}: {rmse:.3f} deg ({len(arr)} points)")

    all_errors = np.concatenate([
        np.asarray(v, dtype=float)
        for v in errors.values() if len(v)
    ])
    print(
        f"  COMBINED: {np.sqrt(np.mean(all_errors**2)):.3f} deg"
    )

    save_csv(out / "g41c_video13_comparison.csv", comparison)

    return peak_time


def plot_results(rows, comparison, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; CSV/NPZ outputs are preserved.")
        return

    t = np.array([r["time_s"] for r in rows])

    for key in JOINTS:
        plt.figure(figsize=(10, 4))
        plt.plot(
            t, [r[f"{key}_deg"] for r in rows],
            label=f"MuJoCo {key.upper()}"
        )

        ct = [
            r["time_s"] for r in comparison
            if np.isfinite(r[f"hw_{key}_deg"])
        ]
        cv = [
            r[f"hw_{key}_deg"] for r in comparison
            if np.isfinite(r[f"hw_{key}_deg"])
        ]

        plt.scatter(ct, cv, label="Video-13 hardware markers")
        plt.xlabel("Time since bag start (s)")
        plt.ylabel("Flexion angle (deg)")
        plt.title(f"{key.upper()}: hardware vs dynamic simulation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"g41c_{key}_comparison.png", dpi=160)
        plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, [r["encoder_tick"] for r in rows])
    plt.xlabel("Time since bag start (s)")
    plt.ylabel("Index encoder (ticks)")
    plt.title("Measured motor encoder: complete closing/opening sweep")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out / "g41c_motor_encoder.png", dpi=160)
    plt.close()


def show_dynamic_replay(xml, times, qpos, encoder, peak_time):
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    # Display the dynamic CLOSING branch only, then hold its
    # final recorded state. This is not a new simulation.
    end = int(np.searchsorted(times, peak_time, side="right"))
    end = max(1, min(end, len(times)))

    print("\n" + "=" * 100)
    print("G41C GUI — RECORDED DYNAMIC CLOSING-BRANCH REPLAY")
    print("These qpos samples come from the completed MuJoCo simulation.")
    print("No joint trajectories are imposed during the simulation.")
    print("Playback speed: 2x. GUI stays open after playback.")
    print("=" * 100)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(end):
            if not viewer.is_running():
                return

            data.qpos[:] = qpos[i]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()

            if i + 1 < end:
                dt = max(0, times[i+1] - times[i])
                time.sleep(min(dt / 2.0, 0.05))

        print(
            f"Closing replay complete; encoder near "
            f"{encoder[end-1]:.1f} ticks."
        )
        print("Viewer remains open until you close it.")

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bag", type=Path)
    p.add_argument("--xml", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--no-gui", action="store_true")
    args = p.parse_args()

    bag = args.bag.expanduser().resolve()
    source = args.xml.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Validate the bag BEFORE doing any expensive model work.
    if not bag.is_dir() or not list(bag.glob("*.db3")):
        raise RuntimeError(f"Invalid free-space ROS bag: {bag}")

    t_input, enc_input = read_encoder(bag)

    print("\n" + "=" * 100)
    print("G41C — VERIFIED FREE-SPACE HARDWARE INPUT")
    print("=" * 100)
    print(f"Bag: {bag}")
    print(f"Messages: {len(enc_input)}")
    print(f"Duration: {t_input[-1]:.3f} s")
    print(f"Encoder min/max: {enc_input.min():.1f}/{enc_input.max():.1f}")
    print(f"Initial/final encoder: {enc_input[0]:.1f}/{enc_input[-1]:.1f}")

    direct_xml = out / "g41c_direct.xml"
    free_xml = out / "g41c_direct_no_bottle.xml"

    g32.build_direct_model(source, direct_xml)
    g32.disable_fixture(direct_xml, free_xml)

    model = mujoco.MjModel.from_xml_path(str(free_xml))
    data = mujoco.MjData(model)

    tid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_TENDON, MAIN_TENDON
    )
    aid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, ACTUATOR
    )

    if tid < 0 or aid < 0:
        raise RuntimeError("Required spatial tendon/actuator missing")

    if int(model.actuator_trnid[aid, 0]) != tid:
        raise RuntimeError("Direct actuator does not target idx_flex_spatial")

    mujoco.mj_resetData(model, data)
    apply_tabletop_stage(model, data, reset_to_qpos0=True)

    refs = g12.get_joint_refs(model)

    # Open index finger is the common video-13 reference.
    for key in JOINTS:
        data.qpos[refs[key]["qadr"]] = 0.0

    mujoco.mj_forward(model, data)

    hold = freeze_arm_indices(model, data)
    open_length = float(data.ten_length[tid])

    print("\nMODEL/INPUT CHECK")
    print(f"Open spatial tendon length: {open_length*1000:.4f} mm")
    print(f"Encoder-to-shortening ratio: {SHORTENING_RATIO:.6f}")
    print(f"MuJoCo timestep: {model.opt.timestep:.6f} s")
    print("Bottle: disabled (free-space experiment)")
    print("UR5e: fixed at verified tabletop staging")
    print("Index MCP/PIP/DIP: fully dynamic, independent")
    print("Joint passive parameters: unchanged")

    if not 0.070 < open_length < 0.085:
        raise RuntimeError("Unexpected direct tendon reference length")

    records = []
    recorded_qpos = []
    recorded_times = []
    next_record = 0.0
    record_period = 0.02  # 50 Hz output

    while data.time < float(t_input[-1]):
        encoder = float(np.interp(data.time, t_input, enc_input))

        shortening = (
            SHORTENING_RATIO
            * METERS_PER_TICK
            * max(0.0, encoder - ENCODER_ZERO)
        )

        desired_length = open_length - shortening
        data.ctrl[aid] = desired_length

        mujoco.mj_step(model, data)

        # The real UR5e was held fixed throughout the finger test.
        # Do not impose motion on the three finger joints.
        enforce_static_arm(data, hold)
        mujoco.mj_forward(model, data)

        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError(
                f"Non-finite MuJoCo state at t={data.time:.4f}"
            )

        if data.time + 1e-9 < next_record:
            continue

        row = {
            "time_s": float(data.time),
            "encoder_tick": encoder,
            "target_tendon_mm": desired_length * 1000,
            "actual_tendon_mm": float(data.ten_length[tid]) * 1000,
            "tendon_tracking_error_mm": (
                float(data.ten_length[tid]) - desired_length
            ) * 1000,
        }

        for key in JOINTS:
            row[f"{key}_deg"] = math.degrees(
                float(data.qpos[refs[key]["qadr"]])
            )

        records.append(row)
        recorded_qpos.append(data.qpos.copy())
        recorded_times.append(float(data.time))
        next_record += record_period

    print("\n" + "=" * 100)
    print("G41C — COMPLETE DYNAMIC REPLAY FINISHED")
    print("=" * 100)
    print(f"Simulated time: {data.time:.3f} s")
    print(f"Recorded trajectory samples: {len(records)}")

    tracking = np.array([
        abs(r["tendon_tracking_error_mm"]) for r in records
    ])
    print(f"Mean tendon tracking error: {tracking.mean():.4f} mm")
    print(f"Max tendon tracking error: {tracking.max():.4f} mm")

    save_csv(out / "g41c_full_dynamic_trajectory.csv", records)

    peak_time = analyze_video13(records, t_input, enc_input, out)

    with (out / "g41c_video13_comparison.csv").open() as f:
        comparison = list(csv.DictReader(f))
        for row in comparison:
            for k, v in row.items():
                row[k] = float(v)

    plot_results(records, comparison, out)

    recorded_times = np.asarray(recorded_times)
    recorded_qpos = np.asarray(recorded_qpos)
    recorded_enc = np.array([r["encoder_tick"] for r in records])

    np.savez_compressed(
        out / "g41c_dynamic_states.npz",
        times=recorded_times,
        qpos=recorded_qpos,
        encoder=recorded_enc,
    )

    shutil.copy2(Path(__file__), out / Path(__file__).name)

    print(f"\nAll results: {out}")

    if not args.no_gui:
        show_dynamic_replay(
            free_xml,
            recorded_times,
            recorded_qpos,
            recorded_enc,
            peak_time,
        )


if __name__ == "__main__":
    main()
