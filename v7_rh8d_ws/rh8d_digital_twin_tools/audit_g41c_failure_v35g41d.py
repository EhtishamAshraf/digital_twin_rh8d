#!/usr/bin/env python3
"""Offline autopsy of G41C. No new dynamics or parameter changes."""

import csv
import math
from pathlib import Path

import mujoco
import numpy as np

import audit_index_tendon_transmission_v35g12 as g12
import build_persistent_flexor_wrap_v35g38a as g38a


OUT = Path.home() / "rh8d_results/v35g41c_video13_free_space"
XML = OUT / "g41c_direct_no_bottle.xml"
CSV = OUT / "g41c_full_dynamic_trajectory.csv"
NPZ = OUT / "g41c_dynamic_states.npz"

for path in (XML, CSV, NPZ):
    if not path.is_file():
        raise SystemExit(f"Missing G41C artifact: {path}")

with CSV.open(newline="") as f:
    rows = list(csv.DictReader(f))

with np.load(NPZ, allow_pickle=False) as z:
    qpos = z["qpos"].copy()
    times = z["times"].copy()

if len(rows) != len(qpos):
    raise SystemExit("CSV and dynamic states have different lengths")

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

if qpos.shape[1] != model.nq:
    raise SystemExit("Saved qpos does not match candidate XML")

refs = g12.get_joint_refs(model)

tid = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_TENDON, "idx_flex_spatial"
)

aid = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR,
    "g32_index_tendon_position"
)

if tid < 0 or aid < 0:
    raise SystemExit("Missing expected tendon/actuator")

encoder = np.array(
    [float(r["encoder_tick"]) for r in rows]
)
peak = int(np.argmax(encoder))

print("=" * 110)
print("G41D — RECORDED FREE-SPACE DYNAMICS AUTOPSY")
print("=" * 110)
print(f"Recorded samples: {len(rows)}")
print(f"Closing-branch peak: {encoder[peak]:.2f} ticks")
print(f"Peak time: {times[peak]:.3f} s")
print(f"MuJoCo timestep: {model.opt.timestep:.6f} s")

print("\nDIRECT TENDON ACTUATOR")
print("  Gain parameters:", model.actuator_gainprm[aid, :3])
print("  Bias parameters:", model.actuator_biasprm[aid, :3])
print("  Gear:", model.actuator_gear[aid, :3])

if hasattr(model, "actuator_ctrlrange"):
    print("  Control range:", model.actuator_ctrlrange[aid])
    print("  Control limited:", model.actuator_ctrllimited[aid])

if hasattr(model, "actuator_forcerange"):
    print("  Force range:", model.actuator_forcerange[aid])
    print("  Force limited:", model.actuator_forcelimited[aid])


def inspect(i):
    data.qpos[:] = qpos[i]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    desired_mm = float(rows[i]["target_tendon_mm"])
    data.ctrl[aid] = desired_mm * 0.001

    mujoco.mj_forward(model, data)

    J = g38a.tendon_jacobian_row(model, data, tid)

    result = {
        "sample": i,
        "time_s": float(times[i]),
        "encoder": float(rows[i]["encoder_tick"]),
        "target_mm": desired_mm,
        "actual_mm": float(data.ten_length[tid]) * 1000,
        "tracking_mm": float(
            rows[i]["tendon_tracking_error_mm"]
        ),
        # Reconstructed at zero joint velocity:
        # this is not the recorded dynamic actuator force.
        "static_force_est_N": float(data.actuator_force[aid]),
    }

    for key in ("mcp", "pip", "dip"):
        result[f"{key}_deg"] = math.degrees(
            float(data.qpos[refs[key]["qadr"]])
        )
        result[f"{key}_J_mm_rad"] = (
            1000 * float(J[refs[key]["dadr"]])
        )

    return result


audited = []
events = {
    "first_tracking_error_over_1mm": None,
    "first_DIP_over_30deg": None,
    "first_DIP_over_35deg": None,
    "first_DIP_wrong_moment_arm_sign": None,
    "first_MCP_wrong_moment_arm_sign": None,
    "first_PIP_wrong_moment_arm_sign": None,
}

for i in range(peak + 1):
    r = inspect(i)
    audited.append(r)

    conditions = {
        "first_tracking_error_over_1mm":
            abs(r["tracking_mm"]) >= 1.0,
        "first_DIP_over_30deg":
            r["dip_deg"] >= 30.0,
        "first_DIP_over_35deg":
            r["dip_deg"] >= 35.0,
        "first_DIP_wrong_moment_arm_sign":
            r["dip_J_mm_rad"] >= 0.0,
        "first_MCP_wrong_moment_arm_sign":
            r["mcp_J_mm_rad"] >= 0.0,
        "first_PIP_wrong_moment_arm_sign":
            r["pip_J_mm_rad"] >= 0.0,
    }

    for name, condition in conditions.items():
        if condition and events[name] is None:
            events[name] = r

print("\n" + "=" * 110)
print("FIRST FAILURE EVENTS ON THE CLOSING BRANCH")
print("=" * 110)

for name, r in events.items():
    if r is None:
        print(f"{name}: NOT OBSERVED")
        continue

    print(
        f"{name}\n"
        f"  time={r['time_s']:.3f}s "
        f"encoder={r['encoder']:.1f} "
        f"MCP/PIP/DIP="
        f"{r['mcp_deg']:.2f}/"
        f"{r['pip_deg']:.2f}/"
        f"{r['dip_deg']:.2f} deg\n"
        f"  tracking={r['tracking_mm']:+.3f} mm, "
        f"DIP moment arm={r['dip_J_mm_rad']:+.3f} mm/rad"
    )

print("\n" + "=" * 110)
print("DYNAMICS AT HARDWARE VIDEO-13 ENCODER LEVELS")
print("=" * 110)
print(
    "tick   actualEnc  MCP    PIP    DIP  "
    "J_MCP   J_PIP   J_DIP  track_mm force_est_N"
)

for tick in (100, 500, 1000, 1500, 2000, 2500, 3000, 3500):
    r = min(
        audited,
        key=lambda a: abs(a["encoder"] - tick)
    )

    print(
        f"{tick:4d} {r['encoder']:9.1f} "
        f"{r['mcp_deg']:6.1f} "
        f"{r['pip_deg']:6.1f} "
        f"{r['dip_deg']:6.1f} "
        f"{r['mcp_J_mm_rad']:+7.2f} "
        f"{r['pip_J_mm_rad']:+7.2f} "
        f"{r['dip_J_mm_rad']:+7.2f} "
        f"{r['tracking_mm']:+8.3f} "
        f"{r['static_force_est_N']:+10.3f}"
    )

out_csv = OUT / "g41d_closing_branch_autopsy.csv"

with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(audited[0].keys())
    )
    writer.writeheader()
    writer.writerows(audited)

print("\n" + "=" * 110)
print("G41D DECISION GATE")
print("=" * 110)

if events["first_DIP_wrong_moment_arm_sign"] is not None:
    print("STRUCTURAL DEFECT CONFIRMED:")
    print("The simulated trajectory visits a state where")
    print("the main spatial tendon no longer flexes DIP.")
    print("Do not fit passive parameters to this trajectory.")
else:
    print("No DIP sign reversal observed in saved closing states.")
    print("Investigate the actuator tracking and load sharing first.")

print(f"\nAutopsy CSV: {out_csv}")
print("No simulation was rerun and no model was modified.")
