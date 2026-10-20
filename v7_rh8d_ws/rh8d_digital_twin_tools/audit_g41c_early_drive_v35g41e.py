#!/usr/bin/env python3
import csv
import shutil
from pathlib import Path

import mujoco
import numpy as np

import audit_index_tendon_transmission_v35g12 as g12
from rh8d_tabletop_stage_v1 import apply_tabletop_stage

BASE = Path.home() / "rh8d_results/v35g41c_video13_free_space"
OUT = Path.home() / "rh8d_results/v35g41e_input_force_audit"
OUT.mkdir(parents=True, exist_ok=True)

XML = BASE / "g41c_direct_no_bottle.xml"
NPZ = BASE / "g41c_dynamic_states.npz"
TRACE = BASE / "g41c_full_dynamic_trajectory.csv"
JAC = BASE / "g41d_closing_branch_autopsy.csv"

with np.load(NPZ, allow_pickle=False) as z:
    qposes = z["qpos"].copy()
    times = z["times"].copy()

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

rows = read_csv(TRACE)
jacrows = read_csv(JAC)

assert len(rows) == len(qposes), "Recorded state mismatch"

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

refs = g12.get_joint_refs(model)
tendon = model.tendon("idx_flex_spatial").id
act = model.actuator("g32_index_tendon_position").id

apply_tabletop_stage(model, data, reset_to_qpos0=True)
staging_ctrl = data.ctrl.copy()

# Restrict all diagnostic selection to the first closing branch.
peak = int(np.argmax([
    float(r["encoder_tick"]) for r in rows
]))

def get_number(i, field):
    return float(rows[i][field])

def first_where(predicate):
    for i in range(peak + 1):
        if predicate(i):
            return i
    return None

events = [
    ("initial", 0),
    ("first_encoder_300", first_where(
        lambda i: get_number(i, "encoder_tick") >= 300
    )),
    ("first_tracking_over_1mm", first_where(
        lambda i: abs(get_number(
            i, "tendon_tracking_error_mm"
        )) >= 1.0
    )),
    ("first_encoder_500", first_where(
        lambda i: get_number(i, "encoder_tick") >= 500
    )),
    ("first_encoder_1000", first_where(
        lambda i: get_number(i, "encoder_tick") >= 1000
    )),
    ("first_DIP_over_35deg", first_where(
        lambda i: get_number(i, "dip_deg") >= 35.0
    )),
]

print("=" * 105)
print("G41E — EARLY ACTUATOR / CONTACT FORCE DIAGNOSIS")
print("=" * 105)
print("Input: existing G41C dynamically simulated states")
print("Physics: NO NEW mj_step() CALLS")
print("Joint torques are reconstructed at zero velocity.")
print("They are diagnostic snapshots, NOT recorded dynamic torques.")
print()

print("DIRECT ACTUATOR:")
print("  name:", model.actuator(act).name)
print("  tendon:", model.tendon(tendon).name)
print("  kp:", model.actuator_gainprm[act, 0])
print("  force limited:", model.actuator_forcelimited[act])
print()

results = []

for event_name, i in events:
    if i is None:
        continue

    data.qpos[:] = qposes[i]
    data.qvel[:] = 0.0
    data.ctrl[:] = staging_ctrl

    target_mm = get_number(i, "target_tendon_mm")
    data.ctrl[act] = target_mm / 1000.0

    mujoco.mj_forward(model, data)

    actual_mm = float(data.ten_length[tendon]) * 1000.0
    force = float(data.actuator_force[act])

    print("-" * 105)
    print(event_name)
    print(
        f"  time={times[i]:.3f} s, "
        f"encoder={get_number(i,'encoder_tick'):.2f} ticks"
    )
    print(
        "  MCP/PIP/DIP = "
        f"{get_number(i,'mcp_deg'):.2f}/"
        f"{get_number(i,'pip_deg'):.2f}/"
        f"{get_number(i,'dip_deg'):.2f} deg"
    )
    print(
        f"  tendon target={target_mm:.4f} mm, "
        f"actual={actual_mm:.4f} mm, "
        f"error={actual_mm-target_mm:+.4f} mm"
    )
    print(
        f"  reconstructed actuator force = {force:+.3f} N"
    )

    result = {
        "event": event_name,
        "time_s": float(times[i]),
        "encoder_tick": get_number(i, "encoder_tick"),
        "target_tendon_mm": target_mm,
        "actual_tendon_mm": actual_mm,
        "actuator_force_N": force,
        "contact_count": int(data.ncon),
    }

    print(
        "  joint    actuator_Nm    passive_Nm"
        "    constraint_Nm    bias_Nm"
    )

    for key in ("mcp", "pip", "dip"):
        dadr = refs[key]["dadr"]

        act_torque = float(data.qfrc_actuator[dadr])
        passive = float(data.qfrc_passive[dadr])
        constraint = float(data.qfrc_constraint[dadr])
        bias = float(data.qfrc_bias[dadr])

        print(
            f"  {key.upper():4s}   "
            f"{act_torque:+12.6f} "
            f"{passive:+12.6f} "
            f"{constraint:+15.6f} "
            f"{bias:+11.6f}"
        )

        result[f"{key}_actuator_Nm"] = act_torque
        result[f"{key}_passive_Nm"] = passive
        result[f"{key}_constraint_Nm"] = constraint
        result[f"{key}_bias_Nm"] = bias

    print(f"  MuJoCo detected contacts: {data.ncon}")

    for k in range(min(data.ncon, 10)):
        c = data.contact[k]
        g1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1
        ) or f"geom#{c.geom1}"
        g2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2
        ) or f"geom#{c.geom2}"

        print(
            f"    {g1} <-> {g2}: "
            f"distance={1000*c.dist:+.3f} mm"
        )

    results.append(result)

with (OUT / "g41e_force_diagnosis.csv").open(
    "w", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

shutil.copy2(
    Path(__file__),
    OUT / Path(__file__).name
)

print()
print("=" * 105)
print("G41E DIAGNOSIS COMPLETE")
print("=" * 105)
print(f"Results: {OUT}")
print("No parameter tuning. No new dynamics.")
