#!/usr/bin/env python3

from pathlib import Path
import csv
import numpy as np
import mujoco

import test_v43_pd_compliant_coupling as v43

WS = Path.home() / (
    "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
)

XML = WS / (
    "src/ur5e_mujoco_description/mjcf/"
    "tmp_v43_k750_r20.xml"
)

OUT = WS / (
    "rh8d_digital_twin_tools/results/"
    "v43_contact_equilibrium_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

TARGET_TIMES = [
    4.373, 5.382, 5.708,
    6.191, 6.589, 6.970,
    7.238, 7.296, 7.372,
    7.458, 7.992, 8.201, 8.401
]

REAL_STEP = mujoco.mj_step
records = []
cache = {}

def gid(model, name):
    return int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, name
    ))

def recording_step(model, data, *args, **kwargs):
    result = REAL_STEP(model, data, *args, **kwargs)

    if not (4.0 <= data.time <= 8.402):
        return result

    # Retrieve geom IDs for each model.
    model_key = id(model)
    if model_key not in cache:
        cache.clear()
        cache[model_key] = {
            "fixture": gid(model, "index_contact_fixture"),
            "middle": [
                gid(model, "collision_idx_mid_mesh"),
                gid(model, "collision_idx_mid"),
            ],
            "tip": [
                gid(model, "collision_idx_tip"),
            ],
        }

    ids = cache[model_key]

    fm = 0.0
    ft = 0.0
    dm = float("nan")
    dt = float("nan")

    middle_dists = []
    tip_dists = []

    for i in range(data.ncon):
        c = data.contact[i]
        a, b = int(c.geom1), int(c.geom2)

        fixture = ids["fixture"]
        if fixture not in (a, b):
            continue

        partner = b if a == fixture else a

        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)

        fn = max(0.0, float(force[0]))

        if partner in ids["middle"]:
            fm += fn
            middle_dists.append(float(c.dist))

        elif partner in ids["tip"]:
            ft += fn
            tip_dists.append(float(c.dist))

    if middle_dists:
        dm = min(middle_dists) * 1000.0

    if tip_dists:
        dt = min(tip_dists) * 1000.0

    records.append({
        "time_s": float(data.time),
        "middle_contact_gap_mm": dm,
        "tip_contact_gap_mm": dt,
        "middle_normal_N": fm,
        "tip_normal_N": ft,
        "total_normal_N": fm + ft,
    })

    return result


print("\nV43 ORIGINAL-STEP FORCE VERIFICATION")
print("=" * 100)

model = mujoco.MjModel.from_xml_path(str(XML))
fixture = gid(model, "index_contact_fixture")

print("Fixture compiled solref:", model.geom_solref[fixture])
print("Fixture compiled solimp:", model.geom_solimp[fixture])
print("Fixture priority:", model.geom_priority[fixture])
print("Physics timestep:", model.opt.timestep)

mujoco.mj_step = recording_step

try:
    result = v43.run_contact(
        XML, "V43_NATIVE_FORCE_VERIFICATION"
    )
finally:
    mujoco.mj_step = REAL_STEP

print("\nReplayed contact RMSE:", result["rmse_deg"])
print("Recorded solver steps:", len(records))

if len(records) < 100:
    raise RuntimeError("Native force capture failed.")

path = OUT / "v43_native_forces.csv"

with path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(records[0].keys())
    )
    writer.writeheader()
    writer.writerows(records)

print("\nNATIVE CONTACT FORCES")
print("=" * 100)
print(
    "time     mid_gap    tip_gap      "
    "Fmid       Ftip       Ftotal"
)
print(
    " s          mm         mm         "
    "N          N           N"
)
print("-" * 100)

for target in TARGET_TIMES:
    r = min(
        records,
        key=lambda x: abs(x["time_s"] - target)
    )
    print(
        f"{r['time_s']:6.3f} "
        f"{r['middle_contact_gap_mm']:+10.4f} "
        f"{r['tip_contact_gap_mm']:+10.4f} "
        f"{r['middle_normal_N']:10.3f} "
        f"{r['tip_normal_N']:10.3f} "
        f"{r['total_normal_N']:11.3f}"
    )

peak = max(
    records, key=lambda x: x["total_normal_N"]
)

print("\nPEAK NATIVE NORMAL FORCE:")
print(
    f"t={peak['time_s']:.3f}s, "
    f"force={peak['total_normal_N']:.3f}N"
)

forces = np.asarray([
    x["total_normal_N"] for x in records
])
print(
    "95th percentile normal force:",
    f"{np.percentile(forces, 95):.3f} N"
)

print("\nSaved:", path)
print("Original physics parameters unchanged.")
