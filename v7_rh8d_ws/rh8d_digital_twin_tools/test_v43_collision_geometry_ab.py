#!/usr/bin/env python3

from pathlib import Path
import csv
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import test_v43_pd_compliant_coupling as v43


WS = Path.home() / (
    "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
)

MJCF = WS / "src/ur5e_mujoco_description/mjcf"
SOURCE = MJCF / "tmp_v43_k750_r20.xml"

OUT = WS / (
    "rh8d_digital_twin_tools/results/"
    "v43_collision_geometry_ab"
)
OUT.mkdir(parents=True, exist_ok=True)

CAPSULE_XML = MJCF / "tmp_v43_mid_capsule_ab.xml"

# --------------------------------------------------
# Build a single comparison model.
# Original source is never modified.
# --------------------------------------------------

tree = ET.parse(SOURCE)
root = tree.getroot()

mesh = root.find(".//geom[@name='collision_idx_mid_mesh']")
capsule = root.find(".//geom[@name='collision_idx_mid']")

if mesh is None or capsule is None:
    raise RuntimeError("Expected middle collision geoms missing")

print("Original middle collision mesh:")
print(" ", mesh.attrib)

print("\nOriginal middle collision capsule:")
print(" ", capsule.attrib)

# Switch off the mesh collision.
mesh.set("contype", "0")
mesh.set("conaffinity", "0")

# Enable the CAD model's preexisting capsule.
capsule.set("contype", "2")
capsule.set("conaffinity", "33")

# Only visualization color; does not affect physics.
capsule.set("rgba", "0 0.8 1 0.45")

tree.write(
    CAPSULE_XML,
    encoding="unicode",
    xml_declaration=True
)

# Compiles from the original MJCF directory,
# preserving relative mesh paths.
mujoco.MjModel.from_xml_path(str(CAPSULE_XML))


def name_id(model, objtype, name):
    return int(
        mujoco.mj_name2id(model, objtype, name)
    )


def capture(xml, label):
    recorded = []
    original_step = mujoco.mj_step
    cache = {}

    def recorder(model, data, *args, **kwargs):
        out = original_step(model, data, *args, **kwargs)

        if not (4.0 <= data.time <= 8.402):
            return out

        key = id(model)

        if cache.get("key") != key:
            cache.clear()
            cache["key"] = key

            names = {
                "fixture": "index_contact_fixture",
                "middle_mesh": "collision_idx_mid_mesh",
                "middle_capsule": "collision_idx_mid",
                "tip": "collision_idx_tip",
            }

            for field, name in names.items():
                cache[field] = name_id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    name
                )

            joints = [
                "Index_Proximal--palm_assembly:1",
                "Index_Middle--Index_Proximal",
                "Index_Distal--Index_Middle",
            ]

            cache["qaddr"] = []

            for j in joints:
                jid = name_id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    j
                )
                if jid < 0:
                    raise RuntimeError(f"Missing joint: {j}")

                cache["qaddr"].append(
                    int(model.jnt_qposadr[jid])
                )

        fm = 0.0
        ft = 0.0

        for i in range(data.ncon):
            contact = data.contact[i]

            a = int(contact.geom1)
            b = int(contact.geom2)

            fixture = cache["fixture"]

            if fixture not in (a, b):
                continue

            partner = b if a == fixture else a

            if partner not in (
                cache["middle_mesh"],
                cache["middle_capsule"],
                cache["tip"],
            ):
                continue

            force = np.zeros(6)
            mujoco.mj_contactForce(
                model, data, i, force
            )

            normal = max(0.0, float(force[0]))

            if partner == cache["tip"]:
                ft += normal
            else:
                fm += normal

        angles = np.degrees(
            data.qpos[cache["qaddr"]]
        )

        recorded.append({
            "time_s": float(data.time),
            "middle_normal_N": fm,
            "tip_normal_N": ft,
            "total_normal_N": fm + ft,
            "mcp_deg": float(angles[0]),
            "pip_deg": float(angles[1]),
            "dip_deg": float(angles[2]),
        })

        return out

    mujoco.mj_step = recorder

    try:
        result = v43.run_contact(xml, label)
    finally:
        mujoco.mj_step = original_step

    if len(recorded) < 100:
        raise RuntimeError(f"Insufficient capture: {label}")

    csv_path = OUT / f"{label}_native.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(recorded[0].keys())
        )
        writer.writeheader()
        writer.writerows(recorded)

    def column(key):
        return np.array(
            [r[key] for r in recorded], dtype=float
        )

    t = column("time_s")
    total = column("total_normal_N")
    fm = column("middle_normal_N")
    ft = column("tip_normal_N")

    dt = float(np.median(np.diff(t)))

    target = np.array([14.5355, 23.8110, 9.3270])

    q = np.column_stack([
        column("mcp_deg"),
        column("pip_deg"),
        column("dip_deg"),
    ])

    rmse = np.sqrt(
        np.mean((q - target)**2, axis=1)
    )

    # A simultaneous *loaded* contact means
    # both parts carry nonzero normal force.
    interval = (t >= 6.191) & (t <= 8.201)

    loaded_both = (fm > 0.02) & (ft > 0.02)
    loaded_any = (fm > 0.02) | (ft > 0.02)

    return {
        "label": label,
        "reported_rmse_deg": float(
            result["rmse_deg"]
        ),
        "minimum_instant_rmse_deg": float(
            np.min(rmse)
        ),
        "terminal_rmse_deg": float(
            rmse[-1]
        ),
        "peak_force_N": float(
            np.max(total)
        ),
        "p95_force_N": float(
            np.percentile(total, 95)
        ),
        "above_2N_ms": float(
            np.sum(total > 2.0) * dt * 1000
        ),
        "above_50N_ms": float(
            np.sum(total > 50.0) * dt * 1000
        ),
        "force_impulse_Ns": float(
            np.trapezoid(total, t)
            if hasattr(np, "trapezoid")
            else np.trapz(total, t)
        ),
        "both_loaded_fraction": float(
            np.mean(loaded_both[interval])
        ),
        "any_loaded_fraction": float(
            np.mean(loaded_any[interval])
        ),
        "terminal_mcp_deg": float(q[-1, 0]),
        "terminal_pip_deg": float(q[-1, 1]),
        "terminal_dip_deg": float(q[-1, 2]),
    }


print("\n" + "=" * 90)
print("V43 COLLISION GEOMETRY A/B")
print("=" * 90)

results = []

for label, xml in [
    ("A_original_mesh", SOURCE),
    ("B_existing_capsule", CAPSULE_XML),
]:
    print(f"\nRunning {label}")
    result = capture(xml, label)
    results.append(result)

    for key, value in result.items():
        print(f"  {key}: {value}")

summary = OUT / "comparison.csv"

with summary.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 90)
print("COMPARISON")
print("=" * 90)

for key in [
    "reported_rmse_deg",
    "terminal_rmse_deg",
    "peak_force_N",
    "p95_force_N",
    "above_2N_ms",
    "above_50N_ms",
    "force_impulse_Ns",
    "both_loaded_fraction",
    "any_loaded_fraction",
]:
    print(
        f"{key:30s} "
        f"mesh={results[0][key]:10.4f} "
        f"capsule={results[1][key]:10.4f}"
    )

print("\nSaved:", summary)
print("Original V43 XML remains unchanged.")
