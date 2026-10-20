#!/usr/bin/env python3
"""Controlled V43 ablations: middle collision shape versus fixture stiffness.

Save alongside test_v43_pd_compliant_coupling.py in rh8d_digital_twin_tools,
then run: python3 v43_controlled_geometry_vs_solver.py
No original XML, hardware data, or physics setup is overwritten.
"""
from __future__ import annotations

import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import test_v43_pd_compliant_coupling as v43

HERE = Path(__file__).resolve().parent
WS = HERE.parent
MJCF = WS / "src/ur5e_mujoco_description/mjcf"
SOURCE = MJCF / "tmp_v43_k750_r20.xml"
OUT = HERE / "results/v43_controlled_geometry_vs_solver"
OUT.mkdir(parents=True, exist_ok=True)

HW_DEG = np.array([14.5355, 23.8110, 9.3270])
JOINTS = (
    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",
)


def object_id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise RuntimeError(f"Missing MuJoCo {kind}: {name}")
    return int(index)


def build_variant(kind):
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    fixture = root.find(".//geom[@name='index_contact_fixture']")
    if fixture is None:
        raise RuntimeError("Missing contact fixture")

    if kind == "capsule_matched_margin":
        mesh = root.find(".//geom[@name='collision_idx_mid_mesh']")
        cap = root.find(".//geom[@name='collision_idx_mid']")
        if mesh is None or cap is None:
            raise RuntimeError("Missing middle collision mesh or capsule")
        # Toggle shape, holding the contact margin at the mesh's 0 mm.
        mesh.set("contype", "0")
        mesh.set("conaffinity", "0")
        cap.set("contype", "2")
        cap.set("conaffinity", "33")
        cap.set("margin", "0")
        cap.set("gap", "0")

    elif kind == "mesh_half_fixture_stiffness":
        # Only the cylinder's direct-format solver stiffness changes.
        # Other geometry, collision margin, damping and tendon parameters
        # are byte-for-byte identical to SOURCE in the XML tree.
        fixture.set("solref", "-14000 -1")
    else:
        raise ValueError(kind)

    destination = MJCF / f"tmp_v43_control_{kind}.xml"
    tree.write(destination, encoding="unicode", xml_declaration=True)
    return destination


def native_capture(xml, label):
    model = mujoco.MjModel.from_xml_path(str(xml))
    fixid = object_id(model, mujoco.mjtObj.mjOBJ_GEOM,
                      "index_contact_fixture")
    print("\nCASE:", label)
    print("XML:", xml)
    print("Compiled fixture solref:", model.geom_solref[fixid])
    print("Compiled fixture priority:", model.geom_priority[fixid])

    samples = []
    original_step = mujoco.mj_step
    cache = {}

    def recorder(m, d, *args, **kwargs):
        result = original_step(m, d, *args, **kwargs)
        if not (4.0 <= float(d.time) <= 8.402):
            return result

        if cache.get("model_id") != id(m):
            cache.clear()
            cache["model_id"] = id(m)
            cache["fixture"] = object_id(
                m, mujoco.mjtObj.mjOBJ_GEOM, "index_contact_fixture")
            cache["middle"] = {
                object_id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in ("collision_idx_mid_mesh", "collision_idx_mid")
            }
            cache["tip"] = object_id(
                m, mujoco.mjtObj.mjOBJ_GEOM, "collision_idx_tip")
            cache["qadr"] = [
                int(m.jnt_qposadr[object_id(
                    m, mujoco.mjtObj.mjOBJ_JOINT, name)])
                for name in JOINTS
            ]

        mid_force = tip_force = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            fixture_id = cache["fixture"]
            if fixture_id not in (a, b):
                continue
            partner = b if a == fixture_id else a
            if partner not in cache["middle"] and partner != cache["tip"]:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, wrench)
            normal = max(0.0, float(wrench[0]))
            if partner == cache["tip"]:
                tip_force += normal
            else:
                mid_force += normal

        angle_deg = np.rad2deg(d.qpos[cache["qadr"]])
        samples.append((
            float(d.time), mid_force, tip_force,
            float(angle_deg[0]), float(angle_deg[1]),
            float(angle_deg[2]),
        ))
        return result

    mujoco.mj_step = recorder
    try:
        res = v43.run_contact(xml, label)
    finally:
        mujoco.mj_step = original_step

    if len(samples) < 4000:
        raise RuntimeError(f"Incomplete solver trace: {label}, {len(samples)}")

    arr = np.asarray(samples, dtype=float)
    t, fm, ft = arr[:, 0], arr[:, 1], arr[:, 2]
    q = arr[:, 3:6]
    force = fm + ft
    dt = float(np.median(np.diff(t)))
    rmse = np.sqrt(np.mean((q - HW_DEG) ** 2, axis=1))

    hold_mask = (t >= 6.2) & (t <= 8.2)
    if not np.any(hold_mask):
        raise RuntimeError("No samples in hold interval")
    both_loaded = (fm > 0.02) & (ft > 0.02)
    any_loaded = (fm > 0.02) | (ft > 0.02)
    pi = int(np.argmax(force))

    row = dict(
        case=label,
        reported_contact_rmse_deg=float(res["rmse_deg"]),
        min_instant_rmse_deg=float(np.min(rmse)),
        terminal_rmse_deg=float(rmse[-1]),
        terminal_mcp_deg=float(q[-1, 0]),
        terminal_pip_deg=float(q[-1, 1]),
        terminal_dip_deg=float(q[-1, 2]),
        peak_force_N=float(force[pi]),
        peak_time_s=float(t[pi]),
        p95_force_N=float(np.percentile(force, 95)),
        above_2N_ms=float(np.sum(force > 2.0) * dt * 1000),
        above_50N_ms=float(np.sum(force > 50.0) * dt * 1000),
        force_impulse_Ns=float(np.trapz(force, t)),
        both_loaded_fraction_hold=float(np.mean(both_loaded[hold_mask])),
        any_loaded_fraction_hold=float(np.mean(any_loaded[hold_mask])),
    )

    csv_path = OUT / f"{label}_solver_steps.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "middle_N", "tip_N",
                         "mcp_deg", "pip_deg", "dip_deg"))
        writer.writerows(samples)

    for key, value in row.items():
        if key != "case":
            print(f"  {key:<32} {value:10.4f}")
    return row


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Expected original V43 candidate: {SOURCE}")

    print("V43 CONTROLLED COLLISION / FIXTURE EXPERIMENT")
    print("Tests are identical except one isolated change per variant.")
    conditions = [
        ("A_original_mesh", SOURCE),
        ("B_capsule_margin_zero", build_variant("capsule_matched_margin")),
        ("C_mesh_half_stiffness", build_variant("mesh_half_fixture_stiffness")),
    ]
    results = [native_capture(xml, label) for label, xml in conditions]

    outcsv = OUT / "comparison.csv"
    with outcsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 100)
    print("CONTROLLED A/B/C COMPARISON")
    print("=" * 100)
    metrics = (
        "reported_contact_rmse_deg", "terminal_rmse_deg", "peak_force_N",
        "p95_force_N", "above_2N_ms", "above_50N_ms",
        "force_impulse_Ns", "both_loaded_fraction_hold",
        "any_loaded_fraction_hold",
    )
    print(f"{'metric':35s} {'A mesh':>12s} {'B capsule 0mm':>15s} {'C softer mesh':>15s}")
    for field in metrics:
        print(f"{field:35s} " + " ".join(f"{x[field]:15.4f}" for x in results))
    print("\nSummary:", outcsv)
    print("Original XML and hardware data were not modified.")
    print("These are model sensitivity tests, not hardware validation.")


if __name__ == "__main__":
    main()
