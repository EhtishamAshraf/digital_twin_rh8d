#!/usr/bin/env python3
import os
import csv
from pathlib import Path

import mujoco
import numpy as np

import test_v43_pd_compliant_coupling as v43

WS = Path.home() / (
    "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
)
MJCF = WS / "src/ur5e_mujoco_description/mjcf"
OUT = WS / (
    "rh8d_digital_twin_tools/results/"
    "v43_runtime_contact_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    ("A_LEGACY_RUNTIME", "tmp_v43_k750_r20.xml", False),
    ("B_XML_28000", "tmp_v43_k750_r20.xml", True),
    (
        "C_XML_14000",
        "tmp_v43_control_mesh_half_fixture_stiffness.xml",
        True,
    ),
]

rows = []

for label, filename, preserve_xml in CASES:
    xml = MJCF / filename

    if not xml.exists():
        raise FileNotFoundError(xml)

    os.environ["RH8D_PRESERVE_XML_SOLREF"] = (
        "1" if preserve_xml else "0"
    )

    recorded = []
    parameters = {}
    real_step = mujoco.mj_step

    def recorder(model, data, *args, **kwargs):
        result = real_step(model, data, *args, **kwargs)

        if not (4.0 <= data.time <= 8.402):
            return result

        if not parameters:
            fixture = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "index_contact_fixture",
            )
            parameters["fixture"] = int(fixture)
            parameters["runtime_solref"] = (
                model.geom_solref[fixture].copy()
            )

        fixture = parameters["fixture"]
        total = 0.0
        middle = 0.0
        tip = 0.0
        effective_ref = None

        for j in range(data.ncon):
            c = data.contact[j]
            a, b = int(c.geom1), int(c.geom2)

            if fixture not in (a, b):
                continue

            other = b if a == fixture else a
            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                other,
            )

            if name not in (
                "collision_idx_mid_mesh",
                "collision_idx_mid",
                "collision_idx_tip",
            ):
                continue

            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, j, f)
            normal = max(0.0, float(f[0]))

            if effective_ref is None:
                effective_ref = c.solref.copy()

            total += normal

            if name == "collision_idx_tip":
                tip += normal
            else:
                middle += normal

        recorded.append((
            float(data.time), total, middle, tip,
            effective_ref,
        ))
        return result

    print("\n" + "=" * 90)
    print("CASE:", label)
    print("Preserve XML solref:", preserve_xml)

    mujoco.mj_step = recorder
    try:
        result = v43.run_contact(xml, label)
    finally:
        mujoco.mj_step = real_step

    if len(recorded) < 100 or not parameters:
        raise RuntimeError(
            f"Runtime instrumentation failed for {label}"
        )

    ref = parameters["runtime_solref"]
    expected = (
        -28000 if label == "B_XML_28000" else
        -14000 if label == "C_XML_14000" else None
    )

    print("ACTUAL runtime fixture solref:", ref)

    if expected is not None and not np.isclose(ref[0], expected):
        raise RuntimeError(
            f"XML bypass failed: expected {expected}, "
            f"found {ref[0]}"
        )

    t = np.asarray([r[0] for r in recorded])
    normal = np.asarray([r[1] for r in recorded])
    mid = np.asarray([r[2] for r in recorded])
    tip = np.asarray([r[3] for r in recorded])

    dt = float(np.median(np.diff(t)))

    peak_index = int(np.argmax(normal))
    peak = recorded[peak_index]

    sample_refs = [
        r[4] for r in recorded if r[4] is not None
    ]

    if not sample_refs:
        raise RuntimeError("No fixture contacts were captured")

    actual_contact_ref = sample_refs[0]

    print("ACTUAL contact solver solref:", actual_contact_ref)

    if expected is not None and not np.isclose(
        actual_contact_ref[0], expected
    ):
        raise RuntimeError(
            "Effective contact solref does not match XML. "
            "Stop before interpreting the results."
        )

    summary = {
        "case": label,
        "runtime_k": float(ref[0]),
        "runtime_c": float(ref[1]),
        "effective_contact_k": float(actual_contact_ref[0]),
        "effective_contact_c": float(actual_contact_ref[1]),
        "reported_rmse_deg": float(result["rmse_deg"]),
        "peak_N": float(normal.max()),
        "peak_time_s": float(peak[0]),
        "peak_middle_N": float(peak[2]),
        "peak_tip_N": float(peak[3]),
        "p95_N": float(np.percentile(normal, 95)),
        "above_2N_ms": float(np.sum(normal > 2) * dt * 1000),
        "above_50N_ms": float(np.sum(normal > 50) * dt * 1000),
        "impulse_Ns": float(np.trapz(normal, t)),
    }
    rows.append(summary)

    for k, value in summary.items():
        print(f"  {k}: {value}")

os.environ.pop("RH8D_PRESERVE_XML_SOLREF", None)

output = OUT / "runtime_abc_summary.csv"
with output.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 90)
print("VERIFIED RUNTIME COMPARISON")
print("=" * 90)

for r in rows:
    print(
        f"{r['case']:<20} "
        f"solref=({r['runtime_k']:.0f},"
        f"{r['runtime_c']:.0f}) "
        f"contactRMSE={r['reported_rmse_deg']:.3f} "
        f"peak={r['peak_N']:.3f}N "
        f">50N={r['above_50N_ms']:.1f}ms"
    )

print("\nSummary:", output)
print("Original V43 model XML unchanged.")
