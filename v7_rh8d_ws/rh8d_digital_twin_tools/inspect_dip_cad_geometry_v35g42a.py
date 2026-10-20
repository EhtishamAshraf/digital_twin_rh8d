#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import audit_index_tendon_transmission_v35g12 as g12
import build_persistent_flexor_wrap_v35g38a as g38a
from rh8d_tabletop_stage_v1 import apply_tabletop_stage

src = (Path.home() / "rh8d_results"
       / "v35g39a_cad_multiguide_flexor"
       / "g39a_cad_multiguide_flexor.xml")
out = Path.home() / "rh8d_results/v35g42a_dip_geometry"
out.mkdir(parents=True, exist_ok=True)

m = mujoco.MjModel.from_xml_path(str(src))
d = mujoco.MjData(m)
refs = g12.get_joint_refs(m)

names = [
    "v35_p05_pip_child",
    "v35_p06_dip_parent",
    "v35_p07_dip_child",
    "v35_p08_distal_turn",
    "v35_p09_anchor",
]

sid = {}
for name in names:
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    if i < 0:
        raise RuntimeError(f"Missing CAD site: {name}")
    sid[name] = i

jid = mujoco.mj_name2id(
    m, mujoco.mjtObj.mjOBJ_JOINT,
    "Index_Distal--Index_Middle"
)
tid = m.tendon("idx_flex_spatial").id
gid = m.geom("g39a_dip_cad_guide").id
radius_mm = 1000.0 * m.geom_size[gid, 0]

def pose(angle):
    apply_tabletop_stage(m, d, reset_to_qpos0=True)
    for key in ("mcp", "pip", "dip"):
        d.qpos[refs[key]["qadr"]] = 0.0
    d.qpos[refs["dip"]["qadr"]] = math.radians(angle)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

pose(0)

center = np.array(d.xanchor[jid])
axis = np.array(d.xaxis[jid])
axis /= np.linalg.norm(axis)

radial = np.array(d.site_xpos[sid[names[1]]]) - center
radial -= np.dot(radial, axis) * axis
radial /= np.linalg.norm(radial)

side = np.cross(axis, radial)
side /= np.linalg.norm(side)

print("=" * 85)
print("G42A — CAD DISTAL TENDON ROUTING")
print("=" * 85)
print(f"DIP guide radius: {radius_mm:.4f} mm")
print("Coordinates are relative to the DIP hinge.")
print("Projection shows CAD waypoint positions, not solved wrap-contact points.")

output = []
plt.figure(figsize=(9, 7))

for angle in (0, 35, 45, 80):
    pose(angle)

    J = g38a.tendon_jacobian_row(m, d, tid)
    dip_j = 1000 * float(J[refs["dip"]["dadr"]])

    print(f"\nDIP = {angle} deg; dL/dq = {dip_j:+.4f} mm/rad")
    print("CAD site                      radial-x    radial-y    axial-z [mm]")

    xy = []
    for name in names:
        v = np.array(d.site_xpos[sid[name]]) - center
        x = 1000 * np.dot(v, radial)
        y = 1000 * np.dot(v, side)
        z = 1000 * np.dot(v, axis)
        xy.append((x, y))

        print(f"{name:29s} {x:+9.3f} {y:+11.3f} {z:+11.3f}")

        output.append({
            "dip_deg": angle,
            "site": name,
            "x_mm": x,
            "y_mm": y,
            "axial_mm": z,
            "dip_j_mm_per_rad": dip_j,
        })

    pts = np.array(xy)
    plt.plot(
        pts[:, 0], pts[:, 1], "-o",
        label=f"DIP {angle}° | J={dip_j:+.2f} mm/rad"
    )

    if angle == 0:
        for name, (x, y) in zip(names, xy):
            plt.annotate(
                name.replace("v35_", ""),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

# Existing G39A DIP guide at the p06 CAD waypoint.
guide = output[1]
circle = plt.Circle(
    (guide["x_mm"], guide["y_mm"]),
    radius_mm,
    fill=False,
    linestyle="--",
)
plt.gca().add_patch(circle)

plt.scatter([0], [0], marker="x", s=100, label="DIP hinge")
plt.axis("equal")
plt.grid(True)
plt.xlabel("Radial coordinate X (mm)")
plt.ylabel("Radial coordinate Y (mm)")
plt.title("RH8D distal CAD waypoints — DIP flexion")
plt.legend(fontsize=8)
plt.tight_layout()

image = out / "g42a_dip_cad_projection.png"
plt.savefig(image, dpi=180)
plt.close()

with (out / "g42a_dip_coordinates.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(output[0]))
    w.writeheader()
    w.writerows(output)

print(f"\nPlot: {image}")
print(f"CSV:  {out / 'g42a_dip_coordinates.csv'}")
print("No dynamics or parameter modifications.")
