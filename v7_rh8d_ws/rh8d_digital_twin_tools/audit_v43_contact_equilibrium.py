#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np
import mujoco
import matplotlib.pyplot as plt

WS = Path.home() / (
    "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
)

XML = WS / (
    "src/ur5e_mujoco_description/mjcf/tmp_v43_k750_r20.xml"
)
TRACE = WS / (
    "rh8d_digital_twin_tools/results/v43_visual_monitor/"
    "v43_k750_r20_exact_contact_trace.npz"
)
OUT = WS / (
    "rh8d_digital_twin_tools/results/"
    "v43_contact_equilibrium_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
z = np.load(TRACE)

t = z["time"]
qpos = z["qpos"]
qvel = z["qvel"]
ctrl = z["ctrl"]

def objid(kind, name):
    i = mujoco.mj_name2id(m, kind, name)
    if i < 0:
        raise RuntimeError(f"Missing: {name}")
    return int(i)

def geom(name):
    return objid(mujoco.mjtObj.mjOBJ_GEOM, name)

def joint(name):
    i = objid(mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(m.jnt_dofadr[i])

fixture = geom("index_contact_fixture")
middle = geom("collision_idx_mid_mesh")
tip = geom("collision_idx_tip")

spool_dof = joint("index_motor_spool")

flexor = objid(
    mujoco.mjtObj.mjOBJ_TENDON,
    "idx_flex_spatial"
)
coupling = objid(
    mujoco.mjtObj.mjOBJ_TENDON,
    "idx_pip_dip_compliant_coupling"
)

act = objid(
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "pos_index_tendon"
)

def distance(g1, g2):
    if not hasattr(mujoco, "mj_geomDistance"):
        return float("nan")

    witness = np.zeros(6)
    try:
        return float(
            mujoco.mj_geomDistance(
                m, d, g1, g2, 0.02, witness
            )
        )
    except (TypeError, AttributeError):
        return float("nan")

def fixture_contacts():
    fm = 0.0
    ft = 0.0
    other = []

    for j in range(d.ncon):
        c = d.contact[j]
        a = int(c.geom1)
        b = int(c.geom2)

        if fixture not in (a, b):
            continue

        partner = b if a == fixture else a
        f = np.zeros(6)

        mujoco.mj_contactForce(m, d, j, f)
        normal = max(0.0, float(f[0]))

        if partner == middle:
            fm += normal
        elif partner == tip:
            ft += normal
        else:
            other.append(
                (
                    mujoco.mj_id2name(
                        m, mujoco.mjtObj.mjOBJ_GEOM,
                        partner
                    ),
                    normal,
                )
            )

    return fm, ft, other


def tendon_passive_force(tid):
    """
    Signed elastic + viscous tendon force in the
    tendon-length coordinate.

    Uses MuJoCo's compiled parameters.

    Does not include constraint-generated friction or
    tendon limit forces.
    """
    length = float(d.ten_length[tid])

    rest = np.asarray(
        m.tendon_lengthspring[tid],
        dtype=float
    ).reshape(-1)

    if rest.size == 1:
        low = high = float(rest[0])
    else:
        low = float(rest[0])
        high = float(rest[1])

    # Respect MuJoCo's tendon spring deadband.
    if length < low:
        extension = length - low
    elif length > high:
        extension = length - high
    else:
        extension = 0.0

    coefficients = np.asarray(
        m.tendon_stiffness[tid],
        dtype=float
    ).reshape(-1)

    # The V43 tendons use linear spring stiffness.
    k = float(coefficients[0])

    damping = float(m.tendon_damping[tid])
    velocity = float(d.ten_velocity[tid])

    return -k * extension - damping * velocity


# 50-ms sampling, plus exact critical times.
sample_times = list(np.arange(t[0], t[-1], 0.05))
critical = [
    4.373, 5.382, 5.708, 6.191,
    6.589, 6.970, 7.238, 7.296,
    7.372, 7.458, 7.992, 8.201,
    8.401
]
sample_times += critical

indices = sorted(set(
    int(np.argmin(np.abs(t - x)))
    for x in sample_times
))

rows = []

for i in indices:
    d.qpos[:] = qpos[i]
    d.qvel[:] = qvel[i]
    d.ctrl[:] = ctrl[i]
    d.time = float(t[i])

    mujoco.mj_forward(m, d)

    fmid, ftip, others = fixture_contacts()

    # This is the recorded servo command minus the
    # recorded spool position, not a modification.
    spool_qadr = int(m.jnt_qposadr[
        objid(
            mujoco.mjtObj.mjOBJ_JOINT,
            "index_motor_spool"
        )
    ])

    row = {
        "time_s": float(t[i]),
        "spool_mm": float(qpos[i, spool_qadr]*1000),
        "spool_ctrl_mm": float(ctrl[i, act]*1000),
        "servo_error_um": float(
            (ctrl[i, act]-qpos[i, spool_qadr])*1e6
        ),
        "middle_distance_mm": 1000*distance(
            fixture, middle
        ),
        "tip_distance_mm": 1000*distance(
            fixture, tip
        ),
        "middle_normal_N": fmid,
        "tip_normal_N": ftip,
        "flexor_length_mm": float(
            d.ten_length[flexor]*1000
        ),
        "flexor_force_N": float(
            tendon_passive_force(flexor)
        ),
        "coupling_force_N": float(
            tendon_passive_force(coupling)
        ),
        "spool_actuator_force_N": float(
            d.qfrc_actuator[spool_dof]
        ),
        "other_fixture_contacts": str(others),
    }
    rows.append(row)

path = OUT / "v43_force_geometry.csv"
with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print("\nV43 FORCE AND GEOMETRY AUDIT")
print("=" * 122)
print(
    " time   mid_gap  tip_gap   Fmid    Ftip   "
    "flexorF  coupleF   spool_err  servoF"
)
print(
    "   s       mm       mm       N       N       "
    "N        N         um         N"
)
print("-" * 122)

for target in critical:
    r = min(rows, key=lambda x: abs(x["time_s"]-target))
    print(
        f"{r['time_s']:6.3f} "
        f"{r['middle_distance_mm']:+8.3f} "
        f"{r['tip_distance_mm']:+8.3f} "
        f"{r['middle_normal_N']:7.3f} "
        f"{r['tip_normal_N']:7.3f} "
        f"{r['flexor_force_N']:+8.3f} "
        f"{r['coupling_force_N']:+8.3f} "
        f"{r['servo_error_um']:+10.2f} "
        f"{r['spool_actuator_force_N']:+8.3f}"
    )

print("\nNON-MIDDLE/TIP CYLINDER CONTACTS")
for target in [6.191, 7.238, 7.296, 8.201]:
    r = min(rows, key=lambda x: abs(x["time_s"]-target))
    print(
        f"t={r['time_s']:.3f}: "
        f"{r['other_fixture_contacts']}"
    )

x = np.asarray([r["time_s"] for r in rows])

def col(name):
    return np.asarray([r[name] for r in rows])

fig, axs = plt.subplots(
    3, 1, figsize=(12, 10), sharex=True,
    constrained_layout=True
)

axs[0].plot(x, col("middle_distance_mm"), label="Middle gap")
axs[0].plot(x, col("tip_distance_mm"), label="Tip gap")
axs[0].axhline(0, ls="--", linewidth=1)
axs[0].set_ylabel("Signed gap (mm)")
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[1].plot(x, col("middle_normal_N"), label="Middle normal")
axs[1].plot(x, col("tip_normal_N"), label="Tip normal")
axs[1].set_ylabel("Contact force (N)")
axs[1].legend()
axs[1].grid(alpha=0.3)

axs[2].plot(x, col("flexor_force_N"), label="Flexor force")
axs[2].plot(x, col("coupling_force_N"), label="Coupling force")
axs[2].set_ylabel("Tendon force (N)")
axs[2].set_xlabel("Simulation time (s)")
axs[2].legend()
axs[2].grid(alpha=0.3)

for ax in axs:
    ax.set_xlim(4.0, 8.4)
    ax.axvline(7.238, ls=":", alpha=0.5)

png = OUT / "v43_force_geometry.png"
fig.savefig(png, dpi=160)
plt.close(fig)

print("\nSaved:")
print(path)
print(png)
print("\nNo dynamics or model parameters were changed.")
