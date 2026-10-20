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
    "src/ur5e_mujoco_description/mjcf/"
    "tmp_v43_k750_r20.xml"
)

TRACE = WS / (
    "rh8d_digital_twin_tools/results/v43_visual_monitor/"
    "v43_k750_r20_exact_contact_trace.npz"
)

OUT = WS / (
    "rh8d_digital_twin_tools/results/"
    "v43_motor_contact_diagnosis"
)
OUT.mkdir(parents=True, exist_ok=True)

m = mujoco.MjModel.from_xml_path(str(XML))
z = np.load(TRACE)

t = np.asarray(z["time"], dtype=float)
qpos = np.asarray(z["qpos"], dtype=float)
mid = np.asarray(z["middle_contact"], dtype=bool)
tip = np.asarray(z["tip_contact"], dtype=bool)

def addr(name):
    jid = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    if jid < 0:
        raise RuntimeError(f"Missing joint: {name}")
    return int(m.jnt_qposadr[jid])

spool = qpos[:, addr("index_motor_spool")] * 1000.0

angles = np.rad2deg(np.column_stack([
    qpos[:, addr("Index_Proximal--palm_assembly:1")],
    qpos[:, addr("Index_Middle--Index_Proximal")],
    qpos[:, addr("Index_Distal--Index_Middle")],
]))

# Smoothing is used only for classifying motor movement;
# raw recorded values remain unchanged.
dt = float(np.median(np.diff(t)))
window = max(3, int(round(0.10 / dt)))
kernel = np.ones(window) / window
smooth = np.convolve(
    np.pad(spool, (window // 2, window - 1 - window // 2),
           mode="edge"),
    kernel,
    mode="valid",
)

velocity = np.gradient(smooth, t)

# Determine whether increasing or decreasing spool displacement
# corresponds to flexion, rather than assuming the sign.
angle_sum = angles.sum(axis=1)
ds = np.diff(spool)
da = np.diff(angle_sum)
active = np.abs(ds) > 1e-6

if np.any(active):
    sign_score = float(np.sum(ds[active] * da[active]))
    closing_sign = 1 if sign_score >= 0 else -1
else:
    closing_sign = 1
    print("WARNING: spool did not move sufficiently to infer direction")

closing_rate = closing_sign * velocity
rate_threshold = 0.02  # mm/s

phase = np.where(
    closing_rate > rate_threshold, "CLOSING",
    np.where(closing_rate < -rate_threshold, "OPENING", "HOLD")
)

hardware = np.array([14.5355, 23.8110, 9.3270])
pose_rmse = np.sqrt(np.mean((angles - hardware)**2, axis=1))

events = [0]
for i in range(1, len(t)):
    if mid[i] != mid[i-1] or tip[i] != tip[i-1]:
        events.append(i)

def state_at(i):
    return (
        f"{t[i]:7.3f}  "
        f"{spool[i]:9.4f}  "
        f"{phase[i]:8s}  "
        f"{angles[i,0]:7.2f}  "
        f"{angles[i,1]:7.2f}  "
        f"{angles[i,2]:7.2f}  "
        f"{int(mid[i])}  {int(tip[i])}  "
        f"{pose_rmse[i]:8.3f}"
    )

print("\nV43 MEASURED-SPOOL AND CONTACT AUDIT")
print("=" * 105)
print("Time     Spool(mm)   Phase     MCP      PIP      DIP     M  T    RMSE")
print("-" * 105)

for i in events:
    print(state_at(i))

print("\nIMPORTANT SAMPLES")
print("-" * 105)

best_i = int(np.argmin(pose_rmse))
max_i = int(np.argmax(spool))
min_i = int(np.argmin(spool))

for label, i in [
    ("Minimum target-pose RMSE", best_i),
    ("Maximum spool position", max_i),
    ("Minimum spool position", min_i),
    ("Final recorded sample", len(t) - 1),
]:
    print(f"{label}:")
    print(" ", state_at(i))

print("\nMOTOR SIGN INFERENCE")
print(
    "Increasing spool =",
    "CLOSING" if closing_sign == 1 else "OPENING"
)
print("Total duration:", round(float(t[-1]-t[0]), 3), "s")
print("Spool range:", round(float(np.ptp(spool)), 4), "mm")

# Per-contact event, print the spool movement over next 0.4 s.
print("\nCONTACT TRANSITIONS AND SUBSEQUENT MOTOR MOVEMENT")
print("-" * 105)

for i in events[1:]:
    j = int(np.searchsorted(t, t[i] + 0.4))
    j = min(j, len(t) - 1)

    delta = spool[j] - spool[i]
    close_delta = closing_sign * delta

    movement = (
        "CONTINUES CLOSING" if close_delta > 0.02
        else "MOTOR OPENS" if close_delta < -0.02
        else "NEARLY HOLDS"
    )

    print(
        f"t={t[i]:.3f}  "
        f"M={int(mid[i])} T={int(tip[i])}  "
        f"next 0.4s spool delta={delta:+.4f} mm  "
        f"{movement}"
    )

with (OUT / "v43_trace.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time_s", "spool_mm", "phase",
        "mcp_deg", "pip_deg", "dip_deg",
        "middle_contact", "tip_contact",
        "target_pose_rmse_deg"
    ])

    for i in range(len(t)):
        writer.writerow([
            t[i], spool[i], phase[i],
            *angles[i],
            int(mid[i]), int(tip[i]),
            pose_rmse[i],
        ])

fig, axs = plt.subplots(
    3, 1, figsize=(13, 10), sharex=True,
    constrained_layout=True
)

axs[0].plot(t, spool)
axs[0].set_ylabel("Spool displacement (mm)")
axs[0].set_title(
    "V43: Recorded spool motion, finger joints and contact"
)
axs[0].grid(alpha=0.3)

for k, name in enumerate(["MCP", "PIP", "DIP"]):
    axs[1].plot(t, angles[:, k], label=name)

axs[1].axhline(hardware[0], ls=":", alpha=.35)
axs[1].axhline(hardware[1], ls=":", alpha=.35)
axs[1].axhline(hardware[2], ls=":", alpha=.35)
axs[1].set_ylabel("Joint angle (deg)")
axs[1].legend()
axs[1].grid(alpha=0.3)

axs[2].step(
    t, mid.astype(int), where="post", label="Middle contact"
)
axs[2].step(
    t, tip.astype(int) + 2, where="post", label="Tip contact"
)
axs[2].set_yticks([0, 1, 2, 3])
axs[2].set_yticklabels([
    "M off", "M on", "T off", "T on"
])
axs[2].set_xlabel("Replay time (s)")
axs[2].legend()
axs[2].grid(alpha=0.3)

for ax in axs:
    ax.axvline(
        t[best_i], color="black", ls="--",
        alpha=0.5
    )

png = OUT / "v43_spool_joints_contact.png"
fig.savefig(png, dpi=160)
plt.close(fig)

print("\nSaved:")
print(png)
print(OUT / "v43_trace.csv")
print("\nNo model parameters were changed.")
