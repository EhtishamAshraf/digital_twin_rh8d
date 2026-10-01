import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# STYLE SETTINGS
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
})

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

contact_csv = SCRIPT_DIR / "contact_pose_comparison.csv"
pareto_csv = SCRIPT_DIR / "constant_model_pareto_frontier.csv"

out_png = SCRIPT_DIR / "figure1_combined.png"
out_pdf = SCRIPT_DIR / "figure1_combined.pdf"

# ============================================================
# LOAD DATA
# ============================================================
contact_df = pd.read_csv(contact_csv)
pareto_df = pd.read_csv(pareto_csv)

# ============================================================
# CREATE FIGURE
# Bigger canvas with comfortable white margins
# ============================================================
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(13.5, 6.5)
)

plt.subplots_adjust(
    left=0.09,
    right=0.95,
    bottom=0.15,
    top=0.88,
    wspace=0.25
)

# ============================================================
# (a) LOADED CONTACT POSE
# ============================================================
joint_cols = ["MCP_deg", "PIP_deg", "DIP_deg"]
joint_labels = ["MCP", "PIP", "DIP"]

wanted_rows = [
    "Hardware contact target",
    "Validated constant model",
    "Best free-preserving re-ID",
]

panel_a_df = contact_df[
    contact_df["model"].isin(wanted_rows)
].copy()

panel_a_df["model"] = pd.Categorical(
    panel_a_df["model"],
    categories=wanted_rows,
    ordered=True
)

panel_a_df = panel_a_df.sort_values("model")

legend_names = {
    "Hardware contact target": "Hardware",
    "Validated constant model": "Validated constant model",
    "Best free-preserving re-ID": "Best free-preserving re-ID",
}

colors = {
    "Hardware contact target": "#1f77b4",
    "Validated constant model": "#ff7f0e",
    "Best free-preserving re-ID": "#2ca02c",
}

x = np.arange(len(joint_labels))
width = 0.24

for i, (_, row) in enumerate(panel_a_df.iterrows()):

    offsets = x + (i - 1) * width
    vals = [row[c] for c in joint_cols]
    model_name = row["model"]

    bars = ax1.bar(
        offsets,
        vals,
        width=width,
        color=colors[model_name],
        label=legend_names[model_name]
    )

    # Values above bars
    for bar, val in zip(bars, vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

ax1.set_title("(a) Loaded contact pose", pad=14)
ax1.set_ylabel("Joint angle (deg)")
ax1.set_xticks(x)
ax1.set_xticklabels(joint_labels, fontsize=13)

ax1.set_ylim(0, 25)
ax1.set_yticks(np.arange(0, 26, 5))

ax1.grid(
    axis="y",
    alpha=0.25
)

ax1.legend(
    loc="upper right",
    frameon=True
)

# ============================================================
# (b) CONSTANT-MODEL TRADE-OFF
# ============================================================

# Convert to NumPy arrays to avoid pandas/matplotlib compatibility issue
x_pareto = pareto_df["free_RMSE_deg"].to_numpy()
y_pareto = pareto_df["contact_RMSE_deg"].to_numpy()

ax2.plot(
    x_pareto,
    y_pareto,
    marker="o",
    markersize=5,
    linewidth=1.6,
    color="#1f77b4",
    label="Pareto frontier"
)

# Best free-space-preserving candidate
best_free = 6.001
best_contact = 9.204

ax2.scatter(
    [best_free],
    [best_contact],
    marker="*",
    s=180,
    color="#ff7f0e",
    label="Best free-preserving re-ID",
    zorder=6
)

# ============================================================
# DIAGNOSTIC GATES
# ============================================================
free_gate = 6.085
contact_gate = 6.0

ax2.axvline(
    free_gate,
    linestyle="--",
    linewidth=1.3,
    color="#1e88e5"
)

ax2.axhline(
    contact_gate,
    linestyle="--",
    linewidth=1.3,
    color="#1e88e5"
)

ax2.text(
    free_gate + 0.06,
    12.75,
    "Free-space gate = 6.085°",
    rotation=90,
    va="top",
    ha="left",
    fontsize=10
)

ax2.text(
    4.95,
    contact_gate + 0.08,
    "Contact gate = 6.0°",
    va="bottom",
    ha="left",
    fontsize=10
)

ax2.set_title(
    "(b) Constant-model trade-off",
    pad=14
)

ax2.set_xlabel("Free-space RMSE (deg)")
ax2.set_ylabel("Contact RMSE (deg)")

ax2.set_xlim(4.8, 9.9)
ax2.set_ylim(5.85, 13.0)

ax2.set_yticks(
    np.arange(6, 14, 1)
)

ax2.grid(
    alpha=0.25
)

ax2.legend(
    loc="upper right",
    frameon=True
)

# ============================================================
# SAVE FIGURE
# ============================================================

# PNG - high resolution with white margin
fig.savefig(
    out_png,
    dpi=300,
    facecolor="white",
    pad_inches=0.3
)

# PDF - larger page with white margin around the plots
fig.savefig(
    out_pdf,
    facecolor="white",
    pad_inches=0.3
)

plt.show()

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")