#!/usr/bin/env python3

import csv
from pathlib import Path


WS = Path(__file__).resolve().parents[1]

CSV_PATH = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "transmission_scale_stiffness_v1/summary.csv"
)


TRANSMISSIONS = [
    (5.0, 4.0, 1.0),
    (6.0, 5.0, 1.0),
    (5.0, 5.0, 1.0),
    (5.0, 6.0, 1.0),
]


FREE_LIMIT = 6.085

MIN_CONTACT_PIP = 18.0
MAX_CONTACT_RMSE = 6.0
MIN_MIDDLE_DUTY = 70.0


def convert(value):

    if value in (
        "",
        "None",
        None,
    ):
        return None

    return float(value)


rows = []

with open(CSV_PATH) as f:

    for r in csv.DictReader(f):

        rows.append(
            {
                key: convert(value)
                for key, value in r.items()
            }
        )


print()
print(
    f"Loaded {len(rows)} completed candidates."
)


# ============================================================
# Control check
# ============================================================

control = next(
    r
    for r in rows
    if (
        r["mcp_mm"] == 5.0
        and r["pip_mm"] == 4.0
        and r["dip_mm"] == 1.0
        and abs(
            r["scale_um"] - 5.10
        ) < 1e-9
        and abs(
            r["tendon_k"] - 2100.0
        ) < 1e-9
    )
)


print()
print("=" * 145)
print("CONTROL CHECK")
print("=" * 145)

print()
print(
    "[MCP,PIP,DIP] = [5,4,1] mm"
)

print(
    "PIP friction = 0.003"
)

print(
    "scale = 5.10 um/tick"
)

print(
    "tendon stiffness = 2100 N/m"
)

print(
    f"free RMSE = "
    f"{control['free_combined']:.3f} deg"
)

print(
    "contact pose = "
    f"({control['contact_mcp']:.3f}, "
    f"{control['contact_pip']:.3f}, "
    f"{control['contact_dip']:.3f}) deg"
)

print(
    f"contact RMSE = "
    f"{control['contact_rmse']:.3f} deg"
)


# ============================================================
# Best free-preserving candidate per transmission
# ============================================================

print()
print("=" * 145)
print(
    "BEST FREE-SPACE-PRESERVING RESULT "
    "FOR EACH TRANSMISSION"
)
print(
    f"(free RMSE <= {FREE_LIMIT:.3f} deg)"
)
print("=" * 145)


free_preserving_best = []


for vector in TRANSMISSIONS:

    mcp, pip, dip = vector


    group = [
        r
        for r in rows
        if (
            r["mcp_mm"] == mcp
            and r["pip_mm"] == pip
            and r["dip_mm"] == dip
            and r["free_combined"]
            <= FREE_LIMIT
        )
    ]


    group.sort(
        key=lambda r:
            r["contact_rmse"]
    )


    print()
    print(
        f"Transmission "
        f"[{mcp:.1f},"
        f"{pip:.1f},"
        f"{dip:.1f}] mm"
    )


    if not group:

        print(
            "  NO candidate preserved "
            "the free-space gate."
        )

        continue


    best = group[0]

    free_preserving_best.append(
        best
    )


    print(
        f"  scale = "
        f"{best['scale_um']:.2f} um/tick"
    )

    print(
        f"  tendon k = "
        f"{best['tendon_k']:.0f} N/m"
    )

    print(
        f"  free MCP/PIP/combined = "
        f"{best['free_mcp']:.3f} / "
        f"{best['free_pip']:.3f} / "
        f"{best['free_combined']:.3f} deg"
    )

    print(
        "  contact pose = "
        f"({best['contact_mcp']:.3f}, "
        f"{best['contact_pip']:.3f}, "
        f"{best['contact_dip']:.3f}) deg"
    )

    print(
        f"  contact RMSE = "
        f"{best['contact_rmse']:.3f} deg"
    )

    print(
        f"  middle duty = "
        f"{best['middle_duty']} %"
    )


# ============================================================
# Pareto frontier
# ============================================================

def dominates(
    a,
    b,
):

    return (
        a["free_combined"]
        <= b["free_combined"]
        and
        a["contact_rmse"]
        <= b["contact_rmse"]
        and
        (
            a["free_combined"]
            < b["free_combined"]
            or
            a["contact_rmse"]
            < b["contact_rmse"]
        )
    )


pareto = []

for r in rows:

    dominated = any(
        dominates(
            other,
            r,
        )
        for other in rows
        if other is not r
    )

    if not dominated:
        pareto.append(r)


pareto.sort(
    key=lambda r:
        r["free_combined"]
)


print()
print("=" * 145)
print("GLOBAL PARETO FRONTIER")
print("=" * 145)


for r in pareto:

    print(
        f"[{r['mcp_mm']:.1f},"
        f"{r['pip_mm']:.1f},"
        f"{r['dip_mm']:.1f}] "
        f"s={r['scale_um']:.2f} "
        f"k={r['tendon_k']:.0f}   "
        f"free={r['free_combined']:.3f}   "
        f"contact={r['contact_rmse']:.3f}   "
        f"pose=("
        f"{r['contact_mcp']:.2f},"
        f"{r['contact_pip']:.2f},"
        f"{r['contact_dip']:.2f})"
    )


# ============================================================
# Strict gate
# ============================================================

def sequence_valid(r):

    return (
        r["dmcp"] is not None
        and
        r["dpip"] is not None
        and
        r["ddip"] is not None
        and
        r["middle_duty"] is not None
        and
        r["middle_duty"]
        >= MIN_MIDDLE_DUTY
    )


qualifying = [
    r
    for r in rows
    if (
        r["free_combined"]
        <= FREE_LIMIT
        and
        r["contact_pip"]
        >= MIN_CONTACT_PIP
        and
        r["contact_rmse"]
        <= MAX_CONTACT_RMSE
        and
        sequence_valid(r)
    )
]


qualifying.sort(
    key=lambda r:
        (
            r["contact_rmse"],
            r["free_combined"],
        )
)


print()
print("=" * 145)
print("STRICT CONSTANT-MODEL GATE")
print("=" * 145)


if qualifying:

    print()
    print(
        f"PASSING CANDIDATES = "
        f"{len(qualifying)}"
    )


    for rank, r in enumerate(
        qualifying[:10],
        start=1,
    ):

        print()
        print(
            f"#{rank}"
        )

        print(
            "  transmission = "
            f"[{r['mcp_mm']:.1f}, "
            f"{r['pip_mm']:.1f}, "
            f"{r['dip_mm']:.1f}] mm"
        )

        print(
            f"  scale = "
            f"{r['scale_um']:.2f} um/tick"
        )

        print(
            f"  tendon k = "
            f"{r['tendon_k']:.0f} N/m"
        )

        print(
            f"  free = "
            f"{r['free_combined']:.3f} deg"
        )

        print(
            "  contact pose = "
            f"({r['contact_mcp']:.3f}, "
            f"{r['contact_pip']:.3f}, "
            f"{r['contact_dip']:.3f}) deg"
        )

        print(
            f"  contact RMSE = "
            f"{r['contact_rmse']:.3f} deg"
        )


else:

    print()
    print("NO candidate passed all gates.")

    print(
        f"  free <= "
        f"{FREE_LIMIT:.3f} deg"
    )

    print(
        f"  contact PIP >= "
        f"{MIN_CONTACT_PIP:.1f} deg"
    )

    print(
        f"  contact RMSE <= "
        f"{MAX_CONTACT_RMSE:.1f} deg"
    )

    print(
        f"  middle duty >= "
        f"{MIN_MIDDLE_DUTY:.0f}%"
    )


# ============================================================
# Best contact among free-preserving candidates
# ============================================================

all_free_preserving = [
    r
    for r in rows
    if (
        r["free_combined"]
        <= FREE_LIMIT
    )
]


best_free_contact = min(
    all_free_preserving,
    key=lambda r:
        r["contact_rmse"],
)


print()
print("=" * 145)
print(
    "BEST CONTACT WHILE PRESERVING FREE-SPACE GATE"
)
print("=" * 145)

print()
print(
    "transmission = "
    f"[{best_free_contact['mcp_mm']:.1f}, "
    f"{best_free_contact['pip_mm']:.1f}, "
    f"{best_free_contact['dip_mm']:.1f}] mm"
)

print(
    f"scale = "
    f"{best_free_contact['scale_um']:.2f} um/tick"
)

print(
    f"tendon k = "
    f"{best_free_contact['tendon_k']:.0f} N/m"
)

print(
    f"free = "
    f"{best_free_contact['free_combined']:.3f} deg"
)

print(
    "contact pose = "
    f"({best_free_contact['contact_mcp']:.3f}, "
    f"{best_free_contact['contact_pip']:.3f}, "
    f"{best_free_contact['contact_dip']:.3f}) deg"
)

print(
    f"contact RMSE = "
    f"{best_free_contact['contact_rmse']:.3f} deg"
)


# ============================================================
# Interpretation
# ============================================================

print()
print("=" * 145)
print("INTERPRETATION")
print("=" * 145)


if qualifying:

    print()
    print(
        "A locally re-identified CONSTANT model passed "
        "both the free-space and contact diagnostic gates."
    )

    print(
        "Do NOT introduce configuration-dependent "
        "transmission yet."
    )


else:

    print()
    print(
        "No locally re-identified constant model "
        "satisfied both experimental regimes."
    )

    print()
    print(
        "The constant-model hierarchy has now tested:"
    )

    print(
        "  passive PIP mechanics"
    )

    print(
        "  constant PIP coefficient"
    )

    print(
        "  coupled constant MCP/PIP/DIP coefficients"
    )

    print(
        "  local scale + tendon-stiffness "
        "re-identification"
    )

    print()
    print(
        "If the Pareto frontier still shows the same "
        "systematic free/contact trade-off, then a "
        "configuration-dependent effective transmission "
        "becomes the next justified structural hypothesis."
    )


print()
print(
    "Source CSV:"
)

print(
    CSV_PATH
)

print()
print("=" * 145)
