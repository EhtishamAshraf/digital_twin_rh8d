#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


OPEN_TIME = 43.50

BOTTLE_DIAMETER_MM = 69.2

# ============================================================
# Previously validated GOOD bottle measurement.
# ============================================================

EDGE1 = np.array(
    [99.49180653, 141.73001790],
    dtype=float,
)

EDGE2 = np.array(
    [317.00324976, 267.33521751],
    dtype=float,
)

BOTTLE_CENTER = (
    0.5 * (EDGE1 + EDGE2)
)

DIAMETER_PX = float(
    np.linalg.norm(
        EDGE2 - EDGE1
    )
)

MM_PER_PX = (
    BOTTLE_DIAMETER_MM
    / DIAMETER_PX
)


def cart(p):

    # Image -> Cartesian image plane.
    return np.array(
        [
            p[0],
            -p[1],
        ],
        dtype=float,
    )


def unit(v):

    v = np.asarray(
        v,
        dtype=float,
    )

    n = np.linalg.norm(v)

    if n < 1e-9:
        raise RuntimeError(
            "Points are too close."
        )

    return v / n


def angle_deg(v):

    return math.degrees(
        math.atan2(
            v[1],
            v[0],
        )
    )


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python3 measure_open_mechanical_axis_v1.py "
            "/path/to/video.mp4"
        )

        sys.exit(1)


    video = Path(
        sys.argv[1]
    ).expanduser().resolve()


    cap = cv2.VideoCapture(
        str(video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open:\n{video}"
        )


    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        OPEN_TIME * 1000.0,
    )


    ok, frame = cap.read()

    cap.release()


    if not ok:
        raise RuntimeError(
            f"Cannot read frame at {OPEN_TIME:.2f} s"
        )


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )


    print()
    print("=" * 100)
    print("OPEN MECHANICAL PROXIMAL-LINK AXIS")
    print("=" * 100)

    print()
    print(
        "We will measure the mechanical MCP -> PIP axis."
    )

    print()
    print(
        "Click SIX points as THREE repeated measurements:"
    )

    print()
    print(
        "  1. MCP joint centre"
    )

    print(
        "  2. PIP joint centre"
    )

    print()
    print(
        "  3. MCP joint centre again"
    )

    print(
        "  4. PIP joint centre again"
    )

    print()
    print(
        "  5. MCP joint centre a third time"
    )

    print(
        "  6. PIP joint centre a third time"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Click the mechanical hinge centres, NOT the"
    )

    print(
        "ends or centreline of the coloured stickers."
    )

    print()


    fig, ax = plt.subplots(
        figsize=(10, 14)
    )

    ax.imshow(
        rgb
    )

    ax.set_title(
        "OPEN @ 43.50 s\n"
        "Click MCP,PIP × 3"
    )


    points = plt.ginput(
        6,
        timeout=0,
    )

    plt.close(fig)


    if len(points) != 6:
        raise RuntimeError(
            "Exactly six points are required."
        )


    points = [
        np.asarray(
            p,
            dtype=float,
        )
        for p in points
    ]


    measurements = []


    for run in range(3):

        mcp = points[
            2 * run
        ]

        pip = points[
            2 * run + 1
        ]


        mc = cart(
            mcp
        )

        pc = cart(
            pip
        )

        bc = cart(
            BOTTLE_CENTER
        )


        forward = unit(
            pc - mc
        )


        side = np.array(
            [
                -forward[1],
                forward[0],
            ],
            dtype=float,
        )


        bottle_vec = (
            bc - mc
        )


        along_px = float(
            np.dot(
                bottle_vec,
                forward,
            )
        )


        side_px = float(
            np.dot(
                bottle_vec,
                side,
            )
        )


        # Positive side is toward bottle.
        if side_px < 0.0:
            side *= -1.0
            side_px *= -1.0


        along_mm = (
            along_px
            * MM_PER_PX
        )


        side_mm = (
            side_px
            * MM_PER_PX
        )


        center_distance_mm = (
            np.linalg.norm(
                bottle_vec
            )
            * MM_PER_PX
        )


        surface_distance_mm = (
            center_distance_mm
            - BOTTLE_DIAMETER_MM / 2.0
        )


        measurements.append(
            {
                "mcp": mcp,
                "pip": pip,
                "forward": forward,
                "axis_angle":
                    angle_deg(forward),

                "along_mm":
                    along_mm,

                "side_mm":
                    side_mm,

                "center_distance_mm":
                    center_distance_mm,

                "surface_distance_mm":
                    surface_distance_mm,
            }
        )


    # ========================================================
    # Statistics
    # ========================================================

    along = np.array(
        [
            m["along_mm"]
            for m in measurements
        ]
    )

    side = np.array(
        [
            m["side_mm"]
            for m in measurements
        ]
    )

    angle = np.array(
        [
            m["axis_angle"]
            for m in measurements
        ]
    )

    distance = np.array(
        [
            m["center_distance_mm"]
            for m in measurements
        ]
    )

    surface = np.array(
        [
            m["surface_distance_mm"]
            for m in measurements
        ]
    )


    print()
    print("=" * 100)
    print("INDIVIDUAL MEASUREMENTS")
    print("=" * 100)


    for i, m in enumerate(
        measurements,
        start=1,
    ):

        print()
        print(
            f"Run {i}:"
        )

        print(
            f"  MCP = {m['mcp']}"
        )

        print(
            f"  PIP = {m['pip']}"
        )

        print(
            f"  mechanical axis angle = "
            f"{m['axis_angle']:.3f} deg"
        )

        print(
            f"  along mechanical finger = "
            f"{m['along_mm']:+.3f} mm"
        )

        print(
            f"  toward bottle           = "
            f"{m['side_mm']:+.3f} mm"
        )


    print()
    print("=" * 100)
    print("FINAL HARDWARE BOTTLE GEOMETRY — MECHANICAL OPEN FRAME")
    print("=" * 100)

    print(
        f"mechanical MCP->PIP angle = "
        f"{np.mean(angle):+.3f} "
        f"+/- {np.std(angle, ddof=1):.3f} deg"
    )

    print()
    print(
        f"along mechanical finger = "
        f"{np.mean(along):+.3f} "
        f"+/- {np.std(along, ddof=1):.3f} mm"
    )

    print(
        f"toward bottle           = "
        f"{np.mean(side):+.3f} "
        f"+/- {np.std(side, ddof=1):.3f} mm"
    )

    print()
    print(
        f"MCP -> bottle center    = "
        f"{np.mean(distance):.3f} "
        f"+/- {np.std(distance, ddof=1):.3f} mm"
    )

    print(
        f"MCP -> bottle surface   = "
        f"{np.mean(surface):.3f} "
        f"+/- {np.std(surface, ddof=1):.3f} mm"
    )


    # ========================================================
    # Diagnostic figure
    # ========================================================

    mean_mcp = np.mean(
        np.asarray(
            [
                m["mcp"]
                for m in measurements
            ]
        ),
        axis=0,
    )

    mean_pip = np.mean(
        np.asarray(
            [
                m["pip"]
                for m in measurements
            ]
        ),
        axis=0,
    )


    fig, ax = plt.subplots(
        figsize=(10, 14)
    )

    ax.imshow(
        rgb
    )


    # Validated bottle diameter.
    ax.plot(
        [
            EDGE1[0],
            EDGE2[0],
        ],
        [
            EDGE1[1],
            EDGE2[1],
        ],
        "--",
        linewidth=2,
        label="validated 69.2-mm bottle diameter",
    )


    ax.plot(
        BOTTLE_CENTER[0],
        BOTTLE_CENTER[1],
        "o",
        markersize=11,
        label="validated bottle centre",
    )


    # All individual clicks.
    for i, m in enumerate(
        measurements,
        start=1,
    ):

        ax.plot(
            [
                m["mcp"][0],
                m["pip"][0],
            ],
            [
                m["mcp"][1],
                m["pip"][1],
            ],
            "o-",
            linewidth=1.5,
            alpha=0.45,
        )


    # Mean mechanical axis.
    ax.plot(
        [
            mean_mcp[0],
            mean_pip[0],
        ],
        [
            mean_mcp[1],
            mean_pip[1],
        ],
        "o-",
        linewidth=4,
        label="mean MCP→PIP mechanical axis",
    )


    ax.plot(
        [
            mean_mcp[0],
            BOTTLE_CENTER[0],
        ],
        [
            mean_mcp[1],
            BOTTLE_CENTER[1],
        ],
        ":",
        linewidth=2,
        label="MCP→bottle centre",
    )


    ax.text(
        mean_mcp[0] + 4,
        mean_mcp[1],
        "MCP",
    )

    ax.text(
        mean_pip[0] + 4,
        mean_pip[1],
        "PIP",
    )


    ax.set_title(
        "Hardware open mechanical finger axis"
    )

    ax.legend(
        loc="lower right"
    )


    result_dir = (
        Path(__file__).parent
        / "results"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    output = (
        result_dir
        / "hardware_open_mechanical_axis_v1.png"
    )


    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


    print()
    print("Saved:")
    print(output)

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
