#!/usr/bin/env python3

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


VIDEO_TIME = 50.50

# Real bottle outside diameter measured physically.
BOTTLE_DIAMETER_MM = 69.2

# Manual-center check.
CENTER_WARNING_PX = 8.0
CENTER_REJECT_PX = 15.0


def unit(v):

    v = np.asarray(
        v,
        dtype=float,
    )

    n = np.linalg.norm(v)

    if n < 1e-9:

        raise RuntimeError(
            "Selected points are too close."
        )

    return v / n


def cart(point):

    # Image:
    #   +x right
    #   +y down
    #
    # Cartesian image plane:
    #   +x right
    #   +y up

    return np.array(
        [
            point[0],
            -point[1],
        ],
        dtype=float,
    )


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python3 measure_hardware_bottle_geometry_v2.py "
            "/path/to/contact_video.mp4"
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
            f"Cannot open video:\n{video}"
        )


    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        VIDEO_TIME * 1000.0,
    )


    ok, frame = cap.read()

    cap.release()


    if not ok:

        raise RuntimeError(
            f"Cannot read video frame at {VIDEO_TIME:.2f} s"
        )


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )


    print()
    print("=" * 100)
    print("HARDWARE BOTTLE GEOMETRY MEASUREMENT V2")
    print("=" * 100)

    print()
    print("Click SIX points:")
    print()

    print(
        "  1. First OUTER edge of the cylindrical bottle body"
    )

    print(
        "  2. Opposite OUTER edge of the same bottle body"
    )

    print()
    print(
        "     IMPORTANT:"
    )

    print(
        "     Points 1 and 2 must define one diameter of the"
    )

    print(
        "     69.2-mm bottle body."
    )

    print(
        "     Do NOT use the handle, cap, highlight, or neck."
    )

    print()
    print(
        "  3. Your visual estimate of the bottle centre"
    )

    print(
        "     This is ONLY a consistency check."
    )

    print(
        "     The calculation will use the midpoint of points 1 and 2."
    )

    print()
    print(
        "  4. Index MCP joint centre"
    )

    print()
    print(
        "  5. Pink marker centreline — proximal point"
    )

    print(
        "  6. Pink marker centreline — distal point"
    )

    print()


    fig, ax = plt.subplots(
        figsize=(10, 14)
    )

    ax.imshow(
        rgb
    )

    ax.set_title(
        "1-2 bottle diameter | "
        "3 bottle centre check | "
        "4 MCP | "
        "5-6 pink proximal→distal"
    )


    pts = plt.ginput(
        6,
        timeout=0,
    )

    plt.close(fig)


    if len(pts) != 6:

        raise RuntimeError(
            "Exactly six points are required."
        )


    edge1 = np.asarray(
        pts[0],
        dtype=float,
    )

    edge2 = np.asarray(
        pts[1],
        dtype=float,
    )

    manual_center = np.asarray(
        pts[2],
        dtype=float,
    )

    mcp = np.asarray(
        pts[3],
        dtype=float,
    )

    pink_prox = np.asarray(
        pts[4],
        dtype=float,
    )

    pink_dist = np.asarray(
        pts[5],
        dtype=float,
    )


    # ============================================================
    # Bottle geometry
    # ============================================================

    diameter_vector = (
        edge2 - edge1
    )


    diameter_px = float(
        np.linalg.norm(
            diameter_vector
        )
    )


    if diameter_px < 20.0:

        raise RuntimeError(
            "Bottle diameter is implausibly small. "
            "Please redo the clicks."
        )


    computed_center = (
        0.5
        * (
            edge1
            + edge2
        )
    )


    center_mismatch_px = float(
        np.linalg.norm(
            manual_center
            - computed_center
        )
    )


    mm_per_px = (
        BOTTLE_DIAMETER_MM
        / diameter_px
    )


    center_mismatch_mm = (
        center_mismatch_px
        * mm_per_px
    )


    # ============================================================
    # Extra center consistency diagnostics
    # ============================================================

    diameter_u = unit(
        diameter_vector
    )


    center_from_edge1 = (
        manual_center
        - edge1
    )


    projection_px = float(
        np.dot(
            center_from_edge1,
            diameter_u,
        )
    )


    projection_fraction = (
        projection_px
        / diameter_px
    )


    perpendicular_vector = (
        center_from_edge1
        -
        projection_px
        * diameter_u
    )


    center_off_diameter_px = float(
        np.linalg.norm(
            perpendicular_vector
        )
    )


    # ============================================================
    # Finger-fixed coordinate system
    # ============================================================

    bc = cart(
        computed_center
    )

    mc = cart(
        mcp
    )

    pp = cart(
        pink_prox
    )

    pd = cart(
        pink_dist
    )


    forward = unit(
        pd - pp
    )


    side = np.array(
        [
            -forward[1],
            forward[0],
        ],
        dtype=float,
    )


    bottle_vector_px = (
        bc - mc
    )


    forward_px = float(
        np.dot(
            bottle_vector_px,
            forward,
        )
    )


    side_px = float(
        np.dot(
            bottle_vector_px,
            side,
        )
    )


    # Make positive side point toward the bottle.
    if side_px < 0.0:

        side *= -1.0

        side_px = -side_px


    forward_mm = (
        forward_px
        * mm_per_px
    )


    side_mm = (
        side_px
        * mm_per_px
    )


    total_distance_mm = float(
        np.linalg.norm(
            bottle_vector_px
        )
        * mm_per_px
    )


    # Distance from MCP to nearest bottle surface
    # in the image plane.
    surface_distance_mm = (
        total_distance_mm
        -
        BOTTLE_DIAMETER_MM / 2.0
    )


    # ============================================================
    # Quality classification
    # ============================================================

    if (
        center_mismatch_px
        <= CENTER_WARNING_PX
        and
        abs(
            projection_fraction
            - 0.5
        )
        <= 0.05
    ):

        quality = "GOOD"

    elif (
        center_mismatch_px
        <= CENTER_REJECT_PX
        and
        abs(
            projection_fraction
            - 0.5
        )
        <= 0.10
    ):

        quality = "USABLE BUT REPEAT ONCE"

    else:

        quality = "REJECT — REDO BOTTLE CLICKS"


    # ============================================================
    # Print
    # ============================================================

    print()
    print("=" * 100)
    print("CLICKED POINTS")
    print("=" * 100)

    print(
        f"edge 1                = {edge1}"
    )

    print(
        f"edge 2                = {edge2}"
    )

    print(
        f"manual bottle centre  = {manual_center}"
    )

    print(
        f"computed midpoint      = {computed_center}"
    )

    print(
        f"MCP                   = {mcp}"
    )

    print(
        f"pink proximal         = {pink_prox}"
    )

    print(
        f"pink distal           = {pink_dist}"
    )


    print()
    print("=" * 100)
    print("BOTTLE-DIAMETER CONSISTENCY CHECK")
    print("=" * 100)

    print(
        f"Diameter in image                 = "
        f"{diameter_px:.3f} px"
    )

    print(
        f"Known real diameter               = "
        f"{BOTTLE_DIAMETER_MM:.3f} mm"
    )

    print(
        f"Scale                             = "
        f"{mm_per_px:.5f} mm/px"
    )

    print()
    print(
        f"Manual centre -> midpoint error   = "
        f"{center_mismatch_px:.3f} px"
    )

    print(
        f"                                  = "
        f"{center_mismatch_mm:.3f} mm"
    )

    print()
    print(
        f"Manual centre position along "
        f"diameter = "
        f"{projection_fraction:.4f}"
    )

    print(
        "Ideal value                        = 0.5000"
    )

    print(
        f"Manual centre off diameter line    = "
        f"{center_off_diameter_px:.3f} px"
    )

    print()
    print(
        f"MEASUREMENT QUALITY: {quality}"
    )


    print()
    print("=" * 100)
    print("BOTTLE CENTRE RELATIVE TO MCP")
    print("=" * 100)

    print(
        "NOTE: These values use the COMPUTED MIDPOINT "
        "of edge points 1 and 2."
    )

    print()

    print(
        f"Along proximal-finger axis         = "
        f"{forward_mm:+.3f} mm"
    )

    print(
        f"Perpendicular toward bottle        = "
        f"{side_mm:+.3f} mm"
    )

    print(
        f"MCP -> bottle-centre distance      = "
        f"{total_distance_mm:.3f} mm"
    )

    print(
        f"MCP -> nearest bottle surface (*)  = "
        f"{surface_distance_mm:.3f} mm"
    )

    print()
    print(
        "(*) Simple image-plane radial distance. "
        "Useful as a geometry sanity check."
    )


    # ============================================================
    # Diagnostic plot
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(10, 14)
    )


    ax.imshow(
        rgb
    )


    # Diameter
    ax.plot(
        [
            edge1[0],
            edge2[0],
        ],
        [
            edge1[1],
            edge2[1],
        ],
        "o-",
        linewidth=2,
        label="69.2-mm diameter",
    )


    # Computed bottle center
    ax.plot(
        computed_center[0],
        computed_center[1],
        "o",
        markersize=12,
        label="computed bottle centre",
    )


    # Manual center
    ax.plot(
        manual_center[0],
        manual_center[1],
        "x",
        markersize=12,
        markeredgewidth=3,
        label="manual centre check",
    )


    # Show error vector
    ax.plot(
        [
            manual_center[0],
            computed_center[0],
        ],
        [
            manual_center[1],
            computed_center[1],
        ],
        "--",
        linewidth=2,
    )


    # MCP
    ax.plot(
        mcp[0],
        mcp[1],
        "o",
        markersize=11,
        label="MCP",
    )


    # Pink axis
    ax.plot(
        [
            pink_prox[0],
            pink_dist[0],
        ],
        [
            pink_prox[1],
            pink_dist[1],
        ],
        "o-",
        linewidth=4,
        label="pink proximal axis",
    )


    # MCP -> bottle center
    ax.plot(
        [
            mcp[0],
            computed_center[0],
        ],
        [
            mcp[1],
            computed_center[1],
        ],
        ":",
        linewidth=2,
        label="MCP → bottle centre",
    )


    ax.text(
        computed_center[0] + 5,
        computed_center[1],
        "computed centre",
    )


    ax.text(
        manual_center[0] + 5,
        manual_center[1],
        (
            f"manual centre\n"
            f"error={center_mismatch_px:.1f}px"
        ),
    )


    ax.text(
        mcp[0] + 5,
        mcp[1],
        "MCP",
    )


    ax.set_title(
        "Hardware bottle geometry\n"
        f"quality: {quality}"
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


    out = (
        result_dir
        / "hardware_bottle_geometry_v2.png"
    )


    fig.savefig(
        out,
        dpi=180,
        bbox_inches="tight",
    )


    plt.close(fig)


    print()
    print("Saved:")
    print(out)

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
