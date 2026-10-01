#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


VIDEO_TIME = 50.50
BOTTLE_DIAMETER_MM = 69.2


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)

    if n < 1e-9:
        raise RuntimeError(
            "Two selected points are too close."
        )

    return v / n


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python3 measure_hardware_bottle_geometry.py "
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
            "Could not read frame."
        )


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )


    print()
    print("=" * 96)
    print("HARDWARE BOTTLE / INDEX GEOMETRY")
    print("=" * 96)

    print()
    print(
        "Please zoom the image before clicking."
    )

    print()
    print(
        "Click these SIX points in order:"
    )

    print(
        "  1. bottle OUTER edge on one side"
    )

    print(
        "  2. bottle OUTER edge on the opposite side"
    )

    print(
        "     IMPORTANT: points 1 and 2 should form a diameter"
    )

    print(
        "     through the bottle centre, preferably approximately"
    )

    print(
        "     along the direction from bottle toward the finger."
    )

    print()

    print(
        "  3. bottle centre / axis centre"
    )

    print(
        "     (the centre of the circular top is a good guide)"
    )

    print()

    print(
        "  4. index MCP joint centre"
    )

    print()

    print(
        "  5. PINK marker axis — proximal point"
    )

    print(
        "  6. PINK marker axis — distal point"
    )

    print()


    fig, ax = plt.subplots(
        figsize=(10, 14)
    )

    ax.imshow(rgb)

    ax.set_title(
        "1 edge, 2 opposite edge, 3 bottle centre, "
        "4 MCP, 5-6 pink proximal->distal"
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


    p1 = np.asarray(
        pts[0],
        dtype=float,
    )

    p2 = np.asarray(
        pts[1],
        dtype=float,
    )

    bottle_center = np.asarray(
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


    # --------------------------------------------------------
    # Pixel scale from measured bottle diameter
    # --------------------------------------------------------

    bottle_diameter_px = float(
        np.linalg.norm(
            p2 - p1
        )
    )


    mm_per_px = (
        BOTTLE_DIAMETER_MM
        / bottle_diameter_px
    )


    # --------------------------------------------------------
    # Convert image coordinates to Cartesian:
    #
    # image:     +x right, +y downward
    # Cartesian: +x right, +y upward
    # --------------------------------------------------------

    def cart(point):

        return np.array(
            [
                point[0],
                -point[1],
            ],
            dtype=float,
        )


    bc = cart(
        bottle_center
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


    # Finger-forward direction from proximal -> distal.
    forward = unit(
        pd - pp
    )


    # One perpendicular direction in image plane.
    side = np.array(
        [
            -forward[1],
            forward[0],
        ]
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


    # Choose side positive toward the bottle.
    if side_px < 0:

        side *= -1.0
        side_px *= -1.0


    forward_mm = (
        forward_px
        * mm_per_px
    )


    side_mm = (
        side_px
        * mm_per_px
    )


    distance_mm = (
        np.linalg.norm(
            bottle_vector_px
        )
        * mm_per_px
    )


    print()
    print("=" * 96)
    print("CLICKED POINTS")
    print("=" * 96)

    print(
        f"edge 1        = {p1}"
    )

    print(
        f"edge 2        = {p2}"
    )

    print(
        f"bottle center = {bottle_center}"
    )

    print(
        f"MCP           = {mcp}"
    )

    print(
        f"pink proximal = {pink_prox}"
    )

    print(
        f"pink distal   = {pink_dist}"
    )


    print()
    print("=" * 96)
    print("IMAGE SCALE")
    print("=" * 96)

    print(
        f"Measured bottle diameter in image = "
        f"{bottle_diameter_px:.3f} px"
    )

    print(
        f"Known real bottle diameter         = "
        f"{BOTTLE_DIAMETER_MM:.3f} mm"
    )

    print(
        f"Scale                              = "
        f"{mm_per_px:.5f} mm/px"
    )


    print()
    print("=" * 96)
    print("BOTTLE CENTRE RELATIVE TO MCP")
    print("=" * 96)

    print(
        f"Along proximal-finger axis = "
        f"{forward_mm:+.3f} mm"
    )

    print(
        f"Perpendicular toward bottle = "
        f"{side_mm:+.3f} mm"
    )

    print(
        f"Total MCP -> bottle-centre distance = "
        f"{distance_mm:.3f} mm"
    )


    # --------------------------------------------------------
    # Draw diagnostic image
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 14)
    )

    ax.imshow(
        rgb
    )


    ax.plot(
        [
            p1[0],
            p2[0],
        ],
        [
            p1[1],
            p2[1],
        ],
        "o-",
        linewidth=2,
    )


    ax.plot(
        bottle_center[0],
        bottle_center[1],
        "o",
        markersize=10,
    )


    ax.plot(
        mcp[0],
        mcp[1],
        "o",
        markersize=10,
    )


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
        linewidth=3,
    )


    ax.plot(
        [
            mcp[0],
            bottle_center[0],
        ],
        [
            mcp[1],
            bottle_center[1],
        ],
        "--",
        linewidth=2,
    )


    ax.text(
        bottle_center[0] + 5,
        bottle_center[1],
        "Bottle center",
    )


    ax.text(
        mcp[0] + 5,
        mcp[1],
        "MCP",
    )


    ax.set_title(
        "Hardware bottle geometry measurement"
    )


    out = (
        Path(__file__).parent
        / "results/"
          "hardware_bottle_geometry.png"
    )


    out.parent.mkdir(
        parents=True,
        exist_ok=True,
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
    print("=" * 96)


if __name__ == "__main__":
    main()
