#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


OPEN_TIME = 43.5
HOLD_TIME = 50.5

MARKERS = [
    ("PROXIMAL / PINK", "pink"),
    ("MIDDLE / ORANGE", "orange"),
    ("DISTAL / GREEN", "green"),
]


def read_frame(cap, time_s):

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        time_s * 1000.0,
    )

    ok, frame = cap.read()

    if not ok:
        raise RuntimeError(
            f"Could not read video at {time_s:.3f} s"
        )

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )


def wrap_deg(angle):

    return (
        (angle + 180.0) % 360.0
        - 180.0
    )


def marker_angle(p0, p1):

    dx = p1[0] - p0[0]

    # Image Y points downward.
    # Convert to normal Cartesian coordinates.
    dy = -(p1[1] - p0[1])

    return math.degrees(
        math.atan2(
            dy,
            dx,
        )
    )


def measure_frame(
    frame,
    title,
):

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    print()
    print(
        "For EACH coloured marker, click TWO points "
        "along its LONG AXIS."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Always click from the PROXIMAL side "
        "toward the DISTAL / fingertip side."
    )

    print()

    for i, (name, _) in enumerate(
        MARKERS,
        start=1,
    ):

        print(
            f"  clicks {2*i-1},{2*i}: {name}"
        )

    print()


    fig, ax = plt.subplots(
        figsize=(9, 12)
    )

    ax.imshow(frame)

    ax.set_title(
        title
        + "\n"
        "Click 2 points along PINK, "
        "then ORANGE, then GREEN"
    )

    points = plt.ginput(
        6,
        timeout=0,
    )

    plt.close(fig)


    if len(points) != 6:

        raise RuntimeError(
            "Exactly 6 clicks required."
        )


    points = [
        np.asarray(
            p,
            dtype=float,
        )
        for p in points
    ]


    result = {}

    for i, (
        marker_name,
        short_name,
    ) in enumerate(MARKERS):

        p0 = points[
            2*i
        ]

        p1 = points[
            2*i + 1
        ]

        angle = marker_angle(
            p0,
            p1,
        )

        result[
            short_name
        ] = {
            "p0": p0,
            "p1": p1,
            "angle": angle,
        }


    return result


def annotate(
    frame,
    measurements,
    title,
    output,
):

    fig, ax = plt.subplots(
        figsize=(9, 12)
    )

    ax.imshow(frame)


    for name, data in measurements.items():

        p0 = data["p0"]
        p1 = data["p1"]

        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            "o-",
            linewidth=3,
        )

        centre = (
            0.5
            * (
                p0 + p1
            )
        )

        ax.text(
            centre[0] + 5,
            centre[1],
            (
                f"{name}\n"
                f"{data['angle']:.1f}°"
            ),
            fontsize=11,
        )


    ax.set_title(title)

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python3 measure_index_contact_pose_v2.py "
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


    open_frame = read_frame(
        cap,
        OPEN_TIME,
    )

    hold_frame = read_frame(
        cap,
        HOLD_TIME,
    )

    cap.release()


    open_m = measure_frame(
        open_frame,
        f"OPEN REFERENCE @ {OPEN_TIME:.2f} s",
    )


    hold_m = measure_frame(
        hold_frame,
        f"CONTACT HOLD @ {HOLD_TIME:.2f} s",
    )


    # --------------------------------------------------------
    # Absolute marker orientations
    # --------------------------------------------------------

    op = open_m[
        "pink"
    ]["angle"]

    om = open_m[
        "orange"
    ]["angle"]

    od = open_m[
        "green"
    ]["angle"]


    hp = hold_m[
        "pink"
    ]["angle"]

    hm = hold_m[
        "orange"
    ]["angle"]

    hd = hold_m[
        "green"
    ]["angle"]


    # --------------------------------------------------------
    # Relative joint orientations within each frame
    # --------------------------------------------------------

    open_pip = wrap_deg(
        om - op
    )

    open_dip = wrap_deg(
        od - om
    )


    hold_pip = wrap_deg(
        hm - hp
    )

    hold_dip = wrap_deg(
        hd - hm
    )


    # --------------------------------------------------------
    # Change from open pose.
    #
    # Sticker mounting offsets cancel here.
    # --------------------------------------------------------

    delta_mcp = wrap_deg(
        hp - op
    )

    delta_pip = wrap_deg(
        hold_pip
        - open_pip
    )

    delta_dip = wrap_deg(
        hold_dip
        - open_dip
    )


    print()
    print("=" * 90)
    print("MARKER ORIENTATIONS")
    print("=" * 90)

    print()
    print("OPEN:")

    print(
        f"  pink   = {op:8.3f} deg"
    )

    print(
        f"  orange = {om:8.3f} deg"
    )

    print(
        f"  green  = {od:8.3f} deg"
    )


    print()
    print("CONTACT HOLD:")

    print(
        f"  pink   = {hp:8.3f} deg"
    )

    print(
        f"  orange = {hm:8.3f} deg"
    )

    print(
        f"  green  = {hd:8.3f} deg"
    )


    print()
    print("=" * 90)
    print("RELATIVE JOINT CHANGE FROM OPEN -> CONTACT")
    print("=" * 90)

    print(
        f"MCP signed change = "
        f"{delta_mcp:8.3f} deg"
    )

    print(
        f"PIP signed change = "
        f"{delta_pip:8.3f} deg"
    )

    print(
        f"DIP signed change = "
        f"{delta_dip:8.3f} deg"
    )


    print()
    print("Magnitudes:")

    print(
        f"  MCP = {abs(delta_mcp):8.3f} deg"
    )

    print(
        f"  PIP = {abs(delta_pip):8.3f} deg"
    )

    print(
        f"  DIP = {abs(delta_dip):8.3f} deg"
    )


    result_dir = (
        Path(__file__).parent
        / "results"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    annotate(
        open_frame,
        open_m,
        f"Index marker axes — OPEN @ {OPEN_TIME:.2f} s",
        result_dir
        / "index_marker_axes_open.png",
    )


    annotate(
        hold_frame,
        hold_m,
        f"Index marker axes — CONTACT @ {HOLD_TIME:.2f} s",
        result_dir
        / "index_marker_axes_contact.png",
    )


    print()
    print("Saved:")

    print(
        result_dir
        / "index_marker_axes_open.png"
    )

    print(
        result_dir
        / "index_marker_axes_contact.png"
    )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
