#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


VIDEO_TIME = 50.5


def angle_between(a, b):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    c = np.dot(a, b) / (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    c = np.clip(c, -1.0, 1.0)

    return math.degrees(
        math.acos(c)
    )


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python3 measure_index_contact_pose.py "
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
            "Could not read requested frame."
        )


    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )


    print()
    print("=" * 80)
    print("HARDWARE CONTACT-POSE MEASUREMENT")
    print("=" * 80)

    print()
    print(
        f"Video time = {VIDEO_TIME:.3f} s"
    )

    print()
    print(
        "Click these FOUR points in order:"
    )

    print(
        "  1. MCP joint centre"
    )

    print(
        "  2. PIP joint centre"
    )

    print(
        "  3. DIP joint centre"
    )

    print(
        "  4. fingertip centre"
    )

    print()
    print(
        "Zoom the Matplotlib window first if needed."
    )

    print(
        "Then left-click the four points."
    )


    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.imshow(
        frame_rgb
    )

    ax.set_title(
        "Click MCP -> PIP -> DIP -> fingertip"
    )

    points = plt.ginput(
        4,
        timeout=0,
    )

    plt.close(
        fig
    )


    if len(points) != 4:

        raise RuntimeError(
            "Exactly four points are required."
        )


    mcp, pip, dip, tip = [
        np.asarray(
            p,
            dtype=float,
        )
        for p in points
    ]


    proximal = (
        pip - mcp
    )

    middle = (
        dip - pip
    )

    distal = (
        tip - dip
    )


    pip_internal = angle_between(
        proximal,
        middle,
    )

    dip_internal = angle_between(
        middle,
        distal,
    )


    print()
    print("=" * 80)
    print("CLICKED IMAGE COORDINATES")
    print("=" * 80)

    print(
        f"MCP = {mcp}"
    )

    print(
        f"PIP = {pip}"
    )

    print(
        f"DIP = {dip}"
    )

    print(
        f"TIP = {tip}"
    )


    print()
    print("=" * 80)
    print("RELATIVE SEGMENT ANGLES")
    print("=" * 80)

    print(
        f"PIP relative bend = "
        f"{pip_internal:.3f} deg"
    )

    print(
        f"DIP relative bend = "
        f"{dip_internal:.3f} deg"
    )


    # --------------------------------------------------------
    # Save annotated frame.
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.imshow(
        frame_rgb
    )


    pts = np.array(
        [
            mcp,
            pip,
            dip,
            tip,
        ]
    )


    ax.plot(
        pts[:, 0],
        pts[:, 1],
        "o-",
        linewidth=2,
    )


    labels = [
        "MCP",
        "PIP",
        "DIP",
        "TIP",
    ]


    for label, p in zip(
        labels,
        pts,
    ):

        ax.text(
            p[0] + 5,
            p[1] - 5,
            label,
            fontsize=12,
        )


    ax.set_title(
        f"Hardware index contact pose @ "
        f"{VIDEO_TIME:.2f} s"
    )


    output = (
        Path(__file__).parent
        / "results/"
          "index_contact_pose_hardware.png"
    )


    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    print()
    print("Saved:")
    print(output)

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
