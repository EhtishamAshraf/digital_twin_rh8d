#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import cv2


MARKERS = [
    ("proximal", "pink/proximal marker"),
    ("middle", "orange/middle marker"),
    ("distal", "green/distal marker"),
]


def wrap_axis(angle):
    """Marker axes are bidirectional, so wrap to [-90, 90)."""
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def read_frame(video, time_sec):

    cap = cv2.VideoCapture(str(video))

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        1000.0 * time_sec,
    )

    ok, frame = cap.read()

    cap.release()

    if not ok:
        raise RuntimeError(
            f"Could not read t={time_sec}s"
        )

    return frame


def choose_axis(frame, title):

    points = []

    image = frame.copy()

    window = title


    def callback(
        event,
        x,
        y,
        flags,
        param,
    ):

        if (
            event
            == cv2.EVENT_LBUTTONDOWN
            and
            len(points) < 2
        ):

            points.append(
                (x, y)
            )


    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window,
        1000,
        800,
    )

    cv2.setMouseCallback(
        window,
        callback,
    )


    while True:

        display = image.copy()


        cv2.putText(
            display,
            "Click TWO points along marker axis; "
            "ENTER=accept, R=reset",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )


        for p in points:

            cv2.circle(
                display,
                p,
                5,
                (255, 255, 255),
                -1,
            )


        if len(points) == 2:

            cv2.line(
                display,
                points[0],
                points[1],
                (255, 255, 255),
                2,
            )


        cv2.imshow(
            window,
            display,
        )


        key = (
            cv2.waitKey(30)
            & 0xFF
        )


        if (
            len(points) == 2
            and
            key in (13, 32)
        ):

            break


        if key == ord("r"):

            points.clear()


        if key in (
            27,
            ord("q"),
        ):

            raise KeyboardInterrupt


    cv2.destroyWindow(
        window
    )


    p1, p2 = points


    dx = (
        p2[0]
        -
        p1[0]
    )


    # Flip image y so normal mathematical CCW sign is used.
    dy = -(
        p2[1]
        -
        p1[1]
    )


    return math.degrees(
        math.atan2(
            dy,
            dx,
        )
    )


def measure_frame(
    frame,
    label,
):

    result = {}


    for key, description in MARKERS:

        result[key] = choose_axis(
            frame,
            f"{label}: {description}",
        )


    return result


def relative_angles(
    axes,
):

    return {

        "MCP":
            axes[
                "proximal"
            ],

        "PIP":
            wrap_axis(
                axes[
                    "middle"
                ]
                -
                axes[
                    "proximal"
                ]
            ),

        "DIP":
            wrap_axis(
                axes[
                    "distal"
                ]
                -
                axes[
                    "middle"
                ]
            ),
    }


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--video",
        required=True,
        type=Path,
    )


    parser.add_argument(
        "--open-time",
        required=True,
        type=float,
    )


    parser.add_argument(
        "--target-time",
        required=True,
        type=float,
    )


    parser.add_argument(
        "--sign",
        type=float,
        default=1.0,
        choices=[
            -1.0,
            1.0,
        ],
    )


    args = parser.parse_args()


    video = (
        args.video
        .expanduser()
        .resolve()
    )


    open_frame = read_frame(
        video,
        args.open_time,
    )


    target_frame = read_frame(
        video,
        args.target_time,
    )


    open_axes = measure_frame(
        open_frame,
        "OPEN",
    )


    target_axes = measure_frame(
        target_frame,
        "TARGET",
    )


    q0 = relative_angles(
        open_axes
    )


    q1 = relative_angles(
        target_axes
    )


    result = {}


    for joint in [
        "MCP",
        "PIP",
        "DIP",
    ]:

        result[joint] = (
            args.sign
            *
            wrap_axis(
                q1[joint]
                -
                q0[joint]
            )
        )


    print()
    print("========================")
    print("OPEN -> TARGET")
    print("========================")


    for joint in [
        "MCP",
        "PIP",
        "DIP",
    ]:

        print(
            f"{joint} = "
            f"{result[joint]:+.3f} deg"
        )


if __name__ == "__main__":
    main()