#!/usr/bin/env python3

import argparse
import csv
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


TRUSTED_DIP_DEG = 9.3270


def wrap_axis(a):
    """Wrap bidirectional relative angle to [-90, 90)."""
    while a >= 90.0:
        a -= 180.0
    while a < -90.0:
        a += 180.0
    return a


def angle_between(a, b):

    dx = b[0] - a[0]

    # Image y increases downward.
    dy = -(b[1] - a[1])

    return math.degrees(
        math.atan2(
            dy,
            dx,
        )
    )


def read_frame(
    cap,
    time_s,
):

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        1000.0 * time_s,
    )

    ok, frame = cap.read()

    if not ok:
        raise RuntimeError(
            f"Cannot read video at {time_s:.3f} s"
        )

    return frame


def crop_hand(frame):

    h, w = frame.shape[:2]

    # Same hand region used in G22A2.
    x1 = int(0.32 * w)
    x2 = int(0.99 * w)

    y1 = int(0.12 * h)
    y2 = int(0.64 * h)

    crop = frame[
        y1:y2,
        x1:x2,
    ].copy()

    return crop


def resize_for_display(frame):

    h, w = frame.shape[:2]

    scale = min(
        1.6,
        1500.0 / w,
        900.0 / h,
    )

    resized = cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR,
    )

    return resized, scale


def click_three_points(
    frame,
    frame_index,
    total_frames,
    time_s,
    encoder,
):

    crop = crop_hand(
        frame
    )

    display, scale = resize_for_display(
        crop
    )

    points = []

    labels = [
        "PIP CENTER",
        "DIP CENTER",
        "DISTAL END CENTER",
    ]

    window = "G22A3 DIP three-point measurement"

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL
        |
        cv2.WINDOW_KEEPRATIO,
    )

    cv2.resizeWindow(
        window,
        1450,
        900,
    )

    cv2.moveWindow(
        window,
        10,
        10,
    )


    def redraw():

        show = display.copy()

        cv2.putText(
            show,
            (
                f"Frame {frame_index}/{total_frames} | "
                f"video={time_s:.3f}s | "
                f"encoder={encoder:.1f}"
            ),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )

        if len(points) < 3:

            instruction = (
                f"CLICK {len(points)+1}: "
                f"{labels[len(points)]}"
            )

        else:

            instruction = (
                "3 points complete - press ENTER/SPACE"
            )

        cv2.putText(
            show,
            instruction,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            show,
            "R = reset current frame | Q/ESC = quit",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        for i, p in enumerate(points):

            cv2.circle(
                show,
                p,
                7,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                show,
                str(i + 1),
                (
                    p[0] + 10,
                    p[1] - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if len(points) >= 2:

            cv2.line(
                show,
                points[0],
                points[1],
                (255, 255, 0),
                3,
            )

        if len(points) >= 3:

            cv2.line(
                show,
                points[1],
                points[2],
                (0, 255, 0),
                3,
            )

        cv2.imshow(
            window,
            show,
        )


    def callback(
        event,
        x,
        y,
        flags,
        param,
    ):

        if (
            event
            ==
            cv2.EVENT_LBUTTONDOWN
            and
            len(points) < 3
        ):

            points.append(
                (
                    int(x),
                    int(y),
                )
            )

            redraw()


    cv2.setMouseCallback(
        window,
        callback,
    )

    redraw()

    while True:

        key = (
            cv2.waitKey(0)
            &
            0xFF
        )

        if (
            len(points) == 3
            and
            key in (
                13,
                32,
            )
        ):
            break

        if key == ord("r"):

            points.clear()

            redraw()

        if key in (
            27,
            ord("q"),
        ):
            raise KeyboardInterrupt

    cv2.destroyWindow(
        window
    )

    # Convert display coordinates back into crop coordinates.
    pts = [
        (
            p[0] / scale,
            p[1] / scale,
        )
        for p in points
    ]

    return pts


def activation_tick(
    encoder,
    q,
    threshold,
):

    encoder = np.asarray(
        encoder,
        dtype=float,
    )

    q = np.asarray(
        q,
        dtype=float,
    )

    for i in range(
        len(q) - 1
    ):

        if (
            q[i] < threshold
            <= q[i + 1]
            and
            q[i + 1] != q[i]
        ):

            f = (
                threshold - q[i]
            ) / (
                q[i + 1] - q[i]
            )

            return (
                encoder[i]
                +
                f
                *
                (
                    encoder[i + 1]
                    -
                    encoder[i]
                )
            )

    return np.nan


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=Path,
        default=(
            Path.home()
            / "Downloads"
            / "14.mp4"
        ),
    )

    parser.add_argument(
        "--trajectory",
        type=Path,
        default=(
            Path.home()
            / "rh8d_results"
            / "v35g22a_h1_marker_trajectory"
            / "g22a_h1_hardware_trajectory.csv"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            / "rh8d_results"
            / "v35g22a3_dip_3point"
        ),
    )

    args = parser.parse_args()

    video = (
        args.video
        .expanduser()
        .resolve()
    )

    trajectory = (
        args.trajectory
        .expanduser()
        .resolve()
    )

    out = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        out
        /
        Path(__file__).name,
    )

    df = pd.read_csv(
        trajectory
    )

    cap = cv2.VideoCapture(
        str(video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video}"
        )

    rows = []

    print()
    print("=" * 120)
    print(
        "G22A3 — H1 DIP THREE-POINT GEOMETRIC MEASUREMENT"
    )
    print("=" * 120)

    print()
    print(
        "For EACH frame click exactly:"
    )

    print(
        "  1. PIP CENTER          = hinge between proximal and middle phalanges"
    )

    print(
        "  2. DIP CENTER          = hinge between middle and distal phalanges"
    )

    print(
        "  3. DISTAL END CENTER   = center of far end of distal phalanx"
    )

    print()
    print(
        "Do NOT click marker edges."
    )

    print(
        "Click the approximate mechanical CENTER of each joint/end."
    )

    print()

    try:

        for idx, row in df.iterrows():

            video_time = float(
                row[
                    "video_time_s"
                ]
            )

            encoder = float(
                row[
                    "encoder_tick"
                ]
            )

            frame = read_frame(
                cap,
                video_time,
            )

            print(
                f"[{idx+1:02d}/{len(df):02d}] "
                f"video={video_time:.3f}s "
                f"encoder={encoder:.1f}"
            )

            (
                pip_center,
                dip_center,
                distal_end,
            ) = click_three_points(
                frame,
                idx + 1,
                len(df),
                video_time,
                encoder,
            )

            middle_angle = angle_between(
                pip_center,
                dip_center,
            )

            distal_angle = angle_between(
                dip_center,
                distal_end,
            )

            relative_dip = wrap_axis(
                distal_angle
                -
                middle_angle
            )

            rows.append({
                "video_time_s":
                    video_time,

                "bag_time_s":
                    float(
                        row[
                            "bag_time_s"
                        ]
                    ),

                "encoder_tick":
                    encoder,

                "mcp_deg":
                    float(
                        row[
                            "mcp_deg"
                        ]
                    ),

                "pip_deg":
                    float(
                        row[
                            "pip_deg"
                        ]
                    ),

                "pip_center_x":
                    pip_center[0],

                "pip_center_y":
                    pip_center[1],

                "dip_center_x":
                    dip_center[0],

                "dip_center_y":
                    dip_center[1],

                "distal_end_x":
                    distal_end[0],

                "distal_end_y":
                    distal_end[1],

                "middle_axis_deg":
                    middle_angle,

                "distal_axis_deg":
                    distal_angle,

                "relative_dip_raw_deg":
                    relative_dip,
            })

    finally:

        cap.release()

        cv2.destroyAllWindows()

    # ------------------------------------------------------------
    # Reference everything to open configuration.
    # ------------------------------------------------------------

    q0 = float(
        rows[0][
            "relative_dip_raw_deg"
        ]
    )

    raw_series = []

    for row in rows:

        q = wrap_axis(
            row[
                "relative_dip_raw_deg"
            ]
            -
            q0
        )

        raw_series.append(
            q
        )

    # ------------------------------------------------------------
    # Temporal unwrap around nearest previous value.
    # ------------------------------------------------------------

    dip = [
        raw_series[0]
    ]

    for value in raw_series[1:]:

        v = value

        while (
            v - dip[-1]
        ) > 90.0:
            v -= 180.0

        while (
            v - dip[-1]
        ) < -90.0:
            v += 180.0

        dip.append(
            v
        )

    for row, q in zip(
        rows,
        dip,
    ):

        row[
            "dip_3point_deg"
        ] = float(q)

    output = (
        out
        /
        "g22a3_h1_dip_3point.csv"
    )

    with output.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print("=" * 120)
    print(
        "G22A3 RESULT"
    )
    print("=" * 120)

    print(
        "video_t   encoder      MCP        PIP        DIP-3POINT"
    )

    print("-" * 120)

    for row in rows:

        print(
            f"{row['video_time_s']:7.3f} "
            f"{row['encoder_tick']:8.1f} "
            f"{row['mcp_deg']:+10.3f} "
            f"{row['pip_deg']:+10.3f} "
            f"{row['dip_3point_deg']:+12.3f}"
        )

    final_dip = float(
        rows[-1][
            "dip_3point_deg"
        ]
    )

    final_error = (
        final_dip
        -
        TRUSTED_DIP_DEG
    )

    print()
    print(
        "FINAL DIP CROSS-CHECK"
    )

    print(
        f"  3-point DIP = "
        f"{final_dip:.3f} deg"
    )

    print(
        f"  trusted DIP = "
        f"{TRUSTED_DIP_DEG:.3f} deg"
    )

    print(
        f"  error       = "
        f"{final_error:+.3f} deg"
    )

    encoder = [
        r[
            "encoder_tick"
        ]
        for r in rows
    ]

    dip_values = [
        r[
            "dip_3point_deg"
        ]
        for r in rows
    ]

    print()
    print(
        "APPROXIMATE DIP ACTIVATION"
    )

    for threshold in [
        1.0,
        3.0,
        5.0,
    ]:

        tick = activation_tick(
            encoder,
            dip_values,
            threshold,
        )

        print(
            f"  {threshold:.0f} deg -> "
            f"{tick:.1f} ticks"
        )

    print()
    print(
        "QUALITY DECISION"
    )

    if abs(
        final_error
    ) <= 2.0:

        print(
            "  PASS: final DIP is within 2 deg "
            "of trusted contact measurement."
        )

    else:

        print(
            "  WARNING: final DIP differs by more than 2 deg."
        )

        print(
            "  Do NOT fit the dynamic model to this trajectory yet."
        )

    print()
    print(
        f"Saved:\n  {output}"
    )

    print()
    print(
        f"All outputs:\n  {out}"
    )


if __name__ == "__main__":
    main()
