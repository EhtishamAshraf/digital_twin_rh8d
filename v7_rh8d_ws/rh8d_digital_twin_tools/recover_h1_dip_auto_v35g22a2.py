#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


TRUSTED = np.array([
    14.5355,
    23.8110,
    9.3270,
])


def wrap180(a):
    while a >= 90.0:
        a -= 180.0
    while a < -90.0:
        a += 180.0
    return a


def angular_distance(a, b):
    return abs(
        wrap180(a - b)
    )


def component(
    frame,
    low,
    high,
):

    h, w = frame.shape[:2]

    # Restrict detection to the hand / marker region.
    x1 = int(0.34 * w)
    x2 = int(0.99 * w)

    y1 = int(0.17 * h)
    y2 = int(0.58 * h)

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

    mask = cv2.inRange(
        hsv,
        np.asarray(low, dtype=np.uint8),
        np.asarray(high, dtype=np.uint8),
    )

    roi = np.zeros_like(mask)

    roi[
        y1:y2,
        x1:x2,
    ] = 255

    mask = cv2.bitwise_and(
        mask,
        roi,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones(
            (3, 3),
            np.uint8,
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones(
            (5, 5),
            np.uint8,
        ),
    )

    n, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask
        )
    )

    candidates = [
        i
        for i in range(1, n)
        if stats[
            i,
            cv2.CC_STAT_AREA
        ] > 100
    ]

    if not candidates:
        raise RuntimeError(
            "Marker component not found."
        )

    idx = max(
        candidates,
        key=lambda i:
            stats[
                i,
                cv2.CC_STAT_AREA
            ],
    )

    ys, xs = np.where(
        labels == idx
    )

    points = np.column_stack([
        xs,
        ys,
    ]).astype(
        np.float32
    )

    centroid = np.asarray(
        centroids[idx],
        dtype=float,
    )

    return (
        points,
        centroid,
        int(
            stats[
                idx,
                cv2.CC_STAT_AREA
            ]
        ),
    )


def vector_angle(a, b):

    dx = (
        b[0]
        -
        a[0]
    )

    dy = -(
        b[1]
        -
        a[1]
    )

    return wrap180(
        math.degrees(
            math.atan2(
                dy,
                dx,
            )
        )
    )


def rectangle_axes(points):

    rect = cv2.minAreaRect(
        points
    )

    box = cv2.boxPoints(
        rect
    )

    axes = []

    for i, j in [
        (0, 1),
        (1, 2),
    ]:

        dx = (
            box[j, 0]
            -
            box[i, 0]
        )

        dy = -(
            box[j, 1]
            -
            box[i, 1]
        )

        axes.append(
            wrap180(
                math.degrees(
                    math.atan2(
                        dy,
                        dx,
                    )
                )
            )
        )

    return axes


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
            / "v35g22a2_auto_dip"
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

    df = pd.read_csv(
        trajectory
    )

    cap = cv2.VideoCapture(
        str(video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {video}"
        )

    distal_angles = []
    areas = []

    previous_angle = None

    for _, row in df.iterrows():

        t = float(
            row[
                "video_time_s"
            ]
        )

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            t * 1000.0,
        )

        ok, frame = cap.read()

        if not ok:
            raise RuntimeError(
                f"Cannot read frame at {t}"
            )

        # Orange marker centroid.
        _, orange_center, _ = component(
            frame,
            [0, 55, 70],
            [20, 255, 255],
        )

        # Green marker.
        green_points, green_center, area = (
            component(
                frame,
                [40, 45, 70],
                [90, 255, 255],
            )
        )

        # Orange -> green gives an independent estimate
        # of the distal-link direction.
        reference = vector_angle(
            orange_center,
            green_center,
        )

        candidates = rectangle_axes(
            green_points
        )

        angle = min(
            candidates,
            key=lambda a:
                angular_distance(
                    a,
                    reference,
                ),
        )

        # Temporal 180-degree unwrap.
        if previous_angle is not None:

            while (
                angle
                -
                previous_angle
            ) > 90.0:
                angle -= 180.0

            while (
                angle
                -
                previous_angle
            ) < -90.0:
                angle += 180.0

        previous_angle = angle

        distal_angles.append(
            angle
        )

        areas.append(
            area
        )

    cap.release()

    distal_angles = np.asarray(
        distal_angles,
        dtype=float,
    )

    distal_delta = (
        distal_angles
        -
        distal_angles[0]
    )

    # Marker-angle equations:
    #
    # MCP = Δtheta_prox
    #
    # PIP = Δtheta_middle - Δtheta_prox
    #
    # therefore:
    #
    # Δtheta_middle = MCP + PIP
    #
    middle_delta = (
        df[
            "mcp_deg"
        ].to_numpy(
            dtype=float
        )
        +
        df[
            "pip_deg"
        ].to_numpy(
            dtype=float
        )
    )

    dip_auto = (
        distal_delta
        -
        middle_delta
    )

    dip_auto = np.asarray([
        wrap180(v)
        for v in dip_auto
    ])

    df[
        "distal_axis_auto_deg"
    ] = distal_angles

    df[
        "distal_axis_delta_auto_deg"
    ] = distal_delta

    df[
        "middle_axis_delta_clicked_deg"
    ] = middle_delta

    df[
        "dip_auto_deg"
    ] = dip_auto

    df[
        "green_area_px"
    ] = areas

    output = (
        out
        /
        "g22a2_h1_with_auto_dip.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 118)
    print(
        "G22A2 — AUTOMATIC DISTAL/DIP RECOVERY"
    )
    print("=" * 118)

    print(
        "video_t   encoder    MCP-click   PIP-click   "
        "DIP-click   DIP-auto"
    )

    print("-" * 118)

    for _, row in df.iterrows():

        print(
            f"{row['video_time_s']:7.3f} "
            f"{row['encoder_tick']:8.1f} "
            f"{row['mcp_deg']:+10.3f} "
            f"{row['pip_deg']:+10.3f} "
            f"{row['dip_deg']:+10.3f} "
            f"{row['dip_auto_deg']:+10.3f}"
        )

    # --------------------------------------------------------
    # Critical independent final-pose cross-check.
    #
    # Use automatic distal-axis change, but TRUSTED MCP/PIP
    # at the final frame rather than the noisy final clicks.
    # --------------------------------------------------------

    final_distal_delta = float(
        distal_delta[-1]
    )

    final_dip_from_trusted_mp = (
        final_distal_delta
        -
        (
            TRUSTED[0]
            +
            TRUSTED[1]
        )
    )

    final_dip_from_trusted_mp = (
        wrap180(
            final_dip_from_trusted_mp
        )
    )

    print()
    print(
        "FINAL DISTAL CROSS-CHECK"
    )

    print(
        f"automatic distal-axis change = "
        f"{final_distal_delta:.3f} deg"
    )

    print(
        "using trusted MCP+PIP:"
    )

    print(
        f"  inferred DIP = "
        f"{final_dip_from_trusted_mp:.3f} deg"
    )

    print(
        f"  trusted DIP  = "
        f"{TRUSTED[2]:.3f} deg"
    )

    print(
        f"  error        = "
        f"{final_dip_from_trusted_mp - TRUSTED[2]:+.3f} deg"
    )

    print()
    print(
        f"Saved:\n  {output}"
    )


if __name__ == "__main__":
    main()
