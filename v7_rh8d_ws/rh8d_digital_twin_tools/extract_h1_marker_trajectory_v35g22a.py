#!/usr/bin/env python3

import argparse
import csv
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

import fit_index_contact_v1 as h1


HW_TARGET = np.array([
    14.5355,
    23.8110,
    9.3270,
])

# Historical synchronization:
# video target 50.5 s corresponds to the hardware-aligned
# bag target around 38.970486 s.
VIDEO_TARGET_TIME = 50.5
BAG_TARGET_TIME = 38.970486

# Deliberately denser near physical contact.
VIDEO_TIMES = [
    43.500,
    44.200,
    45.000,
    46.000,
    47.000,
    48.000,
    48.500,
    48.800,
    48.950,
    49.050,
    49.200,
    49.580,
    49.900,
    50.200,
    50.500,
]

MARKERS = [
    ("proximal", "PINK / proximal"),
    ("middle",   "ORANGE / middle"),
    ("distal",   "GREEN / distal"),
]


def wrap_axis(a):
    while a >= 90.0:
        a -= 180.0
    while a < -90.0:
        a += 180.0
    return a


def get_frame(cap, time_s):
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


def scale_frame(frame):

    h, w = frame.shape[:2]

    scale = min(
        1.0,
        1400.0 / w,
        900.0 / h,
    )

    if scale >= 0.999:
        return frame.copy()

    return cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def choose_axis(frame, title):

    display = scale_frame(frame)

    points = []

    window = "G22A marker measurement"

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL
        |
        cv2.WINDOW_KEEPRATIO,
    )

    cv2.resizeWindow(
        window,
        1200,
        800,
    )

    cv2.moveWindow(
        window,
        20,
        20,
    )

    def redraw():

        show = display.copy()

        cv2.putText(
            show,
            title,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            show,
            "Click TWO points along marker axis | ENTER accept | R reset | Q quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        for pnt in points:

            cv2.circle(
                show,
                pnt,
                6,
                (0, 0, 255),
                -1,
            )

        if len(points) == 2:

            cv2.line(
                show,
                points[0],
                points[1],
                (0, 255, 255),
                2,
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
            len(points) < 2
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

        # Block until a key is pressed.
        # The mouse callback still works while waiting,
        # so the window is NOT continuously redrawn.
        key = (
            cv2.waitKey(0)
            &
            0xFF
        )

        if (
            len(points) == 2
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

    p1, p2 = points

    dx = (
        p2[0]
        -
        p1[0]
    )

    dy = -(
        p2[1]
        -
        p1[1]
    )

    angle = math.degrees(
        math.atan2(
            dy,
            dx,
        )
    )

    return (
        wrap_axis(
            angle
        ),
        p1,
        p2,
    )


def relative_angles(axes):

    return np.array([
        axes["proximal"],
        wrap_axis(
            axes["middle"]
            -
            axes["proximal"]
        ),
        wrap_axis(
            axes["distal"]
            -
            axes["middle"]
        ),
    ])


def encoder_at(
    encoder_data,
    bag_time,
):

    t = np.asarray(
        encoder_data[:, 0],
        dtype=float,
    )

    e = np.asarray(
        encoder_data[:, 1],
        dtype=float,
    )

    return float(
        np.interp(
            bag_time,
            t,
            e,
        )
    )


def threshold_tick(
    rows,
    joint_key,
    threshold,
):

    clean = sorted(
        rows,
        key=lambda r:
            r["encoder_tick"],
    )

    for a, b in zip(
        clean[:-1],
        clean[1:],
    ):

        qa = a[joint_key]
        qb = b[joint_key]

        if (
            qa < threshold
            <= qb
            and
            qb != qa
        ):

            f = (
                threshold - qa
            ) / (
                qb - qa
            )

            return (
                a["encoder_tick"]
                +
                f
                *
                (
                    b["encoder_tick"]
                    -
                    a["encoder_tick"]
                )
            )

    return np.nan


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=Path,
        default=Path.home()
        / "Downloads"
        / "14.mp4",
    )

    parser.add_argument(
        "--bag",
        type=Path,
        default=Path.home()
        / "rh8d_system_id"
        / "index_contact_20260911_164310",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home()
        / "rh8d_results"
        / "v35g22a_h1_marker_trajectory",
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

    bag = (
        args.bag
        .expanduser()
        .resolve()
    )

    out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    if not video.exists():
        raise FileNotFoundError(
            f"Video not found: {video}"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        out_dir
        /
        Path(__file__).name,
    )

    (
        states,
        force_data,
        encoder_data,
    ) = h1.read_bag(
        bag
    )

    encoder_data = np.asarray(
        encoder_data,
        dtype=float,
    ).copy()

    encoder_data[:, 1] = (
        h1.median_filter_5(
            encoder_data[:, 1]
        )
    )

    cap = cv2.VideoCapture(
        str(video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {video}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    nframes = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        nframes / fps
        if fps > 0
        else np.nan
    )

    offset = (
        BAG_TARGET_TIME
        -
        VIDEO_TARGET_TIME
    )

    print()
    print("=" * 110)
    print(
        "G22A — H1 PHYSICAL MCP/PIP/DIP TRAJECTORY"
    )
    print("=" * 110)

    print(
        f"Video    = {video}"
    )

    print(
        f"FPS      = {fps:.6f}"
    )

    print(
        f"Duration = {duration:.3f} s"
    )

    print(
        f"Bag      = {bag}"
    )

    print()
    print(
        "Synchronization:"
    )

    print(
        f"  video {VIDEO_TARGET_TIME:.6f} s"
        f" -> bag {BAG_TARGET_TIME:.6f} s"
    )

    print(
        f"  bag_time = video_time "
        f"{offset:+.6f} s"
    )

    print()
    print(
        "For each frame:"
    )

    print(
        "  1. PINK proximal marker: click two points"
    )

    print(
        "  2. ORANGE middle marker: click two points"
    )

    print(
        "  3. GREEN distal marker: click two points"
    )

    print(
        "ENTER/SPACE accepts each marker."
    )

    raw = []

    try:

        for i, video_time in enumerate(
            VIDEO_TIMES,
            start=1,
        ):

            frame = get_frame(
                cap,
                video_time,
            )

            bag_time = (
                video_time
                +
                offset
            )

            encoder = encoder_at(
                encoder_data,
                bag_time,
            )

            print()
            print("-" * 110)

            print(
                f"[{i:02d}/{len(VIDEO_TIMES):02d}] "
                f"video={video_time:.3f} s  "
                f"bag={bag_time:.6f} s  "
                f"encoder≈{encoder:.1f}"
            )

            print("-" * 110)

            axes = {}
            coords = {}

            for (
                marker,
                description,
            ) in MARKERS:

                angle, p1, p2 = (
                    choose_axis(
                        frame,
                        (
                            f"{i}/{len(VIDEO_TIMES)} "
                            f"t={video_time:.3f}s | "
                            f"{description}"
                        ),
                    )
                )

                axes[
                    marker
                ] = angle

                coords[
                    marker
                ] = (
                    p1,
                    p2,
                )

            q = relative_angles(
                axes
            )

            raw.append({
                "sample":
                    i,

                "video_time_s":
                    video_time,

                "bag_time_s":
                    bag_time,

                "encoder_tick":
                    encoder,

                "prox_axis_deg":
                    axes[
                        "proximal"
                    ],

                "middle_axis_deg":
                    axes[
                        "middle"
                    ],

                "distal_axis_deg":
                    axes[
                        "distal"
                    ],

                "raw_mcp_deg":
                    q[0],

                "raw_pip_deg":
                    q[1],

                "raw_dip_deg":
                    q[2],

                "prox_x1":
                    coords[
                        "proximal"
                    ][0][0],

                "prox_y1":
                    coords[
                        "proximal"
                    ][0][1],

                "prox_x2":
                    coords[
                        "proximal"
                    ][1][0],

                "prox_y2":
                    coords[
                        "proximal"
                    ][1][1],

                "mid_x1":
                    coords[
                        "middle"
                    ][0][0],

                "mid_y1":
                    coords[
                        "middle"
                    ][0][1],

                "mid_x2":
                    coords[
                        "middle"
                    ][1][0],

                "mid_y2":
                    coords[
                        "middle"
                    ][1][1],

                "dist_x1":
                    coords[
                        "distal"
                    ][0][0],

                "dist_y1":
                    coords[
                        "distal"
                    ][0][1],

                "dist_x2":
                    coords[
                        "distal"
                    ][1][0],

                "dist_y2":
                    coords[
                        "distal"
                    ][1][1],
            })

    finally:

        cap.release()
        cv2.destroyAllWindows()

    q0 = np.array([
        raw[0]["raw_mcp_deg"],
        raw[0]["raw_pip_deg"],
        raw[0]["raw_dip_deg"],
    ])

    rows = []

    for r in raw:

        q = np.array([
            r["raw_mcp_deg"],
            r["raw_pip_deg"],
            r["raw_dip_deg"],
        ])

        delta = np.array([
            wrap_axis(
                q[j]
                -
                q0[j]
            )
            for j in range(3)
        ])

        delta *= args.sign

        row = dict(r)

        row[
            "mcp_deg"
        ] = float(
            delta[0]
        )

        row[
            "pip_deg"
        ] = float(
            delta[1]
        )

        row[
            "dip_deg"
        ] = float(
            delta[2]
        )

        rows.append(
            row
        )

    csv_path = (
        out_dir
        /
        "g22a_h1_hardware_trajectory.csv"
    )

    with csv_path.open(
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
    print("=" * 110)
    print(
        "G22A RECOVERED HARDWARE TRAJECTORY"
    )
    print("=" * 110)

    print(
        "video_t   bag_t      encoder    "
        "MCP       PIP       DIP"
    )

    print("-" * 110)

    for r in rows:

        print(
            f"{r['video_time_s']:7.3f}  "
            f"{r['bag_time_s']:9.3f}  "
            f"{r['encoder_tick']:8.1f}  "
            f"{r['mcp_deg']:+8.3f}  "
            f"{r['pip_deg']:+8.3f}  "
            f"{r['dip_deg']:+8.3f}"
        )

    final = np.array([
        rows[-1][
            "mcp_deg"
        ],
        rows[-1][
            "pip_deg"
        ],
        rows[-1][
            "dip_deg"
        ],
    ])

    error = (
        final
        -
        HW_TARGET
    )

    print()
    print(
        "TARGET CROSS-CHECK @ 50.5 s"
    )

    print(
        "  measured = "
        f"{final[0]:.3f} / "
        f"{final[1]:.3f} / "
        f"{final[2]:.3f} deg"
    )

    print(
        "  trusted  = "
        f"{HW_TARGET[0]:.3f} / "
        f"{HW_TARGET[1]:.3f} / "
        f"{HW_TARGET[2]:.3f} deg"
    )

    print(
        "  error    = "
        f"{error[0]:+.3f} / "
        f"{error[1]:+.3f} / "
        f"{error[2]:+.3f} deg"
    )

    print()
    print(
        "APPROXIMATE ACTIVATION TICKS"
    )

    for threshold in [
        1.0,
        3.0,
    ]:

        values = []

        for joint in [
            "mcp_deg",
            "pip_deg",
            "dip_deg",
        ]:

            values.append(
                threshold_tick(
                    rows,
                    joint,
                    threshold,
                )
            )

        print(
            f"  {threshold:.0f} deg: "
            f"MCP/PIP/DIP = "
            f"{values[0]:.1f} / "
            f"{values[1]:.1f} / "
            f"{values[2]:.1f} ticks"
        )

    # ------------------------------------------------------------
    # Also save a compact identification table.
    # ------------------------------------------------------------

    compact = (
        out_dir
        /
        "g22a_identification_table.csv"
    )

    with compact.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "video_time_s",
            "bag_time_s",
            "encoder_tick",
            "mcp_deg",
            "pip_deg",
            "dip_deg",
        ])

        for r in rows:

            writer.writerow([
                r[
                    "video_time_s"
                ],
                r[
                    "bag_time_s"
                ],
                r[
                    "encoder_tick"
                ],
                r[
                    "mcp_deg"
                ],
                r[
                    "pip_deg"
                ],
                r[
                    "dip_deg"
                ],
            ])

    with (
        out_dir
        /
        "manifest.txt"
    ).open(
        "w"
    ) as f:

        f.write(
            "G22A H1 hardware marker trajectory\n"
        )

        f.write(
            f"video={video}\n"
        )

        f.write(
            f"bag={bag}\n"
        )

        f.write(
            f"video_target={VIDEO_TARGET_TIME}\n"
        )

        f.write(
            f"bag_target={BAG_TARGET_TIME}\n"
        )

        f.write(
            f"sign={args.sign}\n"
        )

        f.write(
            "No simulation parameters fitted in G22A.\n"
        )

    print()
    print(
        f"Saved:\n  {csv_path}\n  {compact}"
    )

    print()
    print(
        f"All G22A outputs:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
