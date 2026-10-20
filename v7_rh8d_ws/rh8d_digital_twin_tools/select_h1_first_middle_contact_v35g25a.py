#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import fit_index_contact_v1 as h1


# Same H1 synchronization already established.
VIDEO_TARGET_TIME = 50.500000
BAG_TARGET_TIME = 38.970486

VIDEO_TO_BAG_OFFSET = (
    BAG_TARGET_TIME
    -
    VIDEO_TARGET_TIME
)


def encoder_at(
    encoder_data,
    bag_time,
):

    return float(
        np.interp(
            bag_time,
            encoder_data[:, 0],
            encoder_data[:, 1],
        )
    )


def trajectory_at(
    df,
    video_time,
):

    t = df[
        "video_time_s"
    ].to_numpy(
        dtype=float
    )

    result = {}

    for key in [
        "mcp_deg",
        "pip_deg",
        "dip_3point_deg",
    ]:

        result[
            key
        ] = float(
            np.interp(
                video_time,
                t,
                df[
                    key
                ].to_numpy(
                    dtype=float
                ),
            )
        )

    return result


def read_frame(
    cap,
    frame_index,
):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    ok, frame = cap.read()

    if not ok:
        raise RuntimeError(
            f"Cannot read frame {frame_index}"
        )

    return frame


def prepare_view(
    frame,
    text_lines,
):

    h, w = frame.shape[:2]

    # Tight crop containing bottle + index finger.
    x1 = int(
        0.00 * w
    )

    x2 = int(
        1.00 * w
    )

    y1 = int(
        0.10 * h
    )

    y2 = int(
        0.62 * h
    )

    crop = frame[
        y1:y2,
        x1:x2,
    ].copy()

    scale = min(
        1.8,
        1500.0
        /
        crop.shape[1],
    )

    view = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR,
    )

    # Black header.
    header_h = 145

    canvas = np.zeros(
        (
            view.shape[0]
            +
            header_h,
            view.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        header_h:,
        :,
    ] = view

    y = 28

    for line in text_lines:

        cv2.putText(
            canvas,
            line,
            (
                15,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        y += 28

    return canvas


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=Path,
        default=(
            Path.home()
            /
            "Downloads"
            /
            "14.mp4"
        ),
    )

    parser.add_argument(
        "--bag",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_system_id"
            /
            "index_contact_20260911_164310"
        ),
    )

    parser.add_argument(
        "--trajectory",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g22a3_dip_3point"
            /
            "g22a3_h1_dip_3point.csv"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g25a_real_first_middle_contact"
        ),
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

    if not video.exists():

        raise FileNotFoundError(
            video
        )

    if not trajectory.exists():

        raise FileNotFoundError(
            trajectory
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

    df = pd.read_csv(
        trajectory
    )

    cap = cv2.VideoCapture(
        str(
            video
        )
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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # Start around the region where visual contact develops.
    frame_index = int(
        round(
            46.0
            *
            fps
        )
    )

    min_frame = int(
        round(
            44.5
            *
            fps
        )
    )

    max_frame = int(
        round(
            50.0
            *
            fps
        )
    )

    min_frame = max(
        0,
        min_frame,
    )

    max_frame = min(
        total_frames
        -
        1,
        max_frame,
    )

    window = (
        "G25A select REAL first middle contact"
    )

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL
        |
        cv2.WINDOW_KEEPRATIO,
    )

    cv2.resizeWindow(
        window,
        1200,
        900,
    )

    selected = None

    print()
    print("=" * 120)
    print(
        "G25A — SELECT REAL FIRST MIDDLE-PHALANX CONTACT"
    )
    print("=" * 120)

    print()
    print(
        "Look ONLY at the MIDDLE phalanx "
        "(orange-marked segment)."
    )

    print()
    print(
        "Select the EARLIEST frame where the physical "
        "middle-phalanx surface touches the bottle."
    )

    print()
    print(
        "Ignore the GREEN distal segment."
    )

    print()
    print(
        "Controls:"
    )

    print(
        "  A / left-arrow   = previous frame"
    )

    print(
        "  D / right-arrow  = next frame"
    )

    print(
        "  Z                = -5 frames"
    )

    print(
        "  C                = +5 frames"
    )

    print(
        "  ENTER / SPACE    = accept THIS frame"
    )

    print(
        "  Q / ESC          = quit"
    )

    print()

    try:

        while True:

            frame = read_frame(
                cap,
                frame_index,
            )

            video_time = (
                frame_index
                /
                fps
            )

            bag_time = (
                video_time
                +
                VIDEO_TO_BAG_OFFSET
            )

            encoder = encoder_at(
                encoder_data,
                bag_time,
            )

            q = trajectory_at(
                df,
                video_time,
            )

            lines = [
                (
                    "Find EARLIEST middle-phalanx "
                    "touch with bottle"
                ),

                (
                    f"frame={frame_index}   "
                    f"video={video_time:.4f}s   "
                    f"encoder~{encoder:.1f}"
                ),

                (
                    "estimated H1 pose: "
                    f"MCP={q['mcp_deg']:.2f}  "
                    f"PIP={q['pip_deg']:.2f}  "
                    f"DIP={q['dip_3point_deg']:.2f} deg"
                ),

                (
                    "A/D = 1 frame   "
                    "Z/C = 5 frames   "
                    "ENTER = ACCEPT"
                ),

                (
                    "MIDDLE = orange segment. "
                    "Ignore green distal."
                ),
            ]

            view = prepare_view(
                frame,
                lines,
            )

            cv2.imshow(
                window,
                view,
            )

            key = (
                cv2.waitKey(0)
                &
                0xFF
            )

            # A or left arrow variants.
            if key in (
                ord("a"),
                81,
            ):

                frame_index = max(
                    min_frame,
                    frame_index - 1,
                )

                continue

            # D or right arrow variants.
            if key in (
                ord("d"),
                83,
            ):

                frame_index = min(
                    max_frame,
                    frame_index + 1,
                )

                continue

            if key == ord("z"):

                frame_index = max(
                    min_frame,
                    frame_index - 5,
                )

                continue

            if key == ord("c"):

                frame_index = min(
                    max_frame,
                    frame_index + 5,
                )

                continue

            if key in (
                13,
                32,
            ):

                selected = {
                    "frame":
                        frame_index,

                    "video_time_s":
                        video_time,

                    "bag_time_s":
                        bag_time,

                    "encoder_tick":
                        encoder,

                    "mcp_deg":
                        q[
                            "mcp_deg"
                        ],

                    "pip_deg":
                        q[
                            "pip_deg"
                        ],

                    "dip_deg":
                        q[
                            "dip_3point_deg"
                        ],
                }

                # Save accepted image.
                cv2.imwrite(
                    str(
                        out
                        /
                        "selected_real_first_middle_contact.jpg"
                    ),
                    frame,
                )

                break

            if key in (
                27,
                ord("q"),
            ):

                raise KeyboardInterrupt

    finally:

        cap.release()

        cv2.destroyAllWindows()

    if selected is None:

        raise RuntimeError(
            "No frame selected."
        )

    csv_path = (
        out
        /
        "g25a_real_first_middle_contact.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                selected.keys()
            ),
        )

        writer.writeheader()

        writer.writerow(
            selected
        )

    print()
    print("=" * 120)
    print(
        "G25A HARDWARE FIRST-MIDDLE-CONTACT TARGET"
    )
    print("=" * 120)

    print(
        f"frame       = "
        f"{selected['frame']}"
    )

    print(
        f"video time  = "
        f"{selected['video_time_s']:.6f} s"
    )

    print(
        f"bag time    = "
        f"{selected['bag_time_s']:.6f} s"
    )

    print(
        f"encoder     = "
        f"{selected['encoder_tick']:.3f} ticks"
    )

    print()

    print(
        "estimated physical pose:"
    )

    print(
        f"  MCP = "
        f"{selected['mcp_deg']:.3f} deg"
    )

    print(
        f"  PIP = "
        f"{selected['pip_deg']:.3f} deg"
    )

    print(
        f"  DIP = "
        f"{selected['dip_deg']:.3f} deg"
    )

    print()
    print(
        "Compare later against simulation first-middle event:"
    )

    print(
        "  baseline simulation encoder = 870.963 ticks"
    )

    print(
        "  baseline simulation pose    = "
        "16.937 / 17.145 / 13.608 deg"
    )

    print()
    print(
        f"Saved:\n  {csv_path}"
    )

    print()
    print(
        "Selected image:\n  "
        f"{out / 'selected_real_first_middle_contact.jpg'}"
    )

    print()
    print(
        f"All outputs:\n  {out}"
    )


if __name__ == "__main__":
    main()
