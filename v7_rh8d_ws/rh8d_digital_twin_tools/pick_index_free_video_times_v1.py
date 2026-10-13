#!/usr/bin/env python3

import cv2
import csv
from pathlib import Path


VIDEO = Path(
    "/home/ehtisham/Downloads/13.mp4"
)


LABELS = [
    "open_100",

    "closing_500",
    "closing_1000",
    "closing_1500",
    "closing_2000",
    "closing_2500",
    "closing_3000",
    "closing_3500",

    "opening_3000",
    "opening_2500",
    "opening_2000",
    "opening_1500",
    "opening_1000",
    "opening_500",
    "opening_100",
]


OUT = Path(
    "rh8d_digital_twin_tools/results/"
    "index_free_video_times_v1.csv"
)


def main():

    cap = cv2.VideoCapture(
        str(VIDEO)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {VIDEO}"
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
        else 0.0
    )

    print()
    print("VIDEO")
    print("  path     =", VIDEO)
    print("  fps      =", fps)
    print("  frames   =", nframes)
    print("  duration =", duration, "s")

    print()
    print("CONTROLS")
    print("  A / D        : previous / next frame")
    print("  J / L        : -10 / +10 frames")
    print("  U / O        : -100 / +100 frames")
    print("  ENTER/SPACE  : save current frame for current label")
    print("  R            : redo previous label")
    print("  Q / ESC      : quit")
    print()

    records = []

    index = 0
    frame_id = 0

    window = (
        "FREE-SPACE VIDEO TIMESTAMP PICKER"
    )

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window,
        1200,
        850,
    )

    while index < len(LABELS):

        frame_id = max(
            0,
            min(
                nframes - 1,
                frame_id,
            ),
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_id,
        )

        ok, frame = cap.read()

        if not ok:
            print(
                "Could not read frame",
                frame_id,
            )
            break

        t = frame_id / fps

        display = frame.copy()

        label = LABELS[index]

        cv2.putText(
            display,
            f"TARGET: {label}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            f"frame={frame_id}   time={t:.3f} s",
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            "A/D=1 frame   J/L=10   U/O=100   ENTER=save",
            (25, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            window,
            display,
        )

        key = (
            cv2.waitKey(0)
            & 0xFF
        )

        if key in (
            ord("q"),
            27,
        ):
            break

        elif key == ord("a"):
            frame_id -= 1

        elif key == ord("d"):
            frame_id += 1

        elif key == ord("j"):
            frame_id -= 10

        elif key == ord("l"):
            frame_id += 10

        elif key == ord("u"):
            frame_id -= 100

        elif key == ord("o"):
            frame_id += 100

        elif key in (
            13,
            32,
        ):

            records.append(
                {
                    "label":
                        label,

                    "frame":
                        frame_id,

                    "time_s":
                        t,
                }
            )

            print(
                f"SAVED  {label:<18s} "
                f"frame={frame_id:6d} "
                f"time={t:8.3f} s"
            )

            index += 1

        elif key == ord("r"):

            if records:

                previous = (
                    records.pop()
                )

                index = max(
                    0,
                    index - 1,
                )

                frame_id = int(
                    previous[
                        "frame"
                    ]
                )

                print(
                    "REDO:",
                    LABELS[index],
                )

    cap.release()
    cv2.destroyAllWindows()

    if len(records) != len(LABELS):

        print()
        print(
            "Incomplete:"
            f" {len(records)}/{len(LABELS)} points."
        )

        return

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUT.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "frame",
                "time_s",
            ],
        )

        writer.writeheader()
        writer.writerows(
            records
        )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print(
        OUT.resolve()
    )

    print()

    for row in records:

        print(
            f"{row['label']:<18s} "
            f"{row['frame']:6d} "
            f"{row['time_s']:8.3f} s"
        )


if __name__ == "__main__":
    main()
