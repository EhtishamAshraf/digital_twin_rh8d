#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(HERE),
)

import marker_ANGLE_Extractor as marker


VIDEO = Path(
    "/home/ehtisham/Downloads/13.mp4"
)

TIMESTAMP_CSV = (
    HERE
    / "results"
    / "index_free_video_times_v1.csv"
)

OUT_RAW = (
    HERE
    / "results"
    / "index_free_marker_axes_manual_v1.csv"
)

OUT_FINAL = (
    HERE
    / "results"
    / "index_free_hardware_angles_v1.csv"
)


# ================================================================
# EXISTING TRUSTED FREE-SPACE HARDWARE MCP/PIP MEASUREMENTS
#
# These are used ONLY to:
#   1. determine the global image-angle sign
#   2. cross-check the newly clicked axes
#
# They are NOT used to construct DIP.
# ================================================================

OLD_HW = {

    "open_100":
        (0.00, 0.00),

    "closing_500":
        (13.44, 5.55),

    "closing_1000":
        (30.85, 10.52),

    "closing_1500":
        (46.46, 15.50),

    "closing_2000":
        (58.88, 26.67),

    "closing_2500":
        (59.85, 41.75),

    "closing_3000":
        (60.97, 62.23),

    "closing_3500":
        (64.35, 75.27),

    "opening_3000":
        (52.24, 69.47),

    "opening_2500":
        (46.77, 52.50),

    "opening_2000":
        (37.93, 49.78),

    "opening_1500":
        (37.92, 31.35),

    "opening_1000":
        (34.04, 12.98),

    "opening_500":
        (14.58, 7.93),

    "opening_100":
        (0.00, 0.00),
}


def wrap180_axis(
    angle,
):

    """
    Marker axes are bidirectional.

    Equivalent orientations separated by 180 degrees
    represent the same marker axis.

    Return [-90, 90).
    """

    return marker.wrap_axis(
        float(angle)
    )


def axis_difference(
    a,
    b,
):

    """
    Difference between two bidirectional marker axes.
    """

    return wrap180_axis(
        float(a)
        -
        float(b)
    )


def joint_from_axes(
    axes,
):

    """
    Image-plane joint quantities BEFORE open-state zeroing.

    MCP:
        proximal marker orientation.

    PIP:
        middle relative to proximal.

    DIP:
        distal relative to middle.
    """

    prox = float(
        axes["proximal"]
    )

    mid = float(
        axes["middle"]
    )

    dist = float(
        axes["distal"]
    )

    return {

        "MCP":
            wrap180_axis(
                prox
            ),

        "PIP":
            axis_difference(
                mid,
                prox,
            ),

        "DIP":
            axis_difference(
                dist,
                mid,
            ),
    }


def zero_relative(
    current,
    reference,
):

    """
    Joint motion relative to the initial OPEN 100-tick frame.
    """

    return {

        key:
            axis_difference(
                current[key],
                reference[key],
            )

        for key in (
            "MCP",
            "PIP",
            "DIP",
        )
    }


def read_frame_exact(
    cap,
    frame_number,
):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_number),
    )

    ok, frame = cap.read()

    if not ok:

        raise RuntimeError(
            f"Could not read frame {frame_number}"
        )

    actual = int(
        round(
            cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        )
    ) - 1

    if abs(
        actual
        -
        int(frame_number)
    ) > 1:

        print(
            f"WARNING: requested frame={frame_number}, "
            f"decoder reported frame={actual}"
        )

    return frame


def parse_phase_ticks(
    label,
):

    if label == "open_100":

        return (
            "initial_open",
            100,
        )

    phase, ticks = (
        label.rsplit(
            "_",
            1,
        )
    )

    return (
        phase,
        int(ticks),
    )


def rmse(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                values**2
            )
        )
    )


def main():

    if not VIDEO.exists():

        raise RuntimeError(
            f"Missing video:\n{VIDEO}"
        )

    if not TIMESTAMP_CSV.exists():

        raise RuntimeError(
            f"Missing timestamp CSV:\n"
            f"{TIMESTAMP_CSV}"
        )


    # ============================================================
    # Load user-selected frames.
    # ============================================================

    with TIMESTAMP_CSV.open(
        "r",
        newline="",
    ) as f:

        timestamp_rows = list(
            csv.DictReader(
                f
            )
        )


    if not timestamp_rows:

        raise RuntimeError(
            "Timestamp CSV is empty."
        )


    if timestamp_rows[0][
        "label"
    ] != "open_100":

        raise RuntimeError(
            "First timestamp must be open_100."
        )


    cap = cv2.VideoCapture(
        str(VIDEO)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open:\n{VIDEO}"
        )


    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )


    print()
    print("=" * 100)
    print(
        "RH8D INDEX — FREE-SPACE MANUAL "
        "MCP/PIP/DIP MARKER MEASUREMENT V1"
    )
    print("=" * 100)

    print()
    print(
        f"Video = {VIDEO}"
    )

    print(
        f"FPS   = {fps:.6f}"
    )

    print(
        f"Frames to measure = "
        f"{len(timestamp_rows)}"
    )

    print()
    print(
        "For EACH marker:"
    )

    print(
        "  click TWO points along its long axis"
    )

    print(
        "  ENTER or SPACE = accept"
    )

    print(
        "  R = reset the two clicks"
    )

    print(
        "  Q / ESC = abort"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Click along the actual colored marker axis, "
        "not along the finger silhouette."
    )

    print(
        "The order of the two clicked points does not matter."
    )

    print()


    measured = []


    try:

        for i, row in enumerate(
            timestamp_rows,
            start=1,
        ):

            label = row[
                "label"
            ]

            frame_number = int(
                row[
                    "frame"
                ]
            )

            time_s = float(
                row[
                    "time_s"
                ]
            )


            print()
            print("-" * 100)

            print(
                f"[{i:02d}/"
                f"{len(timestamp_rows):02d}] "
                f"{label}"
            )

            print(
                f"frame={frame_number}, "
                f"time={time_s:.3f} s"
            )

            print("-" * 100)


            frame = read_frame_exact(
                cap,
                frame_number,
            )


            axes = marker.measure_frame(
                frame,
                (
                    f"{label} | "
                    f"frame {frame_number}"
                ),
            )


            joint_absolute = (
                joint_from_axes(
                    axes
                )
            )


            measured.append(
                {
                    "label":
                        label,

                    "frame":
                        frame_number,

                    "time_s":
                        time_s,

                    "proximal_axis_deg":
                        float(
                            axes[
                                "proximal"
                            ]
                        ),

                    "middle_axis_deg":
                        float(
                            axes[
                                "middle"
                            ]
                        ),

                    "distal_axis_deg":
                        float(
                            axes[
                                "distal"
                            ]
                        ),

                    "joint_absolute":
                        joint_absolute,
                }
            )


    finally:

        cap.release()

        cv2.destroyAllWindows()


    if len(measured) != len(
        timestamp_rows
    ):

        raise RuntimeError(
            "Measurement aborted before "
            "all frames were completed."
        )


    # ============================================================
    # Initial OPEN frame defines zero.
    # ============================================================

    reference = measured[
        0
    ][
        "joint_absolute"
    ]


    for row in measured:

        row[
            "raw_relative"
        ] = zero_relative(
            row[
                "joint_absolute"
            ],
            reference,
        )


    # ============================================================
    # Determine ONE global flexion sign.
    #
    # DIP is NOT used here.
    #
    # We choose between +1 and -1 using only the previously
    # measured hardware MCP/PIP values.
    # ============================================================

    sign_results = {}


    for sign in (
        +1.0,
        -1.0,
    ):

        mcp_errors = []
        pip_errors = []


        for row in measured:

            label = row[
                "label"
            ]

            if label not in OLD_HW:
                continue


            hw_mcp, hw_pip = (
                OLD_HW[
                    label
                ]
            )


            raw = row[
                "raw_relative"
            ]


            mcp = (
                sign
                *
                raw["MCP"]
            )

            pip = (
                sign
                *
                raw["PIP"]
            )


            mcp_errors.append(
                mcp
                -
                hw_mcp
            )

            pip_errors.append(
                pip
                -
                hw_pip
            )


        mcp_rmse = rmse(
            mcp_errors
        )

        pip_rmse = rmse(
            pip_errors
        )

        combined = math.sqrt(
            (
                mcp_rmse**2
                +
                pip_rmse**2
            )
            /
            2.0
        )


        sign_results[
            sign
        ] = {
            "mcp_rmse":
                mcp_rmse,

            "pip_rmse":
                pip_rmse,

            "combined":
                combined,
        }


    selected_sign = min(
        sign_results,
        key=lambda s:
            sign_results[
                s
            ][
                "combined"
            ],
    )


    print()
    print("=" * 100)
    print(
        "GLOBAL SIGN CROSS-CHECK"
    )
    print("=" * 100)

    print()

    for sign in (
        +1.0,
        -1.0,
    ):

        r = sign_results[
            sign
        ]

        print(
            f"sign={sign:+.0f}: "
            f"MCP RMSE={r['mcp_rmse']:.3f} deg, "
            f"PIP RMSE={r['pip_rmse']:.3f} deg, "
            f"combined={r['combined']:.3f} deg"
        )


    print()
    print(
        f"SELECTED GLOBAL SIGN = "
        f"{selected_sign:+.0f}"
    )

    print(
        "NOTE: DIP was NOT used to select this sign."
    )


    # ============================================================
    # Build final hardware table.
    # ============================================================

    output_rows = []

    mcp_errors = []
    pip_errors = []


    for row in measured:

        label = row[
            "label"
        ]

        phase, ticks = (
            parse_phase_ticks(
                label
            )
        )


        raw = row[
            "raw_relative"
        ]


        mcp = (
            selected_sign
            *
            raw["MCP"]
        )

        pip = (
            selected_sign
            *
            raw["PIP"]
        )

        dip = (
            selected_sign
            *
            raw["DIP"]
        )


        hw_mcp_old = np.nan
        hw_pip_old = np.nan
        err_mcp = np.nan
        err_pip = np.nan


        if label in OLD_HW:

            (
                hw_mcp_old,
                hw_pip_old,
            ) = OLD_HW[
                label
            ]


            err_mcp = (
                mcp
                -
                hw_mcp_old
            )

            err_pip = (
                pip
                -
                hw_pip_old
            )


            mcp_errors.append(
                err_mcp
            )

            pip_errors.append(
                err_pip
            )


        output_rows.append(
            {
                "label":
                    label,

                "phase":
                    phase,

                "target_ticks":
                    ticks,

                "frame":
                    row["frame"],

                "time_s":
                    row["time_s"],

                "hardware_mcp_manual_deg":
                    mcp,

                "hardware_pip_manual_deg":
                    pip,

                "hardware_dip_manual_deg":
                    dip,

                "old_hardware_mcp_deg":
                    hw_mcp_old,

                "old_hardware_pip_deg":
                    hw_pip_old,

                "mcp_crosscheck_error_deg":
                    err_mcp,

                "pip_crosscheck_error_deg":
                    err_pip,

                "proximal_axis_raw_deg":
                    row[
                        "proximal_axis_deg"
                    ],

                "middle_axis_raw_deg":
                    row[
                        "middle_axis_deg"
                    ],

                "distal_axis_raw_deg":
                    row[
                        "distal_axis_deg"
                    ],
            }
        )


    # ============================================================
    # Save raw axis data.
    # ============================================================

    OUT_RAW.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with OUT_RAW.open(
        "w",
        newline="",
    ) as f:

        fieldnames = [
            "label",
            "frame",
            "time_s",
            "proximal_axis_deg",
            "middle_axis_deg",
            "distal_axis_deg",
            "raw_mcp_zeroed_deg",
            "raw_pip_zeroed_deg",
            "raw_dip_zeroed_deg",
        ]


        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()


        for row in measured:

            raw = row[
                "raw_relative"
            ]

            writer.writerow(
                {
                    "label":
                        row["label"],

                    "frame":
                        row["frame"],

                    "time_s":
                        row["time_s"],

                    "proximal_axis_deg":
                        row[
                            "proximal_axis_deg"
                        ],

                    "middle_axis_deg":
                        row[
                            "middle_axis_deg"
                        ],

                    "distal_axis_deg":
                        row[
                            "distal_axis_deg"
                        ],

                    "raw_mcp_zeroed_deg":
                        raw["MCP"],

                    "raw_pip_zeroed_deg":
                        raw["PIP"],

                    "raw_dip_zeroed_deg":
                        raw["DIP"],
                }
            )


    # ============================================================
    # Save final MCP/PIP/DIP hardware table.
    # ============================================================

    with OUT_FINAL.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                output_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )


    # ============================================================
    # Final report.
    # ============================================================

    new_mcp_rmse = rmse(
        mcp_errors
    )

    new_pip_rmse = rmse(
        pip_errors
    )

    new_combined = math.sqrt(
        (
            new_mcp_rmse**2
            +
            new_pip_rmse**2
        )
        /
        2.0
    )


    print()
    print("=" * 126)
    print(
        "RECOVERED FREE-SPACE HARDWARE JOINT TABLE"
    )
    print("=" * 126)

    print()

    print(
        "PHASE            TICKS    "
        "MCP_NEW   PIP_NEW   DIP_NEW    "
        "MCP_OLD   PIP_OLD    "
        "dMCP      dPIP"
    )

    print(
        "-" * 126
    )


    for row in output_rows:

        print(
            f"{row['phase']:<16s} "
            f"{row['target_ticks']:5d} "
            f"{row['hardware_mcp_manual_deg']:9.3f} "
            f"{row['hardware_pip_manual_deg']:9.3f} "
            f"{row['hardware_dip_manual_deg']:9.3f} "
            f"{row['old_hardware_mcp_deg']:9.3f} "
            f"{row['old_hardware_pip_deg']:9.3f} "
            f"{row['mcp_crosscheck_error_deg']:+9.3f} "
            f"{row['pip_crosscheck_error_deg']:+9.3f}"
        )


    print()
    print("=" * 100)
    print(
        "MANUAL-MARKER CROSS-CHECK AGAINST OLD MCP/PIP"
    )
    print("=" * 100)

    print()

    print(
        f"MCP RMSE      = "
        f"{new_mcp_rmse:.3f} deg"
    )

    print(
        f"PIP RMSE      = "
        f"{new_pip_rmse:.3f} deg"
    )

    print(
        f"combined RMSE = "
        f"{new_combined:.3f} deg"
    )


    print()
    print("Saved raw axes:")
    print(
        OUT_RAW.resolve()
    )

    print()
    print(
        "Saved recovered hardware MCP/PIP/DIP:"
    )

    print(
        OUT_FINAL.resolve()
    )


    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT use the recovered DIP for fitting yet "
        "until the MCP/PIP cross-check and DIP trajectory "
        "have been inspected."
    )

    print()
    print("=" * 100)
    print(
        "MEASUREMENT COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
