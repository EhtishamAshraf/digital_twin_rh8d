#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# RH8D INDEX TENDON CAD DIGITIZATION
#
# PURPOSE
#
# Extract the actual 2-D flexor tendon routing visible in the CAD cross-section.
#
# NO simulation tuning.
# NO MuJoCo modification.
# NO assumed old idx_guide_* geometry.
#
# Coordinate convention:
#
#   origin = MCP joint center
#
#   +x = MCP -> PIP direction
#
#   +y = FRONT / PALMAR side
#
# The user explicitly clicks a FRONT reference point so this convention does
# not depend on how the screenshot is rotated.
# =============================================================================


ROUTE_TASKS = [
    (
        "tendon_palm_entry",
        "TENDON 1/9: click cable centerline at palm/base entry BEFORE MCP",
        "palm",
    ),
    (
        "tendon_mcp_parent",
        "TENDON 2/9: click cable immediately BEFORE MCP bend/guide",
        "palm",
    ),
    (
        "tendon_mcp_child",
        "TENDON 3/9: click cable immediately AFTER MCP bend on proximal phalanx",
        "Index_Proximal",
    ),
    (
        "tendon_pip_parent",
        "TENDON 4/9: click cable immediately BEFORE PIP bend",
        "Index_Proximal",
    ),
    (
        "tendon_pip_child",
        "TENDON 5/9: click cable immediately AFTER PIP bend on middle phalanx",
        "Index_Middle",
    ),
    (
        "tendon_dip_parent",
        "TENDON 6/9: click cable immediately BEFORE DIP bend",
        "Index_Middle",
    ),
    (
        "tendon_dip_child",
        "TENDON 7/9: click cable immediately AFTER DIP bend on distal phalanx",
        "Index_Distal",
    ),
    (
        "tendon_distal_turn",
        "TENDON 8/9: click where cable turns from FRONT toward BACK of fingertip",
        "Index_Distal",
    ),
    (
        "tendon_distal_anchor",
        "TENDON 9/9: click the FINAL backside/dorsal tendon ANCHOR",
        "Index_Distal",
    ),
]


TASKS = [
    (
        "scale_A",
        "SCALE 1/2: click CENTER of first blue dimension circle",
        "scale",
    ),
    (
        "scale_B",
        "SCALE 2/2: click CENTER of second blue dimension circle",
        "scale",
    ),

    (
        "MCP",
        "JOINT 1/3: click MCP joint center",
        "joint",
    ),
    (
        "PIP",
        "JOINT 2/3: click PIP joint center",
        "joint",
    ),
    (
        "DIP",
        "JOINT 3/3: click DIP joint center",
        "joint",
    ),

    (
        "FRONT_REFERENCE",
        "FRONT SIDE: click any point clearly on FRONT/PALMAR side of finger",
        "reference",
    ),
] + ROUTE_TASKS


# =============================================================================
# UTILITIES
# =============================================================================


def distance_point_segment(
    p,
    a,
    b,
):

    p = np.asarray(
        p,
        dtype=float,
    )

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    ab = b - a

    denom = float(
        np.dot(
            ab,
            ab,
        )
    )

    if denom < 1e-12:

        return float(
            np.linalg.norm(
                p - a
            )
        )

    t = float(
        np.dot(
            p - a,
            ab,
        )
        /
        denom
    )

    t = min(
        1.0,
        max(
            0.0,
            t,
        ),
    )

    q = a + t * ab

    return float(
        np.linalg.norm(
            p - q
        )
    )


def min_distance_polyline(
    point,
    polyline,
):

    best = float(
        "inf"
    )

    for i in range(
        len(
            polyline
        )
        -
        1
    ):

        best = min(
            best,
            distance_point_segment(
                point,
                polyline[
                    i
                ],
                polyline[
                    i + 1
                ],
            ),
        )

    return best


def path_length(
    points,
):

    points = np.asarray(
        points,
        dtype=float,
    )

    if len(
        points
    ) < 2:

        return 0.0

    return float(
        np.sum(
            np.linalg.norm(
                np.diff(
                    points,
                    axis=0,
                ),
                axis=1,
            )
        )
    )


# =============================================================================
# INTERACTIVE DIGITIZER
# =============================================================================


class Digitizer:

    def __init__(
        self,
        image,
        window_name,
    ):

        self.image = (
            image.copy()
        )

        self.window_name = (
            window_name
        )

        self.points = []

        self.current = 0

        self.finished = False


        h, w = self.image.shape[
            :2
        ]


        # Keep coordinates exact by rendering a scaled copy and mapping
        # mouse coordinates back to original pixels.
        self.display_scale = min(
            1.0,
            1450.0 / float(
                w
            ),
            850.0 / float(
                h
            ),
        )


        self.display_size = (
            int(
                round(
                    w
                    *
                    self.display_scale
                )
            ),
            int(
                round(
                    h
                    *
                    self.display_scale
                )
            ),
        )


    def original_xy(
        self,
        x,
        y,
    ):

        return np.array(
            [
                float(
                    x
                )
                /
                self.display_scale,

                float(
                    y
                )
                /
                self.display_scale,
            ],
            dtype=float,
        )


    def mouse_callback(
        self,
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
        ):

            if (
                self.current
                >=
                len(
                    TASKS
                )
            ):

                return


            original = (
                self.original_xy(
                    x,
                    y,
                )
            )


            task_name = (
                TASKS[
                    self.current
                ][
                    0
                ]
            )


            self.points.append(
                {
                    "name":
                        task_name,

                    "xy":
                        original,
                }
            )


            self.current += 1


            if (
                self.current
                ==
                len(
                    TASKS
                )
            ):

                self.finished = True


        elif (
            event
            ==
            cv2.EVENT_RBUTTONDOWN
        ):

            if self.points:

                self.points.pop()

                self.current = max(
                    0,
                    self.current
                    -
                    1,
                )

                self.finished = False


    def lookup(
        self,
        name,
    ):

        for item in self.points:

            if (
                item[
                    "name"
                ]
                ==
                name
            ):

                return item[
                    "xy"
                ].copy()

        return None


    def draw(
        self,
    ):

        canvas = (
            self.image.copy()
        )


        # -------------------------------------------------------------
        # Draw all selected points.
        # -------------------------------------------------------------

        for index, item in enumerate(
            self.points
        ):

            xy = item[
                "xy"
            ]

            p = (
                int(
                    round(
                        xy[
                            0
                        ]
                    )
                ),
                int(
                    round(
                        xy[
                            1
                        ]
                    )
                ),
            )


            name = item[
                "name"
            ]


            if (
                name.startswith(
                    "tendon_"
                )
            ):

                color = (
                    255,
                    0,
                    255,
                )

            elif (
                name
                in
                (
                    "MCP",
                    "PIP",
                    "DIP",
                )
            ):

                color = (
                    0,
                    255,
                    255,
                )

            elif (
                name.startswith(
                    "scale_"
                )
            ):

                color = (
                    255,
                    100,
                    0,
                )

            else:

                color = (
                    0,
                    255,
                    0,
                )


            cv2.circle(
                canvas,
                p,
                6,
                color,
                -1,
            )


            cv2.putText(
                canvas,
                name,
                (
                    p[
                        0
                    ]
                    +
                    7,
                    p[
                        1
                    ]
                    -
                    7,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )


        # -------------------------------------------------------------
        # Scale line.
        # -------------------------------------------------------------

        a = self.lookup(
            "scale_A"
        )

        bb = self.lookup(
            "scale_B"
        )


        if (
            a is not None
            and
            bb is not None
        ):

            cv2.line(
                canvas,
                tuple(
                    np.round(
                        a
                    ).astype(
                        int
                    )
                ),
                tuple(
                    np.round(
                        bb
                    ).astype(
                        int
                    )
                ),
                (
                    255,
                    100,
                    0,
                ),
                2,
            )


        # -------------------------------------------------------------
        # Joint-center chain.
        # -------------------------------------------------------------

        mcp = self.lookup(
            "MCP"
        )

        pip = self.lookup(
            "PIP"
        )

        dip = self.lookup(
            "DIP"
        )


        if (
            mcp is not None
            and
            pip is not None
        ):

            cv2.line(
                canvas,
                tuple(
                    np.round(
                        mcp
                    ).astype(
                        int
                    )
                ),
                tuple(
                    np.round(
                        pip
                    ).astype(
                        int
                    )
                ),
                (
                    0,
                    255,
                    255,
                ),
                2,
            )


        if (
            pip is not None
            and
            dip is not None
        ):

            cv2.line(
                canvas,
                tuple(
                    np.round(
                        pip
                    ).astype(
                        int
                    )
                ),
                tuple(
                    np.round(
                        dip
                    ).astype(
                        int
                    )
                ),
                (
                    0,
                    255,
                    255,
                ),
                2,
            )


        # -------------------------------------------------------------
        # Tendon polyline.
        # -------------------------------------------------------------

        route = [
            item[
                "xy"
            ]
            for item in
            self.points
            if item[
                "name"
            ].startswith(
                "tendon_"
            )
        ]


        if len(
            route
        ) >= 2:

            for i in range(
                len(
                    route
                )
                -
                1
            ):

                cv2.line(
                    canvas,
                    tuple(
                        np.round(
                            route[
                                i
                            ]
                        ).astype(
                            int
                        )
                    ),
                    tuple(
                        np.round(
                            route[
                                i + 1
                            ]
                        ).astype(
                            int
                        )
                    ),
                    (
                        255,
                        0,
                        255,
                    ),
                    3,
                )


        # -------------------------------------------------------------
        # Instruction header.
        # -------------------------------------------------------------

        h, w = canvas.shape[
            :2
        ]


        overlay = (
            canvas.copy()
        )


        cv2.rectangle(
            overlay,
            (
                0,
                0,
            ),
            (
                w,
                78,
            ),
            (
                0,
                0,
                0,
            ),
            -1,
        )


        canvas = cv2.addWeighted(
            overlay,
            0.75,
            canvas,
            0.25,
            0,
        )


        if (
            self.current
            <
            len(
                TASKS
            )
        ):

            instruction = (
                TASKS[
                    self.current
                ][
                    1
                ]
            )

        else:

            instruction = (
                "ALL POINTS SELECTED - "
                "ENTER=save   RIGHT CLICK=undo   Q=abort"
            )


        cv2.putText(
            canvas,
            instruction,
            (
                15,
                28,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (
                255,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )


        cv2.putText(
            canvas,
            (
                "LEFT CLICK = select    "
                "RIGHT CLICK = undo last    "
                "Q = abort"
            ),
            (
                15,
                58,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (
                220,
                220,
                220,
            ),
            1,
            cv2.LINE_AA,
        )


        display = cv2.resize(
            canvas,
            self.display_size,
            interpolation=cv2.INTER_AREA,
        )


        return display


    def run(
        self,
    ):

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_AUTOSIZE,
        )


        cv2.setMouseCallback(
            self.window_name,
            self.mouse_callback,
        )


        while True:

            cv2.imshow(
                self.window_name,
                self.draw(),
            )


            key = (
                cv2.waitKey(
                    20
                )
                &
                0xFF
            )


            if (
                key
                in
                (
                    ord(
                        "q"
                    ),
                    27,
                )
            ):

                cv2.destroyAllWindows()

                raise KeyboardInterrupt(
                    "Digitization aborted."
                )


            if (
                key
                in
                (
                    8,
                    127,
                )
            ):

                if self.points:

                    self.points.pop()

                    self.current = max(
                        0,
                        self.current
                        -
                        1,
                    )

                    self.finished = False


            if (
                key
                in
                (
                    10,
                    13,
                )
                and
                self.finished
            ):

                break


        cv2.destroyAllWindows()

        return self.points


# =============================================================================
# ANALYSIS
# =============================================================================


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "image",
        type=Path,
    )


    parser.add_argument(
        "--scale-mm",
        type=float,
        default=25.875,
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "cad_tendon_digitization_v1"
        ),
    )


    args = parser.parse_args()


    image_path = (
        args.image
        .expanduser()
        .resolve()
    )


    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    image = cv2.imread(
        str(
            image_path
        )
    )


    if image is None:

        raise RuntimeError(
            f"Could not read image:\n{image_path}"
        )


    print()
    print("=" * 100)
    print(
        "RH8D INDEX TENDON CAD DIGITIZATION"
    )
    print("=" * 100)

    print()
    print(
        "Scale dimension = "
        f"{args.scale_mm:.3f} mm"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  FRONT/PALMAR side will be defined explicitly by your FRONT_REFERENCE click."
    )

    print(
        "  Click the YELLOW tendon CENTERLINE, not the edges of the channel."
    )

    print(
        "  Final tendon point must be the backside/dorsal distal ANCHOR."
    )

    print()
    print(
        "Right-click if you make a mistake."
    )

    print(
        "After all clicks, press ENTER to save."
    )


    digitizer = Digitizer(
        image,
        "RH8D CAD Tendon Digitization",
    )


    points = digitizer.run()


    raw = {
        item[
            "name"
        ]:
            np.asarray(
                item[
                    "xy"
                ],
                dtype=float,
            )
        for item in points
    }


    # =========================================================================
    # SCALE
    # =========================================================================

    scale_px = float(
        np.linalg.norm(
            raw[
                "scale_B"
            ]
            -
            raw[
                "scale_A"
            ]
        )
    )


    if scale_px < 20:

        raise RuntimeError(
            "Scale clicks are too close. "
            "Re-run and click the two blue circle centers."
        )


    mm_per_px = (
        float(
            args.scale_mm
        )
        /
        scale_px
    )


    px_per_mm = (
        1.0
        /
        mm_per_px
    )


    # =========================================================================
    # FINGER REFERENCE FRAME
    # =========================================================================

    mcp = raw[
        "MCP"
    ]

    pip = raw[
        "PIP"
    ]

    dip = raw[
        "DIP"
    ]


    xhat = (
        pip
        -
        mcp
    )


    xnorm = float(
        np.linalg.norm(
            xhat
        )
    )


    if xnorm < 20:

        raise RuntimeError(
            "MCP and PIP clicks are too close."
        )


    xhat /= xnorm


    # Candidate perpendicular.
    yhat = np.array(
        [
            -xhat[
                1
            ],
            xhat[
                0
            ],
        ],
        dtype=float,
    )


    front_vector = (
        raw[
            "FRONT_REFERENCE"
        ]
        -
        mcp
    )


    if (
        np.dot(
            front_vector,
            yhat,
        )
        <
        0
    ):

        yhat = (
            -yhat
        )


    def to_finger_mm(
        pixel_xy,
    ):

        v = (
            np.asarray(
                pixel_xy,
                dtype=float,
            )
            -
            mcp
        )


        return np.array(
            [
                np.dot(
                    v,
                    xhat,
                )
                *
                mm_per_px,

                np.dot(
                    v,
                    yhat,
                )
                *
                mm_per_px,
            ],
            dtype=float,
        )


    # =========================================================================
    # CONVERT EVERYTHING
    # =========================================================================

    joint_mm = {
        name:
            to_finger_mm(
                raw[
                    name
                ]
            )
        for name in (
            "MCP",
            "PIP",
            "DIP",
        )
    }


    route_rows = []


    route_mm = []


    body_lookup = {
        name:
            body
        for (
            name,
            instruction,
            body
        ) in ROUTE_TASKS
    }


    for order, (
        name,
        instruction,
        body,
    ) in enumerate(
        ROUTE_TASKS,
        start=1,
    ):

        px = raw[
            name
        ]

        mm = to_finger_mm(
            px
        )


        route_mm.append(
            mm
        )


        route_rows.append(
            {
                "order":
                    order,

                "name":
                    name,

                "body":
                    body,

                "u_px":
                    float(
                        px[
                            0
                        ]
                    ),

                "v_px":
                    float(
                        px[
                            1
                        ]
                    ),

                "x_mm_from_MCP":
                    float(
                        mm[
                            0
                        ]
                    ),

                "y_mm_front_positive":
                    float(
                        mm[
                            1
                        ]
                    ),
            }
        )


    route_mm = np.asarray(
        route_mm,
        dtype=float,
    )


    # =========================================================================
    # GEOMETRIC CHECKS
    # =========================================================================

    mcp_pip_mm = float(
        np.linalg.norm(
            joint_mm[
                "PIP"
            ]
            -
            joint_mm[
                "MCP"
            ]
        )
    )


    pip_dip_mm = float(
        np.linalg.norm(
            joint_mm[
                "DIP"
            ]
            -
            joint_mm[
                "PIP"
            ]
        )
    )


    route_length_mm = path_length(
        route_mm
    )


    distances = {}


    for name in (
        "MCP",
        "PIP",
        "DIP",
    ):

        distances[
            name
        ] = min_distance_polyline(
            joint_mm[
                name
            ],
            route_mm,
        )


    anchor_mm = to_finger_mm(
        raw[
            "tendon_distal_anchor"
        ]
    )


    turn_mm = to_finger_mm(
        raw[
            "tendon_distal_turn"
        ]
    )


    # =========================================================================
    # SAVE CSV
    # =========================================================================

    route_csv = (
        output_dir
        /
        "cad_tendon_route_points.csv"
    )


    with open(
        route_csv,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "order",
                "name",
                "body",
                "u_px",
                "v_px",
                "x_mm_from_MCP",
                "y_mm_front_positive",
            ],
        )

        writer.writeheader()

        writer.writerows(
            route_rows
        )


    joint_csv = (
        output_dir
        /
        "cad_joint_centers.csv"
    )


    with open(
        joint_csv,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )


        writer.writerow(
            [
                "joint",
                "u_px",
                "v_px",
                "x_mm_from_MCP",
                "y_mm_front_positive",
            ]
        )


        for name in (
            "MCP",
            "PIP",
            "DIP",
        ):

            writer.writerow(
                [
                    name,
                    raw[
                        name
                    ][
                        0
                    ],
                    raw[
                        name
                    ][
                        1
                    ],
                    joint_mm[
                        name
                    ][
                        0
                    ],
                    joint_mm[
                        name
                    ][
                        1
                    ],
                ]
            )


    # =========================================================================
    # SAVE JSON
    # =========================================================================

    result = {
        "source_image":
            str(
                image_path
            ),

        "known_scale_mm":
            float(
                args.scale_mm
            ),

        "scale_pixel_distance":
            scale_px,

        "mm_per_px":
            mm_per_px,

        "px_per_mm":
            px_per_mm,

        "coordinate_frame":
            {
                "origin":
                    "MCP",

                "x_axis":
                    "MCP_to_PIP",

                "y_axis":
                    "front_palmar_positive",

                "xhat_image":
                    xhat.tolist(),

                "yhat_image":
                    yhat.tolist(),
            },

        "joint_centers_mm":
            {
                name:
                    joint_mm[
                        name
                    ].tolist()
                for name in (
                    "MCP",
                    "PIP",
                    "DIP",
                )
            },

        "joint_spacing_mm":
            {
                "MCP_to_PIP":
                    mcp_pip_mm,

                "PIP_to_DIP":
                    pip_dip_mm,
            },

        "tendon_route":
            route_rows,

        "route_length_2d_mm":
            route_length_mm,

        "static_min_distance_to_joint_mm":
            distances,

        "distal_turn_mm":
            turn_mm.tolist(),

        "distal_anchor_mm":
            anchor_mm.tolist(),
    }


    json_path = (
        output_dir
        /
        "cad_tendon_geometry.json"
    )


    with open(
        json_path,
        "w",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )


    # =========================================================================
    # CREATE FINAL OVERLAY
    # =========================================================================

    overlay = (
        image.copy()
    )


    # Scale
    cv2.line(
        overlay,
        tuple(
            np.round(
                raw[
                    "scale_A"
                ]
            ).astype(
                int
            )
        ),
        tuple(
            np.round(
                raw[
                    "scale_B"
                ]
            ).astype(
                int
            )
        ),
        (
            255,
            100,
            0,
        ),
        3,
    )


    # Joints
    for name in (
        "MCP",
        "PIP",
        "DIP",
    ):

        p = tuple(
            np.round(
                raw[
                    name
                ]
            ).astype(
                int
            )
        )


        cv2.circle(
            overlay,
            p,
            9,
            (
                0,
                255,
                255,
            ),
            2,
        )


        cv2.putText(
            overlay,
            name,
            (
                p[
                    0
                ]
                +
                10,
                p[
                    1
                ]
                -
                10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                0,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )


    # Route
    route_px = [
        raw[
            item[
                0
            ]
        ]
        for item in
        ROUTE_TASKS
    ]


    for i in range(
        len(
            route_px
        )
        -
        1
    ):

        cv2.line(
            overlay,
            tuple(
                np.round(
                    route_px[
                        i
                    ]
                ).astype(
                    int
                )
            ),
            tuple(
                np.round(
                    route_px[
                        i + 1
                    ]
                ).astype(
                    int
                )
            ),
            (
                255,
                0,
                255,
            ),
            4,
        )


    for i, (
        name,
        instruction,
        body,
    ) in enumerate(
        ROUTE_TASKS,
        start=1,
    ):

        p = tuple(
            np.round(
                raw[
                    name
                ]
            ).astype(
                int
            )
        )


        cv2.circle(
            overlay,
            p,
            6,
            (
                255,
                0,
                255,
            ),
            -1,
        )


        cv2.putText(
            overlay,
            str(
                i
            ),
            (
                p[
                    0
                ]
                +
                7,
                p[
                    1
                ]
                -
                7,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                255,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )


    overlay_path = (
        output_dir
        /
        "cad_tendon_digitization_overlay.png"
    )


    cv2.imwrite(
        str(
            overlay_path
        ),
        overlay,
    )


    # =========================================================================
    # NORMALIZED MM-PLOT IMAGE
    # =========================================================================

    margin = 10.0


    all_xy = np.vstack(
        [
            route_mm,
            joint_mm[
                "MCP"
            ],
            joint_mm[
                "PIP"
            ],
            joint_mm[
                "DIP"
            ],
        ]
    )


    xmin = float(
        np.min(
            all_xy[
                :,
                0
            ]
        )
        -
        margin
    )

    xmax = float(
        np.max(
            all_xy[
                :,
                0
            ]
        )
        +
        margin
    )

    ymin = float(
        np.min(
            all_xy[
                :,
                1
            ]
        )
        -
        margin
    )

    ymax = float(
        np.max(
            all_xy[
                :,
                1
            ]
        )
        +
        margin
    )


    width = 1400

    height = 700


    plot = np.full(
        (
            height,
            width,
            3,
        ),
        255,
        dtype=np.uint8,
    )


    def mm_to_plot(
        p,
    ):

        x = (
            (
                p[
                    0
                ]
                -
                xmin
            )
            /
            max(
                xmax
                -
                xmin,
                1e-9,
            )
        )


        # Reverse for image display so +front is visually upward.
        y = (
            (
                ymax
                -
                p[
                    1
                ]
            )
            /
            max(
                ymax
                -
                ymin,
                1e-9,
            )
        )


        return (
            int(
                round(
                    60
                    +
                    x
                    *
                    (
                        width
                        -
                        120
                    )
                )
            ),

            int(
                round(
                    60
                    +
                    y
                    *
                    (
                        height
                        -
                        120
                    )
                )
            ),
        )


    # Axes through MCP.
    origin = mm_to_plot(
        np.array(
            [
                0.0,
                0.0,
            ]
        )
    )


    cv2.line(
        plot,
        (
            40,
            origin[
                1
            ],
        ),
        (
            width
            -
            40,
            origin[
                1
            ],
        ),
        (
            180,
            180,
            180,
        ),
        1,
    )


    cv2.line(
        plot,
        (
            origin[
                0
            ],
            40,
        ),
        (
            origin[
                0
            ],
            height
            -
            40,
        ),
        (
            180,
            180,
            180,
        ),
        1,
    )


    # Tendon
    for i in range(
        len(
            route_mm
        )
        -
        1
    ):

        cv2.line(
            plot,
            mm_to_plot(
                route_mm[
                    i
                ]
            ),
            mm_to_plot(
                route_mm[
                    i + 1
                ]
            ),
            (
                255,
                0,
                255,
            ),
            4,
        )


    for i, p in enumerate(
        route_mm,
        start=1,
    ):

        q = mm_to_plot(
            p
        )


        cv2.circle(
            plot,
            q,
            6,
            (
                255,
                0,
                255,
            ),
            -1,
        )


        cv2.putText(
            plot,
            str(
                i
            ),
            (
                q[
                    0
                ]
                +
                8,
                q[
                    1
                ]
                -
                8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                80,
                80,
                80,
            ),
            1,
            cv2.LINE_AA,
        )


    # Joints
    for name in (
        "MCP",
        "PIP",
        "DIP",
    ):

        q = mm_to_plot(
            joint_mm[
                name
            ]
        )


        cv2.circle(
            plot,
            q,
            9,
            (
                0,
                180,
                220,
            ),
            2,
        )


        cv2.putText(
            plot,
            name,
            (
                q[
                    0
                ]
                +
                10,
                q[
                    1
                ]
                +
                20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                40,
                40,
                40,
            ),
            2,
            cv2.LINE_AA,
        )


    cv2.putText(
        plot,
        "+x = distal direction",
        (
            40,
            30,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (
            30,
            30,
            30,
        ),
        2,
        cv2.LINE_AA,
    )


    cv2.putText(
        plot,
        "+y = FRONT / PALMAR",
        (
            40,
            55,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (
            30,
            30,
            30,
        ),
        2,
        cv2.LINE_AA,
    )


    normalized_path = (
        output_dir
        /
        "cad_tendon_geometry_mm.png"
    )


    cv2.imwrite(
        str(
            normalized_path
        ),
        plot,
    )


    # =========================================================================
    # TEXT SUMMARY
    # =========================================================================

    summary_path = (
        output_dir
        /
        "cad_tendon_summary.txt"
    )


    lines = []

    lines.append(
        "RH8D INDEX CAD TENDON DIGITIZATION"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"scale = {px_per_mm:.6f} px/mm"
    )

    lines.append(
        f"scale = {mm_per_px:.6f} mm/px"
    )

    lines.append(
        ""
    )

    lines.append(
        f"MCP->PIP = {mcp_pip_mm:.3f} mm"
    )

    lines.append(
        f"PIP->DIP = {pip_dip_mm:.3f} mm"
    )

    lines.append(
        ""
    )

    lines.append(
        f"2-D tendon polyline length = "
        f"{route_length_mm:.3f} mm"
    )

    lines.append(
        ""
    )

    lines.append(
        "Static minimum CAD-path distance from joint centers:"
    )

    lines.append(
        f"  MCP = {distances['MCP']:.3f} mm"
    )

    lines.append(
        f"  PIP = {distances['PIP']:.3f} mm"
    )

    lines.append(
        f"  DIP = {distances['DIP']:.3f} mm"
    )

    lines.append(
        ""
    )

    lines.append(
        "Distal turn [x, y_front+] mm:"
    )

    lines.append(
        f"  [{turn_mm[0]:.3f}, {turn_mm[1]:.3f}]"
    )

    lines.append(
        ""
    )

    lines.append(
        "Distal anchor [x, y_front+] mm:"
    )

    lines.append(
        f"  [{anchor_mm[0]:.3f}, {anchor_mm[1]:.3f}]"
    )

    lines.append(
        ""
    )

    lines.append(
        "ROUTE:"
    )


    for row in route_rows:

        lines.append(
            f"  {row['order']:02d} "
            f"{row['name']:<24s} "
            f"body={row['body']:<16s} "
            f"x={row['x_mm_from_MCP']:+8.3f} mm "
            f"y_front={row['y_mm_front_positive']:+8.3f} mm"
        )


    summary = "\n".join(
        lines
    )


    summary_path.write_text(
        summary
    )


    print()
    print("=" * 100)
    print(
        "DIGITIZATION RESULT"
    )
    print("=" * 100)

    print()
    print(
        summary
    )


    print()
    print("=" * 100)
    print(
        "FILES WRITTEN"
    )
    print("=" * 100)


    for p in (
        route_csv,
        joint_csv,
        json_path,
        overlay_path,
        normalized_path,
        summary_path,
    ):

        print(
            f"  {p}"
        )


    print()
    print(
        "FINAL VISUAL CHECK:"
    )

    print(
        "  Magenta polyline must lie on the yellow CAD tendon."
    )

    print(
        "  Point 8 must be where tendon turns toward the BACK."
    )

    print(
        "  Point 9 must be the backside/distal ANCHOR."
    )

    print()


    # Leave final overlay open for visual verification.
    preview = cv2.resize(
        overlay,
        (
            int(
                overlay.shape[
                    1
                ]
                *
                min(
                    1.0,
                    1450.0
                    /
                    overlay.shape[
                        1
                    ],
                )
            ),
            int(
                overlay.shape[
                    0
                ]
                *
                min(
                    1.0,
                    1450.0
                    /
                    overlay.shape[
                        1
                    ],
                )
            ),
        ),
        interpolation=cv2.INTER_AREA,
    )


    cv2.imshow(
        "FINAL CAD DIGITIZATION - press any key to close",
        preview,
    )

    cv2.waitKey(
        0
    )

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
