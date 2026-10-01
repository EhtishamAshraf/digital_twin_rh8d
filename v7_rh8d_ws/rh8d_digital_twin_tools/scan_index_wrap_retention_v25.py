#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv

import numpy as np

import scan_index_stateful_transmission_v24 as v24


# ============================================================
# V25 — LOCAL STATEFUL TRANSMISSION REFINEMENT
#       + MIDDLE-CONTACT RETENTION
#
# V24's old "sequence OK" only checked:
#
#   first middle contact
#        <
#   transmission yield
#        <
#   first tip contact
#
# It did NOT require the middle phalanx to stay wrapped.
#
# V25 explicitly measures:
#
#   1. middle contact duration
#   2. tip contact duration
#   3. simultaneous middle + tip contact
#   4. middle contact remaining after tip first contacts
#
# Hardware target mechanism:
#
#   MCP
#    -> middle contact
#    -> PIP motion while middle remains engaged
#    -> distal/tip contact
#    -> DIP contribution
#
# Bottle position remains frozen V5.
#
# No contact parameters are changed.
# ============================================================


WS = v24.WS

MJCF_DIR = v24.MJCF_DIR

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "wrap_retention_v25"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY = (
    RESULT_DIR
    / "summary.csv"
)


# ============================================================
# HARDWARE TARGET
# ============================================================

HW = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


# ============================================================
# FIXED ARCHITECTURE FROM V24
# ============================================================

FREE_GATE = 6.085

MIN_OVERLAP_S = 0.15


# ============================================================
# SMALL LOCAL GRID
#
# V24 best:
#
#   [3.5, 9.0, 2.0]
#   -> 19.588 / 16.606 / 9.912 deg
#
# Trend suggested:
#
#   smaller MCP arm
#   larger PIP arm
#
# should move toward:
#
#   14.536 / 23.811 / 9.327 deg
#
# Keep DIP near 1.8 because V24 DIP was already close.
# ============================================================

CASES = [

    {
        "name":
            "V24_BEST",

        "yield":
            True,

        "arms_mm":
            [3.5, 9.0, 2.0],
    },

    {
        "name":
            "R3p5_10_1p8",

        "yield":
            True,

        "arms_mm":
            [3.5, 10.0, 1.8],
    },

    {
        "name":
            "R3p5_11_1p8",

        "yield":
            True,

        "arms_mm":
            [3.5, 11.0, 1.8],
    },

    {
        "name":
            "R3p5_12_1p8",

        "yield":
            True,

        "arms_mm":
            [3.5, 12.0, 1.8],
    },

    {
        "name":
            "R3p5_13_1p8",

        "yield":
            True,

        "arms_mm":
            [3.5, 13.0, 1.8],
    },

    {
        "name":
            "R3_11_1p8",

        "yield":
            True,

        "arms_mm":
            [3.0, 11.0, 1.8],
    },

    {
        "name":
            "R3_12_1p8",

        "yield":
            True,

        "arms_mm":
            [3.0, 12.0, 1.8],
    },

    {
        "name":
            "R3_13_1p8",

        "yield":
            True,

        "arms_mm":
            [3.0, 13.0, 1.8],
    },

    {
        "name":
            "R2p5_12_1p8",

        "yield":
            True,

        "arms_mm":
            [2.5, 12.0, 1.8],
    },

    {
        "name":
            "R2p5_13_1p8",

        "yield":
            True,

        "arms_mm":
            [2.5, 13.0, 1.8],
    },
]


# ============================================================
# ACTIVE CONTACT HELPERS
# ============================================================

def active(
    row,
    prefix,
):

    return (
        row[
            f"{prefix}_rows"
        ]
        > 0
        and
        row[
            f"{prefix}_Fn_N"
        ]
        > 1.0e-4
    )


def contact_metrics(
    rows,
):

    if len(
        rows
    ) < 2:

        return {
            "middle_duration_s": 0.0,
            "tip_duration_s": 0.0,
            "overlap_duration_s": 0.0,
            "middle_after_tip_s": 0.0,
            "middle_first_s": -1.0,
            "tip_first_s": -1.0,
        }


    times = np.asarray(
        [
            row["time"]
            for row in rows
        ],
        dtype=float,
    )


    dt = float(
        np.median(
            np.diff(
                times
            )
        )
    )


    t0 = float(
        times[0]
    )


    mid_mask = np.asarray(
        [
            active(
                row,
                "mid",
            )
            for row in rows
        ],
        dtype=bool,
    )


    tip_mask = np.asarray(
        [
            active(
                row,
                "tip",
            )
            for row in rows
        ],
        dtype=bool,
    )


    overlap = (
        mid_mask
        &
        tip_mask
    )


    middle_first = (
        float(
            times[
                np.argmax(
                    mid_mask
                )
            ]
            -
            t0
        )
        if np.any(
            mid_mask
        )
        else -1.0
    )


    tip_first = (
        float(
            times[
                np.argmax(
                    tip_mask
                )
            ]
            -
            t0
        )
        if np.any(
            tip_mask
        )
        else -1.0
    )


    if np.any(
        tip_mask
    ):

        tip_index = int(
            np.argmax(
                tip_mask
            )
        )


        middle_after_tip = float(
            np.sum(
                mid_mask[
                    tip_index:
                ]
            )
            *
            dt
        )


    else:

        middle_after_tip = 0.0


    return {

        "middle_duration_s":
            float(
                np.sum(
                    mid_mask
                )
                *
                dt
            ),

        "tip_duration_s":
            float(
                np.sum(
                    tip_mask
                )
                *
                dt
            ),

        "overlap_duration_s":
            float(
                np.sum(
                    overlap
                )
                *
                dt
            ),

        "middle_after_tip_s":
            middle_after_tip,

        "middle_first_s":
            middle_first,

        "tip_first_s":
            tip_first,
    }


# ============================================================
# CAPTURE V24'S INTERNAL CONTACT RECORDER
# ============================================================

ORIGINAL_RECORDER = (
    v24.viewer_tools.ContactRecorder
)


class CapturingRecorder(
    ORIGINAL_RECORDER
):

    last_rows = []


    def restore(
        self,
    ):

        CapturingRecorder.last_rows = list(
            self.rows
        )

        super().restore()


# ============================================================
# FREE-SPACE GUARD
#
# Run ONCE only.
#
# Post-yield ratio is inactive if breakaway does not happen,
# therefore every V25 ratio has identical free-space mechanics.
# ============================================================

def run_free_guard(
    free_bag,
):

    case = CASES[
        0
    ]


    contact_xml = (
        MJCF_DIR
        /
        "tmp_v25_free_guard_contact.xml"
    )


    free_xml = (
        MJCF_DIR
        /
        "tmp_v25_free_guard_free.xml"
    )


    v24.build_model(
        contact_xml
    )


    v24.ab.make_free_model(
        contact_xml,
        free_xml,
    )


    law = (
        v24.StatefulTransmission()
    )


    law.install()


    old_scale = (
        v24.base.METERS_PER_TICK
    )

    old_tendon = (
        v24.base.TENDON_STIFFNESS
    )

    old_mu = (
        v24.base.CONTACT_MU
    )

    old_k = (
        v24.base.CONTACT_STIFFNESS
    )


    try:

        v24.base.METERS_PER_TICK = (
            v24.SCALE
        )

        v24.base.TENDON_STIFFNESS = (
            v24.TENDON_K
        )

        v24.base.CONTACT_MU = (
            v24.CONTACT_MU
        )

        v24.base.CONTACT_STIFFNESS = (
            v24.CONTACT_K
        )


        law.configure(
            case,
            "FREE",
        )


        metrics, _ = (
            v24.base.run_free(
                free_xml,
                free_bag,
            )
        )


    finally:

        law.restore()

        v24.base.METERS_PER_TICK = (
            old_scale
        )

        v24.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v24.base.CONTACT_MU = (
            old_mu
        )

        v24.base.CONTACT_STIFFNESS = (
            old_k
        )


    event = (
        v24.first_event(
            law.events,
            case[
                "name"
            ],
            "FREE",
        )
    )


    try:
        contact_xml.unlink()
    except FileNotFoundError:
        pass


    try:
        free_xml.unlink()
    except FileNotFoundError:
        pass


    return (
        float(
            metrics[
                "combined"
            ]
        ),
        event is not None,
    )


# ============================================================
# CONTACT-ONLY CANDIDATE
# ============================================================

def run_contact_candidate(
    case,
    contact_bag,
):

    label = (
        case[
            "name"
        ]
        .lower()
        .replace(
            ".",
            "p",
        )
    )


    contact_xml = (
        MJCF_DIR
        /
        f"tmp_v25_{label}.xml"
    )


    v24.build_model(
        contact_xml
    )


    law = (
        v24.StatefulTransmission()
    )


    law.install()


    old_scale = (
        v24.base.METERS_PER_TICK
    )

    old_tendon = (
        v24.base.TENDON_STIFFNESS
    )

    old_mu = (
        v24.base.CONTACT_MU
    )

    old_k = (
        v24.base.CONTACT_STIFFNESS
    )


    CapturingRecorder.last_rows = []


    v24.viewer_tools.ContactRecorder = (
        CapturingRecorder
    )


    try:

        v24.base.METERS_PER_TICK = (
            v24.SCALE
        )

        v24.base.TENDON_STIFFNESS = (
            v24.TENDON_K
        )

        v24.base.CONTACT_MU = (
            v24.CONTACT_MU
        )

        v24.base.CONTACT_STIFFNESS = (
            v24.CONTACT_K
        )


        law.configure(
            case,
            "CONTACT",
        )


        recorder = (
            v24.viewer_tools.ContactRecorder()
        )


        recorder.install()


        try:

            metrics, _ = (
                v24.base.run_contact(
                    contact_xml,
                    contact_bag,
                )
            )


        finally:

            recorder.restore()


    finally:

        v24.viewer_tools.ContactRecorder = (
            ORIGINAL_RECORDER
        )

        law.restore()

        v24.base.METERS_PER_TICK = (
            old_scale
        )

        v24.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v24.base.CONTACT_MU = (
            old_mu
        )

        v24.base.CONTACT_STIFFNESS = (
            old_k
        )


    pose = np.asarray(
        metrics[
            "pose"
        ],
        dtype=float,
    )


    rmse = float(
        np.sqrt(
            np.mean(
                (
                    pose
                    -
                    HW
                )
                ** 2
            )
        )
    )


    event = (
        v24.first_event(
            law.events,
            case[
                "name"
            ],
            "CONTACT",
        )
    )


    rows = (
        CapturingRecorder.last_rows
    )


    contacts = (
        contact_metrics(
            rows
        )
    )


    if (
        event is not None
        and
        rows
    ):

        yield_elapsed = (
            float(
                event[
                    "time"
                ]
            )
            -
            float(
                rows[
                    0
                ][
                    "time"
                ]
            )
        )


    else:

        yield_elapsed = -1.0


    result = {

        "name":
            case[
                "name"
            ],

        "mcp_arm_mm":
            case[
                "arms_mm"
            ][
                0
            ],

        "pip_arm_mm":
            case[
                "arms_mm"
            ][
                1
            ],

        "dip_arm_mm":
            case[
                "arms_mm"
            ][
                2
            ],

        "mcp_deg":
            float(
                pose[
                    0
                ]
            ),

        "pip_deg":
            float(
                pose[
                    1
                ]
            ),

        "dip_deg":
            float(
                pose[
                    2
                ]
            ),

        "contact_rmse":
            rmse,

        "yield_s":
            yield_elapsed,

        **contacts,
    }


    result[
        "retained_wrap"
    ] = (
        result[
            "overlap_duration_s"
        ]
        >= MIN_OVERLAP_S
    )


    try:
        contact_xml.unlink()
    except FileNotFoundError:
        pass


    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "free_bag",
        type=Path,
    )


    parser.add_argument(
        "contact_bag",
        type=Path,
    )


    parser.add_argument(
        "--no-viewer",
        action="store_true",
    )


    args = parser.parse_args()


    free_bag = (
        args.free_bag
        .expanduser()
        .resolve()
    )


    contact_bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )


    v24.ab.patch_replay_modules()


    print()
    print("=" * 122)
    print(
        "RH8D INDEX — WRAP RETENTION "
        "LOCAL REFINEMENT V25"
    )
    print("=" * 122)


    print()
    print(
        "Bottle position = FROZEN V5"
    )

    print(
        "Contact physics = unchanged"
    )

    print(
        "Breakaway architecture = unchanged V24"
    )


    # ========================================================
    # ONE FREE-SPACE CHECK ONLY
    # ========================================================

    print()
    print(
        "Running one free-space architecture guard..."
    )


    (
        free_rmse,
        free_breakaway,
    ) = run_free_guard(
        free_bag
    )


    print(
        f"  free RMSE = "
        f"{free_rmse:.3f} deg"
    )

    print(
        f"  free breakaway = "
        f"{free_breakaway}"
    )


    if (
        free_rmse
        > FREE_GATE
        or
        free_breakaway
    ):

        raise RuntimeError(
            "V25 free-space guard failed."
        )


    print()
    print(
        "PASS: post-yield ratios remain inactive "
        "during free space."
    )


    # ========================================================
    # CONTACT SCAN
    # ========================================================

    results = []


    for i, case in enumerate(
        CASES,
        start=1,
    ):

        print()
        print("=" * 122)

        print(
            f"[{i:02d}/{len(CASES):02d}] "
            f"{case['name']} "
            f"{case['arms_mm']} mm"
        )

        print("=" * 122)


        result = (
            run_contact_candidate(
                case,
                contact_bag,
            )
        )


        results.append(
            result
        )


        print()
        print(
            "Pose MCP/PIP/DIP = "
            f"{result['mcp_deg']:.3f} / "
            f"{result['pip_deg']:.3f} / "
            f"{result['dip_deg']:.3f} deg"
        )


        print(
            f"RMSE = "
            f"{result['contact_rmse']:.3f} deg"
        )


        print()
        print(
            "Contact retention:"
        )

        print(
            f"  middle total = "
            f"{result['middle_duration_s']:.3f} s"
        )

        print(
            f"  tip total = "
            f"{result['tip_duration_s']:.3f} s"
        )

        print(
            f"  middle+tip simultaneous = "
            f"{result['overlap_duration_s']:.3f} s"
        )

        print(
            f"  middle contact after first tip = "
            f"{result['middle_after_tip_s']:.3f} s"
        )

        print(
            f"  retained wrap = "
            f"{result['retained_wrap']}"
        )


    # ========================================================
    # SAVE
    # ========================================================

    with open(
        SUMMARY,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # ========================================================
    # SELECTION
    #
    # First require persistent middle + tip overlap.
    # Only then minimize joint-angle RMSE.
    # ========================================================

    retained = [
        r
        for r in results
        if r[
            "retained_wrap"
        ]
    ]


    if retained:

        selected = min(
            retained,
            key=lambda r:
                r[
                    "contact_rmse"
                ],
        )

        mode = (
            "PERSISTENT MIDDLE+TIP WRAP"
        )


    else:

        selected = min(
            results,
            key=lambda r:
                r[
                    "contact_rmse"
                ],
        )

        mode = (
            "NO CANDIDATE RETAINED MIDDLE WRAP"
        )


    selected_case = next(
        c
        for c in CASES
        if (
            c[
                "name"
            ]
            ==
            selected[
                "name"
            ]
        )
    )


    print()
    print("=" * 122)
    print(
        "V25 SELECTED RESULT"
    )
    print("=" * 122)


    print()
    print(
        f"Selection mode = "
        f"{mode}"
    )

    print(
        f"Candidate = "
        f"{selected['name']}"
    )


    print(
        "Effective post-yield arms = "
        f"{selected['mcp_arm_mm']:.2f} / "
        f"{selected['pip_arm_mm']:.2f} / "
        f"{selected['dip_arm_mm']:.2f} mm"
    )


    print()
    print(
        "Hardware target:"
    )

    print(
        "  14.536 / 23.811 / 9.327 deg"
    )


    print(
        "Prediction:"
    )

    print(
        "  "
        f"{selected['mcp_deg']:.3f} / "
        f"{selected['pip_deg']:.3f} / "
        f"{selected['dip_deg']:.3f} deg"
    )


    print(
        f"RMSE = "
        f"{selected['contact_rmse']:.3f} deg"
    )


    print()
    print(
        f"middle duration = "
        f"{selected['middle_duration_s']:.3f} s"
    )

    print(
        f"tip duration = "
        f"{selected['tip_duration_s']:.3f} s"
    )

    print(
        f"middle+tip overlap = "
        f"{selected['overlap_duration_s']:.3f} s"
    )

    print(
        f"middle after tip = "
        f"{selected['middle_after_tip_s']:.3f} s"
    )


    # ========================================================
    # DECISION
    # ========================================================

    print()
    print("=" * 122)
    print(
        "V25 DECISION"
    )
    print("=" * 122)


    angle_good = (
        selected[
            "contact_rmse"
        ]
        <= 4.0
    )


    wrap_good = (
        selected[
            "retained_wrap"
        ]
    )


    if (
        angle_good
        and
        wrap_good
    ):

        print()
        print(
            "MAJOR SUCCESS:"
        )

        print(
            "joint redistribution and persistent "
            "middle/distal wrapping are both present."
        )

        print()
        print(
            "STOP structural tuning."
        )


    elif (
        angle_good
        and
        not wrap_good
    ):

        print()
        print(
            "ANGLES ARE CLOSE BUT WRAP IS STILL WRONG."
        )

        print()
        print(
            "Do NOT tune tendon ratios further."
        )

        print(
            "V26 should add a physically compliant/"
            "rolling middle-phalanx contact layer while "
            "keeping the bottle pose and V25 transmission "
            "frozen."
        )


    elif (
        selected[
            "pip_deg"
        ]
        >= 20.0
        and
        not wrap_good
    ):

        print()
        print(
            "PIP REDISTRIBUTION IS SUFFICIENT, "
            "BUT CONTACT RETENTION IS NOT."
        )

        print()
        print(
            "This localizes the next problem to the "
            "middle-phalanx/object contact representation."
        )


    else:

        print()
        print(
            "The local transmission trend has not yet "
            "reached the measured PIP state."
        )

        print(
            "Use this result to decide whether one final "
            "very small ratio interpolation is justified."
        )


    print()
    print(
        "Summary:"
    )

    print(
        SUMMARY
    )


    # ========================================================
    # GUI — ONLY SELECTED CANDIDATE
    # ========================================================

    if not args.no_viewer:

        print()
        print("=" * 122)
        print(
            "V25 AUTOMATIC SELECTED GUI REPLAY"
        )
        print("=" * 122)


        print()
        print(
            f"Viewing: "
            f"{selected['name']}"
        )

        print(
            "Index visual geometry remains alpha=1.0."
        )

        print(
            "Bottle position remains frozen."
        )


        v24.visual_replay(
            selected_case,
            contact_bag,
        )


    print()
    print("=" * 122)
    print(
        "V25 COMPLETE"
    )
    print("=" * 122)


if __name__ == "__main__":
    main()
