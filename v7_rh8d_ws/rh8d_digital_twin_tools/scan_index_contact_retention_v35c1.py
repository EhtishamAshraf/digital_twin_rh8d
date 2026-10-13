#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import fit_index_passive_contact_v35b2 as b2


# =============================================================================
# V35C1
#
# CONTACT RETENTION / BOUNCE DIAGNOSTIC
#
# FROZEN:
#   V33 environment
#   cylinder pose / geometry
#   CAD spatial tendon
#   spool scale
#   tendon stiffness
#   V35B2 MCP/PIP/DIP passive mechanics
#
# SCANNED ONLY:
#   coefficient of friction mu
#   effective contact stiffness
#
# NO:
#   contact-triggered joint rules
#   contact-triggered tendon changes
#   prescribed MCP/PIP/DIP
#   object motion
# =============================================================================


MU_VALUES = [
    0.05,
    0.15,
    0.30,
    0.60,
]


CONTACT_K_VALUES = [
    100.0,
    250.0,
    500.0,
    1000.0,
]


# V35B2 calibration result.
BASELINE_RMSE = 2.323

# Preserve kinematic calibration within +1 degree.
RMSE_GATE = (
    BASELINE_RMSE
    +
    1.0
)


def run_one(
    contact_bag,
    mu,
    contact_k,
):

    # ------------------------------------------------------------------
    # Capture the COMPLETE v28 contact analysis.
    #
    # b2.run_candidate() normally returns only a reduced subset.
    # We temporarily wrap analyse() so we can retrieve metrics such as
    # longest continuous overlap without changing any simulation physics.
    # ------------------------------------------------------------------

    original_analyse = (
        b2.v28.analyse
    )

    captured = {}


    def capture_analyse(
        rows,
    ):

        result = original_analyse(
            rows
        )

        captured.clear()

        captured.update(
            result
        )

        return result


    old_mu = (
        b2.v35.CONTACT_MU
    )

    old_k = (
        b2.v35.CONTACT_K
    )


    try:

        b2.v28.analyse = (
            capture_analyse
        )

        b2.v35.CONTACT_MU = (
            float(
                mu
            )
        )

        b2.v35.CONTACT_K = (
            float(
                contact_k
            )
        )


        result = b2.run_candidate(
            b2.SELECTED_MODEL,
            contact_bag,
        )


    finally:

        b2.v28.analyse = (
            original_analyse
        )

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_k
        )


    # ----------------------------------------------------------
    # Extract continuous-contact metrics.
    # ----------------------------------------------------------

    longest_overlap = float(
        captured.get(
            "longest_overlap_s",
            0.0,
        )
    )


    overlap = float(
        captured.get(
            "overlap_s",
            result[
                "overlap"
            ],
        )
    )


    middle_after_tip = float(
        captured.get(
            "middle_after_tip_s",
            result[
                "after_tip"
            ],
        )
    )


    middle_duration = float(
        captured.get(
            "middle_duration_s",
            result[
                "middle"
            ],
        )
    )


    tip_duration = float(
        captured.get(
            "tip_duration_s",
            result[
                "tip"
            ],
        )
    )


    return {

        "mu":
            float(
                mu
            ),

        "contact_k":
            float(
                contact_k
            ),

        "pose":
            result[
                "pose"
            ],

        "rmse":
            float(
                result[
                    "rmse"
                ]
            ),

        "middle":
            middle_duration,

        "tip":
            tip_duration,

        "overlap":
            overlap,

        "longest_overlap":
            longest_overlap,

        "after_tip":
            middle_after_tip,

        "full_analysis":
            captured.copy(),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    args = parser.parse_args()


    contact_bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )


    if not contact_bag.exists():

        raise RuntimeError(
            f"Missing contact bag:\n"
            f"{contact_bag}"
        )


    if not b2.SELECTED_MODEL.exists():

        raise RuntimeError(
            "Missing V35B2 selected model:\n"
            f"{b2.SELECTED_MODEL}\n\n"
            "Run V35B2 first."
        )


    # Exact same replay initialization used by V33/V35A/V35B.
    b2.v30.v24.ab.patch_replay_modules()


    print()
    print("=" * 120)
    print(
        "V35C1 — CONTACT RETENTION / BOUNCE SCAN"
    )
    print("=" * 120)


    print()
    print(
        "FROZEN:"
    )

    print(
        "  V33 environment and cylinder"
    )

    print(
        "  CAD spatial tendon"
    )

    print(
        "  V35B2 passive mechanics:"
    )

    print(
        "    MCP k = 0.0045"
    )

    print(
        "    PIP k = 0.0015"
    )

    print(
        "    PIP friction = 0.00125"
    )

    print(
        "    DIP k = 0.0475"
    )


    print()
    print(
        "SCANNING ONLY:"
    )

    print(
        f"  mu = {MU_VALUES}"
    )

    print(
        f"  contact stiffness = "
        f"{CONTACT_K_VALUES} N/m"
    )


    combinations = list(
        itertools.product(
            MU_VALUES,
            CONTACT_K_VALUES,
        )
    )


    rows = []


    for index, (
        mu,
        contact_k,
    ) in enumerate(
        combinations,
        start=1,
    ):

        print()
        print(
            f"[{index:02d}/{len(combinations):02d}] "
            f"mu={mu:.2f} "
            f"k={contact_k:.0f} N/m"
        )


        result = run_one(
            contact_bag,
            mu,
            contact_k,
        )


        rows.append(
            result
        )


        q = result[
            "pose"
        ]


        print(
            "  q = "
            f"{q[0]:.3f} / "
            f"{q[1]:.3f} / "
            f"{q[2]:.3f} deg"
        )


        print(
            f"  RMSE = "
            f"{result['rmse']:.3f} deg"
        )


        print(
            "  contact: "
            f"middle={result['middle']:.3f}s "
            f"tip={result['tip']:.3f}s "
            f"overlap={result['overlap']:.3f}s "
            f"longest_overlap="
            f"{result['longest_overlap']:.3f}s "
            f"middle_after_tip="
            f"{result['after_tip']:.3f}s"
        )


    # ==================================================================
    # SELECTION
    #
    # First preserve the angle calibration.
    # Then maximize continuous simultaneous contact.
    #
    # This is intentionally NOT an arbitrary weighted sum.
    # ==================================================================

    admissible = [
        row
        for row in rows
        if row[
            "rmse"
        ]
        <=
        RMSE_GATE
    ]


    if not admissible:

        print()
        print(
            "WARNING:"
        )

        print(
            "No contact candidate preserved "
            "V35B2 RMSE within +1 degree."
        )

        print(
            "Selecting the minimum-RMSE candidate "
            "only for diagnosis."
        )


        admissible = rows


    admissible.sort(
        key=lambda row:
            (
                -row[
                    "longest_overlap"
                ],

                -row[
                    "after_tip"
                ],

                row[
                    "rmse"
                ],
            )
    )


    selected = admissible[
        0
    ]


    # Also print all cases sorted by continuous retention.
    ranked = sorted(
        rows,
        key=lambda row:
            (
                -row[
                    "longest_overlap"
                ],

                row[
                    "rmse"
                ],
            )
    )


    print()
    print("=" * 120)
    print(
        "V35C1 RESULTS — SORTED BY LONGEST "
        "CONTINUOUS MIDDLE+TIP CONTACT"
    )
    print("=" * 120)


    for rank, row in enumerate(
        ranked,
        start=1,
    ):

        q = row[
            "pose"
        ]


        gate = (
            "PASS"
            if
            row[
                "rmse"
            ]
            <=
            RMSE_GATE
            else
            "FAIL"
        )


        print(
            f"{rank:02d}. "
            f"mu={row['mu']:.2f} "
            f"k={row['contact_k']:6.0f} "
            f"RMSE={row['rmse']:6.3f} "
            f"gate={gate} "
            f"q="
            f"{q[0]:6.2f}/"
            f"{q[1]:6.2f}/"
            f"{q[2]:6.2f} "
            f"mid={row['middle']:.3f}s "
            f"tip={row['tip']:.3f}s "
            f"overlap={row['overlap']:.3f}s "
            f"LONGEST={row['longest_overlap']:.3f}s "
            f"after_tip={row['after_tip']:.3f}s"
        )


    print()
    print("=" * 120)
    print(
        "V35C1 SELECTED CONTACT MODEL"
    )
    print("=" * 120)


    print(
        f"mu = "
        f"{selected['mu']:.3f}"
    )

    print(
        f"effective contact stiffness = "
        f"{selected['contact_k']:.1f} N/m"
    )

    print(
        f"RMSE = "
        f"{selected['rmse']:.3f} deg"
    )

    print(
        f"longest continuous overlap = "
        f"{selected['longest_overlap']:.3f} s"
    )

    print(
        f"middle after first tip = "
        f"{selected['after_tip']:.3f} s"
    )


    # ==================================================================
    # VISUALIZE ONLY THE SELECTED CASE
    # ==================================================================

    old_mu = (
        b2.v35.CONTACT_MU
    )

    old_k = (
        b2.v35.CONTACT_K
    )


    try:

        b2.v35.CONTACT_MU = (
            selected[
                "mu"
            ]
        )

        b2.v35.CONTACT_K = (
            selected[
                "contact_k"
            ]
        )


        print()
        print("=" * 120)
        print(
            "V35C1 AUTOMATIC SELECTED GUI"
        )
        print("=" * 120)

        print()
        print(
            "Watch specifically for:"
        )

        print(
            "  contact -> rebound -> separation"
        )

        print(
            "versus"
        )

        print(
            "  contact -> damped settling -> "
            "sustained wrapping"
        )

        print()
        print(
            "No contact-triggered control is active."
        )


        b2.visual_replay(
            b2.SELECTED_MODEL,
            contact_bag,
        )


    finally:

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_k
        )


    print()
    print("=" * 120)
    print(
        "V35C1 COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
