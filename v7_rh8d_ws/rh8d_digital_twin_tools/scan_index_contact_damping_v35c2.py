#!/usr/bin/env python3

from __future__ import annotations

import argparse
import functools
import inspect
from pathlib import Path

import numpy as np

import scan_index_contact_retention_v35c1 as c1
import fit_index_passive_contact_v35b2 as b2


# =============================================================================
# V35C2 — CONTACT DAMPING IDENTIFICATION
#
# FROZEN:
#   - V33 environment
#   - H1/V5 cylinder registration
#   - CAD spatial tendon
#   - V35B2 passive joint parameters
#   - friction mu = 0.05
#   - effective contact stiffness = 1000 N/m
#
# SCANNED:
#   - ONLY the damping coefficient of direct-format MuJoCo solref
#
# NO contact-triggered logic.
# NO prescribed joint angles.
# NO tendon switching.
# NO bottle movement.
# =============================================================================


FIXED_MU = 0.05
FIXED_CONTACT_K = 1000.0


DAMPING_VALUES = [
    0.25,
    0.50,
    1.00,   # exact old baseline
    2.00,
    5.00,
    10.00,
    20.00,
]


BASELINE_RMSE = 2.323

RMSE_GATE = (
    BASELINE_RMSE
    +
    1.0
)


CURRENT_DAMPING = 1.0


PATCH_EVENTS = []


# =============================================================================
# RUNTIME SOLREF PATCH
# =============================================================================


def looks_like_model(
    value,
):

    return (
        hasattr(
            value,
            "geom_solref",
        )
        and
        hasattr(
            value,
            "ngeom",
        )
    )


def find_model(
    args,
    kwargs,
):

    for value in args:

        if looks_like_model(
            value
        ):

            return value


    for value in kwargs.values():

        if looks_like_model(
            value
        ):

            return value


    return None


def make_runtime_wrapper(
    module,
    function_name,
    original,
):

    @functools.wraps(
        original
    )
    def wrapped(
        *args,
        **kwargs,
    ):

        model = find_model(
            args,
            kwargs,
        )


        before = None


        if model is not None:

            before = np.array(
                model.geom_solref,
                dtype=float,
                copy=True,
            )


        result = original(
            *args,
            **kwargs,
        )


        if model is None:

            return result


        after = np.asarray(
            model.geom_solref
        )


        # ------------------------------------------------------------
        # Prefer exactly the rows changed by the original runtime
        # contact-law function.
        # ------------------------------------------------------------

        changed = np.any(
            np.abs(
                after
                -
                before
            )
            >
            1e-12,
            axis=1,
        )


        # ------------------------------------------------------------
        # Fallback:
        # if the function happened to write the same stiffness already
        # present in XML, identify rows by the requested direct-format
        # stiffness.
        # ------------------------------------------------------------

        if not np.any(
            changed
        ):

            contact_k = float(
                getattr(
                    module,
                    "CONTACT_STIFFNESS",
                    FIXED_CONTACT_K,
                )
            )


            changed = np.isclose(
                after[
                    :,
                    0
                ],
                -contact_k,
                rtol=1e-8,
                atol=1e-10,
            )


        count = int(
            np.count_nonzero(
                changed
            )
        )


        if count <= 0:

            raise RuntimeError(
                "\nV35C2 could not identify the geoms whose "
                "solref was changed by the runtime contact law.\n"
                "STOP: damping has NOT been applied."
            )


        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # Original runtime law remains responsible for stiffness.
        #
        # We change ONLY solref[:,1].
        # Direct-format value remains negative.
        # ------------------------------------------------------------

        after[
            changed,
            1
        ] = (
            -float(
                CURRENT_DAMPING
            )
        )


        unique_solref = np.unique(
            np.round(
                after[
                    changed,
                    :
                ],
                decimals=8,
            ),
            axis=0,
        )


        PATCH_EVENTS.append(
            {
                "module":
                    module.__name__,

                "function":
                    function_name,

                "count":
                    count,

                "requested_damping":
                    float(
                        CURRENT_DAMPING
                    ),

                "solref":
                    unique_solref.copy(),
            }
        )


        return result


    return wrapped


def install_runtime_damping_patch():

    modules = []


    for module in (
        b2.v30.v24.base,
        b2.v30.v24.viewer_tools.base,
    ):

        if not any(
            id(module)
            ==
            id(existing)
            for existing in modules
        ):

            modules.append(
                module
            )


    installed = []


    for module in modules:

        for (
            name,
            function,
        ) in inspect.getmembers(
            module,
            inspect.isfunction,
        ):

            try:

                source = inspect.getsource(
                    function
                )

            except (
                OSError,
                TypeError,
            ):

                continue


            # We want the actual runtime function that writes solref.
            if (
                "geom_solref"
                not in source
            ):

                continue


            original = function


            setattr(
                module,
                name,
                make_runtime_wrapper(
                    module,
                    name,
                    original,
                ),
            )


            installed.append(
                (
                    module,
                    name,
                    original,
                )
            )


    if not installed:

        print()
        print(
            "Functions available containing 'solref':"
        )


        for module in modules:

            for (
                name,
                function,
            ) in inspect.getmembers(
                module,
                inspect.isfunction,
            ):

                try:

                    source = inspect.getsource(
                        function
                    )

                except Exception:

                    continue


                if (
                    "solref"
                    in source
                ):

                    print(
                        "  "
                        f"{module.__name__}."
                        f"{name}"
                    )


        raise RuntimeError(
            "\nCould not find the runtime geom_solref writer.\n"
            "Nothing has been modified."
        )


    print()
    print(
        "Runtime damping patch installed on:"
    )


    for (
        module,
        name,
        original,
    ) in installed:

        print(
            "  "
            f"{module.__name__}."
            f"{name}"
        )


    return installed


def restore_runtime_patch(
    installed,
):

    for (
        module,
        name,
        original,
    ) in installed:

        setattr(
            module,
            name,
            original,
        )


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================


def get_numeric(
    dictionary,
    names,
    default=np.nan,
):

    for name in names:

        if name not in dictionary:

            continue


        value = dictionary[
            name
        ]


        if isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):

            return float(
                value
            )


    return float(
        default
    )


def run_damping_case(
    contact_bag,
    damping,
):

    global CURRENT_DAMPING


    CURRENT_DAMPING = float(
        damping
    )


    PATCH_EVENTS.clear()


    result = c1.run_one(
        contact_bag,
        FIXED_MU,
        FIXED_CONTACT_K,
    )


    if not PATCH_EVENTS:

        raise RuntimeError(
            "Simulation completed but V35C2 damping patch "
            "was never invoked."
        )


    # ------------------------------------------------------------
    # Verify actual runtime solref.
    # ------------------------------------------------------------

    valid_event = False


    for event in PATCH_EVENTS:

        values = event[
            "solref"
        ]


        if values.size == 0:

            continue


        if np.allclose(
            values[
                :,
                1
            ],
            -float(
                damping
            ),
            rtol=1e-8,
            atol=1e-10,
        ):

            valid_event = True


    if not valid_event:

        raise RuntimeError(
            "Requested damping was not found in final "
            "runtime geom_solref values."
        )


    analysis = result[
        "full_analysis"
    ]


    middle_episodes = get_numeric(
        analysis,
        (
            "middle_episode_count",
            "middle_episodes",
            "n_middle_episodes",
            "num_middle_episodes",
        ),
    )


    tip_episodes = get_numeric(
        analysis,
        (
            "tip_episode_count",
            "tip_episodes",
            "n_tip_episodes",
            "num_tip_episodes",
        ),
    )


    result[
        "damping"
    ] = float(
        damping
    )


    result[
        "middle_episodes"
    ] = middle_episodes


    result[
        "tip_episodes"
    ] = tip_episodes


    return result


# =============================================================================
# MAIN
# =============================================================================


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
            f"{b2.SELECTED_MODEL}"
        )


    b2.v30.v24.ab.patch_replay_modules()


    # Freeze contact law except damping.
    old_mu = (
        b2.v35.CONTACT_MU
    )

    old_k = (
        b2.v35.CONTACT_K
    )


    b2.v35.CONTACT_MU = (
        FIXED_MU
    )

    b2.v35.CONTACT_K = (
        FIXED_CONTACT_K
    )


    installed = []


    try:

        installed = (
            install_runtime_damping_patch()
        )


        print()
        print("=" * 120)
        print(
            "V35C2 — DIRECT CONTACT-DAMPING SCAN"
        )
        print("=" * 120)


        print()
        print(
            "FROZEN:"
        )

        print(
            "  CAD tendon"
        )

        print(
            "  V35B2 passive mechanics"
        )

        print(
            "  cylinder / environment"
        )

        print(
            f"  mu = {FIXED_MU}"
        )

        print(
            f"  contact stiffness = "
            f"{FIXED_CONTACT_K:.1f} N/m"
        )


        print()
        print(
            "SCANNING ONLY direct-format solref damping:"
        )

        print(
            "  "
            f"{DAMPING_VALUES}"
        )


        rows = []


        for index, damping in enumerate(
            DAMPING_VALUES,
            start=1,
        ):

            print()
            print("-" * 120)

            print(
                f"[{index:02d}/"
                f"{len(DAMPING_VALUES):02d}] "
                f"damping = {damping:.3f}"
            )

            print("-" * 120)


            result = run_damping_case(
                contact_bag,
                damping,
            )


            rows.append(
                result
            )


            q = result[
                "pose"
            ]


            last_event = (
                PATCH_EVENTS[
                    -1
                ]
            )


            print(
                "  runtime solref = "
                f"{last_event['solref']}"
            )


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
                f"middle="
                f"{result['middle']:.3f}s "
                f"tip="
                f"{result['tip']:.3f}s "
                f"overlap="
                f"{result['overlap']:.3f}s "
                f"LONGEST="
                f"{result['longest_overlap']:.3f}s "
                f"after_tip="
                f"{result['after_tip']:.3f}s"
            )


            if np.isfinite(
                result[
                    "middle_episodes"
                ]
            ):

                print(
                    "  middle episodes = "
                    f"{result['middle_episodes']:.0f}"
                )


            if np.isfinite(
                result[
                    "tip_episodes"
                ]
            ):

                print(
                    "  tip episodes = "
                    f"{result['tip_episodes']:.0f}"
                )


        # =============================================================
        # SELECTION
        #
        # 1. Keep kinematic calibration within +1 deg.
        # 2. Maximize sustained simultaneous contact.
        # 3. Prefer more middle retention after tip.
        # 4. Prefer lower RMSE.
        #
        # NO arbitrary weighted score.
        # =============================================================

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
                "WARNING: no damping candidate passed "
                "the RMSE guard."
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


        ranked = sorted(
            rows,
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


        print()
        print("=" * 120)
        print(
            "V35C2 RESULTS — SUSTAINED CONTACT RANKING"
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
                f"damp="
                f"{row['damping']:6.2f} "
                f"RMSE="
                f"{row['rmse']:6.3f} "
                f"gate={gate} "
                f"q="
                f"{q[0]:6.2f}/"
                f"{q[1]:6.2f}/"
                f"{q[2]:6.2f} "
                f"mid="
                f"{row['middle']:.3f}s "
                f"tip="
                f"{row['tip']:.3f}s "
                f"overlap="
                f"{row['overlap']:.3f}s "
                f"LONGEST="
                f"{row['longest_overlap']:.3f}s "
                f"after_tip="
                f"{row['after_tip']:.3f}s"
            )


        print()
        print("=" * 120)
        print(
            "V35C2 SELECTED"
        )
        print("=" * 120)


        print(
            f"damping = "
            f"{selected['damping']:.3f}"
        )

        print(
            f"mu = "
            f"{FIXED_MU:.3f}"
        )

        print(
            f"contact stiffness = "
            f"{FIXED_CONTACT_K:.1f} N/m"
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
            f"middle after tip = "
            f"{selected['after_tip']:.3f} s"
        )


        # =============================================================
        # GUI — selected only
        # =============================================================

        global CURRENT_DAMPING

        CURRENT_DAMPING = float(
            selected[
                "damping"
            ]
        )


        print()
        print("=" * 120)
        print(
            "V35C2 SELECTED MUJOCO GUI"
        )
        print("=" * 120)


        print()
        print(
            "Watch the MOTION, not only final angles."
        )

        print(
            "Desired qualitative transition:"
        )

        print(
            "  contact -> damped settling -> "
            "continued wrapping"
        )

        print()
        print(
            "Failure:"
        )

        print(
            "  contact -> rebound -> separation -> "
            "final air gap"
        )


        b2.visual_replay(
            b2.SELECTED_MODEL,
            contact_bag,
        )


    finally:

        if installed:

            restore_runtime_patch(
                installed
            )


        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_k
        )


    print()
    print("=" * 120)
    print(
        "V35C2 COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
