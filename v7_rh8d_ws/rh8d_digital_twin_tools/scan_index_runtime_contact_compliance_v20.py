#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv

import mujoco
import numpy as np

import scan_and_view_index_contact_friction_v18 as v18


# ============================================================
# RH8D INDEX — NORMAL CONTACT COMPLIANCE V20
#
# V19 established that VALID runtime friction variation:
#
#   mu = 0.05 ... 1.20
#
# does NOT create the required PIP redistribution.
#
# Visual replay showed:
#
#   middle contact
#       -> hard reaction / bounce
#       -> loss/reduction of constraint
#       -> no PIP takeover
#       -> no tip wrapping
#
# V20 changes ONLY:
#
#       base.CONTACT_STIFFNESS
#
# FROZEN:
#
#   contact mu = 0.05
#   object / bottle registration
#   contact geometry
#   motor dynamics
#   encoder-to-spool scale
#   trusted constant tendon transmission
#   joint stiffness/damping/friction
#
# Candidate is automatically replayed in native MuJoCo GUI.
#
# NO custom window dimensions are passed to the viewer.
# ============================================================


FIXED_MU = 0.05


# Current trusted runtime value = 1000 N/m.
#
# We test softer values because the actual visual failure is
# impact/bounce rather than excessive penetration.
#
# 2000 is included as a harder control.
CONTACT_STIFFNESS_VALUES = [
    100.0,
    250.0,
    500.0,
    750.0,
    1000.0,
    1500.0,
    2000.0,
]


RESULT_DIR = (
    v18.WS
    / "rh8d_digital_twin_tools/results/"
      "runtime_contact_compliance_v20"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUMMARY = (
    RESULT_DIR
    / "summary.csv"
)


SELECTED_MODEL = (
    v18.MJCF_DIR
    / "ur5e_rh8d_index_runtime_contact_compliance_v20_candidate.xml"
)


BOTTLE = "index_contact_fixture"
MID = "collision_idx_mid"


# ============================================================
# ACTUAL CONTACT-SOLVER PROBE
#
# Reads data.contact[].solref during the REAL replay.
#
# For direct-format contact:
#
#       solref[0] = -stiffness
#
# Therefore:
#
#       effective stiffness = -solref[0]
#
# We also record:
#
#   physical middle/cylinder gap
#   actual normal force
#   contact episode count
#
# so we can quantify the bouncing observed in the viewer.
# ============================================================

class ComplianceProbe:

    def __init__(self):

        self.original_step = (
            mujoco.mj_step
        )

        self.rows = []

        self.cache = {}


    def ids(
        self,
        model,
    ):

        key = id(
            model
        )


        if key in self.cache:

            return self.cache[
                key
            ]


        bottle = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            BOTTLE,
        )


        mid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            MID,
        )


        if (
            bottle < 0
            or
            mid < 0
        ):

            self.cache[
                key
            ] = None

            return None


        result = {
            "bottle":
                bottle,

            "mid":
                mid,
        }


        self.cache[
            key
        ] = result

        return result


    def record(
        self,
        model,
        data,
    ):

        ids = self.ids(
            model
        )


        if ids is None:

            return


        # ----------------------------------------------------
        # Physical signed distance, independent of margin.
        # ----------------------------------------------------

        fromto = np.zeros(
            6,
            dtype=float,
        )


        physical_gap = float(
            mujoco.mj_geomDistance(
                model,
                data,
                ids[
                    "mid"
                ],
                ids[
                    "bottle"
                ],
                1.0,
                fromto,
            )
        )


        # ----------------------------------------------------
        # Actual bottle-middle contact.
        # ----------------------------------------------------

        Fn = 0.0

        contact_rows = 0

        solref0_values = []

        solref1_values = []


        for i in range(
            data.ncon
        ):

            c = data.contact[
                i
            ]


            pair = {
                int(
                    c.geom1
                ),
                int(
                    c.geom2
                ),
            }


            if (
                ids[
                    "bottle"
                ]
                not in pair
                or
                ids[
                    "mid"
                ]
                not in pair
            ):

                continue


            contact_rows += 1


            force = np.zeros(
                6,
                dtype=float,
            )


            mujoco.mj_contactForce(
                model,
                data,
                i,
                force,
            )


            Fn += abs(
                float(
                    force[
                        0
                    ]
                )
            )


            # MjContact normally exposes the mixed contact
            # solver parameters directly.
            try:

                solref0_values.append(
                    float(
                        c.solref[
                            0
                        ]
                    )
                )

                solref1_values.append(
                    float(
                        c.solref[
                            1
                        ]
                    )
                )


            except Exception:

                # Fallback for older bindings.
                solref0_values.append(
                    float(
                        model.geom_solref[
                            ids[
                                "bottle"
                            ],
                            0
                        ]
                    )
                )

                solref1_values.append(
                    float(
                        model.geom_solref[
                            ids[
                                "bottle"
                            ],
                            1
                        ]
                    )
                )


        solref0 = (
            float(
                np.median(
                    solref0_values
                )
            )
            if solref0_values
            else float(
                "nan"
            )
        )


        solref1 = (
            float(
                np.median(
                    solref1_values
                )
            )
            if solref1_values
            else float(
                "nan"
            )
        )


        effective_k = (
            -solref0
            if (
                np.isfinite(
                    solref0
                )
                and
                solref0 < 0.0
            )
            else float(
                "nan"
            )
        )


        self.rows.append(
            {
                "time":
                    float(
                        data.time
                    ),

                "physical_mid_gap_mm":
                    physical_gap
                    * 1000.0,

                "contact_rows":
                    contact_rows,

                "Fn_N":
                    Fn,

                "actual_solref0":
                    solref0,

                "actual_solref1":
                    solref1,

                "effective_stiffness":
                    effective_k,
            }
        )


    def install(
        self,
    ):

        original = (
            self.original_step
        )

        probe = self


        def wrapped(
            model,
            data,
            *args,
            **kwargs,
        ):

            result = original(
                model,
                data,
                *args,
                **kwargs,
            )


            probe.record(
                model,
                data,
            )


            return result


        mujoco.mj_step = wrapped


    def restore(
        self,
    ):

        mujoco.mj_step = (
            self.original_step
        )


# ============================================================
# PROBE ANALYSIS
# ============================================================

def analyse_probe(
    rows,
):

    if not rows:

        raise RuntimeError(
            "V20 compliance probe recorded no samples."
        )


    active = [
        r
        for r in rows
        if (
            r[
                "contact_rows"
            ]
            > 0
            and
            r[
                "Fn_N"
            ]
            > 1.0e-4
        )
    ]


    if not active:

        return {
            "effective_stiffness":
                float(
                    "nan"
                ),

            "actual_solref1":
                float(
                    "nan"
                ),

            "min_mid_gap_mm":
                float(
                    np.min(
                        [
                            r[
                                "physical_mid_gap_mm"
                            ]
                            for r in rows
                        ]
                    )
                ),

            "middle_episodes":
                0,

            "longest_middle_episode_s":
                0.0,

            "probe_max_Fn":
                0.0,
        }


    effective_k = float(
        np.median(
            [
                r[
                    "effective_stiffness"
                ]
                for r in active
                if np.isfinite(
                    r[
                        "effective_stiffness"
                    ]
                )
            ]
        )
    )


    actual_solref1 = float(
        np.median(
            [
                r[
                    "actual_solref1"
                ]
                for r in active
                if np.isfinite(
                    r[
                        "actual_solref1"
                    ]
                )
            ]
        )
    )


    # --------------------------------------------------------
    # Contact episodes.
    #
    # If contact repeatedly appears/disappears, this captures
    # the bounce observed visually.
    # --------------------------------------------------------

    times = np.asarray(
        [
            r[
                "time"
            ]
            for r in rows
        ],
        dtype=float,
    )


    if len(
        times
    ) > 1:

        dt = float(
            np.median(
                np.diff(
                    times
                )
            )
        )

    else:

        dt = 0.0


    mask = [
        (
            r[
                "contact_rows"
            ]
            > 0
            and
            r[
                "Fn_N"
            ]
            > 1.0e-4
        )
        for r in rows
    ]


    episodes = 0

    current_length = 0

    longest_length = 0


    previous = False


    for is_active in mask:

        if is_active:

            if not previous:

                episodes += 1

                current_length = 0


            current_length += 1

            longest_length = max(
                longest_length,
                current_length,
            )


        else:

            current_length = 0


        previous = is_active


    longest_duration = (
        longest_length
        * dt
    )


    return {
        "effective_stiffness":
            effective_k,

        "actual_solref1":
            actual_solref1,

        "min_mid_gap_mm":
            float(
                np.min(
                    [
                        r[
                            "physical_mid_gap_mm"
                        ]
                        for r in rows
                    ]
                )
            ),

        "middle_episodes":
            episodes,

        "longest_middle_episode_s":
            longest_duration,

        "probe_max_Fn":
            float(
                np.max(
                    [
                        r[
                            "Fn_N"
                        ]
                        for r in rows
                    ]
                )
            ),
    }


# ============================================================
# RUN ONE CANDIDATE
# ============================================================

def run_candidate(
    stiffness,
    contact_bag,
):

    # Runtime values ACTUALLY consumed by base.run_contact().
    v18.base.CONTACT_MU = (
        FIXED_MU
    )

    v18.base.CONTACT_STIFFNESS = (
        float(
            stiffness
        )
    )


    # Outer probe remains active while V18 installs its own
    # normal contact recorder inside run_candidate().
    probe = (
        ComplianceProbe()
    )


    probe.install()


    try:

        result = v18.run_candidate(
            FIXED_MU,
            contact_bag,
        )


    finally:

        probe.restore()


    probe_metrics = analyse_probe(
        probe.rows
    )


    result.update(
        probe_metrics
    )


    # ========================================================
    # HARD VALIDATION GATE
    # ========================================================

    effective = result[
        "effective_stiffness"
    ]


    allowed_error = max(
        2.0,
        0.02
        * stiffness,
    )


    if (
        not np.isfinite(
            effective
        )
        or
        abs(
            effective
            -
            stiffness
        )
        > allowed_error
    ):

        raise RuntimeError(
            "\n"
            "V20 RUNTIME STIFFNESS VALIDATION FAILED\n"
            f"  requested stiffness = "
            f"{stiffness:.3f}\n"
            f"  effective stiffness = "
            f"{effective:.3f}\n"
            f"  allowed error = "
            f"{allowed_error:.3f}\n"
            "\n"
            "Do not interpret this scan."
        )


    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "contact_bag",
        type=Path,
    )


    parser.add_argument(
        "--no-viewer",
        action="store_true",
    )


    args = parser.parse_args()


    contact_bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )


    if not contact_bag.exists():

        raise RuntimeError(
            f"Contact bag not found:\n"
            f"{contact_bag}"
        )


    # Freeze V5 fixture.
    v18.ab.patch_replay_modules()


    original_mu = float(
        v18.base.CONTACT_MU
    )

    original_stiffness = float(
        v18.base.CONTACT_STIFFNESS
    )


    # Put V18 raw candidate files in the V20 result folder.
    old_result_dir = (
        v18.RESULT_DIR
    )

    v18.RESULT_DIR = (
        RESULT_DIR
    )


    print()
    print("=" * 118)
    print(
        "RH8D INDEX — VALIDATED RUNTIME "
        "CONTACT COMPLIANCE SCAN V20"
    )
    print("=" * 118)


    print()
    print(
        f"Contact mu frozen = "
        f"{FIXED_MU:.3f}"
    )


    print(
        f"Current runtime stiffness = "
        f"{original_stiffness:.1f}"
    )


    print()
    print(
        "Object position/fixture registration is FROZEN."
    )

    print(
        "Trusted constant tendon mechanics are FROZEN."
    )


    print()
    print(
        "Stiffness candidates [N/m]:"
    )

    print(
        CONTACT_STIFFNESS_VALUES
    )


    results = []


    try:

        for i, stiffness in enumerate(
            CONTACT_STIFFNESS_VALUES,
            start=1,
        ):

            print()
            print("=" * 118)

            print(
                f"[{i:02d}/"
                f"{len(CONTACT_STIFFNESS_VALUES):02d}] "
                f"CONTACT_STIFFNESS = "
                f"{stiffness:.1f} N/m"
            )

            print("=" * 118)


            result = run_candidate(
                stiffness,
                contact_bag,
            )


            result[
                "requested_stiffness"
            ] = stiffness


            results.append(
                result
            )


            print()
            print(
                "RUNTIME CONTACT-LAW VALIDATION:"
            )

            print(
                f"  requested k = "
                f"{stiffness:.1f} N/m"
            )

            print(
                f"  actual k = "
                f"{result['effective_stiffness']:.1f} N/m"
            )

            print(
                f"  actual solref damping = "
                f"{result['actual_solref1']:.3f}"
            )

            print(
                f"  effective mu = "
                f"{result['effective_mu']:.3f}"
            )


            print()
            print(
                "Established contact evaluator:"
            )

            print(
                "  MCP/PIP/DIP = "
                f"{result['contact_mcp']:.3f} / "
                f"{result['contact_pip']:.3f} / "
                f"{result['contact_dip']:.3f} deg"
            )

            print(
                f"  RMSE = "
                f"{result['contact_rmse']:.3f} deg"
            )


            print()
            print(
                "Contact sequence:"
            )

            print(
                f"  total middle-contact duration = "
                f"{result['middle_duration_s']:.3f} s"
            )

            print(
                f"  number of middle-contact episodes = "
                f"{result['middle_episodes']}"
            )

            print(
                f"  longest continuous middle episode = "
                f"{result['longest_middle_episode_s']:.3f} s"
            )

            print(
                f"  minimum physical middle gap = "
                f"{result['min_mid_gap_mm']:+.3f} mm"
            )

            print(
                f"  max middle Fn = "
                f"{result['max_middle_force_N']:.3f} N"
            )


            print()
            print(
                f"  tip-contact duration = "
                f"{result['tip_duration_s']:.3f} s"
            )


            if (
                result[
                    "first_tip_elapsed_s"
                ]
                is None
            ):

                print(
                    "  first tip contact = NEVER"
                )

            else:

                print(
                    f"  first tip contact = "
                    f"{result['first_tip_elapsed_s']:.3f} s"
                )


    finally:

        v18.base.CONTACT_MU = (
            original_mu
        )

        v18.base.CONTACT_STIFFNESS = (
            original_stiffness
        )

        v18.RESULT_DIR = (
            old_result_dir
        )


    # ========================================================
    # SAVE SUMMARY
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
    # BASELINE
    # ========================================================

    baseline = min(
        results,
        key=lambda r:
            abs(
                r[
                    "requested_stiffness"
                ]
                -
                1000.0
            ),
    )


    # ========================================================
    # PHYSICAL ADMISSIBILITY
    #
    # Do not select a candidate merely because a soft contact
    # gives a better angle while allowing huge interpenetration.
    #
    # We tolerate at most 1 mm of physical overlap here.
    # ========================================================

    admissible = [
        r
        for r in results
        if (
            r[
                "min_mid_gap_mm"
            ]
            >= -1.0
        )
    ]


    if not admissible:

        admissible = list(
            results
        )


    with_tip = [
        r
        for r in admissible
        if (
            r[
                "tip_duration_s"
            ]
            > 0.0
        )
    ]


    # ========================================================
    # SELECTION
    #
    # 1. A mechanically valid candidate with tip contact wins.
    #
    # 2. If tip contact never occurs:
    #    avoid candidates whose PIP becomes >0.5 deg worse
    #    than baseline, then prefer longest sustained middle
    #    contact.
    #
    # This makes the viewer show the best MECHANICAL sequence,
    # rather than blindly selecting final pose RMSE.
    # ========================================================

    if with_tip:

        selected = min(
            with_tip,
            key=lambda r:
                r[
                    "contact_rmse"
                ],
        )

        selection_mode = (
            "TIP CONTACT ACHIEVED"
        )


    else:

        not_much_worse_pip = [
            r
            for r in admissible
            if (
                r[
                    "contact_pip"
                ]
                >=
                baseline[
                    "contact_pip"
                ]
                -
                0.5
            )
        ]


        pool = (
            not_much_worse_pip
            if not_much_worse_pip
            else admissible
        )


        selected = max(
            pool,
            key=lambda r:
                (
                    r[
                        "longest_middle_episode_s"
                    ],
                    r[
                        "contact_pip"
                    ],
                    -r[
                        "contact_rmse"
                    ],
                ),
        )


        selection_mode = (
            "BEST SUSTAINED MIDDLE CONTACT; "
            "NO TIP CONTACT"
        )


    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 118)
    print(
        "V20 BASELINE VS SELECTED"
    )
    print("=" * 118)


    print()
    print(
        "BASELINE:"
    )

    print(
        f"  k = "
        f"{baseline['requested_stiffness']:.1f} N/m"
    )

    print(
        "  MCP/PIP/DIP = "
        f"{baseline['contact_mcp']:.3f} / "
        f"{baseline['contact_pip']:.3f} / "
        f"{baseline['contact_dip']:.3f} deg"
    )

    print(
        f"  middle episodes = "
        f"{baseline['middle_episodes']}"
    )

    print(
        f"  longest middle contact = "
        f"{baseline['longest_middle_episode_s']:.3f} s"
    )

    print(
        f"  min physical gap = "
        f"{baseline['min_mid_gap_mm']:+.3f} mm"
    )


    print()
    print(
        "SELECTED:"
    )

    print(
        f"  selection = "
        f"{selection_mode}"
    )

    print(
        f"  k requested/effective = "
        f"{selected['requested_stiffness']:.1f} / "
        f"{selected['effective_stiffness']:.1f} N/m"
    )

    print(
        "  MCP/PIP/DIP = "
        f"{selected['contact_mcp']:.3f} / "
        f"{selected['contact_pip']:.3f} / "
        f"{selected['contact_dip']:.3f} deg"
    )

    print(
        f"  RMSE = "
        f"{selected['contact_rmse']:.3f} deg"
    )

    print(
        f"  middle episodes = "
        f"{selected['middle_episodes']}"
    )

    print(
        f"  longest middle contact = "
        f"{selected['longest_middle_episode_s']:.3f} s"
    )

    print(
        f"  min physical gap = "
        f"{selected['min_mid_gap_mm']:+.3f} mm"
    )

    print(
        f"  tip duration = "
        f"{selected['tip_duration_s']:.3f} s"
    )


    print()
    print(
        "Change from baseline:"
    )

    print(
        f"  dMCP = "
        f"{selected['contact_mcp'] - baseline['contact_mcp']:+.3f} deg"
    )

    print(
        f"  dPIP = "
        f"{selected['contact_pip'] - baseline['contact_pip']:+.3f} deg"
    )

    print(
        f"  dDIP = "
        f"{selected['contact_dip'] - baseline['contact_dip']:+.3f} deg"
    )


    # ========================================================
    # BUILD SELECTED MODEL
    # ========================================================

    v18.build_candidate(
        FIXED_MU,
        SELECTED_MODEL,
    )


    print()
    print(
        "Selected model:"
    )

    print(
        SELECTED_MODEL
    )


    print()
    print(
        "Summary:"
    )

    print(
        SUMMARY
    )


    # ========================================================
    # DECISION
    # ========================================================

    print()
    print("=" * 118)
    print(
        "V20 MECHANICAL DECISION"
    )
    print("=" * 118)


    if (
        selected[
            "tip_duration_s"
        ]
        > 0.0
    ):

        print()
        print(
            "A normal-compliance change produced the missing "
            "middle-contact -> distal-contact transition."
        )

        print(
            "Inspect the automatic replay carefully before "
            "doing any further parameter identification."
        )


    elif (
        selected[
            "longest_middle_episode_s"
        ]
        >
        baseline[
            "longest_middle_episode_s"
        ]
        * 1.25
    ):

        print()
        print(
            "Softer contact substantially improves sustained "
            "middle contact, but the tip still does not wrap."
        )

        print(
            "Compliance contributes to the bounce problem, "
            "but tendon/passive redistribution is still missing."
        )


    else:

        print()
        print(
            "Changing verified normal compliance does not "
            "substantially fix the contact sequence."
        )

        print(
            "Then the hard-looking impact is not the main "
            "cause of missing PIP/distal wrapping."
        )

        print(
            "The next step should return to the tendon/passive "
            "underactuated mechanics, while retaining the "
            "validated contact model."
        )


    # ========================================================
    # AUTOMATIC BEST-CANDIDATE VIEWER
    # ========================================================

    if not args.no_viewer:

        print()
        print("=" * 118)
        print(
            "V20 AUTOMATIC BEST-CANDIDATE MUJOCO REPLAY"
        )
        print("=" * 118)


        print()
        print(
            f"Runtime mu = "
            f"{FIXED_MU:.3f}"
        )

        print(
            f"Runtime contact stiffness = "
            f"{selected['requested_stiffness']:.1f} N/m"
        )

        print(
            "Object position remains unchanged."
        )

        print(
            "The trusted constant tendon remains active."
        )


        old_mu = float(
            v18.base.CONTACT_MU
        )

        old_stiffness = float(
            v18.base.CONTACT_STIFFNESS
        )


        try:

            v18.base.CONTACT_MU = (
                FIXED_MU
            )

            v18.base.CONTACT_STIFFNESS = (
                selected[
                    "requested_stiffness"
                ]
            )


            # Native viewer.
            #
            # V18 visual_replay does NOT request custom
            # width/height, avoiding the previous screen-size
            # error on this machine.
            v18.visual_replay(
                SELECTED_MODEL,
                contact_bag,
            )


        finally:

            v18.base.CONTACT_MU = (
                old_mu
            )

            v18.base.CONTACT_STIFFNESS = (
                old_stiffness
            )


    print()
    print("=" * 118)
    print(
        "V20 COMPLETE"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
