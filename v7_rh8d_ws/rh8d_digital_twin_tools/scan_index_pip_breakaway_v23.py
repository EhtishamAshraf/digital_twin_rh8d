#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import math

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as base
import compare_index_configdep_replay_v6 as ab
import scan_and_view_index_contact_friction_v18 as viewer_tools


# ============================================================
# RH8D INDEX — LOAD-TRIGGERED PIP BREAKAWAY V23
#
# PURPOSE
#
# V22 established:
#
#   baseline contact:
#       MCP/PIP/DIP = 25.999 / 3.206 / 7.849 deg
#
#   zero PIP spring + friction:
#       10.144 / 33.773 / 1.134 deg
#
# Therefore:
#
#   - low PIP resistance can unlock contact redistribution
#   - permanently low resistance destroys free space
#
# V23 tests a mechanically triggered yielding return element.
#
# BEFORE YIELD:
#
#   normal baseline PIP spring/friction
#
# AFTER:
#
#   |PIP generalized constraint load| >= threshold
#
# the passive return element changes to a lower-resistance
# branch.
#
# NO object-contact flag is used.
# NO bottle ID is involved in triggering breakaway.
#
# Object position remains frozen V5.
# ============================================================


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()

MJCF_DIR = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "pip_breakaway_v23"
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
    MJCF_DIR
    / "ur5e_rh8d_index_pip_breakaway_v23_candidate.xml"
)


# ============================================================
# FROZEN TRUSTED MODEL
# ============================================================

SCALE = 5.10e-6

TENDON_K = 2100.0

CONTACT_MU = 0.05

CONTACT_K = 1000.0

OPEN_ENCODER = 100.0


PIP = "Index_Middle--Index_Proximal"

SPOOL = "index_motor_spool"


HW_CONTACT = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


FREE_GATE = 6.085


# ============================================================
# BASELINE + SMALL ARCHITECTURE GRID
# ============================================================

CASES = [

    {
        "name":
            "BASELINE",

        "threshold":
            None,

        "spring_scale":
            1.0,

        "friction_scale":
            1.0,
    },
]


for threshold in [
    0.015,
    0.030,
]:

    for spring_scale in [
        0.00,
        0.25,
        0.50,
    ]:

        for friction_scale in [
            0.00,
            0.25,
        ]:

            CASES.append(
                {
                    "name":
                        (
                            f"T{threshold:.3f}"
                            f"_KS{spring_scale:.2f}"
                            f"_FS{friction_scale:.2f}"
                        ),

                    "threshold":
                        threshold,

                    "spring_scale":
                        spring_scale,

                    "friction_scale":
                        friction_scale,
                }
            )


# ============================================================
# BUILD FROZEN CONSTANT-TENDON CONTROL
# ============================================================

def build_baseline(
    output,
):

    ab.replace_index_with_constant_tendon(
        ab.V5_SOURCE,
        output,
    )


# ============================================================
# STATEFUL PIP BREAKAWAY LAW
# ============================================================

class BreakawayLaw:

    def __init__(
        self,
    ):

        self.original_step = (
            mujoco.mj_step
        )

        self.case = None

        self.phase = None

        self.states = {}

        self.cache = {}

        self.events = []


    def configure(
        self,
        case,
        phase,
    ):

        self.case = case

        self.phase = phase


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


        pip_jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            PIP,
        )


        spool_jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            SPOOL,
        )


        if (
            pip_jid < 0
            or
            spool_jid < 0
        ):

            self.cache[
                key
            ] = None

            return None


        result = {

            "pip_jid":
                pip_jid,

            "pip_dof":
                int(
                    model.jnt_dofadr[
                        pip_jid
                    ]
                ),

            "pip_q":
                int(
                    model.jnt_qposadr[
                        pip_jid
                    ]
                ),

            "spool_q":
                int(
                    model.jnt_qposadr[
                        spool_jid
                    ]
                ),
        }


        self.cache[
            key
        ] = result

        return result


    def state(
        self,
        model,
        ids,
    ):

        key = id(
            model
        )


        if key not in self.states:

            self.states[
                key
            ] = {

                "yielded":
                    False,

                "baseline_k":
                    float(
                        model.jnt_stiffness[
                            ids[
                                "pip_jid"
                            ]
                        ]
                    ),

                "baseline_f":
                    float(
                        model.dof_frictionloss[
                            ids[
                                "pip_dof"
                            ]
                        ]
                    ),

                "event_written":
                    False,
            }


        return self.states[
            key
        ]


    def apply_branch(
        self,
        model,
        ids,
        state,
    ):

        if (
            self.case is None
            or
            self.case[
                "threshold"
            ]
            is None
            or
            not state[
                "yielded"
            ]
        ):

            model.jnt_stiffness[
                ids[
                    "pip_jid"
                ]
            ] = (
                state[
                    "baseline_k"
                ]
            )


            model.dof_frictionloss[
                ids[
                    "pip_dof"
                ]
            ] = (
                state[
                    "baseline_f"
                ]
            )


        else:

            model.jnt_stiffness[
                ids[
                    "pip_jid"
                ]
            ] = (
                state[
                    "baseline_k"
                ]
                *
                self.case[
                    "spring_scale"
                ]
            )


            model.dof_frictionloss[
                ids[
                    "pip_dof"
                ]
            ] = (
                state[
                    "baseline_f"
                ]
                *
                self.case[
                    "friction_scale"
                ]
            )


    def install(
        self,
    ):

        original = (
            self.original_step
        )

        law = self


        def wrapped(
            model,
            data,
            *args,
            **kwargs,
        ):

            if law.case is None:

                return original(
                    model,
                    data,
                    *args,
                    **kwargs,
                )


            ids = law.ids(
                model
            )


            if ids is None:

                return original(
                    model,
                    data,
                    *args,
                    **kwargs,
                )


            state = law.state(
                model,
                ids,
            )


            # Apply current branch BEFORE integration.
            law.apply_branch(
                model,
                ids,
                state,
            )


            result = original(
                model,
                data,
                *args,
                **kwargs,
            )


            threshold = law.case[
                "threshold"
            ]


            # ------------------------------------------------
            # Mechanical trigger.
            #
            # Uses only generalized PIP solver/constraint load.
            #
            # Free-space dry-friction scale is ~0.006 Nm,
            # while V21 bottle-impact constraint torque was
            # much larger (~0.07 Nm).
            #
            # No contact flag or object identity is used.
            # ------------------------------------------------

            if (
                threshold is not None
                and
                not state[
                    "yielded"
                ]
            ):

                constraint_load = abs(
                    float(
                        data.qfrc_constraint[
                            ids[
                                "pip_dof"
                            ]
                        ]
                    )
                )


                if (
                    constraint_load
                    >= threshold
                ):

                    state[
                        "yielded"
                    ] = True


                    encoder_est = (
                        OPEN_ENCODER
                        +
                        float(
                            data.qpos[
                                ids[
                                    "spool_q"
                                ]
                            ]
                        )
                        /
                        SCALE
                    )


                    event = {

                        "case":
                            law.case[
                                "name"
                            ],

                        "phase":
                            law.phase,

                        "time":
                            float(
                                data.time
                            ),

                        "encoder":
                            encoder_est,

                        "pip_deg":
                            math.degrees(
                                float(
                                    data.qpos[
                                        ids[
                                            "pip_q"
                                        ]
                                    ]
                                )
                            ),

                        "constraint_load_Nm":
                            constraint_load,
                    }


                    law.events.append(
                        event
                    )


                    print(
                        "\n"
                        "[BREAKAWAY] "
                        f"{event['case']} "
                        f"{event['phase']} "
                        f"t={event['time']:.3f} "
                        f"encoder={event['encoder']:.1f} "
                        f"PIP={event['pip_deg']:.2f} deg "
                        f"|constraint|="
                        f"{event['constraint_load_Nm']:.4f} Nm"
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
# CONTACT SEQUENCE ANALYSIS
# ============================================================

def first_active(
    rows,
    prefix,
):

    if not rows:

        return None


    t0 = rows[
        0
    ][
        "time"
    ]


    for r in rows:

        if (
            r[
                f"{prefix}_rows"
            ]
            > 0
            and
            r[
                f"{prefix}_Fn_N"
            ]
            > 1.0e-4
        ):

            return (
                float(
                    r[
                        "time"
                    ]
                    -
                    t0
                )
            )


    return None


def active_duration(
    rows,
    prefix,
):

    if len(
        rows
    ) < 2:

        return 0.0


    dt = float(
        np.median(
            np.diff(
                [
                    r[
                        "time"
                    ]
                    for r in rows
                ]
            )
        )
    )


    count = sum(
        1
        for r in rows
        if (
            r[
                f"{prefix}_rows"
            ]
            > 0
            and
            r[
                f"{prefix}_Fn_N"
            ]
            > 1.0e-4
        )
    )


    return (
        count
        * dt
    )


def first_event(
    events,
    case_name,
    phase,
):

    for event in events:

        if (
            event[
                "case"
            ]
            == case_name
            and
            event[
                "phase"
            ]
            == phase
        ):

            return event


    return None


# ============================================================
# RUN CASE
# ============================================================

def run_case(
    case,
    free_bag,
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
        f"tmp_v23_{label}_contact.xml"
    )


    free_xml = (
        MJCF_DIR
        /
        f"tmp_v23_{label}_free.xml"
    )


    build_baseline(
        contact_xml
    )


    ab.make_free_model(
        contact_xml,
        free_xml,
    )


    law = (
        BreakawayLaw()
    )


    law.install()


    old_scale = (
        base.METERS_PER_TICK
    )

    old_tendon = (
        base.TENDON_STIFFNESS
    )

    old_mu = (
        base.CONTACT_MU
    )

    old_contact_k = (
        base.CONTACT_STIFFNESS
    )


    try:

        base.METERS_PER_TICK = (
            SCALE
        )

        base.TENDON_STIFFNESS = (
            TENDON_K
        )

        base.CONTACT_MU = (
            CONTACT_MU
        )

        base.CONTACT_STIFFNESS = (
            CONTACT_K
        )


        # ----------------------------------------------------
        # FREE SPACE
        # ----------------------------------------------------

        law.configure(
            case,
            "FREE",
        )


        free_metrics, _ = (
            base.run_free(
                free_xml,
                free_bag,
            )
        )


        # ----------------------------------------------------
        # CONTACT + sequence recorder
        # ----------------------------------------------------

        recorder = (
            viewer_tools.ContactRecorder()
        )


        recorder.install()


        try:

            law.configure(
                case,
                "CONTACT",
            )


            contact_metrics, _ = (
                base.run_contact(
                    contact_xml,
                    contact_bag,
                )
            )


        finally:

            recorder.restore()


    finally:

        law.restore()

        base.METERS_PER_TICK = (
            old_scale
        )

        base.TENDON_STIFFNESS = (
            old_tendon
        )

        base.CONTACT_MU = (
            old_mu
        )

        base.CONTACT_STIFFNESS = (
            old_contact_k
        )


    pose = np.asarray(
        contact_metrics[
            "pose"
        ],
        dtype=float,
    )


    contact_rmse = float(
        np.sqrt(
            np.mean(
                (
                    pose
                    -
                    HW_CONTACT
                )
                ** 2
            )
        )
    )


    free_event = first_event(
        law.events,
        case[
            "name"
        ],
        "FREE",
    )


    contact_event = first_event(
        law.events,
        case[
            "name"
        ],
        "CONTACT",
    )


    mid_first = first_active(
        recorder.rows,
        "mid",
    )


    tip_first = first_active(
        recorder.rows,
        "tip",
    )


    middle_duration = active_duration(
        recorder.rows,
        "mid",
    )


    tip_duration = active_duration(
        recorder.rows,
        "tip",
    )


    # Contact recorder local time origin.
    if (
        contact_event is not None
        and
        recorder.rows
    ):

        contact_t0 = recorder.rows[
            0
        ][
            "time"
        ]


        break_elapsed = (
            contact_event[
                "time"
            ]
            -
            contact_t0
        )


    else:

        break_elapsed = None


    sequence_ok = (
        mid_first is not None
        and
        break_elapsed is not None
        and
        tip_first is not None
        and
        mid_first
        <=
        break_elapsed
        <=
        tip_first
    )


    result = {

        "name":
            case[
                "name"
            ],

        "threshold_Nm":
            (
                case[
                    "threshold"
                ]
                if case[
                    "threshold"
                ]
                is not None
                else -1.0
            ),

        "spring_scale":
            case[
                "spring_scale"
            ],

        "friction_scale":
            case[
                "friction_scale"
            ],

        "free_mcp_rmse":
            free_metrics[
                "mcp_rmse"
            ],

        "free_pip_rmse":
            free_metrics[
                "pip_rmse"
            ],

        "free_combined":
            free_metrics[
                "combined"
            ],

        "free_breakaway":
            (
                free_event
                is not None
            ),

        "contact_mcp":
            pose[
                0
            ],

        "contact_pip":
            pose[
                1
            ],

        "contact_dip":
            pose[
                2
            ],

        "contact_rmse":
            contact_rmse,

        "contact_breakaway":
            (
                contact_event
                is not None
            ),

        "break_elapsed_s":
            (
                break_elapsed
                if break_elapsed
                is not None
                else -1.0
            ),

        "break_encoder":
            (
                contact_event[
                    "encoder"
                ]
                if contact_event
                is not None
                else -1.0
            ),

        "break_pip_deg":
            (
                contact_event[
                    "pip_deg"
                ]
                if contact_event
                is not None
                else -1.0
            ),

        "middle_first_s":
            (
                mid_first
                if mid_first
                is not None
                else -1.0
            ),

        "tip_first_s":
            (
                tip_first
                if tip_first
                is not None
                else -1.0
            ),

        "middle_duration_s":
            middle_duration,

        "tip_duration_s":
            tip_duration,

        "sequence_ok":
            sequence_ok,
    }


    try:
        contact_xml.unlink()
    except FileNotFoundError:
        pass


    try:
        free_xml.unlink()
    except FileNotFoundError:
        pass


    return result


# ============================================================
# VISUAL REPLAY OF SELECTED CASE
# ============================================================

def visual_replay(
    case,
    contact_bag,
):

    build_baseline(
        SELECTED_MODEL
    )


    law = (
        BreakawayLaw()
    )


    law.install()


    old_scale = (
        viewer_tools.base.METERS_PER_TICK
    )

    old_tendon = (
        viewer_tools.base.TENDON_STIFFNESS
    )

    old_mu = (
        viewer_tools.base.CONTACT_MU
    )

    old_k = (
        viewer_tools.base.CONTACT_STIFFNESS
    )


    try:

        viewer_tools.base.METERS_PER_TICK = (
            SCALE
        )

        viewer_tools.base.TENDON_STIFFNESS = (
            TENDON_K
        )

        viewer_tools.base.CONTACT_MU = (
            CONTACT_MU
        )

        viewer_tools.base.CONTACT_STIFFNESS = (
            CONTACT_K
        )


        law.configure(
            case,
            "CONTACT",
        )


        viewer_tools.visual_replay(
            SELECTED_MODEL,
            contact_bag,
        )


    finally:

        law.restore()

        viewer_tools.base.METERS_PER_TICK = (
            old_scale
        )

        viewer_tools.base.TENDON_STIFFNESS = (
            old_tendon
        )

        viewer_tools.base.CONTACT_MU = (
            old_mu
        )

        viewer_tools.base.CONTACT_STIFFNESS = (
            old_k
        )


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


    if not free_bag.exists():

        raise RuntimeError(
            f"Missing free-space bag:\n"
            f"{free_bag}"
        )


    if not contact_bag.exists():

        raise RuntimeError(
            f"Missing contact bag:\n"
            f"{contact_bag}"
        )


    # Keep V5 fixture registration frozen everywhere.
    ab.patch_replay_modules()


    print()
    print("=" * 122)
    print(
        "RH8D INDEX — LOAD-TRIGGERED "
        "PIP BREAKAWAY SCAN V23"
    )
    print("=" * 122)


    print()
    print(
        "OBJECT POSITION: frozen V5"
    )

    print(
        f"contact mu = "
        f"{CONTACT_MU}"
    )

    print(
        f"contact stiffness = "
        f"{CONTACT_K:.1f} N/m"
    )

    print(
        "constant tendon = "
        "5 / 4 / 1 mm"
    )


    print()
    print(
        f"Testing {len(CASES)} cases."
    )


    results = []


    for index, case in enumerate(
        CASES,
        start=1,
    ):

        print()
        print("=" * 122)

        print(
            f"[{index:02d}/{len(CASES):02d}] "
            f"{case['name']}"
        )

        print("=" * 122)


        result = run_case(
            case,
            free_bag,
            contact_bag,
        )


        results.append(
            result
        )


        print()
        print(
            "Free-space:"
        )

        print(
            f"  RMSE = "
            f"{result['free_combined']:.3f} deg"
        )

        print(
            f"  breakaway occurred = "
            f"{result['free_breakaway']}"
        )


        print()
        print(
            "Contact:"
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
            f"  middle first = "
            f"{result['middle_first_s']:.3f} s"
        )

        print(
            f"  breakaway = "
            f"{result['break_elapsed_s']:.3f} s"
        )

        print(
            f"  tip first = "
            f"{result['tip_first_s']:.3f} s"
        )

        print(
            f"  tip duration = "
            f"{result['tip_duration_s']:.3f} s"
        )

        print(
            f"  desired sequence = "
            f"{result['sequence_ok']}"
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
    # SELECT ONLY FREE-SPACE-PRESERVING MODELS
    # ========================================================

    preserving = [
        r
        for r in results
        if (
            r[
                "free_combined"
            ]
            <= FREE_GATE
            and
            not r[
                "free_breakaway"
            ]
        )
    ]


    if not preserving:

        raise RuntimeError(
            "No V23 model preserved free-space behavior."
        )


    sequence_candidates = [
        r
        for r in preserving
        if r[
            "sequence_ok"
        ]
    ]


    tip_candidates = [
        r
        for r in preserving
        if (
            r[
                "tip_duration_s"
            ]
            > 0.0
        )
    ]


    if sequence_candidates:

        selected = min(
            sequence_candidates,
            key=lambda r:
                r[
                    "contact_rmse"
                ],
        )

        mode = (
            "CORRECT MECHANICAL SEQUENCE"
        )


    elif tip_candidates:

        selected = min(
            tip_candidates,
            key=lambda r:
                r[
                    "contact_rmse"
                ],
        )

        mode = (
            "TIP CONTACT ACHIEVED"
        )


    else:

        selected = min(
            preserving,
            key=lambda r:
                r[
                    "contact_rmse"
                ],
        )

        mode = (
            "NO TIP CONTACT — LOWEST CONTACT RMSE"
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


    baseline = next(
        r
        for r in results
        if (
            r[
                "name"
            ]
            ==
            "BASELINE"
        )
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 122)
    print(
        "V23 SELECTED ARCHITECTURE"
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


    print()
    print(
        f"Threshold = "
        f"{selected['threshold_Nm']:.4f} Nm"
    )

    print(
        f"Post-yield spring scale = "
        f"{selected['spring_scale']:.2f}"
    )

    print(
        f"Post-yield friction scale = "
        f"{selected['friction_scale']:.2f}"
    )


    print()
    print(
        f"Free-space RMSE = "
        f"{selected['free_combined']:.3f} deg"
    )

    print(
        f"Free-space breakaway = "
        f"{selected['free_breakaway']}"
    )


    print()
    print(
        "Contact target:"
    )

    print(
        "  14.536 / 23.811 / 9.327 deg"
    )


    print(
        "Selected prediction:"
    )

    print(
        "  "
        f"{selected['contact_mcp']:.3f} / "
        f"{selected['contact_pip']:.3f} / "
        f"{selected['contact_dip']:.3f} deg"
    )


    print(
        f"Contact RMSE = "
        f"{selected['contact_rmse']:.3f} deg"
    )


    print()
    print(
        "Baseline contact RMSE = "
        f"{baseline['contact_rmse']:.3f} deg"
    )


    print(
        "Improvement = "
        f"{baseline['contact_rmse'] - selected['contact_rmse']:+.3f} deg"
    )


    print()
    print(
        "Event sequence:"
    )

    print(
        f"  middle contact = "
        f"{selected['middle_first_s']:.3f} s"
    )

    print(
        f"  breakaway      = "
        f"{selected['break_elapsed_s']:.3f} s"
    )

    print(
        f"  tip contact    = "
        f"{selected['tip_first_s']:.3f} s"
    )

    print(
        f"  sequence OK    = "
        f"{selected['sequence_ok']}"
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
    print("=" * 122)
    print(
        "V23 STRUCTURAL DECISION"
    )
    print("=" * 122)


    success = (
        selected[
            "free_combined"
        ]
        <= FREE_GATE
        and
        selected[
            "contact_pip"
        ]
        >= 18.0
        and
        selected[
            "tip_duration_s"
        ]
        > 0.0
    )


    if success:

        print()
        print(
            "SUCCESSFUL ARCHITECTURE TEST."
        )

        print()
        print(
            "The load-triggered PIP return element preserves "
            "the trusted free-space behavior while enabling "
            "substantial contact-induced PIP redistribution "
            "and distal contact."
        )

        print()
        print(
            "STOP broad architecture searching."
        )

        print(
            "NEXT: perform a narrow V24 identification of "
            "breakaway threshold and released-state resistance, "
            "then collect held-out hardware contact trials."
        )


    elif (
        selected[
            "contact_pip"
        ]
        >= 15.0
    ):

        print()
        print(
            "PARTIAL STRUCTURAL SUCCESS."
        )

        print()
        print(
            "The mechanism produces strong PIP redistribution "
            "while preserving free-space behavior, but the "
            "full middle -> PIP -> distal sequence is not yet "
            "reproduced."
        )

        print()
        print(
            "NEXT: one narrow refinement only. Do not return "
            "to broad contact/friction/stiffness scans."
        )


    else:

        print()
        print(
            "BREAKAWAY PASSIVE MODEL IS NOT SUFFICIENT."
        )

        print()
        print(
            "Do not spend further time tuning this architecture."
        )

        print(
            "NEXT: move directly to a coupled stateful tendon "
            "transmission / return-element model."
        )


    # ========================================================
    # GUI — SELECTED CANDIDATE ONLY
    # ========================================================

    if not args.no_viewer:

        print()
        print("=" * 122)
        print(
            "V23 AUTOMATIC SELECTED-CANDIDATE GUI REPLAY"
        )
        print("=" * 122)


        print()
        print(
            f"Viewing: "
            f"{selected['name']}"
        )

        print(
            "Bottle position remains frozen V5."
        )

        print(
            "No custom viewer dimensions are used."
        )


        visual_replay(
            selected_case,
            contact_bag,
        )


    print()
    print("=" * 122)
    print(
        "V23 COMPLETE"
    )
    print("=" * 122)


if __name__ == "__main__":
    main()
