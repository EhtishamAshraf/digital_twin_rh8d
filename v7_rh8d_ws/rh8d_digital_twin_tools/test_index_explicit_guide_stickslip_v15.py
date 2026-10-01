#!/usr/bin/env python3

from pathlib import Path
import csv
import math

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as base
import compare_index_configdep_replay_v6 as ab
import diagnose_index_guide_slip_v13 as v13


# ============================================================
# V15 — EXPLICIT STATEFUL GUIDE STICK/SLIP
#
# PURPOSE:
#
# MuJoCo joint frictionloss was shown in V14 to provide only
# ~mN of actual solver force even when its bound was 4 N.
#
# Here we explicitly implement:
#
#       STICK
#          |
#          | |drive| > F_static
#          v
#       SLIDE
#
# During STICK:
#
#   applied guide force = -mechanical drive
#
# up to the static breakaway limit.
#
# During SLIDE:
#
#   kinetic friction opposes motion.
#
# NO object-contact flag enters this law.
#
# V15 is an architectural diagnostic, NOT parameter ID.
# ============================================================


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()

MJCF_DIR = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

SOURCE = (
    MJCF_DIR
    / "ur5e_rh8d_index_proximal_guide_split_v11.xml"
)

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "explicit_guide_stickslip_v15"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    RESULT_DIR
    / "stickslip_timeseries.csv"
)


# ============================================================
# TRUSTED MODEL
# ============================================================

SCALE = 5.10e-6
K_FINGER = 2192.0

SLIP = "index_tendon_slip"
MOTOR_TENDON = "idx_motor_segment"
DISTAL_TENDON = "idx_flex_spatial"


# ============================================================
# TWO TESTS ONLY
#
# Nonzero values are DIAGNOSTIC, not identified.
# ============================================================

TESTS = [
    {
        "name":
            "BASELINE",

        "F_static":
            0.0,

        "F_kinetic":
            0.0,
    },

    {
        "name":
            "EXPLICIT_STICKSLIP",

        "F_static":
            1.0,

        "F_kinetic":
            0.25,
    },
]


# Re-stick only after motion becomes very small.
V_RESTICK = 5.0e-5

# Require load to fall below half the breakaway threshold
# before returning to the stuck state.
RESTICK_RATIO = 0.50

HOLD_ELAPSED = 6.297


# ============================================================
# BUILD MODEL
#
# IMPORTANT:
# remove MuJoCo frictionloss completely.
# ============================================================

def build_model(
    output,
):

    v13.build_candidate(
        0.0,
        output,
    )


# ============================================================
# CUSTOM GUIDE LAW
# ============================================================

class ExplicitGuideLaw:

    def __init__(self):

        self.original_step = (
            mujoco.mj_step
        )

        self.current_test = None
        self.phase = None

        self.states = {}
        self.ids_cache = {}

        self.rows = []


    def configure(
        self,
        test,
        phase,
    ):

        self.current_test = test
        self.phase = phase


    def get_ids(
        self,
        model,
    ):

        key = id(
            model
        )


        if key in self.ids_cache:

            return self.ids_cache[
                key
            ]


        slip_jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            SLIP,
        )


        motor_tid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            MOTOR_TENDON,
        )


        distal_tid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            DISTAL_TENDON,
        )


        if (
            slip_jid < 0
            or
            motor_tid < 0
            or
            distal_tid < 0
        ):

            self.ids_cache[
                key
            ] = None

            return None


        result = {

            "jid":
                slip_jid,

            "qadr":
                int(
                    model.jnt_qposadr[
                        slip_jid
                    ]
                ),

            "dof":
                int(
                    model.jnt_dofadr[
                        slip_jid
                    ]
                ),

            "motor_tid":
                motor_tid,

            "distal_tid":
                distal_tid,
        }


        self.ids_cache[
            key
        ] = result

        return result


    def get_state(
        self,
        model,
    ):

        key = id(
            model
        )


        if key not in self.states:

            F_static = float(
                self.current_test[
                    "F_static"
                ]
            )


            self.states[
                key
            ] = {

                "mode":
                    (
                        "STICK"
                        if F_static > 0.0
                        else "FREE"
                    ),

                "breakaways":
                    0,

                "resticks":
                    0,
            }


        return self.states[
            key
        ]


    def pre_step(
        self,
        model,
        data,
        ids,
        state,
    ):

        dof = ids[
            "dof"
        ]


        # Clear only our hidden-guide external force.
        data.qfrc_applied[
            dof
        ] = 0.0


        # Update passive tendon forces for the CURRENT state.
        mujoco.mj_forward(
            model,
            data,
        )


        # Net mechanical drive on the hidden guide.
        #
        # For this world-attached dummy slide:
        #
        #   qfrc_passive ~= Tmotor - Tdistal
        #
        # subtract bias for completeness.
        drive = (
            float(
                data.qfrc_passive[
                    dof
                ]
            )
            -
            float(
                data.qfrc_bias[
                    dof
                ]
            )
        )


        velocity = float(
            data.qvel[
                dof
            ]
        )


        F_static = float(
            self.current_test[
                "F_static"
            ]
        )

        F_kinetic = float(
            self.current_test[
                "F_kinetic"
            ]
        )


        applied = 0.0


        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        if F_static <= 0.0:

            state[
                "mode"
            ] = "FREE"

            applied = 0.0


        # ----------------------------------------------------
        # STICK
        #
        # Exactly oppose the internal mechanical drive while
        # it lies inside the static-friction envelope.
        # ----------------------------------------------------

        elif (
            state[
                "mode"
            ]
            == "STICK"
        ):

            if abs(
                drive
            ) <= F_static:

                applied = (
                    -drive
                )


            else:

                state[
                    "mode"
                ] = "SLIDE"

                state[
                    "breakaways"
                ] += 1


                direction = (
                    1.0
                    if drive > 0.0
                    else -1.0
                )


                applied = (
                    -F_kinetic
                    * direction
                )


        # ----------------------------------------------------
        # SLIDE
        # ----------------------------------------------------

        else:

            if abs(
                velocity
            ) > 1.0e-7:

                direction = (
                    1.0
                    if velocity > 0.0
                    else -1.0
                )


            elif abs(
                drive
            ) > 1.0e-12:

                direction = (
                    1.0
                    if drive > 0.0
                    else -1.0
                )


            else:

                direction = 0.0


            applied = (
                -F_kinetic
                * direction
            )


            restick_limit = (
                RESTICK_RATIO
                * F_static
            )


            if (
                abs(
                    velocity
                )
                <= V_RESTICK
                and
                abs(
                    drive
                )
                <= restick_limit
            ):

                state[
                    "mode"
                ] = "STICK"

                state[
                    "resticks"
                ] += 1

                applied = (
                    -drive
                )


        data.qfrc_applied[
            dof
        ] = applied


        return (
            drive,
            applied,
        )


    def post_step(
        self,
        model,
        data,
        ids,
        state,
        drive,
        applied,
    ):

        motor = v13.tendon_state(
            model,
            data,
            ids[
                "motor_tid"
            ],
        )


        distal = v13.tendon_state(
            model,
            data,
            ids[
                "distal_tid"
            ],
        )


        slip_q = float(
            data.qpos[
                ids[
                    "qadr"
                ]
            ]
        )


        slip_v = float(
            data.qvel[
                ids[
                    "dof"
                ]
            ]
        )


        self.rows.append(
            {

                "test":
                    self.current_test[
                        "name"
                    ],

                "phase":
                    self.phase,

                "F_static_N":
                    self.current_test[
                        "F_static"
                    ],

                "F_kinetic_N":
                    self.current_test[
                        "F_kinetic"
                    ],

                "time":
                    float(
                        data.time
                    ),

                "mode":
                    state[
                        "mode"
                    ],

                "breakaways":
                    state[
                        "breakaways"
                    ],

                "resticks":
                    state[
                        "resticks"
                    ],

                "slip_mm":
                    slip_q
                    * 1000.0,

                "slip_velocity_mps":
                    slip_v,

                "mechanical_drive_N":
                    drive,

                "applied_guide_force_N":
                    applied,

                "motor_tension_N":
                    motor[
                        "tension_N"
                    ],

                "distal_tension_N":
                    distal[
                        "tension_N"
                    ],

                "Tm_minus_Td_N":
                    (
                        motor[
                            "tension_N"
                        ]
                        -
                        distal[
                            "tension_N"
                        ]
                    ),
            }
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

            if (
                law.current_test
                is None
            ):

                return original(
                    model,
                    data,
                    *args,
                    **kwargs,
                )


            ids = law.get_ids(
                model
            )


            if ids is None:

                return original(
                    model,
                    data,
                    *args,
                    **kwargs,
                )


            state = law.get_state(
                model
            )


            drive, applied = (
                law.pre_step(
                    model,
                    data,
                    ids,
                    state,
                )
            )


            result = original(
                model,
                data,
                *args,
                **kwargs,
            )


            law.post_step(
                model,
                data,
                ids,
                state,
                drive,
                applied,
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
# ANALYSIS
# ============================================================

def percentile(
    values,
    p,
):

    if not values:

        return float(
            "nan"
        )


    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=float,
            ),
            p,
        )
    )


def report_internal(
    name,
    rows,
):

    print()
    print(
        name
    )

    print(
        "-" * 90
    )


    if not rows:

        print(
            "  no samples"
        )

        return


    slip = [
        r[
            "slip_mm"
        ]
        for r in rows
    ]


    gap = [
        abs(
            r[
                "Tm_minus_Td_N"
            ]
        )
        for r in rows
    ]


    force = [
        abs(
            r[
                "applied_guide_force_N"
            ]
        )
        for r in rows
    ]


    stuck = [
        r[
            "mode"
        ]
        == "STICK"
        for r in rows
    ]


    print(
        f"  samples = "
        f"{len(rows)}"
    )


    print(
        f"  slip range = "
        f"{min(slip):+.4f} .. "
        f"{max(slip):+.4f} mm"
    )


    print(
        f"  final slip = "
        f"{slip[-1]:+.4f} mm"
    )


    print(
        f"  stuck fraction = "
        f"{100*np.mean(stuck):.1f}%"
    )


    print(
        f"  total breakaways = "
        f"{rows[-1]['breakaways']}"
    )


    print(
        f"  total resticks = "
        f"{rows[-1]['resticks']}"
    )


    print(
        "  |Tm-Td| N:"
    )

    print(
        f"    median = "
        f"{percentile(gap,50):.3f}"
    )

    print(
        f"    p95 = "
        f"{percentile(gap,95):.3f}"
    )

    print(
        f"    max = "
        f"{max(gap):.3f}"
    )


    print(
        "  |explicit guide force| N:"
    )

    print(
        f"    median = "
        f"{percentile(force,50):.3f}"
    )

    print(
        f"    p95 = "
        f"{percentile(force,95):.3f}"
    )

    print(
        f"    max = "
        f"{max(force):.3f}"
    )


def first_breakaway(
    rows,
):

    previous = 0


    for r in rows:

        current = int(
            r[
                "breakaways"
            ]
        )


        if current > previous:

            return r


        previous = current


    return None


def closest_elapsed(
    rows,
    target,
):

    if not rows:

        return None


    t0 = rows[
        0
    ][
        "time"
    ]


    return min(
        rows,
        key=lambda r:
            abs(
                (
                    r[
                        "time"
                    ]
                    - t0
                )
                - target
            ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    import sys


    if len(
        sys.argv
    ) != 3:

        print(
            "Usage:\n"
            " python3 "
            "test_index_explicit_guide_stickslip_v15.py "
            "/path/to/free_bag "
            "/path/to/contact_bag"
        )

        raise SystemExit(
            1
        )


    free_bag = Path(
        sys.argv[
            1
        ]
    ).expanduser().resolve()


    contact_bag = Path(
        sys.argv[
            2
        ]
    ).expanduser().resolve()


    ab.patch_replay_modules()


    law = (
        ExplicitGuideLaw()
    )

    law.install()


    results = {}


    old_scale = (
        base.METERS_PER_TICK
    )

    old_k = (
        base.TENDON_STIFFNESS
    )


    try:

        base.METERS_PER_TICK = (
            SCALE
        )

        base.TENDON_STIFFNESS = (
            K_FINGER
        )


        for test in TESTS:

            label = (
                test[
                    "name"
                ].lower()
            )


            contact_xml = (
                MJCF_DIR
                /
                f"tmp_v15_{label}_contact.xml"
            )


            free_xml = (
                MJCF_DIR
                /
                f"tmp_v15_{label}_free.xml"
            )


            build_model(
                contact_xml
            )


            ab.make_free_model(
                contact_xml,
                free_xml,
            )


            print()
            print("=" * 110)
            print(
                f"V15 — "
                f"{test['name']}"
            )
            print("=" * 110)


            print(
                f"F_static  = "
                f"{test['F_static']:.3f} N"
            )

            print(
                f"F_kinetic = "
                f"{test['F_kinetic']:.3f} N"
            )


            law.configure(
                test,
                "FREE",
            )


            free_metrics, _ = (
                base.run_free(
                    free_xml,
                    free_bag,
                )
            )


            law.configure(
                test,
                "CONTACT",
            )


            contact_metrics, _ = (
                base.run_contact(
                    contact_xml,
                    contact_bag,
                )
            )


            results[
                test[
                    "name"
                ]
            ] = {

                "free":
                    free_metrics,

                "contact":
                    contact_metrics,
            }


            print(
                f"free combined RMSE = "
                f"{free_metrics['combined']:.3f} deg"
            )


            pose = np.asarray(
                contact_metrics[
                    "pose"
                ],
                dtype=float,
            )


            print(
                "contact pose = "
                f"("
                f"{pose[0]:.3f}, "
                f"{pose[1]:.3f}, "
                f"{pose[2]:.3f}"
                f") deg"
            )


            print(
                f"contact RMSE = "
                f"{contact_metrics['rmse']:.3f} deg"
            )


            try:
                contact_xml.unlink()
            except FileNotFoundError:
                pass


            try:
                free_xml.unlink()
            except FileNotFoundError:
                pass


    finally:

        law.restore()

        base.METERS_PER_TICK = (
            old_scale
        )

        base.TENDON_STIFFNESS = (
            old_k
        )


    # ========================================================
    # SAVE
    # ========================================================

    if law.rows:

        with open(
            OUTPUT_CSV,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    law.rows[
                        0
                    ].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                law.rows
            )


    # ========================================================
    # REPORT STATEFUL DIAGNOSTIC
    # ========================================================

    print()
    print("=" * 110)
    print(
        "V15 EXPLICIT GUIDE INTERNAL STATE"
    )
    print("=" * 110)


    for test in TESTS:

        name = test[
            "name"
        ]


        print()
        print("#" * 110)
        print(
            name
        )
        print("#" * 110)


        for phase in [
            "FREE",
            "CONTACT",
        ]:

            rows = [
                r
                for r in law.rows
                if (
                    r[
                        "test"
                    ]
                    == name
                    and
                    r[
                        "phase"
                    ]
                    == phase
                )
            ]


            report_internal(
                phase,
                rows,
            )


            event = first_breakaway(
                rows
            )


            if event is not None:

                t0 = rows[
                    0
                ][
                    "time"
                ]


                print()
                print(
                    "  FIRST BREAKAWAY:"
                )

                print(
                    f"    elapsed = "
                    f"{event['time']-t0:.3f} s"
                )

                print(
                    f"    slip = "
                    f"{event['slip_mm']:+.4f} mm"
                )

                print(
                    f"    mechanical drive = "
                    f"{event['mechanical_drive_N']:+.3f} N"
                )

                print(
                    f"    Tmotor-Tdistal = "
                    f"{event['Tm_minus_Td_N']:+.3f} N"
                )


            if (
                phase
                == "CONTACT"
            ):

                hold = closest_elapsed(
                    rows,
                    HOLD_ELAPSED,
                )


                if hold is not None:

                    print()
                    print(
                        "  VIDEO-ALIGNED HOLD:"
                    )

                    print(
                        f"    mode = "
                        f"{hold['mode']}"
                    )

                    print(
                        f"    slip = "
                        f"{hold['slip_mm']:+.4f} mm"
                    )

                    print(
                        f"    mechanical drive = "
                        f"{hold['mechanical_drive_N']:+.3f} N"
                    )

                    print(
                        f"    explicit guide force = "
                        f"{hold['applied_guide_force_N']:+.3f} N"
                    )

                    print(
                        f"    Tmotor = "
                        f"{hold['motor_tension_N']:.3f} N"
                    )

                    print(
                        f"    Tdistal = "
                        f"{hold['distal_tension_N']:.3f} N"
                    )

                    print(
                        f"    Tmotor-Tdistal = "
                        f"{hold['Tm_minus_Td_N']:+.3f} N"
                    )


    # ========================================================
    # DIRECT ARCHITECTURE EFFECT
    # ========================================================

    baseline = results[
        "BASELINE"
    ]

    stateful = results[
        "EXPLICIT_STICKSLIP"
    ]


    p0 = np.asarray(
        baseline[
            "contact"
        ][
            "pose"
        ],
        dtype=float,
    )


    p1 = np.asarray(
        stateful[
            "contact"
        ][
            "pose"
        ],
        dtype=float,
    )


    print()
    print("=" * 110)
    print(
        "V15 ARCHITECTURE DECISION"
    )
    print("=" * 110)


    print()
    print(
        "BASELINE:"
    )

    print(
        f"  free = "
        f"{baseline['free']['combined']:.3f} deg"
    )

    print(
        "  contact = "
        f"("
        f"{p0[0]:.3f}, "
        f"{p0[1]:.3f}, "
        f"{p0[2]:.3f}"
        f") deg"
    )


    print()
    print(
        "EXPLICIT STICK-SLIP:"
    )

    print(
        f"  free = "
        f"{stateful['free']['combined']:.3f} deg"
    )

    print(
        "  contact = "
        f"("
        f"{p1[0]:.3f}, "
        f"{p1[1]:.3f}, "
        f"{p1[2]:.3f}"
        f") deg"
    )


    print()
    print(
        "Change from baseline:"
    )

    print(
        f"  free RMSE delta = "
        f"{stateful['free']['combined'] - baseline['free']['combined']:+.3f} deg"
    )

    print(
        f"  dMCP = "
        f"{p1[0]-p0[0]:+.3f} deg"
    )

    print(
        f"  dPIP = "
        f"{p1[1]-p0[1]:+.3f} deg"
    )

    print(
        f"  dDIP = "
        f"{p1[2]-p0[2]:+.3f} deg"
    )


    ss_contact_rows = [
        r
        for r in law.rows
        if (
            r[
                "test"
            ]
            == "EXPLICIT_STICKSLIP"
            and
            r[
                "phase"
            ]
            == "CONTACT"
        )
    ]


    max_gap = max(
        abs(
            r[
                "Tm_minus_Td_N"
            ]
        )
        for r in ss_contact_rows
    )


    breakaways = (
        ss_contact_rows[
            -1
        ][
            "breakaways"
        ]
    )


    print()
    print(
        f"Stateful contact max |Tm-Td| = "
        f"{max_gap:.3f} N"
    )

    print(
        f"Stateful contact breakaways = "
        f"{breakaways}"
    )


    print()


    if (
        breakaways >= 1
        and
        max_gap >= 0.5
    ):

        print(
            "RESULT: EXPLICIT GUIDE LAW ENGAGED."
        )

        print()
        print(
            "The architecture can now sustain a real "
            "proximal/distal tension difference and undergo "
            "mechanical breakaway."
        )

        print()
        print(
            "NEXT: identify static/kinetic guide parameters "
            "jointly from free-space hysteresis and the "
            "contact redistribution experiment."
        )


    else:

        print(
            "RESULT: EXPLICIT GUIDE LAW DID NOT ENGAGE "
            "STRONGLY ENOUGH."
        )

        print()
        print(
            "Do not identify parameters yet. Inspect the "
            "generalized-force balance/topology first."
        )


    print()
    print(
        "Timeseries:"
    )

    print(
        OUTPUT_CSV
    )


    print()
    print("=" * 110)


if __name__ == "__main__":
    main()
