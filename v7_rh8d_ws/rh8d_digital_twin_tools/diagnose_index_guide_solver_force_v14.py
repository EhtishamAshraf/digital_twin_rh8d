#!/usr/bin/env python3

from pathlib import Path
import csv

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as base
import compare_index_configdep_replay_v6 as ab
import diagnose_index_guide_slip_v13 as v13


# ============================================================
# V14
#
# Diagnose MuJoCo's ACTUAL generalized constraint force on
# index_tendon_slip.
#
# This distinguishes:
#
#   A) frictionloss is active but our reconstructed tendon
#      tensions do not expose it correctly
#
#   B) frictionloss is effectively inactive in this topology
#
# NO fitting.
# NO structural changes.
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
      "guide_solver_force_v14"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT = (
    RESULT_DIR
    / "solver_force_timeseries.csv"
)


FRICTIONS = [
    0.0,
    4.0,
]


SLIP = v13.SLIP

SCALE = v13.SCALE
K_FINGER = v13.K_FINGER

HOLD_ELAPSED = 6.297


# ============================================================
# RECORDER
# ============================================================

class Recorder:

    def __init__(self):

        self.original_step = (
            mujoco.mj_step
        )

        self.rows = []

        self.phase = None
        self.requested_friction = None

        self.cache = {}


    def set_run(
        self,
        phase,
        friction,
    ):

        self.phase = phase
        self.requested_friction = friction


    def get_ids(
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


        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            SLIP,
        )


        if jid < 0:

            self.cache[
                key
            ] = None

            return None


        qadr = int(
            model.jnt_qposadr[
                jid
            ]
        )


        dof = int(
            model.jnt_dofadr[
                jid
            ]
        )


        motor_tid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            v13.MOTOR_TENDON,
        )


        distal_tid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            v13.DISTAL_TENDON,
        )


        if (
            motor_tid < 0
            or
            distal_tid < 0
        ):

            self.cache[
                key
            ] = None

            return None


        result = {
            "jid":
                jid,

            "qadr":
                qadr,

            "dof":
                dof,

            "motor_tid":
                motor_tid,

            "distal_tid":
                distal_tid,
        }


        self.cache[
            key
        ] = result

        return result


    def friction_efc_force(
        self,
        data,
        dof,
    ):

        # ----------------------------------------------------
        # Dry joint friction is represented by MuJoCo through
        # an EFC constraint row.
        #
        # We search explicitly for FRICTION_DOF rows whose
        # efc_id corresponds to this slip DOF.
        # ----------------------------------------------------

        total = 0.0
        count = 0
        row_types = []


        friction_enum = getattr(
            mujoco.mjtConstraint,
            "mjCNSTR_FRICTION_DOF",
            None,
        )


        for i in range(
            int(
                data.nefc
            )
        ):

            try:

                efc_id = int(
                    data.efc_id[
                        i
                    ]
                )

                efc_type = int(
                    data.efc_type[
                        i
                    ]
                )

            except Exception:

                continue


            if efc_id != dof:

                continue


            row_types.append(
                efc_type
            )


            if (
                friction_enum is None
                or
                efc_type
                == int(
                    friction_enum
                )
            ):

                total += float(
                    data.efc_force[
                        i
                    ]
                )

                count += 1


        return (
            total,
            count,
            row_types,
        )


    def record(
        self,
        model,
        data,
    ):

        if self.phase is None:

            return


        ids = self.get_ids(
            model
        )


        if ids is None:

            return


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


        dof = ids[
            "dof"
        ]


        qadr = ids[
            "qadr"
        ]


        efc_force, efc_count, row_types = (
            self.friction_efc_force(
                data,
                dof,
            )
        )


        compiled_friction = float(
            model.dof_frictionloss[
                dof
            ]
        )


        slip_q = float(
            data.qpos[
                qadr
            ]
        )


        slip_v = float(
            data.qvel[
                dof
            ]
        )


        qacc = float(
            data.qacc[
                dof
            ]
        )


        qfrc_passive = float(
            data.qfrc_passive[
                dof
            ]
        )


        qfrc_constraint = float(
            data.qfrc_constraint[
                dof
            ]
        )


        qfrc_applied = float(
            data.qfrc_applied[
                dof
            ]
        )


        qfrc_bias = float(
            data.qfrc_bias[
                dof
            ]
        )


        tension_difference = (
            motor[
                "tension_N"
            ]
            -
            distal[
                "tension_N"
            ]
        )


        self.rows.append(
            {

                "phase":
                    self.phase,

                "requested_friction_N":
                    self.requested_friction,

                "compiled_friction_N":
                    compiled_friction,

                "time":
                    float(
                        data.time
                    ),

                "slip_mm":
                    slip_q
                    * 1000.0,

                "slip_velocity_mps":
                    slip_v,

                "slip_acceleration_mps2":
                    qacc,

                "motor_tension_N":
                    motor[
                        "tension_N"
                    ],

                "distal_tension_N":
                    distal[
                        "tension_N"
                    ],

                "Tm_minus_Td_N":
                    tension_difference,

                "qfrc_passive_N":
                    qfrc_passive,

                "qfrc_constraint_N":
                    qfrc_constraint,

                "qfrc_applied_N":
                    qfrc_applied,

                "qfrc_bias_N":
                    qfrc_bias,

                "friction_efc_force":
                    efc_force,

                "friction_efc_rows":
                    efc_count,

                "matching_efc_types":
                    ",".join(
                        str(x)
                        for x in row_types
                    ),
            }
        )


    def install(
        self,
    ):

        original = (
            self.original_step
        )

        recorder = self


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


            recorder.record(
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
# REPORT
# ============================================================

def percentile(
    values,
    p,
):

    a = np.asarray(
        values,
        dtype=float,
    )


    if len(
        a
    ) == 0:

        return float(
            "nan"
        )


    return float(
        np.percentile(
            a,
            p,
        )
    )


def report(
    title,
    rows,
):

    print()
    print(
        title
    )

    print(
        "-" * 100
    )


    if not rows:

        print(
            "  no samples"
        )

        return


    def values(
        key,
    ):

        return np.asarray(
            [
                r[
                    key
                ]
                for r in rows
            ],
            dtype=float,
        )


    v = values(
        "slip_velocity_mps"
    )

    tdiff = values(
        "Tm_minus_Td_N"
    )

    passive = values(
        "qfrc_passive_N"
    )

    constraint = values(
        "qfrc_constraint_N"
    )

    efc = values(
        "friction_efc_force"
    )

    compiled = values(
        "compiled_friction_N"
    )


    print(
        f"  samples = "
        f"{len(rows)}"
    )


    print(
        f"  compiled friction = "
        f"{np.median(compiled):.3f} N"
    )


    print(
        f"  moving fraction "
        f"(|v| > 1e-5) = "
        f"{100*np.mean(np.abs(v)>1e-5):.1f}%"
    )


    print(
        "  Tmotor-Tdistal N:"
    )

    print(
        f"    median = "
        f"{percentile(tdiff,50):+.4f}"
    )

    print(
        f"    p05/p95 = "
        f"{percentile(tdiff,5):+.4f} / "
        f"{percentile(tdiff,95):+.4f}"
    )


    print(
        "  qfrc_passive at slip DOF N:"
    )

    print(
        f"    median = "
        f"{percentile(passive,50):+.4f}"
    )

    print(
        f"    p05/p95 = "
        f"{percentile(passive,5):+.4f} / "
        f"{percentile(passive,95):+.4f}"
    )


    print(
        "  qfrc_constraint at slip DOF N:"
    )

    print(
        f"    median = "
        f"{percentile(constraint,50):+.4f}"
    )

    print(
        f"    p05/p95 = "
        f"{percentile(constraint,5):+.4f} / "
        f"{percentile(constraint,95):+.4f}"
    )

    print(
        f"    max |force| = "
        f"{np.max(np.abs(constraint)):.4f}"
    )


    print(
        "  matched friction EFC force:"
    )

    print(
        f"    median = "
        f"{percentile(efc,50):+.4f}"
    )

    print(
        f"    p05/p95 = "
        f"{percentile(efc,5):+.4f} / "
        f"{percentile(efc,95):+.4f}"
    )

    print(
        f"    max |force| = "
        f"{np.max(np.abs(efc)):.4f}"
    )


    row_counts = sorted(
        set(
            int(
                r[
                    "friction_efc_rows"
                ]
            )
            for r in rows
        )
    )


    type_sets = sorted(
        set(
            r[
                "matching_efc_types"
            ]
            for r in rows
        )
    )


    print(
        f"  matched friction EFC row counts = "
        f"{row_counts}"
    )

    print(
        f"  matching EFC types = "
        f"{type_sets}"
    )


def closest_hold(
    rows,
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
                - HOLD_ELAPSED
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
            "diagnose_index_guide_solver_force_v14.py "
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


    recorder = Recorder()

    recorder.install()


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


        for friction in FRICTIONS:

            label = (
                f"{friction:.0f}N"
            )


            contact_xml = (
                MJCF_DIR
                /
                f"tmp_v14_contact_{label}.xml"
            )


            free_xml = (
                MJCF_DIR
                /
                f"tmp_v14_free_{label}.xml"
            )


            v13.build_candidate(
                friction,
                contact_xml,
            )


            ab.make_free_model(
                contact_xml,
                free_xml,
            )


            print()
            print("=" * 110)
            print(
                f"V14 RUN — GUIDE FRICTION "
                f"{friction:.1f} N"
            )
            print("=" * 110)


            recorder.set_run(
                "FREE",
                friction,
            )


            free_metrics, _ = (
                base.run_free(
                    free_xml,
                    free_bag,
                )
            )


            recorder.set_run(
                "CONTACT",
                friction,
            )


            contact_metrics, _ = (
                base.run_contact(
                    contact_xml,
                    contact_bag,
                )
            )


            print(
                f"free RMSE = "
                f"{free_metrics['combined']:.3f} deg"
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

        recorder.restore()

        base.METERS_PER_TICK = (
            old_scale
        )

        base.TENDON_STIFFNESS = (
            old_k
        )


    # --------------------------------------------------------
    # Save timeseries
    # --------------------------------------------------------

    if recorder.rows:

        with open(
            OUTPUT,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    recorder.rows[
                        0
                    ].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                recorder.rows
            )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "V14 ACTUAL SOLVER-FORCE DIAGNOSTIC"
    )
    print("=" * 110)


    for friction in FRICTIONS:

        print()
        print("#" * 110)
        print(
            f"GUIDE FRICTION = "
            f"{friction:.1f} N"
        )
        print("#" * 110)


        for phase in [
            "FREE",
            "CONTACT",
        ]:

            rows = [
                r
                for r in recorder.rows
                if (
                    r[
                        "phase"
                    ]
                    == phase
                    and
                    abs(
                        r[
                            "requested_friction_N"
                        ]
                        - friction
                    )
                    < 1e-9
                )
            ]


            report(
                phase,
                rows,
            )


            if (
                phase
                == "CONTACT"
            ):

                hold = closest_hold(
                    rows
                )


                if hold:

                    print()
                    print(
                        "  VIDEO-ALIGNED HOLD:"
                    )

                    print(
                        f"    slip = "
                        f"{hold['slip_mm']:+.4f} mm"
                    )

                    print(
                        f"    velocity = "
                        f"{hold['slip_velocity_mps']:+.6f} m/s"
                    )

                    print(
                        f"    Tmotor-Tdistal = "
                        f"{hold['Tm_minus_Td_N']:+.4f} N"
                    )

                    print(
                        f"    qfrc_passive = "
                        f"{hold['qfrc_passive_N']:+.4f} N"
                    )

                    print(
                        f"    qfrc_constraint = "
                        f"{hold['qfrc_constraint_N']:+.4f} N"
                    )

                    print(
                        f"    friction EFC force = "
                        f"{hold['friction_efc_force']:+.4f}"
                    )


    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    four_contact = [
        r
        for r in recorder.rows
        if (
            r[
                "phase"
            ]
            == "CONTACT"
            and
            abs(
                r[
                    "requested_friction_N"
                ]
                - 4.0
            )
            < 1e-9
        )
    ]


    print()
    print("=" * 110)
    print(
        "V14 DECISION"
    )
    print("=" * 110)


    if four_contact:

        constraint = np.asarray(
            [
                abs(
                    r[
                        "qfrc_constraint_N"
                    ]
                )
                for r in four_contact
            ]
        )


        efc = np.asarray(
            [
                abs(
                    r[
                        "friction_efc_force"
                    ]
                )
                for r in four_contact
            ]
        )


        print()
        print(
            f"4 N contact "
            f"|qfrc_constraint| p95 = "
            f"{percentile(constraint,95):.4f} N"
        )


        print(
            f"4 N contact "
            f"|friction EFC force| p95 = "
            f"{percentile(efc,95):.4f}"
        )


        if (
            percentile(
                constraint,
                95,
            )
            > 2.0
        ):

            print()
            print(
                "Friction IS active in the MuJoCo "
                "constraint solver."
            )

            print(
                "Our spring/damping tension reconstruction "
                "is therefore not representing the true "
                "force balance at the internal guide."
            )

            print()
            print(
                "NEXT: derive segment forces directly from "
                "generalized force balance before changing "
                "the mechanical model."
            )


        else:

            print()
            print(
                "The requested 4 N friction is NOT producing "
                "a substantial generalized constraint force."
            )

            print()
            print(
                "NEXT: stop using joint frictionloss for "
                "this latent guide and replace it with an "
                "explicit stateful tendon-guide force law."
            )


    print()
    print(
        "Timeseries:"
    )

    print(
        OUTPUT
    )


    print()
    print("=" * 110)


if __name__ == "__main__":
    main()
