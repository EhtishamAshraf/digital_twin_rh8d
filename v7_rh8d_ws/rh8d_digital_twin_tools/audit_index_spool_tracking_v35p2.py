#!/usr/bin/env python3

from pathlib import Path
import csv
import math

import mujoco
import numpy as np

import fit_index_passive_contact_v35b2 as b2
import scan_index_zero_margin_damping_v35e2b as e2b


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

XML = (
    MJCF
    / "ur5e_rh8d_index_cad_spatial_v35i1_best.xml"
)

BAG = Path(
    "/home/ehtisham/rh8d_system_id/"
    "index_contact_20260911_164310"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v35p2_spool_tracking_audit"
)

OUTCSV = (
    OUTDIR
    / "v35p2_spool_trace.csv"
)


TENDON_K = 9000.0

CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0
CONTACT_MU = 0.05

METERS_PER_TICK = 5.10e-6


SPOOL_JOINT = "index_motor_spool"
ACTUATOR = "pos_index_tendon"

FIXTURE = "index_contact_fixture"
MID = "collision_idx_mid_mesh"
TIP = "collision_idx_tip"


def joint_id(model, name):

    j = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if j < 0:
        raise RuntimeError(
            f"Missing joint: {name}"
        )

    return j


def actuator_id(model, name):

    a = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )

    if a < 0:
        raise RuntimeError(
            f"Missing actuator: {name}"
        )

    return a


def geom_id(model, name):

    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        name,
    )


class Audit:

    def __init__(self):

        self.rows = []
        self.events = []

        self.prev_mid = False
        self.prev_tip = False

        self.prev_time = None
        self.prev_target = None

        self.reverse_samples = 0
        self.reverse_duration = 0.0

        self.max_error = None


    def active_contacts(
        self,
        model,
        data,
    ):

        fixture = geom_id(
            model,
            FIXTURE,
        )

        mid_gid = geom_id(
            model,
            MID,
        )

        tip_gid = geom_id(
            model,
            TIP,
        )


        mid = False
        tip = False


        for ci in range(
            data.ncon
        ):

            c = data.contact[
                ci
            ]

            pair = {
                int(c.geom1),
                int(c.geom2),
            }


            if fixture not in pair:
                continue


            if mid_gid in pair:
                mid = True


            if tip_gid in pair:
                tip = True


        return mid, tip


    def inspect(
        self,
        model,
        data,
    ):

        t = float(
            data.time
        )

        ji = joint_id(
            model,
            SPOOL_JOINT,
        )

        qa = int(
            model.jnt_qposadr[
                ji
            ]
        )

        da = int(
            model.jnt_dofadr[
                ji
            ]
        )

        ai = actuator_id(
            model,
            ACTUATOR,
        )


        actual = float(
            data.qpos[
                qa
            ]
        )

        actual_vel = float(
            data.qvel[
                da
            ]
        )

        target = float(
            data.ctrl[
                ai
            ]
        )

        actuator_force = float(
            data.actuator_force[
                ai
            ]
        )


        error = (
            target
            -
            actual
        )


        # ----------------------------------------------------
        # Target velocity from actual replayed control signal.
        # ----------------------------------------------------

        if (
            self.prev_time is not None
            and
            t > self.prev_time
        ):

            target_vel = (
                target
                -
                self.prev_target
            ) / (
                t
                -
                self.prev_time
            )

        else:

            target_vel = 0.0


        mid, tip = self.active_contacts(
            model,
            data,
        )


        row = {

            "time_s":
                t,

            "target_spool_mm":
                target
                * 1000.0,

            "actual_spool_mm":
                actual
                * 1000.0,

            "tracking_error_mm":
                error
                * 1000.0,

            "tracking_error_ticks":
                error
                /
                METERS_PER_TICK,

            "target_vel_mm_s":
                target_vel
                * 1000.0,

            "actual_vel_mm_s":
                actual_vel
                * 1000.0,

            "actuator_force_N":
                actuator_force,

            "qfrc_actuator_N":
                float(
                    data.qfrc_actuator[
                        da
                    ]
                ),

            "qfrc_passive_N":
                float(
                    data.qfrc_passive[
                        da
                    ]
                ),

            "qfrc_constraint_N":
                float(
                    data.qfrc_constraint[
                        da
                    ]
                ),

            "middle_contact":
                int(
                    mid
                ),

            "tip_contact":
                int(
                    tip
                ),
        }


        self.rows.append(
            row
        )


        # ----------------------------------------------------
        # Detect actual spool backing off while target is
        # stationary or still increasing.
        # ----------------------------------------------------

        dt = float(
            model.opt.timestep
        )


        if (
            target_vel >= -1.0e-5
            and
            actual_vel < -2.0e-5
        ):

            self.reverse_samples += 1
            self.reverse_duration += dt


        # ----------------------------------------------------
        # Record worst target-versus-actual error.
        # ----------------------------------------------------

        if (
            self.max_error is None
            or
            abs(
                error
            )
            >
            abs(
                self.max_error[
                    "tracking_error_mm"
                ]
                / 1000.0
            )
        ):

            self.max_error = dict(
                row
            )


        # ----------------------------------------------------
        # Contact event transitions
        # ----------------------------------------------------

        event_parts = []


        if (
            mid
            and
            not self.prev_mid
        ):

            event_parts.append(
                "ON:middle"
            )


        if (
            self.prev_mid
            and
            not mid
        ):

            event_parts.append(
                "OFF:middle"
            )


        if (
            tip
            and
            not self.prev_tip
        ):

            event_parts.append(
                "ON:tip"
            )


        if (
            self.prev_tip
            and
            not tip
        ):

            event_parts.append(
                "OFF:tip"
            )


        if event_parts:

            ev = dict(
                row
            )

            ev[
                "event"
            ] = ",".join(
                event_parts
            )

            self.events.append(
                ev
            )


        self.prev_mid = mid
        self.prev_tip = tip

        self.prev_time = t
        self.prev_target = target


def rms(
    values,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                x * x
            )
        )
    )


def percentile(
    values,
    p,
):

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=float,
            ),
            p,
        )
    )


def print_row(
    r,
):

    print(
        f"    target spool : "
        f"{r['target_spool_mm']:+.4f} mm"
    )

    print(
        f"    actual spool : "
        f"{r['actual_spool_mm']:+.4f} mm"
    )

    print(
        f"    error        : "
        f"{r['tracking_error_mm']:+.4f} mm "
        f"({r['tracking_error_ticks']:+.1f} ticks)"
    )

    print(
        f"    target vel   : "
        f"{r['target_vel_mm_s']:+.4f} mm/s"
    )

    print(
        f"    actual vel   : "
        f"{r['actual_vel_mm_s']:+.4f} mm/s"
    )

    print(
        f"    actuator F   : "
        f"{r['actuator_force_N']:+.4f} N"
    )

    print(
        "    qfrc actuator/passive/constraint: "
        f"{r['qfrc_actuator_N']:+.4f} / "
        f"{r['qfrc_passive_N']:+.4f} / "
        f"{r['qfrc_constraint_N']:+.4f} N"
    )


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print("=" * 150)
    print("V35P2 — MEASURED-ENCODER TARGET VS ACTUAL MUJOCO SPOOL")
    print("=" * 150)

    print(
        f"XML:\n{XML}"
    )

    print()
    print(
        "This test changes NO model parameter."
    )

    print(
        "The actuator ctrl is the spool target generated "
        "from the measured hardware encoder replay."
    )


    audit = Audit()

    real_step = mujoco.mj_step


    def wrapped_step(
        model,
        data,
        *args,
        **kwargs,
    ):

        out = real_step(
            model,
            data,
            *args,
            **kwargs,
        )

        audit.inspect(
            model,
            data,
        )

        return out


    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
    )


    old_tk = float(
        b2.v35.TENDON_K
    )

    old_btk = float(
        b2.v30.v24.base.TENDON_STIFFNESS
    )

    old_ck = float(
        b2.v35.CONTACT_K
    )

    old_mu = float(
        b2.v35.CONTACT_MU
    )

    old_cd = float(
        contact_replay.CONTACT_DAMPING
    )


    b2.v35.TENDON_K = (
        TENDON_K
    )

    b2.v30.v24.base.TENDON_STIFFNESS = (
        TENDON_K
    )

    b2.v35.CONTACT_K = (
        CONTACT_K
    )

    b2.v35.CONTACT_MU = (
        CONTACT_MU
    )

    contact_replay.CONTACT_DAMPING = (
        CONTACT_DAMPING
    )


    mujoco.mj_step = wrapped_step


    try:

        result = e2b.run_one(
            CONTACT_DAMPING,
            XML,
            BAG,
            contact_replay,
        )

    finally:

        mujoco.mj_step = real_step

        b2.v35.TENDON_K = (
            old_tk
        )

        b2.v30.v24.base.TENDON_STIFFNESS = (
            old_btk
        )

        b2.v35.CONTACT_K = (
            old_ck
        )

        b2.v35.CONTACT_MU = (
            old_mu
        )

        contact_replay.CONTACT_DAMPING = (
            old_cd
        )


    if not audit.rows:

        raise RuntimeError(
            "No spool samples captured."
        )


    with OUTCSV.open(
        "w",
        newline="",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                audit.rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            audit.rows
        )


    errors_mm = [
        r[
            "tracking_error_mm"
        ]
        for r in audit.rows
    ]

    errors_ticks = [
        r[
            "tracking_error_ticks"
        ]
        for r in audit.rows
    ]


    contact_rows = [
        r
        for r in audit.rows
        if
        (
            r[
                "middle_contact"
            ]
            or
            r[
                "tip_contact"
            ]
        )
    ]


    print()
    print("=" * 150)
    print("FINAL CONTACT RESULT")
    print("=" * 150)

    print(
        "pose = "
        f"{result['pose'][0]:.3f} / "
        f"{result['pose'][1]:.3f} / "
        f"{result['pose'][2]:.3f} deg"
    )

    print(
        f"RMSE = "
        f"{result['rmse']:.3f} deg"
    )


    print()
    print("=" * 150)
    print("SPOOL TRACKING — WHOLE RUN")
    print("=" * 150)

    print(
        f"RMS error = "
        f"{rms(errors_mm):.4f} mm "
        f"({rms(errors_ticks):.1f} ticks)"
    )

    print(
        f"95% |error| = "
        f"{percentile(np.abs(errors_mm), 95):.4f} mm "
        f"({percentile(np.abs(errors_ticks), 95):.1f} ticks)"
    )

    print(
        f"max |error| = "
        f"{max(abs(x) for x in errors_mm):.4f} mm "
        f"({max(abs(x) for x in errors_ticks):.1f} ticks)"
    )

    print(
        "actual spool reversal while target is "
        "stationary/increasing:"
    )

    print(
        f"  samples  = "
        f"{audit.reverse_samples}"
    )

    print(
        f"  duration = "
        f"{audit.reverse_duration:.4f} s"
    )


    if contact_rows:

        ce = [
            r[
                "tracking_error_mm"
            ]
            for r in contact_rows
        ]

        cet = [
            r[
                "tracking_error_ticks"
            ]
            for r in contact_rows
        ]


        print()
        print("=" * 150)
        print("SPOOL TRACKING — WHILE FINGER CONTACT IS ACTIVE")
        print("=" * 150)

        print(
            f"RMS error = "
            f"{rms(ce):.4f} mm "
            f"({rms(cet):.1f} ticks)"
        )

        print(
            f"95% |error| = "
            f"{percentile(np.abs(ce), 95):.4f} mm "
            f"({percentile(np.abs(cet), 95):.1f} ticks)"
        )

        print(
            f"max |error| = "
            f"{max(abs(x) for x in ce):.4f} mm "
            f"({max(abs(x) for x in cet):.1f} ticks)"
        )


    print()
    print("=" * 150)
    print("CONTACT EVENT SEQUENCE")
    print("=" * 150)


    for i, r in enumerate(
        audit.events,
        1,
    ):

        print()
        print(
            f"[{i}] "
            f"t={r['time_s']:.4f} s   "
            f"{r['event']}"
        )

        print_row(
            r
        )


    print()
    print("=" * 150)
    print("WORST SPOOL TRACKING ERROR")
    print("=" * 150)

    print(
        f"t = "
        f"{audit.max_error['time_s']:.4f} s"
    )

    print_row(
        audit.max_error
    )


    print()
    print(
        f"CSV:\n{OUTCSV}"
    )

    print(
        f"\nExperiment folder:\n{OUTDIR}"
    )

    print("=" * 150)


if __name__ == "__main__":
    main()
