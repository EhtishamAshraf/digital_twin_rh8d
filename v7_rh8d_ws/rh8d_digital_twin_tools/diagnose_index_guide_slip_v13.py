#!/usr/bin/env python3

from pathlib import Path
import csv
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as base
import compare_index_configdep_replay_v6 as ab


# ============================================================
# V13
#
# Diagnostic only.
#
# Questions:
#
# 1. Does index_tendon_slip actually move?
# 2. Does frictionloss create T_motor != T_distal?
# 3. Does the tension difference reach the Coulomb limit?
# 4. Is the guide mostly stuck or continuously sliding?
#
# No parameter identification here.
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
      "guide_slip_diagnostic_v13"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SLIP = "index_tendon_slip"

SPOOL = "index_motor_spool"

MOTOR_TENDON = "idx_motor_segment"
DISTAL_TENDON = "idx_flex_spatial"

BOTTLE = "index_contact_fixture"
MID_GEOM = "collision_idx_mid"
TIP_GEOM = "collision_idx_tip"


SCALE = 5.10e-6
K_FINGER = 2192.0

FRICTIONS = [
    0.0,
    1.0,
    4.0,
]


# From the previously validated contact alignment.
HOLD_ELAPSED = 6.297


# ============================================================
# XML
# ============================================================

def build_candidate(
    friction,
    output,
):

    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()

    found = False


    for joint in root.iter(
        "joint"
    ):

        if (
            joint.attrib.get("name")
            == SLIP
        ):

            joint.set(
                "frictionloss",
                f"{friction:.10g}",
            )

            found = True
            break


    if not found:

        raise RuntimeError(
            f"Missing joint: {SLIP}"
        )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


    mujoco.MjModel.from_xml_path(
        str(
            output
        )
    )


# ============================================================
# MUJOCO HELPERS
# ============================================================

def obj_id(
    model,
    objtype,
    name,
):

    return mujoco.mj_name2id(
        model,
        objtype,
        name,
    )


def tendon_state(
    model,
    data,
    tid,
):

    L = float(
        data.ten_length[
            tid
        ]
    )


    if hasattr(
        data,
        "ten_velocity",
    ):

        V = float(
            data.ten_velocity[
                tid
            ]
        )

    else:

        V = 0.0


    k = float(
        model.tendon_stiffness[
            tid
        ]
    )

    c = float(
        model.tendon_damping[
            tid
        ]
    )


    ls = np.asarray(
        model.tendon_lengthspring[
            tid
        ],
        dtype=float,
    ).reshape(-1)


    if len(ls) >= 2:

        low = float(
            ls[0]
        )

        high = float(
            ls[1]
        )

    else:

        low = float(
            ls[0]
        )

        high = low


    if L > high:

        deformation = (
            L
            - high
        )

    elif L < low:

        deformation = (
            L
            - low
        )

    else:

        deformation = 0.0


    spring_scalar = (
        -k
        * deformation
    )

    damping_scalar = (
        -c
        * V
    )


    total_scalar = (
        spring_scalar
        + damping_scalar
    )


    tension = max(
        0.0,
        -total_scalar,
    )


    return {
        "length_m":
            L,

        "velocity_mps":
            V,

        "deformation_m":
            deformation,

        "spring_scalar_N":
            spring_scalar,

        "damping_scalar_N":
            damping_scalar,

        "tension_N":
            tension,
    }


# ============================================================
# RECORDER
# ============================================================

class Recorder:

    def __init__(self):

        self.original_step = (
            mujoco.mj_step
        )

        self.phase = None
        self.friction = None

        self.rows = []

        self.cache = {}


    def set_run(
        self,
        phase,
        friction,
    ):

        self.phase = phase
        self.friction = friction


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


        slip_jid = obj_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            SLIP,
        )


        spool_jid = obj_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            SPOOL,
        )


        motor_tid = obj_id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            MOTOR_TENDON,
        )


        distal_tid = obj_id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            DISTAL_TENDON,
        )


        # Ignore unrelated models stepped by helper scripts.
        if (
            slip_jid < 0
            or
            spool_jid < 0
            or
            motor_tid < 0
            or
            distal_tid < 0
        ):

            self.cache[
                key
            ] = None

            return None


        result = {

            "slip_jid":
                slip_jid,

            "slip_qadr":
                int(
                    model.jnt_qposadr[
                        slip_jid
                    ]
                ),

            "slip_dof":
                int(
                    model.jnt_dofadr[
                        slip_jid
                    ]
                ),

            "spool_qadr":
                int(
                    model.jnt_qposadr[
                        spool_jid
                    ]
                ),

            "motor_tid":
                motor_tid,

            "distal_tid":
                distal_tid,

            "bottle":
                obj_id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    BOTTLE,
                ),

            "mid":
                obj_id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    MID_GEOM,
                ),

            "tip":
                obj_id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    TIP_GEOM,
                ),
        }


        self.cache[
            key
        ] = result

        return result


    def contact_flags(
        self,
        data,
        ids,
    ):

        middle = False
        tip = False


        if ids["bottle"] < 0:

            return middle, tip


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


            if ids["bottle"] not in pair:

                continue


            if (
                ids["mid"] in pair
                and
                float(
                    c.dist
                ) <= 0.001
            ):

                middle = True


            if (
                ids["tip"] in pair
                and
                float(
                    c.dist
                ) <= 0.001
            ):

                tip = True


        return middle, tip


    def record(
        self,
        model,
        data,
    ):

        if self.phase is None:

            return


        ids = self.ids(
            model
        )


        if ids is None:

            return


        motor = tendon_state(
            model,
            data,
            ids[
                "motor_tid"
            ],
        )


        distal = tendon_state(
            model,
            data,
            ids[
                "distal_tid"
            ],
        )


        slip_q = float(
            data.qpos[
                ids[
                    "slip_qadr"
                ]
            ]
        )


        slip_v = float(
            data.qvel[
                ids[
                    "slip_dof"
                ]
            ]
        )


        spool_q = float(
            data.qpos[
                ids[
                    "spool_qadr"
                ]
            ]
        )


        compiled_friction = float(
            model.dof_frictionloss[
                ids[
                    "slip_dof"
                ]
            ]
        )


        middle, tip = (
            self.contact_flags(
                data,
                ids,
            )
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
                    self.friction,

                "compiled_friction_N":
                    compiled_friction,

                "time":
                    float(
                        data.time
                    ),

                "spool_mm":
                    spool_q
                    * 1000.0,

                "encoder_est":
                    100.0
                    +
                    spool_q
                    / SCALE,

                "slip_mm":
                    slip_q
                    * 1000.0,

                "slip_velocity_mps":
                    slip_v,

                "motor_tension_N":
                    motor[
                        "tension_N"
                    ],

                "distal_tension_N":
                    distal[
                        "tension_N"
                    ],

                "tension_difference_N":
                    tension_difference,

                "abs_tension_difference_N":
                    abs(
                        tension_difference
                    ),

                "motor_length_mm":
                    motor[
                        "length_m"
                    ]
                    * 1000.0,

                "distal_length_mm":
                    distal[
                        "length_m"
                    ]
                    * 1000.0,

                "middle_contact":
                    middle,

                "tip_contact":
                    tip,
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
# REPORT HELPERS
# ============================================================

def percentile(
    values,
    p,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(
        values
    ) == 0:

        return float(
            "nan"
        )


    return float(
        np.percentile(
            values,
            p,
        )
    )


def report_group(
    title,
    rows,
    friction,
):

    print()
    print(
        title
    )

    print(
        "-" * 96
    )


    if not rows:

        print(
            "  no samples"
        )

        return


    slip = np.asarray(
        [
            r[
                "slip_mm"
            ]
            for r in rows
        ],
        dtype=float,
    )


    vel = np.asarray(
        [
            r[
                "slip_velocity_mps"
            ]
            for r in rows
        ],
        dtype=float,
    )


    tm = np.asarray(
        [
            r[
                "motor_tension_N"
            ]
            for r in rows
        ],
        dtype=float,
    )


    td = np.asarray(
        [
            r[
                "distal_tension_N"
            ]
            for r in rows
        ],
        dtype=float,
    )


    gap = np.asarray(
        [
            r[
                "abs_tension_difference_N"
            ]
            for r in rows
        ],
        dtype=float,
    )


    stuck = (
        np.abs(
            vel
        )
        < 1.0e-5
    )


    print(
        f"  samples = "
        f"{len(rows)}"
    )


    print(
        f"  slip range = "
        f"{slip.min():+.4f} .. "
        f"{slip.max():+.4f} mm"
    )


    print(
        f"  slip final = "
        f"{slip[-1]:+.4f} mm"
    )


    print(
        f"  |slip velocity| p95 = "
        f"{percentile(abs(vel),95):.6f} m/s"
    )


    print(
        f"  stuck fraction "
        f"(|v| < 1e-5) = "
        f"{100.0*np.mean(stuck):.1f}%"
    )


    print(
        "  motor tension N: "
        f"median={percentile(tm,50):.3f}  "
        f"p95={percentile(tm,95):.3f}  "
        f"max={tm.max():.3f}"
    )


    print(
        "  distal tension N: "
        f"median={percentile(td,50):.3f}  "
        f"p95={percentile(td,95):.3f}  "
        f"max={td.max():.3f}"
    )


    print(
        "  |Tm-Td| N: "
        f"median={percentile(gap,50):.4f}  "
        f"p95={percentile(gap,95):.4f}  "
        f"max={gap.max():.4f}"
    )


    if friction > 0.0:

        utilization = (
            gap
            /
            friction
        )


        near_limit = (
            utilization
            >= 0.90
        )


        print(
            f"  |Tm-Td| / friction: "
            f"median={percentile(utilization,50):.3f}  "
            f"p95={percentile(utilization,95):.3f}"
        )


        print(
            f"  fraction >= 90% friction limit = "
            f"{100.0*np.mean(near_limit):.1f}%"
        )


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
            "diagnose_index_guide_slip_v13.py "
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


    metrics = {}


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
                f"f{friction:.1f}"
                .replace(
                    ".",
                    "p",
                )
            )


            contact_xml = (
                MJCF_DIR
                /
                f"tmp_v13_contact_{label}.xml"
            )


            free_xml = (
                MJCF_DIR
                /
                f"tmp_v13_free_{label}.xml"
            )


            build_candidate(
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
                f"V13 — FRICTION "
                f"{friction:.2f} N"
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


            metrics[
                friction
            ] = (
                free_metrics,
                contact_metrics,
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


    # ========================================================
    # SAVE
    # ========================================================

    output = (
        RESULT_DIR
        / "guide_slip_timeseries.csv"
    )


    if recorder.rows:

        with open(
            output,
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


    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 110)
    print(
        "V13 INTERNAL GUIDE-STATE DIAGNOSTIC"
    )
    print("=" * 110)


    for friction in FRICTIONS:

        free = [
            r
            for r in recorder.rows
            if (
                r[
                    "phase"
                ]
                == "FREE"
                and
                abs(
                    r[
                        "requested_friction_N"
                    ]
                    - friction
                )
                < 1.0e-9
            )
        ]


        contact = [
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
                    - friction
                )
                < 1.0e-9
            )
        ]


        print()
        print("#" * 110)
        print(
            f"GUIDE FRICTION = "
            f"{friction:.2f} N"
        )
        print("#" * 110)


        report_group(
            "FREE SPACE",
            free,
            friction,
        )


        report_group(
            "CONTACT",
            contact,
            friction,
        )


        first_middle = None
        first_tip = None


        for r in contact:

            if (
                first_middle is None
                and
                r[
                    "middle_contact"
                ]
            ):

                first_middle = r


            if (
                first_tip is None
                and
                r[
                    "tip_contact"
                ]
            ):

                first_tip = r


        if contact:

            t0 = contact[
                0
            ][
                "time"
            ]


            print()
            print(
                "Contact events:"
            )


            if first_middle:

                print(
                    "  first middle contact:"
                )

                print(
                    f"    elapsed = "
                    f"{first_middle['time']-t0:.3f} s"
                )

                print(
                    f"    encoder = "
                    f"{first_middle['encoder_est']:.1f}"
                )

                print(
                    f"    slip = "
                    f"{first_middle['slip_mm']:+.4f} mm"
                )

                print(
                    f"    Tmotor/Tdistal = "
                    f"{first_middle['motor_tension_N']:.3f} / "
                    f"{first_middle['distal_tension_N']:.3f} N"
                )

                print(
                    f"    Tmotor-Tdistal = "
                    f"{first_middle['tension_difference_N']:+.3f} N"
                )

            else:

                print(
                    "  no middle contact"
                )


            if first_tip:

                print(
                    "  first tip contact:"
                )

                print(
                    f"    elapsed = "
                    f"{first_tip['time']-t0:.3f} s"
                )

                print(
                    f"    encoder = "
                    f"{first_tip['encoder_est']:.1f}"
                )

                print(
                    f"    slip = "
                    f"{first_tip['slip_mm']:+.4f} mm"
                )

            else:

                print(
                    "  no tip contact"
                )


            hold = closest_elapsed(
                contact,
                HOLD_ELAPSED,
            )


            if hold:

                print()
                print(
                    f"Video-aligned diagnostic "
                    f"at ~{HOLD_ELAPSED:.3f} s:"
                )

                print(
                    f"  encoder = "
                    f"{hold['encoder_est']:.1f}"
                )

                print(
                    f"  slip = "
                    f"{hold['slip_mm']:+.4f} mm"
                )

                print(
                    f"  slip velocity = "
                    f"{hold['slip_velocity_mps']:+.6f} m/s"
                )

                print(
                    f"  motor tension = "
                    f"{hold['motor_tension_N']:.3f} N"
                )

                print(
                    f"  distal tension = "
                    f"{hold['distal_tension_N']:.3f} N"
                )

                print(
                    f"  Tmotor-Tdistal = "
                    f"{hold['tension_difference_N']:+.3f} N"
                )


    # ========================================================
    # DIRECT 0 vs 4 N COMPARISON
    # ========================================================

    print()
    print("=" * 110)
    print(
        "V13 DECISION"
    )
    print("=" * 110)


    zero_contact = [
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
            )
            < 1.0e-9
        )
    ]


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
            < 1.0e-9
        )
    ]


    if (
        zero_contact
        and
        four_contact
    ):

        z_slip = np.asarray(
            [
                r[
                    "slip_mm"
                ]
                for r in zero_contact
            ]
        )


        f_slip = np.asarray(
            [
                r[
                    "slip_mm"
                ]
                for r in four_contact
            ]
        )


        z_gap = np.asarray(
            [
                r[
                    "abs_tension_difference_N"
                ]
                for r in zero_contact
            ]
        )


        f_gap = np.asarray(
            [
                r[
                    "abs_tension_difference_N"
                ]
                for r in four_contact
            ]
        )


        print()
        print(
            "0 N contact:"
        )

        print(
            f"  slip range = "
            f"{z_slip.min():+.4f} .. "
            f"{z_slip.max():+.4f} mm"
        )

        print(
            f"  |Tm-Td| p95 = "
            f"{percentile(z_gap,95):.4f} N"
        )


        print()
        print(
            "4 N contact:"
        )

        print(
            f"  slip range = "
            f"{f_slip.min():+.4f} .. "
            f"{f_slip.max():+.4f} mm"
        )

        print(
            f"  |Tm-Td| p95 = "
            f"{percentile(f_gap,95):.4f} N"
        )


        print()
        print(
            "Interpretation rule:"
        )

        print(
            "  If 4 N creates a large tension difference "
            "but the joint trajectory remains unchanged, "
            "simple Coulomb guide friction is mechanically "
            "insufficient."
        )

        print(
            "  If the tension difference remains near zero, "
            "the current latent-slip construction is not "
            "actually engaging guide friction as intended."
        )

        print(
            "  If the slip coordinate is pinned at a joint "
            "limit, fix that numerical topology before any "
            "further friction identification."
        )


    print()
    print(
        "Timeseries:"
    )

    print(
        output
    )


    print()
    print("=" * 110)


if __name__ == "__main__":
    main()
