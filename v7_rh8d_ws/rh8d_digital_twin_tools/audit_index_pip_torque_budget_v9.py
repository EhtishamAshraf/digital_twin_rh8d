#!/usr/bin/env python3

from pathlib import Path
import csv
import math

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as base
import compare_index_configdep_replay_v6 as ab


# ============================================================
# PURPOSE
#
# Diagnose WHY the real finger redistributes strongly into PIP
# after contact while the ideal tendon model does not.
#
# NO parameters are fitted here.
# NO hardware joint angles are imposed.
#
# We measure in the trusted CONSTANT model:
#
#   tendon tension
#   available PIP tendon torque
#   passive PIP generalized torque
#   middle/tip bottle contacts
#
# during:
#
#   1. validated free-space replay
#   2. measured-encoder contact replay
#
# Key question:
#
# Does contact create a clearly higher tendon-load regime that
# could physically overcome a PIP breakaway/stiction mechanism?
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
      "pip_torque_budget_v9"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CONTACT_MODEL = (
    MJCF_DIR
    / "tmp_v9_constant_contact.xml"
)

FREE_MODEL = (
    MJCF_DIR
    / "tmp_v9_constant_free.xml"
)


# ============================================================
# Trusted constant-model parameters
# ============================================================

METERS_PER_TICK = 5.10e-6
TENDON_STIFFNESS = 2100.0

OPEN_ENCODER = 100.0

# Constant PIP transmission coefficient.
PIP_ARM = 0.004


# Hardware-video aligned target:
#
# bag 38.970486
# approach start 32.673486
#
# therefore approximately:
CONTACT_TARGET_ELAPSED = (
    38.970486
    - 32.673486
)


# ============================================================
# Names
# ============================================================

TENDON = "idx_flex_spatial"

SPOOL = "index_motor_spool"

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

BOTTLE = "index_contact_fixture"

IDX_PROX = "collision_idx_prox"
IDX_MID = "collision_idx_mid"
IDX_TIP = "collision_idx_tip"


# ============================================================
# Utility
# ============================================================

def object_id(
    model,
    objtype,
    name,
):

    return mujoco.mj_name2id(
        model,
        objtype,
        name,
    )


def joint_qpos_adr(
    model,
    name,
):

    jid = object_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        return None

    return int(
        model.jnt_qposadr[
            jid
        ]
    )


def joint_dof_adr(
    model,
    name,
):

    jid = object_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        return None

    return int(
        model.jnt_dofadr[
            jid
        ]
    )


# ============================================================
# PASSIVE TENDON FORCE RECONSTRUCTION
#
# This MuJoCo Python version does not expose data.ten_force.
#
# For a passive tendon:
#
#   F_spring = -k * deformation
#   F_damp   = -c * tendon_velocity
#
# springlength=[low, high] defines the zero-force interval.
#
# Positive tendon extension beyond the upper spring length
# produces a negative generalized tendon force. We report its
# positive magnitude as physical tensile load.
# ============================================================

def tendon_tension_components(
    model,
    data,
    tid,
):

    length = float(
        data.ten_length[
            tid
        ]
    )


    ten_velocity_array = getattr(
        data,
        "ten_velocity",
        None,
    )


    if ten_velocity_array is None:

        velocity = 0.0

    else:

        velocity = float(
            ten_velocity_array[
                tid
            ]
        )


    stiffness = float(
        model.tendon_stiffness[
            tid
        ]
    )


    damping = float(
        model.tendon_damping[
            tid
        ]
    )


    length_spring = getattr(
        model,
        "tendon_lengthspring",
        None,
    )


    if length_spring is None:

        # Very old/fallback API.
        rest = float(
            model.tendon_length0[
                tid
            ]
        )

        lower = rest
        upper = rest

    else:

        values = np.asarray(
            length_spring[
                tid
            ],
            dtype=float,
        ).reshape(-1)


        if len(values) >= 2:

            lower = float(
                values[0]
            )

            upper = float(
                values[1]
            )

        else:

            lower = float(
                values[0]
            )

            upper = lower


    # Deformation relative to the deadband.
    if length > upper:

        deformation = (
            length
            - upper
        )

    elif length < lower:

        deformation = (
            length
            - lower
        )

    else:

        deformation = 0.0


    spring_scalar = (
        -stiffness
        * deformation
    )


    damping_scalar = (
        -damping
        * velocity
    )


    total_scalar = (
        spring_scalar
        + damping_scalar
    )


    # Negative scalar force corresponds to tensile restoring
    # load for the closing direction used in this model.
    tension = max(
        0.0,
        -total_scalar,
    )


    return {
        "length":
            length,

        "velocity":
            velocity,

        "deformation":
            deformation,

        "spring_scalar":
            spring_scalar,

        "damping_scalar":
            damping_scalar,

        "tension":
            tension,
    }


def pip_resistance_components(
    model,
    data,
    jid,
    qadr,
    dof,
):

    q = float(
        data.qpos[
            qadr
        ]
    )

    qvel = float(
        data.qvel[
            dof
        ]
    )


    stiffness = float(
        model.jnt_stiffness[
            jid
        ]
    )


    damping = float(
        model.dof_damping[
            dof
        ]
    )


    frictionloss = float(
        model.dof_frictionloss[
            dof
        ]
    )


    qpos_spring = getattr(
        model,
        "qpos_spring",
        None,
    )


    if qpos_spring is None:

        qref = float(
            model.qpos0[
                qadr
            ]
        )

    else:

        qref = float(
            qpos_spring[
                qadr
            ]
        )


    spring_tau = (
        -stiffness
        * (
            q
            - qref
        )
    )


    damping_tau = (
        -damping
        * qvel
    )


    # frictionloss is a Coulomb/static-friction bound rather
    # than a continuously applied passive torque. For the
    # breakaway question, include it as an opposing threshold.
    resistance_budget = (
        abs(
            spring_tau
            + damping_tau
        )
        +
        frictionloss
    )


    return {
        "spring_tau":
            spring_tau,

        "damping_tau":
            damping_tau,

        "frictionloss":
            frictionloss,

        "budget":
            resistance_budget,
    }


# ============================================================
# Recorder
# ============================================================

class StepRecorder:

    def __init__(self):

        self.phase = None
        self.rows = []

        self._original_step = (
            mujoco.mj_step
        )

        self._cache = {}


    def set_phase(
        self,
        phase,
    ):

        self.phase = phase


    def _ids(
        self,
        model,
    ):

        key = id(
            model
        )


        if key in self._cache:

            return self._cache[
                key
            ]


        result = {

            "tid":
                object_id(
                    model,
                    mujoco.mjtObj.mjOBJ_TENDON,
                    TENDON,
                ),

            "mcp_q":
                joint_qpos_adr(
                    model,
                    MCP,
                ),

            "pip_q":
                joint_qpos_adr(
                    model,
                    PIP,
                ),

            "pip_jid":
                object_id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    PIP,
                ),

            "dip_q":
                joint_qpos_adr(
                    model,
                    DIP,
                ),

            "spool_q":
                joint_qpos_adr(
                    model,
                    SPOOL,
                ),

            "spool_dof":
                joint_dof_adr(
                    model,
                    SPOOL,
                ),

            "pip_dof":
                joint_dof_adr(
                    model,
                    PIP,
                ),

            "bottle":
                object_id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    BOTTLE,
                ),

            "prox":
                object_id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    IDX_PROX,
                ),

            "mid":
                object_id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    IDX_MID,
                ),

            "tip":
                object_id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    IDX_TIP,
                ),
        }


        self._cache[
            key
        ] = result


        return result


    def _contact_status(
        self,
        data,
        ids,
    ):

        bottle = ids[
            "bottle"
        ]


        status = {

            "prox_contact":
                False,

            "mid_contact":
                False,

            "tip_contact":
                False,

            "prox_dist":
                None,

            "mid_dist":
                None,

            "tip_dist":
                None,
        }


        if bottle < 0:

            return status


        mapping = {
            ids["prox"]:
                (
                    "prox_contact",
                    "prox_dist",
                ),

            ids["mid"]:
                (
                    "mid_contact",
                    "mid_dist",
                ),

            ids["tip"]:
                (
                    "tip_contact",
                    "tip_dist",
                ),
        }


        for i in range(
            data.ncon
        ):

            c = data.contact[
                i
            ]


            if (
                c.geom1
                != bottle
                and
                c.geom2
                != bottle
            ):

                continue


            other = (
                c.geom2
                if c.geom1
                == bottle
                else c.geom1
            )


            if other not in mapping:

                continue


            contact_key, dist_key = (
                mapping[
                    other
                ]
            )


            dist_mm = (
                float(
                    c.dist
                )
                * 1000.0
            )


            previous = status[
                dist_key
            ]


            if (
                previous is None
                or
                dist_mm < previous
            ):

                status[
                    dist_key
                ] = dist_mm


            # MuJoCo may instantiate a contact slightly before
            # negative geometric penetration because of margin.
            #
            # <= +1 mm is treated as meaningful near-contact.
            if dist_mm <= 1.0:

                status[
                    contact_key
                ] = True


        return status


    def _record(
        self,
        model,
        data,
    ):

        if self.phase is None:

            return


        ids = self._ids(
            model
        )


        if (
            ids["tid"] < 0
            or
            ids["pip_q"] is None
            or
            ids["spool_q"] is None
        ):

            return


        tendon_state = tendon_tension_components(
            model,
            data,
            ids["tid"],
        )


        tension = tendon_state[
            "tension"
        ]


        pip_resistance = pip_resistance_components(
            model,
            data,
            ids["pip_jid"],
            ids["pip_q"],
            ids["pip_dof"],
        )


        # For the trusted fixed model:
        #
        # tau_PIP = T * 0.004
        #
        pip_tendon_torque = (
            tension
            * PIP_ARM
        )


        spool = float(
            data.qpos[
                ids["spool_q"]
            ]
        )


        spool_velocity = float(
            data.qvel[
                ids["spool_dof"]
            ]
        )


        encoder_est = (
            OPEN_ENCODER
            +
            spool
            / METERS_PER_TICK
        )


        contacts = (
            self._contact_status(
                data,
                ids,
            )
        )


        self.rows.append(
            {

                "phase":
                    self.phase,

                "time":
                    float(
                        data.time
                    ),

                "encoder_est":
                    encoder_est,

                "spool_mm":
                    spool
                    * 1000.0,

                "spool_velocity":
                    spool_velocity,

                "mcp_deg":
                    math.degrees(
                        float(
                            data.qpos[
                                ids["mcp_q"]
                            ]
                        )
                    ),

                "pip_deg":
                    math.degrees(
                        float(
                            data.qpos[
                                ids["pip_q"]
                            ]
                        )
                    ),

                "dip_deg":
                    math.degrees(
                        float(
                            data.qpos[
                                ids["dip_q"]
                            ]
                        )
                    ),

                "tendon_length_mm":
                    tendon_state[
                        "length"
                    ]
                    * 1000.0,

                "tendon_velocity_mps":
                    tendon_state[
                        "velocity"
                    ],

                "tendon_deformation_mm":
                    tendon_state[
                        "deformation"
                    ]
                    * 1000.0,

                "tendon_spring_scalar_N":
                    tendon_state[
                        "spring_scalar"
                    ],

                "tendon_damping_scalar_N":
                    tendon_state[
                        "damping_scalar"
                    ],

                "tension_N":
                    tension,

                "pip_tendon_torque_Nm":
                    pip_tendon_torque,

                "pip_spring_torque_Nm":
                    pip_resistance[
                        "spring_tau"
                    ],

                "pip_damping_torque_Nm":
                    pip_resistance[
                        "damping_tau"
                    ],

                "pip_frictionloss_Nm":
                    pip_resistance[
                        "frictionloss"
                    ],

                "pip_resistance_budget_Nm":
                    pip_resistance[
                        "budget"
                    ],

                **contacts,
            }
        )


    def install(
        self,
    ):

        original = (
            self._original_step
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


            recorder._record(
                model,
                data,
            )


            return result


        mujoco.mj_step = wrapped


    def restore(
        self,
    ):

        mujoco.mj_step = (
            self._original_step
        )


# ============================================================
# Statistics
# ============================================================

def percentile(
    values,
    p,
):

    if not values:

        return None

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=float,
            ),
            p,
        )
    )


def summarize_subset(
    name,
    rows,
):

    print()
    print(
        name
    )

    print(
        "-" * 85
    )


    if not rows:

        print(
            "  no samples"
        )

        return


    tension = [
        r["tension_N"]
        for r in rows
    ]


    drive = [
        r[
            "pip_tendon_torque_Nm"
        ]
        for r in rows
    ]


    passive = [
        r[
            "pip_resistance_budget_Nm"
        ]
        for r in rows
    ]


    print(
        f"  samples = "
        f"{len(rows)}"
    )


    print(
        "  tension N: "
        f"median={percentile(tension,50):.3f}  "
        f"p95={percentile(tension,95):.3f}  "
        f"max={max(tension):.3f}"
    )


    print(
        "  available PIP tendon torque Nm: "
        f"median={percentile(drive,50):.5f}  "
        f"p95={percentile(drive,95):.5f}  "
        f"max={max(drive):.5f}"
    )


    print(
        "  PIP resistance budget Nm: "
        f"median={percentile(passive,50):.5f}  "
        f"p95={percentile(passive,95):.5f}  "
        f"max={max(passive):.5f}"
    )


    ratios = []


    for d, p in zip(
        drive,
        passive,
    ):

        if p > 1.0e-6:

            ratios.append(
                d / p
            )


    if ratios:

        print(
            "  tendon/passive torque ratio: "
            f"median={percentile(ratios,50):.3f}  "
            f"p95={percentile(ratios,95):.3f}"
        )


def closest_time(
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
                    r["time"]
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
            " python3 audit_index_pip_torque_budget_v9.py "
            "/path/to/free_bag "
            "/path/to/contact_bag"
        )

        raise SystemExit(
            1
        )


    free_bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    contact_bag = Path(
        sys.argv[2]
    ).expanduser().resolve()


    # --------------------------------------------------------
    # Build the trusted constant model on the frozen V5
    # fixture/contact-mask geometry.
    # --------------------------------------------------------

    ab.replace_index_with_constant_tendon(
        ab.V5_SOURCE,
        CONTACT_MODEL,
    )


    ab.make_free_model(
        CONTACT_MODEL,
        FREE_MODEL,
    )


    # Force old replay helpers onto the frozen V5 fixture.
    ab.patch_replay_modules()


    old_scale = (
        base.METERS_PER_TICK
    )

    old_k = (
        base.TENDON_STIFFNESS
    )


    recorder = (
        StepRecorder()
    )


    recorder.install()


    try:

        base.METERS_PER_TICK = (
            METERS_PER_TICK
        )

        base.TENDON_STIFFNESS = (
            TENDON_STIFFNESS
        )


        print()
        print("=" * 110)
        print(
            "RH8D INDEX — PIP TORQUE-BUDGET "
            "AUDIT V9"
        )
        print("=" * 110)


        print()
        print(
            "Running trusted FREE-SPACE replay..."
        )


        recorder.set_phase(
            "FREE"
        )


        free_metrics, _ = (
            base.run_free(
                FREE_MODEL,
                free_bag,
            )
        )


        print(
            f"  free combined RMSE = "
            f"{free_metrics['combined']:.3f} deg"
        )


        print()
        print(
            "Running measured-encoder CONTACT replay..."
        )


        recorder.set_phase(
            "CONTACT"
        )


        contact_metrics, _ = (
            base.run_contact(
                CONTACT_MODEL,
                contact_bag,
            )
        )


        print(
            f"  contact RMSE = "
            f"{contact_metrics['rmse']:.3f} deg"
        )


    finally:

        recorder.restore()

        base.METERS_PER_TICK = (
            old_scale
        )

        base.TENDON_STIFFNESS = (
            old_k
        )


    # --------------------------------------------------------
    # Split data.
    # --------------------------------------------------------

    free_rows = [
        r
        for r in recorder.rows
        if r[
            "phase"
        ]
        == "FREE"
    ]


    contact_rows = [
        r
        for r in recorder.rows
        if r[
            "phase"
        ]
        == "CONTACT"
    ]


    # Normalize each replay's time.
    for collection in [
        free_rows,
        contact_rows,
    ]:

        if collection:

            t0 = collection[
                0
            ][
                "time"
            ]


            for r in collection:

                r[
                    "elapsed"
                ] = (
                    r[
                        "time"
                    ]
                    - t0
                )


    # --------------------------------------------------------
    # Save raw diagnostic CSV.
    # --------------------------------------------------------

    output_csv = (
        RESULT_DIR
        / "torque_budget.csv"
    )


    all_rows = (
        free_rows
        + contact_rows
    )


    if all_rows:

        fields = list(
            all_rows[
                0
            ].keys()
        )


        with open(
            output_csv,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()

            writer.writerows(
                all_rows
            )


    # --------------------------------------------------------
    # Define comparable regions.
    # --------------------------------------------------------

    # Free-space CLOSING around the encoder range where the
    # contact experiment operates.
    free_same_encoder = [
        r
        for r in free_rows
        if (
            700.0
            <= r[
                "encoder_est"
            ]
            <= 1150.0
            and
            r[
                "spool_velocity"
            ]
            > 1.0e-5
        )
    ]


    contact_pre_middle = []


    first_middle_index = None
    first_tip_index = None


    for i, r in enumerate(
        contact_rows
    ):

        if (
            first_middle_index
            is None
            and
            r[
                "mid_contact"
            ]
        ):

            first_middle_index = i


        if (
            first_tip_index
            is None
            and
            r[
                "tip_contact"
            ]
        ):

            first_tip_index = i


    if first_middle_index is None:

        contact_pre_middle = (
            contact_rows
        )

    else:

        contact_pre_middle = (
            contact_rows[
                :first_middle_index
            ]
        )


    middle_before_tip = []


    if (
        first_middle_index
        is not None
    ):

        end = (
            first_tip_index
            if (
                first_tip_index
                is not None
            )
            else len(
                contact_rows
            )
        )


        middle_before_tip = [
            r
            for r in contact_rows[
                first_middle_index:
                end
            ]
            if r[
                "mid_contact"
            ]
        ]


    after_tip = []


    if (
        first_tip_index
        is not None
    ):

        after_tip = (
            contact_rows[
                first_tip_index:
            ]
        )


    # --------------------------------------------------------
    # Report.
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "TORQUE / TENSION REGIMES"
    )
    print("=" * 110)


    summarize_subset(
        "FREE SPACE: closing, encoder 700-1150",
        free_same_encoder,
    )


    summarize_subset(
        "CONTACT: before first meaningful middle contact",
        contact_pre_middle,
    )


    summarize_subset(
        "CONTACT: middle contact before first tip contact",
        middle_before_tip,
    )


    summarize_subset(
        "CONTACT: from first tip contact onward",
        after_tip,
    )


    print()
    print("=" * 110)
    print(
        "CONTACT EVENT TIMING"
    )
    print("=" * 110)


    if (
        first_middle_index
        is None
    ):

        print()
        print(
            "No meaningful middle contact detected."
        )

    else:

        r = contact_rows[
            first_middle_index
        ]

        print()
        print(
            "First middle contact:"
        )

        print(
            f"  elapsed = "
            f"{r['elapsed']:.3f} s"
        )

        print(
            f"  encoder ~= "
            f"{r['encoder_est']:.1f} ticks"
        )

        print(
            f"  tension = "
            f"{r['tension_N']:.3f} N"
        )

        print(
            f"  PIP = "
            f"{r['pip_deg']:.3f} deg"
        )


    if (
        first_tip_index
        is None
    ):

        print()
        print(
            "No meaningful tip contact detected."
        )

    else:

        r = contact_rows[
            first_tip_index
        ]

        print()
        print(
            "First tip contact:"
        )

        print(
            f"  elapsed = "
            f"{r['elapsed']:.3f} s"
        )

        print(
            f"  encoder ~= "
            f"{r['encoder_est']:.1f} ticks"
        )

        print(
            f"  tension = "
            f"{r['tension_N']:.3f} N"
        )

        print(
            f"  PIP = "
            f"{r['pip_deg']:.3f} deg"
        )


    # --------------------------------------------------------
    # Video-aligned hold.
    # --------------------------------------------------------

    hold = closest_time(
        contact_rows,
        CONTACT_TARGET_ELAPSED,
    )


    print()
    print("=" * 110)
    print(
        "VIDEO-ALIGNED HOLD DIAGNOSTIC"
    )
    print("=" * 110)


    if hold is not None:

        print()
        print(
            f"elapsed = "
            f"{hold['elapsed']:.3f} s"
        )

        print(
            f"encoder ~= "
            f"{hold['encoder_est']:.1f} ticks"
        )

        print(
            "SIM pose = "
            f"("
            f"{hold['mcp_deg']:.3f}, "
            f"{hold['pip_deg']:.3f}, "
            f"{hold['dip_deg']:.3f}"
            f") deg"
        )

        print(
            f"tendon tension = "
            f"{hold['tension_N']:.3f} N"
        )

        print(
            f"available PIP tendon torque = "
            f"{hold['pip_tendon_torque_Nm']:.5f} Nm"
        )

        print(
            f"PIP resistance budget = "
            f"{hold['pip_resistance_budget_Nm']:.5f} Nm"
        )

        print(
            f"  spring contribution = "
            f"{hold['pip_spring_torque_Nm']:+.5f} Nm"
        )

        print(
            f"  damping contribution = "
            f"{hold['pip_damping_torque_Nm']:+.5f} Nm"
        )

        print(
            f"  friction-loss bound = "
            f"{hold['pip_frictionloss_Nm']:.5f} Nm"
        )

        print(
            f"middle contact = "
            f"{hold['mid_contact']}"
        )

        print(
            f"tip contact = "
            f"{hold['tip_contact']}"
        )


        # Compare against free-space at approximately the
        # same encoder.
        comparable = [
            r
            for r in free_rows
            if (
                abs(
                    r[
                        "encoder_est"
                    ]
                    -
                    hold[
                        "encoder_est"
                    ]
                )
                <= 50.0
                and
                r[
                    "spool_velocity"
                ]
                > 1.0e-5
            )
        ]


        if comparable:

            free_tension = [
                r[
                    "tension_N"
                ]
                for r in comparable
            ]


            median_free = percentile(
                free_tension,
                50,
            )


            print()
            print(
                "Free-space at approximately "
                "the same encoder:"
            )

            print(
                f"  median tension = "
                f"{median_free:.3f} N"
            )

            print(
                f"  contact/free tension ratio = "
                f"{hold['tension_N']/max(median_free,1e-9):.3f} x"
            )


    print()
    print("=" * 110)
    print(
        "DECISION GUIDE"
    )
    print("=" * 110)

    print()
    print(
        "If contact tension / available PIP torque rises "
        "well above the comparable free-space regime, "
        "a tension-driven breakaway/stiction model is "
        "physically testable next."
    )

    print()
    print(
        "If free-space and contact tension regimes overlap "
        "strongly, a simple tension threshold cannot explain "
        "the hardware redistribution. Then the next model "
        "should represent tendon-guide history/friction or "
        "another stateful transmission mechanism instead."
    )


    print()
    print(
        "Raw diagnostic CSV:"
    )

    print(
        output_csv
    )


    print()
    print("=" * 110)


    for path in [
        CONTACT_MODEL,
        FREE_MODEL,
    ]:

        try:
            path.unlink()

        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
