#!/usr/bin/env python3

from pathlib import Path
import csv
import math

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as base
import compare_index_configdep_replay_v6 as ab
import diagnose_index_guide_slip_v13 as tendon_tools
import scan_and_view_index_contact_friction_v18 as viewer_tools


# ============================================================
# RH8D INDEX — PIP CONTACT TORQUE AUDIT V21
#
# Diagnostic only.
#
# Determine WHY PIP fails to flex after middle-phalanx
# contact.
#
# Frozen:
#
#   bottle position
#   contact geometry
#   mu = 0.05
#   normal stiffness = 1000 N/m
#   constant tendon coefficients:
#
#       MCP  -0.005 m/rad
#       PIP  -0.004 m/rad
#       DIP  -0.001 m/rad
#
#   motor dynamics
#   passive joint parameters
#
# No fitting.
# No parameter scan.
# ============================================================


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()

MJCF_DIR = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

MODEL_PATH = (
    MJCF_DIR
    / "tmp_v21_torque_audit.xml"
)

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "pip_contact_torque_v21"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_OUT = (
    RESULT_DIR
    / "pip_contact_torque_timeseries.csv"
)


# ============================================================
# FROZEN TRUSTED VALUES
# ============================================================

SCALE = 5.10e-6
TENDON_K = 2100.0

CONTACT_MU = 0.05
CONTACT_K = 1000.0

OPEN_ENCODER = 100.0


# ============================================================
# NAMES
# ============================================================

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

SPOOL = "index_motor_spool"

TENDON = "idx_flex_spatial"

BOTTLE = "index_contact_fixture"

PROX = "collision_idx_prox"
MID = "collision_idx_mid"
TIP = "collision_idx_tip"


# Constant tendon dL/dq magnitudes.
MOMENT_ARMS = {
    MCP: 0.005,
    PIP: 0.004,
    DIP: 0.001,
}


# ============================================================
# HELPERS
# ============================================================

def obj_id(
    model,
    objtype,
    name,
):

    idx = mujoco.mj_name2id(
        model,
        objtype,
        name,
    )

    if idx < 0:

        raise RuntimeError(
            f"Missing MuJoCo object: {name}"
        )

    return idx


def joint_info(
    model,
    name,
):

    jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return {
        "jid":
            jid,

        "qadr":
            int(
                model.jnt_qposadr[
                    jid
                ]
            ),

        "dof":
            int(
                model.jnt_dofadr[
                    jid
                ]
            ),
    }


def joint_passive_components(
    model,
    data,
    info,
):

    jid = info[
        "jid"
    ]

    qadr = info[
        "qadr"
    ]

    dof = info[
        "dof"
    ]


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


    friction_bound = float(
        model.dof_frictionloss[
            dof
        ]
    )


    try:

        rest = float(
            model.qpos_spring[
                qadr
            ]
        )

    except Exception:

        rest = float(
            model.qpos0[
                qadr
            ]
        )


    spring_torque = (
        -stiffness
        * (
            q
            -
            rest
        )
    )


    damping_torque = (
        -damping
        * qvel
    )


    return {
        "q":
            q,

        "qvel":
            qvel,

        "spring":
            spring_torque,

        "damping":
            damping_torque,

        "friction_bound":
            friction_bound,
    }


# ============================================================
# CONTACT GENERALIZED FORCE
#
# mj_contactForce gives the contact wrench in the contact
# frame.
#
# c.frame stores contact-frame axes as rows, therefore:
#
#     world_vector = local_vector @ frame
#
# The normal points from geom1 -> geom2.
#
# Therefore the returned contact force is applied in that
# direction to geom2 and with opposite sign to geom1.
# ============================================================

def contact_generalized_force(
    model,
    data,
    bottle_gid,
    index_geoms,
):

    total_qfrc = np.zeros(
        model.nv,
        dtype=float,
    )


    total_normal_force = 0.0

    contacts = 0


    for i in range(
        data.ncon
    ):

        c = data.contact[
            i
        ]


        g1 = int(
            c.geom1
        )

        g2 = int(
            c.geom2
        )


        if (
            bottle_gid
            not in (
                g1,
                g2,
            )
        ):

            continue


        finger_gid = None

        sign = None


        if (
            g1
            in index_geoms
            and
            g2
            == bottle_gid
        ):

            finger_gid = g1

            # mj_contactForce convention corresponds to the
            # force on geom2. Finger is geom1 -> opposite.
            sign = -1.0


        elif (
            g2
            in index_geoms
            and
            g1
            == bottle_gid
        ):

            finger_gid = g2

            # Finger is geom2.
            sign = +1.0


        else:

            continue


        force6 = np.zeros(
            6,
            dtype=float,
        )


        mujoco.mj_contactForce(
            model,
            data,
            i,
            force6,
        )


        frame = np.asarray(
            c.frame,
            dtype=float,
        ).reshape(
            3,
            3,
        )


        world_force = (
            sign
            *
            (
                force6[
                    :3
                ]
                @ frame
            )
        )


        world_torque = (
            sign
            *
            (
                force6[
                    3:
                ]
                @ frame
            )
        )


        body_id = int(
            model.geom_bodyid[
                finger_gid
            ]
        )


        qfrc_one = np.zeros(
            model.nv,
            dtype=float,
        )


        mujoco.mj_applyFT(
            model,
            data,
            world_force,
            world_torque,
            np.asarray(
                c.pos,
                dtype=float,
            ),
            body_id,
            qfrc_one,
        )


        total_qfrc += (
            qfrc_one
        )


        total_normal_force += abs(
            float(
                force6[
                    0
                ]
            )
        )


        contacts += 1


    return (
        total_qfrc,
        total_normal_force,
        contacts,
    )


# ============================================================
# RECORDER
# ============================================================

class Recorder:

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


        try:

            info = {

                "mcp":
                    joint_info(
                        model,
                        MCP,
                    ),

                "pip":
                    joint_info(
                        model,
                        PIP,
                    ),

                "dip":
                    joint_info(
                        model,
                        DIP,
                    ),

                "spool":
                    joint_info(
                        model,
                        SPOOL,
                    ),

                "tendon":
                    obj_id(
                        model,
                        mujoco.mjtObj.mjOBJ_TENDON,
                        TENDON,
                    ),

                "bottle":
                    obj_id(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        BOTTLE,
                    ),

                "prox":
                    obj_id(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        PROX,
                    ),

                "mid":
                    obj_id(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        MID,
                    ),

                "tip":
                    obj_id(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        TIP,
                    ),
            }


        except RuntimeError:

            info = None


        self.cache[
            key
        ] = info

        return info


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
        # Tendon tension
        # ----------------------------------------------------

        tendon = tendon_tools.tendon_state(
            model,
            data,
            ids[
                "tendon"
            ],
        )


        tension = float(
            tendon[
                "tension_N"
            ]
        )


        # ----------------------------------------------------
        # Joint passive terms
        # ----------------------------------------------------

        mcp_passive = (
            joint_passive_components(
                model,
                data,
                ids[
                    "mcp"
                ],
            )
        )


        pip_passive = (
            joint_passive_components(
                model,
                data,
                ids[
                    "pip"
                ],
            )
        )


        dip_passive = (
            joint_passive_components(
                model,
                data,
                ids[
                    "dip"
                ],
            )
        )


        # ----------------------------------------------------
        # Contact-generated generalized torques
        # ----------------------------------------------------

        index_geoms = {
            ids[
                "prox"
            ],
            ids[
                "mid"
            ],
            ids[
                "tip"
            ],
        }


        (
            qfrc_contact,
            normal_force,
            contact_count,
        ) = contact_generalized_force(
            model,
            data,
            ids[
                "bottle"
            ],
            index_geoms,
        )


        # ----------------------------------------------------
        # Signed tendon flexion torques.
        #
        # L contains negative coefficients for flexion:
        #
        #     tau = -T dL/dq
        #
        # therefore positive flexion torque is:
        #
        #     T * |coefficient|
        # ----------------------------------------------------

        tendon_tau_mcp = (
            tension
            *
            MOMENT_ARMS[
                MCP
            ]
        )


        tendon_tau_pip = (
            tension
            *
            MOMENT_ARMS[
                PIP
            ]
        )


        tendon_tau_dip = (
            tension
            *
            MOMENT_ARMS[
                DIP
            ]
        )


        mcp_dof = ids[
            "mcp"
        ][
            "dof"
        ]

        pip_dof = ids[
            "pip"
        ][
            "dof"
        ]

        dip_dof = ids[
            "dip"
        ][
            "dof"
        ]


        # qfrc_constraint includes contact + dry friction +
        # any other active solver constraints.
        #
        # Remove our independently reconstructed bottle/index
        # contact torque to expose the residual solver term.
        residual_constraint_pip = (
            float(
                data.qfrc_constraint[
                    pip_dof
                ]
            )
            -
            float(
                qfrc_contact[
                    pip_dof
                ]
            )
        )


        spool_q = float(
            data.qpos[
                ids[
                    "spool"
                ][
                    "qadr"
                ]
            ]
        )


        self.rows.append(
            {

                "time":
                    float(
                        data.time
                    ),

                "encoder_est":
                    OPEN_ENCODER
                    +
                    spool_q
                    / SCALE,

                "mcp_deg":
                    math.degrees(
                        mcp_passive[
                            "q"
                        ]
                    ),

                "pip_deg":
                    math.degrees(
                        pip_passive[
                            "q"
                        ]
                    ),

                "dip_deg":
                    math.degrees(
                        dip_passive[
                            "q"
                        ]
                    ),

                "tendon_tension_N":
                    tension,

                "tendon_tau_mcp_Nm":
                    tendon_tau_mcp,

                "tendon_tau_pip_Nm":
                    tendon_tau_pip,

                "tendon_tau_dip_Nm":
                    tendon_tau_dip,

                "pip_spring_tau_Nm":
                    pip_passive[
                        "spring"
                    ],

                "pip_damping_tau_Nm":
                    pip_passive[
                        "damping"
                    ],

                "pip_friction_bound_Nm":
                    pip_passive[
                        "friction_bound"
                    ],

                "contact_tau_mcp_Nm":
                    float(
                        qfrc_contact[
                            mcp_dof
                        ]
                    ),

                "contact_tau_pip_Nm":
                    float(
                        qfrc_contact[
                            pip_dof
                        ]
                    ),

                "contact_tau_dip_Nm":
                    float(
                        qfrc_contact[
                            dip_dof
                        ]
                    ),

                "qfrc_passive_pip_Nm":
                    float(
                        data.qfrc_passive[
                            pip_dof
                        ]
                    ),

                "qfrc_constraint_pip_Nm":
                    float(
                        data.qfrc_constraint[
                            pip_dof
                        ]
                    ),

                "residual_constraint_pip_Nm":
                    residual_constraint_pip,

                "contact_normal_force_N":
                    normal_force,

                "index_bottle_contacts":
                    contact_count,
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


def print_state(
    title,
    row,
    t0,
):

    print()
    print(
        title
    )

    print(
        "-" * 100
    )


    if row is None:

        print(
            "  not detected"
        )

        return


    print(
        f"  elapsed = "
        f"{row['time']-t0:.3f} s"
    )

    print(
        f"  encoder ~= "
        f"{row['encoder_est']:.1f} ticks"
    )


    print(
        "  pose MCP/PIP/DIP = "
        f"{row['mcp_deg']:.3f} / "
        f"{row['pip_deg']:.3f} / "
        f"{row['dip_deg']:.3f} deg"
    )


    print(
        f"  tendon tension = "
        f"{row['tendon_tension_N']:.3f} N"
    )


    print()
    print(
        "  FLEXION TORQUE AVAILABLE FROM TENDON:"
    )

    print(
        f"    MCP = "
        f"{row['tendon_tau_mcp_Nm']:+.5f} Nm"
    )

    print(
        f"    PIP = "
        f"{row['tendon_tau_pip_Nm']:+.5f} Nm"
    )

    print(
        f"    DIP = "
        f"{row['tendon_tau_dip_Nm']:+.5f} Nm"
    )


    print()
    print(
        "  PIP PASSIVE TERMS:"
    )

    print(
        f"    spring  = "
        f"{row['pip_spring_tau_Nm']:+.5f} Nm"
    )

    print(
        f"    damping = "
        f"{row['pip_damping_tau_Nm']:+.5f} Nm"
    )

    print(
        f"    dry-friction bound = "
        f"{row['pip_friction_bound_Nm']:.5f} Nm"
    )


    print()
    print(
        "  BOTTLE CONTACT GENERALIZED TORQUES:"
    )

    print(
        f"    MCP = "
        f"{row['contact_tau_mcp_Nm']:+.5f} Nm"
    )

    print(
        f"    PIP = "
        f"{row['contact_tau_pip_Nm']:+.5f} Nm"
    )

    print(
        f"    DIP = "
        f"{row['contact_tau_dip_Nm']:+.5f} Nm"
    )


    print()
    print(
        "  MUJOCO PIP GENERALIZED FORCE:"
    )

    print(
        f"    qfrc_passive = "
        f"{row['qfrc_passive_pip_Nm']:+.5f} Nm"
    )

    print(
        f"    qfrc_constraint = "
        f"{row['qfrc_constraint_pip_Nm']:+.5f} Nm"
    )

    print(
        f"    residual constraint "
        f"(constraint - bottle contact) = "
        f"{row['residual_constraint_pip_Nm']:+.5f} Nm"
    )


    print()
    print(
        f"  total bottle/index normal force = "
        f"{row['contact_normal_force_N']:.3f} N"
    )

    print(
        f"  active bottle/index contacts = "
        f"{row['index_bottle_contacts']}"
    )


def first_where(
    rows,
    predicate,
):

    for r in rows:

        if predicate(
            r
        ):

            return r


    return None


# ============================================================
# MAIN
# ============================================================

def main():

    import sys


    if len(
        sys.argv
    ) != 2:

        raise SystemExit(
            "Usage:\n"
            " python3 audit_index_pip_contact_torque_v21.py "
            "/path/to/contact_bag"
        )


    contact_bag = Path(
        sys.argv[
            1
        ]
    ).expanduser().resolve()


    if not contact_bag.exists():

        raise RuntimeError(
            f"Contact bag missing:\n"
            f"{contact_bag}"
        )


    # --------------------------------------------------------
    # Build exact frozen trusted constant-tendon control.
    # --------------------------------------------------------

    ab.replace_index_with_constant_tendon(
        ab.V5_SOURCE,
        MODEL_PATH,
    )


    ab.patch_replay_modules()


    # --------------------------------------------------------
    # Recorder
    # --------------------------------------------------------

    recorder = Recorder()

    recorder.install()


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


        print()
        print("=" * 112)
        print(
            "V21 — TRUSTED CONTACT PIP "
            "TORQUE-BALANCE AUDIT"
        )
        print("=" * 112)


        print()
        print(
            "Frozen object registration: V5"
        )

        print(
            f"Runtime mu = "
            f"{CONTACT_MU:.3f}"
        )

        print(
            f"Runtime contact stiffness = "
            f"{CONTACT_K:.1f} N/m"
        )

        print(
            "Tendon coefficients = "
            "MCP 5 mm, PIP 4 mm, DIP 1 mm"
        )


        metrics, _ = (
            base.run_contact(
                MODEL_PATH,
                contact_bag,
            )
        )


        pose = np.asarray(
            metrics[
                "pose"
            ],
            dtype=float,
        )


        print()
        print(
            "Established replay result:"
        )

        print(
            "  MCP/PIP/DIP = "
            f"{pose[0]:.3f} / "
            f"{pose[1]:.3f} / "
            f"{pose[2]:.3f} deg"
        )

        print(
            f"  RMSE = "
            f"{metrics['rmse']:.3f} deg"
        )


    finally:

        recorder.restore()

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


    rows = recorder.rows


    if not rows:

        raise RuntimeError(
            "No V21 samples recorded."
        )


    # --------------------------------------------------------
    # Save raw diagnostic.
    # --------------------------------------------------------

    with open(
        CSV_OUT,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    t0 = rows[
        0
    ][
        "time"
    ]


    active = [
        r
        for r in rows
        if (
            r[
                "index_bottle_contacts"
            ]
            > 0
            and
            r[
                "contact_normal_force_N"
            ]
            > 1.0e-4
        )
    ]


    first_contact = (
        active[
            0
        ]
        if active
        else None
    )


    peak_contact = (
        max(
            active,
            key=lambda r:
                r[
                    "contact_normal_force_N"
                ],
        )
        if active
        else None
    )


    max_pip_contact = (
        max(
            active,
            key=lambda r:
                r[
                    "pip_deg"
                ],
        )
        if active
        else None
    )


    last_contact = (
        active[
            -1
        ]
        if active
        else None
    )


    print()
    print("=" * 112)
    print(
        "V21 KEY MECHANICAL STATES"
    )
    print("=" * 112)


    print_state(
        "FIRST BOTTLE/INDEX CONTACT",
        first_contact,
        t0,
    )


    print_state(
        "PEAK BOTTLE/INDEX NORMAL FORCE",
        peak_contact,
        t0,
    )


    print_state(
        "MAXIMUM PIP ANGLE WHILE IN CONTACT",
        max_pip_contact,
        t0,
    )


    print_state(
        "LAST BOTTLE/INDEX CONTACT",
        last_contact,
        t0,
    )


    # ========================================================
    # ACTIVE CONTACT STATISTICS
    # ========================================================

    print()
    print("=" * 112)
    print(
        "V21 ACTIVE-CONTACT PIP TORQUE STATISTICS"
    )
    print("=" * 112)


    if not active:

        print(
            "No active bottle/index contact detected."
        )


    else:

        tendon_tau = np.asarray(
            [
                r[
                    "tendon_tau_pip_Nm"
                ]
                for r in active
            ],
            dtype=float,
        )


        contact_tau = np.asarray(
            [
                r[
                    "contact_tau_pip_Nm"
                ]
                for r in active
            ],
            dtype=float,
        )


        spring_tau = np.asarray(
            [
                r[
                    "pip_spring_tau_Nm"
                ]
                for r in active
            ],
            dtype=float,
        )


        damping_tau = np.asarray(
            [
                r[
                    "pip_damping_tau_Nm"
                ]
                for r in active
            ],
            dtype=float,
        )


        residual = np.asarray(
            [
                r[
                    "residual_constraint_pip_Nm"
                ]
                for r in active
            ],
            dtype=float,
        )


        friction_bound = np.asarray(
            [
                r[
                    "pip_friction_bound_Nm"
                ]
                for r in active
            ],
            dtype=float,
        )


        print()
        print(
            "Positive torque = PIP flexion."
        )

        print(
            "Negative torque = PIP extension."
        )


        print()
        print(
            "PIP tendon flexion torque:"
        )

        print(
            f"  median = "
            f"{percentile(tendon_tau,50):+.5f} Nm"
        )

        print(
            f"  p05/p95 = "
            f"{percentile(tendon_tau,5):+.5f} / "
            f"{percentile(tendon_tau,95):+.5f} Nm"
        )


        print()
        print(
            "PIP bottle-contact torque:"
        )

        print(
            f"  median = "
            f"{percentile(contact_tau,50):+.5f} Nm"
        )

        print(
            f"  p05/p95 = "
            f"{percentile(contact_tau,5):+.5f} / "
            f"{percentile(contact_tau,95):+.5f} Nm"
        )

        print(
            f"  minimum = "
            f"{np.min(contact_tau):+.5f} Nm"
        )

        print(
            f"  maximum = "
            f"{np.max(contact_tau):+.5f} Nm"
        )


        print()
        print(
            "PIP spring+damping torque:"
        )

        sd = (
            spring_tau
            +
            damping_tau
        )

        print(
            f"  median = "
            f"{percentile(sd,50):+.5f} Nm"
        )

        print(
            f"  p05/p95 = "
            f"{percentile(sd,5):+.5f} / "
            f"{percentile(sd,95):+.5f} Nm"
        )


        print()
        print(
            "PIP dry-friction bound:"
        )

        print(
            f"  median = "
            f"{percentile(friction_bound,50):.5f} Nm"
        )


        print()
        print(
            "Residual PIP solver constraint "
            "(after bottle contact subtraction):"
        )

        print(
            f"  median = "
            f"{percentile(residual,50):+.5f} Nm"
        )

        print(
            f"  p05/p95 = "
            f"{percentile(residual,5):+.5f} / "
            f"{percentile(residual,95):+.5f} Nm"
        )


        opposing_contact_fraction = float(
            np.mean(
                contact_tau
                < 0.0
            )
        )


        contact_vs_tendon = (
            np.abs(
                contact_tau
            )
            /
            np.maximum(
                tendon_tau,
                1.0e-9,
            )
        )


        print()
        print(
            f"Fraction of active-contact samples where "
            f"contact torque opposes PIP flexion = "
            f"{100.0*opposing_contact_fraction:.1f}%"
        )


        print(
            f"|contact torque| / tendon torque "
            f"median = "
            f"{percentile(contact_vs_tendon,50):.3f}"
        )

        print(
            f"|contact torque| / tendon torque "
            f"p95 = "
            f"{percentile(contact_vs_tendon,95):.3f}"
        )


        # ====================================================
        # DECISION
        # ====================================================

        print()
        print("=" * 112)
        print(
            "V21 MECHANICAL DECISION"
        )
        print("=" * 112)


        median_contact = percentile(
            contact_tau,
            50,
        )


        median_tendon = percentile(
            tendon_tau,
            50,
        )


        median_passive_opposition = abs(
            min(
                0.0,
                percentile(
                    sd,
                    50,
                ),
            )
        )


        median_friction = percentile(
            friction_bound,
            50,
        )


        if (
            median_contact
            < 0.0
            and
            abs(
                median_contact
            )
            >=
            0.5
            *
            median_tendon
        ):

            print()
            print(
                "CONTACT LOAD IS STRONGLY OPPOSING "
                "PIP FLEXION."
            )

            print()
            print(
                "The middle-phalanx/cylinder contact itself "
                "generates a substantial extension moment "
                "relative to the available tendon flexion "
                "torque."
            )

            print()
            print(
                "NEXT: inspect/replace the simplified rigid "
                "capsule contact surface with a physically "
                "appropriate finger-pad/contact representation "
                "while keeping bottle registration frozen."
            )


        elif (
            median_passive_opposition
            +
            median_friction
            >=
            0.8
            *
            median_tendon
        ):

            print()
            print(
                "PIP PASSIVE RESISTANCE IS CONSUMING MOST "
                "OF THE AVAILABLE TENDON TORQUE."
            )

            print()
            print(
                "NEXT: identify the PIP return-element / "
                "breakaway mechanics rather than changing "
                "contact or bottle position."
            )


        else:

            print()
            print(
                "Neither bottle-contact torque nor the "
                "current passive PIP resistance alone "
                "explains the missing PIP flexion."
            )

            print()
            print(
                "NEXT: the constant 5/4/1 tendon transmission "
                "must be replaced by a load/history-dependent "
                "underactuated tendon model."
            )


    print()
    print(
        "Raw V21 diagnostic:"
    )

    print(
        CSV_OUT
    )


    # ========================================================
    # AUTOMATIC VISUAL REPLAY
    #
    # Keep current trusted baseline:
    #
    #   mu = 0.05
    #   k  = 1000 N/m
    #
    # We do NOT promote V20 k=100 because it did not improve
    # PIP or distal wrapping.
    # ========================================================

    print()
    print("=" * 112)
    print(
        "V21 AUTOMATIC TRUSTED-BASELINE MUJOCO REPLAY"
    )
    print("=" * 112)


    old_mu = (
        viewer_tools.base.CONTACT_MU
    )

    old_contact_k = (
        viewer_tools.base.CONTACT_STIFFNESS
    )


    try:

        viewer_tools.base.CONTACT_MU = (
            CONTACT_MU
        )

        viewer_tools.base.CONTACT_STIFFNESS = (
            CONTACT_K
        )


        viewer_tools.visual_replay(
            MODEL_PATH,
            contact_bag,
        )


    finally:

        viewer_tools.base.CONTACT_MU = (
            old_mu
        )

        viewer_tools.base.CONTACT_STIFFNESS = (
            old_contact_k
        )


    try:

        MODEL_PATH.unlink()

    except FileNotFoundError:

        pass


    print()
    print("=" * 112)
    print(
        "V21 COMPLETE"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
