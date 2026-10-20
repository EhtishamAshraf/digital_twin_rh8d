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

MJCF = WS / "src/ur5e_mujoco_description/mjcf"

XML = (
    MJCF
    / "ur5e_rh8d_index_segmented_v36a2.xml"
)

BAG = Path(
    "/home/ehtisham/rh8d_system_id/"
    "index_contact_20260911_164310"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v36a3_equivalence_audit"
)

OUTCSV = (
    OUTDIR
    / "v36a3_internal_trace.csv"
)


CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0
CONTACT_MU = 0.05


JOINTS = {
    "mcp":
        "Index_Proximal--palm_assembly:1",

    "pip":
        "Index_Middle--Index_Proximal",

    "dip":
        "Index_Distal--Index_Middle",

    "spool":
        "index_motor_spool",

    "s1":
        "idx_v36_slip_mcp",

    "s2":
        "idx_v36_slip_pip",

    "e1":
        "idx_v36_ext1",

    "e2":
        "idx_v36_ext2",

    "e3":
        "idx_v36_ext3",
}


GEOM_TENDONS = [
    "idx_v36_seg1_geom",
    "idx_v36_seg2_geom",
    "idx_v36_seg3_geom",
]

STATE_TENDONS = [
    "idx_v36_seg1_state",
    "idx_v36_seg2_state",
    "idx_v36_seg3_state",
]


def jid(model, name):

    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing joint: {name}"
        )

    return i


def tid(model, name):

    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        name,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing tendon: {name}"
        )

    return i


def qadr(model, name):

    return int(
        model.jnt_qposadr[
            jid(model, name)
        ]
    )


def dadr(model, name):

    return int(
        model.jnt_dofadr[
            jid(model, name)
        ]
    )


class Audit:

    def __init__(self):

        self.initialized = False

        self.rows = []

        self.geom0 = None
        self.state0 = None
        self.spool0 = None


    def inspect(
        self,
        model,
        data,
    ):

        tg = [
            tid(
                model,
                x,
            )
            for x in GEOM_TENDONS
        ]

        ts = [
            tid(
                model,
                x,
            )
            for x in STATE_TENDONS
        ]


        if not self.initialized:

            self.geom0 = np.array(
                [
                    data.ten_length[x]
                    for x in tg
                ],
                dtype=float,
            )

            self.state0 = np.array(
                [
                    data.ten_length[x]
                    for x in ts
                ],
                dtype=float,
            )

            self.spool0 = float(
                data.qpos[
                    qadr(
                        model,
                        JOINTS["spool"],
                    )
                ]
            )

            self.initialized = True


        geom = np.array(
            [
                data.ten_length[x]
                for x in tg
            ],
            dtype=float,
        )

        state = np.array(
            [
                data.ten_length[x]
                for x in ts
            ],
            dtype=float,
        )


        dgeom = (
            geom
            -
            self.geom0
        )

        dstate = (
            state
            -
            self.state0
        )


        # Equality residual:
        #
        #   ΔL_geom - ΔL_state
        #
        eq_res = (
            dgeom
            -
            dstate
        )


        spool = float(
            data.qpos[
                qadr(
                    model,
                    JOINTS["spool"],
                )
            ]
        )

        du = (
            spool
            -
            self.spool0
        )


        e = []
        edot = []
        eacc = []

        K = []
        D = []
        A = []


        for key in [
            "e1",
            "e2",
            "e3",
        ]:

            jname = JOINTS[
                key
            ]

            ji = jid(
                model,
                jname,
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


            e.append(
                float(
                    data.qpos[
                        qa
                    ]
                )
            )

            edot.append(
                float(
                    data.qvel[
                        da
                    ]
                )
            )

            eacc.append(
                float(
                    data.qacc[
                        da
                    ]
                )
            )

            K.append(
                float(
                    model.jnt_stiffness[
                        ji
                    ]
                )
            )

            D.append(
                float(
                    model.dof_damping[
                        da
                    ]
                )
            )

            A.append(
                float(
                    model.dof_armature[
                        da
                    ]
                )
            )


        e = np.asarray(
            e,
            dtype=float,
        )

        edot = np.asarray(
            edot,
            dtype=float,
        )

        eacc = np.asarray(
            eacc,
            dtype=float,
        )

        K = np.asarray(
            K,
            dtype=float,
        )

        D = np.asarray(
            D,
            dtype=float,
        )

        A = np.asarray(
            A,
            dtype=float,
        )


        # Elastic/viscous tensile-force estimate.
        T = (
            K * e
            +
            D * edot
        )


        s = []
        sdot = []
        sacc = []
        sarm = []


        for key in [
            "s1",
            "s2",
        ]:

            jname = JOINTS[
                key
            ]

            ji = jid(
                model,
                jname,
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


            s.append(
                float(
                    data.qpos[
                        qa
                    ]
                )
            )

            sdot.append(
                float(
                    data.qvel[
                        da
                    ]
                )
            )

            sacc.append(
                float(
                    data.qacc[
                        da
                    ]
                )
            )

            sarm.append(
                float(
                    model.dof_armature[
                        da
                    ]
                )
            )


        s = np.asarray(
            s
        )

        sdot = np.asarray(
            sdot
        )

        sacc = np.asarray(
            sacc
        )

        sarm = np.asarray(
            sarm
        )


        slip_inertia = (
            sarm
            *
            sacc
        )


        # Cable conservation:
        #
        # e1+e2+e3 should equal
        #
        # ΔLgeom1 + ΔLgeom2 + ΔLgeom3 + spool displacement
        #
        expected_ext = (
            float(
                np.sum(
                    dgeom
                )
            )
            +
            du
        )


        conservation = (
            float(
                np.sum(
                    e
                )
            )
            -
            expected_ext
        )


        qdeg = {}


        for key in [
            "mcp",
            "pip",
            "dip",
        ]:

            qdeg[key] = math.degrees(
                float(
                    data.qpos[
                        qadr(
                            model,
                            JOINTS[key],
                        )
                    ]
                )
            )


        row = {

            "time_s":
                float(
                    data.time
                ),

            "mcp_deg":
                qdeg["mcp"],

            "pip_deg":
                qdeg["pip"],

            "dip_deg":
                qdeg["dip"],

            "spool_mm":
                spool
                *
                1000.0,

            "s1_mm":
                s[0]
                *
                1000.0,

            "s2_mm":
                s[1]
                *
                1000.0,

            "s1_acc":
                sacc[0],

            "s2_acc":
                sacc[1],

            "s1_inertia_N":
                slip_inertia[0],

            "s2_inertia_N":
                slip_inertia[1],

            "e1_mm":
                e[0]
                *
                1000.0,

            "e2_mm":
                e[1]
                *
                1000.0,

            "e3_mm":
                e[2]
                *
                1000.0,

            "T1_N":
                T[0],

            "T2_N":
                T[1],

            "T3_N":
                T[2],

            "dT12_N":
                T[0]
                -
                T[1],

            "dT23_N":
                T[1]
                -
                T[2],

            "eq1_um":
                eq_res[0]
                *
                1.0e6,

            "eq2_um":
                eq_res[1]
                *
                1.0e6,

            "eq3_um":
                eq_res[2]
                *
                1.0e6,

            "conservation_um":
                conservation
                *
                1.0e6,
        }


        self.rows.append(
            row
        )


def nearest_pose(
    rows,
    pose,
):

    q = np.asarray(
        [
            [
                r["mcp_deg"],
                r["pip_deg"],
                r["dip_deg"],
            ]
            for r in rows
        ],
        dtype=float,
    )

    target = np.asarray(
        pose,
        dtype=float,
    )

    error = np.linalg.norm(
        q
        -
        target[
            None,
            :
        ],
        axis=1,
    )

    return rows[
        int(
            np.argmin(
                error
            )
        )
    ]


def maxabs(
    rows,
    key,
):

    return max(
        abs(
            float(
                r[key]
            )
        )
        for r in rows
    )


def percentile_abs(
    rows,
    key,
    p,
):

    v = np.array(
        [
            abs(
                float(
                    r[key]
                )
            )
            for r in rows
        ],
        dtype=float,
    )

    return float(
        np.percentile(
            v,
            p,
        )
    )


def main():

    if not XML.exists():
        raise RuntimeError(
            f"Missing:\n{XML}"
        )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    audit = Audit()

    real_step = mujoco.mj_step


    def wrapped_step(
        model,
        data,
        *args,
        **kwargs,
    ):

        result = real_step(
            model,
            data,
            *args,
            **kwargs,
        )

        audit.inspect(
            model,
            data,
        )

        return result


    print()
    print("=" * 150)
    print("V36A3 — ZERO-FRICTION SEGMENTED-CABLE INTERNAL AUDIT")
    print("=" * 150)

    print(
        "NO PHYSICAL PARAMETER IS BEING CHANGED."
    )

    print(
        "Checking:"
    )

    print(
        "  1. equality residuals"
    )

    print(
        "  2. cable conservation"
    )

    print(
        "  3. T1/T2/T3 equality"
    )

    print(
        "  4. artificial slip-coordinate inertial forces"
    )


    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
    )


    old_cd = float(
        contact_replay.CONTACT_DAMPING
    )

    old_mu = float(
        b2.v35.CONTACT_MU
    )

    old_ck = float(
        b2.v35.CONTACT_K
    )

    old_tk = float(
        b2.v35.TENDON_K
    )

    old_btk = float(
        b2.v30.v24.base.TENDON_STIFFNESS
    )


    contact_replay.CONTACT_DAMPING = (
        CONTACT_DAMPING
    )

    b2.v35.CONTACT_MU = (
        CONTACT_MU
    )

    b2.v35.CONTACT_K = (
        CONTACT_K
    )


    # Original whole spatial tendon must remain passive.
    b2.v35.TENDON_K = 0.0

    b2.v30.v24.base.TENDON_STIFFNESS = (
        0.0
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

        contact_replay.CONTACT_DAMPING = (
            old_cd
        )

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_ck
        )

        b2.v35.TENDON_K = (
            old_tk
        )

        b2.v30.v24.base.TENDON_STIFFNESS = (
            old_btk
        )


    if not audit.rows:

        raise RuntimeError(
            "No audit rows captured."
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


    target_row = nearest_pose(
        audit.rows,
        result["pose"],
    )


    print()
    print("=" * 150)
    print("RESULT POSE")
    print("=" * 150)

    print(
        "pose = "
        f"{result['pose'][0]:.3f} / "
        f"{result['pose'][1]:.3f} / "
        f"{result['pose'][2]:.3f} deg"
    )

    print(
        f"hardware RMSE = "
        f"{result['rmse']:.3f} deg"
    )


    r = target_row


    print()
    print("=" * 150)
    print("INTERNAL STATE NEAREST RESULT POSE")
    print("=" * 150)

    print(
        f"time = "
        f"{r['time_s']:.4f} s"
    )

    print(
        "q = "
        f"{r['mcp_deg']:.3f} / "
        f"{r['pip_deg']:.3f} / "
        f"{r['dip_deg']:.3f} deg"
    )

    print(
        f"spool = "
        f"{r['spool_mm']:.4f} mm"
    )

    print()
    print(
        "slip coordinates:"
    )

    print(
        f"  s1 = "
        f"{r['s1_mm']:+.6f} mm"
    )

    print(
        f"  s2 = "
        f"{r['s2_mm']:+.6f} mm"
    )

    print()
    print(
        "segment extensions:"
    )

    print(
        f"  e1/e2/e3 = "
        f"{r['e1_mm']:+.6f} / "
        f"{r['e2_mm']:+.6f} / "
        f"{r['e3_mm']:+.6f} mm"
    )

    print()
    print(
        "estimated segment tensions:"
    )

    print(
        f"  T1/T2/T3 = "
        f"{r['T1_N']:+.6f} / "
        f"{r['T2_N']:+.6f} / "
        f"{r['T3_N']:+.6f} N"
    )

    print(
        f"  T1-T2 = "
        f"{r['dT12_N']:+.6f} N"
    )

    print(
        f"  T2-T3 = "
        f"{r['dT23_N']:+.6f} N"
    )

    print()
    print(
        "artificial slip-coordinate inertia:"
    )

    print(
        f"  M*a s1 = "
        f"{r['s1_inertia_N']:+.6f} N"
    )

    print(
        f"  M*a s2 = "
        f"{r['s2_inertia_N']:+.6f} N"
    )

    print()
    print(
        "equality residuals:"
    )

    print(
        f"  r1/r2/r3 = "
        f"{r['eq1_um']:+.3f} / "
        f"{r['eq2_um']:+.3f} / "
        f"{r['eq3_um']:+.3f} um"
    )

    print(
        f"cable conservation residual = "
        f"{r['conservation_um']:+.3f} um"
    )


    print()
    print("=" * 150)
    print("WHOLE-RUN DIAGNOSTIC MAGNITUDES")
    print("=" * 150)

    print(
        "95th percentile |T1-T2| = "
        f"{percentile_abs(audit.rows, 'dT12_N', 95):.6f} N"
    )

    print(
        "95th percentile |T2-T3| = "
        f"{percentile_abs(audit.rows, 'dT23_N', 95):.6f} N"
    )

    print(
        "max |T1-T2| = "
        f"{maxabs(audit.rows, 'dT12_N'):.6f} N"
    )

    print(
        "max |T2-T3| = "
        f"{maxabs(audit.rows, 'dT23_N'):.6f} N"
    )

    print()
    print(
        "95th percentile |M*a s1| = "
        f"{percentile_abs(audit.rows, 's1_inertia_N', 95):.6f} N"
    )

    print(
        "95th percentile |M*a s2| = "
        f"{percentile_abs(audit.rows, 's2_inertia_N', 95):.6f} N"
    )

    print()
    print(
        "max equality residuals:"
    )

    print(
        f"  seg1 = "
        f"{maxabs(audit.rows, 'eq1_um'):.3f} um"
    )

    print(
        f"  seg2 = "
        f"{maxabs(audit.rows, 'eq2_um'):.3f} um"
    )

    print(
        f"  seg3 = "
        f"{maxabs(audit.rows, 'eq3_um'):.3f} um"
    )

    print(
        "max cable-conservation residual = "
        f"{maxabs(audit.rows, 'conservation_um'):.3f} um"
    )


    print()
    print("=" * 150)
    print("INTERPRETATION")
    print("=" * 150)

    p95_dt = max(
        percentile_abs(
            audit.rows,
            "dT12_N",
            95,
        ),
        percentile_abs(
            audit.rows,
            "dT23_N",
            95,
        ),
    )

    p95_inertia = max(
        percentile_abs(
            audit.rows,
            "s1_inertia_N",
            95,
        ),
        percentile_abs(
            audit.rows,
            "s2_inertia_N",
            95,
        ),
    )

    max_eq = max(
        maxabs(
            audit.rows,
            "eq1_um",
        ),
        maxabs(
            audit.rows,
            "eq2_um",
        ),
        maxabs(
            audit.rows,
            "eq3_um",
        ),
    )


    if (
        p95_dt > 0.10
        and
        p95_inertia > 0.05
    ):

        print(
            "PRIMARY SUSPECT: artificial inertia of the "
            "frictionless slip coordinates."
        )

        print(
            "Zero guide friction is NOT producing "
            "approximately equal dynamic segment tension."
        )

    elif max_eq > 50.0:

        print(
            "PRIMARY SUSPECT: soft tendon-equality constraints."
        )

        print(
            "The three geometric segments are not following "
            "their cable-state coordinates accurately enough."
        )

    else:

        print(
            "No single failure is obvious from thresholds."
        )

        print(
            "Inspect the numerical values above before changing "
            "the architecture."
        )


    print()
    print(
        f"CSV:\n{OUTCSV}"
    )

    print("=" * 150)


if __name__ == "__main__":
    main()
