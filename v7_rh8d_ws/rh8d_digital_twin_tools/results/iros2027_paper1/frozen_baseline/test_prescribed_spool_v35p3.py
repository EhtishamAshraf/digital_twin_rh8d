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
    / "ur5e_rh8d_index_cad_spatial_v35i1_best.xml"
)

BAG = Path(
    "/home/ehtisham/rh8d_system_id/"
    "index_contact_20260911_164310"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v35p3_prescribed_spool"
)

OUTCSV = OUTDIR / "v35p3_summary.csv"


CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0
CONTACT_MU = 0.05

CASES = [
    ("SERVO_K9000",      9000.0, False),
    ("PRESCRIBED_K9000", 9000.0, True),
    ("PRESCRIBED_K2100", 2100.0, True),
]


SPOOL = "index_motor_spool"
ACTUATOR = "pos_index_tendon"

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

FIXTURE = "index_contact_fixture"
MID = "collision_idx_mid_mesh"
TIP = "collision_idx_tip"


def jid(model, name):
    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )
    if i < 0:
        raise RuntimeError(f"Missing joint: {name}")
    return i


def aid(model, name):
    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )
    if i < 0:
        raise RuntimeError(f"Missing actuator: {name}")
    return i


def gid(model, name):
    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        name,
    )
    if i < 0:
        raise RuntimeError(f"Missing geom: {name}")
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

    def __init__(self, prescribed):
        self.prescribed = prescribed

        self.prev_target = None
        self.prev_t = None

        self.prev_mid = False
        self.prev_tip = False

        self.mid_on = 0
        self.mid_off = 0
        self.tip_on = 0
        self.tip_off = 0

        self.first_mid = None
        self.first_tip = None

        self.rows = []

        self.max_pre_step_error_mm = 0.0
        self.max_post_step_drift_mm = 0.0


    def contacts(self, model, data):

        fixture = gid(model, FIXTURE)
        mid_gid = gid(model, MID)
        tip_gid = gid(model, TIP)

        mid = False
        tip = False

        for ci in range(data.ncon):

            c = data.contact[ci]

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


    def snapshot(self, model, data):

        qa = qadr(model, SPOOL)
        da = dadr(model, SPOOL)
        ai = aid(model, ACTUATOR)

        return {
            "time":
                float(data.time),

            "target_mm":
                float(data.ctrl[ai]) * 1000.0,

            "actual_mm":
                float(data.qpos[qa]) * 1000.0,

            "spool_vel_mm_s":
                float(data.qvel[da]) * 1000.0,

            "mcp":
                math.degrees(
                    float(
                        data.qpos[
                            qadr(model, MCP)
                        ]
                    )
                ),

            "pip":
                math.degrees(
                    float(
                        data.qpos[
                            qadr(model, PIP)
                        ]
                    )
                ),

            "dip":
                math.degrees(
                    float(
                        data.qpos[
                            qadr(model, DIP)
                        ]
                    )
                ),

            "actuator_force":
                float(
                    data.actuator_force[ai]
                ),
        }


    def inspect_events(self, model, data):

        mid, tip = self.contacts(
            model,
            data,
        )

        if mid and not self.prev_mid:

            self.mid_on += 1

            if self.first_mid is None:
                self.first_mid = self.snapshot(
                    model,
                    data,
                )

        if self.prev_mid and not mid:
            self.mid_off += 1


        if tip and not self.prev_tip:

            self.tip_on += 1

            if self.first_tip is None:
                self.first_tip = self.snapshot(
                    model,
                    data,
                )

        if self.prev_tip and not tip:
            self.tip_off += 1


        self.prev_mid = mid
        self.prev_tip = tip


    def before_step(self, model, data):

        qa = qadr(model, SPOOL)
        da = dadr(model, SPOOL)
        ai = aid(model, ACTUATOR)

        target = float(
            data.ctrl[ai]
        )

        t = float(
            data.time
        )


        if (
            self.prev_target is not None
            and
            self.prev_t is not None
            and
            t > self.prev_t
        ):
            target_vel = (
                target
                -
                self.prev_target
            ) / (
                t
                -
                self.prev_t
            )
        else:
            target_vel = 0.0


        error_before = (
            target
            -
            float(data.qpos[qa])
        )

        self.max_pre_step_error_mm = max(
            self.max_pre_step_error_mm,
            abs(error_before) * 1000.0,
        )


        if self.prescribed:

            # ------------------------------------------------
            # The hardware encoder is an observed input.
            #
            # Set only the hidden motor/spool coordinate.
            # MCP/PIP/DIP remain fully dynamic.
            # ------------------------------------------------

            data.qpos[qa] = target

            data.qvel[da] = target_vel

            data.qacc[da] = 0.0


        self.prev_target = target
        self.prev_t = t


    def after_step(
        self,
        model,
        data,
        target_used,
    ):

        qa = qadr(model, SPOOL)

        drift = (
            float(data.qpos[qa])
            -
            target_used
        )

        self.max_post_step_drift_mm = max(
            self.max_post_step_drift_mm,
            abs(drift) * 1000.0,
        )


        self.inspect_events(
            model,
            data,
        )

        s = self.snapshot(
            model,
            data,
        )

        self.rows.append(
            s
        )


def run_case(
    label,
    tendon_k,
    prescribed,
    contact_replay,
):

    audit = Audit(
        prescribed=prescribed,
    )

    real_step = mujoco.mj_step


    def wrapped_step(
        model,
        data,
        *args,
        **kwargs,
    ):

        ai = aid(
            model,
            ACTUATOR,
        )

        target = float(
            data.ctrl[ai]
        )


        audit.before_step(
            model,
            data,
        )


        out = real_step(
            model,
            data,
            *args,
            **kwargs,
        )


        audit.after_step(
            model,
            data,
            target,
        )


        return out


    old_tk = float(
        b2.v35.TENDON_K
    )

    old_btk = float(
        b2.v30.v24.base.TENDON_STIFFNESS
    )


    b2.v35.TENDON_K = float(
        tendon_k
    )

    b2.v30.v24.base.TENDON_STIFFNESS = float(
        tendon_k
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

        b2.v35.TENDON_K = old_tk
        b2.v30.v24.base.TENDON_STIFFNESS = (
            old_btk
        )


    mid = audit.first_mid
    tip = audit.first_tip


    if (
        mid is not None
        and
        tip is not None
        and
        tip["time"] >= mid["time"]
    ):

        delay = (
            tip["time"]
            -
            mid["time"]
        )

        dpip = (
            tip["pip"]
            -
            mid["pip"]
        )

        ddip = (
            tip["dip"]
            -
            mid["dip"]
        )

    else:

        delay = math.nan
        dpip = math.nan
        ddip = math.nan


    q = np.asarray(
        result["pose"],
        dtype=float,
    )


    row = {
        "case":
            label,

        "tendon_k":
            tendon_k,

        "prescribed":
            prescribed,

        "mcp_deg":
            q[0],

        "pip_deg":
            q[1],

        "dip_deg":
            q[2],

        "rmse_deg":
            result["rmse"],

        "mid_on":
            audit.mid_on,

        "mid_off":
            audit.mid_off,

        "tip_on":
            audit.tip_on,

        "tip_off":
            audit.tip_off,

        "middle_to_tip_delay_s":
            delay,

        "middle_to_tip_dPIP_deg":
            dpip,

        "middle_to_tip_dDIP_deg":
            ddip,

        "first_mid_MCP":
            (
                mid["mcp"]
                if mid is not None
                else math.nan
            ),

        "first_mid_PIP":
            (
                mid["pip"]
                if mid is not None
                else math.nan
            ),

        "first_mid_DIP":
            (
                mid["dip"]
                if mid is not None
                else math.nan
            ),

        "max_pre_step_error_mm":
            audit.max_pre_step_error_mm,

        "max_post_step_drift_mm":
            audit.max_post_step_drift_mm,

        "target_mid_gap_mm":
            result.get(
                "target_mid_gap_mm"
            ),

        "target_tip_gap_mm":
            result.get(
                "target_tip_gap_mm"
            ),
    }


    trace_path = (
        OUTDIR
        /
        f"{label}_trace.csv"
    )

    if audit.rows:

        with trace_path.open(
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


    return row


def ff(x, n=3):

    try:
        x = float(x)

        if math.isnan(x):
            return "nan"

        return f"{x:.{n}f}"

    except Exception:
        return str(x)


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print("=" * 175)
    print("V35P3 — MEASURED ENCODER AS PRESCRIBED SPOOL COORDINATE")
    print("=" * 175)

    print(
        f"MODEL:\n{XML}"
    )

    print()
    print(
        "This experiment does NOT prescribe finger joints."
    )

    print(
        "Only index_motor_spool is overwritten from the "
        "already-measured hardware encoder target."
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


    contact_replay.CONTACT_DAMPING = (
        CONTACT_DAMPING
    )

    b2.v35.CONTACT_MU = (
        CONTACT_MU
    )

    b2.v35.CONTACT_K = (
        CONTACT_K
    )


    rows = []


    try:

        for i, (
            label,
            K,
            prescribed,
        ) in enumerate(
            CASES,
            1,
        ):

            print()
            print("-" * 175)

            print(
                f"[{i}/{len(CASES)}] {label}"
            )

            print(
                f"tendon K = {K:.0f} N/m"
            )

            print(
                f"spool mode = "
                f"{'PRESCRIBED FROM MEASURED ENCODER' if prescribed else 'OLD FINITE SERVO'}"
            )


            r = run_case(
                label,
                K,
                prescribed,
                contact_replay,
            )

            rows.append(r)


            print(
                "pose = "
                f"{r['mcp_deg']:.3f} / "
                f"{r['pip_deg']:.3f} / "
                f"{r['dip_deg']:.3f} deg"
            )

            print(
                f"RMSE = "
                f"{r['rmse_deg']:.3f} deg"
            )

            print(
                "middle/tip on-off = "
                f"{r['mid_on']}/{r['mid_off']}  "
                f"{r['tip_on']}/{r['tip_off']}"
            )

            print(
                "first middle pose = "
                f"{ff(r['first_mid_MCP'])} / "
                f"{ff(r['first_mid_PIP'])} / "
                f"{ff(r['first_mid_DIP'])}"
            )

            print(
                "middle -> first tip:"
            )

            print(
                f"  delay = "
                f"{ff(r['middle_to_tip_delay_s'])} s"
            )

            print(
                f"  dPIP  = "
                f"{ff(r['middle_to_tip_dPIP_deg'])} deg"
            )

            print(
                f"  dDIP  = "
                f"{ff(r['middle_to_tip_dDIP_deg'])} deg"
            )

            print(
                "spool enforcement:"
            )

            print(
                f"  max pre-step target error = "
                f"{r['max_pre_step_error_mm']:.4f} mm"
            )

            print(
                f"  max within-step drift     = "
                f"{r['max_post_step_drift_mm']:.6f} mm"
            )

            print(
                "target gaps M/T = "
                f"{ff(r['target_mid_gap_mm'])} / "
                f"{ff(r['target_tip_gap_mm'])} mm"
            )


    finally:

        contact_replay.CONTACT_DAMPING = (
            old_cd
        )

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_ck
        )


    with OUTCSV.open(
        "w",
        newline="",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    print()
    print("=" * 200)
    print("V35P3 SUMMARY")
    print("=" * 200)

    print(
        "case                K       mode         RMSE    "
        "MCP/PIP/DIP             M/T on-off    "
        "dPIP M->T   dDIP M->T    "
        "preErr(mm)  stepDrift(mm)     gaps M/T"
    )

    print("-" * 200)


    for r in rows:

        mode = (
            "PRESCRIBED"
            if r["prescribed"]
            else "SERVO"
        )

        print(
            f"{r['case']:20s} "
            f"{r['tendon_k']:6.0f}  "
            f"{mode:10s} "
            f"{r['rmse_deg']:6.3f}   "
            f"{r['mcp_deg']:6.2f}/"
            f"{r['pip_deg']:6.2f}/"
            f"{r['dip_deg']:6.2f}      "
            f"{r['mid_on']:1d}/{r['mid_off']:1d} "
            f"{r['tip_on']:1d}/{r['tip_off']:1d}        "
            f"{ff(r['middle_to_tip_dPIP_deg']):>7s}     "
            f"{ff(r['middle_to_tip_dDIP_deg']):>7s}      "
            f"{r['max_pre_step_error_mm']:7.4f}      "
            f"{r['max_post_step_drift_mm']:9.6f}      "
            f"{ff(r['target_mid_gap_mm']):>6s}/"
            f"{ff(r['target_tip_gap_mm']):>6s}"
        )


    print()
    print(
        f"CSV:\n{OUTCSV}"
    )

    print(
        f"\nExperiment folder:\n{OUTDIR}"
    )

    print("=" * 200)


if __name__ == "__main__":
    main()
