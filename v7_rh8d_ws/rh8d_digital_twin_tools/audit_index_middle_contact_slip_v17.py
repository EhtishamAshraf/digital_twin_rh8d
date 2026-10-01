#!/usr/bin/env python3

from pathlib import Path
import csv
import math

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as base
import compare_index_configdep_replay_v6 as ab


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()

MJCF_DIR = WS / "src/ur5e_mujoco_description/mjcf"

MODEL = MJCF_DIR / "tmp_v17_trusted_contact.xml"

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "middle_contact_slip_v17"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_OUT = RESULT_DIR / "middle_contact_slip.csv"


SCALE = 5.10e-6
K = 2100.0
OPEN_ENCODER = 100.0

HOLD_ELAPSED = 6.297

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

SPOOL = "index_motor_spool"

BOTTLE = "index_contact_fixture"
MID = "collision_idx_mid"


def oid(model, typ, name):
    i = mujoco.mj_name2id(
        model,
        typ,
        name,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing object: {name}"
        )

    return i


def qadr(model, name):
    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return int(
        model.jnt_qposadr[jid]
    )


def geom_point_velocity(
    model,
    data,
    gid,
    point,
):
    v6 = np.zeros(
        6,
        dtype=float,
    )

    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_GEOM,
        gid,
        v6,
        0,
    )

    omega = v6[:3]
    linear = v6[3:]

    r = (
        np.asarray(
            point,
            dtype=float,
        )
        -
        data.geom_xpos[gid]
    )

    return (
        linear
        +
        np.cross(
            omega,
            r,
        )
    )


class Recorder:

    def __init__(self):
        self.original_step = mujoco.mj_step
        self.rows = []
        self.cache = {}


    def ids(self, model):
        key = id(model)

        if key in self.cache:
            return self.cache[key]

        try:
            d = {
                "mcp_q":
                    qadr(model, MCP),

                "pip_q":
                    qadr(model, PIP),

                "dip_q":
                    qadr(model, DIP),

                "spool_q":
                    qadr(model, SPOOL),

                "bottle":
                    oid(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        BOTTLE,
                    ),

                "mid":
                    oid(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        MID,
                    ),
            }

        except RuntimeError:
            d = None

        self.cache[key] = d
        return d


    def record(self, model, data):

        ids = self.ids(model)

        if ids is None:
            return


        Fn = 0.0
        Ft = 0.0

        max_slip = 0.0

        min_dist = None

        mu_values = []

        contact_rows = 0


        for i in range(data.ncon):

            c = data.contact[i]

            pair = {
                int(c.geom1),
                int(c.geom2),
            }

            if (
                ids["bottle"] not in pair
                or
                ids["mid"] not in pair
            ):
                continue


            contact_rows += 1

            dist = float(c.dist)

            if (
                min_dist is None
                or
                dist < min_dist
            ):
                min_dist = dist


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


            normal_force = abs(
                float(force[0])
            )

            tangent_force = math.hypot(
                float(force[1]),
                float(force[2]),
            )


            Fn += normal_force
            Ft += tangent_force


            # Contact frame normal.
            frame = np.asarray(
                c.frame,
                dtype=float,
            ).reshape(
                3,
                3,
            )

            normal = frame[0]


            vb = geom_point_velocity(
                model,
                data,
                ids["bottle"],
                c.pos,
            )

            vm = geom_point_velocity(
                model,
                data,
                ids["mid"],
                c.pos,
            )


            relative = (
                vm
                -
                vb
            )


            vn = (
                np.dot(
                    relative,
                    normal,
                )
                *
                normal
            )


            vt = (
                relative
                -
                vn
            )


            slip_speed = float(
                np.linalg.norm(
                    vt
                )
            )


            max_slip = max(
                max_slip,
                slip_speed,
            )


            # MuJoCo's compiled contact friction value.
            try:
                mu_values.append(
                    float(c.friction[0])
                )
            except Exception:
                pass


        # Independent signed geometry distance.
        fromto = np.zeros(
            6,
            dtype=float,
        )

        geom_gap = mujoco.mj_geomDistance(
            model,
            data,
            ids["mid"],
            ids["bottle"],
            1.0,
            fromto,
        )


        mu = (
            float(
                np.median(
                    mu_values
                )
            )
            if mu_values
            else float("nan")
        )


        friction_capacity = (
            mu * Fn
            if (
                Fn > 0.0
                and
                np.isfinite(mu)
            )
            else 0.0
        )


        utilization = (
            Ft
            /
            friction_capacity
            if friction_capacity > 1e-12
            else 0.0
        )


        spool = float(
            data.qpos[
                ids["spool_q"]
            ]
        )


        self.rows.append({
            "time":
                float(data.time),

            "encoder_est":
                OPEN_ENCODER
                +
                spool / SCALE,

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

            "middle_geom_gap_mm":
                float(geom_gap)
                * 1000.0,

            "middle_contact_rows":
                contact_rows,

            "middle_contact_dist_mm":
                (
                    min_dist * 1000.0
                    if min_dist is not None
                    else None
                ),

            "Fn_N":
                Fn,

            "Ft_N":
                Ft,

            "mu":
                mu,

            "friction_capacity_N":
                friction_capacity,

            "friction_utilization":
                utilization,

            "tangential_slip_speed_mps":
                max_slip,
        })


    def install(self):
        original = self.original_step
        rec = self

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

            rec.record(
                model,
                data,
            )

            return result

        mujoco.mj_step = wrapped


    def restore(self):
        mujoco.mj_step = (
            self.original_step
        )


def percentile(values, p):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return float("nan")

    return float(
        np.percentile(
            values,
            p,
        )
    )


def print_state(
    label,
    r,
    t0,
):
    print()
    print(label)
    print("-" * 88)

    if r is None:
        print("  not detected")
        return

    print(
        f"  elapsed = "
        f"{r['time']-t0:.3f} s"
    )

    print(
        f"  encoder = "
        f"{r['encoder_est']:.1f}"
    )

    print(
        "  MCP/PIP/DIP = "
        f"{r['mcp_deg']:.3f} / "
        f"{r['pip_deg']:.3f} / "
        f"{r['dip_deg']:.3f} deg"
    )

    print(
        f"  middle gap = "
        f"{r['middle_geom_gap_mm']:+.3f} mm"
    )

    print(
        f"  Fn = "
        f"{r['Fn_N']:.3f} N"
    )

    print(
        f"  Ft = "
        f"{r['Ft_N']:.3f} N"
    )

    print(
        f"  mu = "
        f"{r['mu']:.3f}"
    )

    print(
        f"  Ft/(mu Fn) = "
        f"{r['friction_utilization']:.3f}"
    )

    print(
        f"  tangential slip speed = "
        f"{r['tangential_slip_speed_mps']:.6f} m/s"
    )


def main():

    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            " python3 audit_index_middle_contact_slip_v17.py "
            "/path/to/contact_bag"
        )


    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    ab.replace_index_with_constant_tendon(
        ab.V5_SOURCE,
        MODEL,
    )

    ab.patch_replay_modules()


    rec = Recorder()
    rec.install()


    old_scale = base.METERS_PER_TICK
    old_k = base.TENDON_STIFFNESS


    try:
        base.METERS_PER_TICK = SCALE
        base.TENDON_STIFFNESS = K

        print()
        print("=" * 110)
        print(
            "V17 — TRUSTED MIDDLE-CONTACT "
            "FRICTION / SLIP AUDIT"
        )
        print("=" * 110)

        metrics, _ = base.run_contact(
            MODEL,
            bag,
        )

        print(
            f"Contact RMSE = "
            f"{metrics['rmse']:.3f} deg"
        )

    finally:
        rec.restore()

        base.METERS_PER_TICK = (
            old_scale
        )

        base.TENDON_STIFFNESS = (
            old_k
        )


    rows = rec.rows

    if not rows:
        raise RuntimeError(
            "No replay samples recorded."
        )


    t0 = rows[0]["time"]

    active = [
        r
        for r in rows
        if (
            r["middle_contact_rows"] > 0
            and
            r["Fn_N"] > 1e-4
        )
    ]


    # Save full timeseries.
    with open(
        CSV_OUT,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


    print()
    print("=" * 110)
    print(
        "V17 ACTIVE MIDDLE-CONTACT STATISTICS"
    )
    print("=" * 110)


    if not active:

        print(
            "No meaningful middle contact."
        )

        return


    dt = np.diff(
        [
            r["time"]
            for r in rows
        ]
    )

    median_dt = (
        float(np.median(dt))
        if len(dt)
        else 0.0
    )


    print()
    print(
        f"active samples = "
        f"{len(active)}"
    )

    print(
        f"approx total active time = "
        f"{len(active)*median_dt:.3f} s"
    )


    Fn = [
        r["Fn_N"]
        for r in active
    ]

    Ft = [
        r["Ft_N"]
        for r in active
    ]

    util = [
        r["friction_utilization"]
        for r in active
    ]

    slip = [
        r["tangential_slip_speed_mps"]
        for r in active
    ]


    print()
    print(
        "Normal force Fn [N]:"
    )

    print(
        f"  median = "
        f"{percentile(Fn,50):.3f}"
    )

    print(
        f"  p95    = "
        f"{percentile(Fn,95):.3f}"
    )

    print(
        f"  max    = "
        f"{max(Fn):.3f}"
    )


    print()
    print(
        "Tangential force Ft [N]:"
    )

    print(
        f"  median = "
        f"{percentile(Ft,50):.3f}"
    )

    print(
        f"  p95    = "
        f"{percentile(Ft,95):.3f}"
    )

    print(
        f"  max    = "
        f"{max(Ft):.3f}"
    )


    print()
    print(
        "Friction utilization Ft/(mu Fn):"
    )

    print(
        f"  median = "
        f"{percentile(util,50):.3f}"
    )

    print(
        f"  p95    = "
        f"{percentile(util,95):.3f}"
    )

    print(
        f"  max    = "
        f"{max(util):.3f}"
    )


    print()
    print(
        "Tangential contact speed [m/s]:"
    )

    print(
        f"  median = "
        f"{percentile(slip,50):.6f}"
    )

    print(
        f"  p95    = "
        f"{percentile(slip,95):.6f}"
    )

    print(
        f"  max    = "
        f"{max(slip):.6f}"
    )


    first = active[0]

    peak = max(
        active,
        key=lambda r:
            r["Fn_N"],
    )


    last = active[-1]


    hold = min(
        rows,
        key=lambda r:
            abs(
                (
                    r["time"]
                    -
                    t0
                )
                -
                HOLD_ELAPSED
            ),
    )


    print()
    print("=" * 110)
    print(
        "V17 KEY STATES"
    )
    print("=" * 110)


    print_state(
        "FIRST MIDDLE CONTACT",
        first,
        t0,
    )


    print_state(
        "PEAK MIDDLE NORMAL FORCE",
        peak,
        t0,
    )


    print_state(
        "LAST MIDDLE CONTACT",
        last,
        t0,
    )


    print_state(
        "VIDEO-ALIGNED HOLD",
        hold,
        t0,
    )


    print()
    print("=" * 110)
    print(
        "V17 DECISION"
    )
    print("=" * 110)


    util95 = percentile(
        util,
        95,
    )

    slip95 = percentile(
        slip,
        95,
    )


    if (
        util95 >= 0.80
    ):

        print()
        print(
            "Middle contact is operating close to the "
            "available friction limit."
        )

        print(
            "Contact friction/material behavior is therefore "
            "a credible contributor to the loss of contact."
        )

        print()
        print(
            "NEXT: vary only contact friction while keeping "
            "tendon/passive mechanics frozen."
        )


    else:

        print()
        print(
            "Middle contact is NOT close to the available "
            "friction limit."
        )

        print(
            "Simply increasing cylinder/finger friction is "
            "unlikely to explain the missing PIP wrapping."
        )

        print()
        print(
            "NEXT: return to the underactuated passive-joint/"
            "tendon force distribution, using the contact "
            "constraint as a measured mechanical boundary."
        )


    print()
    print(
        f"p95 friction utilization = "
        f"{util95:.3f}"
    )

    print(
        f"p95 tangential speed = "
        f"{slip95:.6f} m/s"
    )


    print()
    print(
        "Timeseries:"
    )

    print(
        CSV_OUT
    )

    print()
    print("=" * 110)


    try:
        MODEL.unlink()

    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
