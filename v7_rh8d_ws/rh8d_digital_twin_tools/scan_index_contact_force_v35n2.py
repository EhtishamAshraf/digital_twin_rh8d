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

OUT = (
    MJCF
    / "v35n2_contact_force_scan.csv"
)


# Targeted scan only.
CONTACT_K_VALUES = [
    50.0,
    100.0,
    150.0,
    200.0,
    300.0,
    500.0,
    1000.0,
]

CONTACT_DAMPING = 20.0
CONTACT_MU = 0.05

FIXTURE = "index_contact_fixture"

MID = "collision_idx_mid_mesh"
TIP = "collision_idx_tip"


class ContactStats:

    def __init__(self):

        self.prev_mid = False
        self.prev_tip = False

        self.mid_on = 0
        self.mid_off = 0

        self.tip_on = 0
        self.tip_off = 0

        self.mid_forces = []
        self.tip_forces = []

        self.mid_positive = []
        self.tip_positive = []

        self.max_mid_force = 0.0
        self.max_tip_force = 0.0

        self.max_mid_pen = 0.0
        self.max_tip_pen = 0.0


    def inspect(self, model, data):

        fixture = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            FIXTURE,
        )

        if fixture < 0:
            return


        mid_active = False
        tip_active = False

        mid_force = 0.0
        tip_force = 0.0

        mid_pen = 0.0
        tip_pen = 0.0


        for ci in range(data.ncon):

            c = data.contact[ci]

            g1 = int(c.geom1)
            g2 = int(c.geom2)

            if fixture not in (g1, g2):
                continue


            other = (
                g2
                if g1 == fixture
                else g1
            )


            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                other,
            )

            if name not in (MID, TIP):
                continue


            force6 = np.zeros(
                6,
                dtype=float,
            )

            mujoco.mj_contactForce(
                model,
                data,
                ci,
                force6,
            )

            fn = max(
                0.0,
                float(force6[0]),
            )

            penetration = max(
                0.0,
                -float(c.dist),
            )


            if name == MID:

                mid_active = True

                mid_force += fn

                mid_pen = max(
                    mid_pen,
                    penetration,
                )


            elif name == TIP:

                tip_active = True

                tip_force += fn

                tip_pen = max(
                    tip_pen,
                    penetration,
                )


        if (
            mid_active
            and not self.prev_mid
        ):
            self.mid_on += 1

        if (
            self.prev_mid
            and not mid_active
        ):
            self.mid_off += 1


        if (
            tip_active
            and not self.prev_tip
        ):
            self.tip_on += 1

        if (
            self.prev_tip
            and not tip_active
        ):
            self.tip_off += 1


        self.prev_mid = mid_active
        self.prev_tip = tip_active


        if mid_active:

            self.mid_forces.append(
                mid_force
            )

            if mid_force > 0:

                self.mid_positive.append(
                    mid_force
                )

            self.max_mid_force = max(
                self.max_mid_force,
                mid_force,
            )

            self.max_mid_pen = max(
                self.max_mid_pen,
                mid_pen,
            )


        if tip_active:

            self.tip_forces.append(
                tip_force
            )

            if tip_force > 0:

                self.tip_positive.append(
                    tip_force
                )

            self.max_tip_force = max(
                self.max_tip_force,
                tip_force,
            )

            self.max_tip_pen = max(
                self.max_tip_pen,
                tip_pen,
            )


    @staticmethod
    def median(values):

        if not values:
            return math.nan

        return float(
            np.median(
                values
            )
        )


    @staticmethod
    def mean(values):

        if not values:
            return math.nan

        return float(
            np.mean(
                values
            )
        )


    def summary(self):

        return {

            "mid_on":
                self.mid_on,

            "mid_off":
                self.mid_off,

            "tip_on":
                self.tip_on,

            "tip_off":
                self.tip_off,

            "mid_force_median":
                self.median(
                    self.mid_positive
                ),

            "mid_force_mean":
                self.mean(
                    self.mid_positive
                ),

            "mid_force_max":
                self.max_mid_force,

            "tip_force_median":
                self.median(
                    self.tip_positive
                ),

            "tip_force_max":
                self.max_tip_force,

            "mid_pen_max_mm":
                self.max_mid_pen
                * 1000.0,

            "tip_pen_max_mm":
                self.max_tip_pen
                * 1000.0,
        }


def run_one(
    contact_k,
    contact_replay,
):

    stats = ContactStats()

    real_step = mujoco.mj_step


    def audited_step(
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

        stats.inspect(
            model,
            data,
        )

        return out


    mujoco.mj_step = audited_step


    try:

        b2.v35.CONTACT_K = float(
            contact_k
        )

        result = e2b.run_one(
            CONTACT_DAMPING,
            XML,
            BAG,
            contact_replay,
        )


    finally:

        mujoco.mj_step = real_step


    s = stats.summary()

    q = result["pose"]


    row = {

        "contact_k":
            contact_k,

        "mcp_deg":
            q[0],

        "pip_deg":
            q[1],

        "dip_deg":
            q[2],

        "rmse_deg":
            result["rmse"],

        "mid_on":
            s["mid_on"],

        "mid_off":
            s["mid_off"],

        "tip_on":
            s["tip_on"],

        "tip_off":
            s["tip_off"],

        "mid_force_median_N":
            s["mid_force_median"],

        "mid_force_mean_N":
            s["mid_force_mean"],

        "mid_force_max_N":
            s["mid_force_max"],

        "tip_force_median_N":
            s["tip_force_median"],

        "tip_force_max_N":
            s["tip_force_max"],

        "mid_pen_max_mm":
            s["mid_pen_max_mm"],

        "tip_pen_max_mm":
            s["tip_pen_max_mm"],

        "target_mid_gap_mm":
            result.get(
                "target_mid_gap_mm"
            ),

        "target_tip_gap_mm":
            result.get(
                "target_tip_gap_mm"
            ),

        "overlap_longest_s":
            result.get(
                "overlap_longest"
            ),
    }

    return row


def f(value, digits=3):

    try:

        if math.isnan(
            float(value)
        ):
            return "nan"

    except Exception:
        pass

    return (
        f"{float(value):.{digits}f}"
    )


def main():

    if not XML.exists():

        raise RuntimeError(
            f"Missing XML:\n{XML}"
        )

    if not BAG.exists():

        raise RuntimeError(
            f"Missing bag:\n{BAG}"
        )


    print()
    print("=" * 150)
    print("V35N2 — TARGETED CONTACT FORCE / STIFFNESS TEST")
    print("=" * 150)

    print(
        f"XML:\n{XML}"
    )

    print()
    print(
        "Everything except contact stiffness is frozen."
    )

    print(
        f"contact damping = "
        f"{CONTACT_DAMPING}"
    )

    print(
        f"contact mu      = "
        f"{CONTACT_MU}"
    )


    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
    )


    old_damping = float(
        contact_replay.CONTACT_DAMPING
    )

    old_mu = float(
        b2.v35.CONTACT_MU
    )

    old_k = float(
        b2.v35.CONTACT_K
    )


    contact_replay.CONTACT_DAMPING = (
        CONTACT_DAMPING
    )

    b2.v35.CONTACT_MU = (
        CONTACT_MU
    )


    rows = []


    try:

        for i, k in enumerate(
            CONTACT_K_VALUES,
            1,
        ):

            print()
            print(
                f"[{i}/{len(CONTACT_K_VALUES)}] "
                f"CONTACT_K = {k:g}"
            )

            row = run_one(
                k,
                contact_replay,
            )

            rows.append(
                row
            )


            print(
                "  pose = "
                f"{row['mcp_deg']:.2f}/"
                f"{row['pip_deg']:.2f}/"
                f"{row['dip_deg']:.2f} deg"
            )

            print(
                f"  RMSE = "
                f"{row['rmse_deg']:.3f} deg"
            )

            print(
                "  events mid on/off = "
                f"{row['mid_on']}/"
                f"{row['mid_off']}   "
                "tip on/off = "
                f"{row['tip_on']}/"
                f"{row['tip_off']}"
            )

            print(
                "  middle Fn "
                "median/mean/max = "
                f"{f(row['mid_force_median_N'])}/"
                f"{f(row['mid_force_mean_N'])}/"
                f"{f(row['mid_force_max_N'])} N"
            )

            print(
                "  tip Fn "
                "median/max = "
                f"{f(row['tip_force_median_N'])}/"
                f"{f(row['tip_force_max_N'])} N"
            )

            print(
                "  max penetration "
                "mid/tip = "
                f"{f(row['mid_pen_max_mm'])}/"
                f"{f(row['tip_pen_max_mm'])} mm"
            )

            print(
                "  target gaps "
                "mid/tip = "
                f"{f(row['target_mid_gap_mm'])}/"
                f"{f(row['target_tip_gap_mm'])} mm"
            )


    finally:

        contact_replay.CONTACT_DAMPING = (
            old_damping
        )

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_k
        )


    fields = list(
        rows[0].keys()
    )


    with OUT.open(
        "w",
        newline="",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    print()
    print("=" * 170)

    print(
        "SUMMARY"
    )

    print("=" * 170)

    print(
        " K     RMSE     pose MCP/PIP/DIP          "
        "mid on/off   tip on/off   "
        "mid Fn med/max       mid pen     "
        "gaps mid/tip"
    )

    print("-" * 170)


    for r in rows:

        print(
            f"{r['contact_k']:4.0f}   "
            f"{r['rmse_deg']:6.3f}   "
            f"{r['mcp_deg']:6.2f}/"
            f"{r['pip_deg']:6.2f}/"
            f"{r['dip_deg']:6.2f}       "
            f"{r['mid_on']:2d}/{r['mid_off']:<2d}          "
            f"{r['tip_on']:2d}/{r['tip_off']:<2d}         "
            f"{f(r['mid_force_median_N']):>6s}/"
            f"{f(r['mid_force_max_N']):>6s} N   "
            f"{f(r['mid_pen_max_mm']):>6s} mm   "
            f"{f(r['target_mid_gap_mm']):>6s}/"
            f"{f(r['target_tip_gap_mm']):>6s} mm"
        )


    print()
    print(
        f"CSV:\n{OUT}"
    )

    print("=" * 170)


if __name__ == "__main__":
    main()
