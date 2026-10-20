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
    / "v35o3_friction_k9000"
)

OUTCSV = (
    OUTDIR
    / "v35o3_friction_summary.csv"
)


TENDON_K = 9000.0

CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

MU_VALUES = [
    0.05,
    0.15,
    0.30,
    0.60,
    1.00,
    1.20,
]


FIXTURE = "index_contact_fixture"
MID = "collision_idx_mid_mesh"
TIP = "collision_idx_tip"


class Audit:

    def __init__(self):

        self.prev_mid = False
        self.prev_tip = False

        self.mid_on = 0
        self.mid_off = 0

        self.tip_on = 0
        self.tip_off = 0

        self.tip_start = None
        self.tip_episodes = []

        self.mid_start = None
        self.mid_episodes = []

        self.tip_active_time = 0.0
        self.mid_active_time = 0.0

        self.tip_mu = []
        self.tip_fn = []
        self.tip_ft = []
        self.tip_util = []

        self.last_time = 0.0


    def inspect(
        self,
        model,
        data,
    ):

        self.last_time = float(
            data.time
        )

        fixture = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            FIXTURE,
        )

        if fixture < 0:
            return


        mid = False
        tip = False


        for ci in range(
            data.ncon
        ):

            c = data.contact[
                ci
            ]

            g1 = int(
                c.geom1
            )

            g2 = int(
                c.geom2
            )

            if fixture not in (
                g1,
                g2,
            ):
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


            if name == MID:

                mid = True


            if name == TIP:

                tip = True


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
                    float(
                        force6[0]
                    ),
                )

                ft = math.hypot(
                    float(
                        force6[1]
                    ),
                    float(
                        force6[2]
                    ),
                )


                try:

                    mu_eff = float(
                        c.friction[0]
                    )

                except Exception:

                    mu_eff = math.nan


                if math.isfinite(
                    mu_eff
                ):

                    self.tip_mu.append(
                        mu_eff
                    )


                if fn > 1.0e-8:

                    self.tip_fn.append(
                        fn
                    )

                    self.tip_ft.append(
                        ft
                    )


                    if (
                        math.isfinite(
                            mu_eff
                        )
                        and
                        mu_eff > 1.0e-9
                    ):

                        self.tip_util.append(
                            ft
                            /
                            (
                                mu_eff
                                * fn
                            )
                        )


        # ====================================================
        # Transition accounting
        # ====================================================

        t = float(
            data.time
        )


        if (
            mid
            and
            not self.prev_mid
        ):

            self.mid_on += 1

            self.mid_start = t


        if (
            self.prev_mid
            and
            not mid
        ):

            self.mid_off += 1

            if self.mid_start is not None:

                self.mid_episodes.append(
                    t
                    -
                    self.mid_start
                )

                self.mid_start = None


        if (
            tip
            and
            not self.prev_tip
        ):

            self.tip_on += 1

            self.tip_start = t


        if (
            self.prev_tip
            and
            not tip
        ):

            self.tip_off += 1

            if self.tip_start is not None:

                self.tip_episodes.append(
                    t
                    -
                    self.tip_start
                )

                self.tip_start = None


        if tip:

            self.tip_active_time += float(
                model.opt.timestep
            )


        if mid:

            self.mid_active_time += float(
                model.opt.timestep
            )


        self.prev_mid = mid
        self.prev_tip = tip


    def finalize(self):

        if (
            self.prev_tip
            and
            self.tip_start is not None
        ):

            self.tip_episodes.append(
                self.last_time
                -
                self.tip_start
            )

            self.tip_start = None


        if (
            self.prev_mid
            and
            self.mid_start is not None
        ):

            self.mid_episodes.append(
                self.last_time
                -
                self.mid_start
            )

            self.mid_start = None


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
    def maximum(values):

        if not values:
            return math.nan

        return float(
            np.max(
                values
            )
        )


    def summary(self):

        self.finalize()

        return {

            "mid_on":
                self.mid_on,

            "mid_off":
                self.mid_off,

            "tip_on":
                self.tip_on,

            "tip_off":
                self.tip_off,

            "mid_active_s":
                self.mid_active_time,

            "tip_active_s":
                self.tip_active_time,

            "longest_mid_s":
                self.maximum(
                    self.mid_episodes
                ),

            "longest_tip_s":
                self.maximum(
                    self.tip_episodes
                ),

            "effective_mu_median":
                self.median(
                    self.tip_mu
                ),

            "tip_fn_median":
                self.median(
                    self.tip_fn
                ),

            "tip_fn_max":
                self.maximum(
                    self.tip_fn
                ),

            "tip_ft_median":
                self.median(
                    self.tip_ft
                ),

            "friction_util_median":
                self.median(
                    self.tip_util
                ),

            "friction_util_max":
                self.maximum(
                    self.tip_util
                ),
        }


def run_one(
    requested_mu,
    contact_replay,
):

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


    mujoco.mj_step = wrapped_step


    try:

        b2.v35.TENDON_K = (
            TENDON_K
        )

        b2.v30.v24.base.TENDON_STIFFNESS = (
            TENDON_K
        )

        b2.v35.CONTACT_K = (
            CONTACT_K
        )

        b2.v35.CONTACT_MU = float(
            requested_mu
        )


        result = e2b.run_one(
            CONTACT_DAMPING,
            XML,
            BAG,
            contact_replay,
        )


    finally:

        mujoco.mj_step = real_step


    s = audit.summary()

    q = result[
        "pose"
    ]


    return {

        "requested_mu":
            requested_mu,

        "effective_mu":
            s[
                "effective_mu_median"
            ],

        "mcp_deg":
            q[0],

        "pip_deg":
            q[1],

        "dip_deg":
            q[2],

        "rmse_deg":
            result[
                "rmse"
            ],

        "mid_on":
            s[
                "mid_on"
            ],

        "mid_off":
            s[
                "mid_off"
            ],

        "tip_on":
            s[
                "tip_on"
            ],

        "tip_off":
            s[
                "tip_off"
            ],

        "mid_active_s":
            s[
                "mid_active_s"
            ],

        "tip_active_s":
            s[
                "tip_active_s"
            ],

        "longest_mid_s":
            s[
                "longest_mid_s"
            ],

        "longest_tip_s":
            s[
                "longest_tip_s"
            ],

        "tip_fn_median":
            s[
                "tip_fn_median"
            ],

        "tip_fn_max":
            s[
                "tip_fn_max"
            ],

        "tip_ft_median":
            s[
                "tip_ft_median"
            ],

        "friction_util_median":
            s[
                "friction_util_median"
            ],

        "friction_util_max":
            s[
                "friction_util_max"
            ],

        "mid_gap_mm":
            result.get(
                "target_mid_gap_mm"
            ),

        "tip_gap_mm":
            result.get(
                "target_tip_gap_mm"
            ),

        "overlap_s":
            result.get(
                "overlap_longest"
            ),
    }


def fmt(
    x,
    n=3,
):

    try:

        x = float(
            x
        )

        if math.isnan(
            x
        ):
            return "nan"

        return (
            f"{x:.{n}f}"
        )

    except Exception:

        return str(
            x
        )


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print("=" * 175)
    print("V35O3 — CONTACT FRICTION TEST AT Kflex=9000")
    print("=" * 175)

    print(
        f"XML:\n{XML}"
    )

    print()

    print(
        f"Tendon K        = "
        f"{TENDON_K:.0f} N/m"
    )

    print(
        f"Contact K       = "
        f"{CONTACT_K:.0f}"
    )

    print(
        f"Contact damping = "
        f"{CONTACT_DAMPING:.1f}"
    )


    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
    )


    old_tendon_k = float(
        b2.v35.TENDON_K
    )

    old_base_tendon_k = float(
        b2.v30.v24.base.TENDON_STIFFNESS
    )

    old_contact_k = float(
        b2.v35.CONTACT_K
    )

    old_mu = float(
        b2.v35.CONTACT_MU
    )

    old_cd = float(
        contact_replay.CONTACT_DAMPING
    )


    contact_replay.CONTACT_DAMPING = (
        CONTACT_DAMPING
    )


    rows = []


    try:

        for i, mu in enumerate(
            MU_VALUES,
            1,
        ):

            print()
            print(
                f"[{i}/{len(MU_VALUES)}] "
                f"requested mu = {mu:.2f}"
            )


            r = run_one(
                mu,
                contact_replay,
            )

            rows.append(
                r
            )


            print(
                "  effective contact mu = "
                f"{fmt(r['effective_mu'])}"
            )

            print(
                "  pose = "
                f"{r['mcp_deg']:.2f}/"
                f"{r['pip_deg']:.2f}/"
                f"{r['dip_deg']:.2f} deg"
            )

            print(
                f"  RMSE = "
                f"{r['rmse_deg']:.3f} deg"
            )

            print(
                "  mid on/off = "
                f"{r['mid_on']}/"
                f"{r['mid_off']}   "
                "tip on/off = "
                f"{r['tip_on']}/"
                f"{r['tip_off']}"
            )

            print(
                "  tip total active = "
                f"{fmt(r['tip_active_s'])} s   "
                "longest episode = "
                f"{fmt(r['longest_tip_s'])} s"
            )

            print(
                "  friction utilization "
                "median/max = "
                f"{fmt(r['friction_util_median'])}/"
                f"{fmt(r['friction_util_max'])}"
            )

            print(
                "  gaps mid/tip = "
                f"{fmt(r['mid_gap_mm'])}/"
                f"{fmt(r['tip_gap_mm'])} mm"
            )


    finally:

        b2.v35.TENDON_K = (
            old_tendon_k
        )

        b2.v30.v24.base.TENDON_STIFFNESS = (
            old_base_tendon_k
        )

        b2.v35.CONTACT_K = (
            old_contact_k
        )

        b2.v35.CONTACT_MU = (
            old_mu
        )

        contact_replay.CONTACT_DAMPING = (
            old_cd
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
    print("=" * 190)
    print("SUMMARY")
    print("=" * 190)

    print(
        " mu_req  mu_eff   RMSE     final MCP/PIP/DIP       "
        "mid on/off   tip on/off   tip active   longest tip   "
        "fric util med/max      gaps mid/tip"
    )

    print("-" * 190)


    for r in rows:

        print(
            f"{r['requested_mu']:6.2f}  "
            f"{fmt(r['effective_mu']):>6s}  "
            f"{r['rmse_deg']:6.3f}   "
            f"{r['mcp_deg']:6.2f}/"
            f"{r['pip_deg']:6.2f}/"
            f"{r['dip_deg']:6.2f}        "
            f"{r['mid_on']:2d}/{r['mid_off']:<2d}          "
            f"{r['tip_on']:2d}/{r['tip_off']:<2d}         "
            f"{fmt(r['tip_active_s']):>7s}s     "
            f"{fmt(r['longest_tip_s']):>7s}s      "
            f"{fmt(r['friction_util_median']):>6s}/"
            f"{fmt(r['friction_util_max']):>6s}        "
            f"{fmt(r['mid_gap_mm']):>6s}/"
            f"{fmt(r['tip_gap_mm']):>6s}"
        )


    print()
    print(
        f"CSV:\n{OUTCSV}"
    )

    print(
        f"\nExperiment folder:\n{OUTDIR}"
    )

    print("=" * 190)


if __name__ == "__main__":
    main()
