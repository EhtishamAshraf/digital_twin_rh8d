#!/usr/bin/env python3

from pathlib import Path
import csv
import math

import mujoco

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
    / "v35p1_contact_dimension"
)

OUTCSV = (
    OUTDIR
    / "v35p1_contact_dimension_summary.csv"
)


# ============================================================
# Frozen mechanics
# ============================================================

TENDON_K = 9000.0

CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0

SLIDING_MU = 0.05

# Already present in the fixture XML.
TORSIONAL_FRICTION = 0.005
ROLLING_FRICTION = 0.0001


CASES = [
    ("CONDIM3", 3),
    ("CONDIM4", 4),
    ("CONDIM6", 6),
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

        self.mid_active = 0.0
        self.tip_active = 0.0
        self.overlap_active = 0.0

        self.overlap_start = None
        self.overlap_episodes = []

        self.tip_start = None
        self.tip_episodes = []

        self.mid_start = None
        self.mid_episodes = []

        self.first_tip_time = None
        self.middle_after_first_tip = 0.0

        self.max_mid_pen = 0.0
        self.max_tip_pen = 0.0

        self.contact_dims = set()

        self.first_friction = None

        self.last_time = 0.0


    @staticmethod
    def maximum(values):

        if not values:
            return 0.0

        return float(
            max(values)
        )


    def inspect(
        self,
        model,
        data,
    ):

        dt = float(
            model.opt.timestep
        )

        t = float(
            data.time
        )

        self.last_time = t


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


            if name not in (
                MID,
                TIP,
            ):
                continue


            try:

                self.contact_dims.add(
                    int(c.dim)
                )

            except Exception:
                pass


            if self.first_friction is None:

                try:

                    f = list(
                        c.friction
                    )

                    self.first_friction = [
                        float(x)
                        for x in f
                    ]

                except Exception:

                    self.first_friction = []


            pen = max(
                0.0,
                -float(
                    c.dist
                ),
            )


            if name == MID:

                mid = True

                self.max_mid_pen = max(
                    self.max_mid_pen,
                    pen,
                )


            elif name == TIP:

                tip = True

                self.max_tip_pen = max(
                    self.max_tip_pen,
                    pen,
                )


        # ====================================================
        # Middle transitions
        # ====================================================

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
                    t - self.mid_start
                )

                self.mid_start = None


        # ====================================================
        # Tip transitions
        # ====================================================

        if (
            tip
            and
            not self.prev_tip
        ):

            self.tip_on += 1
            self.tip_start = t

            if self.first_tip_time is None:

                self.first_tip_time = t


        if (
            self.prev_tip
            and
            not tip
        ):

            self.tip_off += 1

            if self.tip_start is not None:

                self.tip_episodes.append(
                    t - self.tip_start
                )

                self.tip_start = None


        # ====================================================
        # Simultaneous middle + tip
        # ====================================================

        overlap = (
            mid
            and
            tip
        )


        if overlap:

            self.overlap_active += dt


        previous_overlap = (
            self.prev_mid
            and
            self.prev_tip
        )


        if (
            overlap
            and
            not previous_overlap
        ):

            self.overlap_start = t


        if (
            previous_overlap
            and
            not overlap
        ):

            if self.overlap_start is not None:

                self.overlap_episodes.append(
                    t
                    -
                    self.overlap_start
                )

                self.overlap_start = None


        # ====================================================
        # Active durations
        # ====================================================

        if mid:
            self.mid_active += dt

        if tip:
            self.tip_active += dt


        if (
            self.first_tip_time is not None
            and
            mid
        ):

            self.middle_after_first_tip += dt


        self.prev_mid = mid
        self.prev_tip = tip


    def finalize(self):

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
            self.prev_tip
            and
            self.overlap_start is not None
        ):

            self.overlap_episodes.append(
                self.last_time
                -
                self.overlap_start
            )

            self.overlap_start = None


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
                self.mid_active,

            "tip_active_s":
                self.tip_active,

            "overlap_active_s":
                self.overlap_active,

            "longest_mid_s":
                self.maximum(
                    self.mid_episodes
                ),

            "longest_tip_s":
                self.maximum(
                    self.tip_episodes
                ),

            "longest_overlap_s":
                self.maximum(
                    self.overlap_episodes
                ),

            "middle_after_tip_s":
                self.middle_after_first_tip,

            "max_mid_pen_mm":
                self.max_mid_pen
                * 1000.0,

            "max_tip_pen_mm":
                self.max_tip_pen
                * 1000.0,

            "effective_dims":
                ",".join(
                    str(x)
                    for x in sorted(
                        self.contact_dims
                    )
                ),

            "first_friction":
                (
                    " ".join(
                        f"{x:.6g}"
                        for x in self.first_friction
                    )
                    if self.first_friction is not None
                    else ""
                ),
        }


def run_case(
    label,
    condim,
    contact_replay,
):

    audit = Audit()

    patched = False

    real_step = mujoco.mj_step


    def wrapped_step(
        model,
        data,
        *args,
        **kwargs,
    ):

        nonlocal patched


        # ----------------------------------------------------
        # Patch the freshly loaded model once, immediately
        # before its first dynamic step.
        # ----------------------------------------------------

        if not patched:

            gid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                FIXTURE,
            )

            if gid < 0:

                raise RuntimeError(
                    f"Missing geom: {FIXTURE}"
                )


            model.geom_condim[
                gid
            ] = int(
                condim
            )


            model.geom_friction[
                gid,
                0
            ] = SLIDING_MU

            model.geom_friction[
                gid,
                1
            ] = TORSIONAL_FRICTION

            model.geom_friction[
                gid,
                2
            ] = ROLLING_FRICTION


            patched = True


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

        b2.v35.CONTACT_MU = (
            SLIDING_MU
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

        "case":
            label,

        "requested_condim":
            condim,

        "effective_dims":
            s[
                "effective_dims"
            ],

        "first_contact_friction":
            s[
                "first_friction"
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

        "overlap_active_s":
            s[
                "overlap_active_s"
            ],

        "longest_mid_s":
            s[
                "longest_mid_s"
            ],

        "longest_tip_s":
            s[
                "longest_tip_s"
            ],

        "longest_overlap_s":
            s[
                "longest_overlap_s"
            ],

        "middle_after_tip_s":
            s[
                "middle_after_tip_s"
            ],

        "max_mid_pen_mm":
            s[
                "max_mid_pen_mm"
            ],

        "max_tip_pen_mm":
            s[
                "max_tip_pen_mm"
            ],

        "mid_gap_mm":
            result.get(
                "target_mid_gap_mm"
            ),

        "tip_gap_mm":
            result.get(
                "target_tip_gap_mm"
            ),
    }


def f(
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

    if not XML.exists():

        raise RuntimeError(
            f"Missing XML:\n{XML}"
        )


    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print("=" * 175)
    print("V35P1 — CONTACT DIMENSION / ROLLING-MOMENT TEST")
    print("=" * 175)

    print(
        f"XML:\n{XML}"
    )

    print()
    print(
        "Frozen:"
    )

    print(
        f"  tendon K        = "
        f"{TENDON_K:.0f} N/m"
    )

    print(
        f"  contact K       = "
        f"{CONTACT_K:.0f}"
    )

    print(
        f"  contact damping = "
        f"{CONTACT_DAMPING:.1f}"
    )

    print(
        f"  sliding mu      = "
        f"{SLIDING_MU:.3f}"
    )

    print(
        f"  torsional coeff = "
        f"{TORSIONAL_FRICTION}"
    )

    print(
        f"  rolling coeff   = "
        f"{ROLLING_FRICTION}"
    )

    print()
    print(
        "VARIABLE ONLY: fixture condim = 3 / 4 / 6"
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

        for i, (
            label,
            condim,
        ) in enumerate(
            CASES,
            1,
        ):

            print()
            print(
                f"[{i}/{len(CASES)}] "
                f"{label}"
            )


            r = run_case(
                label,
                condim,
                contact_replay,
            )

            rows.append(
                r
            )


            print(
                "  effective contact dims = "
                f"{r['effective_dims']}"
            )

            print(
                "  first contact friction = "
                f"{r['first_contact_friction']}"
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
                "  middle on/off = "
                f"{r['mid_on']}/"
                f"{r['mid_off']}   "
                "tip on/off = "
                f"{r['tip_on']}/"
                f"{r['tip_off']}"
            )

            print(
                "  active middle/tip/overlap = "
                f"{f(r['mid_active_s'])}/"
                f"{f(r['tip_active_s'])}/"
                f"{f(r['overlap_active_s'])} s"
            )

            print(
                "  longest middle/tip/overlap = "
                f"{f(r['longest_mid_s'])}/"
                f"{f(r['longest_tip_s'])}/"
                f"{f(r['longest_overlap_s'])} s"
            )

            print(
                "  middle after first tip = "
                f"{f(r['middle_after_tip_s'])} s"
            )

            print(
                "  max penetration mid/tip = "
                f"{f(r['max_mid_pen_mm'])}/"
                f"{f(r['max_tip_pen_mm'])} mm"
            )

            print(
                "  target gaps mid/tip = "
                f"{f(r['mid_gap_mm'])}/"
                f"{f(r['tip_gap_mm'])} mm"
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
    print("=" * 195)
    print("SUMMARY")
    print("=" * 195)

    print(
        " case     dim_eff   RMSE     final MCP/PIP/DIP       "
        "mid on/off   tip on/off   overlap total/longest   "
        "middle-after-tip   penetration M/T    gaps M/T"
    )

    print("-" * 195)


    for r in rows:

        print(
            f"{r['case']:8s} "
            f"{r['effective_dims']:>7s}   "
            f"{r['rmse_deg']:6.3f}   "
            f"{r['mcp_deg']:6.2f}/"
            f"{r['pip_deg']:6.2f}/"
            f"{r['dip_deg']:6.2f}       "
            f"{r['mid_on']:2d}/{r['mid_off']:<2d}          "
            f"{r['tip_on']:2d}/{r['tip_off']:<2d}         "
            f"{f(r['overlap_active_s']):>6s}/"
            f"{f(r['longest_overlap_s']):>6s}s       "
            f"{f(r['middle_after_tip_s']):>7s}s          "
            f"{f(r['max_mid_pen_mm']):>6s}/"
            f"{f(r['max_tip_pen_mm']):>6s} mm   "
            f"{f(r['mid_gap_mm']):>6s}/"
            f"{f(r['tip_gap_mm']):>6s}"
        )


    print()
    print(
        f"CSV:\n{OUTCSV}"
    )

    print(
        f"\nExperiment folder:\n{OUTDIR}"
    )

    print("=" * 195)


if __name__ == "__main__":
    main()
