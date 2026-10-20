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
    / "ur5e_rh8d_index_segmented_v36a4.xml"
)

BAG = Path(
    "/home/ehtisham/rh8d_system_id/"
    "index_contact_20260911_164310"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v36b1_guide_friction"
)

OUTCSV = (
    OUTDIR
    / "v36b1_guide_friction_summary.csv"
)


# ============================================================
# Frozen model parameters
# ============================================================

CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0
CONTACT_MU = 0.05


# Whole V35 tendon MUST remain mechanically disabled.
FULL_TENDON_K = 0.0


# ============================================================
# Only variables in this experiment:
#
# Coulomb guide-force thresholds [N]
# ============================================================

CASES = [

    ("F0_F0",       0.00, 0.00),

    ("MCP_0p10",    0.10, 0.00),
    ("MCP_0p30",    0.30, 0.00),

    ("PIP_0p10",    0.00, 0.10),
    ("PIP_0p30",    0.00, 0.30),

    ("BOTH_0p10",   0.10, 0.10),
    ("BOTH_0p30",   0.30, 0.30),
]


FIXTURE = "index_contact_fixture"

MID = "collision_idx_mid_mesh"

TIP = "collision_idx_tip"


J_MCP = (
    "Index_Proximal--palm_assembly:1"
)

J_PIP = (
    "Index_Middle--Index_Proximal"
)

J_DIP = (
    "Index_Distal--Index_Middle"
)

SLIP_MCP = (
    "idx_v36_slip_mcp"
)

SLIP_PIP = (
    "idx_v36_slip_pip"
)

EXT_JOINTS = [
    "idx_v36_ext1",
    "idx_v36_ext2",
    "idx_v36_ext3",
]


# ============================================================
# MuJoCo helpers
# ============================================================

def joint_id(
    model,
    name,
):

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


def qadr(
    model,
    name,
):

    j = joint_id(
        model,
        name,
    )

    return int(
        model.jnt_qposadr[
            j
        ]
    )


def dadr(
    model,
    name,
):

    j = joint_id(
        model,
        name,
    )

    return int(
        model.jnt_dofadr[
            j
        ]
    )


def geom_id(
    model,
    name,
):

    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        name,
    )


# ============================================================
# Dynamic diagnostic recorder
# ============================================================

class Audit:

    def __init__(
        self,
        fg_mcp,
        fg_pip,
    ):

        self.fg_mcp = float(
            fg_mcp
        )

        self.fg_pip = float(
            fg_pip
        )

        self.prev_mid = False
        self.prev_tip = False

        self.mid_on = 0
        self.mid_off = 0

        self.tip_on = 0
        self.tip_off = 0

        self.first_mid = None
        self.first_tip = None

        self.mid_active = 0.0
        self.tip_active = 0.0
        self.overlap_active = 0.0

        self.overlap_start = None
        self.overlap_episodes = []

        self.last_time = 0.0

        self.max_abs_dt12 = 0.0
        self.max_abs_dt23 = 0.0


    def tensions(
        self,
        model,
        data,
    ):

        out = []


        for name in EXT_JOINTS:

            ji = joint_id(
                model,
                name,
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


            e = float(
                data.qpos[
                    qa
                ]
            )

            edot = float(
                data.qvel[
                    da
                ]
            )


            K = float(
                model.jnt_stiffness[
                    ji
                ]
            )

            D = float(
                model.dof_damping[
                    da
                ]
            )


            # Signed elastic/viscous cable-force estimate.
            #
            # Positive = tension-producing extension.
            T = (
                K * e
                +
                D * edot
            )


            out.append(
                T
            )


        return out


    def snapshot(
        self,
        model,
        data,
    ):

        T1, T2, T3 = self.tensions(
            model,
            data,
        )


        s1_da = dadr(
            model,
            SLIP_MCP,
        )

        s2_da = dadr(
            model,
            SLIP_PIP,
        )


        s1_qa = qadr(
            model,
            SLIP_MCP,
        )

        s2_qa = qadr(
            model,
            SLIP_PIP,
        )


        snap = {

            "time":
                float(
                    data.time
                ),

            "mcp":
                math.degrees(
                    float(
                        data.qpos[
                            qadr(
                                model,
                                J_MCP,
                            )
                        ]
                    )
                ),

            "pip":
                math.degrees(
                    float(
                        data.qpos[
                            qadr(
                                model,
                                J_PIP,
                            )
                        ]
                    )
                ),

            "dip":
                math.degrees(
                    float(
                        data.qpos[
                            qadr(
                                model,
                                J_DIP,
                            )
                        ]
                    )
                ),

            "s1_mm":
                float(
                    data.qpos[
                        s1_qa
                    ]
                )
                * 1000.0,

            "s2_mm":
                float(
                    data.qpos[
                        s2_qa
                    ]
                )
                * 1000.0,

            "s1_vel_mm_s":
                float(
                    data.qvel[
                        s1_da
                    ]
                )
                * 1000.0,

            "s2_vel_mm_s":
                float(
                    data.qvel[
                        s2_da
                    ]
                )
                * 1000.0,

            "T1":
                T1,

            "T2":
                T2,

            "T3":
                T3,

            "dT12":
                T1
                -
                T2,

            "dT23":
                T2
                -
                T3,
        }


        self.max_abs_dt12 = max(
            self.max_abs_dt12,
            abs(
                snap["dT12"]
            ),
        )

        self.max_abs_dt23 = max(
            self.max_abs_dt23,
            abs(
                snap["dT23"]
            ),
        )


        return snap


    def inspect(
        self,
        model,
        data,
    ):

        dt = float(
            model.opt.timestep
        )

        self.last_time = float(
            data.time
        )


        fixture = geom_id(
            model,
            FIXTURE,
        )

        middle_gid = geom_id(
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
                int(
                    c.geom1
                ),
                int(
                    c.geom2
                ),
            }


            if fixture not in pair:
                continue


            if middle_gid in pair:
                mid = True


            if tip_gid in pair:
                tip = True


        # ----------------------------------------------------
        # First middle contact
        # ----------------------------------------------------

        if (
            mid
            and
            not self.prev_mid
        ):

            self.mid_on += 1

            if self.first_mid is None:

                self.first_mid = self.snapshot(
                    model,
                    data,
                )


        if (
            self.prev_mid
            and
            not mid
        ):

            self.mid_off += 1


        # ----------------------------------------------------
        # First tip contact
        # ----------------------------------------------------

        if (
            tip
            and
            not self.prev_tip
        ):

            self.tip_on += 1

            if self.first_tip is None:

                self.first_tip = self.snapshot(
                    model,
                    data,
                )


        if (
            self.prev_tip
            and
            not tip
        ):

            self.tip_off += 1


        # ----------------------------------------------------
        # Contact durations
        # ----------------------------------------------------

        if mid:

            self.mid_active += dt


        if tip:

            self.tip_active += dt


        overlap = (
            mid
            and
            tip
        )


        previous_overlap = (
            self.prev_mid
            and
            self.prev_tip
        )


        if overlap:

            self.overlap_active += dt


        if (
            overlap
            and
            not previous_overlap
        ):

            self.overlap_start = (
                float(
                    data.time
                )
            )


        if (
            previous_overlap
            and
            not overlap
            and
            self.overlap_start is not None
        ):

            self.overlap_episodes.append(
                float(
                    data.time
                )
                -
                self.overlap_start
            )

            self.overlap_start = None


        # Always update tension diagnostics.
        self.snapshot(
            model,
            data,
        )


        self.prev_mid = mid
        self.prev_tip = tip


    def finalize(self):

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


    def longest_overlap(self):

        self.finalize()

        if not self.overlap_episodes:
            return 0.0

        return float(
            max(
                self.overlap_episodes
            )
        )


# ============================================================
# Run one guide-friction hypothesis
# ============================================================

def run_case(
    label,
    fg_mcp,
    fg_pip,
    contact_replay,
):

    audit = Audit(
        fg_mcp,
        fg_pip,
    )


    real_step = mujoco.mj_step

    patched = False


    def wrapped_step(
        model,
        data,
        *args,
        **kwargs,
    ):

        nonlocal patched


        # ----------------------------------------------------
        # Apply guide friction exactly once to the freshly
        # loaded V36 model before dynamics begin.
        # ----------------------------------------------------

        if not patched:

            da1 = dadr(
                model,
                SLIP_MCP,
            )

            da2 = dadr(
                model,
                SLIP_PIP,
            )


            model.dof_frictionloss[
                da1
            ] = float(
                fg_mcp
            )

            model.dof_frictionloss[
                da2
            ] = float(
                fg_pip
            )


            patched = True


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


    mujoco.mj_step = wrapped_step


    old_tk = float(
        b2.v35.TENDON_K
    )

    old_btk = float(
        b2.v30.v24.base.TENDON_STIFFNESS
    )


    try:

        # Whole original V35 tendon remains non-mechanical.
        b2.v35.TENDON_K = (
            FULL_TENDON_K
        )

        b2.v30.v24.base.TENDON_STIFFNESS = (
            FULL_TENDON_K
        )


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


    q = result[
        "pose"
    ]


    row = {

        "case":
            label,

        "Fg_MCP_N":
            fg_mcp,

        "Fg_PIP_N":
            fg_pip,

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
            audit.mid_on,

        "mid_off":
            audit.mid_off,

        "tip_on":
            audit.tip_on,

        "tip_off":
            audit.tip_off,

        "overlap_total_s":
            audit.overlap_active,

        "overlap_longest_s":
            audit.longest_overlap(),

        "first_mid_time":
            (
                mid["time"]
                if mid is not None
                else math.nan
            ),

        "first_tip_time":
            (
                tip["time"]
                if tip is not None
                else math.nan
            ),

        "middle_to_tip_delay_s":
            delay,

        "middle_to_tip_dPIP_deg":
            dpip,

        "middle_to_tip_dDIP_deg":
            ddip,

        "mid_MCP_deg":
            (
                mid["mcp"]
                if mid is not None
                else math.nan
            ),

        "mid_PIP_deg":
            (
                mid["pip"]
                if mid is not None
                else math.nan
            ),

        "mid_DIP_deg":
            (
                mid["dip"]
                if mid is not None
                else math.nan
            ),

        "mid_dT12_N":
            (
                mid["dT12"]
                if mid is not None
                else math.nan
            ),

        "mid_dT23_N":
            (
                mid["dT23"]
                if mid is not None
                else math.nan
            ),

        "mid_s1_vel_mm_s":
            (
                mid["s1_vel_mm_s"]
                if mid is not None
                else math.nan
            ),

        "mid_s2_vel_mm_s":
            (
                mid["s2_vel_mm_s"]
                if mid is not None
                else math.nan
            ),

        "tip_dT12_N":
            (
                tip["dT12"]
                if tip is not None
                else math.nan
            ),

        "tip_dT23_N":
            (
                tip["dT23"]
                if tip is not None
                else math.nan
            ),

        "tip_s1_vel_mm_s":
            (
                tip["s1_vel_mm_s"]
                if tip is not None
                else math.nan
            ),

        "tip_s2_vel_mm_s":
            (
                tip["s2_vel_mm_s"]
                if tip is not None
                else math.nan
            ),

        "max_abs_dT12_N":
            audit.max_abs_dt12,

        "max_abs_dT23_N":
            audit.max_abs_dt23,

        "target_mid_gap_mm":
            result.get(
                "target_mid_gap_mm"
            ),

        "target_tip_gap_mm":
            result.get(
                "target_tip_gap_mm"
            ),
    }


    return row


def ff(
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
            f"Missing V36A4 XML:\n{XML}"
        )


    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print("=" * 185)
    print("V36B1 — INTERNAL GUIDE-HOLE FRICTION CAUSAL TEST")
    print("=" * 185)

    print(
        f"MODEL:\n{XML}"
    )

    print()
    print(
        "Frozen:"
    )

    print(
        "  validated V36A4 segmented cable"
    )

    print(
        "  full original V35 tendon stiffness = 0"
    )

    print(
        f"  contact K       = {CONTACT_K}"
    )

    print(
        f"  contact damping = {CONTACT_DAMPING}"
    )

    print(
        f"  contact mu      = {CONTACT_MU}"
    )

    print()
    print(
        "VARIABLE ONLY:"
    )

    print(
        "  MCP guide frictionloss [N]"
    )

    print(
        "  PIP guide frictionloss [N]"
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
            fmcp,
            fpip,
        ) in enumerate(
            CASES,
            1,
        ):

            print()
            print("-" * 185)

            print(
                f"[{i}/{len(CASES)}] "
                f"{label}"
            )

            print(
                f"Fg MCP/PIP = "
                f"{fmcp:.3f} / "
                f"{fpip:.3f} N"
            )


            r = run_case(
                label,
                fmcp,
                fpip,
                contact_replay,
            )


            rows.append(
                r
            )


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
                "episodes M/T = "
                f"{r['mid_on']}/"
                f"{r['tip_on']}"
            )

            print(
                "overlap total/longest = "
                f"{r['overlap_total_s']:.3f} / "
                f"{r['overlap_longest_s']:.3f} s"
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
                "at first middle:"
            )

            print(
                "  q = "
                f"{ff(r['mid_MCP_deg'])} / "
                f"{ff(r['mid_PIP_deg'])} / "
                f"{ff(r['mid_DIP_deg'])}"
            )

            print(
                "  dT12/dT23 = "
                f"{ff(r['mid_dT12_N'], 4)} / "
                f"{ff(r['mid_dT23_N'], 4)} N"
            )

            print(
                "  slip velocities s1/s2 = "
                f"{ff(r['mid_s1_vel_mm_s'], 4)} / "
                f"{ff(r['mid_s2_vel_mm_s'], 4)} mm/s"
            )

            print(
                "at first tip:"
            )

            print(
                "  dT12/dT23 = "
                f"{ff(r['tip_dT12_N'], 4)} / "
                f"{ff(r['tip_dT23_N'], 4)} N"
            )

            print(
                "  slip velocities s1/s2 = "
                f"{ff(r['tip_s1_vel_mm_s'], 4)} / "
                f"{ff(r['tip_s2_vel_mm_s'], 4)} mm/s"
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
    print("=" * 210)
    print("V36B1 SUMMARY")
    print("=" * 210)

    print(
        "case         Fmcp Fpip   RMSE     MCP/PIP/DIP             "
        "M/T     overlap   dPIP(M->T)  dDIP(M->T)   "
        "dT12@M dT23@M   dT12@T dT23@T      gaps M/T"
    )

    print("-" * 210)


    for r in rows:

        print(
            f"{r['case']:12s} "
            f"{r['Fg_MCP_N']:4.2f} "
            f"{r['Fg_PIP_N']:4.2f}   "
            f"{r['rmse_deg']:6.3f}   "
            f"{r['mcp_deg']:6.2f}/"
            f"{r['pip_deg']:6.2f}/"
            f"{r['dip_deg']:6.2f}       "
            f"{r['mid_on']:1d}/"
            f"{r['tip_on']:1d}      "
            f"{r['overlap_longest_s']:6.3f}     "
            f"{ff(r['middle_to_tip_dPIP_deg']):>7s}      "
            f"{ff(r['middle_to_tip_dDIP_deg']):>7s}      "
            f"{ff(r['mid_dT12_N'], 3):>6s} "
            f"{ff(r['mid_dT23_N'], 3):>6s}    "
            f"{ff(r['tip_dT12_N'], 3):>6s} "
            f"{ff(r['tip_dT23_N'], 3):>6s}     "
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

    print("=" * 210)


if __name__ == "__main__":
    main()
