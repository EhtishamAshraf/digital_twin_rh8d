#!/usr/bin/env python3

from pathlib import Path
import csv
import math
import shutil

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

TOOLS = (
    WS
    / "rh8d_digital_twin_tools"
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
    TOOLS
    / "results"
    / "v35o1_flexor_stiffness"
)

OUTCSV = (
    OUTDIR
    / "v35o1_flexor_stiffness_summary.csv"
)


K_VALUES = [
    2100.0,
    3000.0,
    4500.0,
    6500.0,
    9000.0,
]


CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0
CONTACT_MU = 0.05

FIXTURE = "index_contact_fixture"
TIP = "collision_idx_tip"
MID = "collision_idx_mid_mesh"

TENDON = "idx_flex_spatial"

JOINTS = [
    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",
]


def oid(model, obj, name):

    i = mujoco.mj_name2id(
        model,
        obj,
        name,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing {name}"
        )

    return i


def qadr(model, joint):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint,
    )

    return int(
        model.jnt_qposadr[jid]
    )


def dadr(model, joint):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint,
    )

    return int(
        model.jnt_dofadr[jid]
    )


class EventAudit:

    def __init__(self):

        self.prev_mid = False
        self.prev_tip = False

        self.mid_on = 0
        self.mid_off = 0

        self.tip_on = 0
        self.tip_off = 0

        self.first_tip = None

        self.max_actuator_force = 0.0


    def inspect(self, model, data):

        fixture = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            FIXTURE,
        )

        if fixture < 0:
            return


        mid = False
        tip = False


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

            if name == MID:
                mid = True

            if name == TIP:
                tip = True


        if mid and not self.prev_mid:
            self.mid_on += 1

        if self.prev_mid and not mid:
            self.mid_off += 1

        if tip and not self.prev_tip:

            self.tip_on += 1

            if self.first_tip is None:

                tid = oid(
                    model,
                    mujoco.mjtObj.mjOBJ_TENDON,
                    TENDON,
                )

                dip_qa = qadr(
                    model,
                    JOINTS[2],
                )

                dip_da = dadr(
                    model,
                    JOINTS[2],
                )


                tendon_k = float(
                    model.tendon_stiffness[
                        tid
                    ]
                )

                L = float(
                    data.ten_length[
                        tid
                    ]
                )

                L0 = float(
                    model.tendon_lengthspring[
                        tid,
                        0
                    ]
                )

                extension = max(
                    0.0,
                    L - L0,
                )

                spring_force = (
                    tendon_k
                    * extension
                )


                self.first_tip = {
                    "time_s":
                        float(data.time),

                    "dip_deg":
                        math.degrees(
                            float(
                                data.qpos[
                                    dip_qa
                                ]
                            )
                        ),

                    "dip_qvel":
                        float(
                            data.qvel[
                                dip_da
                            ]
                        ),

                    "tendon_length_mm":
                        L * 1000.0,

                    "tendon_extension_mm":
                        extension * 1000.0,

                    "tendon_spring_force_N":
                        spring_force,

                    "dip_qfrc_passive":
                        float(
                            data.qfrc_passive[
                                dip_da
                            ]
                        ),

                    "dip_qfrc_constraint":
                        float(
                            data.qfrc_constraint[
                                dip_da
                            ]
                        ),
                }


        if self.prev_tip and not tip:
            self.tip_off += 1


        self.prev_mid = mid
        self.prev_tip = tip


        if data.actuator_force.size:

            self.max_actuator_force = max(
                self.max_actuator_force,
                float(
                    np.max(
                        np.abs(
                            data.actuator_force
                        )
                    )
                ),
            )


def run_one(
    tendon_k,
    contact_replay,
):

    audit = EventAudit()

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

        audit.inspect(
            model,
            data,
        )

        return out


    mujoco.mj_step = audited_step


    old_v35_k = float(
        b2.v35.TENDON_K
    )

    old_base_k = float(
        b2.v30.v24.base.TENDON_STIFFNESS
    )


    try:

        # ------------------------------------------------------
        # Important:
        # run_candidate() propagates v35.TENDON_K into the
        # lower replay module, so this really changes the
        # spatial flexor stiffness at runtime.
        # ------------------------------------------------------

        b2.v35.TENDON_K = float(
            tendon_k
        )

        b2.v30.v24.base.TENDON_STIFFNESS = float(
            tendon_k
        )


        result = e2b.run_one(
            CONTACT_DAMPING,
            XML,
            BAG,
            contact_replay,
        )


    finally:

        mujoco.mj_step = real_step

        b2.v35.TENDON_K = old_v35_k

        b2.v30.v24.base.TENDON_STIFFNESS = (
            old_base_k
        )


    q = result["pose"]

    ft = audit.first_tip or {}


    return {

        "tendon_k_N_m":
            tendon_k,

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

        "first_tip_time_s":
            ft.get(
                "time_s",
                math.nan,
            ),

        "first_tip_dip_deg":
            ft.get(
                "dip_deg",
                math.nan,
            ),

        "first_tip_dip_qvel":
            ft.get(
                "dip_qvel",
                math.nan,
            ),

        "first_tip_tendon_length_mm":
            ft.get(
                "tendon_length_mm",
                math.nan,
            ),

        "first_tip_extension_mm":
            ft.get(
                "tendon_extension_mm",
                math.nan,
            ),

        "first_tip_tendon_force_N":
            ft.get(
                "tendon_spring_force_N",
                math.nan,
            ),

        "first_tip_dip_qfrc_passive":
            ft.get(
                "dip_qfrc_passive",
                math.nan,
            ),

        "first_tip_dip_qfrc_constraint":
            ft.get(
                "dip_qfrc_constraint",
                math.nan,
            ),

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

        "max_actuator_force":
            audit.max_actuator_force,
    }


def fmt(x, nd=3):

    try:

        if math.isnan(
            float(x)
        ):
            return "nan"

    except Exception:
        return str(x)

    return f"{float(x):.{nd}f}"


def main():

    if not XML.exists():
        raise RuntimeError(
            f"Missing XML:\n{XML}"
        )

    if not BAG.exists():
        raise RuntimeError(
            f"Missing bag:\n{BAG}"
        )


    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Keep the exact source model with the experiment output.
    shutil.copy2(
        XML,
        OUTDIR / XML.name,
    )


    print()
    print("=" * 160)
    print("V35O1 — FLEXOR AXIAL STIFFNESS / FIRST-TIP TORQUE TEST")
    print("=" * 160)

    print(
        f"Source XML:\n{XML}"
    )

    print(
        f"\nResults folder:\n{OUTDIR}"
    )

    print()
    print(
        "Frozen:"
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

    print(
        "  all joint passive parameters unchanged"
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

    old_contact_k = float(
        b2.v35.CONTACT_K
    )


    contact_replay.CONTACT_DAMPING = (
        CONTACT_DAMPING
    )

    b2.v35.CONTACT_MU = CONTACT_MU
    b2.v35.CONTACT_K = CONTACT_K


    rows = []


    try:

        for i, k in enumerate(
            K_VALUES,
            1,
        ):

            print()
            print(
                f"[{i}/{len(K_VALUES)}] "
                f"TENDON_K = {k:.0f} N/m"
            )


            r = run_one(
                k,
                contact_replay,
            )

            rows.append(
                r
            )


            print(
                "  final pose = "
                f"{r['mcp_deg']:.2f}/"
                f"{r['pip_deg']:.2f}/"
                f"{r['dip_deg']:.2f} deg"
            )

            print(
                f"  RMSE = "
                f"{r['rmse_deg']:.3f} deg"
            )

            print(
                "  events mid on/off = "
                f"{r['mid_on']}/{r['mid_off']}   "
                "tip on/off = "
                f"{r['tip_on']}/{r['tip_off']}"
            )

            print(
                "  first tip: "
                f"DIP={fmt(r['first_tip_dip_deg'])} deg   "
                f"qvel={fmt(r['first_tip_dip_qvel'],4)} rad/s"
            )

            print(
                "  first tip: "
                f"extension={fmt(r['first_tip_extension_mm'])} mm   "
                f"spring F={fmt(r['first_tip_tendon_force_N'])} N"
            )

            print(
                "  first tip: "
                f"DIP passive="
                f"{fmt(r['first_tip_dip_qfrc_passive'],6)} Nm   "
                f"constraint="
                f"{fmt(r['first_tip_dip_qfrc_constraint'],6)} Nm"
            )

            print(
                "  final gaps mid/tip = "
                f"{fmt(r['target_mid_gap_mm'])}/"
                f"{fmt(r['target_tip_gap_mm'])} mm"
            )


    finally:

        contact_replay.CONTACT_DAMPING = old_cd
        b2.v35.CONTACT_MU = old_mu
        b2.v35.CONTACT_K = old_contact_k


    with OUTCSV.open(
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
        writer.writerows(
            rows
        )


    print()
    print("=" * 185)
    print("SUMMARY")
    print("=" * 185)

    print(
        " Kflex   RMSE     final MCP/PIP/DIP      "
        "tip on/off    first-tip DIP   first-tip qvel   "
        "first-tip F     DIP passive       gaps mid/tip"
    )

    print("-" * 185)


    for r in rows:

        print(
            f"{r['tendon_k_N_m']:6.0f}  "
            f"{r['rmse_deg']:6.3f}   "
            f"{r['mcp_deg']:6.2f}/"
            f"{r['pip_deg']:6.2f}/"
            f"{r['dip_deg']:6.2f}       "
            f"{r['tip_on']:2d}/{r['tip_off']:<2d}          "
            f"{fmt(r['first_tip_dip_deg']):>7s}°       "
            f"{fmt(r['first_tip_dip_qvel'],4):>8s}       "
            f"{fmt(r['first_tip_tendon_force_N']):>7s} N   "
            f"{fmt(r['first_tip_dip_qfrc_passive'],6):>10s}   "
            f"{fmt(r['target_mid_gap_mm']):>6s}/"
            f"{fmt(r['target_tip_gap_mm']):>6s}"
        )


    print()
    print(
        f"All output is in:\n{OUTDIR}"
    )

    print("=" * 185)


if __name__ == "__main__":
    main()
