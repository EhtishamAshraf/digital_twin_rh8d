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
    / "v35n1_contact_events.csv"
)


FIXTURE = "index_contact_fixture"

INDEX_GEOMS = {
    "collision_idx_prox",
    "collision_idx_mid",
    "collision_idx_mid_mesh",
    "collision_idx_tip",
}

JOINTS = [
    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",
]

TENDON = "idx_flex_spatial"
ACTUATOR = "pos_index_tendon"
SPOOL = "index_motor_spool"


def name_of_geom(model, gid):

    if gid < 0:
        return "NONE"

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        gid,
    )

    if name is None:
        return f"geom_{gid}"

    return name


def id_of(model, obj, name):

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


def qpos_addr(model, joint):

    jid = id_of(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint,
    )

    return int(
        model.jnt_qposadr[jid]
    )


def dof_addr(model, joint):

    jid = id_of(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint,
    )

    return int(
        model.jnt_dofadr[jid]
    )


class Audit:

    def __init__(self):

        self.rows = []

        self.previous_active = set()

        self.step_count = 0


    def inspect(
        self,
        model,
        data,
    ):

        self.step_count += 1

        fixture_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            FIXTURE,
        )

        if fixture_id < 0:
            return


        active = set()
        contacts = []


        for ci in range(
            data.ncon
        ):

            c = data.contact[ci]

            g1 = int(
                c.geom1
            )

            g2 = int(
                c.geom2
            )


            if fixture_id not in (
                g1,
                g2,
            ):
                continue


            other = (
                g2
                if g1 == fixture_id
                else g1
            )

            other_name = name_of_geom(
                model,
                other,
            )


            if other_name not in INDEX_GEOMS:
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


            key = other_name

            active.add(
                key
            )

            contacts.append(
                (
                    other_name,
                    float(c.dist),
                    float(force6[0]),
                    np.array(
                        c.pos,
                        dtype=float,
                    ),
                )
            )


        if active == self.previous_active:
            return


        appeared = (
            active
            -
            self.previous_active
        )

        disappeared = (
            self.previous_active
            -
            active
        )


        q_deg = []

        qvel = []


        for joint in JOINTS:

            qa = qpos_addr(
                model,
                joint,
            )

            da = dof_addr(
                model,
                joint,
            )

            q_deg.append(
                math.degrees(
                    float(
                        data.qpos[qa]
                    )
                )
            )

            qvel.append(
                float(
                    data.qvel[da]
                )
            )


        spool_qa = qpos_addr(
            model,
            SPOOL,
        )

        spool_da = dof_addr(
            model,
            SPOOL,
        )


        tid = id_of(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            TENDON,
        )

        aid = id_of(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            ACTUATOR,
        )


        contact_summary = []

        for (
            geom,
            dist,
            fn,
            pos,
        ) in contacts:

            contact_summary.append(
                f"{geom}:"
                f"d={dist*1000:+.4f}mm,"
                f"Fn={fn:.4f}N"
            )


        event = []

        if appeared:

            event.append(
                "ON:"
                +
                ",".join(
                    sorted(
                        appeared
                    )
                )
            )

        if disappeared:

            event.append(
                "OFF:"
                +
                ",".join(
                    sorted(
                        disappeared
                    )
                )
            )


        row = {
            "time_s":
                float(
                    data.time
                ),

            "event":
                " | ".join(
                    event
                ),

            "active_contacts":
                ",".join(
                    sorted(
                        active
                    )
                ),

            "contact_detail":
                " ; ".join(
                    contact_summary
                ),

            "mcp_deg":
                q_deg[0],

            "pip_deg":
                q_deg[1],

            "dip_deg":
                q_deg[2],

            "mcp_qvel":
                qvel[0],

            "pip_qvel":
                qvel[1],

            "dip_qvel":
                qvel[2],

            "spool_mm":
                float(
                    data.qpos[
                        spool_qa
                    ]
                )
                * 1000.0,

            "spool_vel_mm_s":
                float(
                    data.qvel[
                        spool_da
                    ]
                )
                * 1000.0,

            "tendon_length_mm":
                float(
                    data.ten_length[
                        tid
                    ]
                )
                * 1000.0,

            "tendon_velocity_mm_s":
                float(
                    data.ten_velocity[
                        tid
                    ]
                )
                * 1000.0,

            "actuator_force":
                float(
                    data.actuator_force[
                        aid
                    ]
                ),

            "qfrc_constraint_mcp":
                float(
                    data.qfrc_constraint[
                        dof_addr(
                            model,
                            JOINTS[0],
                        )
                    ]
                ),

            "qfrc_constraint_pip":
                float(
                    data.qfrc_constraint[
                        dof_addr(
                            model,
                            JOINTS[1],
                        )
                    ]
                ),

            "qfrc_constraint_dip":
                float(
                    data.qfrc_constraint[
                        dof_addr(
                            model,
                            JOINTS[2],
                        )
                    ]
                ),

            "qfrc_passive_mcp":
                float(
                    data.qfrc_passive[
                        dof_addr(
                            model,
                            JOINTS[0],
                        )
                    ]
                ),

            "qfrc_passive_pip":
                float(
                    data.qfrc_passive[
                        dof_addr(
                            model,
                            JOINTS[1],
                        )
                    ]
                ),

            "qfrc_passive_dip":
                float(
                    data.qfrc_passive[
                        dof_addr(
                            model,
                            JOINTS[2],
                        )
                    ]
                ),
        }


        self.rows.append(
            row
        )


        print()
        print(
            "=" * 118
        )

        print(
            f"t={data.time:.4f}s   "
            f"{row['event']}"
        )

        print(
            f"active = "
            f"{row['active_contacts'] or 'NONE'}"
        )

        print(
            f"q = "
            f"{q_deg[0]:.3f} / "
            f"{q_deg[1]:.3f} / "
            f"{q_deg[2]:.3f} deg"
        )

        print(
            f"qvel = "
            f"{qvel[0]:+.4f} / "
            f"{qvel[1]:+.4f} / "
            f"{qvel[2]:+.4f} rad/s"
        )

        print(
            f"spool = "
            f"{row['spool_mm']:.3f} mm   "
            f"v={row['spool_vel_mm_s']:+.3f} mm/s"
        )

        print(
            f"tendon = "
            f"{row['tendon_length_mm']:.3f} mm   "
            f"v={row['tendon_velocity_mm_s']:+.3f} mm/s"
        )

        print(
            f"actuator force = "
            f"{row['actuator_force']:+.5f}"
        )

        print(
            "constraint torque MCP/PIP/DIP = "
            f"{row['qfrc_constraint_mcp']:+.6f} / "
            f"{row['qfrc_constraint_pip']:+.6f} / "
            f"{row['qfrc_constraint_dip']:+.6f}"
        )

        print(
            "passive torque MCP/PIP/DIP    = "
            f"{row['qfrc_passive_mcp']:+.6f} / "
            f"{row['qfrc_passive_pip']:+.6f} / "
            f"{row['qfrc_passive_dip']:+.6f}"
        )

        if contact_summary:

            print(
                "contacts:"
            )

            for s in contact_summary:

                print(
                    "  ",
                    s,
                )


        self.previous_active = active


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
    print("=" * 118)
    print("V35N1 — EVENT-RESOLVED INDEX CONTACT AUDIT")
    print("=" * 118)

    print(
        f"XML:\n{XML}"
    )

    print(
        f"\nBAG:\n{BAG}"
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

    old_k = float(
        b2.v35.CONTACT_K
    )


    contact_replay.CONTACT_DAMPING = 20.0

    b2.v35.CONTACT_MU = 0.05

    b2.v35.CONTACT_K = 1000.0


    audit = Audit()


    # ------------------------------------------------------------
    # e2b.run_one() itself temporarily wraps mujoco.mj_step.
    #
    # We wrap it first. Its wrapper will therefore call ours,
    # allowing us to observe the exact same simulation used for
    # the normal numerical evaluation.
    # ------------------------------------------------------------

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


    try:

        result = e2b.run_one(
            20.0,
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
            old_k
        )


    if audit.rows:

        fields = list(
            audit.rows[0].keys()
        )

        with OUT.open(
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()

            writer.writerows(
                audit.rows
            )


    print()
    print("=" * 118)
    print("NORMAL I1 RESULT")
    print("=" * 118)

    print(
        "pose = "
        f"{result['pose'][0]:.3f} / "
        f"{result['pose'][1]:.3f} / "
        f"{result['pose'][2]:.3f} deg"
    )

    print(
        f"RMSE = "
        f"{result['rmse']:.3f} deg"
    )

    print(
        "transitions mid/tip = "
        f"{result.get('mid_transitions')} / "
        f"{result.get('tip_transitions')}"
    )

    print()
    print(
        f"Captured contact-state changes = "
        f"{len(audit.rows)}"
    )

    print(
        f"CSV:\n{OUT}"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
