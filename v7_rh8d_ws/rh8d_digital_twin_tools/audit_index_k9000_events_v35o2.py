#!/usr/bin/env python3

from pathlib import Path
import csv
import shutil

import mujoco

import fit_index_passive_contact_v35b2 as b2
import scan_index_zero_margin_damping_v35e2b as e2b
import audit_index_contact_events_v35n1 as n1


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
    / "v35o2_k9000_event_audit"
)

OUTCSV = (
    OUTDIR
    / "v35o2_k9000_contact_events.csv"
)


TENDON_K = 9000.0

CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0
CONTACT_MU = 0.05


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

    shutil.copy2(
        XML,
        OUTDIR / XML.name,
    )


    print()
    print("=" * 150)
    print("V35O2 — K=9000 EVENT-RESOLVED CONTACT AUDIT")
    print("=" * 150)

    print(
        f"XML:\n{XML}"
    )

    print()
    print(
        f"Tendon stiffness = "
        f"{TENDON_K:.0f} N/m"
    )

    print(
        f"Contact K        = "
        f"{CONTACT_K:.0f}"
    )

    print(
        f"Contact damping  = "
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
        CONTACT_MU
    )

    contact_replay.CONTACT_DAMPING = (
        CONTACT_DAMPING
    )


    audit = n1.Audit()

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
            CONTACT_DAMPING,
            XML,
            BAG,
            contact_replay,
        )

    finally:

        mujoco.mj_step = real_step

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


    if audit.rows:

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


    print()
    print()
    print("=" * 150)
    print("POST-RUN CONTACT EVENT SEQUENCE")
    print("=" * 150)


    tip_start = None
    mid_start = None


    for i, row in enumerate(
        audit.rows,
        1,
    ):

        t = float(
            row["time_s"]
        )

        event = row[
            "event"
        ]


        print()
        print(
            f"[{i}] "
            f"t={t:.4f} s   "
            f"{event}"
        )

        print(
            f"    active  : "
            f"{row['active_contacts'] or 'NONE'}"
        )

        print(
            f"    contact : "
            f"{row['contact_detail'] or 'NONE'}"
        )

        print(
            "    q       : "
            f"{float(row['mcp_deg']):7.3f} / "
            f"{float(row['pip_deg']):7.3f} / "
            f"{float(row['dip_deg']):7.3f} deg"
        )

        print(
            "    qvel    : "
            f"{float(row['mcp_qvel']):+8.4f} / "
            f"{float(row['pip_qvel']):+8.4f} / "
            f"{float(row['dip_qvel']):+8.4f} rad/s"
        )

        print(
            "    spool   : "
            f"{float(row['spool_mm']):7.3f} mm   "
            f"v="
            f"{float(row['spool_vel_mm_s']):+8.3f} mm/s"
        )

        print(
            "    tendon  : "
            f"{float(row['tendon_length_mm']):8.3f} mm   "
            f"v="
            f"{float(row['tendon_velocity_mm_s']):+8.3f} mm/s"
        )

        print(
            "    actuator: "
            f"{float(row['actuator_force']):+9.5f}"
        )

        print(
            "    constraint torque MCP/PIP/DIP: "
            f"{float(row['qfrc_constraint_mcp']):+10.6f} / "
            f"{float(row['qfrc_constraint_pip']):+10.6f} / "
            f"{float(row['qfrc_constraint_dip']):+10.6f}"
        )

        print(
            "    passive torque    MCP/PIP/DIP: "
            f"{float(row['qfrc_passive_mcp']):+10.6f} / "
            f"{float(row['qfrc_passive_pip']):+10.6f} / "
            f"{float(row['qfrc_passive_dip']):+10.6f}"
        )


        if (
            "ON:collision_idx_tip"
            in event
        ):

            tip_start = t


        if (
            "OFF:collision_idx_tip"
            in event
            and tip_start is not None
        ):

            print(
                "    >>> TIP CONTACT DURATION = "
                f"{t - tip_start:.4f} s"
            )

            tip_start = None


        if (
            "ON:collision_idx_mid_mesh"
            in event
        ):

            mid_start = t


        if (
            "OFF:collision_idx_mid_mesh"
            in event
            and mid_start is not None
        ):

            print(
                "    >>> MIDDLE CONTACT DURATION = "
                f"{t - mid_start:.4f} s"
            )

            mid_start = None


    print()
    print("=" * 150)
    print("K=9000 FINAL NUMERICAL RESULT")
    print("=" * 150)

    q = result[
        "pose"
    ]

    print(
        "pose = "
        f"{q[0]:.3f} / "
        f"{q[1]:.3f} / "
        f"{q[2]:.3f} deg"
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

    print(
        "gaps mid/tip = "
        f"{result.get('target_mid_gap_mm'):+.3f} / "
        f"{result.get('target_tip_gap_mm'):+.3f} mm"
    )

    print()
    print(
        f"CSV:\n{OUTCSV}"
    )

    print(
        f"\nExperiment folder:\n{OUTDIR}"
    )

    print("=" * 150)


if __name__ == "__main__":
    main()
