#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

import scan_index_contact_friction_v35g7 as g7
import audit_index_tendon_transmission_v35g12 as g12


# =====================================================================
# G29A — MEASURED ENCODER vs SIMULATED SPOOL AUDIT
#
# Question:
#
# We already replay the measured HARDWARE encoder trajectory.
# Does the MuJoCo position actuator introduce another significant
# spool lag before that measurement reaches the tendon?
#
# If yes, system-identification replay is double-counting actuator
# dynamics.
#
# NO parameter changes.
# NO physics tuning.
# =====================================================================


SCALE = 5.70e-6

TENDON_NAME = "idx_flex_spatial"
SPOOL_NAME = "index_motor_spool"

CHECK_ENCODERS = [
    416.1,
    611.7,
    673.573,
    800.0,
    900.0,
    1027.0,
]


def nearest_index(values, target):
    values = np.asarray(values, dtype=float)
    return int(
        np.argmin(
            np.abs(values - target)
        )
    )


def tendon_jacobian_row(
    model,
    data,
    tendon_id,
):
    """
    Return one complete tendon Jacobian row for both dense and
    sparse MuJoCo representations.
    """

    import numpy as np

    raw = np.asarray(
        data.ten_J,
        dtype=float,
    )

    # Dense 2-D representation.
    if (
        raw.ndim == 2
        and
        raw.shape
        ==
        (
            model.ntendon,
            model.nv,
        )
    ):
        return raw[
            tendon_id,
            :
        ].copy()

    # Dense flattened representation.
    if (
        raw.ndim == 1
        and
        raw.size
        ==
        model.ntendon
        *
        model.nv
    ):
        return raw.reshape(
            model.ntendon,
            model.nv,
        )[
            tendon_id,
            :
        ].copy()

    # Sparse MuJoCo tendon Jacobian representation.
    J = np.zeros(
        model.nv,
        dtype=float,
    )

    rownnz = np.asarray(
        model.ten_J_rownnz,
        dtype=int,
    )

    rowadr = np.asarray(
        model.ten_J_rowadr,
        dtype=int,
    )

    colind = np.asarray(
        model.ten_J_colind,
        dtype=int,
    )

    nnz = int(
        rownnz[
            tendon_id
        ]
    )

    adr = int(
        rowadr[
            tendon_id
        ]
    )

    for k in range(
        nnz
    ):
        idx = (
            adr
            +
            k
        )

        col = int(
            colind[
                idx
            ]
        )

        J[
            col
        ] = float(
            raw[
                idx
            ]
        )

    return J



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            /
            "rh8d_results"
            /
            "v35g29a_spool_tracking_audit"
        ),
    )

    args = parser.parse_args()

    bag = args.contact_bag.expanduser().resolve()
    xml = args.xml.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Dynamic replay exactly as currently used.
    # --------------------------------------------------------------

    g7.b2.v30.v24.ab.patch_replay_modules()

    replay = g7.b2.v30.v24.base.contact_replay

    old_scale = float(
        g7.b2.v35.METERS_PER_TICK
    )

    g7.b2.v35.METERS_PER_TICK = SCALE

    try:
        (
            rows,
            qpos,
            times,
            summary,
        ) = g12.run_audit(
            xml,
            bag,
            replay,
        )

    finally:
        g7.b2.v35.METERS_PER_TICK = old_scale

    model = mujoco.MjModel.from_xml_path(
        str(xml)
    )

    data = mujoco.MjData(
        model
    )

    spool_jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_NAME,
    )

    tendon_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    if spool_jid < 0:
        raise RuntimeError(
            "index_motor_spool not found"
        )

    if tendon_id < 0:
        raise RuntimeError(
            "idx_flex_spatial not found"
        )

    spool_qadr = int(
        model.jnt_qposadr[
            spool_jid
        ]
    )

    encoder = np.asarray(
        [
            row["encoder_tick"]
            for row in rows
        ],
        dtype=float,
    )

    qpos = np.asarray(
        qpos,
        dtype=float,
    )

    # --------------------------------------------------------------
    # Measured encoder reference.
    # --------------------------------------------------------------

    open_i = nearest_index(
        encoder,
        100.0,
    )

    open_encoder = float(
        encoder[
            open_i
        ]
    )

    open_spool = float(
        qpos[
            open_i,
            spool_qadr
        ]
    )

    tendon_k = float(
        model.tendon_stiffness[
            tendon_id
        ]
    )

    spring_reference = (
        g12.tendon_spring_reference(
            model,
            tendon_id,
        )
    )

    refs = g12.get_joint_refs(
        model
    )

    print()
    print("=" * 150)
    print(
        "G29A — MEASURED ENCODER vs DYNAMIC SIMULATED SPOOL"
    )
    print("=" * 150)

    print()
    print(
        f"open encoder = "
        f"{open_encoder:.3f} ticks"
    )

    print(
        f"open spool   = "
        f"{1000.0 * open_spool:+.6f} mm"
    )

    print(
        f"scale        = "
        f"{SCALE * 1e6:.6f} um/tick"
    )

    print(
        f"tendon K     = "
        f"{tendon_k:.1f} N/m"
    )

    print()
    print(
        "The 'target spool' below is the spool displacement implied "
        "directly by the MEASURED hardware encoder."
    )

    print()

    print(
        "enc       target_spool  actual_spool   error     "
        "error_ticks   F_actual   F_direct   dF      dTau_MCP"
    )

    print("-" * 150)

    output_rows = []

    for requested_encoder in CHECK_ENCODERS:

        i = nearest_index(
            encoder,
            requested_encoder,
        )

        enc = float(
            encoder[
                i
            ]
        )

        target_spool = (
            open_spool
            +
            SCALE
            *
            (
                enc
                -
                open_encoder
            )
        )

        actual_spool = float(
            qpos[
                i,
                spool_qadr
            ]
        )

        spool_error = (
            actual_spool
            -
            target_spool
        )

        error_ticks = (
            spool_error
            /
            SCALE
        )

        # ----------------------------------------------------------
        # Actual dynamic configuration.
        # ----------------------------------------------------------

        data.qpos[:] = qpos[i]
        data.qvel[:] = 0.0

        mujoco.mj_forward(
            model,
            data,
        )

        actual_length = float(
            data.ten_length[
                tendon_id
            ]
        )

        actual_extension = (
            g12.signed_extension(
                actual_length,
                spring_reference,
            )
        )

        actual_force = (
            tendon_k
            *
            actual_extension
        )

        # ----------------------------------------------------------
        # Same finger configuration, but with spool placed exactly
        # where the measured encoder says it should be.
        # ----------------------------------------------------------

        direct_qpos = qpos[i].copy()

        direct_qpos[
            spool_qadr
        ] = target_spool

        data.qpos[:] = direct_qpos
        data.qvel[:] = 0.0

        mujoco.mj_forward(
            model,
            data,
        )

        direct_length = float(
            data.ten_length[
                tendon_id
            ]
        )

        direct_extension = (
            g12.signed_extension(
                direct_length,
                spring_reference,
            )
        )

        direct_force = (
            tendon_k
            *
            direct_extension
        )

        force_deficit = (
            direct_force
            -
            actual_force
        )

        # G28A equivalent-tendon generalized moment arms.
        #
        # data.ten_J can be dense or sparse depending on MuJoCo
        # build/version, so always reconstruct the complete row.
        J = tendon_jacobian_row(
            model,
            data,
            tendon_id,
        )

        mcp_J = abs(
            float(
                J[
                    refs[
                        "mcp"
                    ][
                        "dadr"
                    ]
                ]
            )
        )

        pip_J = abs(
            float(
                J[
                    refs[
                        "pip"
                    ][
                        "dadr"
                    ]
                ]
            )
        )

        dip_J = abs(
            float(
                J[
                    refs[
                        "dip"
                    ][
                        "dadr"
                    ]
                ]
            )
        )

        dtau_mcp = (
            force_deficit
            *
            mcp_J
        )

        dtau_pip = (
            force_deficit
            *
            pip_J
        )

        dtau_dip = (
            force_deficit
            *
            dip_J
        )

        row = rows[i]

        output_rows.append({
            "requested_encoder":
                requested_encoder,

            "actual_encoder":
                enc,

            "target_spool_mm":
                1000.0
                *
                target_spool,

            "actual_spool_mm":
                1000.0
                *
                actual_spool,

            "spool_error_mm":
                1000.0
                *
                spool_error,

            "equivalent_encoder_error_ticks":
                error_ticks,

            "actual_extension_mm":
                1000.0
                *
                actual_extension,

            "direct_input_extension_mm":
                1000.0
                *
                direct_extension,

            "actual_tendon_force_N":
                actual_force,

            "direct_input_tendon_force_N":
                direct_force,

            "force_deficit_N":
                force_deficit,

            "mcp_torque_deficit_Nm":
                dtau_mcp,

            "pip_torque_deficit_Nm":
                dtau_pip,

            "dip_torque_deficit_Nm":
                dtau_dip,

            "mcp_deg":
                row[
                    "mcp_deg"
                ],

            "pip_deg":
                row[
                    "pip_deg"
                ],

            "dip_deg":
                row[
                    "dip_deg"
                ],
        })

        print(
            f"{enc:7.1f} "
            f"{1000*target_spool:+12.4f} "
            f"{1000*actual_spool:+12.4f} "
            f"{1000*spool_error:+8.4f} "
            f"{error_ticks:+11.2f} "
            f"{actual_force:+9.4f} "
            f"{direct_force:+9.4f} "
            f"{force_deficit:+8.4f} "
            f"{dtau_mcp:+10.6f}"
        )

    # --------------------------------------------------------------
    # Contact-onset diagnostic.
    # --------------------------------------------------------------

    contact_i = nearest_index(
        encoder,
        673.573,
    )

    contact_result = min(
        output_rows,
        key=lambda r:
            abs(
                r[
                    "actual_encoder"
                ]
                -
                encoder[
                    contact_i
                ]
            ),
    )

    spool_error_mm = abs(
        contact_result[
            "spool_error_mm"
        ]
    )

    tick_error = abs(
        contact_result[
            "equivalent_encoder_error_ticks"
        ]
    )

    force_deficit = abs(
        contact_result[
            "force_deficit_N"
        ]
    )

    print()
    print("=" * 150)
    print(
        "G29A CONTACT-REGION DIAGNOSTIC"
    )
    print("=" * 150)

    print(
        f"|spool tracking error| = "
        f"{spool_error_mm:.4f} mm"
    )

    print(
        f"equivalent encoder lag = "
        f"{tick_error:.1f} ticks"
    )

    print(
        f"tendon-force deficit   = "
        f"{force_deficit:.4f} N"
    )

    print(
        f"MCP torque deficit     = "
        f"{abs(contact_result['mcp_torque_deficit_Nm']):.6f} Nm"
    )

    print()

    if (
        spool_error_mm >= 0.10
        or
        tick_error >= 20.0
    ):

        print(
            "RESULT: SIMULATED SPOOL TRACKING ERROR IS "
            "MECHANICALLY SIGNIFICANT."
        )

        print()
        print(
            "The measured hardware encoder is already an observed "
            "actuator state."
        )

        print(
            "Next experiment should replay that measured spool state "
            "directly and remove the extra simulated servo lag."
        )

        print(
            "Finger joints remain fully dynamic; only the measured "
            "actuator input is prescribed."
        )

    else:

        print(
            "RESULT: SIMULATED SPOOL TRACKING ERROR IS TOO SMALL "
            "TO EXPLAIN THE H1 MISMATCH."
        )

        print()
        print(
            "Next model class should be a lumped palm-transmission "
            "compliance/friction/hysteresis element."
        )

    # --------------------------------------------------------------
    # Save.
    # --------------------------------------------------------------

    csv_path = (
        out
        /
        "g29a_spool_tracking.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                output_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    print()
    print(
        f"CSV:\n  {csv_path}"
    )

    print()
    print(
        f"All outputs:\n  {out}"
    )


if __name__ == "__main__":
    main()
