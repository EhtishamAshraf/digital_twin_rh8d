#!/usr/bin/env python3

from pathlib import Path
import math

import mujoco
import numpy as np


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


JOINTS = [
    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",
]

LABELS = [
    "MCP",
    "PIP",
    "DIP",
]

TENDON = "idx_flex_spatial"


CASES = [
    (
        "open",
        [0.0, 0.0, 0.0],
    ),

    (
        "hardware_contact",
        [14.5355, 23.8110, 9.3270],
    ),

    (
        "baseline_contact",
        [15.794, 20.801, 4.607],
    ),

    (
        "J1_contact",
        [11.464, 27.530, 13.046],
    ),

    (
        "J1_1mm",
        [6.2133, 6.2710, 6.8549],
    ),

    (
        "J1_3mm",
        [16.3927, 16.8490, 12.1712],
    ),

    (
        "J1_5mm",
        [27.8273, 28.7400, 8.8048],
    ),

    (
        "J1_8mm",
        [50.2082, 50.9347, 9.8891],
    ),
]


def get_id(
    model,
    obj,
    name,
):

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


def qadr(
    model,
    name,
):

    jid = get_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return int(
        model.jnt_qposadr[jid]
    )


def set_angles(
    model,
    data,
    angles_deg,
):

    for name, deg in zip(
        JOINTS,
        angles_deg,
    ):

        data.qpos[
            qadr(
                model,
                name,
            )
        ] = math.radians(
            deg
        )


def tendon_length(
    model,
    data,
    tid,
):

    mujoco.mj_forward(
        model,
        data,
    )

    return float(
        data.ten_length[
            tid
        ]
    )


def moment_arm(
    model,
    data,
    tid,
    joint,
):

    qa = qadr(
        model,
        joint,
    )

    q0 = float(
        data.qpos[
            qa
        ]
    )

    eps = 1e-5


    data.qpos[
        qa
    ] = q0 + eps

    lp = tendon_length(
        model,
        data,
        tid,
    )


    data.qpos[
        qa
    ] = q0 - eps

    lm = tendon_length(
        model,
        data,
        tid,
    )


    data.qpos[
        qa
    ] = q0

    mujoco.mj_forward(
        model,
        data,
    )


    return -(
        lp - lm
    ) / (
        2.0 * eps
    )


def main():

    if not XML.exists():

        raise RuntimeError(
            f"Missing XML:\n{XML}"
        )


    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )

    data = mujoco.MjData(
        model
    )


    tid = get_id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON,
    )


    print()
    print(
        "=" * 128
    )

    print(
        "V35L1 — PASSIVE RETURN TORQUE AUDIT"
    )

    print(
        "=" * 128
    )


    print()
    print(
        "Joint parameters:"
    )


    for label, name in zip(
        LABELS,
        JOINTS,
    ):

        jid = get_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        print(
            f"{label}: "
            f"k={model.jnt_stiffness[jid]:.6f} Nm/rad   "
            f"springref={math.degrees(model.qpos_spring[qadr(model,name)]):+.3f} deg   "
            f"damping={model.dof_damping[model.jnt_dofadr[jid]]:.6f}   "
            f"frictionloss={model.dof_frictionloss[model.jnt_dofadr[jid]]:.6f}"
        )


    print()
    print(
        "=" * 128
    )

    print(
        "CONFIGURATION AUDIT"
    )

    print(
        "=" * 128
    )


    for case, angles in CASES:

        set_angles(
            model,
            data,
            angles,
        )

        mujoco.mj_forward(
            model,
            data,
        )


        print()
        print(
            "-" * 128
        )

        print(
            f"{case}: "
            f"{angles[0]:.3f}/"
            f"{angles[1]:.3f}/"
            f"{angles[2]:.3f} deg"
        )

        print(
            "-" * 128
        )


        for label, name in zip(
            LABELS,
            JOINTS,
        ):

            jid = get_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            qa = qadr(
                model,
                name,
            )


            q = float(
                data.qpos[
                    qa
                ]
            )

            qref = float(
                model.qpos_spring[
                    qa
                ]
            )

            k = float(
                model.jnt_stiffness[
                    jid
                ]
            )


            tau_passive = (
                -k
                *
                (
                    q - qref
                )
            )


            r = moment_arm(
                model,
                data,
                tid,
                name,
            )


            if abs(r) > 1e-9:

                tension_equiv = (
                    abs(
                        tau_passive
                    )
                    /
                    abs(r)
                )

            else:

                tension_equiv = (
                    math.inf
                )


            print(
                f"{label}: "
                f"r={r * 1000:+7.3f} mm   "
                f"tau_spring={tau_passive:+.6f} Nm   "
                f"equiv_F={tension_equiv:7.3f} N"
            )


    print()
    print(
        "=" * 128
    )

    print(
        "Interpretation:"
    )

    print(
        "If DIP equivalent tension is several times MCP/PIP "
        "through the operating region, the current independent "
        "joint-spring approximation is strongly biasing load "
        "redistribution away from DIP."
    )

    print(
        "=" * 128
    )


if __name__ == "__main__":
    main()
