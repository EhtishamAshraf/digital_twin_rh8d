#!/usr/bin/env python3

from pathlib import Path
import csv
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
    / "ur5e_rh8d_index_cad_spatial_v35j1_inextensible.xml"
)

OUT = (
    MJCF
    / "v35k1_moment_arm_curves.csv"
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


def jid(model, name):

    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing joint: {name}"
        )

    return i


def qadr(model, name):

    return int(
        model.jnt_qposadr[
            jid(
                model,
                name,
            )
        ]
    )


def tendon_id(model):

    i = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing tendon: {TENDON}"
        )

    return i


def set_angles(
    model,
    data,
    deg,
):

    for name, value in zip(
        JOINTS,
        deg,
    ):

        data.qpos[
            qadr(
                model,
                name,
            )
        ] = math.radians(
            float(value)
        )


def length(
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
    joint_name,
):

    qa = qadr(
        model,
        joint_name,
    )

    q0 = float(
        data.qpos[
            qa
        ]
    )

    eps = 1.0e-5


    data.qpos[
        qa
    ] = q0 + eps

    lp = length(
        model,
        data,
        tid,
    )


    data.qpos[
        qa
    ] = q0 - eps

    lm = length(
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


    dldq = (
        lp - lm
    ) / (
        2.0 * eps
    )


    # Positive number = flexion leverage magnitude.
    return (
        -dldq
    )


def measure(
    model,
    data,
    tid,
    angles,
    case,
):

    set_angles(
        model,
        data,
        angles,
    )


    L = length(
        model,
        data,
        tid,
    )


    arms = [
        moment_arm(
            model,
            data,
            tid,
            name,
        )
        for name in JOINTS
    ]


    return {
        "case":
            case,

        "mcp_deg":
            angles[0],

        "pip_deg":
            angles[1],

        "dip_deg":
            angles[2],

        "length_mm":
            L * 1000.0,

        "mcp_arm_mm":
            arms[0] * 1000.0,

        "pip_arm_mm":
            arms[1] * 1000.0,

        "dip_arm_mm":
            arms[2] * 1000.0,
    }


def print_row(row):

    print(
        f"{row['case']:24s} "
        f"q="
        f"{row['mcp_deg']:6.1f}/"
        f"{row['pip_deg']:6.1f}/"
        f"{row['dip_deg']:6.1f} deg    "
        f"r="
        f"{row['mcp_arm_mm']:+7.3f}/"
        f"{row['pip_arm_mm']:+7.3f}/"
        f"{row['dip_arm_mm']:+7.3f} mm"
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

    tid = tendon_id(
        model
    )


    rows = []


    print()
    print(
        "=" * 120
    )

    print(
        "V35K1 — CURRENT CAD SPATIAL TENDON MOMENT-ARM MAP"
    )

    print(
        "=" * 120
    )

    print()
    print(
        "Positive r means tendon shortening generates flexion."
    )

    print(
        "Negative r means the tendon would generate EXTENSION."
    )

    print()


    # ============================================================
    # 1. All three joints flex together.
    # ============================================================

    for deg in range(
        0,
        81,
        5,
    ):

        row = measure(
            model,
            data,
            tid,
            [
                deg,
                deg,
                deg,
            ],
            "equal_flexion",
        )

        rows.append(
            row
        )

        print_row(
            row
        )


    # ============================================================
    # 2. Isolate each joint independently.
    # ============================================================

    for joint_index, label in enumerate(
        LABELS
    ):

        print()
        print(
            "-" * 120
        )

        print(
            f"ISOLATED {label}"
        )

        print(
            "-" * 120
        )


        for deg in range(
            0,
            81,
            5,
        ):

            q = [
                0.0,
                0.0,
                0.0,
            ]

            q[
                joint_index
            ] = deg


            row = measure(
                model,
                data,
                tid,
                q,
                f"isolated_{label.lower()}",
            )

            rows.append(
                row
            )

            print_row(
                row
            )


    # ============================================================
    # 3. Important experimentally observed configurations.
    # ============================================================

    special = [

        (
            "hardware_contact",
            [
                14.5355,
                23.8110,
                9.3270,
            ],
        ),

        (
            "baseline_contact",
            [
                15.794,
                20.801,
                4.607,
            ],
        ),

        (
            "J1_contact",
            [
                11.464,
                27.530,
                13.046,
            ],
        ),

        (
            "J1_1mm",
            [
                6.2133,
                6.2710,
                6.8549,
            ],
        ),

        (
            "J1_3mm",
            [
                16.3927,
                16.8490,
                12.1712,
            ],
        ),

        (
            "J1_5mm",
            [
                27.8273,
                28.7400,
                8.8048,
            ],
        ),

        (
            "J1_8mm",
            [
                50.2082,
                50.9347,
                9.8891,
            ],
        ),
    ]


    print()
    print(
        "=" * 120
    )

    print(
        "IMPORTANT OBSERVED CONFIGURATIONS"
    )

    print(
        "=" * 120
    )


    for name, q in special:

        row = measure(
            model,
            data,
            tid,
            q,
            name,
        )

        rows.append(
            row
        )

        print_row(
            row
        )


    fields = [
        "case",
        "mcp_deg",
        "pip_deg",
        "dip_deg",
        "length_mm",
        "mcp_arm_mm",
        "pip_arm_mm",
        "dip_arm_mm",
    ]


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
            rows
        )


    print()
    print(
        "=" * 120
    )

    print(
        f"CSV saved:\n{OUT}"
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()
