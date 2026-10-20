#!/usr/bin/env python3

from pathlib import Path
import math
import mujoco


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser()

XML = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
    / "ur5e_rh8d_index_cad_spatial_v35m1_passive_return.xml"
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

TENDON = "idx_return_spatial"


CASES = [
    (
        "open",
        [0.0, 0.0, 0.0],
    ),

    (
        "hardware",
        [14.5355, 23.8110, 9.3270],
    ),

    (
        "I1",
        [15.794, 20.801, 4.607],
    ),

    (
        "M1",
        [15.236, 11.008, 25.729],
    ),
]


def get_id(model, obj, name):

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


def qadr(model, name):

    j = get_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return int(
        model.jnt_qposadr[j]
    )


def forward_length(
    model,
    data,
    tid,
):

    mujoco.mj_forward(
        model,
        data,
    )

    return float(
        data.ten_length[tid]
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
        data.qpos[qa]
    )

    eps = 1e-5

    data.qpos[qa] = q0 + eps

    lp = forward_length(
        model,
        data,
        tid,
    )

    data.qpos[qa] = q0 - eps

    lm = forward_length(
        model,
        data,
        tid,
    )

    data.qpos[qa] = q0

    mujoco.mj_forward(
        model,
        data,
    )

    # dL/dq:
    #
    # For a dorsal return tendon we WANT:
    # flexion -> tendon gets longer.
    #
    # Therefore positive dL/dq means the spring
    # generates restoring extension torque.
    return (
        lp - lm
    ) / (
        2.0 * eps
    )


def main():

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
    print("=" * 112)
    print("M1 PASSIVE RETURN MOMENT-ARM AUDIT")
    print("=" * 112)

    print(
        f"return stiffness = "
        f"{model.tendon_stiffness[tid]:.3f} N/m"
    )

    print()
    print(
        "For the return tendon, positive dL/dq is desired:"
    )

    print(
        "joint flexion should STRETCH the elastic return element."
    )


    for case, q in CASES:

        for name, deg in zip(
            JOINTS,
            q,
        ):

            data.qpos[
                qadr(
                    model,
                    name,
                )
            ] = math.radians(
                deg
            )


        mujoco.mj_forward(
            model,
            data,
        )

        L = float(
            data.ten_length[
                tid
            ]
        )

        spring_L0 = float(
            model.tendon_lengthspring[
                tid,
                0
            ]
        )

        extension = (
            L - spring_L0
        )

        tension = max(
            0.0,
            model.tendon_stiffness[tid]
            * extension,
        )


        print()
        print("-" * 112)

        print(
            f"{case}: "
            f"{q[0]:.3f}/"
            f"{q[1]:.3f}/"
            f"{q[2]:.3f} deg"
        )

        print(
            f"L={L*1000:.3f} mm   "
            f"extension={extension*1000:+.3f} mm   "
            f"spring tension≈{tension:.3f} N"
        )


        for label, joint in zip(
            LABELS,
            JOINTS,
        ):

            r = moment_arm(
                model,
                data,
                tid,
                joint,
            )

            tau = (
                -tension
                * r
            )

            print(
                f"{label}: "
                f"dL/dq={r*1000:+7.3f} mm/rad   "
                f"return torque≈{tau:+.6f} Nm"
            )


    print()
    print("=" * 112)


if __name__ == "__main__":
    main()
