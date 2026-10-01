#!/usr/bin/env python3

from pathlib import Path
import math

import mujoco
import numpy as np


WS = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_tabletop_sysid_v1.xml"
)


# ============================================================
# Identified/provisional hand model
# ============================================================

TENDON_NAME = "idx_flex_spatial"
TENDON_STIFFNESS = 2100.0

SCALE = 5.1e-6       # m / encoder tick
OPEN_TICKS = 100.0
MAX_TICKS = 2200.0
STEP_TICKS = 25.0

FIXTURE_NAME = "index_contact_fixture"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"

SPOOL_NAME = "index_motor_spool"
INDEX_ACTUATOR = "pos_index_tendon"


# ============================================================
# Known old bottle location.
#
# Only used to choose WHICH SIDE of the index finger the
# bottle should be searched on.
# ============================================================

OLD_BOTTLE_XY = np.array(
    [
        0.475,
       -0.340,
    ],
    dtype=float,
)


# ============================================================
# Search coordinates.
#
# "along" = distance from MCP along open index finger
# "normal" = lateral distance from finger centreline
#
# Bottle radius = 34.6 mm
# Finger collision radius ~7 mm
# Tangency therefore occurs near ~42 mm.
# ============================================================

ALONG_MM = [
     0.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
]

NORMAL_MM = [
    38.0,
    42.0,
    46.0,
    50.0,
    54.0,
    58.0,
]


# Contact force below this is ignored as numerical/noise contact.
FORCE_THRESHOLD = 0.005  # N


ARM_POSE = {
    "shoulder_pan_joint":   1.4927,
    "shoulder_lift_joint": -0.8678,
    "elbow_joint":         -2.43,
    "wrist_1_joint":       -0.6596,
    "wrist_2_joint":       -0.0347,
    "wrist_3_joint":        0.126,
}


ARM_ACTUATORS = {
    "shoulder_pan":  "shoulder_pan_joint",
    "shoulder_lift": "shoulder_lift_joint",
    "elbow":         "elbow_joint",
    "wrist_1":       "wrist_1_joint",
    "wrist_2":       "wrist_2_joint",
    "wrist_3":       "wrist_3_joint",
}


HAND_QPOS = {
    "forearm:1--base:1":                  0.0,
    "palm_axis:1--forearm:1":             0.0,
    "palm_assembly:1--palm_axis:1":       0.0,
    "Thumb_axis--palm_assembly:1":         1.57,
    "Thumb_Methacarpal--Thumb_axis":       0.0,

    "Index_Proximal--palm_assembly:1":     0.0,
    "Index_Middle--Index_Proximal":        0.0,
    "Index_Distal--Index_Middle":          0.0,

    "Middle_Proximal--palm_assembly:1":    0.0,
    "Ring_Proximal--palm_assembly:1":      0.0,
}


OTHER_CONTROLS = {
    "pos_forearm":          0.0,
    "pos_palm_axis":        0.0,
    "pos_palm_assembly":    0.0,
    "pos_thumb_axis":       1.57,
    "pos_thumb_tendon":     0.0,
    "pos_middle_tendon":    0.0,
    "pos_ring_small_tendon":0.0,
}


def oid(model, typ, name):

    i = mujoco.mj_name2id(
        model,
        typ,
        name,
    )

    if i < 0:
        raise RuntimeError(
            f"Missing MuJoCo object: {name}"
        )

    return i


def set_joint(model, data, name, value):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    data.qpos[
        model.jnt_qposadr[jid]
    ] = float(value)


def set_ctrl_if_present(
    model,
    data,
    name,
    value,
):

    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )

    if aid >= 0:
        data.ctrl[aid] = float(value)


def joint_angle_deg(
    model,
    data,
    name,
):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    qadr = model.jnt_qposadr[
        jid
    ]

    return math.degrees(
        data.qpos[qadr]
    )


def configure_pose(
    model,
    data,
):

    mujoco.mj_resetData(
        model,
        data,
    )


    for name, value in ARM_POSE.items():
        set_joint(
            model,
            data,
            name,
            value,
        )


    for actuator, joint in ARM_ACTUATORS.items():
        set_ctrl_if_present(
            model,
            data,
            actuator,
            ARM_POSE[joint],
        )


    for name, value in HAND_QPOS.items():
        set_joint(
            model,
            data,
            name,
            value,
        )


    for name, value in OTHER_CONTROLS.items():
        set_ctrl_if_present(
            model,
            data,
            name,
            value,
        )


    set_joint(
        model,
        data,
        SPOOL_NAME,
        0.0,
    )


    set_ctrl_if_present(
        model,
        data,
        INDEX_ACTUATOR,
        0.0,
    )


    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


def bottle_contact_forces(
    model,
    data,
    fixture_gid,
):

    forces = {}


    for i in range(data.ncon):

        c = data.contact[i]


        if (
            c.geom1 != fixture_gid
            and
            c.geom2 != fixture_gid
        ):
            continue


        other = (
            c.geom2
            if c.geom1 == fixture_gid
            else c.geom1
        )


        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            other,
        )

        if name is None:
            name = f"geom_{other}"


        wrench = np.zeros(
            6,
            dtype=float,
        )

        mujoco.mj_contactForce(
            model,
            data,
            i,
            wrench,
        )


        normal_force = max(
            0.0,
            float(wrench[0]),
        )


        if normal_force < FORCE_THRESHOLD:
            continue


        forces[name] = (
            forces.get(name, 0.0)
            + normal_force
        )


    return forces


def is_index_geom(name):

    return name.startswith(
        "collision_idx_"
    )


def settle(
    model,
    data,
    seconds,
):

    n = max(
        1,
        int(
            round(
                seconds
                / model.opt.timestep
            )
        ),
    )

    for _ in range(n):
        mujoco.mj_step(
            model,
            data,
        )


def main():

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(
        model
    )


    tendon_id = oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    model.tendon_stiffness[
        tendon_id
    ] = TENDON_STIFFNESS


    fixture_gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FIXTURE_NAME,
    )


    index_actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        INDEX_ACTUATOR,
    )


    # ========================================================
    # Determine index-plane XY basis from the actual startup pose.
    # ========================================================

    configure_pose(
        model,
        data,
    )


    mcp_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        MCP_NAME,
    )

    pip_jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        PIP_NAME,
    )


    mcp_world = data.xanchor[
        mcp_jid
    ].copy()

    pip_world = data.xanchor[
        pip_jid
    ].copy()


    forward_xy = (
        pip_world[:2]
        - mcp_world[:2]
    )

    forward_xy /= np.linalg.norm(
        forward_xy
    )


    # 90-degree planar normal.
    normal_xy = np.array(
        [
            -forward_xy[1],
             forward_xy[0],
        ]
    )


    # Keep normal pointing toward the same side as the old bottle.
    old_delta = (
        OLD_BOTTLE_XY
        - mcp_world[:2]
    )


    if np.dot(
        old_delta,
        normal_xy,
    ) < 0:

        normal_xy *= -1.0


    print()
    print("=" * 120)
    print("TABLETOP BOTTLE XY SEARCH V1")
    print("=" * 120)

    print()
    print(
        "MCP world =",
        mcp_world,
    )

    print(
        "PIP world =",
        pip_world,
    )

    print(
        "forward XY =",
        forward_xy,
    )

    print(
        "normal XY  =",
        normal_xy,
    )

    print()
    print(
        f"Testing "
        f"{len(ALONG_MM)*len(NORMAL_MM)} "
        f"bottle positions..."
    )


    results = []


    for along_mm in ALONG_MM:

        for normal_mm in NORMAL_MM:

            bottle_xy = (
                mcp_world[:2]
                +
                forward_xy
                * (
                    along_mm
                    / 1000.0
                )
                +
                normal_xy
                * (
                    normal_mm
                    / 1000.0
                )
            )


            model.geom_pos[
                fixture_gid,
                0
            ] = bottle_xy[0]

            model.geom_pos[
                fixture_gid,
                1
            ] = bottle_xy[1]


            configure_pose(
                model,
                data,
            )


            # Allow arm + passive finger to settle at OPEN.
            settle(
                model,
                data,
                0.30,
            )


            open_forces = bottle_contact_forces(
                model,
                data,
                fixture_gid,
            )


            first_any = None
            first_mid = None
            first_prox = None
            first_tip = None
            first_other = None

            mid_pose = None
            tip_pose = None

            max_other_force = 0.0


            ticks_values = np.arange(
                OPEN_TICKS,
                MAX_TICKS + 0.1,
                STEP_TICKS,
            )


            for ticks in ticks_values:

                displacement = max(
                    0.0,
                    (
                        ticks
                        - OPEN_TICKS
                    )
                    * SCALE,
                )


                data.ctrl[
                    index_actuator
                ] = displacement


                # Quasi-static settlement at each motor position.
                settle(
                    model,
                    data,
                    0.04,
                )


                forces = bottle_contact_forces(
                    model,
                    data,
                    fixture_gid,
                )


                active = set(
                    forces.keys()
                )


                if active and first_any is None:

                    first_any = (
                        float(ticks),
                        sorted(active),
                    )


                if (
                    "collision_idx_mid"
                    in active
                    and
                    first_mid is None
                ):

                    first_mid = float(
                        ticks
                    )

                    mid_pose = (
                        joint_angle_deg(
                            model,
                            data,
                            MCP_NAME,
                        ),
                        joint_angle_deg(
                            model,
                            data,
                            PIP_NAME,
                        ),
                        joint_angle_deg(
                            model,
                            data,
                            DIP_NAME,
                        ),
                    )


                if (
                    "collision_idx_prox"
                    in active
                    and
                    first_prox is None
                ):

                    first_prox = float(
                        ticks
                    )


                if (
                    "collision_idx_tip"
                    in active
                    and
                    first_tip is None
                ):

                    first_tip = float(
                        ticks
                    )

                    tip_pose = (
                        joint_angle_deg(
                            model,
                            data,
                            MCP_NAME,
                        ),
                        joint_angle_deg(
                            model,
                            data,
                            PIP_NAME,
                        ),
                        joint_angle_deg(
                            model,
                            data,
                            DIP_NAME,
                        ),
                    )


                other_names = [
                    name
                    for name in active
                    if not is_index_geom(
                        name
                    )
                ]


                if other_names:

                    if first_other is None:

                        first_other = (
                            float(ticks),
                            sorted(
                                other_names
                            ),
                        )


                    for name in other_names:

                        max_other_force = max(
                            max_other_force,
                            forces[name],
                        )


            # =================================================
            # Physical ranking.
            # =================================================

            score = 0.0


            if open_forces:
                score += 10000.0


            if first_other is not None:
                score += 5000.0


            if first_mid is None:
                score += 3000.0


            if first_tip is None:
                score += 1500.0


            if (
                first_mid is not None
                and
                first_tip is not None
            ):

                if first_mid >= first_tip:

                    # Wrong sequence.
                    score += 4000.0

                else:

                    # Weakly prefer hardware-like tip onset.
                    score += (
                        0.10
                        * abs(
                            first_tip
                            - 1020.0
                        )
                    )


            if first_any is not None:

                first_names = (
                    first_any[1]
                )

                if (
                    "collision_idx_mid"
                    not in first_names
                ):

                    score += 1000.0


            valid = (
                not open_forces

                and

                first_other is None

                and

                first_mid is not None

                and

                first_tip is not None

                and

                first_mid < first_tip

                and

                first_any is not None

                and

                "collision_idx_mid"
                in first_any[1]
            )


            result = {
                "along_mm":
                    along_mm,

                "normal_mm":
                    normal_mm,

                "xy":
                    bottle_xy.copy(),

                "open_forces":
                    open_forces,

                "first_any":
                    first_any,

                "first_mid":
                    first_mid,

                "first_prox":
                    first_prox,

                "first_tip":
                    first_tip,

                "first_other":
                    first_other,

                "mid_pose":
                    mid_pose,

                "tip_pose":
                    tip_pose,

                "score":
                    score,

                "valid":
                    valid,
            }


            results.append(
                result
            )


            def ft(v):

                return (
                    "----"
                    if v is None
                    else f"{v:6.0f}"
                )


            print(
                f"along={along_mm:5.1f} mm  "
                f"normal={normal_mm:5.1f} mm  "
                f"open={'YES' if open_forces else ' no'}  "
                f"mid={ft(first_mid)}  "
                f"tip={ft(first_tip)}  "
                f"prox={ft(first_prox)}  "
                f"other={'YES' if first_other else ' no'}"
            )


    ranked = sorted(
        results,
        key=lambda r:
            r["score"],
    )


    print()
    print("=" * 120)
    print("TOP 12 TABLETOP GEOMETRY CANDIDATES")
    print("=" * 120)


    for rank, r in enumerate(
        ranked[:12],
        start=1,
    ):

        print()
        print(
            f"#{rank}  "
            f"along={r['along_mm']:.1f} mm  "
            f"normal={r['normal_mm']:.1f} mm  "
            f"score={r['score']:.3f}"
        )

        print(
            f"  bottle XY = "
            f"{r['xy'][0]:.9f} "
            f"{r['xy'][1]:.9f}"
        )

        print(
            f"  open contacts = "
            f"{r['open_forces']}"
        )

        print(
            f"  first any  = "
            f"{r['first_any']}"
        )

        print(
            f"  first prox = "
            f"{r['first_prox']}"
        )

        print(
            f"  first mid  = "
            f"{r['first_mid']}"
        )

        print(
            f"  first tip  = "
            f"{r['first_tip']}"
        )

        print(
            f"  first other = "
            f"{r['first_other']}"
        )

        print(
            f"  pose at MID = "
            f"{r['mid_pose']}"
        )

        print(
            f"  pose at TIP = "
            f"{r['tip_pose']}"
        )


    valid = [
        r
        for r in ranked
        if r["valid"]
    ]


    print()
    print("=" * 120)
    print("MIDDLE-FIRST PHYSICALLY VALID CANDIDATES")
    print("=" * 120)


    if not valid:

        print("NONE")

    else:

        for r in valid:

            print()
            print(
                f"along={r['along_mm']:.1f} mm  "
                f"normal={r['normal_mm']:.1f} mm"
            )

            print(
                f"  bottle center = "
                f"{r['xy'][0]:.9f} "
                f"{r['xy'][1]:.9f} "
                f"0.799000000"
            )

            print(
                f"  middle contact = "
                f"{r['first_mid']:.0f} ticks"
            )

            print(
                f"  tip contact    = "
                f"{r['first_tip']:.0f} ticks"
            )

            print(
                f"  MCP/PIP/DIP @ middle = "
                f"{r['mid_pose']}"
            )

            print(
                f"  MCP/PIP/DIP @ tip    = "
                f"{r['tip_pose']}"
            )


    print()
    print("=" * 120)


if __name__ == "__main__":
    main()
