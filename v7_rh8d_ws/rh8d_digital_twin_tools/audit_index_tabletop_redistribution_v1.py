#!/usr/bin/env python3

from pathlib import Path
import math
import csv

import mujoco
import numpy as np


WS = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_tabletop_sysid_v1.xml"
)


# ============================================================
# Provisional clean middle-first geometry
# ============================================================

BOTTLE_POS = np.array(
    [
         0.505349318,
        -0.371980074,
         0.799000000,
    ],
    dtype=float,
)


# ============================================================
# Current Candidate-A mechanics
# ============================================================

TENDON_NAME = "idx_flex_spatial"
TENDON_STIFFNESS = 2100.0

SCALE = 5.1e-6

OPEN_TICKS = 100.0
MAX_TICKS = 2200.0
STEP_TICKS = 25.0

SETTLE_TIME = 0.060

FORCE_THRESHOLD = 0.005


FIXTURE_NAME = "index_contact_fixture"

SPOOL_NAME = "index_motor_spool"
INDEX_ACTUATOR = "pos_index_tendon"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"


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
    "forearm:1--base:1":               0.0,
    "palm_axis:1--forearm:1":          0.0,
    "palm_assembly:1--palm_axis:1":    0.0,

    "Thumb_axis--palm_assembly:1":      1.57,
    "Thumb_Methacarpal--Thumb_axis":    0.0,

    "Index_Proximal--palm_assembly:1":  0.0,
    "Index_Middle--Index_Proximal":     0.0,
    "Index_Distal--Index_Middle":       0.0,

    "Middle_Proximal--palm_assembly:1": 0.0,
    "Ring_Proximal--palm_assembly:1":   0.0,
}


OTHER_CONTROLS = {
    "pos_forearm":           0.0,
    "pos_palm_axis":         0.0,
    "pos_palm_assembly":     0.0,

    "pos_thumb_axis":        1.57,
    "pos_thumb_tendon":      0.0,

    "pos_middle_tendon":     0.0,
    "pos_ring_small_tendon": 0.0,
}


def oid(model, typ, name):

    idx = mujoco.mj_name2id(
        model,
        typ,
        name,
    )

    if idx < 0:
        raise RuntimeError(
            f"Missing MuJoCo object: {name}"
        )

    return idx


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


def angle_deg(model, data, name):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return math.degrees(
        data.qpos[
            model.jnt_qposadr[jid]
        ]
    )


def settle(model, data, seconds):

    steps = max(
        1,
        int(
            round(
                seconds
                / model.opt.timestep
            )
        ),
    )

    for _ in range(steps):

        mujoco.mj_step(
            model,
            data,
        )


def configure_pose(model, data):

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


def contact_forces(
    model,
    data,
    fixture_gid,
):

    result = {}


    for i in range(
        data.ncon
    ):

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


        fn = max(
            0.0,
            float(wrench[0]),
        )


        if fn < FORCE_THRESHOLD:
            continue


        result[name] = (
            result.get(
                name,
                0.0,
            )
            + fn
        )


    return result


def slope_deg_per_100ticks(
    rows,
    key,
):

    if len(rows) < 2:
        return None


    x = np.array(
        [
            r["ticks"]
            for r in rows
        ],
        dtype=float,
    )


    y = np.array(
        [
            r[key]
            for r in rows
        ],
        dtype=float,
    )


    p = np.polyfit(
        x,
        y,
        1,
    )


    return float(
        p[0]
        * 100.0
    )


def main():

    model = mujoco.MjModel.from_xml_path(
        str(
            MODEL_PATH
        )
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


    model.geom_pos[
        fixture_gid
    ] = BOTTLE_POS


    index_actuator = oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        INDEX_ACTUATOR,
    )


    configure_pose(
        model,
        data,
    )


    settle(
        model,
        data,
        0.30,
    )


    open_contacts = contact_forces(
        model,
        data,
        fixture_gid,
    )


    print()
    print("=" * 120)
    print(
        "INDEX TABLETOP CONTACT REDISTRIBUTION AUDIT V1"
    )
    print("=" * 120)

    print()
    print(
        "Bottle =",
        BOTTLE_POS,
    )

    print(
        "Open contacts =",
        open_contacts,
    )


    rows = []


    first_mid = None
    first_tip = None


    for ticks in np.arange(
        OPEN_TICKS,
        MAX_TICKS + 0.1,
        STEP_TICKS,
    ):

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


        settle(
            model,
            data,
            SETTLE_TIME,
        )


        forces = contact_forces(
            model,
            data,
            fixture_gid,
        )


        row = {
            "ticks":
                float(ticks),

            "disp_mm":
                1000.0
                * displacement,

            "mcp":
                angle_deg(
                    model,
                    data,
                    MCP_NAME,
                ),

            "pip":
                angle_deg(
                    model,
                    data,
                    PIP_NAME,
                ),

            "dip":
                angle_deg(
                    model,
                    data,
                    DIP_NAME,
                ),

            "prox_force":
                forces.get(
                    "collision_idx_prox",
                    0.0,
                ),

            "mid_force":
                forces.get(
                    "collision_idx_mid",
                    0.0,
                ),

            "tip_force":
                forces.get(
                    "collision_idx_tip",
                    0.0,
                ),

            "other":
                "|".join(
                    sorted(
                        name
                        for name in forces
                        if not name.startswith(
                            "collision_idx_"
                        )
                    )
                ),
        }


        rows.append(
            row
        )


        if (
            row["mid_force"]
            >= FORCE_THRESHOLD
            and
            first_mid is None
        ):

            first_mid = len(rows) - 1


        if (
            row["tip_force"]
            >= FORCE_THRESHOLD
            and
            first_tip is None
        ):

            first_tip = len(rows) - 1


    # ========================================================
    # CSV
    # ========================================================

    csv_path = (
        WS
        / "rh8d_digital_twin_tools/"
          "index_tabletop_redistribution_v1.csv"
    )


    with open(
        csv_path,
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


    # ========================================================
    # Event summary
    # ========================================================

    print()
    print("=" * 120)
    print("EVENT SUMMARY")
    print("=" * 120)


    if first_mid is None:

        print(
            "Middle contact: NONE"
        )

    else:

        r = rows[
            first_mid
        ]

        print()
        print(
            f"Middle contact = "
            f"{r['ticks']:.0f} ticks"
        )

        print(
            f"  pose = "
            f"({r['mcp']:.3f}, "
            f"{r['pip']:.3f}, "
            f"{r['dip']:.3f}) deg"
        )

        print(
            f"  middle force = "
            f"{r['mid_force']:.5f} N"
        )


    if first_tip is None:

        print()
        print(
            "Tip contact: NONE"
        )

    else:

        r = rows[
            first_tip
        ]

        print()
        print(
            f"Tip contact = "
            f"{r['ticks']:.0f} ticks"
        )

        print(
            f"  pose = "
            f"({r['mcp']:.3f}, "
            f"{r['pip']:.3f}, "
            f"{r['dip']:.3f}) deg"
        )

        print(
            f"  middle force = "
            f"{r['mid_force']:.5f} N"
        )

        print(
            f"  tip force    = "
            f"{r['tip_force']:.5f} N"
        )


    # ========================================================
    # Contact persistence
    # ========================================================

    if (
        first_mid is not None
        and
        first_tip is not None
        and
        first_tip >= first_mid
    ):

        segment = rows[
            first_mid:
            first_tip + 1
        ]


        mid_duty = np.mean(
            [
                r["mid_force"]
                >= FORCE_THRESHOLD
                for r in segment
            ]
        )


        print()
        print(
            f"Middle-contact duty "
            f"from MID -> TIP = "
            f"{100*mid_duty:.1f}%"
        )


        tiprow = rows[
            first_tip
        ]


        print(
            "Middle still active "
            "when tip contacts = "
            + (
                "YES"
                if tiprow["mid_force"]
                >= FORCE_THRESHOLD
                else "NO"
            )
        )


    # ========================================================
    # Motion redistribution
    # ========================================================

    print()
    print("=" * 120)
    print("MOTION REDISTRIBUTION")
    print("=" * 120)


    if first_mid is not None:

        mid_tick = rows[
            first_mid
        ][
            "ticks"
        ]


        pre = [
            r
            for r in rows
            if (
                mid_tick - 200
                <= r["ticks"]
                < mid_tick
            )
        ]


        post_end = (
            rows[first_tip]["ticks"]
            if first_tip is not None
            else mid_tick + 200
        )


        post = [
            r
            for r in rows
            if (
                mid_tick
                <= r["ticks"]
                <= post_end
            )
        ]


        print()
        print(
            "Before middle contact "
            "(deg / 100 ticks):"
        )


        for key in [
            "mcp",
            "pip",
            "dip",
        ]:

            print(
                f"  {key.upper():3s} = "
                f"{slope_deg_per_100ticks(pre, key):+.3f}"
            )


        print()
        print(
            "Middle contact -> tip contact "
            "(deg / 100 ticks):"
        )


        for key in [
            "mcp",
            "pip",
            "dip",
        ]:

            print(
                f"  {key.upper():3s} = "
                f"{slope_deg_per_100ticks(post, key):+.3f}"
            )


    # ========================================================
    # Print rows around transitions
    # ========================================================

    print()
    print("=" * 120)
    print("ROWS AROUND CONTACT TRANSITIONS")
    print("=" * 120)


    important_indices = set()


    for idx in [
        first_mid,
        first_tip,
    ]:

        if idx is None:
            continue

        for j in range(
            max(0, idx - 4),
            min(
                len(rows),
                idx + 7,
            ),
        ):

            important_indices.add(
                j
            )


    print()
    print(
        " ticks | disp mm |    MCP |    PIP |    DIP "
        "| Fprox   | Fmid    | Ftip    | other"
    )

    print(
        "-" * 120
    )


    for i in sorted(
        important_indices
    ):

        r = rows[i]

        print(
            f"{r['ticks']:6.0f} | "
            f"{r['disp_mm']:7.3f} | "
            f"{r['mcp']:6.2f} | "
            f"{r['pip']:6.2f} | "
            f"{r['dip']:6.2f} | "
            f"{r['prox_force']:7.4f} | "
            f"{r['mid_force']:7.4f} | "
            f"{r['tip_force']:7.4f} | "
            f"{r['other']}"
        )


    print()
    print(
        "CSV written to:"
    )

    print(
        csv_path
    )

    print()
    print("=" * 120)


if __name__ == "__main__":
    main()
