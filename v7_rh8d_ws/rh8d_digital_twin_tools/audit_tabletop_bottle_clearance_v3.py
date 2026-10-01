#!/usr/bin/env python3

from pathlib import Path
import math

import mujoco
import numpy as np


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()

MODEL_PATH = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_configdep_geom_v3.xml"
)


ARM_STARTUP = {
    "shoulder_pan_joint":   1.4927,
    "shoulder_lift_joint": -0.8678,
    "elbow_joint":         -2.4300,
    "wrist_1_joint":       -0.6596,
    "wrist_2_joint":       -0.0347,
    "wrist_3_joint":        0.1260,
}


MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"


STATES = {
    "OPEN":
        (0.0, 0.0, 0.0),

    "HW_CONTACT":
        (14.5355, 23.8110, 9.3270),
}


def get_joint_adr(model, name):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(
            f"Missing joint: {name}"
        )

    return int(
        model.jnt_qposadr[jid]
    )


def geom_name(model, gid):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        gid,
    )

    return (
        name
        if name is not None
        else "<unnamed>"
    )


def body_name(model, bid):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        bid,
    )

    return (
        name
        if name is not None
        else "<unnamed>"
    )


def set_state(
    model,
    data,
    qpos0,
    finger_adrs,
    finger_deg,
):

    data.qpos[:] = qpos0
    data.qvel[:] = 0.0


    for name, value in ARM_STARTUP.items():

        adr = get_joint_adr(
            model,
            name,
        )

        data.qpos[adr] = value


    for adr, deg in zip(
        finger_adrs,
        finger_deg,
    ):

        data.qpos[adr] = math.radians(
            deg
        )


    mujoco.mj_forward(
        model,
        data,
    )


def geom_distance(
    model,
    data,
    g1,
    g2,
):

    fromto = np.zeros(
        6,
        dtype=float,
    )

    try:

        distance = mujoco.mj_geomDistance(
            model,
            data,
            g1,
            g2,
            1.0,
            fromto,
        )

        return (
            float(distance),
            fromto,
        )

    except Exception as exc:

        return (
            None,
            str(exc),
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

    qpos0 = model.qpos0.copy()


    finger_adrs = [
        get_joint_adr(
            model,
            MCP,
        ),
        get_joint_adr(
            model,
            PIP,
        ),
        get_joint_adr(
            model,
            DIP,
        ),
    ]


    # ========================================================
    # Locate the physical bottle cylinder automatically.
    #
    # Expected:
    # radius ≈ 34.6 mm
    # half height ≈ 75 mm
    # ========================================================

    candidates = []


    for gid in range(
        model.ngeom
    ):

        if (
            int(model.geom_type[gid])
            != int(
                mujoco.mjtGeom.mjGEOM_CYLINDER
            )
        ):
            continue


        radius = float(
            model.geom_size[gid, 0]
        )

        halfheight = float(
            model.geom_size[gid, 1]
        )


        if (
            abs(radius - 0.0346)
            < 0.006
            and
            abs(halfheight - 0.075)
            < 0.035
        ):

            candidates.append(
                gid
            )


    print()
    print("=" * 105)
    print(
        "BOTTLE CANDIDATES"
    )
    print("=" * 105)


    if not candidates:

        print(
            "No ~69.2-mm diameter cylinder found."
        )

        return


    for gid in candidates:

        bid = int(
            model.geom_bodyid[gid]
        )

        print()
        print(
            f"geom id    = {gid}"
        )

        print(
            f"name       = {geom_name(model, gid)}"
        )

        print(
            f"body       = {body_name(model, bid)}"
        )

        print(
            f"local pos  = {model.geom_pos[gid]}"
        )

        print(
            f"size       = {model.geom_size[gid]}"
        )


    # Pick closest match to known bottle dimensions.
    bottle_gid = min(
        candidates,
        key=lambda gid:
            (
                abs(
                    float(
                        model.geom_size[
                            gid,
                            0
                        ]
                    )
                    - 0.0346
                )
                +
                abs(
                    float(
                        model.geom_size[
                            gid,
                            1
                        ]
                    )
                    - 0.075
                )
            )
    )


    palm_gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "collision_palm",
    )


    if palm_gid < 0:

        raise RuntimeError(
            "collision_palm not found"
        )


    index_names = [
        "collision_idx_prox",
        "collision_idx_mid",
        "collision_idx_tip",
    ]


    index_gids = []

    for name in index_names:

        gid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )

        if gid >= 0:

            index_gids.append(
                (
                    name,
                    gid,
                )
            )


    print()
    print("=" * 105)
    print(
        "SELECTED BOTTLE"
    )
    print("=" * 105)

    print(
        f"name = "
        f"{geom_name(model, bottle_gid)}"
    )


    # ========================================================
    # Evaluate OPEN and hardware-contact geometry
    # ========================================================

    for label, q_deg in STATES.items():

        set_state(
            model,
            data,
            qpos0,
            finger_adrs,
            q_deg,
        )


        print()
        print("=" * 105)
        print(label)
        print("=" * 105)


        print(
            "Bottle world center = "
            f"{data.geom_xpos[bottle_gid]}"
        )

        print(
            "Palm collision center = "
            f"{data.geom_xpos[palm_gid]}"
        )


        distance, nearest = geom_distance(
            model,
            data,
            palm_gid,
            bottle_gid,
        )


        print()
        print(
            "PALM -> BOTTLE"
        )


        if distance is None:

            print(
                "  distance unavailable:"
            )

            print(
                nearest
            )

        else:

            print(
                f"  signed distance = "
                f"{distance*1000.0:+.3f} mm"
            )

            if distance < 0:

                print(
                    "  *** PENETRATION ***"
                )

            else:

                print(
                    "  physical collision gap "
                    "(for collision_palm)"
                )

            print(
                "  nearest points = "
                f"{nearest}"
            )


        print()
        print(
            "INDEX -> BOTTLE"
        )


        for name, gid in index_gids:

            distance, _ = geom_distance(
                model,
                data,
                gid,
                bottle_gid,
            )


            if distance is None:

                print(
                    f"  {name:<22} unavailable"
                )

            else:

                print(
                    f"  {name:<22} "
                    f"{distance*1000.0:+8.3f} mm"
                )


        # ----------------------------------------------------
        # Actual active contacts involving the bottle
        # ----------------------------------------------------

        print()
        print(
            "ACTIVE CONTACTS WITH BOTTLE"
        )


        found = False


        for i in range(
            data.ncon
        ):

            c = data.contact[i]


            if (
                c.geom1 != bottle_gid
                and
                c.geom2 != bottle_gid
            ):

                continue


            found = True


            other = (
                c.geom2
                if c.geom1 == bottle_gid
                else c.geom1
            )


            print(
                f"  {geom_name(model, other):<28} "
                f"dist={c.dist*1000.0:+8.3f} mm"
            )


        if not found:

            print(
                "  none"
            )


    # ========================================================
    # Simple XY direction from palm toward bottle.
    #
    # We will use this only AFTER reviewing the distances.
    # ========================================================

    set_state(
        model,
        data,
        qpos0,
        finger_adrs,
        STATES["OPEN"],
    )


    delta_xy = (
        data.geom_xpos[
            bottle_gid,
            :2
        ]
        -
        data.geom_xpos[
            palm_gid,
            :2
        ]
    )


    norm = np.linalg.norm(
        delta_xy
    )


    print()
    print("=" * 105)
    print(
        "PALM -> BOTTLE XY DIRECTION"
    )
    print("=" * 105)


    if norm > 1e-9:

        direction = (
            delta_xy
            / norm
        )

        print(
            f"unit XY = "
            f"[{direction[0]:+.6f}, "
            f"{direction[1]:+.6f}]"
        )

    else:

        print(
            "centers coincide in XY"
        )


    print()
    print("=" * 105)


if __name__ == "__main__":
    main()
