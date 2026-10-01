#!/usr/bin/env python3

import math
import mujoco
import numpy as np

import search_index_tabletop_xy_v1 as base


# ============================================================
# Small neighborhood around:
#
# along  = 30 mm
# normal = 50 mm
# ============================================================

ALONG_MM = [
    24.0,
    26.0,
    28.0,
    30.0,
    32.0,
    34.0,
    36.0,
]

NORMAL_MM = [
    46.0,
    48.0,
    50.0,
    52.0,
    54.0,
]


OPEN_TICKS = 100.0
MAX_TICKS = 1900.0
STEP_TICKS = 25.0

SETTLE_TIME = 0.060

FORCE_THRESHOLD = 0.005


def angle_deg(
    model,
    data,
    name,
):

    jid = base.oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return math.degrees(
        data.qpos[
            model.jnt_qposadr[
                jid
            ]
        ]
    )


def main():

    model = mujoco.MjModel.from_xml_path(
        str(
            base.MODEL_PATH
        )
    )

    data = mujoco.MjData(
        model
    )


    tendon_id = base.oid(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        base.TENDON_NAME,
    )

    model.tendon_stiffness[
        tendon_id
    ] = base.TENDON_STIFFNESS


    fixture_gid = base.oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        base.FIXTURE_NAME,
    )


    actuator = base.oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        base.INDEX_ACTUATOR,
    )


    # ========================================================
    # Recover the same tabletop coordinate system.
    # ========================================================

    base.configure_pose(
        model,
        data,
    )


    mcp_jid = base.oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        base.MCP_NAME,
    )

    pip_jid = base.oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        base.PIP_NAME,
    )


    mcp_world = (
        data.xanchor[
            mcp_jid
        ].copy()
    )

    pip_world = (
        data.xanchor[
            pip_jid
        ].copy()
    )


    forward_xy = (
        pip_world[:2]
        -
        mcp_world[:2]
    )

    forward_xy /= np.linalg.norm(
        forward_xy
    )


    normal_xy = np.array(
        [
            -forward_xy[1],
             forward_xy[0],
        ]
    )


    old_delta = (
        base.OLD_BOTTLE_XY
        -
        mcp_world[:2]
    )


    if np.dot(
        old_delta,
        normal_xy,
    ) < 0:

        normal_xy *= -1.0


    print()
    print("=" * 125)
    print(
        "LOCAL TABLETOP REDISTRIBUTION SEARCH V2"
    )
    print("=" * 125)

    print()
    print(
        "We require:"
    )

    print(
        "  no open contact"
    )

    print(
        "  no non-index contact"
    )

    print(
        "  middle contact before tip"
    )

    print(
        "  then inspect MCP/PIP/DIP redistribution"
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


            base.configure_pose(
                model,
                data,
            )


            base.settle(
                model,
                data,
                0.30,
            )


            open_forces = (
                base.bottle_contact_forces(
                    model,
                    data,
                    fixture_gid,
                )
            )


            rows = []

            first_mid = None
            first_tip = None
            first_other = None


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
                    * base.SCALE,
                )


                data.ctrl[
                    actuator
                ] = displacement


                base.settle(
                    model,
                    data,
                    SETTLE_TIME,
                )


                forces = (
                    base.bottle_contact_forces(
                        model,
                        data,
                        fixture_gid,
                    )
                )


                mid_active = (
                    forces.get(
                        "collision_idx_mid",
                        0.0,
                    )
                    >= FORCE_THRESHOLD
                )


                tip_active = (
                    forces.get(
                        "collision_idx_tip",
                        0.0,
                    )
                    >= FORCE_THRESHOLD
                )


                others = [
                    name
                    for name in forces
                    if not name.startswith(
                        "collision_idx_"
                    )
                ]


                row = {
                    "ticks":
                        float(ticks),

                    "mcp":
                        angle_deg(
                            model,
                            data,
                            base.MCP_NAME,
                        ),

                    "pip":
                        angle_deg(
                            model,
                            data,
                            base.PIP_NAME,
                        ),

                    "dip":
                        angle_deg(
                            model,
                            data,
                            base.DIP_NAME,
                        ),

                    "mid":
                        mid_active,

                    "tip":
                        tip_active,

                    "others":
                        others,
                }


                rows.append(
                    row
                )


                idx = len(
                    rows
                ) - 1


                if (
                    mid_active
                    and
                    first_mid is None
                ):

                    first_mid = idx


                if (
                    tip_active
                    and
                    first_tip is None
                ):

                    first_tip = idx


                if (
                    others
                    and
                    first_other is None
                ):

                    first_other = idx


            # =================================================
            # Evaluate middle -> tip interval.
            # =================================================

            topology_ok = (
                not open_forces

                and

                first_other is None

                and

                first_mid is not None

                and

                first_tip is not None

                and

                first_mid < first_tip
            )


            mid_duty = 0.0

            dmcp = None
            dpip = None
            ddip = None
            tick_gap = None


            if topology_ok:

                a = rows[
                    first_mid
                ]

                b = rows[
                    first_tip
                ]


                tick_gap = (
                    b["ticks"]
                    -
                    a["ticks"]
                )


                dmcp = (
                    b["mcp"]
                    -
                    a["mcp"]
                )

                dpip = (
                    b["pip"]
                    -
                    a["pip"]
                )

                ddip = (
                    b["dip"]
                    -
                    a["dip"]
                )


                segment = rows[
                    first_mid:
                    first_tip + 1
                ]


                mid_duty = float(
                    np.mean(
                        [
                            r["mid"]
                            for r in segment
                        ]
                    )
                )


            # =================================================
            # Score only after topology is satisfied.
            #
            # Prefer:
            #  - persistent middle contact
            #  - positive PIP redistribution
            #  - small MCP drift
            #
            # Do NOT fit final HW pose here.
            # =================================================

            if not topology_ok:

                score = 10000.0

            else:

                score = 0.0


                score += (
                    200.0
                    * (
                        1.0
                        - mid_duty
                    )
                )


                # Reward PIP flexion.
                score -= (
                    20.0
                    * dpip
                )


                # Penalize MCP motion after contact.
                score += (
                    10.0
                    * abs(
                        dmcp
                    )
                )


                # Weakly discourage DIP taking over before
                # distal contact.
                score += (
                    2.0
                    * max(
                        0.0,
                        ddip
                        - dpip
                    )
                )


            results.append(
                {
                    "along":
                        along_mm,

                    "normal":
                        normal_mm,

                    "xy":
                        bottle_xy.copy(),

                    "topology":
                        topology_ok,

                    "mid_tick":
                        (
                            None
                            if first_mid is None
                            else
                            rows[
                                first_mid
                            ][
                                "ticks"
                            ]
                        ),

                    "tip_tick":
                        (
                            None
                            if first_tip is None
                            else
                            rows[
                                first_tip
                            ][
                                "ticks"
                            ]
                        ),

                    "gap":
                        tick_gap,

                    "mid_duty":
                        mid_duty,

                    "dmcp":
                        dmcp,

                    "dpip":
                        dpip,

                    "ddip":
                        ddip,

                    "score":
                        score,
                }
            )


            def val(x):

                if x is None:
                    return "   ---"

                return f"{x:+6.2f}"


            print(
                f"along={along_mm:5.1f}  "
                f"normal={normal_mm:5.1f}  "
                f"topology={'YES' if topology_ok else ' no'}  "
                f"dMCP={val(dmcp)}  "
                f"dPIP={val(dpip)}  "
                f"dDIP={val(ddip)}  "
                f"mid-duty={100*mid_duty:5.1f}%"
            )


    ranked = sorted(
        results,
        key=lambda r:
            r["score"],
    )


    print()
    print("=" * 125)
    print(
        "TOP LOCAL REDISTRIBUTION CANDIDATES"
    )
    print("=" * 125)


    good = [
        r
        for r in ranked
        if r["topology"]
    ]


    if not good:

        print(
            "NONE"
        )

    else:

        for rank, r in enumerate(
            good[:15],
            start=1,
        ):

            print()
            print(
                f"#{rank} "
                f"along={r['along']:.1f} mm  "
                f"normal={r['normal']:.1f} mm"
            )

            print(
                "  bottle center = "
                f"{r['xy'][0]:.9f} "
                f"{r['xy'][1]:.9f} "
                "0.799000000"
            )

            print(
                f"  middle -> tip = "
                f"{r['mid_tick']:.0f} -> "
                f"{r['tip_tick']:.0f} ticks"
            )

            print(
                f"  gap = "
                f"{r['gap']:.0f} ticks"
            )

            print(
                f"  middle duty = "
                f"{100*r['mid_duty']:.1f}%"
            )

            print(
                f"  delta MCP = "
                f"{r['dmcp']:+.3f} deg"
            )

            print(
                f"  delta PIP = "
                f"{r['dpip']:+.3f} deg"
            )

            print(
                f"  delta DIP = "
                f"{r['ddip']:+.3f} deg"
            )

            print(
                f"  score = "
                f"{r['score']:.3f}"
            )


    print()
    print("=" * 125)
    print(
        "KEY QUESTION"
    )
    print("=" * 125)


    strong = [
        r
        for r in good
        if (
            r["dpip"] is not None

            and

            r["dpip"] >= 5.0

            and

            r["mid_duty"] >= 0.80
        )
    ]


    if strong:

        print(
            "At least one nearby geometry gives substantial "
            "PIP redistribution."
        )

        print(
            "Geometry is still an important variable."
        )

    else:

        print(
            "NO nearby geometry gives >= 5 deg PIP gain "
            "with persistent middle contact."
        )

        print(
            "This strongly points to the tendon/passive "
            "mechanics as the remaining mismatch."
        )


    print()
    print("=" * 125)


if __name__ == "__main__":
    main()
