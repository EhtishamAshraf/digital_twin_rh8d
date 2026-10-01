#!/usr/bin/env python3

import math
import mujoco
import numpy as np

import search_index_tabletop_xy_v1 as base


# ============================================================
# Freeze the best middle-first tabletop geometry
# ============================================================

BOTTLE_POS = np.array(
    [
         0.503422950,
        -0.372517762,
         0.799000000,
    ],
    dtype=float,
)


# ============================================================
# Test only interface sliding friction
# ============================================================

FRICTIONS = [
    0.02,
    0.05,
    0.10,
    0.20,
    0.40,
    0.60,
    0.80,
    1.20,
]


OPEN_TICKS = 100.0
MAX_TICKS = 2000.0
STEP_TICKS = 25.0

SETTLE_TIME = 0.060

FORCE_THRESHOLD = 0.005


INDEX_CONTACT_GEOMS = [
    "collision_idx_prox",
    "collision_idx_mid",
    "collision_idx_tip",
]


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

    qadr = model.jnt_qposadr[
        jid
    ]

    return math.degrees(
        data.qpos[
            qadr
        ]
    )


def run_one(
    mu,
):

    model = mujoco.MjModel.from_xml_path(
        str(
            base.MODEL_PATH
        )
    )

    data = mujoco.MjData(
        model
    )


    # --------------------------------------------------------
    # Candidate-A tendon stiffness
    # --------------------------------------------------------

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


    model.geom_pos[
        fixture_gid
    ] = BOTTLE_POS


    # --------------------------------------------------------
    # Set BOTH sides of the index/bottle interface to the
    # tested friction so the scan actually changes the pair.
    #
    # Torsional / rolling friction kept very small.
    # --------------------------------------------------------

    model.geom_friction[
        fixture_gid
    ] = [
        mu,
        0.001,
        0.0001,
    ]


    for geom_name in INDEX_CONTACT_GEOMS:

        gid = base.oid(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )

        model.geom_friction[
            gid
        ] = [
            mu,
            0.001,
            0.0001,
        ]


    actuator = base.oid(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        base.INDEX_ACTUATOR,
    )


    base.configure_pose(
        model,
        data,
    )


    base.settle(
        model,
        data,
        0.30,
    )


    open_forces = base.bottle_contact_forces(
        model,
        data,
        fixture_gid,
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


        forces = base.bottle_contact_forces(
            model,
            data,
            fixture_gid,
        )


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

            "fprox":
                forces.get(
                    "collision_idx_prox",
                    0.0,
                ),

            "fmid":
                forces.get(
                    "collision_idx_mid",
                    0.0,
                ),

            "ftip":
                forces.get(
                    "collision_idx_tip",
                    0.0,
                ),

            "others":
                [
                    n
                    for n in forces
                    if not n.startswith(
                        "collision_idx_"
                    )
                ],
        }


        rows.append(
            row
        )


        idx = len(
            rows
        ) - 1


        if (
            row["fmid"]
            >= FORCE_THRESHOLD
            and
            first_mid is None
        ):

            first_mid = idx


        if (
            row["ftip"]
            >= FORCE_THRESHOLD
            and
            first_tip is None
        ):

            first_tip = idx


        if (
            row["others"]
            and
            first_other is None
        ):

            first_other = idx


    topology = (
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


    result = {

        "mu":
            mu,

        "topology":
            topology,

        "open_forces":
            open_forces,

        "mid_tick":
            None,

        "tip_tick":
            None,

        "mid_duty":
            0.0,

        "dmcp":
            None,

        "dpip":
            None,

        "ddip":
            None,

        "max_mid_force":
            0.0,

        "mid_force_at_tip":
            0.0,

        "tip_force_at_tip":
            0.0,
    }


    if (
        first_mid is not None
    ):

        result[
            "mid_tick"
        ] = rows[
            first_mid
        ][
            "ticks"
        ]


    if (
        first_tip is not None
    ):

        result[
            "tip_tick"
        ] = rows[
            first_tip
        ][
            "ticks"
        ]


    if topology:

        a = rows[
            first_mid
        ]

        b = rows[
            first_tip
        ]


        result[
            "dmcp"
        ] = (
            b["mcp"]
            -
            a["mcp"]
        )


        result[
            "dpip"
        ] = (
            b["pip"]
            -
            a["pip"]
        )


        result[
            "ddip"
        ] = (
            b["dip"]
            -
            a["dip"]
        )


        segment = rows[
            first_mid:
            first_tip + 1
        ]


        result[
            "mid_duty"
        ] = float(
            np.mean(
                [
                    r["fmid"]
                    >= FORCE_THRESHOLD
                    for r in segment
                ]
            )
        )


        result[
            "max_mid_force"
        ] = max(
            r["fmid"]
            for r in segment
        )


        result[
            "mid_force_at_tip"
        ] = b[
            "fmid"
        ]


        result[
            "tip_force_at_tip"
        ] = b[
            "ftip"
        ]


    return result


def main():

    print()
    print("=" * 125)
    print(
        "INDEX TABLETOP FRICTION SENSITIVITY V1"
    )
    print("=" * 125)

    print()
    print(
        "Frozen bottle =",
        BOTTLE_POS,
    )

    print(
        "Only sliding friction changes."
    )

    print()


    results = []


    for mu in FRICTIONS:

        r = run_one(
            mu
        )

        results.append(
            r
        )


        def val(
            x,
        ):

            if x is None:

                return "   ---"

            return f"{x:+7.3f}"


        print(
            f"mu={mu:4.2f}  "
            f"topology={'YES' if r['topology'] else ' no'}  "
            f"mid={str(r['mid_tick']):>6s}  "
            f"tip={str(r['tip_tick']):>6s}  "
            f"dMCP={val(r['dmcp'])}  "
            f"dPIP={val(r['dpip'])}  "
            f"dDIP={val(r['ddip'])}  "
            f"mid-duty={100*r['mid_duty']:5.1f}%  "
            f"Fmid,max={r['max_mid_force']:7.3f}"
        )


    good = [
        r
        for r in results
        if r[
            "topology"
        ]
    ]


    print()
    print("=" * 125)
    print(
        "VALID MIDDLE-FIRST RESULTS"
    )
    print("=" * 125)


    if not good:

        print(
            "NONE"
        )

    else:

        for r in good:

            print()
            print(
                f"friction mu = "
                f"{r['mu']:.3f}"
            )

            print(
                f"  middle contact = "
                f"{r['mid_tick']:.0f} ticks"
            )

            print(
                f"  tip contact    = "
                f"{r['tip_tick']:.0f} ticks"
            )

            print(
                f"  middle duty    = "
                f"{100*r['mid_duty']:.1f}%"
            )

            print(
                f"  delta MCP      = "
                f"{r['dmcp']:+.3f} deg"
            )

            print(
                f"  delta PIP      = "
                f"{r['dpip']:+.3f} deg"
            )

            print(
                f"  delta DIP      = "
                f"{r['ddip']:+.3f} deg"
            )

            print(
                f"  max middle F   = "
                f"{r['max_mid_force']:.4f} N"
            )

            print(
                f"  middle F @ tip = "
                f"{r['mid_force_at_tip']:.4f} N"
            )

            print(
                f"  tip F @ tip    = "
                f"{r['tip_force_at_tip']:.4f} N"
            )


    print()
    print("=" * 125)
    print(
        "INTERPRETATION"
    )
    print("=" * 125)


    if good:

        best_pip = max(
            good,
            key=lambda r:
                r[
                    "dpip"
                ],
        )


        print()
        print(
            f"Largest PIP redistribution = "
            f"{best_pip['dpip']:+.3f} deg"
        )

        print(
            f"at friction mu = "
            f"{best_pip['mu']:.3f}"
        )


        if (
            best_pip[
                "dpip"
            ]
            >= 5.0
        ):

            print()
            print(
                "Contact friction CAN materially change "
                "the missing PIP redistribution."
            )

            print(
                "Do not modify tendon/passive mechanics yet."
            )

        else:

            print()
            print(
                "Even low interface friction does NOT produce "
                "substantial PIP redistribution."
            )

            print(
                "The remaining mismatch is therefore much more "
                "likely in the tendon/passive joint mechanics."
            )


    print()
    print("=" * 125)


if __name__ == "__main__":
    main()
