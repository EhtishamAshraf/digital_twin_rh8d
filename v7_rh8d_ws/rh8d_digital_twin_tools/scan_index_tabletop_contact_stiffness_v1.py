#!/usr/bin/env python3

import math
import mujoco
import numpy as np

import search_index_tabletop_xy_v1 as base


# ============================================================
# Frozen tabletop geometry
# ============================================================

BOTTLE_POS = np.array(
    [
         0.503422950,
        -0.372517762,
         0.799000000,
    ],
    dtype=float,
)


# Best-case low sliding friction from previous scan.
MU = 0.05


# Old 28000 N/m was identified with invalid old geometry.
# Scan a broad range.
CONTACT_STIFFNESSES = [
      500.0,
     1000.0,
     2000.0,
     5000.0,
    10000.0,
    20000.0,
    28000.0,
]


CONTACT_DAMPING = 1.0

OPEN_TICKS = 100.0
MAX_TICKS = 2000.0
STEP_TICKS = 25.0

SETTLE_TIME = 0.060
FORCE_THRESHOLD = 0.005

TARGET_TICKS = 1029.0

HW_TARGET = np.array(
    [
        14.5355,
        23.8110,
         9.3270,
    ],
    dtype=float,
)


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

    return math.degrees(
        data.qpos[
            model.jnt_qposadr[jid]
        ]
    )


def run_one(k_contact):

    model = mujoco.MjModel.from_xml_path(
        str(base.MODEL_PATH)
    )

    data = mujoco.MjData(
        model
    )


    # ========================================================
    # Candidate-A tendon
    # ========================================================

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


    # ========================================================
    # Freeze friction, vary only contact stiffness.
    # Apply same parameters to both sides of contact.
    # ========================================================

    contact_geoms = [
        fixture_gid
    ]


    for name in INDEX_CONTACT_GEOMS:

        contact_geoms.append(
            base.oid(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
        )


    for gid in contact_geoms:

        model.geom_friction[
            gid
        ] = [
            MU,
            0.001,
            0.0001,
        ]

        model.geom_solref[
            gid
        ] = [
            -k_contact,
            -CONTACT_DAMPING,
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


    # Include exact hardware hold encoder in trajectory.
    ticks_array = list(
        np.arange(
            OPEN_TICKS,
            MAX_TICKS + 0.1,
            STEP_TICKS,
        )
    )

    ticks_array.append(
        TARGET_TICKS
    )

    ticks_array = sorted(
        set(
            float(x)
            for x in ticks_array
        )
    )


    rows = []

    first_mid = None
    first_tip = None
    first_other = None


    for ticks in ticks_array:

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
                ticks,

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


        idx = len(rows) - 1


        if (
            row["fmid"]
            >= FORCE_THRESHOLD
            and first_mid is None
        ):

            first_mid = idx


        if (
            row["ftip"]
            >= FORCE_THRESHOLD
            and first_tip is None
        ):

            first_tip = idx


        if (
            row["others"]
            and first_other is None
        ):

            first_other = idx


    target_row = min(
        rows,
        key=lambda r:
            abs(
                r["ticks"]
                - TARGET_TICKS
            ),
    )


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

        "k":
            k_contact,

        "topology":
            topology,

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

        "target":
            target_row,
    }


    if first_mid is not None:

        result["mid_tick"] = (
            rows[first_mid]["ticks"]
        )


    if first_tip is not None:

        result["tip_tick"] = (
            rows[first_tip]["ticks"]
        )


    if topology:

        a = rows[first_mid]
        b = rows[first_tip]


        result["dmcp"] = (
            b["mcp"]
            - a["mcp"]
        )

        result["dpip"] = (
            b["pip"]
            - a["pip"]
        )

        result["ddip"] = (
            b["dip"]
            - a["dip"]
        )


        segment = rows[
            first_mid:
            first_tip + 1
        ]


        result["mid_duty"] = float(
            np.mean(
                [
                    r["fmid"]
                    >= FORCE_THRESHOLD
                    for r in segment
                ]
            )
        )


        result["max_mid_force"] = max(
            r["fmid"]
            for r in segment
        )


        result["mid_force_at_tip"] = (
            b["fmid"]
        )

        result["tip_force_at_tip"] = (
            b["ftip"]
        )


    return result


def main():

    print()
    print("=" * 125)
    print(
        "INDEX TABLETOP CONTACT-STIFFNESS SENSITIVITY V1"
    )
    print("=" * 125)

    print()
    print(
        "Bottle =",
        BOTTLE_POS,
    )

    print(
        f"Sliding friction fixed at mu = {MU}"
    )

    print(
        "Only contact stiffness changes."
    )


    results = []


    for k in CONTACT_STIFFNESSES:

        r = run_one(
            k
        )

        results.append(
            r
        )


        t = r[
            "target"
        ]


        if r["dpip"] is None:

            dpip_text = "---"

        else:

            dpip_text = (
                f"{r['dpip']:+.3f}"
            )


        print()
        print(
            f"k={k:8.1f} N/m  "
            f"topology={'YES' if r['topology'] else ' no'}  "
            f"middle={r['mid_tick']}  "
            f"tip={r['tip_tick']}  "
            f"dPIP={dpip_text}"
        )

        print(
            f"  @1029 ticks: "
            f"MCP={t['mcp']:.3f}, "
            f"PIP={t['pip']:.3f}, "
            f"DIP={t['dip']:.3f}, "
            f"Fmid={t['fmid']:.4f}, "
            f"Ftip={t['ftip']:.4f}"
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


    for r in good:

        t = r[
            "target"
        ]


        target_pose = np.array(
            [
                t["mcp"],
                t["pip"],
                t["dip"],
            ]
        )


        err = (
            target_pose
            - HW_TARGET
        )


        print()
        print(
            f"k = {r['k']:.1f} N/m"
        )

        print(
            f"  middle -> tip = "
            f"{r['mid_tick']:.0f} -> "
            f"{r['tip_tick']:.0f} ticks"
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
            f"  max middle force = "
            f"{r['max_mid_force']:.4f} N"
        )

        print(
            f"  middle F @ tip = "
            f"{r['mid_force_at_tip']:.4f} N"
        )

        print(
            f"  tip F @ tip = "
            f"{r['tip_force_at_tip']:.4f} N"
        )

        print(
            f"  pose @ 1029 = "
            f"({t['mcp']:.3f}, "
            f"{t['pip']:.3f}, "
            f"{t['dip']:.3f}) deg"
        )

        print(
            f"  HW pose      = "
            f"({HW_TARGET[0]:.3f}, "
            f"{HW_TARGET[1]:.3f}, "
            f"{HW_TARGET[2]:.3f}) deg"
        )

        print(
            f"  error        = "
            f"({err[0]:+.3f}, "
            f"{err[1]:+.3f}, "
            f"{err[2]:+.3f}) deg"
        )


    print()
    print("=" * 125)
    print(
        "INTERPRETATION"
    )
    print("=" * 125)


    if not good:

        print(
            "No tested stiffness preserved "
            "middle-before-tip topology."
        )

    else:

        best = max(
            good,
            key=lambda r:
                r[
                    "dpip"
                ],
        )


        print(
            f"Largest PIP redistribution = "
            f"{best['dpip']:+.3f} deg"
        )

        print(
            f"at contact stiffness "
            f"{best['k']:.1f} N/m."
        )


        if best["dpip"] >= 5.0:

            print()
            print(
                "Contact compliance materially changes "
                "PIP redistribution."
            )

            print(
                "Do NOT change the hand transmission yet."
            )

        else:

            print()
            print(
                "Contact compliance also fails to produce "
                "the missing PIP motion."
            )

            print(
                "Bottle placement, sliding friction, and "
                "contact stiffness have now all been tested."
            )

            print(
                "The next identification stage should target "
                "PIP transmission/passive mechanics while "
                "simultaneously preserving the free-space fit."
            )


    print()
    print("=" * 125)


if __name__ == "__main__":
    main()
