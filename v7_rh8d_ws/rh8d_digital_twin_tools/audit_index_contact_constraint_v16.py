#!/usr/bin/env python3

from pathlib import Path
import csv
import math

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as base
import compare_index_configdep_replay_v6 as ab


# ============================================================
# V16 — CONTACT-CONSTRAINT AUDIT
#
# Diagnostic only.
#
# Trusted constant tendon + frozen V5 fixture.
#
# Questions:
#
# 1. Which link actually contacts the cylinder first?
# 2. Is the proximal phalanx becoming load-bearing?
# 3. How much penetration occurs?
# 4. What normal forces are generated?
# 5. At the video-aligned hold, is MCP actually constrained?
#
# No parameter fitting.
# No joint angles imposed.
# ============================================================


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()

MJCF_DIR = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

MODEL_PATH = (
    MJCF_DIR
    / "tmp_v16_trusted_contact.xml"
)

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "contact_constraint_v16"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    RESULT_DIR
    / "contact_timeseries.csv"
)


SCALE = 5.10e-6
TENDON_K = 2100.0

OPEN_ENCODER = 100.0

HOLD_ELAPSED = 6.297


MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

SPOOL = "index_motor_spool"

BOTTLE = "index_contact_fixture"

PROX = "collision_idx_prox"
MID = "collision_idx_mid"
TIP = "collision_idx_tip"


# ============================================================
# HELPERS
# ============================================================

def obj_id(
    model,
    objtype,
    name,
):

    idx = mujoco.mj_name2id(
        model,
        objtype,
        name,
    )

    if idx < 0:

        raise RuntimeError(
            f"Missing MuJoCo object: {name}"
        )

    return idx


def joint_qadr(
    model,
    name,
):

    jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return int(
        model.jnt_qposadr[
            jid
        ]
    )


def geom_signed_distance(
    model,
    data,
    g1,
    g2,
):

    fromto = np.zeros(
        6,
        dtype=float,
    )

    d = mujoco.mj_geomDistance(
        model,
        data,
        g1,
        g2,
        1.0,
        fromto,
    )

    return float(
        d
    )


# ============================================================
# RECORDER
# ============================================================

class ContactRecorder:

    def __init__(self):

        self.original_step = (
            mujoco.mj_step
        )

        self.rows = []

        self.cache = {}


    def ids(
        self,
        model,
    ):

        key = id(
            model
        )


        if key in self.cache:

            return self.cache[
                key
            ]


        try:

            result = {

                "mcp_q":
                    joint_qadr(
                        model,
                        MCP,
                    ),

                "pip_q":
                    joint_qadr(
                        model,
                        PIP,
                    ),

                "dip_q":
                    joint_qadr(
                        model,
                        DIP,
                    ),

                "spool_q":
                    joint_qadr(
                        model,
                        SPOOL,
                    ),

                "bottle":
                    obj_id(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        BOTTLE,
                    ),

                "prox":
                    obj_id(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        PROX,
                    ),

                "mid":
                    obj_id(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        MID,
                    ),

                "tip":
                    obj_id(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        TIP,
                    ),
            }


        except RuntimeError:

            result = None


        self.cache[
            key
        ] = result

        return result


    def contact_force_for_geom(
        self,
        model,
        data,
        bottle,
        geom,
    ):

        total_normal = 0.0

        min_contact_dist = None

        count = 0


        for i in range(
            data.ncon
        ):

            c = data.contact[
                i
            ]


            pair = {
                int(
                    c.geom1
                ),
                int(
                    c.geom2
                ),
            }


            if (
                bottle not in pair
                or
                geom not in pair
            ):

                continue


            count += 1


            dist = float(
                c.dist
            )


            if (
                min_contact_dist is None
                or
                dist
                < min_contact_dist
            ):

                min_contact_dist = (
                    dist
                )


            force = np.zeros(
                6,
                dtype=float,
            )


            mujoco.mj_contactForce(
                model,
                data,
                i,
                force,
            )


            # First component is normal force in the
            # local contact frame.
            total_normal += abs(
                float(
                    force[
                        0
                    ]
                )
            )


        return (
            count,
            min_contact_dist,
            total_normal,
        )


    def record(
        self,
        model,
        data,
    ):

        ids = self.ids(
            model
        )


        if ids is None:

            return


        # Geometric distance is recorded whether or not the
        # solver has instantiated a contact row.
        prox_geom_dist = (
            geom_signed_distance(
                model,
                data,
                ids[
                    "prox"
                ],
                ids[
                    "bottle"
                ],
            )
        )


        mid_geom_dist = (
            geom_signed_distance(
                model,
                data,
                ids[
                    "mid"
                ],
                ids[
                    "bottle"
                ],
            )
        )


        tip_geom_dist = (
            geom_signed_distance(
                model,
                data,
                ids[
                    "tip"
                ],
                ids[
                    "bottle"
                ],
            )
        )


        (
            prox_n,
            prox_contact_dist,
            prox_force,
        ) = self.contact_force_for_geom(
            model,
            data,
            ids[
                "bottle"
            ],
            ids[
                "prox"
            ],
        )


        (
            mid_n,
            mid_contact_dist,
            mid_force,
        ) = self.contact_force_for_geom(
            model,
            data,
            ids[
                "bottle"
            ],
            ids[
                "mid"
            ],
        )


        (
            tip_n,
            tip_contact_dist,
            tip_force,
        ) = self.contact_force_for_geom(
            model,
            data,
            ids[
                "bottle"
            ],
            ids[
                "tip"
            ],
        )


        spool = float(
            data.qpos[
                ids[
                    "spool_q"
                ]
            ]
        )


        self.rows.append(
            {

                "time":
                    float(
                        data.time
                    ),

                "encoder_est":
                    OPEN_ENCODER
                    +
                    spool
                    / SCALE,

                "spool_mm":
                    spool
                    * 1000.0,

                "mcp_deg":
                    math.degrees(
                        float(
                            data.qpos[
                                ids[
                                    "mcp_q"
                                ]
                            ]
                        )
                    ),

                "pip_deg":
                    math.degrees(
                        float(
                            data.qpos[
                                ids[
                                    "pip_q"
                                ]
                            ]
                        )
                    ),

                "dip_deg":
                    math.degrees(
                        float(
                            data.qpos[
                                ids[
                                    "dip_q"
                                ]
                            ]
                        )
                    ),

                "prox_distance_mm":
                    prox_geom_dist
                    * 1000.0,

                "mid_distance_mm":
                    mid_geom_dist
                    * 1000.0,

                "tip_distance_mm":
                    tip_geom_dist
                    * 1000.0,

                "prox_contact_rows":
                    prox_n,

                "mid_contact_rows":
                    mid_n,

                "tip_contact_rows":
                    tip_n,

                "prox_normal_force_N":
                    prox_force,

                "mid_normal_force_N":
                    mid_force,

                "tip_normal_force_N":
                    tip_force,

                "prox_contact_dist_mm":
                    (
                        prox_contact_dist
                        * 1000.0
                        if prox_contact_dist
                        is not None
                        else None
                    ),

                "mid_contact_dist_mm":
                    (
                        mid_contact_dist
                        * 1000.0
                        if mid_contact_dist
                        is not None
                        else None
                    ),

                "tip_contact_dist_mm":
                    (
                        tip_contact_dist
                        * 1000.0
                        if tip_contact_dist
                        is not None
                        else None
                    ),
            }
        )


    def install(
        self,
    ):

        original = (
            self.original_step
        )

        recorder = self


        def wrapped(
            model,
            data,
            *args,
            **kwargs,
        ):

            result = original(
                model,
                data,
                *args,
                **kwargs,
            )


            recorder.record(
                model,
                data,
            )


            return result


        mujoco.mj_step = wrapped


    def restore(
        self,
    ):

        mujoco.mj_step = (
            self.original_step
        )


# ============================================================
# REPORT HELPERS
# ============================================================

def first_where(
    rows,
    predicate,
):

    for r in rows:

        if predicate(
            r
        ):

            return r

    return None


def closest_elapsed(
    rows,
    target,
):

    if not rows:

        return None


    t0 = rows[
        0
    ][
        "time"
    ]


    return min(
        rows,
        key=lambda r:
            abs(
                (
                    r[
                        "time"
                    ]
                    - t0
                )
                - target
            ),
    )


def print_event(
    label,
    row,
    t0,
):

    print()
    print(
        label
    )

    print(
        "-" * 75
    )


    if row is None:

        print(
            "  not detected"
        )

        return


    print(
        f"  elapsed = "
        f"{row['time']-t0:.3f} s"
    )

    print(
        f"  encoder = "
        f"{row['encoder_est']:.1f} ticks"
    )

    print(
        "  pose MCP/PIP/DIP = "
        f"{row['mcp_deg']:.3f} / "
        f"{row['pip_deg']:.3f} / "
        f"{row['dip_deg']:.3f} deg"
    )

    print(
        "  geometric gaps prox/mid/tip = "
        f"{row['prox_distance_mm']:+.3f} / "
        f"{row['mid_distance_mm']:+.3f} / "
        f"{row['tip_distance_mm']:+.3f} mm"
    )

    print(
        "  normal forces prox/mid/tip = "
        f"{row['prox_normal_force_N']:.3f} / "
        f"{row['mid_normal_force_N']:.3f} / "
        f"{row['tip_normal_force_N']:.3f} N"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    import sys


    if len(
        sys.argv
    ) != 2:

        print(
            "Usage:\n"
            " python3 "
            "audit_index_contact_constraint_v16.py "
            "/path/to/contact_bag"
        )

        raise SystemExit(
            1
        )


    contact_bag = Path(
        sys.argv[
            1
        ]
    ).expanduser().resolve()


    if not contact_bag.exists():

        raise RuntimeError(
            f"Missing contact bag:\n"
            f"{contact_bag}"
        )


    # Frozen V5 fixture + trusted constant tendon.
    ab.replace_index_with_constant_tendon(
        ab.V5_SOURCE,
        MODEL_PATH,
    )


    ab.patch_replay_modules()


    # --------------------------------------------------------
    # Print compiled contact properties before replay.
    # --------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(
            MODEL_PATH
        )
    )


    print()
    print("=" * 110)
    print(
        "V16 COMPILED CONTACT PARAMETERS"
    )
    print("=" * 110)


    for name in [
        BOTTLE,
        PROX,
        MID,
        TIP,
    ]:

        gid = obj_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )


        print()
        print(
            name
        )

        print(
            f"  contype/conaffinity = "
            f"{model.geom_contype[gid]} / "
            f"{model.geom_conaffinity[gid]}"
        )

        print(
            f"  margin = "
            f"{model.geom_margin[gid]*1000.0:.3f} mm"
        )

        print(
            f"  gap = "
            f"{model.geom_gap[gid]*1000.0:.3f} mm"
        )

        print(
            f"  solref = "
            f"{model.geom_solref[gid]}"
        )

        print(
            f"  solimp = "
            f"{model.geom_solimp[gid]}"
        )

        print(
            f"  friction = "
            f"{model.geom_friction[gid]}"
        )


    # --------------------------------------------------------
    # Replay
    # --------------------------------------------------------

    recorder = ContactRecorder()

    recorder.install()


    old_scale = (
        base.METERS_PER_TICK
    )

    old_k = (
        base.TENDON_STIFFNESS
    )


    try:

        base.METERS_PER_TICK = (
            SCALE
        )

        base.TENDON_STIFFNESS = (
            TENDON_K
        )


        print()
        print("=" * 110)
        print(
            "RUNNING TRUSTED CONTACT REPLAY"
        )
        print("=" * 110)


        metrics, _ = (
            base.run_contact(
                MODEL_PATH,
                contact_bag,
            )
        )


        pose = np.asarray(
            metrics[
                "pose"
            ],
            dtype=float,
        )


        print()
        print(
            f"Replay contact pose = "
            f"("
            f"{pose[0]:.3f}, "
            f"{pose[1]:.3f}, "
            f"{pose[2]:.3f}"
            f") deg"
        )

        print(
            f"Replay contact RMSE = "
            f"{metrics['rmse']:.3f} deg"
        )


    finally:

        recorder.restore()

        base.METERS_PER_TICK = (
            old_scale
        )

        base.TENDON_STIFFNESS = (
            old_k
        )


    rows = recorder.rows


    if not rows:

        raise RuntimeError(
            "No contact replay samples recorded."
        )


    t0 = rows[
        0
    ][
        "time"
    ]


    for r in rows:

        r[
            "elapsed"
        ] = (
            r[
                "time"
            ]
            - t0
        )


    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # --------------------------------------------------------
    # Event extraction
    # --------------------------------------------------------

    prox_first = first_where(
        rows,
        lambda r:
            r[
                "prox_contact_rows"
            ]
            > 0
            and
            r[
                "prox_normal_force_N"
            ]
            > 1.0e-4,
    )


    mid_first = first_where(
        rows,
        lambda r:
            r[
                "mid_contact_rows"
            ]
            > 0
            and
            r[
                "mid_normal_force_N"
            ]
            > 1.0e-4,
    )


    tip_first = first_where(
        rows,
        lambda r:
            r[
                "tip_contact_rows"
            ]
            > 0
            and
            r[
                "tip_normal_force_N"
            ]
            > 1.0e-4,
    )


    hold = closest_elapsed(
        rows,
        HOLD_ELAPSED,
    )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "V16 FIRST CONTACT EVENTS"
    )
    print("=" * 110)


    print_event(
        "PROXIMAL",
        prox_first,
        t0,
    )


    print_event(
        "MIDDLE",
        mid_first,
        t0,
    )


    print_event(
        "TIP",
        tip_first,
        t0,
    )


    # Determine sequence.
    sequence = []


    for name, event in [
        (
            "PROX",
            prox_first,
        ),
        (
            "MID",
            mid_first,
        ),
        (
            "TIP",
            tip_first,
        ),
    ]:

        if event is not None:

            sequence.append(
                (
                    event[
                        "time"
                    ],
                    name,
                )
            )


    sequence.sort()


    print()
    print(
        "Contact sequence:"
    )

    if sequence:

        print(
            "  "
            +
            " -> ".join(
                name
                for _, name
                in sequence
            )
        )

    else:

        print(
            "  none detected"
        )


    print()
    print("=" * 110)
    print(
        "V16 VIDEO-ALIGNED HOLD"
    )
    print("=" * 110)


    print_event(
        f"HOLD (~{HOLD_ELAPSED:.3f} s)",
        hold,
        t0,
    )


    print()
    print(
        "Hardware-aligned geometric reference from V5:"
    )

    print(
        "  proximal = +5.483 mm"
    )

    print(
        "  middle   = -0.011 mm"
    )

    print(
        "  tip      = -0.015 mm"
    )


    # --------------------------------------------------------
    # Max penetration / force
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "V16 CONTACT SEVERITY"
    )
    print("=" * 110)


    for prefix, label in [
        (
            "prox",
            "PROXIMAL",
        ),
        (
            "mid",
            "MIDDLE",
        ),
        (
            "tip",
            "TIP",
        ),
    ]:

        distances = np.asarray(
            [
                r[
                    f"{prefix}_distance_mm"
                ]
                for r in rows
            ],
            dtype=float,
        )


        forces = np.asarray(
            [
                r[
                    f"{prefix}_normal_force_N"
                ]
                for r in rows
            ],
            dtype=float,
        )


        min_distance = float(
            np.min(
                distances
            )
        )


        max_penetration = max(
            0.0,
            -min_distance,
        )


        print()
        print(
            label
        )

        print(
            f"  minimum signed gap = "
            f"{min_distance:+.3f} mm"
        )

        print(
            f"  maximum penetration = "
            f"{max_penetration:.3f} mm"
        )

        print(
            f"  maximum normal force = "
            f"{np.max(forces):.3f} N"
        )


    # --------------------------------------------------------
    # Decision aid
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "V16 DECISION AID"
    )
    print("=" * 110)


    if hold is not None:

        prox_bad = (
            hold[
                "prox_distance_mm"
            ]
            < -0.5
        )

        penetration_bad = (
            min(
                hold[
                    "prox_distance_mm"
                ],
                hold[
                    "mid_distance_mm"
                ],
                hold[
                    "tip_distance_mm"
                ],
            )
            < -1.0
        )


        if (
            prox_bad
            or
            penetration_bad
        ):

            print()
            print(
                "The trusted dynamic replay is not enforcing "
                "the hardware-aligned contact geometry closely."
            )

            print()
            print(
                "NEXT: identify/fix the contact constraint "
                "before adding further tendon hysteresis."
            )


        else:

            print()
            print(
                "Contact geometry is being enforced closely "
                "enough that the remaining error is likely "
                "inside the tendon/passive-joint mechanics."
            )


    print()
    print(
        "Timeseries:"
    )

    print(
        OUTPUT_CSV
    )


    print()
    print("=" * 110)


    try:
        MODEL_PATH.unlink()
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
