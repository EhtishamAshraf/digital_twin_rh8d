#!/usr/bin/env python3

from pathlib import Path
import argparse
import math

import mujoco
import numpy as np

import scan_index_stateful_transmission_v24 as v24


# ============================================================
# V27 — WRAP GEOMETRY / CONTACT-PATH AUDIT
#
# Diagnostic only.
#
# No fitting.
# No parameter changes.
# No artificial compliant pad.
#
# Uses frozen V25 architecture:
#
#   pre-yield  = 5 / 4 / 1 mm
#   post-yield = 2.5 / 13 / 1.8 mm
#
# Goal:
#
# Determine whether loss of middle contact comes from:
#
#   A) incorrect contact geometry / registration
#
# or
#
#   B) dynamic separation despite geometry that supports
#      simultaneous middle + distal contact.
# ============================================================


WS = v24.WS

MJCF_DIR = v24.MJCF_DIR


MODEL_PATH = (
    MJCF_DIR
    / "tmp_v27_wrap_geometry_audit.xml"
)


V25_CASE = {

    "name":
        "V25_FROZEN",

    "yield":
        True,

    "arms_mm":
        [
            2.5,
            13.0,
            1.8,
        ],
}


HW_TARGET = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


V25_PREDICTION = np.array(
    [
        14.918,
        24.219,
        8.862,
    ],
    dtype=float,
)


MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

MID = "collision_idx_mid"
TIP = "collision_idx_tip"
BOTTLE = "index_contact_fixture"


# ============================================================
# GEOMETRY DISTANCE
# ============================================================

def geom_distance(
    model,
    data,
    gid1,
    gid2,
):

    fromto = np.zeros(
        6,
        dtype=float,
    )


    return float(
        mujoco.mj_geomDistance(
            model,
            data,
            gid1,
            gid2,
            0.10,
            fromto,
        )
    )


# ============================================================
# RECORDER
# ============================================================

class Recorder:

    def __init__(self):

        self.original_step = (
            mujoco.mj_step
        )

        self.rows = []

        self.cache = {}

        self.first_qpos = None
        self.first_geom_pos = None
        self.first_body_pos = None


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


        def jid(name):

            j = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            if j < 0:

                raise RuntimeError(
                    f"Missing joint: {name}"
                )


            return {

                "jid":
                    j,

                "qadr":
                    int(
                        model.jnt_qposadr[
                            j
                        ]
                    ),

                "dof":
                    int(
                        model.jnt_dofadr[
                            j
                        ]
                    ),
            }


        def gid(name):

            g = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )

            if g < 0:

                raise RuntimeError(
                    f"Missing geom: {name}"
                )

            return g


        result = {

            "mcp":
                jid(
                    MCP
                ),

            "pip":
                jid(
                    PIP
                ),

            "dip":
                jid(
                    DIP
                ),

            "mid":
                gid(
                    MID
                ),

            "tip":
                gid(
                    TIP
                ),

            "bottle":
                gid(
                    BOTTLE
                ),
        }


        self.cache[
            key
        ] = result

        return result


    def force_for_pair(
        self,
        model,
        data,
        ga,
        gb,
    ):

        total = 0.0

        rows = 0


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


            if pair != {
                ga,
                gb,
            }:

                continue


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


            total += abs(
                float(
                    force[
                        0
                    ]
                )
            )

            rows += 1


        return (
            total,
            rows,
        )


    def record(
        self,
        model,
        data,
    ):

        ids = self.ids(
            model
        )


        if (
            self.first_qpos
            is None
        ):

            self.first_qpos = (
                data.qpos.copy()
            )

            self.first_geom_pos = (
                model.geom_pos.copy()
            )

            self.first_body_pos = (
                model.body_pos.copy()
            )


        mid_gap = (
            geom_distance(
                model,
                data,
                ids[
                    "mid"
                ],
                ids[
                    "bottle"
                ],
            )
            *
            1000.0
        )


        tip_gap = (
            geom_distance(
                model,
                data,
                ids[
                    "tip"
                ],
                ids[
                    "bottle"
                ],
            )
            *
            1000.0
        )


        (
            mid_force,
            mid_rows,
        ) = self.force_for_pair(
            model,
            data,
            ids[
                "mid"
            ],
            ids[
                "bottle"
            ],
        )


        (
            tip_force,
            tip_rows,
        ) = self.force_for_pair(
            model,
            data,
            ids[
                "tip"
            ],
            ids[
                "bottle"
            ],
        )


        self.rows.append(
            {

                "time":
                    float(
                        data.time
                    ),

                "mcp_deg":
                    math.degrees(
                        float(
                            data.qpos[
                                ids[
                                    "mcp"
                                ][
                                    "qadr"
                                ]
                            ]
                        )
                    ),

                "pip_deg":
                    math.degrees(
                        float(
                            data.qpos[
                                ids[
                                    "pip"
                                ][
                                    "qadr"
                                ]
                            ]
                        )
                    ),

                "dip_deg":
                    math.degrees(
                        float(
                            data.qpos[
                                ids[
                                    "dip"
                                ][
                                    "qadr"
                                ]
                            ]
                        )
                    ),

                "mid_gap_mm":
                    mid_gap,

                "tip_gap_mm":
                    tip_gap,

                "mid_force_N":
                    mid_force,

                "tip_force_N":
                    tip_force,

                "mid_rows":
                    mid_rows,

                "tip_rows":
                    tip_rows,
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
# PRINT MIDDLE-PHALANX GEOMETRY
# ============================================================

def audit_middle_geometry(
    model,
):

    print()
    print("=" * 112)
    print(
        "V27 MIDDLE-PHALANX GEOMETRY"
    )
    print("=" * 112)


    middle_bodies = []


    for bid in range(
        model.nbody
    ):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            bid,
        )


        name = (
            name
            or
            ""
        )


        if (
            "Index_Middle"
            in name
            or
            "index_middle"
            in name.lower()
        ):

            middle_bodies.append(
                bid
            )

            print()
            print(
                f"BODY {bid}: "
                f"{name}"
            )


    if not middle_bodies:

        print(
            "No body containing Index_Middle found."
        )

        return


    for gid in range(
        model.ngeom
    ):

        bid = int(
            model.geom_bodyid[
                gid
            ]
        )


        if (
            bid
            not in middle_bodies
        ):

            continue


        gname = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            gid,
        )


        gname = (
            gname
            or
            "<unnamed>"
        )


        mesh_id = int(
            model.geom_dataid[
                gid
            ]
        )


        mesh_name = None


        if (
            int(
                model.geom_type[
                    gid
                ]
            )
            ==
            int(
                mujoco.mjtGeom.mjGEOM_MESH
            )
            and
            mesh_id >= 0
        ):

            mesh_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_MESH,
                mesh_id,
            )


        print()
        print(
            f"  GEOM {gid}: "
            f"{gname}"
        )

        print(
            f"    type = "
            f"{int(model.geom_type[gid])}"
        )

        print(
            f"    mesh = "
            f"{mesh_name}"
        )

        print(
            f"    pos = "
            f"{model.geom_pos[gid]}"
        )

        print(
            f"    size = "
            f"{model.geom_size[gid]}"
        )

        print(
            f"    contype = "
            f"{int(model.geom_contype[gid])}"
        )

        print(
            f"    conaffinity = "
            f"{int(model.geom_conaffinity[gid])}"
        )

        print(
            f"    rgba = "
            f"{model.geom_rgba[gid]}"
        )


# ============================================================
# STATIC GEOMETRY CHECK
# ============================================================

def static_pose_gap(
    recorder,
    target_deg,
):

    model = mujoco.MjModel.from_xml_path(
        str(
            MODEL_PATH
        )
    )


    # Reproduce runtime model-position modifications.
    model.geom_pos[:] = (
        recorder.first_geom_pos
    )

    model.body_pos[:] = (
        recorder.first_body_pos
    )


    data = mujoco.MjData(
        model
    )


    data.qpos[:] = (
        recorder.first_qpos
    )


    joints = [
        MCP,
        PIP,
        DIP,
    ]


    for name, angle in zip(
        joints,
        target_deg,
    ):

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )


        qadr = int(
            model.jnt_qposadr[
                jid
            ]
        )


        # Hardware quantities are relative to the open pose.
        data.qpos[
            qadr
        ] = (
            recorder.first_qpos[
                qadr
            ]
            +
            math.radians(
                float(
                    angle
                )
            )
        )


    mujoco.mj_forward(
        model,
        data,
    )


    mid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        MID,
    )


    tip = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        TIP,
    )


    bottle = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        BOTTLE,
    )


    return {

        "mid_gap_mm":
            1000.0
            *
            geom_distance(
                model,
                data,
                mid,
                bottle,
            ),

        "tip_gap_mm":
            1000.0
            *
            geom_distance(
                model,
                data,
                tip,
                bottle,
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "contact_bag",
        type=Path,
    )


    parser.add_argument(
        "--no-viewer",
        action="store_true",
    )


    args = parser.parse_args()


    contact_bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )


    if not contact_bag.exists():

        raise RuntimeError(
            f"Missing contact bag:\n"
            f"{contact_bag}"
        )


    v24.ab.patch_replay_modules()


    v24.build_model(
        MODEL_PATH
    )


    # ========================================================
    # PRINT ACTUAL MIDDLE GEOMETRY
    # ========================================================

    model_for_audit = (
        mujoco.MjModel.from_xml_path(
            str(
                MODEL_PATH
            )
        )
    )


    audit_middle_geometry(
        model_for_audit
    )


    # ========================================================
    # DYNAMIC V25 REPLAY
    # ========================================================

    law = (
        v24.StatefulTransmission()
    )


    law.install()


    recorder = (
        Recorder()
    )


    recorder.install()


    old_scale = (
        v24.base.METERS_PER_TICK
    )

    old_tendon = (
        v24.base.TENDON_STIFFNESS
    )

    old_mu = (
        v24.base.CONTACT_MU
    )

    old_contact_k = (
        v24.base.CONTACT_STIFFNESS
    )


    try:

        v24.base.METERS_PER_TICK = (
            v24.SCALE
        )

        v24.base.TENDON_STIFFNESS = (
            v24.TENDON_K
        )

        v24.base.CONTACT_MU = (
            v24.CONTACT_MU
        )

        v24.base.CONTACT_STIFFNESS = (
            v24.CONTACT_K
        )


        law.configure(
            V25_CASE,
            "CONTACT",
        )


        metrics, _ = (
            v24.base.run_contact(
                MODEL_PATH,
                contact_bag,
            )
        )


    finally:

        recorder.restore()

        law.restore()

        v24.base.METERS_PER_TICK = (
            old_scale
        )

        v24.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v24.base.CONTACT_MU = (
            old_mu
        )

        v24.base.CONTACT_STIFFNESS = (
            old_contact_k
        )


    rows = (
        recorder.rows
    )


    if not rows:

        raise RuntimeError(
            "V27 recorded no samples."
        )


    # ========================================================
    # USE ANGLES RELATIVE TO FIRST/OPEN SAMPLE
    # ========================================================

    q0 = np.array(
        [
            rows[0][
                "mcp_deg"
            ],
            rows[0][
                "pip_deg"
            ],
            rows[0][
                "dip_deg"
            ],
        ],
        dtype=float,
    )


    for row in rows:

        q = np.array(
            [
                row[
                    "mcp_deg"
                ],
                row[
                    "pip_deg"
                ],
                row[
                    "dip_deg"
                ],
            ],
            dtype=float,
        )


        row[
            "relative_pose"
        ] = (
            q
            -
            q0
        )


        row[
            "hw_rmse"
        ] = float(
            np.sqrt(
                np.mean(
                    (
                        row[
                            "relative_pose"
                        ]
                        -
                        HW_TARGET
                    )
                    ** 2
                )
            )
        )


    # ========================================================
    # STATIC CHECKS
    # ========================================================

    static_hw = static_pose_gap(
        recorder,
        HW_TARGET,
    )


    static_v25 = static_pose_gap(
        recorder,
        V25_PREDICTION,
    )


    print()
    print("=" * 112)
    print(
        "V27 STATIC GEOMETRY CHECK"
    )
    print("=" * 112)


    print()
    print(
        "At exact HARDWARE target "
        "14.536 / 23.811 / 9.327 deg:"
    )

    print(
        f"  middle physical gap = "
        f"{static_hw['mid_gap_mm']:+.3f} mm"
    )

    print(
        f"  tip physical gap    = "
        f"{static_hw['tip_gap_mm']:+.3f} mm"
    )


    print()
    print(
        "At V25 predicted pose "
        "14.918 / 24.219 / 8.862 deg:"
    )

    print(
        f"  middle physical gap = "
        f"{static_v25['mid_gap_mm']:+.3f} mm"
    )

    print(
        f"  tip physical gap    = "
        f"{static_v25['tip_gap_mm']:+.3f} mm"
    )


    # ========================================================
    # CLOSEST DYNAMIC POSE TO HW TARGET
    # ========================================================

    closest = min(
        rows,
        key=lambda r:
            r[
                "hw_rmse"
            ],
    )


    t0 = rows[
        0
    ][
        "time"
    ]


    print()
    print("=" * 112)
    print(
        "V27 DYNAMIC POSE CLOSEST TO HARDWARE"
    )
    print("=" * 112)


    print()
    print(
        f"elapsed = "
        f"{closest['time'] - t0:.3f} s"
    )


    pose = closest[
        "relative_pose"
    ]


    print(
        "MCP/PIP/DIP = "
        f"{pose[0]:.3f} / "
        f"{pose[1]:.3f} / "
        f"{pose[2]:.3f} deg"
    )


    print(
        f"angle RMSE to HW = "
        f"{closest['hw_rmse']:.3f} deg"
    )


    print(
        f"middle gap/Fn = "
        f"{closest['mid_gap_mm']:+.3f} mm / "
        f"{closest['mid_force_N']:.3f} N"
    )


    print(
        f"tip gap/Fn = "
        f"{closest['tip_gap_mm']:+.3f} mm / "
        f"{closest['tip_force_N']:.3f} N"
    )


    # ========================================================
    # FIRST TIP CONTACT
    # ========================================================

    first_tip = next(
        (
            row
            for row in rows
            if (
                row[
                    "tip_rows"
                ]
                > 0
                and
                row[
                    "tip_force_N"
                ]
                > 1.0e-4
            )
        ),
        None,
    )


    print()
    print("=" * 112)
    print(
        "V27 FIRST TIP CONTACT"
    )
    print("=" * 112)


    if first_tip is None:

        print(
            "No tip contact."
        )


    else:

        pose = first_tip[
            "relative_pose"
        ]


        print()
        print(
            f"elapsed = "
            f"{first_tip['time'] - t0:.3f} s"
        )

        print(
            "MCP/PIP/DIP = "
            f"{pose[0]:.3f} / "
            f"{pose[1]:.3f} / "
            f"{pose[2]:.3f} deg"
        )


        print(
            f"middle gap/Fn = "
            f"{first_tip['mid_gap_mm']:+.3f} mm / "
            f"{first_tip['mid_force_N']:.3f} N"
        )


        print(
            f"tip gap/Fn = "
            f"{first_tip['tip_gap_mm']:+.3f} mm / "
            f"{first_tip['tip_force_N']:.3f} N"
        )


    # ========================================================
    # GEOMETRIC SIMULTANEOUS-NEAR WINDOW
    # ========================================================

    near_both = [
        row
        for row in rows
        if (
            abs(
                row[
                    "mid_gap_mm"
                ]
            )
            <= 1.5
            and
            abs(
                row[
                    "tip_gap_mm"
                ]
            )
            <= 1.5
        )
    ]


    if len(
        rows
    ) > 1:

        dt = float(
            np.median(
                np.diff(
                    [
                        r[
                            "time"
                        ]
                        for r in rows
                    ]
                )
            )
        )


    else:

        dt = 0.0


    near_duration = (
        len(
            near_both
        )
        *
        dt
    )


    print()
    print("=" * 112)
    print(
        "V27 SIMULTANEOUS GEOMETRIC PROXIMITY"
    )
    print("=" * 112)


    print()
    print(
        "Criterion:"
    )

    print(
        "  |middle gap| <= 1.5 mm"
    )

    print(
        "  |tip gap|    <= 1.5 mm"
    )


    print()
    print(
        f"duration satisfying both = "
        f"{near_duration:.3f} s"
    )


    # ========================================================
    # DECISION
    # ========================================================

    print()
    print("=" * 112)
    print(
        "V27 DECISION"
    )
    print("=" * 112)


    geometry_supports_hw = (
        abs(
            static_hw[
                "mid_gap_mm"
            ]
        )
        <= 1.5
        and
        abs(
            static_hw[
                "tip_gap_mm"
            ]
        )
        <= 1.5
    )


    if geometry_supports_hw:

        print()
        print(
            "THE FROZEN GEOMETRY CAN SUPPORT THE "
            "HARDWARE CONTACT CONFIGURATION."
        )

        print()
        print(
            "Therefore the current problem is primarily "
            "DYNAMIC CONTACT RETENTION, not bottle pose."
        )

        print()
        print(
            "Do NOT move the bottle."
        )

        print(
            "Do NOT tune tendon ratios further."
        )

        print()
        print(
            "Next: replace the point-like capsule interaction "
            "with a physically faithful distributed/rolling "
            "middle-phalanx surface using the existing CAD/"
            "visual mesh if available."
        )


    else:

        print()
        print(
            "THE CURRENT COLLISION GEOMETRY DOES NOT "
            "SUPPORT THE MEASURED HARDWARE WRAP EVEN "
            "STATISTICALLY."
        )

        print()
        print(
            "Before changing compliance or dynamics, "
            "the middle/distal collision geometry must be "
            "corrected from CAD/mesh geometry."
        )

        print()
        print(
            "Bottle position remains frozen."
        )


    # ========================================================
    # GUI — ONE USEFUL REPLAY
    # ========================================================

    if not args.no_viewer:

        print()
        print("=" * 112)
        print(
            "V27 AUTOMATIC V25 GUI REPLAY"
        )
        print("=" * 112)


        print()
        print(
            "Watch specifically:"
        )

        print(
            "  middle contact -> separation distance "
            "while PIP rotates -> first tip contact"
        )


        v24.visual_replay(
            V25_CASE,
            contact_bag,
        )


    try:
        MODEL_PATH.unlink()
    except FileNotFoundError:
        pass


    print()
    print("=" * 112)
    print(
        "V27 COMPLETE"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
