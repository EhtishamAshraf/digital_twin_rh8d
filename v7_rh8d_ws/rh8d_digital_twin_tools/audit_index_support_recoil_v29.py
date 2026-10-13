#!/usr/bin/env python3

from pathlib import Path
import argparse
import math

import mujoco
import numpy as np

import scan_index_stateful_transmission_v24 as v24


# ============================================================
# V29 — INDEX SUPPORT / PALM RECOIL AUDIT
#
# V27 produced an important discrepancy:
#
# Static hardware target:
#     middle gap ~ 0 mm
#
# Dynamic replay at nearly same MCP/PIP/DIP:
#     middle gap ~ +2.57 mm
#
# Therefore some OTHER generalized coordinate / parent-body
# pose may be changing during contact.
#
# V29 quantifies:
#
#   - support/palm translation
#   - support/palm rotation
#   - upstream joint motion
#   - bottle motion
#
# and then performs counterfactual geometry tests:
#
#   A) actual dynamic qpos
#   B) same finger pose but upstream support joints restored
#   C) same finger pose but all non-index joints restored
#
# NO parameters are changed.
# ============================================================


WS = v24.WS
MJCF_DIR = v24.MJCF_DIR

MODEL_PATH = (
    MJCF_DIR
    / "tmp_v29_support_recoil.xml"
)


# ============================================================
# FROZEN V25 ARCHITECTURE
# ============================================================

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


MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

SPOOL = "index_motor_spool"

MID = "collision_idx_mid"
TIP = "collision_idx_tip"
BOTTLE = "index_contact_fixture"


# ============================================================
# HELPERS
# ============================================================

def joint_id(
    model,
    name,
):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:

        raise RuntimeError(
            f"Missing joint: {name}"
        )

    return jid


def geom_id(
    model,
    name,
):

    gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        name,
    )

    if gid < 0:

        raise RuntimeError(
            f"Missing geom: {name}"
        )

    return gid


def qpos_width(
    model,
    jid,
):

    jtype = int(
        model.jnt_type[
            jid
        ]
    )


    if (
        jtype
        ==
        int(
            mujoco.mjtJoint.mjJNT_FREE
        )
    ):

        return 7


    if (
        jtype
        ==
        int(
            mujoco.mjtJoint.mjJNT_BALL
        )
    ):

        return 4


    return 1


def geom_gap_mm(
    model,
    data,
    gid1,
    gid2,
):

    fromto = np.zeros(
        6,
        dtype=float,
    )


    distance = mujoco.mj_geomDistance(
        model,
        data,
        gid1,
        gid2,
        0.1,
        fromto,
    )


    return (
        1000.0
        *
        float(
            distance
        )
    )


def rotation_difference_deg(
    R0,
    R1,
):

    Rrel = (
        R0.T
        @
        R1
    )


    value = (
        np.trace(
            Rrel
        )
        -
        1.0
    ) / 2.0


    value = float(
        np.clip(
            value,
            -1.0,
            1.0,
        )
    )


    return math.degrees(
        math.acos(
            value
        )
    )


# ============================================================
# RECORDER
# ============================================================

class Recorder:

    def __init__(
        self,
    ):

        self.original_step = (
            mujoco.mj_step
        )

        self.rows = []

        self.cache = {}


        self.qpos0 = None

        self.geom_pos0 = None

        self.body_pos0 = None


        self.support_body_id = None

        self.support_name = None

        self.ancestor_joints = None


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


        mcp_jid = joint_id(
            model,
            MCP,
        )

        pip_jid = joint_id(
            model,
            PIP,
        )

        dip_jid = joint_id(
            model,
            DIP,
        )

        spool_jid = joint_id(
            model,
            SPOOL,
        )


        mcp_body = int(
            model.jnt_bodyid[
                mcp_jid
            ]
        )


        support_body = int(
            model.body_parentid[
                mcp_body
            ]
        )


        self.support_body_id = (
            support_body
        )


        support_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            support_body,
        )


        self.support_name = (
            support_name
            or
            f"body_{support_body}"
        )


        # ----------------------------------------------------
        # Find all joints upstream of MCP.
        # ----------------------------------------------------

        ancestor_joints = []

        body = support_body


        while body > 0:

            joint_adr = int(
                model.body_jntadr[
                    body
                ]
            )

            joint_num = int(
                model.body_jntnum[
                    body
                ]
            )


            for local in range(
                joint_num
            ):

                ancestor_joints.append(
                    joint_adr
                    +
                    local
                )


            parent = int(
                model.body_parentid[
                    body
                ]
            )


            if parent == body:

                break


            body = parent


        self.ancestor_joints = (
            ancestor_joints
        )


        result = {

            "mcp_jid":
                mcp_jid,

            "pip_jid":
                pip_jid,

            "dip_jid":
                dip_jid,

            "spool_jid":
                spool_jid,

            "mcp_q":
                int(
                    model.jnt_qposadr[
                        mcp_jid
                    ]
                ),

            "pip_q":
                int(
                    model.jnt_qposadr[
                        pip_jid
                    ]
                ),

            "dip_q":
                int(
                    model.jnt_qposadr[
                        dip_jid
                    ]
                ),

            "mid":
                geom_id(
                    model,
                    MID,
                ),

            "tip":
                geom_id(
                    model,
                    TIP,
                ),

            "bottle":
                geom_id(
                    model,
                    BOTTLE,
                ),

            "support":
                support_body,
        }


        self.cache[
            key
        ] = result

        return result


    def record(
        self,
        model,
        data,
    ):

        ids = self.ids(
            model
        )


        if (
            self.qpos0
            is None
        ):

            self.qpos0 = (
                data.qpos.copy()
            )

            self.geom_pos0 = (
                model.geom_pos.copy()
            )

            self.body_pos0 = (
                model.body_pos.copy()
            )


        support = ids[
            "support"
        ]


        self.rows.append(
            {

                "time":
                    float(
                        data.time
                    ),

                "qpos":
                    data.qpos.copy(),

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

                "mid_gap_mm":
                    geom_gap_mm(
                        model,
                        data,
                        ids[
                            "mid"
                        ],
                        ids[
                            "bottle"
                        ],
                    ),

                "tip_gap_mm":
                    geom_gap_mm(
                        model,
                        data,
                        ids[
                            "tip"
                        ],
                        ids[
                            "bottle"
                        ],
                    ),

                "support_xpos":
                    data.xpos[
                        support
                    ].copy(),

                "support_xmat":
                    data.xmat[
                        support
                    ].reshape(
                        3,
                        3,
                    ).copy(),

                "bottle_xpos":
                    data.geom_xpos[
                        ids[
                            "bottle"
                        ]
                    ].copy(),
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


        mujoco.mj_step = (
            wrapped
        )


    def restore(
        self,
    ):

        mujoco.mj_step = (
            self.original_step
        )


# ============================================================
# RESTORE JOINTS TO INITIAL QPOS
# ============================================================

def restore_joint_set(
    model,
    qpos,
    qpos0,
    joint_ids,
):

    result = qpos.copy()


    for jid in joint_ids:

        qadr = int(
            model.jnt_qposadr[
                jid
            ]
        )


        width = qpos_width(
            model,
            jid,
        )


        result[
            qadr:
            qadr
            +
            width
        ] = qpos0[
            qadr:
            qadr
            +
            width
        ]


    return result


# ============================================================
# COUNTERFACTUAL GAP
# ============================================================

def evaluate_qpos(
    recorder,
    qpos,
):

    model = mujoco.MjModel.from_xml_path(
        str(
            MODEL_PATH
        )
    )


    # Reproduce runtime position modifications.
    model.geom_pos[:] = (
        recorder.geom_pos0
    )

    model.body_pos[:] = (
        recorder.body_pos0
    )


    data = mujoco.MjData(
        model
    )


    data.qpos[:] = (
        qpos
    )

    data.qvel[:] = 0.0


    mujoco.mj_forward(
        model,
        data,
    )


    mid = geom_id(
        model,
        MID,
    )

    tip = geom_id(
        model,
        TIP,
    )

    bottle = geom_id(
        model,
        BOTTLE,
    )


    return {

        "middle_gap_mm":
            geom_gap_mm(
                model,
                data,
                mid,
                bottle,
            ),

        "tip_gap_mm":
            geom_gap_mm(
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


    law = (
        v24.StatefulTransmission()
    )

    law.install()


    recorder = Recorder()

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


        v24.base.run_contact(
            MODEL_PATH,
            contact_bag,
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


    rows = recorder.rows


    if not rows:

        raise RuntimeError(
            "No V29 rows recorded."
        )


    # ========================================================
    # RELATIVE FINGER POSE
    # ========================================================

    q0_deg = np.array(
        [
            rows[
                0
            ][
                "mcp_deg"
            ],

            rows[
                0
            ][
                "pip_deg"
            ],

            rows[
                0
            ][
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
            "relative"
        ] = (
            q
            -
            q0_deg
        )


        row[
            "rmse_hw"
        ] = float(
            np.sqrt(
                np.mean(
                    (
                        row[
                            "relative"
                        ]
                        -
                        HW_TARGET
                    )
                    ** 2
                )
            )
        )


    closest = min(
        rows,
        key=lambda row:
            row[
                "rmse_hw"
            ],
    )


    first = rows[
        0
    ]


    # ========================================================
    # SUPPORT BODY MOTION
    # ========================================================

    support_translation_mm = (
        1000.0
        *
        np.linalg.norm(
            closest[
                "support_xpos"
            ]
            -
            first[
                "support_xpos"
            ]
        )
    )


    support_rotation_deg = (
        rotation_difference_deg(
            first[
                "support_xmat"
            ],
            closest[
                "support_xmat"
            ],
        )
    )


    bottle_translation_mm = (
        1000.0
        *
        np.linalg.norm(
            closest[
                "bottle_xpos"
            ]
            -
            first[
                "bottle_xpos"
            ]
        )
    )


    print()
    print("=" * 116)
    print(
        "V29 SUPPORT / PALM RECOIL AUDIT"
    )
    print("=" * 116)


    print()
    print(
        f"Support body = "
        f"{recorder.support_name}"
    )


    print()
    print(
        "Dynamic state closest to hardware target:"
    )


    pose = closest[
        "relative"
    ]


    print(
        "  MCP/PIP/DIP = "
        f"{pose[0]:.3f} / "
        f"{pose[1]:.3f} / "
        f"{pose[2]:.3f} deg"
    )


    print(
        f"  angle RMSE = "
        f"{closest['rmse_hw']:.3f} deg"
    )


    print(
        f"  middle gap = "
        f"{closest['mid_gap_mm']:+.3f} mm"
    )


    print(
        f"  tip gap = "
        f"{closest['tip_gap_mm']:+.3f} mm"
    )


    print()
    print(
        "SUPPORT MOTION FROM INITIAL STATE:"
    )


    delta_xyz = (
        1000.0
        *
        (
            closest[
                "support_xpos"
            ]
            -
            first[
                "support_xpos"
            ]
        )
    )


    print(
        "  translation xyz = "
        f"[{delta_xyz[0]:+.3f}, "
        f"{delta_xyz[1]:+.3f}, "
        f"{delta_xyz[2]:+.3f}] mm"
    )


    print(
        f"  translation norm = "
        f"{support_translation_mm:.3f} mm"
    )


    print(
        f"  rotation = "
        f"{support_rotation_deg:.4f} deg"
    )


    print()
    print(
        f"Bottle world displacement = "
        f"{bottle_translation_mm:.6f} mm"
    )


    # ========================================================
    # UPSTREAM JOINT CHANGES
    # ========================================================

    model = mujoco.MjModel.from_xml_path(
        str(
            MODEL_PATH
        )
    )


    print()
    print("=" * 116)
    print(
        "V29 UPSTREAM JOINT CHANGES"
    )
    print("=" * 116)


    for jid in recorder.ancestor_joints:

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )


        name = (
            name
            or
            f"joint_{jid}"
        )


        qadr = int(
            model.jnt_qposadr[
                jid
            ]
        )


        width = qpos_width(
            model,
            jid,
        )


        before = recorder.qpos0[
            qadr:
            qadr
            +
            width
        ]


        after = closest[
            "qpos"
        ][
            qadr:
            qadr
            +
            width
        ]


        jtype = int(
            model.jnt_type[
                jid
            ]
        )


        if width == 1:

            delta = float(
                after[
                    0
                ]
                -
                before[
                    0
                ]
            )


            if (
                jtype
                ==
                int(
                    mujoco.mjtJoint.mjJNT_SLIDE
                )
            ):

                print(
                    f"  {name}: "
                    f"{1000.0 * delta:+.4f} mm"
                )


            else:

                print(
                    f"  {name}: "
                    f"{math.degrees(delta):+.5f} deg"
                )


        else:

            print(
                f"  {name}: "
                f"qpos change norm = "
                f"{np.linalg.norm(after-before):.6e}"
            )


    # ========================================================
    # COUNTERFACTUALS
    # ========================================================

    dynamic_qpos = (
        closest[
            "qpos"
        ].copy()
    )


    actual = evaluate_qpos(
        recorder,
        dynamic_qpos,
    )


    upstream_restored_qpos = (
        restore_joint_set(
            model,
            dynamic_qpos,
            recorder.qpos0,
            recorder.ancestor_joints,
        )
    )


    upstream_restored = (
        evaluate_qpos(
            recorder,
            upstream_restored_qpos,
        )
    )


    # --------------------------------------------------------
    # Restore ALL joints except:
    #
    # MCP/PIP/DIP and hidden motor spool.
    #
    # This tells us whether ANY non-index DOF is responsible.
    # --------------------------------------------------------

    preserve = {

        joint_id(
            model,
            MCP,
        ),

        joint_id(
            model,
            PIP,
        ),

        joint_id(
            model,
            DIP,
        ),

        joint_id(
            model,
            SPOOL,
        ),
    }


    non_index_joints = [

        jid

        for jid in range(
            model.njnt
        )

        if jid
        not in preserve
    ]


    all_other_restored_qpos = (
        restore_joint_set(
            model,
            dynamic_qpos,
            recorder.qpos0,
            non_index_joints,
        )
    )


    all_other_restored = (
        evaluate_qpos(
            recorder,
            all_other_restored_qpos,
        )
    )


    print()
    print("=" * 116)
    print(
        "V29 GAP DECOMPOSITION"
    )
    print("=" * 116)


    print()
    print(
        "A) ACTUAL DYNAMIC QPOS"
    )

    print(
        f"   middle = "
        f"{actual['middle_gap_mm']:+.3f} mm"
    )

    print(
        f"   tip    = "
        f"{actual['tip_gap_mm']:+.3f} mm"
    )


    print()
    print(
        "B) SAME FINGER QPOS, "
        "UPSTREAM SUPPORT JOINTS RESET"
    )

    print(
        f"   middle = "
        f"{upstream_restored['middle_gap_mm']:+.3f} mm"
    )

    print(
        f"   tip    = "
        f"{upstream_restored['tip_gap_mm']:+.3f} mm"
    )


    print()
    print(
        "C) SAME MCP/PIP/DIP, "
        "ALL NON-INDEX JOINTS RESET"
    )

    print(
        f"   middle = "
        f"{all_other_restored['middle_gap_mm']:+.3f} mm"
    )

    print(
        f"   tip    = "
        f"{all_other_restored['tip_gap_mm']:+.3f} mm"
    )


    # ========================================================
    # DECISION
    # ========================================================

    print()
    print("=" * 116)
    print(
        "V29 DECISION"
    )
    print("=" * 116)


    recovered_by_support = (
        actual[
            "middle_gap_mm"
        ]
        -
        upstream_restored[
            "middle_gap_mm"
        ]
    )


    recovered_by_all = (
        actual[
            "middle_gap_mm"
        ]
        -
        all_other_restored[
            "middle_gap_mm"
        ]
    )


    print()
    print(
        f"Middle-gap recovered by upstream reset = "
        f"{recovered_by_support:+.3f} mm"
    )


    print(
        f"Middle-gap recovered by all non-index reset = "
        f"{recovered_by_all:+.3f} mm"
    )


    if (
        recovered_by_support
        >= 1.0
    ):

        print()
        print(
            "MAJOR FINDING:"
        )

        print(
            "The hand/support chain is recoiling enough "
            "during contact to explain a substantial part "
            "of the lost middle-phalanx contact."
        )

        print()
        print(
            "NEXT:"
        )

        print(
            "Hold/replay the upstream arm/palm configuration "
            "as in the hardware experiment and rerun V25 "
            "before adding any new contact model."
        )


    elif (
        recovered_by_all
        >= 1.0
    ):

        print()
        print(
            "A NON-INDEX DEGREE OF FREEDOM is responsible "
            "for a substantial part of the contact loss."
        )

        print()
        print(
            "NEXT: identify that DOF and constrain/replay it "
            "according to the hardware experiment."
        )


    else:

        print()
        print(
            "SUPPORT / ARM RECOIL DOES NOT EXPLAIN "
            "THE MIDDLE-CONTACT LOSS."
        )

        print()
        print(
            "Then V27/V28 genuinely localize the remaining "
            "problem to distributed compliant contact / "
            "surface mechanics."
        )

        print()
        print(
            "NEXT: build the effective compliant contact "
            "layer; do not revisit bottle pose or tendon "
            "ratios."
        )


    try:
        MODEL_PATH.unlink()
    except FileNotFoundError:
        pass


    print()
    print("=" * 116)
    print(
        "V29 COMPLETE"
    )
    print("=" * 116)


if __name__ == "__main__":
    main()
