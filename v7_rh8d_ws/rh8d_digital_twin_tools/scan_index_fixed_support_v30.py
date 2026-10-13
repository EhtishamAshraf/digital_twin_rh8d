#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv

import mujoco
import numpy as np

import scan_index_stateful_transmission_v24 as v24


# ============================================================
# V30 — FIXED SUPPORT / ARM CONTACT REPLAY
#
# V29 found:
#
# support translation = 7.264 mm
#
# dynamic middle gap:
#     +2.574 mm
#
# same finger pose with upstream joints reset:
#     -0.439 mm
#
# recovered middle gap:
#     3.013 mm
#
# Therefore previous contact tuning was confounded by
# unintended motion of the UR5e/palm support chain.
#
# V30 holds all joints upstream of index MCP at their initial
# replay configuration.
#
# This represents the experimental boundary condition:
# the UR5e was holding the hand pose during the low-force
# index-contact experiment.
#
# NO bottle movement.
# NO pad.
# NO mesh-collision replacement.
# NO contact parameter changes.
# ============================================================


WS = v24.WS
MJCF_DIR = v24.MJCF_DIR

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "fixed_support_v30"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY = RESULT_DIR / "summary.csv"

SELECTED_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_fixed_support_v30_candidate.xml"
)


HW = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


FREE_GATE = 6.085

MIN_OVERLAP_S = 0.15


MCP = "Index_Proximal--palm_assembly:1"

MID = "collision_idx_mid"
TIP = "collision_idx_tip"
BOTTLE = "index_contact_fixture"


# ============================================================
# SMALL RE-EVALUATION SET
#
# Important:
#
# V25 ratios are NOT assumed correct because they were tuned
# while the arm/support chain recoiled.
# ============================================================

CASES = [

    {
        "name":
            "BASELINE_FIXED",

        "yield":
            False,

        "arms_mm":
            [5.0, 4.0, 1.0],
    },

    {
        "name":
            "YIELD_5_4_1",

        "yield":
            True,

        "arms_mm":
            [5.0, 4.0, 1.0],
    },

    {
        "name":
            "YIELD_4_8_1p5",

        "yield":
            True,

        "arms_mm":
            [4.0, 8.0, 1.5],
    },

    {
        "name":
            "YIELD_3p5_9_2",

        "yield":
            True,

        "arms_mm":
            [3.5, 9.0, 2.0],
    },

    {
        "name":
            "YIELD_3_11_1p8",

        "yield":
            True,

        "arms_mm":
            [3.0, 11.0, 1.8],
    },

    {
        "name":
            "YIELD_2p5_13_1p8",

        "yield":
            True,

        "arms_mm":
            [2.5, 13.0, 1.8],
    },
]


# ============================================================
# JOINT DIMENSIONS
# ============================================================

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


def dof_width(
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

        return 6


    if (
        jtype
        ==
        int(
            mujoco.mjtJoint.mjJNT_BALL
        )
    ):

        return 3


    return 1


# ============================================================
# FIXED-UPSTREAM BOUNDARY CONDITION
# ============================================================

class FixedSupport:

    def __init__(self):

        # Instantiate AFTER StatefulTransmission.install().
        # Therefore this wraps the transmission model.
        self.original_step = (
            mujoco.mj_step
        )

        self.cache = {}

        self.reference = {}


    def information(
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


        mcp = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            MCP,
        )


        if mcp < 0:

            self.cache[
                key
            ] = None

            return None


        mcp_body = int(
            model.jnt_bodyid[
                mcp
            ]
        )


        support_body = int(
            model.body_parentid[
                mcp_body
            ]
        )


        support_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            support_body,
        )


        # ----------------------------------------------------
        # All joints in the kinematic chain ABOVE MCP.
        # ----------------------------------------------------

        ancestors = []

        body = support_body


        while body > 0:

            adr = int(
                model.body_jntadr[
                    body
                ]
            )

            num = int(
                model.body_jntnum[
                    body
                ]
            )


            for offset in range(
                num
            ):

                ancestors.append(
                    adr
                    +
                    offset
                )


            parent = int(
                model.body_parentid[
                    body
                ]
            )


            if parent == body:

                break


            body = parent


        result = {

            "support_body":
                support_body,

            "support_name":
                support_name
                or
                f"body_{support_body}",

            "joints":
                ancestors,
        }


        self.cache[
            key
        ] = result

        return result


    def ensure_reference(
        self,
        model,
        data,
        info,
    ):

        key = id(
            model
        )


        if key in self.reference:

            return self.reference[
                key
            ]


        qpos_ref = {}

        support_xpos = (
            data.xpos[
                info[
                    "support_body"
                ]
            ].copy()
        )


        for jid in info[
            "joints"
        ]:

            qadr = int(
                model.jnt_qposadr[
                    jid
                ]
            )


            width = qpos_width(
                model,
                jid,
            )


            qpos_ref[
                jid
            ] = (
                data.qpos[
                    qadr:
                    qadr
                    +
                    width
                ].copy()
            )


        state = {

            "qpos":
                qpos_ref,

            "support_xpos":
                support_xpos,
        }


        self.reference[
            key
        ] = state


        print()
        print(
            "[FIXED SUPPORT]"
        )

        print(
            f"  body = "
            f"{info['support_name']}"
        )


        print(
            "  locked upstream joints:"
        )


        for jid in info[
            "joints"
        ]:

            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jid,
            )

            print(
                f"    {name}"
            )


        return state


    def apply(
        self,
        model,
        data,
        info,
        state,
    ):

        for jid in info[
            "joints"
        ]:

            qadr = int(
                model.jnt_qposadr[
                    jid
                ]
            )


            qwidth = qpos_width(
                model,
                jid,
            )


            data.qpos[
                qadr:
                qadr
                +
                qwidth
            ] = (
                state[
                    "qpos"
                ][
                    jid
                ]
            )


            dadr = int(
                model.jnt_dofadr[
                    jid
                ]
            )


            dwidth = dof_width(
                model,
                jid,
            )


            data.qvel[
                dadr:
                dadr
                +
                dwidth
            ] = 0.0


    def install(self):

        original = (
            self.original_step
        )

        clamp = self


        def wrapped(
            model,
            data,
            *args,
            **kwargs,
        ):

            info = clamp.information(
                model
            )


            if info is None:

                return original(
                    model,
                    data,
                    *args,
                    **kwargs,
                )


            state = clamp.ensure_reference(
                model,
                data,
                info,
            )


            # Fixed boundary condition before integration.
            clamp.apply(
                model,
                data,
                info,
                state,
            )


            result = original(
                model,
                data,
                *args,
                **kwargs,
            )


            # Remove any numerical motion accumulated during
            # this step.
            clamp.apply(
                model,
                data,
                info,
                state,
            )


            # Recompute world transforms and contacts at the
            # corrected fixed-support configuration.
            mujoco.mj_forward(
                model,
                data,
            )


            return result


        mujoco.mj_step = (
            wrapped
        )


    def restore(self):

        mujoco.mj_step = (
            self.original_step
        )


# ============================================================
# CONTACT RECORDER
# ============================================================

class Recorder:

    def __init__(self):

        self.original_step = (
            mujoco.mj_step
        )

        self.rows = []

        self.cache = {}

        self.initial_support = {}


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


        def gid(name):

            return mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )


        mcp = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            MCP,
        )


        mcp_body = int(
            model.jnt_bodyid[
                mcp
            ]
        )


        support = int(
            model.body_parentid[
                mcp_body
            ]
        )


        result = {

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

            "support":
                support,
        }


        self.cache[
            key
        ] = result

        return result


    @staticmethod
    def normal_force(
        model,
        data,
        index,
    ):

        force = np.zeros(
            6,
            dtype=float,
        )


        mujoco.mj_contactForce(
            model,
            data,
            index,
            force,
        )


        return abs(
            float(
                force[
                    0
                ]
            )
        )


    def record(
        self,
        model,
        data,
    ):

        ids = self.ids(
            model
        )


        key = id(
            model
        )


        if key not in self.initial_support:

            self.initial_support[
                key
            ] = (
                data.xpos[
                    ids[
                        "support"
                    ]
                ].copy()
            )


        mid_force = 0.0
        tip_force = 0.0

        mid_rows = 0
        tip_rows = 0


        for i in range(
            data.ncon
        ):

            contact = data.contact[
                i
            ]


            pair = {
                int(
                    contact.geom1
                ),
                int(
                    contact.geom2
                ),
            }


            if pair == {
                ids[
                    "mid"
                ],
                ids[
                    "bottle"
                ],
            }:

                mid_rows += 1

                mid_force += (
                    self.normal_force(
                        model,
                        data,
                        i,
                    )
                )


            elif pair == {
                ids[
                    "tip"
                ],
                ids[
                    "bottle"
                ],
            }:

                tip_rows += 1

                tip_force += (
                    self.normal_force(
                        model,
                        data,
                        i,
                    )
                )


        support_delta = (
            1000.0
            *
            np.linalg.norm(
                data.xpos[
                    ids[
                        "support"
                    ]
                ]
                -
                self.initial_support[
                    key
                ]
            )
        )


        self.rows.append(
            {

                "time":
                    float(
                        data.time
                    ),

                "mid_active":
                    (
                        mid_rows > 0
                        and
                        mid_force
                        > 1.0e-4
                    ),

                "tip_active":
                    (
                        tip_rows > 0
                        and
                        tip_force
                        > 1.0e-4
                    ),

                "mid_force_N":
                    mid_force,

                "tip_force_N":
                    tip_force,

                "support_drift_mm":
                    float(
                        support_delta
                    ),
            }
        )


    def install(self):

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


    def restore(self):

        mujoco.mj_step = (
            self.original_step
        )


# ============================================================
# METRICS
# ============================================================

def longest_duration(
    mask,
    dt,
):

    longest = 0
    current = 0


    for value in mask:

        if value:

            current += 1

            longest = max(
                longest,
                current,
            )

        else:

            current = 0


    return float(
        longest
        *
        dt
    )


def analyse(
    rows,
):

    if len(
        rows
    ) < 2:

        return {

            "middle_duration_s":
                0.0,

            "tip_duration_s":
                0.0,

            "overlap_s":
                0.0,

            "longest_overlap_s":
                0.0,

            "middle_after_tip_s":
                0.0,

            "max_support_drift_mm":
                0.0,
        }


    times = np.asarray(
        [
            r[
                "time"
            ]
            for r in rows
        ],
        dtype=float,
    )


    dt = float(
        np.median(
            np.diff(
                times
            )
        )
    )


    middle = np.asarray(
        [
            r[
                "mid_active"
            ]
            for r in rows
        ],
        dtype=bool,
    )


    tip = np.asarray(
        [
            r[
                "tip_active"
            ]
            for r in rows
        ],
        dtype=bool,
    )


    both = (
        middle
        &
        tip
    )


    if np.any(
        tip
    ):

        first_tip = int(
            np.argmax(
                tip
            )
        )


        middle_after_tip = float(
            np.sum(
                middle[
                    first_tip:
                ]
            )
            *
            dt
        )


    else:

        middle_after_tip = 0.0


    return {

        "middle_duration_s":
            float(
                np.sum(
                    middle
                )
                *
                dt
            ),

        "tip_duration_s":
            float(
                np.sum(
                    tip
                )
                *
                dt
            ),

        "overlap_s":
            float(
                np.sum(
                    both
                )
                *
                dt
            ),

        "longest_overlap_s":
            longest_duration(
                both,
                dt,
            ),

        "middle_after_tip_s":
            middle_after_tip,

        "max_support_drift_mm":
            float(
                np.max(
                    [
                        r[
                            "support_drift_mm"
                        ]
                        for r in rows
                    ]
                )
            ),
    }


# ============================================================
# FREE-SPACE GUARD
# ============================================================

def free_guard(
    free_bag,
):

    contact_xml = (
        MJCF_DIR
        /
        "tmp_v30_free_contact.xml"
    )


    free_xml = (
        MJCF_DIR
        /
        "tmp_v30_free.xml"
    )


    v24.build_model(
        contact_xml
    )


    v24.ab.make_free_model(
        contact_xml,
        free_xml,
    )


    case = {

        "name":
            "FREE_GUARD",

        "yield":
            True,

        "arms_mm":
            [5.0, 4.0, 1.0],
    }


    law = (
        v24.StatefulTransmission()
    )

    law.install()


    clamp = (
        FixedSupport()
    )

    clamp.install()


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
            case,
            "FREE",
        )


        metrics, _ = (
            v24.base.run_free(
                free_xml,
                free_bag,
            )
        )


    finally:

        clamp.restore()

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


    event = (
        v24.first_event(
            law.events,
            case[
                "name"
            ],
            "FREE",
        )
    )


    try:
        contact_xml.unlink()
    except FileNotFoundError:
        pass


    try:
        free_xml.unlink()
    except FileNotFoundError:
        pass


    return (
        float(
            metrics[
                "combined"
            ]
        ),
        event is not None,
    )


# ============================================================
# CONTACT CASE
# ============================================================

def run_case(
    case,
    contact_bag,
):

    label = (
        case[
            "name"
        ]
        .lower()
        .replace(
            ".",
            "p",
        )
    )


    model_path = (
        MJCF_DIR
        /
        f"tmp_v30_{label}.xml"
    )


    v24.build_model(
        model_path
    )


    law = (
        v24.StatefulTransmission()
    )

    law.install()


    clamp = (
        FixedSupport()
    )

    clamp.install()


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
            case,
            "CONTACT",
        )


        metrics, _ = (
            v24.base.run_contact(
                model_path,
                contact_bag,
            )
        )


    finally:

        recorder.restore()

        clamp.restore()

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


    pose = np.asarray(
        metrics[
            "pose"
        ],
        dtype=float,
    )


    rmse = float(
        np.sqrt(
            np.mean(
                (
                    pose
                    -
                    HW
                )
                ** 2
            )
        )
    )


    retention = analyse(
        recorder.rows
    )


    result = {

        "name":
            case[
                "name"
            ],

        "yield":
            case[
                "yield"
            ],

        "mcp_arm_mm":
            case[
                "arms_mm"
            ][
                0
            ],

        "pip_arm_mm":
            case[
                "arms_mm"
            ][
                1
            ],

        "dip_arm_mm":
            case[
                "arms_mm"
            ][
                2
            ],

        "mcp_deg":
            float(
                pose[
                    0
                ]
            ),

        "pip_deg":
            float(
                pose[
                    1
                ]
            ),

        "dip_deg":
            float(
                pose[
                    2
                ]
            ),

        "rmse_deg":
            rmse,

        **retention,
    }


    try:
        model_path.unlink()
    except FileNotFoundError:
        pass


    return result


# ============================================================
# FIXED-SUPPORT GUI
# ============================================================

def visual_fixed(
    case,
    contact_bag,
):

    v24.build_model(
        SELECTED_MODEL
    )


    law = (
        v24.StatefulTransmission()
    )

    law.install()


    clamp = (
        FixedSupport()
    )

    clamp.install()


    old_scale = (
        v24.viewer_tools.base.METERS_PER_TICK
    )

    old_tendon = (
        v24.viewer_tools.base.TENDON_STIFFNESS
    )

    old_mu = (
        v24.viewer_tools.base.CONTACT_MU
    )

    old_k = (
        v24.viewer_tools.base.CONTACT_STIFFNESS
    )


    try:

        v24.viewer_tools.base.METERS_PER_TICK = (
            v24.SCALE
        )

        v24.viewer_tools.base.TENDON_STIFFNESS = (
            v24.TENDON_K
        )

        v24.viewer_tools.base.CONTACT_MU = (
            v24.CONTACT_MU
        )

        v24.viewer_tools.base.CONTACT_STIFFNESS = (
            v24.CONTACT_K
        )


        law.configure(
            case,
            "CONTACT",
        )


        v24.viewer_tools.visual_replay(
            SELECTED_MODEL,
            contact_bag,
        )


    finally:

        clamp.restore()

        law.restore()

        v24.viewer_tools.base.METERS_PER_TICK = (
            old_scale
        )

        v24.viewer_tools.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v24.viewer_tools.base.CONTACT_MU = (
            old_mu
        )

        v24.viewer_tools.base.CONTACT_STIFFNESS = (
            old_k
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "free_bag",
        type=Path,
    )


    parser.add_argument(
        "contact_bag",
        type=Path,
    )


    parser.add_argument(
        "--no-viewer",
        action="store_true",
    )


    args = parser.parse_args()


    free_bag = (
        args.free_bag
        .expanduser()
        .resolve()
    )


    contact_bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )


    if not free_bag.exists():

        raise RuntimeError(
            f"Missing free bag:\n"
            f"{free_bag}"
        )


    if not contact_bag.exists():

        raise RuntimeError(
            f"Missing contact bag:\n"
            f"{contact_bag}"
        )


    v24.ab.patch_replay_modules()


    print()
    print("=" * 122)
    print(
        "RH8D INDEX — FIXED SUPPORT REPLAY V30"
    )
    print("=" * 122)


    print()
    print(
        "Critical change:"
    )

    print(
        "  UR5e + palm support chain is held at "
        "its initial replay pose."
    )


    print()
    print(
        "Still frozen:"
    )

    print(
        "  bottle position = V5"
    )

    print(
        "  collision geometry = capsule baseline"
    )

    print(
        "  contact mu = 0.05"
    )

    print(
        "  contact k = 1000 N/m"
    )

    print(
        "  no artificial compliant pad"
    )


    # ========================================================
    # FREE-SPACE GUARD
    # ========================================================

    print()
    print(
        "Running fixed-support free-space guard..."
    )


    (
        free_rmse,
        free_breakaway,
    ) = free_guard(
        free_bag
    )


    print()
    print(
        f"Fixed-support free RMSE = "
        f"{free_rmse:.3f} deg"
    )

    print(
        f"Free breakaway = "
        f"{free_breakaway}"
    )


    if (
        free_rmse
        >
        FREE_GATE
    ):

        print()
        print(
            "WARNING: fixed support changed the "
            "trusted free-space result."
        )


    # ========================================================
    # CONTACT RE-EVALUATION
    # ========================================================

    results = []


    for index, case in enumerate(
        CASES,
        start=1,
    ):

        print()
        print("=" * 122)

        print(
            f"[{index:02d}/{len(CASES):02d}] "
            f"{case['name']}"
        )

        print(
            f"post-yield arms = "
            f"{case['arms_mm']} mm"
        )

        print("=" * 122)


        result = run_case(
            case,
            contact_bag,
        )


        results.append(
            result
        )


        print()
        print(
            "Contact MCP/PIP/DIP = "
            f"{result['mcp_deg']:.3f} / "
            f"{result['pip_deg']:.3f} / "
            f"{result['dip_deg']:.3f} deg"
        )


        print(
            f"RMSE = "
            f"{result['rmse_deg']:.3f} deg"
        )


        print()
        print(
            "Retention:"
        )

        print(
            f"  middle total = "
            f"{result['middle_duration_s']:.3f} s"
        )

        print(
            f"  tip total = "
            f"{result['tip_duration_s']:.3f} s"
        )

        print(
            f"  overlap total = "
            f"{result['overlap_s']:.3f} s"
        )

        print(
            f"  longest continuous overlap = "
            f"{result['longest_overlap_s']:.3f} s"
        )

        print(
            f"  middle after first tip = "
            f"{result['middle_after_tip_s']:.3f} s"
        )


        print(
            f"  max support drift = "
            f"{result['max_support_drift_mm']:.6f} mm"
        )


    # ========================================================
    # SAVE
    # ========================================================

    with open(
        SUMMARY,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # ========================================================
    # SELECT
    # ========================================================

    retained = [
        r
        for r in results
        if (
            r[
                "longest_overlap_s"
            ]
            >= MIN_OVERLAP_S
        )
    ]


    if retained:

        selected = min(
            retained,
            key=lambda r:
                r[
                    "rmse_deg"
                ],
        )

        mode = (
            "PERSISTENT HARDWARE-LIKE WRAP"
        )


    else:

        selected = max(
            results,
            key=lambda r:
                (
                    r[
                        "longest_overlap_s"
                    ],
                    -r[
                        "rmse_deg"
                    ],
                ),
        )

        mode = (
            "NO CASE PASSED 0.15 s OVERLAP"
        )


    selected_case = next(
        case
        for case in CASES
        if (
            case[
                "name"
            ]
            ==
            selected[
                "name"
            ]
        )
    )


    print()
    print("=" * 122)
    print(
        "V30 SELECTED FIXED-SUPPORT RESULT"
    )
    print("=" * 122)


    print()
    print(
        f"Selection mode = "
        f"{mode}"
    )


    print(
        f"Candidate = "
        f"{selected['name']}"
    )


    print(
        "Transmission = "
        f"{selected['mcp_arm_mm']:.2f} / "
        f"{selected['pip_arm_mm']:.2f} / "
        f"{selected['dip_arm_mm']:.2f} mm"
    )


    print()
    print(
        "Hardware:"
    )

    print(
        "  14.536 / 23.811 / 9.327 deg"
    )


    print(
        "Simulation:"
    )

    print(
        "  "
        f"{selected['mcp_deg']:.3f} / "
        f"{selected['pip_deg']:.3f} / "
        f"{selected['dip_deg']:.3f} deg"
    )


    print(
        f"RMSE = "
        f"{selected['rmse_deg']:.3f} deg"
    )


    print()
    print(
        f"Longest middle+tip overlap = "
        f"{selected['longest_overlap_s']:.3f} s"
    )


    print(
        f"Middle after tip = "
        f"{selected['middle_after_tip_s']:.3f} s"
    )


    print(
        f"Maximum support drift = "
        f"{selected['max_support_drift_mm']:.6f} mm"
    )


    # ========================================================
    # DECISION
    # ========================================================

    print()
    print("=" * 122)
    print(
        "V30 DECISION"
    )
    print("=" * 122)


    if (
        selected[
            "longest_overlap_s"
        ]
        >= MIN_OVERLAP_S
    ):

        print()
        print(
            "FIXING THE HARDWARE SUPPORT BOUNDARY "
            "RECOVERS PERSISTENT MIDDLE+TIP WRAPPING."
        )

        print()
        print(
            "This means the dominant previous contact-retention "
            "error came from unintended UR5e/palm recoil."
        )


        if (
            selected[
                "rmse_deg"
            ]
            <= 3.0
        ):

            print()
            print(
                "MAJOR RESULT:"
            )

            print(
                "The corrected boundary condition now gives "
                "both hardware-like contact topology and "
                "reasonable joint-angle agreement."
            )

            print()
            print(
                "STOP adding contact structures."
            )

            print(
                "Next: one narrow re-identification under "
                "fixed support, then held-out validation."
            )


        else:

            print()
            print(
                "Contact topology is now correct, but the "
                "transmission must be re-identified under "
                "this corrected boundary condition."
            )


    else:

        print()
        print(
            "FIXING SUPPORT REMOVES THE LARGE RECOIL, "
            "BUT PERSISTENT WRAP IS STILL INCOMPLETE."
        )

        print()
        print(
            "Only now would a compliant/distributed contact "
            "model be justified."
        )

        print(
            "Do not return to bottle-position tuning."
        )


    print()
    print(
        "Summary:"
    )

    print(
        SUMMARY
    )


    # ========================================================
    # GUI — SELECTED ONLY
    # ========================================================

    if not args.no_viewer:

        print()
        print("=" * 122)
        print(
            "V30 AUTOMATIC FIXED-SUPPORT GUI"
        )
        print("=" * 122)


        print()
        print(
            f"Viewing: "
            f"{selected['name']}"
        )

        print(
            "Index finger alpha = 1.0"
        )

        print(
            "UR5e/palm upstream support = FIXED"
        )

        print(
            "Bottle = frozen V5"
        )


        visual_fixed(
            selected_case,
            contact_bag,
        )


    print()
    print("=" * 122)
    print(
        "V30 COMPLETE"
    )
    print("=" * 122)


if __name__ == "__main__":
    main()
