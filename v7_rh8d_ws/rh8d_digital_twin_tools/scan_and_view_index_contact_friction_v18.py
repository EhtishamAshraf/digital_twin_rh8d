#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as base
import compare_index_configdep_replay_v6 as ab


# ============================================================
# RH8D INDEX — CONTACT FRICTION V18
#
# PURPOSE
#
# V17 showed:
#
#   effective contact mu = 0.05
#   Ft / (mu Fn) = 1.0 during middle contact
#
# Therefore the middle-link contact is friction-limited.
#
# V18 changes ONLY sliding contact friction.
#
# FROZEN:
#
#   - trusted constant [5,4,1] transmission
#   - encoder/spool scale
#   - tendon stiffness
#   - motor delay/rate model
#   - passive joint mechanics
#   - V5 bottle registration
#   - collision geometry
#   - contact normal stiffness/damping
#
# After the scan, the selected diagnostic candidate is replayed
# automatically in the native MuJoCo viewer using EXACTLY the
# same base.run_contact() command path as the terminal results.
#
# No manually imposed finger angles are used.
# ============================================================


# ============================================================
# PATHS
# ============================================================

WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
).expanduser()

MJCF_DIR = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "contact_friction_v18"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_CSV = (
    RESULT_DIR
    / "summary.csv"
)

BEST_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_contact_friction_v18_candidate.xml"
)


# ============================================================
# TRUSTED MODEL PARAMETERS
# ============================================================

SCALE = 5.10e-6

TENDON_K = 2100.0

OPEN_ENCODER = 100.0


# ============================================================
# CORRECTED RECORDER TIME ALIGNMENT
#
# V16:
#
#   baseline first middle contact
#       recorder elapsed ~5.430 s
#
# Previous hardware replay:
#
#   same event bag time
#       ~36.115486 s
#
# Thus recorder t=0 corresponds approximately to:
#
#   36.115486 - 5.430
#       = 30.685486 s bag time
#
# Hardware video-aligned target:
#
#   bag time ~38.970486
#
# Therefore:
#
#   38.970486 - 30.685486
#       = 8.285 s
#
# This replaces the previously used 6.297 s.
# ============================================================

HOLD_ELAPSED = 8.285


# ============================================================
# HARDWARE CONTACT TARGET
#
# Comparison only.
# Never imposed in MuJoCo.
# ============================================================

HW_POSE = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


# ============================================================
# CONTACT FRICTION CANDIDATES
#
# V17 baseline = 0.05.
#
# These are CALIBRATION candidates, not yet claimed as
# identified physical material coefficients.
# ============================================================

MU_VALUES = [
    0.05,
    0.10,
    0.20,
    0.40,
    0.80,
    1.20,
]


# ============================================================
# NAMES
# ============================================================

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

SPOOL = "index_motor_spool"

BOTTLE = "index_contact_fixture"

PROX = "collision_idx_prox"
MID = "collision_idx_mid"
TIP = "collision_idx_tip"


# ============================================================
# XML HELPERS
# ============================================================

def patch_geom_sliding_friction(
    geom,
    mu,
):

    raw = geom.attrib.get(
        "friction",
        "",
    ).split()


    try:

        values = [
            float(x)
            for x in raw
        ]

    except Exception:

        values = []


    # MuJoCo geom friction:
    #
    #   sliding torsional rolling
    #
    if len(values) < 3:

        values = [
            mu,
            0.005,
            0.0001,
        ]

    else:

        values[0] = mu


    geom.set(
        "friction",
        " ".join(
            f"{x:.10g}"
            for x in values
        ),
    )


def patch_pair_sliding_friction(
    pair,
    mu,
):

    raw = pair.attrib.get(
        "friction",
        "",
    ).split()


    try:

        values = [
            float(x)
            for x in raw
        ]

    except Exception:

        values = []


    # Explicit pair friction can contain five components:
    #
    #   sliding1 sliding2 torsional rolling1 rolling2
    #
    # Preserve the non-sliding terms whenever possible.
    if len(values) < 5:

        values = [
            mu,
            mu,
            0.005,
            0.0001,
            0.0001,
        ]

    else:

        values[0] = mu
        values[1] = mu


    pair.set(
        "friction",
        " ".join(
            f"{x:.10g}"
            for x in values
        ),
    )


def patch_safe_offscreen_size(
    root,
):

    # This is only a safeguard for any offscreen rendering.
    #
    # The interactive viewer below DOES NOT receive a custom
    # window width/height, avoiding the previous desktop-size
    # mismatch problem.

    visual = root.find(
        "visual"
    )


    if visual is None:

        return


    global_visual = visual.find(
        "global"
    )


    if global_visual is None:

        global_visual = ET.SubElement(
            visual,
            "global",
        )


    global_visual.set(
        "offwidth",
        "640",
    )

    global_visual.set(
        "offheight",
        "480",
    )


def build_candidate(
    mu,
    output,
):

    # Start from frozen V5 fixture and restore the trusted
    # constant tendon transmission.

    ab.replace_index_with_constant_tendon(
        ab.V5_SOURCE,
        output,
    )


    tree = ET.parse(
        output
    )

    root = tree.getroot()


    # --------------------------------------------------------
    # Patch all relevant bottle/index geoms as fallback.
    # --------------------------------------------------------

    target_geoms = {
        BOTTLE,
        PROX,
        MID,
        TIP,
    }


    geom_count = 0


    for geom in root.iter(
        "geom"
    ):

        if (
            geom.attrib.get(
                "name"
            )
            in target_geoms
        ):

            patch_geom_sliding_friction(
                geom,
                mu,
            )

            geom_count += 1


    # --------------------------------------------------------
    # Explicit contact-pair friction overrides geom friction.
    #
    # V17's observed effective mu=0.05 despite much higher
    # compiled geom friction indicates that an explicit pair
    # or equivalent contact override is likely present.
    # --------------------------------------------------------

    pair_count = 0


    contact = root.find(
        "contact"
    )


    if contact is not None:

        for pair in contact.findall(
            "pair"
        ):

            g1 = pair.attrib.get(
                "geom1",
                "",
            )

            g2 = pair.attrib.get(
                "geom2",
                "",
            )


            pair_set = {
                g1,
                g2,
            }


            bottle_in_pair = (
                BOTTLE
                in pair_set
            )


            index_in_pair = bool(
                pair_set
                & {
                    PROX,
                    MID,
                    TIP,
                }
            )


            if (
                bottle_in_pair
                and
                index_in_pair
            ):

                patch_pair_sliding_friction(
                    pair,
                    mu,
                )

                pair_count += 1


    patch_safe_offscreen_size(
        root
    )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


    # Compile immediately.
    mujoco.MjModel.from_xml_path(
        str(
            output
        )
    )


    return (
        geom_count,
        pair_count,
    )


# ============================================================
# MUJOCO HELPERS
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


def signed_geom_distance(
    model,
    data,
    g1,
    g2,
):

    fromto = np.zeros(
        6,
        dtype=float,
    )


    return float(
        mujoco.mj_geomDistance(
            model,
            data,
            g1,
            g2,
            1.0,
            fromto,
        )
    )


# ============================================================
# CONTACT RECORDER
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


    def contact_for_geom(
        self,
        model,
        data,
        bottle,
        geom,
    ):

        Fn = 0.0
        Ft = 0.0

        contact_rows = 0

        mus = []


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


            normal = abs(
                float(
                    force[
                        0
                    ]
                )
            )


            tangent = math.hypot(
                float(
                    force[
                        1
                    ]
                ),
                float(
                    force[
                        2
                    ]
                ),
            )


            Fn += normal
            Ft += tangent

            contact_rows += 1


            try:

                mus.append(
                    float(
                        c.friction[
                            0
                        ]
                    )
                )

            except Exception:

                pass


        mu = (
            float(
                np.median(
                    mus
                )
            )
            if mus
            else float(
                "nan"
            )
        )


        capacity = (
            mu
            * Fn
            if (
                Fn > 0.0
                and
                np.isfinite(
                    mu
                )
            )
            else 0.0
        )


        utilization = (
            Ft
            /
            capacity
            if capacity > 1.0e-12
            else 0.0
        )


        return {
            "rows":
                contact_rows,

            "Fn":
                Fn,

            "Ft":
                Ft,

            "mu":
                mu,

            "utilization":
                utilization,
        }


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


        middle = self.contact_for_geom(
            model,
            data,
            ids[
                "bottle"
            ],
            ids[
                "mid"
            ],
        )


        tip = self.contact_for_geom(
            model,
            data,
            ids[
                "bottle"
            ],
            ids[
                "tip"
            ],
        )


        mid_gap = signed_geom_distance(
            model,
            data,
            ids[
                "mid"
            ],
            ids[
                "bottle"
            ],
        )


        tip_gap = signed_geom_distance(
            model,
            data,
            ids[
                "tip"
            ],
            ids[
                "bottle"
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
                    mid_gap
                    * 1000.0,

                "tip_gap_mm":
                    tip_gap
                    * 1000.0,

                "mid_rows":
                    middle[
                        "rows"
                    ],

                "mid_Fn_N":
                    middle[
                        "Fn"
                    ],

                "mid_Ft_N":
                    middle[
                        "Ft"
                    ],

                "mid_mu":
                    middle[
                        "mu"
                    ],

                "mid_utilization":
                    middle[
                        "utilization"
                    ],

                "tip_rows":
                    tip[
                        "rows"
                    ],

                "tip_Fn_N":
                    tip[
                        "Fn"
                    ],

                "tip_Ft_N":
                    tip[
                        "Ft"
                    ],

                "tip_mu":
                    tip[
                        "mu"
                    ],

                "tip_utilization":
                    tip[
                        "utilization"
                    ],
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
# ANALYSIS HELPERS
# ============================================================

def percentile(
    values,
    p,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    if len(
        values
    ) == 0:

        return float(
            "nan"
        )


    return float(
        np.percentile(
            values,
            p,
        )
    )


def closest_elapsed(
    rows,
    target,
):

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


def pose_from_row(
    row,
):

    return np.array(
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


def rmse_pose(
    pose,
):

    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(
                        pose,
                        dtype=float,
                    )
                    -
                    HW_POSE
                )
                ** 2
            )
        )
    )


def first_active(
    rows,
    prefix,
):

    for r in rows:

        if (
            r[
                f"{prefix}_rows"
            ]
            > 0
            and
            r[
                f"{prefix}_Fn_N"
            ]
            > 1.0e-4
        ):

            return r


    return None


# ============================================================
# RUN ONE FRICTION CANDIDATE
# ============================================================

def run_candidate(
    mu,
    bag,
):

    label = (
        f"mu_{mu:.2f}"
        .replace(
            ".",
            "p",
        )
    )


    model_path = (
        MJCF_DIR
        /
        f"tmp_v18_{label}.xml"
    )


    geom_count, pair_count = (
        build_candidate(
            mu,
            model_path,
        )
    )


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


        metrics, log = (
            base.run_contact(
                model_path,
                bag,
            )
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
            f"No samples recorded for mu={mu}"
        )


    # --------------------------------------------------------
    # Save raw timeseries/log
    # --------------------------------------------------------

    csv_path = (
        RESULT_DIR
        /
        f"{label}_timeseries.csv"
    )


    with open(
        csv_path,
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


    (
        RESULT_DIR
        /
        f"{label}_replay.txt"
    ).write_text(
        log
    )


    # --------------------------------------------------------
    # Correctly aligned hold
    # --------------------------------------------------------

    hold = closest_elapsed(
        rows,
        HOLD_ELAPSED,
    )


    hold_pose = pose_from_row(
        hold
    )


    hold_rmse = rmse_pose(
        hold_pose
    )


    # --------------------------------------------------------
    # Active contacts
    # --------------------------------------------------------

    middle_active = [
        r
        for r in rows
        if (
            r[
                "mid_rows"
            ]
            > 0
            and
            r[
                "mid_Fn_N"
            ]
            > 1.0e-4
        )
    ]


    tip_active = [
        r
        for r in rows
        if (
            r[
                "tip_rows"
            ]
            > 0
            and
            r[
                "tip_Fn_N"
            ]
            > 1.0e-4
        )
    ]


    if len(
        rows
    ) > 1:

        dt = np.diff(
            [
                r[
                    "time"
                ]
                for r in rows
            ]
        )

        median_dt = float(
            np.median(
                dt
            )
        )

    else:

        median_dt = 0.0


    middle_duration = (
        len(
            middle_active
        )
        * median_dt
    )


    tip_duration = (
        len(
            tip_active
        )
        * median_dt
    )


    first_mid = first_active(
        rows,
        "mid",
    )


    first_tip = first_active(
        rows,
        "tip",
    )


    t0 = rows[
        0
    ][
        "time"
    ]


    # --------------------------------------------------------
    # Effective MuJoCo contact friction
    # --------------------------------------------------------

    effective_mu = (
        percentile(
            [
                r[
                    "mid_mu"
                ]
                for r in middle_active
                if np.isfinite(
                    r[
                        "mid_mu"
                    ]
                )
            ],
            50,
        )
        if middle_active
        else float(
            "nan"
        )
    )


    util95 = (
        percentile(
            [
                r[
                    "mid_utilization"
                ]
                for r in middle_active
            ],
            95,
        )
        if middle_active
        else float(
            "nan"
        )
    )


    base_pose = np.asarray(
        metrics[
            "pose"
        ],
        dtype=float,
    )


    result = {

        "requested_mu":
            mu,

        "effective_mu":
            effective_mu,

        "patched_geoms":
            geom_count,

        "patched_pairs":
            pair_count,

        "contact_mcp":
            base_pose[
                0
            ],

        "contact_pip":
            base_pose[
                1
            ],

        "contact_dip":
            base_pose[
                2
            ],

        "contact_rmse":
            metrics[
                "rmse"
            ],

        "hold_mcp":
            hold_pose[
                0
            ],

        "hold_pip":
            hold_pose[
                1
            ],

        "hold_dip":
            hold_pose[
                2
            ],

        "hold_rmse":
            hold_rmse,

        "hold_mid_gap_mm":
            hold[
                "mid_gap_mm"
            ],

        "hold_tip_gap_mm":
            hold[
                "tip_gap_mm"
            ],

        "hold_mid_force_N":
            hold[
                "mid_Fn_N"
            ],

        "hold_tip_force_N":
            hold[
                "tip_Fn_N"
            ],

        "middle_duration_s":
            middle_duration,

        "tip_duration_s":
            tip_duration,

        "middle_util_p95":
            util95,

        "max_middle_force_N":
            max(
                [
                    r[
                        "mid_Fn_N"
                    ]
                    for r in rows
                ]
            ),

        "max_tip_force_N":
            max(
                [
                    r[
                        "tip_Fn_N"
                    ]
                    for r in rows
                ]
            ),

        "first_middle_elapsed_s":
            (
                first_mid[
                    "time"
                ]
                - t0
                if first_mid
                is not None
                else None
            ),

        "first_tip_elapsed_s":
            (
                first_tip[
                    "time"
                ]
                - t0
                if first_tip
                is not None
                else None
            ),
    }


    try:

        model_path.unlink()

    except FileNotFoundError:

        pass


    return result


# ============================================================
# INTERACTIVE VIEWER
#
# IMPORTANT:
#
# This uses the SAME base.run_contact() replay function used
# to generate the numerical terminal results.
#
# We only intercept mj_step to:
#
#   - synchronize the GUI
#   - pace approximately in real time
#
# No finger qpos is manually imposed.
#
# NO CUSTOM VIEWER WINDOW WIDTH / HEIGHT IS REQUESTED.
# ============================================================

class RealtimeViewer:

    def __init__(self):

        self.original_step = (
            mujoco.mj_step
        )

        self.viewer = None
        self.target_model_id = None

        self.wall_start = None
        self.sim_start = None

        self.last_sync_sim_time = None

        self.launch_error = None


    def is_target_model(
        self,
        model,
    ):

        bottle_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            BOTTLE,
        )


        mid_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            MID,
        )


        return (
            bottle_id >= 0
            and
            mid_id >= 0
        )


    def launch_if_needed(
        self,
        model,
        data,
    ):

        if self.viewer is not None:

            return


        if self.launch_error is not None:

            return


        if not self.is_target_model(
            model
        ):

            return


        try:

            import mujoco.viewer


            # Native passive viewer:
            #
            # deliberately NO width/height arguments.
            self.viewer = (
                mujoco.viewer.launch_passive(
                    model,
                    data,
                )
            )


            self.target_model_id = id(
                model
            )


            # ----------------------------------------------
            # Close-up camera around index/cylinder.
            # ----------------------------------------------

            bottle = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                BOTTLE,
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


            focus = np.mean(
                np.vstack(
                    [
                        data.geom_xpos[
                            bottle
                        ],
                        data.geom_xpos[
                            mid
                        ],
                        data.geom_xpos[
                            tip
                        ],
                    ]
                ),
                axis=0,
            )


            self.viewer.cam.lookat[:] = (
                focus
            )

            self.viewer.cam.distance = (
                0.17
            )

            self.viewer.cam.azimuth = (
                135.0
            )

            self.viewer.cam.elevation = (
                -35.0
            )


            # Previous debugging showed contact-point graphics
            # can produce distracting orange artifacts.
            try:

                self.viewer.opt.flags[
                    mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
                ] = False

            except Exception:

                pass


            self.wall_start = (
                time.perf_counter()
            )

            self.sim_start = float(
                data.time
            )

            self.last_sync_sim_time = (
                self.sim_start
            )


            print()
            print("=" * 110)
            print(
                "MUJOCO VIEWER OPENED"
            )
            print("=" * 110)

            print()
            print(
                "The exact selected hardware-contact replay "
                "is now running in real time."
            )

            print(
                "No MCP/PIP/DIP angles are manually imposed."
            )

            print(
                "Close the MuJoCo window after inspecting "
                "the replay."
            )


        except Exception as exc:

            self.launch_error = exc


            print()
            print("=" * 110)
            print(
                "VIEWER LAUNCH ERROR"
            )
            print("=" * 110)

            print(
                repr(
                    exc
                )
            )

            print()
            print(
                "Numerical replay will continue, but the "
                "interactive window could not be opened."
            )


    def install(
        self,
    ):

        original = (
            self.original_step
        )

        visual = self


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


            visual.launch_if_needed(
                model,
                data,
            )


            if (
                visual.viewer
                is None
            ):

                return result


            if (
                id(
                    model
                )
                != visual.target_model_id
            ):

                return result


            # User may close window before replay ends.
            if not visual.viewer.is_running():

                return result


            # ----------------------------------------------
            # Real-time pacing.
            # ----------------------------------------------

            sim_elapsed = (
                float(
                    data.time
                )
                -
                visual.sim_start
            )


            wall_elapsed = (
                time.perf_counter()
                -
                visual.wall_start
            )


            sleep_time = (
                sim_elapsed
                -
                wall_elapsed
            )


            if sleep_time > 0.0:

                time.sleep(
                    sleep_time
                )


            # ----------------------------------------------
            # Viewer synchronization at ~60 Hz.
            # ----------------------------------------------

            if (
                visual.last_sync_sim_time
                is None
                or
                float(
                    data.time
                )
                -
                visual.last_sync_sim_time
                >=
                1.0
                /
                60.0
            ):

                visual.viewer.sync()

                visual.last_sync_sim_time = (
                    float(
                        data.time
                    )
                )


            return result


        mujoco.mj_step = wrapped


    def restore(
        self,
    ):

        mujoco.mj_step = (
            self.original_step
        )


    def wait_until_closed(
        self,
    ):

        if (
            self.viewer is None
            or
            not self.viewer.is_running()
        ):

            return


        print()
        print(
            "Replay finished."
        )

        print(
            "The viewer will remain open so you can inspect "
            "the final configuration."
        )

        print(
            "Close the MuJoCo window to finish the script."
        )


        while self.viewer.is_running():

            self.viewer.sync()

            time.sleep(
                0.05
            )


# ============================================================
# VISUAL REPLAY
# ============================================================

def visual_replay(
    model_path,
    bag,
):

    viewer = (
        RealtimeViewer()
    )


    viewer.install()


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
            "AUTOMATIC VISUAL HARDWARE REPLAY"
        )
        print("=" * 110)


        metrics, _ = (
            base.run_contact(
                model_path,
                bag,
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
            "Visual replay numerical result:"
        )

        print(
            f"  MCP = "
            f"{pose[0]:.3f} deg"
        )

        print(
            f"  PIP = "
            f"{pose[1]:.3f} deg"
        )

        print(
            f"  DIP = "
            f"{pose[2]:.3f} deg"
        )

        print(
            f"  RMSE = "
            f"{metrics['rmse']:.3f} deg"
        )


    finally:

        viewer.restore()

        base.METERS_PER_TICK = (
            old_scale
        )

        base.TENDON_STIFFNESS = (
            old_k
        )


    viewer.wait_until_closed()


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
        help=(
            "Run numerical friction scan only; "
            "do not launch MuJoCo GUI."
        ),
    )


    args = parser.parse_args()


    bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )


    if not bag.exists():

        raise RuntimeError(
            f"Missing contact bag:\n"
            f"{bag}"
        )


    # Old replay helpers still contain the earlier fixture
    # constant; force all helpers to the frozen V5 geometry.
    ab.patch_replay_modules()


    print()
    print("=" * 118)
    print(
        "RH8D INDEX — CONTACT FRICTION "
        "CALIBRATION + VISUAL REPLAY V18"
    )
    print("=" * 118)


    print()
    print(
        "Only sliding contact friction is changing."
    )

    print(
        "Tendon, motor, joint passive mechanics, "
        "collision geometry and normal contact law "
        "are frozen."
    )


    print()
    print(
        f"Corrected hardware-aligned recorder time = "
        f"{HOLD_ELAPSED:.3f} s"
    )


    print()
    print(
        "Friction candidates:"
    )

    print(
        MU_VALUES
    )


    results = []


    for i, mu in enumerate(
        MU_VALUES,
        start=1,
    ):

        print()
        print("=" * 118)

        print(
            f"[{i:02d}/{len(MU_VALUES):02d}] "
            f"requested sliding mu = "
            f"{mu:.3f}"
        )

        print("=" * 118)


        result = run_candidate(
            mu,
            bag,
        )


        results.append(
            result
        )


        print()
        print(
            f"effective MuJoCo mu = "
            f"{result['effective_mu']:.3f}"
        )

        print(
            f"patched geoms/pairs = "
            f"{result['patched_geoms']} / "
            f"{result['patched_pairs']}"
        )


        print()
        print(
            "Existing replay contact metric:"
        )

        print(
            f"  pose = "
            f"("
            f"{result['contact_mcp']:.3f}, "
            f"{result['contact_pip']:.3f}, "
            f"{result['contact_dip']:.3f}"
            f") deg"
        )

        print(
            f"  RMSE = "
            f"{result['contact_rmse']:.3f} deg"
        )


        print()
        print(
            f"Corrected hold at "
            f"{HOLD_ELAPSED:.3f} s:"
        )

        print(
            f"  pose = "
            f"("
            f"{result['hold_mcp']:.3f}, "
            f"{result['hold_pip']:.3f}, "
            f"{result['hold_dip']:.3f}"
            f") deg"
        )

        print(
            f"  hold RMSE = "
            f"{result['hold_rmse']:.3f} deg"
        )

        print(
            f"  middle gap/force = "
            f"{result['hold_mid_gap_mm']:+.3f} mm / "
            f"{result['hold_mid_force_N']:.3f} N"
        )

        print(
            f"  tip gap/force = "
            f"{result['hold_tip_gap_mm']:+.3f} mm / "
            f"{result['hold_tip_force_N']:.3f} N"
        )


        print()
        print(
            f"middle contact duration = "
            f"{result['middle_duration_s']:.3f} s"
        )

        print(
            f"tip contact duration = "
            f"{result['tip_duration_s']:.3f} s"
        )

        print(
            f"middle friction-utilization p95 = "
            f"{result['middle_util_p95']:.3f}"
        )

        print(
            f"max middle normal force = "
            f"{result['max_middle_force_N']:.3f} N"
        )

        print(
            f"max tip normal force = "
            f"{result['max_tip_force_N']:.3f} N"
        )


        if (
            result[
                "first_tip_elapsed_s"
            ]
            is None
        ):

            print(
                "tip contact = NEVER"
            )

        else:

            print(
                f"first tip contact = "
                f"{result['first_tip_elapsed_s']:.3f} s"
            )


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(
        SUMMARY_CSV,
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
    # BASELINE TIMING SANITY
    # ========================================================

    baseline = min(
        results,
        key=lambda r:
            abs(
                r[
                    "requested_mu"
                ]
                -
                0.05
            ),
    )


    replay_pose = np.array(
        [
            baseline[
                "contact_mcp"
            ],
            baseline[
                "contact_pip"
            ],
            baseline[
                "contact_dip"
            ],
        ]
    )


    aligned_pose = np.array(
        [
            baseline[
                "hold_mcp"
            ],
            baseline[
                "hold_pip"
            ],
            baseline[
                "hold_dip"
            ],
        ]
    )


    timing_delta = (
        aligned_pose
        -
        replay_pose
    )


    print()
    print("=" * 118)
    print(
        "V18 CORRECTED TIMING SANITY CHECK"
    )
    print("=" * 118)


    print()
    print(
        "Baseline existing replay pose:"
    )

    print(
        f"  "
        f"{replay_pose[0]:.3f} / "
        f"{replay_pose[1]:.3f} / "
        f"{replay_pose[2]:.3f} deg"
    )


    print(
        f"Recorder pose at "
        f"{HOLD_ELAPSED:.3f} s:"
    )

    print(
        f"  "
        f"{aligned_pose[0]:.3f} / "
        f"{aligned_pose[1]:.3f} / "
        f"{aligned_pose[2]:.3f} deg"
    )


    print(
        "Difference:"
    )

    print(
        f"  "
        f"{timing_delta[0]:+.3f} / "
        f"{timing_delta[1]:+.3f} / "
        f"{timing_delta[2]:+.3f} deg"
    )


    timing_ok = (
        np.max(
            np.abs(
                timing_delta
            )
        )
        <= 1.0
    )


    print()
    print(
        f"TIMING ALIGNMENT PASS "
        f"(<=1 deg/joint) = "
        f"{timing_ok}"
    )


    # ========================================================
    # SELECT DIAGNOSTIC BEST
    #
    # We use the existing replay's hardware comparison metric
    # because it is the established contact-replay evaluator.
    #
    # This is still CALIBRATION, not held-out validation.
    # ========================================================

    selected = min(
        results,
        key=lambda r:
            r[
                "contact_rmse"
            ],
    )


    build_candidate(
        selected[
            "requested_mu"
        ],
        BEST_MODEL,
    )


    print()
    print("=" * 118)
    print(
        "V18 DIAGNOSTIC BEST CONTACT-FRICTION CANDIDATE"
    )
    print("=" * 118)


    print()
    print(
        f"requested mu = "
        f"{selected['requested_mu']:.3f}"
    )

    print(
        f"effective MuJoCo mu = "
        f"{selected['effective_mu']:.3f}"
    )


    print()
    print(
        "Contact prediction:"
    )

    print(
        f"  MCP = "
        f"{selected['contact_mcp']:.3f} deg"
    )

    print(
        f"  PIP = "
        f"{selected['contact_pip']:.3f} deg"
    )

    print(
        f"  DIP = "
        f"{selected['contact_dip']:.3f} deg"
    )

    print(
        f"  RMSE = "
        f"{selected['contact_rmse']:.3f} deg"
    )


    print()
    print(
        f"middle contact duration = "
        f"{selected['middle_duration_s']:.3f} s"
    )

    print(
        f"tip contact duration = "
        f"{selected['tip_duration_s']:.3f} s"
    )

    print(
        f"middle utilization p95 = "
        f"{selected['middle_util_p95']:.3f}"
    )


    print()
    print(
        "Hardware target:"
    )

    print(
        "  MCP/PIP/DIP = "
        "14.536 / 23.811 / 9.327 deg"
    )


    print()
    print(
        "Candidate model:"
    )

    print(
        BEST_MODEL
    )


    print()
    print(
        "Summary CSV:"
    )

    print(
        SUMMARY_CSV
    )


    # ========================================================
    # INTERPRETATION BEFORE GUI
    # ========================================================

    pip_gain = (
        selected[
            "contact_pip"
        ]
        -
        baseline[
            "contact_pip"
        ]
    )


    mcp_change = (
        selected[
            "contact_mcp"
        ]
        -
        baseline[
            "contact_mcp"
        ]
    )


    print()
    print("=" * 118)
    print(
        "V18 MECHANICAL EFFECT"
    )
    print("=" * 118)


    print()
    print(
        f"dMCP relative to mu=0.05 = "
        f"{mcp_change:+.3f} deg"
    )

    print(
        f"dPIP relative to mu=0.05 = "
        f"{pip_gain:+.3f} deg"
    )


    if (
        pip_gain > 2.0
        and
        mcp_change < 0.0
    ):

        print()
        print(
            "Higher contact friction produced the desired "
            "redistribution direction: MCP decreased while "
            "PIP increased."
        )


    elif pip_gain > 2.0:

        print()
        print(
            "Higher contact friction increased PIP motion, "
            "but MCP did not decrease."
        )

        print(
            "Contact friction contributes, but it does not "
            "fully reproduce the hardware redistribution."
        )


    else:

        print()
        print(
            "Increasing contact friction did not produce a "
            "large PIP redistribution."
        )

        print(
            "Contact friction alone is insufficient; the "
            "remaining discrepancy lies in the underactuated "
            "tendon/passive-joint mechanics."
        )


    # ========================================================
    # AUTOMATIC GUI REPLAY
    # ========================================================

    if not args.no_viewer:

        print()
        print("=" * 118)
        print(
            "LAUNCHING SELECTED CANDIDATE IN MUJOCO"
        )
        print("=" * 118)

        print()
        print(
            f"Selected mu = "
            f"{selected['requested_mu']:.3f}"
        )

        print(
            "The simulator will replay the SAME hardware "
            "contact command used for the terminal metrics."
        )


        visual_replay(
            BEST_MODEL,
            bag,
        )


    print()
    print("=" * 118)
    print(
        "V18 COMPLETE"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
