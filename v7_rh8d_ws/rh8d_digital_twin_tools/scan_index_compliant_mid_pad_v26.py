#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import scan_index_stateful_transmission_v24 as v24


# ============================================================
# RH8D INDEX — COMPLIANT MIDDLE-PHALANX PAD V26
#
# V25:
#
#   14.918 / 24.219 / 8.862 deg
#   RMSE = 0.420 deg
#
# but:
#
#   middle + tip overlap = 0.009 s
#
# Therefore tendon-ratio tuning is STOPPED.
#
# V26 changes ONLY the effective middle-phalanx contact
# surface by adding a small mechanically compliant pad.
#
# Bottle position remains frozen V5.
#
# V25 stateful transmission is frozen:
#
#   pre-yield  = 5 / 4 / 1 mm
#   post-yield = 2.5 / 13 / 1.8 mm
#
# No object-specific switch is used.
# ============================================================


WS = v24.WS
MJCF_DIR = v24.MJCF_DIR

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "compliant_mid_pad_v26"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY = (
    RESULT_DIR
    / "summary.csv"
)

SELECTED_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_compliant_mid_pad_v26_candidate.xml"
)


# ============================================================
# FROZEN V25 MODEL
# ============================================================

V25_CASE = {
    "name":
        "V25_FROZEN",

    "yield":
        True,

    "arms_mm":
        [2.5, 13.0, 1.8],
}


HW = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


FREE_GATE = 6.085


# ============================================================
# PAD GEOMETRY
#
# Current rigid middle capsule radius ~= 7 mm.
#
# Pad is placed on the negative local-x side of the existing
# middle collision capsule.
#
# Full pad thickness = 2 mm.
# Allowed compression travel = 3 mm.
# ============================================================

PAD_HALF_THICKNESS = 0.0010
PAD_HALF_WIDTH = 0.0070
PAD_HALF_LENGTH = 0.0100

PAD_TRAVEL = 0.0030

PAD_DAMPING = 1.0

PAD_MASS = 0.001


CASES = [

    {
        "name":
            "NO_PAD_V25_CONTROL",

        "pad":
            False,

        "stiffness":
            -1.0,
    },

    {
        "name":
            "PAD_K100",

        "pad":
            True,

        "stiffness":
            100.0,
    },

    {
        "name":
            "PAD_K250",

        "pad":
            True,

        "stiffness":
            250.0,
    },

    {
        "name":
            "PAD_K500",

        "pad":
            True,

        "stiffness":
            500.0,
    },

    {
        "name":
            "PAD_K1000",

        "pad":
            True,

        "stiffness":
            1000.0,
    },
]


MID_GEOM = "collision_idx_mid"
TIP_GEOM = "collision_idx_tip"
BOTTLE_GEOM = "index_contact_fixture"

PAD_BODY = "index_mid_compliant_pad_body"
PAD_JOINT = "index_mid_pad_compression"
PAD_GEOM = "collision_idx_mid_pad"


# ============================================================
# XML HELPERS
# ============================================================

def find_geom_and_parent_body(
    root,
    geom_name,
):

    for body in root.iter(
        "body"
    ):

        for geom in body.findall(
            "geom"
        ):

            if (
                geom.attrib.get(
                    "name"
                )
                ==
                geom_name
            ):

                return geom, body


    raise RuntimeError(
        f"Could not find geom {geom_name}"
    )


def build_model(
    output,
    case,
):

    # Start from exactly the same constant-tendon V5 model
    # used by V24/V25.
    v24.build_model(
        output
    )


    if not case[
        "pad"
    ]:

        mujoco.MjModel.from_xml_path(
            str(
                output
            )
        )

        return


    tree = ET.parse(
        output
    )

    root = tree.getroot()


    mid_geom, parent_body = (
        find_geom_and_parent_body(
            root,
            MID_GEOM,
        )
    )


    # --------------------------------------------------------
    # Existing middle capsule geometry.
    # --------------------------------------------------------

    fromto = np.asarray(
        [
            float(x)
            for x in
            mid_geom.attrib[
                "fromto"
            ].split()
        ],
        dtype=float,
    )


    p1 = fromto[
        :3
    ]

    p2 = fromto[
        3:
    ]


    center = (
        0.5
        *
        (
            p1
            +
            p2
        )
    )


    radius = float(
        mid_geom.attrib[
            "size"
        ].split()[
            0
        ]
    )


    # Existing rigid capsule outer surface is approximately:
    #
    #   center_x - radius
    #
    # Put the pad immediately outside it.
    pad_center = (
        center.copy()
    )


    pad_center[
        0
    ] = (
        center[
            0
        ]
        -
        radius
        -
        PAD_HALF_THICKNESS
    )


    print(
        f"[PAD BUILD] "
        f"{case['name']} "
        f"middle radius="
        f"{1000*radius:.3f} mm "
        f"pad center="
        f"{pad_center}"
    )


    # --------------------------------------------------------
    # Compliant child body.
    # --------------------------------------------------------

    pad_body = ET.SubElement(
        parent_body,
        "body",
        {
            "name":
                PAD_BODY,

            "pos":
                (
                    f"{pad_center[0]:.9f} "
                    f"{pad_center[1]:.9f} "
                    f"{pad_center[2]:.9f}"
                ),
        },
    )


    ET.SubElement(
        pad_body,
        "joint",
        {
            "name":
                PAD_JOINT,

            "type":
                "slide",

            # +x moves the pad inward toward the finger.
            "axis":
                "1 0 0",

            "limited":
                "true",

            "range":
                f"0 {PAD_TRAVEL:.9f}",

            "stiffness":
                f"{case['stiffness']:.9f}",

            "damping":
                f"{PAD_DAMPING:.9f}",

            "springref":
                "0",
        },
    )


    geom_attributes = {

        "name":
            PAD_GEOM,

        "type":
            "box",

        "pos":
            "0 0 0",

        "size":
            (
                f"{PAD_HALF_THICKNESS:.9f} "
                f"{PAD_HALF_WIDTH:.9f} "
                f"{PAD_HALF_LENGTH:.9f}"
            ),

        "mass":
            f"{PAD_MASS:.9f}",

        # Solid diagnostic pad.
        "rgba":
            "0.25 0.25 0.25 1",
    }


    # Copy the existing middle collision filtering/contact
    # attributes so bottle/contact masks remain consistent.
    if (
        "class"
        in mid_geom.attrib
    ):

        geom_attributes[
            "class"
        ] = (
            mid_geom.attrib[
                "class"
            ]
        )


    for attr in [
        "contype",
        "conaffinity",
        "condim",
        "friction",
        "solref",
        "solimp",
        "margin",
        "priority",
    ]:

        if (
            attr
            in mid_geom.attrib
        ):

            geom_attributes[
                attr
            ] = (
                mid_geom.attrib[
                    attr
                ]
            )


    ET.SubElement(
        pad_body,
        "geom",
        geom_attributes,
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


# ============================================================
# CONTACT RECORDER
# ============================================================

class RetentionRecorder:

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


        def geom_id(name):

            return mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )


        bottle = geom_id(
            BOTTLE_GEOM
        )

        mid = geom_id(
            MID_GEOM
        )

        tip = geom_id(
            TIP_GEOM
        )

        pad = geom_id(
            PAD_GEOM
        )


        pad_joint = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            PAD_JOINT,
        )


        pad_qadr = (
            int(
                model.jnt_qposadr[
                    pad_joint
                ]
            )
            if pad_joint >= 0
            else -1
        )


        result = {

            "bottle":
                bottle,

            "mid":
                mid,

            "tip":
                tip,

            "pad":
                pad,

            "pad_qadr":
                pad_qadr,
        }


        self.cache[
            key
        ] = result

        return result


    @staticmethod
    def contact_force(
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


        if (
            ids[
                "bottle"
            ]
            < 0
        ):

            return


        middle_force = 0.0

        tip_force = 0.0

        middle_rows = 0

        tip_rows = 0


        middle_geoms = {
            ids[
                "mid"
            ]
        }


        if (
            ids[
                "pad"
            ]
            >= 0
        ):

            middle_geoms.add(
                ids[
                    "pad"
                ]
            )


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
                ids[
                    "bottle"
                ]
                not in pair
            ):

                continue


            other = next(
                (
                    g
                    for g in pair
                    if g
                    != ids[
                        "bottle"
                    ]
                ),
                None,
            )


            if (
                other
                in middle_geoms
            ):

                middle_rows += 1

                middle_force += (
                    self.contact_force(
                        model,
                        data,
                        i,
                    )
                )


            if (
                other
                ==
                ids[
                    "tip"
                ]
            ):

                tip_rows += 1

                tip_force += (
                    self.contact_force(
                        model,
                        data,
                        i,
                    )
                )


        pad_mm = 0.0


        if (
            ids[
                "pad_qadr"
            ]
            >= 0
        ):

            pad_mm = (
                1000.0
                *
                float(
                    data.qpos[
                        ids[
                            "pad_qadr"
                        ]
                    ]
                )
            )


        self.rows.append(
            {

                "time":
                    float(
                        data.time
                    ),

                "middle_active":
                    (
                        middle_rows > 0
                        and
                        middle_force
                        > 1.0e-4
                    ),

                "tip_active":
                    (
                        tip_rows > 0
                        and
                        tip_force
                        > 1.0e-4
                    ),

                "middle_force_N":
                    middle_force,

                "tip_force_N":
                    tip_force,

                "pad_compression_mm":
                    pad_mm,
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
# ANALYSE RETENTION
# ============================================================

def analyse_rows(
    rows,
):

    if len(
        rows
    ) < 2:

        return {
            "middle_duration_s": 0.0,
            "tip_duration_s": 0.0,
            "overlap_duration_s": 0.0,
            "middle_after_tip_s": 0.0,
            "max_pad_compression_mm": 0.0,
            "max_middle_force_N": 0.0,
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


    mid = np.asarray(
        [
            r[
                "middle_active"
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
        mid
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


        after_tip = float(
            np.sum(
                mid[
                    first_tip:
                ]
            )
            *
            dt
        )


    else:

        after_tip = 0.0


    return {

        "middle_duration_s":
            float(
                np.sum(
                    mid
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

        "overlap_duration_s":
            float(
                np.sum(
                    both
                )
                *
                dt
            ),

        "middle_after_tip_s":
            after_tip,

        "max_pad_compression_mm":
            float(
                np.max(
                    [
                        r[
                            "pad_compression_mm"
                        ]
                        for r in rows
                    ]
                )
            ),

        "max_middle_force_N":
            float(
                np.max(
                    [
                        r[
                            "middle_force_N"
                        ]
                        for r in rows
                    ]
                )
            ),
    }


# ============================================================
# RUN CANDIDATE
# ============================================================

def run_candidate(
    case,
    free_bag,
    contact_bag,
):

    label = (
        case[
            "name"
        ]
        .lower()
    )


    contact_xml = (
        MJCF_DIR
        /
        f"tmp_v26_{label}_contact.xml"
    )


    free_xml = (
        MJCF_DIR
        /
        f"tmp_v26_{label}_free.xml"
    )


    build_model(
        contact_xml,
        case,
    )


    v24.ab.make_free_model(
        contact_xml,
        free_xml,
    )


    law = (
        v24.StatefulTransmission()
    )


    law.install()


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


        # ----------------------------------------------------
        # FREE SPACE
        # ----------------------------------------------------

        law.configure(
            V25_CASE,
            "FREE",
        )


        free_metrics, _ = (
            v24.base.run_free(
                free_xml,
                free_bag,
            )
        )


        free_event = (
            v24.first_event(
                law.events,
                V25_CASE[
                    "name"
                ],
                "FREE",
            )
        )


        # ----------------------------------------------------
        # CONTACT
        # ----------------------------------------------------

        recorder = (
            RetentionRecorder()
        )


        recorder.install()


        try:

            law.configure(
                V25_CASE,
                "CONTACT",
            )


            contact_metrics, _ = (
                v24.base.run_contact(
                    contact_xml,
                    contact_bag,
                )
            )


        finally:

            recorder.restore()


    finally:

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
        contact_metrics[
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


    retention = analyse_rows(
        recorder.rows
    )


    result = {

        "name":
            case[
                "name"
            ],

        "pad_stiffness_N_per_m":
            case[
                "stiffness"
            ],

        "free_rmse":
            float(
                free_metrics[
                    "combined"
                ]
            ),

        "free_breakaway":
            free_event is not None,

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

        "contact_rmse":
            rmse,

        **retention,
    }


    try:
        contact_xml.unlink()
    except FileNotFoundError:
        pass


    try:
        free_xml.unlink()
    except FileNotFoundError:
        pass


    return result


# ============================================================
# VISUAL REPLAY
# ============================================================

def visual_replay(
    case,
    contact_bag,
):

    build_model(
        SELECTED_MODEL,
        case,
    )


    law = (
        v24.StatefulTransmission()
    )


    law.install()


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
            V25_CASE,
            "CONTACT",
        )


        v24.viewer_tools.visual_replay(
            SELECTED_MODEL,
            contact_bag,
        )


    finally:

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


    v24.ab.patch_replay_modules()


    print()
    print("=" * 122)
    print(
        "RH8D INDEX — COMPLIANT "
        "MIDDLE-PHALANX PAD V26"
    )
    print("=" * 122)


    print()
    print(
        "FROZEN:"
    )

    print(
        "  bottle position = V5"
    )

    print(
        "  V25 transmission = "
        "2.5 / 13 / 1.8 mm after yield"
    )

    print(
        "  breakaway threshold = 0.015 Nm"
    )

    print(
        "  contact mu = 0.05"
    )

    print(
        "  contact solver stiffness = 1000 N/m"
    )


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

        print("=" * 122)


        result = run_candidate(
            case,
            free_bag,
            contact_bag,
        )


        results.append(
            result
        )


        print()
        print(
            f"Free RMSE = "
            f"{result['free_rmse']:.3f} deg"
        )

        print(
            f"Free breakaway = "
            f"{result['free_breakaway']}"
        )


        print()
        print(
            "Contact MCP/PIP/DIP = "
            f"{result['mcp_deg']:.3f} / "
            f"{result['pip_deg']:.3f} / "
            f"{result['dip_deg']:.3f} deg"
        )

        print(
            f"Contact RMSE = "
            f"{result['contact_rmse']:.3f} deg"
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
            f"  middle+tip overlap = "
            f"{result['overlap_duration_s']:.3f} s"
        )

        print(
            f"  middle after tip = "
            f"{result['middle_after_tip_s']:.3f} s"
        )

        print(
            f"  max pad compression = "
            f"{result['max_pad_compression_mm']:.3f} mm"
        )

        print(
            f"  max middle force = "
            f"{result['max_middle_force_N']:.3f} N"
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
    # VALID SET
    # ========================================================

    valid = [
        r
        for r in results
        if (
            r[
                "free_rmse"
            ]
            <= FREE_GATE
            and
            not r[
                "free_breakaway"
            ]
        )
    ]


    if not valid:

        raise RuntimeError(
            "No V26 model preserves free space."
        )


    retained = [
        r
        for r in valid
        if (
            r[
                "overlap_duration_s"
            ]
            >= 0.15
        )
    ]


    if retained:

        selected = min(
            retained,
            key=lambda r:
                r[
                    "contact_rmse"
                ],
        )

        mode = (
            "PERSISTENT MIDDLE + DISTAL CONTACT"
        )


    else:

        # If the architecture does not yet pass,
        # pick the model with the greatest actual overlap,
        # then angle error.
        selected = max(
            valid,
            key=lambda r:
                (
                    r[
                        "overlap_duration_s"
                    ],
                    -r[
                        "contact_rmse"
                    ],
                ),
        )

        mode = (
            "NO CANDIDATE PASSED RETENTION"
        )


    selected_case = next(
        c
        for c in CASES
        if (
            c[
                "name"
            ]
            ==
            selected[
                "name"
            ]
        )
    )


    control = next(
        r
        for r in results
        if (
            r[
                "name"
            ]
            ==
            "NO_PAD_V25_CONTROL"
        )
    )


    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 122)
    print(
        "V26 SELECTED RESULT"
    )
    print("=" * 122)


    print()
    print(
        f"Selection = "
        f"{mode}"
    )

    print(
        f"Candidate = "
        f"{selected['name']}"
    )


    print()
    print(
        f"Pad stiffness = "
        f"{selected['pad_stiffness_N_per_m']:.1f} N/m"
    )


    print()
    print(
        "Hardware target:"
    )

    print(
        "  14.536 / 23.811 / 9.327 deg"
    )


    print(
        "Prediction:"
    )

    print(
        "  "
        f"{selected['mcp_deg']:.3f} / "
        f"{selected['pip_deg']:.3f} / "
        f"{selected['dip_deg']:.3f} deg"
    )


    print(
        f"RMSE = "
        f"{selected['contact_rmse']:.3f} deg"
    )


    print()
    print(
        f"Middle+tip overlap:"
    )

    print(
        f"  control = "
        f"{control['overlap_duration_s']:.3f} s"
    )

    print(
        f"  selected = "
        f"{selected['overlap_duration_s']:.3f} s"
    )


    print(
        f"Middle after first tip:"
    )

    print(
        f"  control = "
        f"{control['middle_after_tip_s']:.3f} s"
    )

    print(
        f"  selected = "
        f"{selected['middle_after_tip_s']:.3f} s"
    )


    print(
        f"Max pad compression = "
        f"{selected['max_pad_compression_mm']:.3f} mm"
    )


    # ========================================================
    # DECISION
    # ========================================================

    print()
    print("=" * 122)
    print(
        "V26 DECISION"
    )
    print("=" * 122)


    if (
        selected[
            "overlap_duration_s"
        ]
        >= 0.15
        and
        selected[
            "contact_rmse"
        ]
        <= 3.0
    ):

        print()
        print(
            "CONTACT-RETENTION ARCHITECTURE WORKS."
        )

        print()
        print(
            "The compliant middle pad preserves the "
            "identified joint redistribution while producing "
            "sustained middle + distal contact."
        )

        print()
        print(
            "STOP adding model structures."
        )

        print(
            "Next: narrow identification of pad stiffness/"
            "geometry and then held-out hardware validation."
        )


    elif (
        selected[
            "overlap_duration_s"
        ]
        >
        control[
            "overlap_duration_s"
        ]
        +
        0.05
    ):

        print()
        print(
            "COMPLIANCE HELPS CONTACT RETENTION."
        )

        print(
            "Only one small local pad refinement is justified."
        )


    else:

        print()
        print(
            "THE SIMPLE COMPLIANT PAD DOES NOT FIX "
            "CONTACT RETENTION."
        )

        print()
        print(
            "Do not tune tendon ratios, friction, or bottle "
            "position again."
        )

        print(
            "Next diagnostic should inspect the actual "
            "hardware middle-phalanx contact surface/geometry "
            "and reproduce that geometry explicitly."
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
            "V26 AUTOMATIC SELECTED GUI"
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
            "Bottle position = frozen V5"
        )


        visual_replay(
            selected_case,
            contact_bag,
        )


    print()
    print("=" * 122)
    print(
        "V26 COMPLETE"
    )
    print("=" * 122)


if __name__ == "__main__":
    main()
