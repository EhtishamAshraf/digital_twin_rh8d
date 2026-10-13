#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import scan_index_fixed_support_v30 as v30


# ============================================================
# V32 — DISTRIBUTED COMPLIANT MIDDLE-PHALANX SKIN
#
# Corrected architecture from V30/V31:
#
#   support = FIXED
#   bottle = frozen V5
#   pre-yield transmission = 5 / 4 / 1 mm
#   post-yield transmission = 4 / 8 / 1.5 mm
#   breakaway = 0.015 Nm
#   mu = 0.05
#   contact stiffness = 1000 N/m
#
# V31 ruled out friction as the missing retention mechanism.
#
# V32 adds 3 independently compliant contact patches along
# the middle phalanx while retaining the original rigid
# collision capsule underneath as a hard core.
#
# No contact/object switch is used.
# ============================================================


WS = v30.WS
MJCF_DIR = v30.v24.MJCF_DIR

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "distributed_skin_v32"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY = RESULT_DIR / "summary.csv"

SELECTED_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_distributed_skin_v32_candidate.xml"
)


# ============================================================
# FROZEN CORRECTED TRANSMISSION
# ============================================================

CASE = {

    "name":
        "V30_SELECTED",

    "yield":
        True,

    "arms_mm":
        [4.0, 8.0, 1.5],
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
MIN_OVERLAP = 0.15


MID_CORE = "collision_idx_mid"
TIP = "collision_idx_tip"
BOTTLE = "index_contact_fixture"


# ============================================================
# SKIN GEOMETRY
#
# rigid middle capsule radius = 7.0 mm
# visual phalanx x extent     ~7.87 mm
#
# Effective outer layer ~= 0.8 mm.
#
# Each patch:
# full thickness = 0.8 mm
# compression toward core = +local x
# ============================================================

SKIN_HALF_THICKNESS = 0.00040

SKIN_HALF_WIDTH = 0.0055

SKIN_HALF_LENGTH = 0.0038

SKIN_TRAVEL = 0.0012

SKIN_DAMPING = 0.75

SKIN_MASS = 0.00020


# Along the existing middle capsule centerline.
PATCH_Z = [
    -0.0082,
    -0.01294,
    -0.0177,
]


PATCH_NAMES = [
    "prox",
    "mid",
    "dist",
]


CASES = [

    {
        "name":
            "NO_SKIN_CONTROL",

        "skin":
            False,

        "k":
            -1.0,
    },

    {
        "name":
            "SKIN_K250",

        "skin":
            True,

        "k":
            250.0,
    },

    {
        "name":
            "SKIN_K500",

        "skin":
            True,

        "k":
            500.0,
    },

    {
        "name":
            "SKIN_K1000",

        "skin":
            True,

        "k":
            1000.0,
    },
]


# ============================================================
# XML
# ============================================================

def middle_body_and_core(
    root,
):

    for body in root.iter(
        "body"
    ):

        if (
            body.attrib.get(
                "name"
            )
            !=
            "Index_Middle"
        ):

            continue


        for geom in body.findall(
            "geom"
        ):

            if (
                geom.attrib.get(
                    "name"
                )
                ==
                MID_CORE
            ):

                return (
                    body,
                    geom,
                )


    raise RuntimeError(
        "Could not find Index_Middle / collision_idx_mid."
    )


def build_model(
    output,
    case,
):

    # Start from exact V30 model.
    v30.v24.build_model(
        output
    )


    if not case[
        "skin"
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


    middle_body, core = (
        middle_body_and_core(
            root
        )
    )


    radius = float(
        core.attrib[
            "size"
        ].split()[
            0
        ]
    )


    # Capsule center x from V27 ~= -5.03 mm.
    fromto = np.asarray(
        [
            float(v)
            for v in
            core.attrib[
                "fromto"
            ].split()
        ],
        dtype=float,
    )


    center_x = float(
        0.5
        *
        (
            fromto[
                0
            ]
            +
            fromto[
                3
            ]
        )
    )


    # Pad center such that its inner face begins at rigid
    # capsule surface and outer face extends ~0.8 mm.
    skin_x = (
        center_x
        -
        radius
        -
        SKIN_HALF_THICKNESS
    )


    print()
    print(
        f"[V32 BUILD] {case['name']}"
    )

    print(
        f"  rigid radius = "
        f"{1000*radius:.3f} mm"
    )

    print(
        f"  skin outer thickness = "
        f"{2000*SKIN_HALF_THICKNESS:.3f} mm"
    )

    print(
        f"  element stiffness = "
        f"{case['k']:.1f} N/m"
    )


    for label, z in zip(
        PATCH_NAMES,
        PATCH_Z,
    ):

        body_name = (
            f"idx_mid_skin_{label}_body"
        )

        joint_name = (
            f"idx_mid_skin_{label}_slide"
        )

        geom_name = (
            f"collision_idx_mid_skin_{label}"
        )


        patch_body = ET.SubElement(
            middle_body,
            "body",
            {
                "name":
                    body_name,

                "pos":
                    (
                        f"{skin_x:.9f} "
                        f"0 "
                        f"{z:.9f}"
                    ),
            },
        )


        ET.SubElement(
            patch_body,
            "joint",
            {
                "name":
                    joint_name,

                "type":
                    "slide",

                # Bottle pushes skin inward toward finger.
                "axis":
                    "1 0 0",

                "limited":
                    "true",

                "range":
                    f"0 {SKIN_TRAVEL:.9f}",

                "springref":
                    "0",

                "stiffness":
                    f"{case['k']:.9f}",

                "damping":
                    f"{SKIN_DAMPING:.9f}",

                "armature":
                    "0.000001",
            },
        )


        ET.SubElement(
            patch_body,
            "geom",
            {
                "name":
                    geom_name,

                "type":
                    "box",

                "size":
                    (
                        f"{SKIN_HALF_THICKNESS:.9f} "
                        f"{SKIN_HALF_WIDTH:.9f} "
                        f"{SKIN_HALF_LENGTH:.9f}"
                    ),

                "mass":
                    f"{SKIN_MASS:.9f}",

                "contype":
                    "2",

                "conaffinity":
                    "33",

                "condim":
                    "4",

                # Explicitly keep the validated V31 interface
                # friction for these new collision elements.
                "friction":
                    "0.05 0.005 0.0001",

                # Same basic collision solver family as
                # original middle capsule.
                "solref":
                    "0.05 1.0",

                "solimp":
                    "0.80 0.95 0.001",

                "margin":
                    "0.001",

                # Almost transparent cyan diagnostic skin.
                "rgba":
                    "0 0.8 1 0.10",
            },
        )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


    # Compile now.
    mujoco.MjModel.from_xml_path(
        str(
            output
        )
    )


# ============================================================
# RECORDER
# ============================================================

class SkinRecorder:

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


        def gid(
            name,
        ):

            return mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )


        def jid(
            name,
        ):

            return mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )


        result = {

            "bottle":
                gid(
                    BOTTLE
                ),

            "core":
                gid(
                    MID_CORE
                ),

            "tip":
                gid(
                    TIP
                ),

            "patch_geoms":
                {},

            "patch_joints":
                {},
        }


        for label in PATCH_NAMES:

            result[
                "patch_geoms"
            ][
                label
            ] = gid(
                f"collision_idx_mid_skin_{label}"
            )


            joint = jid(
                f"idx_mid_skin_{label}_slide"
            )


            result[
                "patch_joints"
            ][
                label
            ] = (
                int(
                    model.jnt_qposadr[
                        joint
                    ]
                )
                if joint >= 0
                else -1
            )


        self.cache[
            key
        ] = result

        return result


    @staticmethod
    def force(
        model,
        data,
        contact_index,
    ):

        f = np.zeros(
            6,
            dtype=float,
        )


        mujoco.mj_contactForce(
            model,
            data,
            contact_index,
            f,
        )


        return abs(
            float(
                f[
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


        bottle = ids[
            "bottle"
        ]


        if bottle < 0:

            return


        middle_force = 0.0

        tip_force = 0.0


        active_patch = {

            label:
                False

            for label in PATCH_NAMES
        }


        patch_force = {

            label:
                0.0

            for label in PATCH_NAMES
        }


        core_active = False

        effective_mu = []


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


            if bottle not in pair:

                continue


            other = next(
                (
                    g
                    for g in pair
                    if g != bottle
                ),
                None,
            )


            fn = self.force(
                model,
                data,
                i,
            )


            if (
                other
                ==
                ids[
                    "core"
                ]
                and
                fn > 1.0e-4
            ):

                core_active = True

                middle_force += fn


            for label in PATCH_NAMES:

                if (
                    other
                    ==
                    ids[
                        "patch_geoms"
                    ][
                        label
                    ]
                    and
                    fn > 1.0e-4
                ):

                    active_patch[
                        label
                    ] = True

                    patch_force[
                        label
                    ] += fn

                    middle_force += fn


                    try:

                        effective_mu.append(
                            float(
                                c.friction[
                                    0
                                ]
                            )
                        )

                    except Exception:

                        pass


            if (
                other
                ==
                ids[
                    "tip"
                ]
                and
                fn > 1.0e-4
            ):

                tip_force += fn


        middle_active = (
            core_active
            or
            any(
                active_patch.values()
            )
        )


        compressions = {}


        for label in PATCH_NAMES:

            qadr = ids[
                "patch_joints"
            ][
                label
            ]


            compressions[
                label
            ] = (
                1000.0
                *
                float(
                    data.qpos[
                        qadr
                    ]
                )
                if qadr >= 0
                else 0.0
            )


        self.rows.append(
            {

                "time":
                    float(
                        data.time
                    ),

                "middle_active":
                    middle_active,

                "tip_active":
                    (
                        tip_force
                        >
                        1.0e-4
                    ),

                "core_active":
                    core_active,

                "prox_active":
                    active_patch[
                        "prox"
                    ],

                "mid_active":
                    active_patch[
                        "mid"
                    ],

                "dist_active":
                    active_patch[
                        "dist"
                    ],

                "middle_force_N":
                    middle_force,

                "tip_force_N":
                    tip_force,

                "prox_force_N":
                    patch_force[
                        "prox"
                    ],

                "mid_force_N":
                    patch_force[
                        "mid"
                    ],

                "dist_force_N":
                    patch_force[
                        "dist"
                    ],

                "prox_compression_mm":
                    compressions[
                        "prox"
                    ],

                "mid_compression_mm":
                    compressions[
                        "mid"
                    ],

                "dist_compression_mm":
                    compressions[
                        "dist"
                    ],

                "effective_skin_mu":
                    (
                        float(
                            np.median(
                                effective_mu
                            )
                        )
                        if effective_mu
                        else float(
                            "nan"
                        )
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


        mujoco.mj_step = wrapped


    def restore(self):

        mujoco.mj_step = (
            self.original_step
        )


# ============================================================
# METRICS
# ============================================================

def longest_true(
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

        return {}


    time = np.asarray(
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
                time
            )
        )
    )


    middle = np.asarray(
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


    result = {

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
            longest_true(
                both,
                dt,
            ),

        "middle_after_tip_s":
            middle_after_tip,

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


    for label in PATCH_NAMES:

        mask = np.asarray(
            [
                r[
                    f"{label}_active"
                ]
                for r in rows
            ],
            dtype=bool,
        )


        result[
            f"{label}_duration_s"
        ] = float(
            np.sum(
                mask
            )
            *
            dt
        )


        result[
            f"max_{label}_compression_mm"
        ] = float(
            np.max(
                [
                    r[
                        f"{label}_compression_mm"
                    ]
                    for r in rows
                ]
            )
        )


    mu = [

        r[
            "effective_skin_mu"
        ]

        for r in rows

        if np.isfinite(
            r[
                "effective_skin_mu"
            ]
        )
    ]


    result[
        "effective_skin_mu"
    ] = (
        float(
            np.median(
                mu
            )
        )
        if mu
        else float(
            "nan"
        )
    )


    return result


# ============================================================
# FREE-SPACE GUARD — ONE SKIN MODEL ONLY
# ============================================================

def free_guard(
    free_bag,
):

    guard_case = {

        "name":
            "FREE_SKIN_GUARD",

        "skin":
            True,

        "k":
            500.0,
    }


    contact_xml = (
        MJCF_DIR
        /
        "tmp_v32_free_contact.xml"
    )


    free_xml = (
        MJCF_DIR
        /
        "tmp_v32_free.xml"
    )


    build_model(
        contact_xml,
        guard_case,
    )


    v30.v24.ab.make_free_model(
        contact_xml,
        free_xml,
    )


    law = (
        v30.v24.StatefulTransmission()
    )

    law.install()


    clamp = (
        v30.FixedSupport()
    )

    clamp.install()


    old_scale = (
        v30.v24.base.METERS_PER_TICK
    )

    old_tendon = (
        v30.v24.base.TENDON_STIFFNESS
    )

    old_mu = (
        v30.v24.base.CONTACT_MU
    )

    old_contact_k = (
        v30.v24.base.CONTACT_STIFFNESS
    )


    try:

        v30.v24.base.METERS_PER_TICK = (
            v30.v24.SCALE
        )

        v30.v24.base.TENDON_STIFFNESS = (
            v30.v24.TENDON_K
        )

        v30.v24.base.CONTACT_MU = 0.05

        v30.v24.base.CONTACT_STIFFNESS = (
            v30.v24.CONTACT_K
        )


        law.configure(
            CASE,
            "FREE",
        )


        metrics, _ = (
            v30.v24.base.run_free(
                free_xml,
                free_bag,
            )
        )


    finally:

        clamp.restore()

        law.restore()

        v30.v24.base.METERS_PER_TICK = (
            old_scale
        )

        v30.v24.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v30.v24.base.CONTACT_MU = (
            old_mu
        )

        v30.v24.base.CONTACT_STIFFNESS = (
            old_contact_k
        )


    try:
        contact_xml.unlink()
    except FileNotFoundError:
        pass


    try:
        free_xml.unlink()
    except FileNotFoundError:
        pass


    return float(
        metrics[
            "combined"
        ]
    )


# ============================================================
# CONTACT RUN
# ============================================================

def run_case(
    skin_case,
    contact_bag,
):

    label = (
        skin_case[
            "name"
        ]
        .lower()
    )


    model_path = (
        MJCF_DIR
        /
        f"tmp_v32_{label}.xml"
    )


    build_model(
        model_path,
        skin_case,
    )


    law = (
        v30.v24.StatefulTransmission()
    )

    law.install()


    clamp = (
        v30.FixedSupport()
    )

    clamp.install()


    recorder = (
        SkinRecorder()
    )

    recorder.install()


    old_scale = (
        v30.v24.base.METERS_PER_TICK
    )

    old_tendon = (
        v30.v24.base.TENDON_STIFFNESS
    )

    old_mu = (
        v30.v24.base.CONTACT_MU
    )

    old_contact_k = (
        v30.v24.base.CONTACT_STIFFNESS
    )


    try:

        v30.v24.base.METERS_PER_TICK = (
            v30.v24.SCALE
        )

        v30.v24.base.TENDON_STIFFNESS = (
            v30.v24.TENDON_K
        )

        v30.v24.base.CONTACT_MU = 0.05

        v30.v24.base.CONTACT_STIFFNESS = (
            v30.v24.CONTACT_K
        )


        law.configure(
            CASE,
            "CONTACT",
        )


        metrics, _ = (
            v30.v24.base.run_contact(
                model_path,
                contact_bag,
            )
        )


    finally:

        recorder.restore()

        clamp.restore()

        law.restore()

        v30.v24.base.METERS_PER_TICK = (
            old_scale
        )

        v30.v24.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v30.v24.base.CONTACT_MU = (
            old_mu
        )

        v30.v24.base.CONTACT_STIFFNESS = (
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


    extra = analyse(
        recorder.rows
    )


    result = {

        "name":
            skin_case[
                "name"
            ],

        "skin_k":
            skin_case[
                "k"
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

        **extra,
    }


    try:
        model_path.unlink()
    except FileNotFoundError:
        pass


    return result


# ============================================================
# GUI
# ============================================================

def visual_selected(
    skin_case,
    contact_bag,
):

    build_model(
        SELECTED_MODEL,
        skin_case,
    )


    law = (
        v30.v24.StatefulTransmission()
    )

    law.install()


    clamp = (
        v30.FixedSupport()
    )

    clamp.install()


    old_scale = (
        v30.v24.viewer_tools.base.METERS_PER_TICK
    )

    old_tendon = (
        v30.v24.viewer_tools.base.TENDON_STIFFNESS
    )

    old_mu = (
        v30.v24.viewer_tools.base.CONTACT_MU
    )

    old_k = (
        v30.v24.viewer_tools.base.CONTACT_STIFFNESS
    )


    try:

        v30.v24.viewer_tools.base.METERS_PER_TICK = (
            v30.v24.SCALE
        )

        v30.v24.viewer_tools.base.TENDON_STIFFNESS = (
            v30.v24.TENDON_K
        )

        v30.v24.viewer_tools.base.CONTACT_MU = 0.05

        v30.v24.viewer_tools.base.CONTACT_STIFFNESS = (
            v30.v24.CONTACT_K
        )


        law.configure(
            CASE,
            "CONTACT",
        )


        v30.v24.viewer_tools.visual_replay(
            SELECTED_MODEL,
            contact_bag,
        )


    finally:

        clamp.restore()

        law.restore()

        v30.v24.viewer_tools.base.METERS_PER_TICK = (
            old_scale
        )

        v30.v24.viewer_tools.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v30.v24.viewer_tools.base.CONTACT_MU = (
            old_mu
        )

        v30.v24.viewer_tools.base.CONTACT_STIFFNESS = (
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


    v30.v24.ab.patch_replay_modules()


    print()
    print("=" * 122)
    print(
        "RH8D INDEX — DISTRIBUTED "
        "MIDDLE SKIN V32"
    )
    print("=" * 122)


    print()
    print(
        "Frozen corrected architecture:"
    )

    print(
        "  support = FIXED"
    )

    print(
        "  transmission = 4 / 8 / 1.5 mm after yield"
    )

    print(
        "  bottle = V5 frozen"
    )

    print(
        "  mu = 0.05"
    )

    print(
        "  contact k = 1000 N/m"
    )


    print()
    print(
        "Running one free-space skin guard..."
    )


    free_rmse = free_guard(
        free_bag
    )


    print(
        f"Free RMSE = "
        f"{free_rmse:.3f} deg"
    )


    if (
        free_rmse
        >
        FREE_GATE
    ):

        raise RuntimeError(
            "V32 skin changes free-space model too much."
        )


    results = []


    for i, skin_case in enumerate(
        CASES,
        start=1,
    ):

        print()
        print("=" * 122)

        print(
            f"[{i:02d}/{len(CASES):02d}] "
            f"{skin_case['name']}"
        )

        print("=" * 122)


        result = run_case(
            skin_case,
            contact_bag,
        )


        results.append(
            result
        )


        print()
        print(
            "Pose MCP/PIP/DIP = "
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
            f"  overlap = "
            f"{result['overlap_s']:.3f} s"
        )

        print(
            f"  longest overlap = "
            f"{result['longest_overlap_s']:.3f} s"
        )

        print(
            f"  middle after tip = "
            f"{result['middle_after_tip_s']:.3f} s"
        )


        if skin_case[
            "skin"
        ]:

            print()
            print(
                "Skin element contact duration:"
            )

            print(
                f"  proximal = "
                f"{result['prox_duration_s']:.3f} s"
            )

            print(
                f"  middle   = "
                f"{result['mid_duration_s']:.3f} s"
            )

            print(
                f"  distal   = "
                f"{result['dist_duration_s']:.3f} s"
            )


            print()
            print(
                "Maximum skin compression:"
            )

            print(
                f"  proximal = "
                f"{result['max_prox_compression_mm']:.3f} mm"
            )

            print(
                f"  middle   = "
                f"{result['max_mid_compression_mm']:.3f} mm"
            )

            print(
                f"  distal   = "
                f"{result['max_dist_compression_mm']:.3f} mm"
            )


            print(
                f"effective skin mu = "
                f"{result['effective_skin_mu']:.3f}"
            )


            if (
                not np.isfinite(
                    result[
                        "effective_skin_mu"
                    ]
                )
                or
                abs(
                    result[
                        "effective_skin_mu"
                    ]
                    -
                    0.05
                )
                >
                0.005
            ):

                raise RuntimeError(
                    "V32 skin contact friction is not "
                    "the requested mu=0.05. "
                    "Do not interpret the experiment."
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


    control = results[
        0
    ]


    retained = [
        r
        for r in results
        if (
            r[
                "longest_overlap_s"
            ]
            >=
            MIN_OVERLAP
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
            "PERSISTENT MIDDLE + TIP WRAP"
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
            "NO SKIN PASSED RETENTION"
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


    print()
    print("=" * 122)
    print(
        "V32 SELECTED RESULT"
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
        "Longest simultaneous middle+tip contact:"
    )

    print(
        f"  no-skin control = "
        f"{control['longest_overlap_s']:.3f} s"
    )

    print(
        f"  selected = "
        f"{selected['longest_overlap_s']:.3f} s"
    )


    print(
        f"Middle after first tip = "
        f"{selected['middle_after_tip_s']:.3f} s"
    )


    # ========================================================
    # DECISION
    # ========================================================

    print()
    print("=" * 122)
    print(
        "V32 DECISION"
    )
    print("=" * 122)


    if (
        selected[
            "longest_overlap_s"
        ]
        >=
        MIN_OVERLAP
    ):

        print()
        print(
            "DISTRIBUTED COMPLIANCE RECOVERS "
            "PERSISTENT WRAPPING."
        )


        if (
            selected[
                "rmse_deg"
            ]
            <= 3.0
        ):

            print()
            print(
                "MAJOR STRUCTURAL SUCCESS."
            )

            print()
            print(
                "We now have:"
            )

            print(
                "  - validated free-space behavior"
            )

            print(
                "  - fixed hardware-like support"
            )

            print(
                "  - correct contact-induced redistribution"
            )

            print(
                "  - persistent middle + distal contact"
            )

            print()
            print(
                "STOP architecture development."
            )

            print(
                "Next: narrow identification and held-out "
                "hardware validation."
            )


        else:

            print()
            print(
                "Contact topology is now correct."
            )

            print(
                "Only a small transmission/skin parameter "
                "rebalance is justified."
            )


    elif (
        selected[
            "longest_overlap_s"
        ]
        >
        control[
            "longest_overlap_s"
        ]
        +
        0.05
    ):

        print()
        print(
            "DISTRIBUTED SKIN HELPS SUBSTANTIALLY "
            "BUT DOES NOT YET PASS THE RETENTION GATE."
        )

        print()
        print(
            "Only ONE local refinement around this skin "
            "model is justified."
        )


    else:

        print()
        print(
            "THIS DISTRIBUTED SKIN MODEL DOES NOT "
            "RECOVER WRAPPING."
        )

        print()
        print(
            "Do not run another blind stiffness scan."
        )

        print(
            "Next step would require using actual hardware "
            "surface/compliance measurements rather than "
            "further simulator-only tuning."
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
            "V32 AUTOMATIC SELECTED-SKIN GUI"
        )
        print("=" * 122)


        print()
        print(
            f"Viewing: "
            f"{selected['name']}"
        )

        print(
            "Support = FIXED"
        )

        print(
            "Bottle = frozen V5"
        )

        print(
            "Index visual finger = solid"
        )

        print(
            "Skin patches = faint cyan"
        )


        visual_selected(
            selected_case,
            contact_bag,
        )


    print()
    print("=" * 122)
    print(
        "V32 COMPLETE"
    )
    print("=" * 122)


if __name__ == "__main__":
    main()
