#!/usr/bin/env python3

from pathlib import Path
import argparse
import copy
import csv
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import scan_index_stateful_transmission_v24 as v24


# ============================================================
# V28 — MIDDLE PHALANX MESH-COLLISION TEST
#
# V27 established:
#
# At exact hardware target:
#   middle gap = -0.010 mm
#   tip gap    = -0.014 mm
#
# But dynamically near the same joint angles:
#   middle gap ~= +2.57 mm
#
# Therefore:
#
#   final registration is not the main problem.
#
# Test whether the simplified middle capsule gives the wrong
# rolling/contact path and contact normal.
#
# ONLY CHANGE:
#
#   collision_idx_mid capsule
#           ->
#   existing Index_Middle "phalanx" mesh collision
#
# Everything else stays frozen at V25.
# ============================================================


WS = v24.WS
MJCF_DIR = v24.MJCF_DIR

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "middle_mesh_collision_v28"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY = (
    RESULT_DIR
    / "summary.csv"
)

MESH_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_middle_mesh_collision_v28.xml"
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


HW = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


BOTTLE = "index_contact_fixture"

MID_CAPSULE = "collision_idx_mid"

MID_MESH_COLLISION = (
    "collision_idx_mid_mesh"
)

TIP = "collision_idx_tip"


CASES = [

    {
        "name":
            "CAPSULE_CONTROL",

        "use_mesh":
            False,
    },

    {
        "name":
            "PHALANX_MESH_COLLISION",

        "use_mesh":
            True,
    },
]


# ============================================================
# XML HELPERS
# ============================================================

def get_middle_body(
    root,
):

    for body in root.iter(
        "body"
    ):

        if (
            body.attrib.get(
                "name"
            )
            ==
            "Index_Middle"
        ):

            return body


    raise RuntimeError(
        "Index_Middle body not found."
    )


def find_geom(
    body,
    name,
):

    for geom in body.findall(
        "geom"
    ):

        if (
            geom.attrib.get(
                "name"
            )
            ==
            name
        ):

            return geom


    return None


def find_visual_phalanx_mesh(
    body,
):

    candidates = []


    for geom in body.findall(
        "geom"
    ):

        if (
            geom.attrib.get(
                "mesh"
            )
            ==
            "phalanx"
        ):

            candidates.append(
                geom
            )


    if not candidates:

        raise RuntimeError(
            "Could not find Index_Middle visual "
            "geom using mesh='phalanx'."
        )


    # V27 showed one such visual geom.
    return candidates[
        0
    ]


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(
    output,
    use_mesh,
):

    # Exact V25 mechanical base.
    v24.build_model(
        output
    )


    if not use_mesh:

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


    middle_body = get_middle_body(
        root
    )


    capsule = find_geom(
        middle_body,
        MID_CAPSULE,
    )


    if capsule is None:

        raise RuntimeError(
            f"{MID_CAPSULE} not found."
        )


    visual = (
        find_visual_phalanx_mesh(
            middle_body
        )
    )


    print()
    print(
        "[V28 BUILD] Visual middle geom:"
    )

    print(
        visual.attrib
    )


    print()
    print(
        "[V28 BUILD] Existing capsule:"
    )

    print(
        capsule.attrib
    )


    # --------------------------------------------------------
    # Disable old capsule COLLISION.
    #
    # We also make it visually transparent to avoid confusing
    # the GUI. This does not affect the actual solid visual
    # phalanx mesh.
    # --------------------------------------------------------

    capsule.set(
        "contype",
        "0",
    )

    capsule.set(
        "conaffinity",
        "0",
    )

    capsule.set(
        "rgba",
        "1 0.5 0 0",
    )


    # --------------------------------------------------------
    # Duplicate the actual visual mesh geometry.
    #
    # It has exactly the same pose/orientation/scale as the
    # rendered middle phalanx.
    # --------------------------------------------------------

    mesh_collision = (
        copy.deepcopy(
            visual
        )
    )


    mesh_collision.set(
        "name",
        MID_MESH_COLLISION,
    )


    mesh_collision.set(
        "type",
        "mesh",
    )


    # No duplicated mass/inertia contribution.
    mesh_collision.set(
        "mass",
        "0",
    )


    # --------------------------------------------------------
    # Copy the collision/contact filtering and solver settings
    # from the old middle capsule.
    # --------------------------------------------------------

    mesh_collision.set(
        "contype",
        "2",
    )

    mesh_collision.set(
        "conaffinity",
        "33",
    )


    for attribute in [
        "condim",
        "friction",
        "solref",
        "solimp",
        "margin",
        "priority",
    ]:

        if (
            attribute
            in capsule.attrib
        ):

            mesh_collision.set(
                attribute,
                capsule.attrib[
                    attribute
                ],
            )


    # Remove visual material so the duplicate collision mesh
    # does not cover the solid finger.
    if (
        "material"
        in mesh_collision.attrib
    ):

        del mesh_collision.attrib[
            "material"
        ]


    # Slightly visible cyan collision shell.
    # The actual visual phalanx remains solid/opaque.
    mesh_collision.set(
        "rgba",
        "0 0.8 1 0.06",
    )


    middle_body.append(
        mesh_collision
    )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


    # Compile immediately.
    model = mujoco.MjModel.from_xml_path(
        str(
            output
        )
    )


    gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        MID_MESH_COLLISION,
    )


    if gid < 0:

        raise RuntimeError(
            "Mesh collision geom did not compile."
        )


    print()
    print(
        "[V28 BUILD] compiled middle mesh collision:"
    )

    print(
        f"  geom id = "
        f"{gid}"
    )

    print(
        f"  type = "
        f"{int(model.geom_type[gid])}"
    )

    print(
        f"  contype = "
        f"{int(model.geom_contype[gid])}"
    )

    print(
        f"  conaffinity = "
        f"{int(model.geom_conaffinity[gid])}"
    )


# ============================================================
# CONTACT RECORDER
# ============================================================

class Recorder:

    def __init__(
        self,
        middle_name,
    ):

        self.middle_name = (
            middle_name
        )

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


        result = {

            "bottle":
                gid(
                    BOTTLE
                ),

            "middle":
                gid(
                    self.middle_name
                ),

            "tip":
                gid(
                    TIP
                ),
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
            or
            ids[
                "middle"
            ]
            < 0
            or
            ids[
                "tip"
            ]
            < 0
        ):

            return


        middle_force = 0.0
        tip_force = 0.0

        middle_rows = 0
        tip_rows = 0

        effective_mu = []


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
                    "middle"
                ],
                ids[
                    "bottle"
                ],
            }:

                middle_rows += 1

                middle_force += (
                    self.contact_force(
                        model,
                        data,
                        i,
                    )
                )


                try:

                    effective_mu.append(
                        float(
                            contact.friction[
                                0
                            ]
                        )
                    )

                except Exception:

                    pass


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
                    self.contact_force(
                        model,
                        data,
                        i,
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

                "middle_rows":
                    middle_rows,

                "tip_rows":
                    tip_rows,

                "effective_mu":
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
# RETENTION METRICS
# ============================================================

def longest_true_duration(
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


    return (
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

            "max_middle_force_N":
                0.0,

            "effective_mu":
                float(
                    "nan"
                ),
        }


    times = np.asarray(
        [
            row[
                "time"
            ]
            for row in rows
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
            row[
                "middle_active"
            ]
            for row in rows
        ],
        dtype=bool,
    )


    tip = np.asarray(
        [
            row[
                "tip_active"
            ]
            for row in rows
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


    mu_values = [

        row[
            "effective_mu"
        ]

        for row in rows

        if np.isfinite(
            row[
                "effective_mu"
            ]
        )
    ]


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
            float(
                longest_true_duration(
                    both,
                    dt,
                )
            ),

        "middle_after_tip_s":
            middle_after_tip,

        "max_middle_force_N":
            float(
                np.max(
                    [
                        row[
                            "middle_force_N"
                        ]
                        for row in rows
                    ]
                )
            ),

        "effective_mu":
            (
                float(
                    np.median(
                        mu_values
                    )
                )
                if mu_values
                else float(
                    "nan"
                )
            ),
    }


# ============================================================
# RUN ONE CASE
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
    )


    model_path = (
        MJCF_DIR
        /
        f"tmp_v28_{label}.xml"
    )


    build_model(
        model_path,
        case[
            "use_mesh"
        ],
    )


    middle_name = (
        MID_MESH_COLLISION
        if case[
            "use_mesh"
        ]
        else MID_CAPSULE
    )


    law = (
        v24.StatefulTransmission()
    )


    law.install()


    recorder = Recorder(
        middle_name
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
                model_path,
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


    if (
        case[
            "use_mesh"
        ]
    ):

        # Keep this exact model for GUI.
        import shutil

        shutil.copy2(
            model_path,
            MESH_MODEL,
        )


    try:
        model_path.unlink()
    except FileNotFoundError:
        pass


    return result


# ============================================================
# GUI
# ============================================================

def visual_mesh(
    contact_bag,
):

    # Rebuild in case user reruns GUI independently.
    build_model(
        MESH_MODEL,
        True,
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
            MESH_MODEL,
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


    print()
    print("=" * 122)
    print(
        "RH8D INDEX — MIDDLE PHALANX "
        "MESH COLLISION TEST V28"
    )
    print("=" * 122)


    print()
    print(
        "Everything frozen except middle collision shape."
    )

    print(
        "V25 transmission = 2.5 / 13 / 1.8 mm"
    )

    print(
        "Bottle position = frozen V5"
    )


    results = []


    for i, case in enumerate(
        CASES,
        start=1,
    ):

        print()
        print("=" * 122)

        print(
            f"[{i}/2] "
            f"{case['name']}"
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
            "Pose MCP/PIP/DIP = "
            f"{result['mcp_deg']:.3f} / "
            f"{result['pip_deg']:.3f} / "
            f"{result['dip_deg']:.3f} deg"
        )

        print(
            f"Angle RMSE = "
            f"{result['rmse_deg']:.3f} deg"
        )


        print()
        print(
            "Contact retention:"
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
            f"  total overlap = "
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


        print()
        print(
            f"max middle force = "
            f"{result['max_middle_force_N']:.3f} N"
        )

        print(
            f"runtime effective mu = "
            f"{result['effective_mu']:.3f}"
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

    mesh = results[
        1
    ]


    print()
    print("=" * 122)
    print(
        "V28 CONTROL VS MESH"
    )
    print("=" * 122)


    print()
    print(
        "CAPSULE:"
    )

    print(
        f"  RMSE = "
        f"{control['rmse_deg']:.3f} deg"
    )

    print(
        f"  overlap = "
        f"{control['overlap_s']:.3f} s"
    )

    print(
        f"  longest continuous overlap = "
        f"{control['longest_overlap_s']:.3f} s"
    )


    print()
    print(
        "PHALANX MESH:"
    )

    print(
        f"  RMSE = "
        f"{mesh['rmse_deg']:.3f} deg"
    )

    print(
        f"  overlap = "
        f"{mesh['overlap_s']:.3f} s"
    )

    print(
        f"  longest continuous overlap = "
        f"{mesh['longest_overlap_s']:.3f} s"
    )


    improvement = (
        mesh[
            "longest_overlap_s"
        ]
        -
        control[
            "longest_overlap_s"
        ]
    )


    print()
    print(
        f"Continuous-overlap improvement = "
        f"{improvement:+.3f} s"
    )


    # ========================================================
    # DECISION
    # ========================================================

    print()
    print("=" * 122)
    print(
        "V28 DECISION"
    )
    print("=" * 122)


    if (
        mesh[
            "longest_overlap_s"
        ]
        >= 0.15
        and
        mesh[
            "rmse_deg"
        ]
        <= 3.0
    ):

        print()
        print(
            "MESH CONTACT SOLVES THE PRINCIPAL "
            "WRAP-RETENTION PROBLEM."
        )

        print()
        print(
            "Use the CAD-derived phalanx surface as the "
            "basis of the final collision representation."
        )

        print()
        print(
            "Next: only a narrow contact-material/"
            "compliance identification is needed."
        )


    elif (
        improvement
        >= 0.05
    ):

        print()
        print(
            "MIDDLE SURFACE GEOMETRY MATTERS "
            "SUBSTANTIALLY."
        )

        print()
        print(
            "The mesh improves rolling/contact retention."
        )

        print(
            "Next: refine this CAD-derived contact "
            "representation, not tendon ratios."
        )


    else:

        print()
        print(
            "THE PHALANX MESH DOES NOT RESOLVE "
            "DYNAMIC CONTACT RETENTION."
        )

        print()
        print(
            "The remaining missing mechanism is then likely "
            "surface compliance / distributed skin contact / "
            "rolling resistance rather than gross geometry."
        )

        print()
        print(
            "Do NOT move the bottle."
        )

        print(
            "Do NOT change tendon ratios."
        )


    print()
    print(
        "Summary:"
    )

    print(
        SUMMARY
    )


    # ========================================================
    # GUI — ALWAYS SHOW MESH TEST
    # ========================================================

    if not args.no_viewer:

        print()
        print("=" * 122)
        print(
            "V28 AUTOMATIC PHALANX-MESH GUI"
        )
        print("=" * 122)


        print()
        print(
            "Index visual mesh remains solid."
        )

        print(
            "Cyan collision-mesh overlay is nearly "
            "transparent."
        )

        print(
            "Watch whether the middle phalanx rolls along "
            "the bottle instead of peeling away."
        )


        visual_mesh(
            contact_bag,
        )


    print()
    print("=" * 122)
    print(
        "V28 COMPLETE"
    )
    print("=" * 122)


if __name__ == "__main__":
    main()
