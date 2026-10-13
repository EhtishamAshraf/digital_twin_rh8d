#!/usr/bin/env python3
from __future__ import annotations

import itertools
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

import build_audit_index_spatial_tendon_v34a as b


OUT = (
    b.MJCF_DIR
    / "ur5e_rh8d_index_spatial_v34a2.xml"
)

TMP = (
    b.MJCF_DIR
    / ".ur5e_rh8d_index_spatial_v34a2_tmp.xml"
)


# =============================================================================
# STRUCTURAL AUDIT CONFIGURATION
# =============================================================================

POSES = [
    (5, 5, 2),
    (30, 10, 5),
    (45, 25, 10),
    (60, 50, 20),
]


# Old effective moment arms.
#
# IMPORTANT:
# These are used ONLY as an initialization scale.
# They are NOT imposed on the new model.
TARGET = np.array(
    [
        0.005,
        0.004,
        0.001,
    ],
    dtype=float,
)


# Physically bounded effective guide radii.
#
# These are deliberately small searches around finger-scale
# internal routing radii.
R1 = [
    0.004,
    0.005,
    0.006,
]

R2 = [
    0.003,
    0.004,
    0.005,
]

R3 = [
    0.001,
    0.0015,
    0.002,
    0.0025,
]


# =============================================================================
# XML HELPERS
# =============================================================================

def remove_named(
    root,
    tag,
    names,
):

    names = set(
        names
    )

    for parent in root.iter():

        for child in list(
            parent
        ):

            if (
                child.tag
                ==
                tag
                and
                child.attrib.get(
                    "name"
                )
                in
                names
            ):

                parent.remove(
                    child
                )


def add_wrap(
    body,
    geom_name,
    side_name,
    radius,
    side,
):

    # -------------------------------------------------------------
    # The index joints rotate about local +/-X.
    #
    # MuJoCo cylinders point along local Z by default.
    # This quaternion rotates the cylinder axis onto X.
    # -------------------------------------------------------------

    ET.SubElement(
        body,
        "geom",
        name=geom_name,
        type="cylinder",
        pos="0 0 0",
        quat="0.7071067812 0 0.7071067812 0",
        size=f"{radius:.10g} 0.006",
        rgba="1 0 1 0.22",
        contype="0",
        conaffinity="0",
        group="4",
    )


    # -------------------------------------------------------------
    # sidesite tells MuJoCo which side of the cylinder the tendon
    # must wrap around.
    #
    # This is crucial; otherwise MuJoCo may choose the wrong branch.
    # -------------------------------------------------------------

    y = (
        side
        *
        (
            radius
            +
            0.004
        )
    )


    ET.SubElement(
        body,
        "site",
        name=side_name,
        pos=f"0 {y:.10g} 0",
        size="0.001",
        rgba="1 1 0 0.8",
        group="4",
    )


def build(
    side,
    radii,
    stiffness=0.0,
    damping=0.0,
    springlength=None,
):

    tree = ET.parse(
        b.BASE_MODEL
    )

    root = (
        tree.getroot()
    )


    b.make_index_visuals_solid(
        root
    )


    # V34A2 is still a transmission audit.
    # Bottle contact must not influence it yet.
    b.disable_v5_fixture_for_free_audit(
        root
    )


    # =============================================================
    # MOVING SPOOL CABLE SITE
    # =============================================================

    spool_body = (
        b.body_named(
            root,
            b.SPOOL_BODY,
        )
    )


    if (
        spool_body.find(
            f"./site[@name='{b.SPOOL_SITE}']"
        )
        is None
    ):

        ET.SubElement(
            spool_body,
            "site",
            name=b.SPOOL_SITE,
            pos="0 0 0",
            size="0.0015",
            rgba="1 0 1 1",
        )


    spool_pos = np.array(
        [
            float(x)
            for x
            in spool_body.attrib.get(
                "pos",
                "0 0 0",
            ).split()
        ]
    )


    palm_pos = np.array(
        [
            float(x)
            for x
            in b.site_named(
                root,
                b.PALM_ORIGIN,
            ).attrib[
                "pos"
            ].split()
        ]
    )


    axis = (
        spool_pos
        -
        palm_pos
    )


    axis /= np.linalg.norm(
        axis
    )


    b.joint_named(
        root,
        b.SPOOL,
    ).set(
        "axis",
        " ".join(
            f"{x:.10g}"
            for x
            in axis
        ),
    )


    # =============================================================
    # REMOVE OLD CONSTANT INDEX TENDON
    # =============================================================

    tendon_parent = (
        root.find(
            ".//tendon"
        )
    )


    for child in list(
        tendon_parent
    ):

        if (
            child.attrib.get(
                "name"
            )
            ==
            b.TENDON
        ):

            tendon_parent.remove(
                child
            )


    # Remove anything left from a previous V34A2-generated XML.
    remove_named(
        root,
        "geom",
        {
            "idx_wrap_mcp_v34",
            "idx_wrap_pip_v34",
            "idx_wrap_dip_v34",
        },
    )


    remove_named(
        root,
        "site",
        {
            "idx_side_mcp_v34",
            "idx_side_pip_v34",
            "idx_side_dip_v34",
        },
    )


    # =============================================================
    # CAD-INFORMED INTERNAL FLEXOR SIDE
    # =============================================================

    # Preserve x/z from the RH8D model.
    #
    # We only test which local side of the finger corresponds to
    # the mechanically correct flexor path.

    for name in (
        b.GUIDE_P,
        b.GUIDE_M,
        b.GUIDE_D,
        b.ANCHOR,
    ):

        site = (
            b.site_named(
                root,
                name,
            )
        )


        pos = np.array(
            [
                float(x)
                for x
                in site.attrib[
                    "pos"
                ].split()
            ]
        )


        pos[
            1
        ] = (
            side
            *
            abs(
                pos[
                    1
                ]
            )
        )


        site.set(
            "pos",
            " ".join(
                f"{x:.10g}"
                for x
                in pos
            ),
        )


    # =============================================================
    # JOINT WRAPPING GEOMETRY
    # =============================================================

    add_wrap(
        b.body_named(
            root,
            "Index_Proximal",
        ),
        "idx_wrap_mcp_v34",
        "idx_side_mcp_v34",
        radii[
            0
        ],
        side,
    )


    add_wrap(
        b.body_named(
            root,
            "Index_Middle",
        ),
        "idx_wrap_pip_v34",
        "idx_side_pip_v34",
        radii[
            1
        ],
        side,
    )


    add_wrap(
        b.body_named(
            root,
            "Index_Distal",
        ),
        "idx_wrap_dip_v34",
        "idx_side_dip_v34",
        radii[
            2
        ],
        side,
    )


    # =============================================================
    # PHYSICAL SPATIAL FLEXOR TENDON
    # =============================================================

    attrs = {
        "name":
            b.TENDON,

        "rgba":
            "1 0 1 1",

        "width":
            "0.0015",

        "stiffness":
            f"{stiffness:.10g}",

        "damping":
            f"{damping:.10g}",

        "frictionloss":
            "0",
    }


    if (
        springlength
        is not None
    ):

        attrs[
            "springlength"
        ] = (
            f"{springlength:.10g}"
        )


    tendon = (
        ET.SubElement(
            tendon_parent,
            "spatial",
            attrs,
        )
    )


    ET.SubElement(
        tendon,
        "site",
        site=b.SPOOL_SITE,
    )


    ET.SubElement(
        tendon,
        "site",
        site=b.PALM_ORIGIN,
    )


    ET.SubElement(
        tendon,
        "geom",
        geom="idx_wrap_mcp_v34",
        sidesite="idx_side_mcp_v34",
    )


    ET.SubElement(
        tendon,
        "site",
        site=b.GUIDE_P,
    )


    ET.SubElement(
        tendon,
        "geom",
        geom="idx_wrap_pip_v34",
        sidesite="idx_side_pip_v34",
    )


    ET.SubElement(
        tendon,
        "site",
        site=b.GUIDE_M,
    )


    ET.SubElement(
        tendon,
        "geom",
        geom="idx_wrap_dip_v34",
        sidesite="idx_side_dip_v34",
    )


    ET.SubElement(
        tendon,
        "site",
        site=b.ANCHOR,
    )


    return tree


# =============================================================================
# COMPILE
# =============================================================================

def compile_tree(
    tree,
):

    tree.write(
        TMP,
        encoding="utf-8",
        xml_declaration=True,
    )


    model = (
        mujoco.MjModel.from_xml_path(
            str(
                TMP
            )
        )
    )


    return (
        model,
        mujoco.MjData(
            model
        ),
    )


# =============================================================================
# CANDIDATE AUDIT
# =============================================================================

def audit(
    side,
    radii,
):

    try:

        tree = (
            build(
                side,
                radii,
            )
        )


        model, data = (
            compile_tree(
                tree
            )
        )


        tendon_id = (
            b.id_tendon(
                model,
                b.TENDON,
            )
        )


        all_arms = []

        gains = []


        for pose in POSES:

            arms, gain = (
                b.moment_arms(
                    model,
                    data,
                    tendon_id,
                    pose,
                )
            )


            all_arms.append(
                arms
            )


            gains.append(
                gain
            )


        all_arms = np.asarray(
            all_arms
        )


        gains = np.asarray(
            gains
        )


        # ---------------------------------------------------------
        # HARD STRUCTURAL REQUIREMENTS
        # ---------------------------------------------------------

        if (
            np.any(
                ~np.isfinite(
                    all_arms
                )
            )
            or
            np.any(
                all_arms
                <=
                0.00005
            )
            or
            np.any(
                all_arms
                >
                0.020
            )
            or
            np.any(
                gains
                <=
                0.2
            )
        ):

            return None


        # ---------------------------------------------------------
        # Near-open effective scale.
        #
        # Previous 5/4/1 values are only an initialization prior.
        # ---------------------------------------------------------

        fit = np.sum(
            (
                (
                    all_arms[
                        0
                    ]
                    -
                    TARGET
                )
                /
                np.array(
                    [
                        0.0025,
                        0.0020,
                        0.0015,
                    ]
                )
            )
            ** 2
        )


        # Prefer geometry that remains mechanically well behaved
        # over the whole configuration range.
        variation = np.sum(
            np.std(
                all_arms,
                axis=0,
            )
            /
            np.maximum(
                np.mean(
                    all_arms,
                    axis=0,
                ),
                1e-6,
            )
        )


        # Spool should remain close to one-to-one cable excursion.
        spool_score = (
            20.0
            *
            np.mean(
                (
                    gains
                    -
                    1.0
                )
                ** 2
            )
        )


        score = float(
            fit
            +
            0.35
            *
            variation
            +
            spool_score
        )


        return {
            "score":
                score,

            "side":
                side,

            "radii":
                radii,

            "arms":
                all_arms,

            "gains":
                gains,
        }


    except Exception:

        return None


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print(
        "=" * 110
    )

    print(
        "V34A2 — CAD-INFORMED SPATIAL TENDON "
        "WITH JOINT WRAPPING"
    )

    print(
        "=" * 110
    )


    print()
    print(
        "moving spool -> palm lead -> MCP wrap "
        "-> proximal guide -> PIP wrap "
        "-> middle guide -> DIP wrap -> anchor"
    )


    print()
    print(
        "No contact-triggered switching."
    )

    print(
        "No joint-angle commands."
    )

    print(
        "No object-specific tuning."
    )

    print()


    # =============================================================
    # STRUCTURAL ROUTE SEARCH
    # =============================================================

    rows = []


    for (
        side,
        r1,
        r2,
        r3,
    ) in itertools.product(
        (
            -1,
            +1,
        ),
        R1,
        R2,
        R3,
    ):

        result = audit(
            side,
            (
                r1,
                r2,
                r3,
            ),
        )


        if (
            result
            is not None
        ):

            rows.append(
                result
            )


    if not rows:

        raise RuntimeError(
            "No all-positive wrapped spatial route found. "
            "STOP here: inspect wrap topology/orientation."
        )


    rows.sort(
        key=lambda x:
            x[
                "score"
            ]
    )


    best = (
        rows[
            0
        ]
    )


    print(
        "Top valid wrapped routes:"
    )


    for i, row in enumerate(
        rows[
            :10
        ],
        start=1,
    ):

        arms = (
            row[
                "arms"
            ][
                0
            ]
            *
            1000.0
        )


        radii_mm = (
            np.array(
                row[
                    "radii"
                ]
            )
            *
            1000.0
        )


        print(
            f" {i:02d}. "
            f"score={row['score']:.4f} "
            f"side={row['side']:+d} "
            f"radii="
            f"{radii_mm[0]:.2f}/"
            f"{radii_mm[1]:.2f}/"
            f"{radii_mm[2]:.2f} mm "
            f"near-open arms="
            f"{arms[0]:.3f}/"
            f"{arms[1]:.3f}/"
            f"{arms[2]:.3f} mm "
            f"dL/dx="
            f"{row['gains'][0]:.5f}"
        )


    side = (
        best[
            "side"
        ]
    )


    radii = (
        best[
            "radii"
        ]
    )


    # =============================================================
    # MEASURE OPEN TENDON LENGTH
    # =============================================================

    tree0 = (
        build(
            side,
            radii,
        )
    )


    model0, data0 = (
        compile_tree(
            tree0
        )
    )


    tendon_id0 = (
        b.id_tendon(
            model0,
            b.TENDON,
        )
    )


    b.set_finger_pose(
        model0,
        data0,
        (
            0,
            0,
            0,
        ),
        spool_m=0,
    )


    L0 = (
        b.tendon_length(
            model0,
            data0,
            tendon_id0,
        )
    )


    # =============================================================
    # FINAL ELASTIC V34A2 MODEL
    # =============================================================

    tree = (
        build(
            side,
            radii,
            b.TENDON_STIFFNESS,
            b.TENDON_DAMPING,
            L0,
        )
    )


    tree.write(
        OUT,
        encoding="utf-8",
        xml_declaration=True,
    )


    print()
    print(
        "=" * 110
    )

    print(
        "SELECTED V34A2 ROUTE"
    )

    print(
        "=" * 110
    )


    print(
        f" flexor side sign = "
        f"{side:+d}"
    )


    print(
        " wrap radii MCP/PIP/DIP = "
        f"{radii[0]*1000:.3f} / "
        f"{radii[1]*1000:.3f} / "
        f"{radii[2]*1000:.3f} mm"
    )


    print(
        f" open tendon length L0 = "
        f"{L0*1000:.3f} mm"
    )


    print(
        f" model = {OUT}"
    )


    # =============================================================
    # FINAL CONFIGURATION-DEPENDENT AUDIT
    # =============================================================

    model = (
        mujoco.MjModel.from_xml_path(
            str(
                OUT
            )
        )
    )


    data = (
        mujoco.MjData(
            model
        )
    )


    tendon_id = (
        b.id_tendon(
            model,
            b.TENDON,
        )
    )


    print()
    print(
        "=" * 110
    )

    print(
        "CONFIGURATION-DEPENDENT MOMENT-ARM AUDIT"
    )

    print(
        "=" * 110
    )


    for pose in POSES:

        arms, gain = (
            b.moment_arms(
                model,
                data,
                tendon_id,
                pose,
            )
        )


        print(
            f" q="
            f"{pose[0]:5.1f}/"
            f"{pose[1]:5.1f}/"
            f"{pose[2]:5.1f} deg "
            f"arms="
            f"{arms[0]*1000:7.3f}/"
            f"{arms[1]*1000:7.3f}/"
            f"{arms[2]*1000:7.3f} mm "
            f"dL/dx="
            f"{gain:.5f}"
        )


    # =============================================================
    # NATIVE MUJOCO VISUALIZATION
    # =============================================================

    data = (
        mujoco.MjData(
            model
        )
    )


    frozen = (
        b.create_fixed_state(
            model,
            data,
        )
    )


    actuator_id = (
        b.id_actuator(
            model,
            b.ACTUATOR,
        )
    )


    print()
    print(
        "=" * 110
    )

    print(
        "V34A2 NATIVE MUJOCO VISUALIZATION"
    )

    print(
        "=" * 110
    )


    print(
        "Magenta line = ACTIVE tendon."
    )

    print(
        "Transparent magenta cylinders = effective joint wraps."
    )

    print(
        "Yellow sites = enforced wrap side."
    )

    print(
        "Bottle is non-colliding for this architecture audit."
    )


    print()
    print(
        "Check visually:"
    )

    print(
        "  tendon stays on the internal/flexor side;"
    )

    print(
        "  MCP/PIP/DIP all close in the hardware direction;"
    )

    print(
        "  cable does not jump through the dorsal side;"
    )

    print(
        "  no joint angle is commanded directly."
    )


    viewer = (
        mujoco.viewer.launch_passive(
            model,
            data,
        )
    )


    try:

        body_id = (
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                "Index_Middle",
            )
        )


        mujoco.mj_forward(
            model,
            data,
        )


        if (
            body_id
            >=
            0
        ):

            viewer.cam.lookat[:] = (
                data.xpos[
                    body_id
                ]
            )

            viewer.cam.distance = (
                0.22
            )


        steps = max(
            1,
            int(
                3.0
                /
                model.opt.timestep
            ),
        )


        for k in range(
            steps
        ):

            if not viewer.is_running():

                break


            t = (
                k
                *
                model.opt.timestep
            )


            u = min(
                1.0,
                t
                /
                2.0,
            )


            smooth = (
                3
                *
                u
                *
                u
                -
                2
                *
                u
                *
                u
                *
                u
            )


            data.ctrl[
                actuator_id
            ] = (
                b.VIEW_SPOOL_TARGET
                *
                smooth
            )


            mujoco.mj_step(
                model,
                data,
            )


            b.enforce_fixed_state(
                model,
                data,
                frozen,
            )


            viewer.sync()


            time.sleep(
                model.opt.timestep
            )


        q = np.degrees(
            [
                data.qpos[
                    b.qadr(
                        model,
                        b.MCP,
                    )
                ],

                data.qpos[
                    b.qadr(
                        model,
                        b.PIP,
                    )
                ],

                data.qpos[
                    b.qadr(
                        model,
                        b.DIP,
                    )
                ],
            ]
        )


        print()
        print(
            "Structural free-space pose "
            "@ encoder-equivalent 1000 ticks:"
        )


        print(
            f" MCP/PIP/DIP = "
            f"{q[0]:.3f} / "
            f"{q[1]:.3f} / "
            f"{q[2]:.3f} deg"
        )


        print()
        print(
            "Viewer remains open. "
            "Close it to finish."
        )


        while (
            viewer.is_running()
        ):

            viewer.sync()

            time.sleep(
                0.05
            )


    finally:

        viewer.close()


    try:

        TMP.unlink()


    except FileNotFoundError:

        pass


    print()
    print(
        "=" * 110
    )

    print(
        "V34A2 COMPLETE"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":

    main()
