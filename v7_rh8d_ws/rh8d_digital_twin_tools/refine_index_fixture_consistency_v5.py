#!/usr/bin/env python3

from pathlib import Path
import math
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


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

SOURCE = (
    MJCF_DIR
    / "ur5e_rh8d_index_configdep_geom_v4.xml"
)

CANDIDATE_OUTPUT = (
    MJCF_DIR
    / "ur5e_rh8d_index_configdep_geom_v5_candidate.xml"
)

PASS_OUTPUT = (
    MJCF_DIR
    / "ur5e_rh8d_index_configdep_geom_v5.xml"
)

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "index_configdep_v5"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL NAMES
# ============================================================

BOTTLE = "index_contact_fixture"

PALM = "collision_palm"

IDX_PROX = "collision_idx_prox"
IDX_MID  = "collision_idx_mid"
IDX_TIP  = "collision_idx_tip"

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"


# ============================================================
# TABLETOP ARM POSE
# ============================================================

ARM_STARTUP = {
    "shoulder_pan_joint":   1.4927,
    "shoulder_lift_joint": -0.8678,
    "elbow_joint":         -2.4300,
    "wrist_1_joint":       -0.6596,
    "wrist_2_joint":       -0.0347,
    "wrist_3_joint":        0.1260,
}


# ============================================================
# FINGER STATES
#
# HW_CONTACT is the measured hardware pose change relative
# to the open reference.
# ============================================================

Q_OPEN = (
    0.0,
    0.0,
    0.0,
)

Q_HW = (
    14.5355,
    23.8110,
    9.3270,
)


# ============================================================
# GEOMETRY GATES
# ============================================================

# OPEN must be physically clear.
MIN_OPEN_PALM_MM = 3.0
MIN_OPEN_INDEX_MM = 2.0

# At measured hardware contact pose:
MIN_HW_PALM_MM = 3.0
MIN_HW_PROX_MM = 1.0

# Middle and distal should be near the bottle surface.
# This is intentionally not exact zero because:
#   - collision surfaces are effective approximations
#   - manual angle measurements have uncertainty
#   - MuJoCo uses soft/margin-based contact
STRICT_CONTACT_ABS_MM = 1.5


# ============================================================
# SEARCH
# ============================================================

COARSE_RANGE_MM = 20.0
COARSE_STEP_MM = 1.0

LOCAL_RANGE_MM = 2.0
LOCAL_STEP_MM = 0.1


# ============================================================
# HELPERS
# ============================================================

def obj_id(model, objtype, name):

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


def joint_adr(model, name):

    jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return int(
        model.jnt_qposadr[jid]
    )


def set_state(
    model,
    data,
    qpos0,
    finger_adrs,
    q_deg,
):

    data.qpos[:] = qpos0
    data.qvel[:] = 0.0


    for name, value in ARM_STARTUP.items():

        data.qpos[
            joint_adr(
                model,
                name,
            )
        ] = value


    for adr, deg in zip(
        finger_adrs,
        q_deg,
    ):

        data.qpos[adr] = math.radians(
            float(deg)
        )


    mujoco.mj_forward(
        model,
        data,
    )


def geom_distance_mm(
    model,
    data,
    gid1,
    gid2,
):

    fromto = np.zeros(
        6,
        dtype=float,
    )

    d = mujoco.mj_geomDistance(
        model,
        data,
        gid1,
        gid2,
        1.0,
        fromto,
    )

    return float(
        d * 1000.0
    )


def evaluate_position(
    model,
    data,
    qpos0,
    finger_adrs,
    ids,
    x,
    y,
):

    bottle_gid = ids["bottle"]

    model.geom_pos[
        bottle_gid,
        0
    ] = x

    model.geom_pos[
        bottle_gid,
        1
    ] = y


    # --------------------------------------------------------
    # OPEN
    # --------------------------------------------------------

    set_state(
        model,
        data,
        qpos0,
        finger_adrs,
        Q_OPEN,
    )


    open_dist = {

        "palm":
            geom_distance_mm(
                model,
                data,
                ids["palm"],
                bottle_gid,
            ),

        "prox":
            geom_distance_mm(
                model,
                data,
                ids["prox"],
                bottle_gid,
            ),

        "mid":
            geom_distance_mm(
                model,
                data,
                ids["mid"],
                bottle_gid,
            ),

        "tip":
            geom_distance_mm(
                model,
                data,
                ids["tip"],
                bottle_gid,
            ),
    }


    # Early reject physically invalid OPEN geometry.
    if (
        open_dist["palm"]
        < MIN_OPEN_PALM_MM
    ):
        return None


    if min(
        open_dist["prox"],
        open_dist["mid"],
        open_dist["tip"],
    ) < MIN_OPEN_INDEX_MM:

        return None


    # --------------------------------------------------------
    # MEASURED HARDWARE CONTACT POSE
    # --------------------------------------------------------

    set_state(
        model,
        data,
        qpos0,
        finger_adrs,
        Q_HW,
    )


    hw_dist = {

        "palm":
            geom_distance_mm(
                model,
                data,
                ids["palm"],
                bottle_gid,
            ),

        "prox":
            geom_distance_mm(
                model,
                data,
                ids["prox"],
                bottle_gid,
            ),

        "mid":
            geom_distance_mm(
                model,
                data,
                ids["mid"],
                bottle_gid,
            ),

        "tip":
            geom_distance_mm(
                model,
                data,
                ids["tip"],
                bottle_gid,
            ),
    }


    if (
        hw_dist["palm"]
        < MIN_HW_PALM_MM
    ):
        return None


    if (
        hw_dist["prox"]
        < MIN_HW_PROX_MM
    ):
        return None


    # --------------------------------------------------------
    # SCORE
    #
    # Primary objective:
    #   middle and tip simultaneously approach surface.
    #
    # Small regularization discourages unnecessary fixture
    # movement away from the previous calibrated position.
    # --------------------------------------------------------

    mid = hw_dist["mid"]
    tip = hw_dist["tip"]


    return {
        "x": x,
        "y": y,

        "open": open_dist,
        "hw": hw_dist,

        "score":
            mid * mid
            +
            tip * tip
            +
            0.20
            * (
                mid - tip
            ) ** 2,
    }


def search_grid(
    model,
    data,
    qpos0,
    finger_adrs,
    ids,
    center_x,
    center_y,
    range_mm,
    step_mm,
):

    offsets = np.arange(
        -range_mm,
        range_mm + 0.5 * step_mm,
        step_mm,
    )


    results = []


    for dx_mm in offsets:

        for dy_mm in offsets:

            result = evaluate_position(
                model,
                data,
                qpos0,
                finger_adrs,
                ids,
                center_x
                + dx_mm / 1000.0,
                center_y
                + dy_mm / 1000.0,
            )


            if result is not None:

                results.append(
                    result
                )


    results.sort(
        key=lambda r:
            r["score"]
    )


    return results


def write_model(
    source,
    output,
    x,
    y,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()


    bottle = None


    for geom in root.iter("geom"):

        if (
            geom.attrib.get("name")
            == BOTTLE
        ):

            bottle = geom
            break


    if bottle is None:

        raise RuntimeError(
            f"Missing {BOTTLE}"
        )


    old = [
        float(v)
        for v in bottle.attrib[
            "pos"
        ].split()
    ]


    bottle.set(
        "pos",
        (
            f"{x:.9f} "
            f"{y:.9f} "
            f"{old[2]:.9f}"
        ),
    )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


def render_closeups(
    model_path,
    best,
):

    try:

        model = mujoco.MjModel.from_xml_path(
            str(
                model_path
            )
        )

        data = mujoco.MjData(
            model
        )

        qpos0 = (
            model.qpos0.copy()
        )


        finger_adrs = [
            joint_adr(
                model,
                MCP,
            ),
            joint_adr(
                model,
                PIP,
            ),
            joint_adr(
                model,
                DIP,
            ),
        ]


        bottle_gid = obj_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            BOTTLE,
        )

        prox_gid = obj_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            IDX_PROX,
        )

        mid_gid = obj_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            IDX_MID,
        )

        tip_gid = obj_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            IDX_TIP,
        )


        renderer = mujoco.Renderer(
            model,
            height=480,
            width=640,
        )


        try:
            from PIL import Image, ImageDraw
            use_pil = True

        except ImportError:
            use_pil = False


        for label, q_deg, dist_dict in [

            (
                "open",
                Q_OPEN,
                best["open"],
            ),

            (
                "hw_contact",
                Q_HW,
                best["hw"],
            ),
        ]:

            set_state(
                model,
                data,
                qpos0,
                finger_adrs,
                q_deg,
            )


            # ------------------------------------------------
            # Close-up camera focused on bottle/index region
            # ------------------------------------------------

            centers = np.vstack(
                [
                    data.geom_xpos[
                        bottle_gid
                    ],
                    data.geom_xpos[
                        prox_gid
                    ],
                    data.geom_xpos[
                        mid_gid
                    ],
                    data.geom_xpos[
                        tip_gid
                    ],
                ]
            )


            focus = centers.mean(
                axis=0
            )


            cam = mujoco.MjvCamera()

            cam.type = (
                mujoco.mjtCamera.mjCAMERA_FREE
            )

            cam.lookat[:] = focus

            cam.distance = 0.17
            cam.azimuth = 135.0
            cam.elevation = -35.0


            renderer.update_scene(
                data,
                camera=cam,
            )

            image = renderer.render()


            output = (
                RESULT_DIR
                / f"{label}_closeup.png"
            )


            if use_pil:

                img = Image.fromarray(
                    image
                )

                draw = ImageDraw.Draw(
                    img
                )


                text = (
                    f"{label.upper()}\n"
                    f"MCP {q_deg[0]:.2f} deg\n"
                    f"PIP {q_deg[1]:.2f} deg\n"
                    f"DIP {q_deg[2]:.2f} deg\n"
                    f"prox {dist_dict['prox']:+.2f} mm\n"
                    f"mid  {dist_dict['mid']:+.2f} mm\n"
                    f"tip  {dist_dict['tip']:+.2f} mm"
                )


                draw.rectangle(
                    (
                        8,
                        8,
                        205,
                        130,
                    ),
                    fill=(
                        255,
                        255,
                        255,
                    ),
                )


                draw.multiline_text(
                    (
                        14,
                        14,
                    ),
                    text,
                    fill=(
                        0,
                        0,
                        0,
                    ),
                    spacing=2,
                )


                img.save(
                    output
                )


            else:

                import imageio.v3 as iio

                iio.imwrite(
                    output,
                    image,
                )


            print(
                f"Saved close-up: "
                f"{output}"
            )


        renderer.close()


    except Exception as exc:

        print()
        print(
            "WARNING: geometry search succeeded, "
            "but close-up rendering failed:"
        )

        print(
            str(exc)
        )


def main():

    if not SOURCE.exists():

        raise RuntimeError(
            f"Missing source:\n{SOURCE}"
        )


    model = mujoco.MjModel.from_xml_path(
        str(
            SOURCE
        )
    )

    data = mujoco.MjData(
        model
    )

    qpos0 = model.qpos0.copy()


    finger_adrs = [
        joint_adr(
            model,
            MCP,
        ),
        joint_adr(
            model,
            PIP,
        ),
        joint_adr(
            model,
            DIP,
        ),
    ]


    ids = {

        "bottle":
            obj_id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                BOTTLE,
            ),

        "palm":
            obj_id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                PALM,
            ),

        "prox":
            obj_id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                IDX_PROX,
            ),

        "mid":
            obj_id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                IDX_MID,
            ),

        "tip":
            obj_id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                IDX_TIP,
            ),
    }


    original_pos = (
        model.geom_pos[
            ids["bottle"]
        ].copy()
    )


    print()
    print("=" * 110)
    print(
        "RH8D INDEX — TABLETOP FIXTURE "
        "GEOMETRY CONSISTENCY REFINEMENT V5"
    )
    print("=" * 110)


    print()
    print(
        "Starting bottle position:"
    )

    print(
        original_pos
    )


    # ========================================================
    # COARSE
    # ========================================================

    print()
    print(
        "Coarse XY search:"
    )

    print(
        f"  +/- {COARSE_RANGE_MM:.1f} mm"
    )

    print(
        f"  step {COARSE_STEP_MM:.1f} mm"
    )


    coarse = search_grid(
        model,
        data,
        qpos0,
        finger_adrs,
        ids,
        original_pos[0],
        original_pos[1],
        COARSE_RANGE_MM,
        COARSE_STEP_MM,
    )


    if not coarse:

        print()
        print(
            "NO geometrically valid coarse candidate."
        )

        print(
            "Do not continue to tendon dynamics."
        )

        return


    best_coarse = coarse[0]


    print()
    print(
        "Best coarse:"
    )

    print(
        f"  x = "
        f"{best_coarse['x']:.9f}"
    )

    print(
        f"  y = "
        f"{best_coarse['y']:.9f}"
    )

    print(
        f"  HW mid = "
        f"{best_coarse['hw']['mid']:+.3f} mm"
    )

    print(
        f"  HW tip = "
        f"{best_coarse['hw']['tip']:+.3f} mm"
    )


    # ========================================================
    # LOCAL
    # ========================================================

    print()
    print(
        "Local refinement:"
    )

    print(
        f"  +/- {LOCAL_RANGE_MM:.1f} mm"
    )

    print(
        f"  step {LOCAL_STEP_MM:.1f} mm"
    )


    local = search_grid(
        model,
        data,
        qpos0,
        finger_adrs,
        ids,
        best_coarse["x"],
        best_coarse["y"],
        LOCAL_RANGE_MM,
        LOCAL_STEP_MM,
    )


    if not local:

        raise RuntimeError(
            "Local refinement produced no valid candidates."
        )


    best = local[0]


    strict_pass = (
        abs(
            best["hw"]["mid"]
        )
        <= STRICT_CONTACT_ABS_MM
        and
        abs(
            best["hw"]["tip"]
        )
        <= STRICT_CONTACT_ABS_MM
    )


    # ========================================================
    # SAVE
    # ========================================================

    write_model(
        SOURCE,
        CANDIDATE_OUTPUT,
        best["x"],
        best["y"],
    )


    # Compile candidate.
    mujoco.MjModel.from_xml_path(
        str(
            CANDIDATE_OUTPUT
        )
    )


    if strict_pass:

        shutil.copy2(
            CANDIDATE_OUTPUT,
            PASS_OUTPUT,
        )


    # ========================================================
    # RESULT
    # ========================================================

    dx_mm = (
        best["x"]
        - original_pos[0]
    ) * 1000.0

    dy_mm = (
        best["y"]
        - original_pos[1]
    ) * 1000.0


    print()
    print("=" * 110)
    print(
        "BEST FIXTURE REGISTRATION V5"
    )
    print("=" * 110)


    print()
    print(
        "Bottle position:"
    )

    print(
        f"  x = {best['x']:.9f} m"
    )

    print(
        f"  y = {best['y']:.9f} m"
    )

    print(
        f"  z = {original_pos[2]:.9f} m"
    )


    print()
    print(
        "Change relative to V4:"
    )

    print(
        f"  dx = {dx_mm:+.3f} mm"
    )

    print(
        f"  dy = {dy_mm:+.3f} mm"
    )


    print()
    print(
        "OPEN signed distances:"
    )

    for key in [
        "palm",
        "prox",
        "mid",
        "tip",
    ]:

        print(
            f"  {key:<5} "
            f"{best['open'][key]:+8.3f} mm"
        )


    print()
    print(
        "HW_CONTACT signed distances:"
    )

    for key in [
        "palm",
        "prox",
        "mid",
        "tip",
    ]:

        print(
            f"  {key:<5} "
            f"{best['hw'][key]:+8.3f} mm"
        )


    print()
    print(
        f"Strict middle/tip contact tolerance = "
        f"+/- {STRICT_CONTACT_ABS_MM:.1f} mm"
    )


    print()
    print(
        f"STRICT GEOMETRY PASS = "
        f"{strict_pass}"
    )


    print()
    print(
        "Candidate XML:"
    )

    print(
        CANDIDATE_OUTPUT
    )


    if strict_pass:

        print()
        print(
            "Frozen V5 XML:"
        )

        print(
            PASS_OUTPUT
        )


    # ========================================================
    # TWO CLOSE-UP IMAGES
    # ========================================================

    print()
    print("=" * 110)
    print(
        "CLOSE-UP VISUALIZATION"
    )
    print("=" * 110)


    render_closeups(
        CANDIDATE_OUTPUT,
        best,
    )


    print()
    print("=" * 110)


    if strict_pass:

        print(
            "RESULT: PASS — fixture geometry is "
            "consistent enough to freeze for the "
            "next tendon/motor replay stage."
        )

    else:

        print(
            "RESULT: DO NOT START DYNAMIC FITTING YET."
        )

        print(
            "The best XY fixture registration could not "
            "simultaneously place middle and tip near the "
            "measured hardware contact surface."
        )

        print(
            "Next step would be to inspect collision "
            "surface geometry / hardware-pose uncertainty, "
            "not widen the search blindly."
        )


    print()
    print("=" * 110)


if __name__ == "__main__":
    main()
