#!/usr/bin/env python3

from pathlib import Path
import csv
import inspect
import math
import shutil
import types
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as base

try:
    import replay_index_tabletop_hardware_contact_v1 as contact_replay
except Exception:
    contact_replay = None


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

V5_SOURCE = (
    MJCF_DIR
    / "ur5e_rh8d_index_configdep_geom_v5.xml"
)

V6_SOURCE = (
    MJCF_DIR
    / "ur5e_rh8d_index_configdep_spool_v6.xml"
)

CONTROL_CONTACT = (
    MJCF_DIR
    / "tmp_v6_ab_constant_contact.xml"
)

CONTROL_FREE = (
    MJCF_DIR
    / "tmp_v6_ab_constant_free.xml"
)

CONFIG_CONTACT = (
    MJCF_DIR
    / "tmp_v6_ab_configdep_contact.xml"
)

CONFIG_FREE = (
    MJCF_DIR
    / "tmp_v6_ab_configdep_free.xml"
)

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "configdep_replay_v6"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_CSV = (
    RESULT_DIR
    / "summary.csv"
)


# ============================================================
# IDENTIFIED / PROVISIONAL PARAMETERS
# ============================================================

METERS_PER_TICK = 5.10e-6
TENDON_STIFFNESS = 2100.0

OPEN_ENCODER = 100.0

# Video-aligned hardware hold was approximately 1029 ticks.
# This is used only to position the spool in the static
# visualization after the dynamic replay has produced its
# predicted joint pose.
VIS_ENCODER = 1029.0


# ============================================================
# HARDWARE CONTACT TARGET
#
# Used ONLY for comparison after simulation.
# Never imposed during dynamic replay.
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
# V5 FIXTURE POSITION
# ============================================================

OLD_BOTTLE_POS = np.array(
    [
        0.503422950,
       -0.372517762,
        0.799000000,
    ]
)

V5_BOTTLE_POS = np.array(
    [
        0.504122950,
       -0.357017762,
        0.799000000,
    ]
)


# ============================================================
# NAMES
# ============================================================

TENDON = "idx_flex_spatial"

SPOOL = "index_motor_spool"

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

BOTTLE = "index_contact_fixture"
SUPPORT = "index_sysid_support"

IDX_PROX = "collision_idx_prox"
IDX_MID = "collision_idx_mid"
IDX_TIP = "collision_idx_tip"


ARM_STARTUP = {
    "shoulder_pan_joint":   1.4927,
    "shoulder_lift_joint": -0.8678,
    "elbow_joint":         -2.4300,
    "wrist_1_joint":       -0.6596,
    "wrist_2_joint":       -0.0347,
    "wrist_3_joint":        0.1260,
}


# ============================================================
# XML BUILDERS
# ============================================================

def find_tendon_section(root):

    section = root.find("tendon")

    if section is None:
        raise RuntimeError(
            "<tendon> section missing"
        )

    return section


def replace_index_with_constant_tendon(
    source,
    output,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    section = find_tendon_section(
        root
    )


    old = None
    old_index = None


    for i, element in enumerate(
        list(section)
    ):

        if (
            element.attrib.get("name")
            == TENDON
        ):

            old = element
            old_index = i
            break


    if old is None:

        raise RuntimeError(
            f"Missing tendon {TENDON}"
        )


    section.remove(
        old
    )


    fixed = ET.Element(
        "fixed",
        {
            "name":
                TENDON,

            "stiffness":
                f"{TENDON_STIFFNESS:.10g}",

            "damping":
                "2.0",

            "springlength":
                "-0.050 0.0",
        },
    )


    ET.SubElement(
        fixed,
        "joint",
        {
            "joint":
                SPOOL,

            "coef":
                "1.0",
        },
    )


    ET.SubElement(
        fixed,
        "joint",
        {
            "joint":
                MCP,

            "coef":
                "-0.00500",
        },
    )


    ET.SubElement(
        fixed,
        "joint",
        {
            "joint":
                PIP,

            "coef":
                "-0.00400",
        },
    )


    ET.SubElement(
        fixed,
        "joint",
        {
            "joint":
                DIP,

            "coef":
                "-0.00100",
        },
    )


    section.insert(
        old_index,
        fixed,
    )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


def make_free_model(
    contact_model,
    free_model,
):

    tree = ET.parse(
        contact_model
    )

    root = tree.getroot()


    for geom in root.iter("geom"):

        name = geom.attrib.get(
            "name",
            "",
        )


        if name in {
            BOTTLE,
            SUPPORT,
        }:

            geom.set(
                "contype",
                "0",
            )

            geom.set(
                "conaffinity",
                "0",
            )


    tree.write(
        free_model,
        encoding="unicode",
        xml_declaration=True,
    )


# ============================================================
# PATCH OLD HARD-CODED BOTTLE POSITION IF THE EXISTING
# REPLAY MODULE EXPOSES IT AS A GLOBAL.
# ============================================================

def replacement_like(
    original,
    new_array,
):

    if isinstance(
        original,
        np.ndarray,
    ):

        return np.asarray(
            new_array,
            dtype=original.dtype,
        )


    if isinstance(
        original,
        tuple,
    ):

        return tuple(
            float(x)
            for x in new_array
        )


    if isinstance(
        original,
        list,
    ):

        return [
            float(x)
            for x in new_array
        ]


    return original


def patch_bottle_globals(
    module,
):

    if module is None:
        return []


    patched = []


    for name, value in list(
        vars(module).items()
    ):

        upper = name.upper()


        if not (
            (
                "BOTTLE" in upper
                or
                "CYLINDER" in upper
            )
            and
            (
                "POS" in upper
                or
                "XY" in upper
                or
                "CENTER" in upper
            )
        ):

            continue


        try:

            arr = np.asarray(
                value,
                dtype=float,
            )

        except Exception:

            continue


        if arr.shape == (3,):

            if np.linalg.norm(
                arr
                - OLD_BOTTLE_POS
            ) < 0.010:

                setattr(
                    module,
                    name,
                    replacement_like(
                        value,
                        V5_BOTTLE_POS,
                    ),
                )

                patched.append(
                    (
                        module.__name__,
                        name,
                        arr.copy(),
                        V5_BOTTLE_POS.copy(),
                    )
                )


        elif arr.shape == (2,):

            if np.linalg.norm(
                arr
                - OLD_BOTTLE_POS[:2]
            ) < 0.010:

                setattr(
                    module,
                    name,
                    replacement_like(
                        value,
                        V5_BOTTLE_POS[:2],
                    ),
                )

                patched.append(
                    (
                        module.__name__,
                        name,
                        arr.copy(),
                        V5_BOTTLE_POS[:2].copy(),
                    )
                )


    return patched


def patch_replay_modules():

    modules = [
        base,
        contact_replay,
    ]


    for value in vars(
        base
    ).values():

        if isinstance(
            value,
            types.ModuleType,
        ):

            path = getattr(
                value,
                "__file__",
                None,
            )


            if (
                path
                and
                str(WS) in path
            ):

                modules.append(
                    value
                )


    unique = []

    seen = set()


    for module in modules:

        if module is None:
            continue

        if id(module) in seen:
            continue

        seen.add(
            id(module)
        )

        unique.append(
            module
        )


    patched = []


    for module in unique:

        patched.extend(
            patch_bottle_globals(
                module
            )
        )


    return patched


# ============================================================
# RUN EXACT EXISTING VALIDATED PIPELINES
# ============================================================

def run_pair(
    label,
    free_xml,
    contact_xml,
    free_bag,
    contact_bag,
):

    old_scale = (
        base.METERS_PER_TICK
    )

    old_stiffness = (
        base.TENDON_STIFFNESS
    )


    try:

        base.METERS_PER_TICK = (
            METERS_PER_TICK
        )

        base.TENDON_STIFFNESS = (
            TENDON_STIFFNESS
        )


        print()
        print(
            f"Running {label} FREE-SPACE replay..."
        )


        (
            free_metrics,
            free_log,
        ) = base.run_free(
            free_xml,
            free_bag,
        )


        print()
        print(
            f"Running {label} CONTACT replay..."
        )


        (
            contact_metrics,
            contact_log,
        ) = base.run_contact(
            contact_xml,
            contact_bag,
        )


    finally:

        base.METERS_PER_TICK = (
            old_scale
        )

        base.TENDON_STIFFNESS = (
            old_stiffness
        )


    (
        RESULT_DIR
        / f"{label.lower()}_free.txt"
    ).write_text(
        free_log
    )


    (
        RESULT_DIR
        / f"{label.lower()}_contact.txt"
    ).write_text(
        contact_log
    )


    return {
        "label":
            label,

        "free":
            free_metrics,

        "contact":
            contact_metrics,
    }


# ============================================================
# RENDER STATIC RECONSTRUCTION OF THE SIMULATED HOLD POSE
#
# IMPORTANT:
# q comes from the replay prediction, NOT from hardware.
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


def joint_adr(
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


def geom_distance_mm(
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
        d * 1000.0
    )


def render_hold(
    model_path,
    sim_pose,
    label,
    contact_rmse,
):

    model = mujoco.MjModel.from_xml_path(
        str(
            model_path
        )
    )

    data = mujoco.MjData(
        model
    )


    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0


    # Tabletop arm configuration.
    for name, value in ARM_STARTUP.items():

        data.qpos[
            joint_adr(
                model,
                name,
            )
        ] = value


    # SIMULATION-PREDICTED finger pose.
    for name, deg in zip(
        [
            MCP,
            PIP,
            DIP,
        ],
        sim_pose,
    ):

        data.qpos[
            joint_adr(
                model,
                name,
            )
        ] = math.radians(
            float(deg)
        )


    # Hardware-aligned measured encoder is used only to
    # display the corresponding spool/tendon configuration.
    spool_displacement = (
        (
            VIS_ENCODER
            - OPEN_ENCODER
        )
        * METERS_PER_TICK
    )


    spool_adr = joint_adr(
        model,
        SPOOL,
    )


    spool_displacement = min(
        float(
            model.jnt_range[
                obj_id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    SPOOL,
                ),
                1,
            ]
        ),
        max(
            0.0,
            spool_displacement,
        ),
    )


    data.qpos[
        spool_adr
    ] = spool_displacement


    mujoco.mj_forward(
        model,
        data,
    )


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


    prox_d = geom_distance_mm(
        model,
        data,
        prox_gid,
        bottle_gid,
    )

    mid_d = geom_distance_mm(
        model,
        data,
        mid_gid,
        bottle_gid,
    )

    tip_d = geom_distance_mm(
        model,
        data,
        tip_gid,
        bottle_gid,
    )


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


    option = mujoco.MjvOption()

    mujoco.mjv_defaultOption(
        option
    )


    try:

        option.flags[
            mujoco.mjtVisFlag.mjVIS_TENDON
        ] = True

    except Exception:

        pass


    try:

        option.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
        ] = False

    except Exception:

        pass


    renderer = mujoco.Renderer(
        model,
        height=480,
        width=640,
    )


    renderer.update_scene(
        data,
        camera=cam,
        scene_option=option,
    )


    image = renderer.render()


    output = (
        RESULT_DIR
        / (
            f"{label.lower()}_"
            "hold_closeup.png"
        )
    )


    try:

        from PIL import (
            Image,
            ImageDraw,
        )


        img = Image.fromarray(
            image
        )

        draw = ImageDraw.Draw(
            img
        )


        text = (
            f"{label}\n"
            f"SIM MCP {sim_pose[0]:.2f} deg\n"
            f"SIM PIP {sim_pose[1]:.2f} deg\n"
            f"SIM DIP {sim_pose[2]:.2f} deg\n"
            f"TARGET 14.54 / 23.81 / 9.33 deg\n"
            f"RMSE {contact_rmse:.2f} deg\n"
            f"encoder ~{VIS_ENCODER:.0f} ticks\n"
            f"spool {spool_displacement*1000.0:.2f} mm\n"
            f"prox gap {prox_d:+.2f} mm\n"
            f"mid gap {mid_d:+.2f} mm\n"
            f"tip gap {tip_d:+.2f} mm"
        )


        draw.rectangle(
            (
                8,
                8,
                315,
                185,
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


    except ImportError:

        import imageio.v3 as iio

        iio.imwrite(
            output,
            image,
        )


    renderer.close()


    print(
        f"Saved close-up: {output}"
    )


    return {
        "prox_gap":
            prox_d,

        "mid_gap":
            mid_d,

        "tip_gap":
            tip_d,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    import sys


    if len(
        sys.argv
    ) != 3:

        print(
            "Usage:\n"
            " python3 compare_index_configdep_replay_v6.py "
            "/path/to/free_space_bag "
            "/path/to/contact_bag"
        )

        raise SystemExit(
            1
        )


    free_bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    contact_bag = Path(
        sys.argv[2]
    ).expanduser().resolve()


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


    if not V5_SOURCE.exists():

        raise RuntimeError(
            f"Missing V5:\n"
            f"{V5_SOURCE}"
        )


    if not V6_SOURCE.exists():

        raise RuntimeError(
            f"Missing V6:\n"
            f"{V6_SOURCE}"
        )


    print()
    print("=" * 120)
    print(
        "RH8D INDEX — CONSTANT vs CONFIGURATION-"
        "DEPENDENT TRANSMISSION A/B REPLAY V6"
    )
    print("=" * 120)


    print()
    print(
        "Shared parameters:"
    )

    print(
        f"  encoder/spool scale = "
        f"{METERS_PER_TICK*1e6:.2f} um/tick"
    )

    print(
        f"  tendon stiffness = "
        f"{TENDON_STIFFNESS:.0f} N/m"
    )

    print(
        "  fixture = frozen V5"
    )

    print(
        "  motor model = existing validated replay"
    )


    # --------------------------------------------------------
    # Build CONTROL.
    # --------------------------------------------------------

    replace_index_with_constant_tendon(
        V5_SOURCE,
        CONTROL_CONTACT,
    )


    make_free_model(
        CONTROL_CONTACT,
        CONTROL_FREE,
    )


    # --------------------------------------------------------
    # Build CONFIGURATION-DEPENDENT model.
    # --------------------------------------------------------

    shutil.copy2(
        V6_SOURCE,
        CONFIG_CONTACT,
    )


    make_free_model(
        CONFIG_CONTACT,
        CONFIG_FREE,
    )


    # Compile all models now.
    for path in [
        CONTROL_CONTACT,
        CONTROL_FREE,
        CONFIG_CONTACT,
        CONFIG_FREE,
    ]:

        mujoco.MjModel.from_xml_path(
            str(
                path
            )
        )


    # --------------------------------------------------------
    # Patch any old replay-level fixture constant.
    # --------------------------------------------------------

    patched = patch_replay_modules()


    print()
    print(
        "Replay bottle-global patches:"
    )


    if patched:

        for (
            module_name,
            variable,
            old,
            new,
        ) in patched:

            print(
                f"  {module_name}.{variable}: "
                f"{old} -> {new}"
            )

    else:

        print(
            "  none found; replay is expected "
            "to use model geometry directly."
        )


    # --------------------------------------------------------
    # RUN CONTROL
    # --------------------------------------------------------

    control = run_pair(
        "CONSTANT",
        CONTROL_FREE,
        CONTROL_CONTACT,
        free_bag,
        contact_bag,
    )


    # --------------------------------------------------------
    # RUN CONFIGURATION-DEPENDENT
    # --------------------------------------------------------

    config = run_pair(
        "CONFIGDEP",
        CONFIG_FREE,
        CONFIG_CONTACT,
        free_bag,
        contact_bag,
    )


    # --------------------------------------------------------
    # Extract.
    # --------------------------------------------------------

    control_pose = np.asarray(
        control[
            "contact"
        ][
            "pose"
        ],
        dtype=float,
    )


    config_pose = np.asarray(
        config[
            "contact"
        ][
            "pose"
        ],
        dtype=float,
    )


    # --------------------------------------------------------
    # Render exactly two close-up images:
    # same replay time, same camera, one per model.
    # --------------------------------------------------------

    print()
    print("=" * 120)
    print(
        "CLOSE-UP A/B VISUALIZATION"
    )
    print("=" * 120)


    control_visual = render_hold(
        CONTROL_CONTACT,
        control_pose,
        "CONSTANT",
        control[
            "contact"
        ][
            "rmse"
        ],
    )


    config_visual = render_hold(
        CONFIG_CONTACT,
        config_pose,
        "CONFIGDEP",
        config[
            "contact"
        ][
            "rmse"
        ],
    )


    # --------------------------------------------------------
    # Save summary.
    # --------------------------------------------------------

    rows = []


    for result, visual in [
        (
            control,
            control_visual,
        ),
        (
            config,
            config_visual,
        ),
    ]:

        pose = np.asarray(
            result[
                "contact"
            ][
                "pose"
            ],
            dtype=float,
        )


        rows.append(
            {
                "model":
                    result[
                        "label"
                    ],

                "free_mcp_rmse":
                    result[
                        "free"
                    ][
                        "mcp_rmse"
                    ],

                "free_pip_rmse":
                    result[
                        "free"
                    ][
                        "pip_rmse"
                    ],

                "free_combined_rmse":
                    result[
                        "free"
                    ][
                        "combined"
                    ],

                "contact_mcp":
                    pose[0],

                "contact_pip":
                    pose[1],

                "contact_dip":
                    pose[2],

                "contact_rmse":
                    result[
                        "contact"
                    ][
                        "rmse"
                    ],

                "middle_duty":
                    result[
                        "contact"
                    ].get(
                        "middle_duty",
                        None,
                    ),

                "dmcp":
                    result[
                        "contact"
                    ].get(
                        "dmcp",
                        None,
                    ),

                "dpip":
                    result[
                        "contact"
                    ].get(
                        "dpip",
                        None,
                    ),

                "ddip":
                    result[
                        "contact"
                    ].get(
                        "ddip",
                        None,
                    ),

                "visual_prox_gap":
                    visual[
                        "prox_gap"
                    ],

                "visual_mid_gap":
                    visual[
                        "mid_gap"
                    ],

                "visual_tip_gap":
                    visual[
                        "tip_gap"
                    ],
            }
        )


    with open(
        SUMMARY_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 120)
    print(
        "A/B RESULTS"
    )
    print("=" * 120)


    for result in [
        control,
        config,
    ]:

        pose = np.asarray(
            result[
                "contact"
            ][
                "pose"
            ],
            dtype=float,
        )


        error = (
            pose
            - HW_POSE
        )


        print()
        print(
            result[
                "label"
            ]
        )

        print(
            "-" * 70
        )


        print(
            "Free-space MCP/PIP/combined RMSE:"
        )

        print(
            f"  "
            f"{result['free']['mcp_rmse']:.3f} / "
            f"{result['free']['pip_rmse']:.3f} / "
            f"{result['free']['combined']:.3f} deg"
        )


        print(
            "Contact predicted pose:"
        )

        print(
            f"  MCP = {pose[0]:.3f} deg"
        )

        print(
            f"  PIP = {pose[1]:.3f} deg"
        )

        print(
            f"  DIP = {pose[2]:.3f} deg"
        )


        print(
            "Contact error SIM-HW:"
        )

        print(
            f"  MCP = {error[0]:+.3f} deg"
        )

        print(
            f"  PIP = {error[1]:+.3f} deg"
        )

        print(
            f"  DIP = {error[2]:+.3f} deg"
        )


        print(
            f"Contact RMSE = "
            f"{result['contact']['rmse']:.3f} deg"
        )


        if (
            result[
                "contact"
            ].get(
                "middle_duty",
                None,
            )
            is not None
        ):

            print(
                f"Middle-contact duty = "
                f"{result['contact']['middle_duty']:.1f}%"
            )


        if (
            result[
                "contact"
            ].get(
                "dpip",
                None,
            )
            is not None
        ):

            print(
                "Middle -> tip redistribution:"
            )

            print(
                f"  dMCP = "
                f"{result['contact']['dmcp']:+.3f} deg"
            )

            print(
                f"  dPIP = "
                f"{result['contact']['dpip']:+.3f} deg"
            )

            print(
                f"  dDIP = "
                f"{result['contact']['ddip']:+.3f} deg"
            )


    free_delta = (
        config[
            "free"
        ][
            "combined"
        ]
        -
        control[
            "free"
        ][
            "combined"
        ]
    )


    contact_delta = (
        config[
            "contact"
        ][
            "rmse"
        ]
        -
        control[
            "contact"
        ][
            "rmse"
        ]
    )


    pip_improvement = (
        abs(
            control_pose[1]
            - HW_POSE[1]
        )
        -
        abs(
            config_pose[1]
            - HW_POSE[1]
        )
    )


    print()
    print("=" * 120)
    print(
        "DIRECT EFFECT OF CONFIGURATION-"
        "DEPENDENT TRANSMISSION"
    )
    print("=" * 120)


    print()
    print(
        f"Free-space combined RMSE change "
        f"(CONFIG - CONSTANT): "
        f"{free_delta:+.3f} deg"
    )


    print(
        f"Contact RMSE change "
        f"(CONFIG - CONSTANT): "
        f"{contact_delta:+.3f} deg"
    )


    print(
        f"PIP absolute-error reduction: "
        f"{pip_improvement:+.3f} deg"
    )


    print()
    print(
        "Hardware contact target:"
    )

    print(
        "  MCP/PIP/DIP = "
        "14.536 / 23.811 / 9.327 deg"
    )


    print()
    print(
        "Summary:"
    )

    print(
        SUMMARY_CSV
    )


    print()
    print(
        "Two close-up images:"
    )

    print(
        RESULT_DIR
        / "constant_hold_closeup.png"
    )

    print(
        RESULT_DIR
        / "configdep_hold_closeup.png"
    )


    print()
    print("=" * 120)


    # --------------------------------------------------------
    # Important sanity condition.
    # --------------------------------------------------------

    control_free = (
        control[
            "free"
        ][
            "combined"
        ]
    )


    if abs(
        control_free
        - 5.085
    ) <= 0.35:

        print(
            "CONTROL SANITY: PASS — constant free-space "
            "result remains consistent with the trusted "
            "dynamic baseline."
        )

    else:

        print(
            "CONTROL SANITY: WARNING — constant free-space "
            "result does not reproduce the trusted ~5.085 "
            "deg baseline closely enough."
        )

        print(
            "Interpret the A/B results cautiously until "
            "that discrepancy is explained."
        )


    print()
    print("=" * 120)


if __name__ == "__main__":
    main()
