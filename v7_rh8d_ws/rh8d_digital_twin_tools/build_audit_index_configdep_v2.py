#!/usr/bin/env python3

from pathlib import Path
import csv
import math
import random
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


# =====================================================================
# RH8D INDEX — CONFIGURATION-DEPENDENT SPATIAL TENDON SEARCH V2
#
# Improvements over V1:
#
#   - search palm origin y,z
#   - search proximal guide y,z
#   - search middle guide y,z
#   - search distal guide y,z
#   - search anchor y,z
#
#   - HARD reject wrong flexion sign
#   - HARD reject unreasonable open moment arms
#   - HARD reject extreme moment arms
#   - require PIP leverage growth with configuration
#
# This is still GEOMETRY ONLY.
#
# No dynamic fitting.
# No contact-force switch.
# No final identified XML is modified.
# =====================================================================


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
    / "ur5e_rh8d_index_tabletop_sysid_v1.xml"
)

OUTPUT = (
    MJCF_DIR
    / "ur5e_rh8d_index_configdep_geom_v2.xml"
)

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "index_configdep_v2"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_OUT = (
    RESULT_DIR
    / "moment_arm_candidates_v2.csv"
)


# =====================================================================
# Names
# =====================================================================

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

TENDON = "idx_flex_spatial"


# =====================================================================
# Existing approximate geometric reference
#
# x remains fixed in V2.
#
# y,z are searched.
# =====================================================================

ORIGIN_X = -0.0520

GUIDE_X = -0.0100

ANCHOR_X = -0.0055


# Original/reference coordinates.
REF = {

    "origin_y": -0.0090,
    "origin_z":  0.0685,

    "prox_y":   0.0065,
    "prox_z":   0.0120,

    "mid_y":    0.0065,
    "mid_z":    0.0100,

    "dist_y":   0.0085,
    "dist_z":   0.0080,

    "anchor_y": 0.0015,
    "anchor_z": -0.0120,
}


# =====================================================================
# Search bounds
#
# Kept local to finger/palm geometry.
# =====================================================================

BOUNDS = {

    # Palm-local coordinates
    "origin_y": (-0.0200,  0.0080),
    "origin_z": ( 0.0520,  0.0840),

    # Link-local coordinates
    "prox_y":   (-0.0140,  0.0140),
    "prox_z":   (-0.0040,  0.0200),

    "mid_y":    (-0.0140,  0.0140),
    "mid_z":    (-0.0040,  0.0200),

    "dist_y":   (-0.0140,  0.0140),
    "dist_z":   (-0.0040,  0.0180),

    "anchor_y": (-0.0120,  0.0120),
    "anchor_z": (-0.0200,  0.0040),
}


PARAM_NAMES = list(
    BOUNDS.keys()
)


# =====================================================================
# Configurations used only to inspect r(q)
# =====================================================================

Q_OPEN = np.array(
    [0.0, 0.0, 0.0]
)

Q_MID = np.array(
    [30.0, 15.0, 5.0]
)

# Measured hardware contact configuration.
Q_HW_CONTACT = np.array(
    [14.5355, 23.8110, 9.3270]
)

Q_FLEXED = np.array(
    [50.0, 45.0, 15.0]
)


# =====================================================================
# Open-state reference
#
# This comes from the validated constant free-space family.
#
# We do NOT require exact equality.
# =====================================================================

TARGET_OPEN = np.array(
    [-5.0, -4.0, -1.0]
)


# =====================================================================
# Structural gates
# =====================================================================

OPEN_MIN = np.array(
    [3.0, 2.0, 0.25]
)

OPEN_MAX = np.array(
    [8.0, 7.0, 4.0]
)


MAX_ANY_ARM_MM = 14.0


# Require configuration-induced increase in |PIP leverage|.
MIN_PIP_GROWTH = 1.20


# Avoid destroying MCP leverage.
MIN_MCP_CONTACT_RATIO = 0.55
MAX_MCP_CONTACT_RATIO = 1.80


# Avoid huge DIP amplification.
MAX_DIP_CONTACT_RATIO = 3.00


# =====================================================================
# Search size
# =====================================================================

RNG_SEED = 29

GLOBAL_SAMPLES = 50000

LOCAL_SEEDS = 30

LOCAL_SAMPLES_PER_SEED = 600

TOP_TO_SAVE = 40


# =====================================================================
# XML helpers
# =====================================================================

def find_body(root, name):

    for body in root.iter("body"):

        if body.attrib.get("name") == name:
            return body

    raise RuntimeError(
        f"Body not found: {name}"
    )


def ensure_site(
    body,
    name,
    pos,
    rgba,
):

    target = None

    for site in body.findall("site"):

        if site.attrib.get("name") == name:
            target = site
            break


    if target is None:

        target = ET.SubElement(
            body,
            "site",
        )


    target.set(
        "name",
        name,
    )

    target.set(
        "pos",
        " ".join(
            f"{float(x):.10g}"
            for x in pos
        ),
    )

    target.set(
        "size",
        "0.0018",
    )

    target.set(
        "rgba",
        rgba,
    )


def prepare_model():

    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()


    palm = find_body(
        root,
        "palm_assembly:1",
    )

    prox = find_body(
        root,
        "Index_Proximal",
    )

    middle = find_body(
        root,
        "Index_Middle",
    )

    distal = find_body(
        root,
        "Index_Distal",
    )


    ensure_site(
        palm,
        "cfg2_idx_origin",
        [
            ORIGIN_X,
            REF["origin_y"],
            REF["origin_z"],
        ],
        "0 1 1 1",
    )

    ensure_site(
        prox,
        "cfg2_idx_guide_p",
        [
            GUIDE_X,
            REF["prox_y"],
            REF["prox_z"],
        ],
        "1 1 0 1",
    )

    ensure_site(
        middle,
        "cfg2_idx_guide_m",
        [
            GUIDE_X,
            REF["mid_y"],
            REF["mid_z"],
        ],
        "1 0.5 0 1",
    )

    ensure_site(
        distal,
        "cfg2_idx_guide_d",
        [
            GUIDE_X,
            REF["dist_y"],
            REF["dist_z"],
        ],
        "1 0 1 1",
    )

    ensure_site(
        distal,
        "cfg2_idx_anchor",
        [
            ANCHOR_X,
            REF["anchor_y"],
            REF["anchor_z"],
        ],
        "0 1 0 1",
    )


    tendon_parent = root.find(
        "tendon"
    )

    if tendon_parent is None:

        raise RuntimeError(
            "<tendon> section missing."
        )


    old = None
    old_index = None


    for i, element in enumerate(
        list(tendon_parent)
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
            f"Could not find {TENDON}"
        )


    tendon_parent.remove(
        old
    )


    spatial = ET.Element(
        "spatial",
        {
            "name": TENDON,
            "width": "0.0015",
            "rgba": "1 0 1 1",
        },
    )


    for name in [
        "cfg2_idx_origin",
        "cfg2_idx_guide_p",
        "cfg2_idx_guide_m",
        "cfg2_idx_guide_d",
        "cfg2_idx_anchor",
    ]:

        ET.SubElement(
            spatial,
            "site",
            {
                "site": name
            },
        )


    tendon_parent.insert(
        old_index,
        spatial,
    )


    tree.write(
        OUTPUT,
        encoding="unicode",
        xml_declaration=True,
    )


# =====================================================================
# MuJoCo helpers
# =====================================================================

def name_id(
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
            f"MuJoCo object missing: {name}"
        )

    return idx


def joint_address(
    model,
    name,
):

    jid = name_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    return int(
        model.jnt_qposadr[jid]
    )


def set_configuration(
    model,
    data,
    qpos0,
    addresses,
    q_deg,
):

    data.qpos[:] = qpos0


    for adr, q in zip(
        addresses,
        q_deg,
    ):

        data.qpos[adr] = math.radians(
            float(q)
        )


    mujoco.mj_forward(
        model,
        data,
    )


def moment_arms_mm(
    model,
    data,
    qpos0,
    addresses,
    tendon_id,
    q_deg,
):

    set_configuration(
        model,
        data,
        qpos0,
        addresses,
        q_deg,
    )


    qbase = (
        data.qpos.copy()
    )


    arms = []


    eps = 1.0e-5


    for adr in addresses:

        q0 = float(
            qbase[adr]
        )


        if abs(q0) < 2.0 * eps:

            data.qpos[:] = qbase

            mujoco.mj_forward(
                model,
                data,
            )

            L0 = float(
                data.ten_length[
                    tendon_id
                ]
            )


            data.qpos[adr] = (
                q0
                + eps
            )

            mujoco.mj_forward(
                model,
                data,
            )

            L1 = float(
                data.ten_length[
                    tendon_id
                ]
            )


            derivative = (
                L1 - L0
            ) / eps


        else:

            data.qpos[:] = qbase

            data.qpos[adr] = (
                q0
                + eps
            )

            mujoco.mj_forward(
                model,
                data,
            )

            Lplus = float(
                data.ten_length[
                    tendon_id
                ]
            )


            data.qpos[:] = qbase

            data.qpos[adr] = (
                q0
                - eps
            )

            mujoco.mj_forward(
                model,
                data,
            )

            Lminus = float(
                data.ten_length[
                    tendon_id
                ]
            )


            derivative = (
                Lplus
                - Lminus
            ) / (
                2.0
                * eps
            )


        arms.append(
            derivative
            * 1000.0
        )


    data.qpos[:] = qbase

    mujoco.mj_forward(
        model,
        data,
    )


    return np.asarray(
        arms,
        dtype=float,
    )


# =====================================================================
# Parameter application
# =====================================================================

def apply_parameters(
    model,
    site_ids,
    p,
):

    model.site_pos[
        site_ids["origin"],
        :
    ] = [
        ORIGIN_X,
        p["origin_y"],
        p["origin_z"],
    ]


    model.site_pos[
        site_ids["prox"],
        :
    ] = [
        GUIDE_X,
        p["prox_y"],
        p["prox_z"],
    ]


    model.site_pos[
        site_ids["mid"],
        :
    ] = [
        GUIDE_X,
        p["mid_y"],
        p["mid_z"],
    ]


    model.site_pos[
        site_ids["dist"],
        :
    ] = [
        GUIDE_X,
        p["dist_y"],
        p["dist_z"],
    ]


    model.site_pos[
        site_ids["anchor"],
        :
    ] = [
        ANCHOR_X,
        p["anchor_y"],
        p["anchor_z"],
    ]


# =====================================================================
# Candidate evaluation
# =====================================================================

def evaluate(
    model,
    data,
    qpos0,
    addresses,
    tendon_id,
    site_ids,
    p,
):

    apply_parameters(
        model,
        site_ids,
        p,
    )


    # ---------------------------------------------------------
    # OPEN: cheap early rejection
    # ---------------------------------------------------------

    open_arm = moment_arms_mm(
        model,
        data,
        qpos0,
        addresses,
        tendon_id,
        Q_OPEN,
    )


    # Correct sign mandatory.
    if not np.all(
        open_arm < 0.0
    ):
        return None


    abs_open = np.abs(
        open_arm
    )


    if not np.all(
        abs_open >= OPEN_MIN
    ):
        return None


    if not np.all(
        abs_open <= OPEN_MAX
    ):
        return None


    # ---------------------------------------------------------
    # Hardware contact configuration
    # ---------------------------------------------------------

    contact_arm = moment_arms_mm(
        model,
        data,
        qpos0,
        addresses,
        tendon_id,
        Q_HW_CONTACT,
    )


    if not np.all(
        contact_arm < 0.0
    ):
        return None


    if np.max(
        np.abs(
            contact_arm
        )
    ) > MAX_ANY_ARM_MM:

        return None


    # ---------------------------------------------------------
    # Configuration-dependent ratios
    # ---------------------------------------------------------

    pip_growth = (
        abs(
            contact_arm[1]
        )
        /
        abs(
            open_arm[1]
        )
    )


    if (
        pip_growth
        < MIN_PIP_GROWTH
    ):
        return None


    mcp_ratio = (
        abs(
            contact_arm[0]
        )
        /
        abs(
            open_arm[0]
        )
    )


    if not (
        MIN_MCP_CONTACT_RATIO
        <= mcp_ratio
        <= MAX_MCP_CONTACT_RATIO
    ):
        return None


    dip_ratio = (
        abs(
            contact_arm[2]
        )
        /
        max(
            abs(
                open_arm[2]
            ),
            0.25,
        )
    )


    if (
        dip_ratio
        > MAX_DIP_CONTACT_RATIO
    ):
        return None


    # ---------------------------------------------------------
    # Intermediate/flexed sign validation
    # ---------------------------------------------------------

    mid_arm = moment_arms_mm(
        model,
        data,
        qpos0,
        addresses,
        tendon_id,
        Q_MID,
    )


    flex_arm = moment_arms_mm(
        model,
        data,
        qpos0,
        addresses,
        tendon_id,
        Q_FLEXED,
    )


    if not np.all(
        mid_arm < 0.0
    ):
        return None


    if not np.all(
        flex_arm < 0.0
    ):
        return None


    if (
        np.max(
            np.abs(
                mid_arm
            )
        )
        > MAX_ANY_ARM_MM
    ):
        return None


    if (
        np.max(
            np.abs(
                flex_arm
            )
        )
        > MAX_ANY_ARM_MM
    ):
        return None


    # ---------------------------------------------------------
    # Score
    #
    # IMPORTANT:
    #
    # We do not prescribe a desired contact moment arm.
    #
    # We prefer:
    #   - good open-state agreement
    #   - strong but smooth PIP growth
    #   - minimal MCP distortion
    #   - moderate DIP change
    # ---------------------------------------------------------

    open_error = np.sum(
        (
            open_arm
            - TARGET_OPEN
        ) ** 2
    )


    mcp_change = (
        contact_arm[0]
        - open_arm[0]
    ) ** 2


    dip_change = (
        contact_arm[2]
        - open_arm[2]
    ) ** 2


    # Prefer useful PIP growth roughly in 1.3–2.2 region,
    # without forcing one exact value.
    if pip_growth <= 2.2:

        growth_cost = (
            pip_growth
            - 1.55
        ) ** 2

    else:

        growth_cost = (
            pip_growth
            - 2.2
        ) ** 2 * 4.0


    # Penalize abrupt mid/flex changes.
    smoothness = (
        np.sum(
            (
                mid_arm
                - open_arm
            ) ** 2
        )
        +
        np.sum(
            (
                flex_arm
                - contact_arm
            ) ** 2
        )
    )


    score = (
        3.0
        * open_error
        +
        0.35
        * mcp_change
        +
        0.25
        * dip_change
        +
        5.0
        * growth_cost
        +
        0.03
        * smoothness
    )


    return {

        "score":
            float(score),

        "params":
            dict(p),

        "open":
            open_arm.copy(),

        "mid":
            mid_arm.copy(),

        "contact":
            contact_arm.copy(),

        "flex":
            flex_arm.copy(),

        "pip_growth":
            float(pip_growth),

        "mcp_ratio":
            float(mcp_ratio),

        "dip_ratio":
            float(dip_ratio),
    }


# =====================================================================
# Sampling
# =====================================================================

def uniform_candidate(
    rng,
):

    p = {}


    for name in PARAM_NAMES:

        low, high = (
            BOUNDS[name]
        )

        p[name] = rng.uniform(
            low,
            high,
        )


    return p


def local_candidate(
    rng,
    center,
    scale=0.12,
):

    p = {}


    for name in PARAM_NAMES:

        low, high = (
            BOUNDS[name]
        )


        width = (
            high - low
        )


        value = rng.gauss(
            center[name],
            scale
            * width,
        )


        value = min(
            high,
            max(
                low,
                value,
            ),
        )


        p[name] = value


    return p


# =====================================================================
# Save selected geometry
# =====================================================================

def write_selected_geometry(
    params,
):

    tree = ET.parse(
        OUTPUT
    )

    root = tree.getroot()


    values = {

        "cfg2_idx_origin":
            [
                ORIGIN_X,
                params["origin_y"],
                params["origin_z"],
            ],

        "cfg2_idx_guide_p":
            [
                GUIDE_X,
                params["prox_y"],
                params["prox_z"],
            ],

        "cfg2_idx_guide_m":
            [
                GUIDE_X,
                params["mid_y"],
                params["mid_z"],
            ],

        "cfg2_idx_guide_d":
            [
                GUIDE_X,
                params["dist_y"],
                params["dist_z"],
            ],

        "cfg2_idx_anchor":
            [
                ANCHOR_X,
                params["anchor_y"],
                params["anchor_z"],
            ],
    }


    for site in root.iter("site"):

        name = site.attrib.get(
            "name"
        )


        if name in values:

            site.set(
                "pos",
                " ".join(
                    f"{float(x):.10g}"
                    for x in values[
                        name
                    ]
                ),
            )


    tree.write(
        OUTPUT,
        encoding="unicode",
        xml_declaration=True,
    )


# =====================================================================
# Main
# =====================================================================

def main():

    print()
    print("=" * 120)
    print(
        "RH8D INDEX — CONFIGURATION-DEPENDENT "
        "SPATIAL TENDON SEARCH V2"
    )
    print("=" * 120)


    prepare_model()


    model = mujoco.MjModel.from_xml_path(
        str(
            OUTPUT
        )
    )

    data = mujoco.MjData(
        model
    )


    qpos0 = (
        model.qpos0.copy()
    )


    addresses = [

        joint_address(
            model,
            MCP,
        ),

        joint_address(
            model,
            PIP,
        ),

        joint_address(
            model,
            DIP,
        ),
    ]


    tendon_id = name_id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON,
    )


    site_ids = {

        "origin":
            name_id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                "cfg2_idx_origin",
            ),

        "prox":
            name_id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                "cfg2_idx_guide_p",
            ),

        "mid":
            name_id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                "cfg2_idx_guide_m",
            ),

        "dist":
            name_id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                "cfg2_idx_guide_d",
            ),

        "anchor":
            name_id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                "cfg2_idx_anchor",
            ),
    }


    rng = random.Random(
        RNG_SEED
    )


    accepted = []


    print()
    print(
        f"Global candidates = "
        f"{GLOBAL_SAMPLES}"
    )

    print(
        "Hard structural filtering enabled."
    )


    for i in range(
        GLOBAL_SAMPLES
    ):

        p = uniform_candidate(
            rng
        )


        result = evaluate(
            model,
            data,
            qpos0,
            addresses,
            tendon_id,
            site_ids,
            p,
        )


        if result is not None:

            accepted.append(
                result
            )


        if (
            (i + 1) % 10000
            == 0
        ):

            print(
                f"  {i+1:6d}/"
                f"{GLOBAL_SAMPLES}  "
                f"accepted="
                f"{len(accepted)}"
            )


    accepted.sort(
        key=lambda r:
            r["score"]
    )


    print()
    print(
        f"Global structurally valid = "
        f"{len(accepted)}"
    )


    # =================================================================
    # Local refinement around best valid global candidates
    # =================================================================

    if accepted:

        seeds = accepted[
            :min(
                LOCAL_SEEDS,
                len(
                    accepted
                ),
            )
        ]


        print()
        print(
            f"Local refinement seeds = "
            f"{len(seeds)}"
        )


        for seed_index, seed in enumerate(
            seeds,
            start=1,
        ):

            for _ in range(
                LOCAL_SAMPLES_PER_SEED
            ):

                p = local_candidate(
                    rng,
                    seed["params"],
                )


                result = evaluate(
                    model,
                    data,
                    qpos0,
                    addresses,
                    tendon_id,
                    site_ids,
                    p,
                )


                if result is not None:

                    accepted.append(
                        result
                    )


            if (
                seed_index % 5
                == 0
            ):

                print(
                    f"  refined "
                    f"{seed_index}/"
                    f"{len(seeds)} seeds"
                )


    accepted.sort(
        key=lambda r:
            r["score"]
    )


    # =================================================================
    # No valid geometry
    # =================================================================

    if not accepted:

        print()
        print("=" * 120)
        print(
            "RESULT: NO STRUCTURALLY VALID "
            "SPATIAL ROUTING FOUND"
        )
        print("=" * 120)

        print()
        print(
            "Do NOT fit dynamics."
        )

        print(
            "Next step would be to allow "
            "additional routing topology / pulley-wrap "
            "elements rather than widening parameters blindly."
        )

        return


    selected = accepted[0]


    # =================================================================
    # Save top candidates
    # =================================================================

    top = accepted[
        :TOP_TO_SAVE
    ]


    fields = [

        "rank",
        "score",

        *PARAM_NAMES,

        "open_mcp",
        "open_pip",
        "open_dip",

        "mid_mcp",
        "mid_pip",
        "mid_dip",

        "contact_mcp",
        "contact_pip",
        "contact_dip",

        "flex_mcp",
        "flex_pip",
        "flex_dip",

        "pip_growth",
        "mcp_ratio",
        "dip_ratio",
    ]


    with open(
        CSV_OUT,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()


        for rank, r in enumerate(
            top,
            start=1,
        ):

            row = {

                "rank":
                    rank,

                "score":
                    r["score"],

                **r["params"],

                "open_mcp":
                    r["open"][0],

                "open_pip":
                    r["open"][1],

                "open_dip":
                    r["open"][2],

                "mid_mcp":
                    r["mid"][0],

                "mid_pip":
                    r["mid"][1],

                "mid_dip":
                    r["mid"][2],

                "contact_mcp":
                    r["contact"][0],

                "contact_pip":
                    r["contact"][1],

                "contact_dip":
                    r["contact"][2],

                "flex_mcp":
                    r["flex"][0],

                "flex_pip":
                    r["flex"][1],

                "flex_dip":
                    r["flex"][2],

                "pip_growth":
                    r["pip_growth"],

                "mcp_ratio":
                    r["mcp_ratio"],

                "dip_ratio":
                    r["dip_ratio"],
            }


            writer.writerow(
                row
            )


    write_selected_geometry(
        selected["params"]
    )


    # =================================================================
    # Print selected routing
    # =================================================================

    print()
    print("=" * 120)
    print(
        "BEST STRUCTURALLY VALID ROUTING V2"
    )
    print("=" * 120)


    print()
    print(
        "Site parameters:"
    )


    for name in PARAM_NAMES:

        print(
            f"  {name:<10} = "
            f"{selected['params'][name]*1000.0:+8.3f} mm"
        )


    print()
    print(
        f"{'STATE':<14}"
        f"{'MCP':>12}"
        f"{'PIP':>12}"
        f"{'DIP':>12}"
    )

    print(
        "-" * 50
    )


    for label, key in [

        ("OPEN", "open"),
        ("MID", "mid"),
        ("HW_CONTACT", "contact"),
        ("FLEXED", "flex"),
    ]:

        values = selected[
            key
        ]


        print(
            f"{label:<14}"
            f"{values[0]:>12.3f}"
            f"{values[1]:>12.3f}"
            f"{values[2]:>12.3f}"
        )


    print()
    print(
        "Configuration ratios:"
    )

    print(
        f"  PIP contact/open = "
        f"{selected['pip_growth']:.3f} x"
    )

    print(
        f"  MCP contact/open = "
        f"{selected['mcp_ratio']:.3f} x"
    )

    print(
        f"  DIP contact/open = "
        f"{selected['dip_ratio']:.3f} x"
    )


    print()
    print(
        f"Score = "
        f"{selected['score']:.4f}"
    )


    print()
    print(
        f"Total structurally valid candidates = "
        f"{len(accepted)}"
    )


    print()
    print(
        "Selected geometry:"
    )

    print(
        OUTPUT
    )


    print()
    print(
        "Top candidates:"
    )

    print(
        CSV_OUT
    )


    print()
    print("=" * 120)
    print(
        "RESULT: PASS — spatial tendon geometry "
        "has correct flexion sign and genuine "
        "configuration-dependent PIP leverage."
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
