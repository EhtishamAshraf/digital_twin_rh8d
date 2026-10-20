#!/usr/bin/env python3

from pathlib import Path
import csv
import math
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import fit_spatial_free_and_test_contact_v35p7 as p7
import test_v43_pd_compliant_coupling as v43


# ============================================================
# PATHS
# ============================================================

WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = WS / "src/ur5e_mujoco_description/mjcf"
MESHES = WS / "src/ur5e_mujoco_description/meshes"

SOURCE = MJCF / "ur5e_rh8d_index_spatial_v38_final.xml"

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v44_spatial_staged_coupling"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_CSV = OUTDIR / "v44_summary.csv"

SELECTED_XML = OUTDIR / "v44_selected.xml"
SELECTED_TRACE = OUTDIR / "v44_selected_trace.npz"


# ============================================================
# JOINT / GEOM NAMES
# ============================================================

MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

FIXTURE = "index_contact_fixture"

MID_GEOMS = [
    "collision_idx_mid_mesh",
    "collision_idx_mid",
]

TIP_GEOMS = [
    "collision_idx_tip",
]

COUPLING = "idx_pd_spatial_staged_return"

SITE_A = "v44_ret_pip_parent"
SITE_B = "v44_ret_pip_child"
SITE_C = "v44_ret_dip_parent"
SITE_D = "v44_ret_dip_child"


# ============================================================
# HARDWARE TARGET
# ============================================================

HW = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


# ============================================================
# ACTUAL CAD REFERENCE SITES FROM V38
#
# We preserve their x/z longitudinal locations and only choose
# the cross-sectional side (local y) of the elastic return.
#
# These locations are around PIP and DIP:
#
# p04 = proximal side of PIP
# p05 = middle side of PIP
# p06 = middle side of DIP
# p07 = distal side of DIP
# ============================================================

CAD_SITE_NAMES = [
    "v35_p04_pip_parent",
    "v35_p05_pip_child",
    "v35_p06_dip_parent",
    "v35_p07_dip_child",
]


# ============================================================
# SMALL STRUCTURAL ROUTE SET
#
# This is NOT an optimizer.
#
# We test only a few physically realizable S-routes.
# The script will automatically reject routes that do not:
#
#   tension -> +PIP torque
#   tension -> -DIP torque
#   approximately zero MCP torque
#
# It also requires route length to increase during late
# PIP/DIP recruitment so a slack cable can become taut.
# ============================================================

ROUTES = [
    (
        "S1",
        (+0.009, -0.009, +0.003, +0.003),
    ),

    (
        "S2",
        (-0.006, +0.009, +0.009, +0.003),
    ),

    (
        "S3",
        (+0.006, -0.006, +0.009, +0.003),
    ),

    (
        "S4",
        (-0.006, +0.006, +0.003, +0.006),
    ),
]


# ============================================================
# ENGAGEMENT TEST
#
# Hardware/video hypothesis:
#
# early:
#   PIP moving, DIP still relatively quiet
#
# later:
#   distal coupling becomes loaded
#
# Each pair means:
#
#   (PIP angle, DIP angle)
#
# where the elastic element just becomes taut.
# ============================================================

ENGAGE_CONFIGS = [
    (16.0, 3.0),
    (18.0, 4.0),
    (20.0, 5.0),
]


# Only 3 stiffnesses.
K_VALUES = [
    4000.0,
    6000.0,
    8000.0,
]


# ============================================================
# XML HELPERS
# ============================================================

def vec(text):

    return np.asarray(
        [
            float(x)
            for x in text.split()
        ],
        dtype=float,
    )


def source_root():

    return ET.parse(
        SOURCE
    ).getroot()


def find_body(
    root,
    name,
):

    b = root.find(
        f".//body[@name='{name}']"
    )

    if b is None:

        raise RuntimeError(
            f"Missing body {name}"
        )

    return b


def source_site_positions():

    root = source_root()

    result = {}

    for name in CAD_SITE_NAMES:

        s = root.find(
            f".//site[@name='{name}']"
        )

        if s is None:

            raise RuntimeError(
                f"Missing source CAD site {name}"
            )

        result[
            name
        ] = vec(
            s.get(
                "pos"
            )
        )

    return result


BASE_POS = source_site_positions()


# ============================================================
# BUILD SPATIAL RETURN
# ============================================================

def build_candidate(
    route,
    stiffness,
    slack_length,
    output,
):

    tree = ET.parse(
        SOURCE
    )

    root = tree.getroot()


    # --------------------------------------------------------
    # Make mesh references independent of where result XML is
    # stored.
    # --------------------------------------------------------

    compiler = root.find(
        "compiler"
    )

    if compiler is None:

        raise RuntimeError(
            "No compiler element."
        )

    compiler.set(
        "meshdir",
        str(
            MESHES
        ),
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


    yA, yB, yC, yD = route


    # Preserve x/z from the already identified CAD route.
    pA = BASE_POS[
        "v35_p04_pip_parent"
    ].copy()

    pB = BASE_POS[
        "v35_p05_pip_child"
    ].copy()

    pC = BASE_POS[
        "v35_p06_dip_parent"
    ].copy()

    pD = BASE_POS[
        "v35_p07_dip_child"
    ].copy()


    pA[1] = yA
    pB[1] = yB
    pC[1] = yC
    pD[1] = yD


    def add_site(
        body,
        name,
        pos,
        rgba,
    ):

        ET.SubElement(
            body,
            "site",
            {
                "name":
                    name,

                "pos":
                    " ".join(
                        f"{x:.12g}"
                        for x in pos
                    ),

                "size":
                    "0.0012",

                "rgba":
                    rgba,

                "group":
                    "4",
            },
        )


    add_site(
        prox,
        SITE_A,
        pA,
        "0 1 1 1",
    )

    add_site(
        middle,
        SITE_B,
        pB,
        "0 1 1 1",
    )

    add_site(
        middle,
        SITE_C,
        pC,
        "0 1 1 1",
    )

    add_site(
        distal,
        SITE_D,
        pD,
        "0 1 1 1",
    )


    tendon_root = root.find(
        "tendon"
    )

    if tendon_root is None:

        raise RuntimeError(
            "No tendon block."
        )


    spatial = ET.SubElement(
        tendon_root,
        "spatial",
        {
            "name":
                COUPLING,

            "width":
                "0.0012",

            "rgba":
                "0 1 1 1",

            "stiffness":
                f"{float(stiffness):.12g}",

            # No always-on viscous coupling.
            "damping":
                "0",

            "frictionloss":
                "0",

            # CRITICAL:
            #
            # 0 <= L <= slack_length:
            #     zero elastic force
            #
            # L > slack_length:
            #     elastic tension
            #
            "springlength":
                f"0 {float(slack_length):.12g}",
        },
    )


    for name in [
        SITE_A,
        SITE_B,
        SITE_C,
        SITE_D,
    ]:

        ET.SubElement(
            spatial,
            "site",
            {
                "site":
                    name
            },
        )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


    # Fail immediately if malformed.
    mujoco.MjModel.from_xml_path(
        str(
            output
        )
    )


# ============================================================
# MUJOCO KINEMATIC HELPERS
# ============================================================

def qpos_addr(
    model,
    joint,
):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint,
    )

    if jid < 0:

        raise RuntimeError(
            f"Missing joint {joint}"
        )

    return int(
        model.jnt_qposadr[
            jid
        ]
    )


def tendon_id(
    model,
):

    tid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        COUPLING,
    )

    if tid < 0:

        raise RuntimeError(
            "V44 tendon missing."
        )

    return tid


def set_pd(
    model,
    data,
    pip_deg,
    dip_deg,
):

    data.qpos[:] = (
        model.qpos0
    )

    data.qvel[:] = 0.0


    data.qpos[
        qpos_addr(
            model,
            PIP,
        )
    ] = math.radians(
        pip_deg
    )

    data.qpos[
        qpos_addr(
            model,
            DIP,
        )
    ] = math.radians(
        dip_deg
    )


    mujoco.mj_forward(
        model,
        data,
    )


def tendon_length(
    model,
    data,
    tid,
    pip_deg,
    dip_deg,
):

    set_pd(
        model,
        data,
        pip_deg,
        dip_deg,
    )

    return float(
        data.ten_length[
            tid
        ]
    )


def length_gradient(
    model,
    data,
    tid,
    pip_deg,
    dip_deg,
):

    h = 0.05


    lp1 = tendon_length(
        model,
        data,
        tid,
        pip_deg + h,
        dip_deg,
    )

    lp0 = tendon_length(
        model,
        data,
        tid,
        pip_deg - h,
        dip_deg,
    )

    dL_dPIP = (
        lp1
        -
        lp0
    ) / math.radians(
        2 * h
    )


    ld1 = tendon_length(
        model,
        data,
        tid,
        pip_deg,
        dip_deg + h,
    )

    ld0 = tendon_length(
        model,
        data,
        tid,
        pip_deg,
        dip_deg - h,
    )

    dL_dDIP = (
        ld1
        -
        ld0
    ) / math.radians(
        2 * h
    )


    return (
        dL_dPIP,
        dL_dDIP,
    )


# ============================================================
# ROUTE SELECTION
# ============================================================

def choose_route():

    print()
    print("=" * 170)
    print(
        "V44 SPATIAL ROUTE SIGN CHECK"
    )
    print("=" * 170)

    valid = []


    for name, route in ROUTES:

        probe = (
            OUTDIR
            /
            f"probe_{name}.xml"
        )


        build_candidate(
            route,
            0.0,
            1.0,
            probe,
        )


        model = mujoco.MjModel.from_xml_path(
            str(
                probe
            )
        )

        data = mujoco.MjData(
            model
        )

        tid = tendon_id(
            model
        )


        dP, dD = length_gradient(
            model,
            data,
            tid,
            20.0,
            8.0,
        )


        # Under positive tendon tension:
        #
        # tau ~= -T * dL/dq
        #
        # Therefore desired:
        #
        # dL/dPIP < 0  -> +PIP torque
        # dL/dDIP > 0  -> -DIP torque


        L_open = tendon_length(
            model,
            data,
            tid,
            0.0,
            0.0,
        )

        L_early = tendon_length(
            model,
            data,
            tid,
            12.0,
            2.0,
        )

        L_engage = tendon_length(
            model,
            data,
            tid,
            18.0,
            4.0,
        )

        L_target = tendon_length(
            model,
            data,
            tid,
            23.811,
            9.327,
        )


        growth = (
            L_target
            -
            L_engage
        )


        balance = (
            min(
                abs(
                    dP
                ),
                abs(
                    dD
                ),
            )
            /
            max(
                abs(
                    dP
                ),
                abs(
                    dD
                ),
                1e-12,
            )
        )


        sign_ok = (
            dP < 0.0
            and
            dD > 0.0
        )


        stage_ok = (
            L_open
            <=
            L_engage
            and
            L_target
            >
            L_engage
            +
            2e-5
        )


        print()
        print(
            f"{name}: y = "
            f"{route}"
        )

        print(
            "  dL/dPIP, dL/dDIP = "
            f"{dP*1000:+.4f} / "
            f"{dD*1000:+.4f} mm/rad"
        )

        print(
            "  L open/early/engage/target = "
            f"{L_open*1000:.3f} / "
            f"{L_early*1000:.3f} / "
            f"{L_engage*1000:.3f} / "
            f"{L_target*1000:.3f} mm"
        )

        print(
            f"  post-engagement stretch = "
            f"{growth*1000:.4f} mm"
        )

        print(
            f"  moment-arm balance = "
            f"{balance:.3f}"
        )

        print(
            "  sign/stage = "
            f"{sign_ok}/{stage_ok}"
        )


        if (
            sign_ok
            and
            stage_ok
        ):

            # Prefer useful late stretch while avoiding
            # extremely unbalanced PIP/DIP leverage.
            score = (
                balance
                +
                min(
                    growth
                    *
                    1000.0,
                    0.25,
                )
            )

            valid.append(
                (
                    score,
                    name,
                    route,
                    probe,
                )
            )


    if not valid:

        raise RuntimeError(
            "No physically valid V44 S-route. "
            "STOP: do not run parameter cases."
        )


    valid.sort(
        reverse=True,
        key=lambda x:
            x[0],
    )


    score, name, route, probe = (
        valid[0]
    )


    print()
    print("=" * 170)

    print(
        f"SELECTED STRUCTURAL ROUTE: "
        f"{name}"
    )

    print(
        f"y coordinates = "
        f"{route}"
    )

    print(
        f"route score = "
        f"{score:.4f}"
    )

    print("=" * 170)


    return (
        name,
        route,
        probe,
    )


# ============================================================
# CONTACT TRACE RECORDER
# ============================================================

def geom_id(
    model,
    name,
):

    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        name,
    )


def has_contact(
    data,
    fixture,
    targets,
):

    targets = {
        x
        for x in targets
        if x >= 0
    }


    for i in range(
        data.ncon
    ):

        c = data.contact[
            i
        ]

        a = int(
            c.geom1
        )

        b = int(
            c.geom2
        )


        if (
            a == fixture
            and
            b in targets
        ):

            return True


        if (
            b == fixture
            and
            a in targets
        ):

            return True


    return False


def run_contact_trace(
    xml,
    label,
):

    trace_t = []
    trace_angles = []
    trace_mid = []
    trace_tip = []


    REAL_STEP = (
        mujoco.mj_step
    )


    cache = {
        "model_id":
            None
    }


    def recorder(
        model,
        data,
        *args,
        **kwargs,
    ):

        out = REAL_STEP(
            model,
            data,
            *args,
            **kwargs,
        )


        if (
            cache[
                "model_id"
            ]
            !=
            id(
                model
            )
        ):

            cache[
                "model_id"
            ] = id(
                model
            )


            cache[
                "mcp"
            ] = qpos_addr(
                model,
                MCP,
            )

            cache[
                "pip"
            ] = qpos_addr(
                model,
                PIP,
            )

            cache[
                "dip"
            ] = qpos_addr(
                model,
                DIP,
            )


            cache[
                "fixture"
            ] = geom_id(
                model,
                FIXTURE,
            )


            cache[
                "mid"
            ] = [
                geom_id(
                    model,
                    n,
                )
                for n in MID_GEOMS
            ]


            cache[
                "tip"
            ] = [
                geom_id(
                    model,
                    n,
                )
                for n in TIP_GEOMS
            ]


        # If replay setup resets simulation time,
        # keep only final continuous run.
        if (
            trace_t
            and
            float(
                data.time
            )
            <
            trace_t[-1]
            -
            1e-9
        ):

            trace_t.clear()
            trace_angles.clear()
            trace_mid.clear()
            trace_tip.clear()


        angles = np.degrees(
            [
                data.qpos[
                    cache[
                        "mcp"
                    ]
                ],

                data.qpos[
                    cache[
                        "pip"
                    ]
                ],

                data.qpos[
                    cache[
                        "dip"
                    ]
                ],
            ]
        )


        trace_t.append(
            float(
                data.time
            )
        )

        trace_angles.append(
            angles
        )


        trace_mid.append(
            has_contact(
                data,
                cache[
                    "fixture"
                ],
                cache[
                    "mid"
                ],
            )
        )


        trace_tip.append(
            has_contact(
                data,
                cache[
                    "fixture"
                ],
                cache[
                    "tip"
                ],
            )
        )


        return out


    # Same successful wrapping method used by the V43 viewer.
    mujoco.mj_step = (
        recorder
    )


    try:

        result = v43.run_contact(
            xml,
            label,
        )


    finally:

        mujoco.mj_step = (
            REAL_STEP
        )


    if len(
        trace_t
    ) < 100:

        raise RuntimeError(
            "Contact trace capture failed."
        )


    t = np.asarray(
        trace_t,
        dtype=float,
    )

    ang = np.asarray(
        trace_angles,
        dtype=float,
    )

    mid = np.asarray(
        trace_mid,
        dtype=bool,
    )

    tip = np.asarray(
        trace_tip,
        dtype=bool,
    )


    return (
        result,
        t,
        ang,
        mid,
        tip,
    )


# ============================================================
# STABILITY ANALYSIS
# ============================================================

def analyze_trace(
    result,
    t,
    ang,
    mid,
    tip,
):

    reported = np.array(
        [
            result[
                "mcp_deg"
            ],

            result[
                "pip_deg"
            ],

            result[
                "dip_deg"
            ],
        ],
        dtype=float,
    )


    # Locate the actual trajectory sample corresponding
    # most closely to the evaluator's reported pose.
    distance = np.linalg.norm(
        ang
        -
        reported[
            None,
            :
        ],
        axis=1,
    )


    i_target = int(
        np.argmin(
            distance
        )
    )


    target_time = float(
        t[
            i_target
        ]
    )


    # Analyze 0.75 s after the reported target pose.
    end_time = (
        target_time
        +
        0.75
    )


    i_end = int(
        np.searchsorted(
            t,
            end_time,
            side="right",
        )
    )


    i_end = max(
        i_end,
        i_target
        +
        1,
    )

    i_end = min(
        i_end,
        len(
            t
        ),
    )


    wm = mid[
        i_target:
        i_end
    ]

    wt = tip[
        i_target:
        i_end
    ]

    wa = ang[
        i_target:
        i_end
    ]


    both = (
        wm
        &
        wt
    )


    any_contact = (
        wm
        |
        wt
    )


    both_fraction = float(
        np.mean(
            both
        )
    )

    any_fraction = float(
        np.mean(
            any_contact
        )
    )


    def off_events(
        x,
    ):

        if len(
            x
        ) < 2:

            return 0

        return int(
            np.sum(
                x[
                    :-1
                ]
                &
                (
                    ~x[
                        1:
                    ]
                )
            )
        )


    detach_events = (
        off_events(
            wm
        )
        +
        off_events(
            wt
        )
    )


    # Pose at the end of the 0.75 s hold window.
    hold_end_pose = (
        wa[
            -1
        ]
    )


    hold_drift_rmse = float(
        np.sqrt(
            np.mean(
                (
                    hold_end_pose
                    -
                    reported
                )
                **
                2
            )
        )
    )


    terminal_pose = (
        ang[
            -1
        ]
    )


    terminal_both = bool(
        mid[
            -1
        ]
        and
        tip[
            -1
        ]
    )


    return {
        "target_time":
            target_time,

        "target_match_error_deg":
            float(
                distance[
                    i_target
                ]
            ),

        "both_fraction_075":
            both_fraction,

        "any_fraction_075":
            any_fraction,

        "detach_events_075":
            detach_events,

        "hold_end_mcp":
            float(
                hold_end_pose[
                    0
                ]
            ),

        "hold_end_pip":
            float(
                hold_end_pose[
                    1
                ]
            ),

        "hold_end_dip":
            float(
                hold_end_pose[
                    2
                ]
            ),

        "hold_drift_rmse":
            hold_drift_rmse,

        "terminal_mcp":
            float(
                terminal_pose[
                    0
                ]
            ),

        "terminal_pip":
            float(
                terminal_pose[
                    1
                ]
            ),

        "terminal_dip":
            float(
                terminal_pose[
                    2
                ]
            ),

        "terminal_mid":
            int(
                mid[
                    -1
                ]
            ),

        "terminal_tip":
            int(
                tip[
                    -1
                ]
            ),

        "terminal_both":
            int(
                terminal_both
            ),
    }


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate(
    label,
    xml,
    route_name,
    K,
    engage_pip,
    engage_dip,
    slack,
):

    print()
    print("-" * 180)

    print(
        f"{label}"
    )


    fm, fp, fc, _ = p7.run_free(
        xml
    )


    (
        cr,
        t,
        ang,
        mid,
        tip,
    ) = run_contact_trace(
        xml,
        label,
    )


    stability = analyze_trace(
        cr,
        t,
        ang,
        mid,
        tip,
    )


    # --------------------------------------------------------
    # Selection score.
    #
    # A transient low RMSE is no longer enough.
    # --------------------------------------------------------

    score = (
        float(
            cr[
                "rmse_deg"
            ]
        )

        +
        0.35
        *
        float(
            fc
        )

        +
        4.0
        *
        (
            1.0
            -
            stability[
                "both_fraction_075"
            ]
        )

        +
        0.75
        *
        stability[
            "hold_drift_rmse"
        ]

        +
        0.40
        *
        stability[
            "detach_events_075"
        ]
    )


    # Require correct distal redistribution.
    ddip = float(
        cr[
            "middle_to_tip_dDIP_deg"
        ]
    )


    if (
        not math.isfinite(
            ddip
        )
        or
        ddip
        <=
        0.0
    ):

        score += 20.0


    row = {
        "label":
            label,

        "route":
            route_name,

        "K_Npm":
            K,

        "engage_pip_deg":
            engage_pip,

        "engage_dip_deg":
            engage_dip,

        "slack_length_m":
            slack,

        "free_mcp_rmse":
            fm,

        "free_pip_rmse":
            fp,

        "free_combined_rmse":
            fc,

        "reported_mcp":
            cr[
                "mcp_deg"
            ],

        "reported_pip":
            cr[
                "pip_deg"
            ],

        "reported_dip":
            cr[
                "dip_deg"
            ],

        "reported_contact_rmse":
            cr[
                "rmse_deg"
            ],

        "mid_events":
            cr[
                "mid_on"
            ],

        "tip_events":
            cr[
                "tip_on"
            ],

        "dPIP":
            cr[
                "middle_to_tip_dPIP_deg"
            ],

        "dDIP":
            cr[
                "middle_to_tip_dDIP_deg"
            ],

        **stability,

        "score":
            score,
    }


    print(
        "FREE RMSE = "
        f"{fc:.3f} deg"
    )

    print(
        "reported pose = "
        f"{cr['mcp_deg']:.3f} / "
        f"{cr['pip_deg']:.3f} / "
        f"{cr['dip_deg']:.3f}"
    )

    print(
        f"reported contact RMSE = "
        f"{cr['rmse_deg']:.3f} deg"
    )

    print(
        "events M/T = "
        f"{cr['mid_on']} / "
        f"{cr['tip_on']}"
    )

    print(
        "middle->tip dPIP/dDIP = "
        f"{cr['middle_to_tip_dPIP_deg']:+.3f} / "
        f"{cr['middle_to_tip_dDIP_deg']:+.3f}"
    )

    print(
        "0.75 s after reported pose:"
    )

    print(
        f"  BOTH-contact fraction = "
        f"{stability['both_fraction_075']:.3f}"
    )

    print(
        f"  ANY-contact fraction = "
        f"{stability['any_fraction_075']:.3f}"
    )

    print(
        f"  detach events = "
        f"{stability['detach_events_075']}"
    )

    print(
        f"  pose drift RMSE = "
        f"{stability['hold_drift_rmse']:.3f} deg"
    )

    print(
        "  hold-end pose = "
        f"{stability['hold_end_mcp']:.2f} / "
        f"{stability['hold_end_pip']:.2f} / "
        f"{stability['hold_end_dip']:.2f}"
    )

    print(
        "terminal contact M/T = "
        f"{stability['terminal_mid']} / "
        f"{stability['terminal_tip']}"
    )

    print(
        f"SCORE = "
        f"{score:.3f}"
    )


    return (
        row,
        t,
        ang,
        mid,
        tip,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 180)

    print(
        "V44 — SLACK SPATIAL PIP/DIP COUPLING "
        "+ CONTACT-PERSISTENCE EVALUATOR"
    )

    print("=" * 180)

    print()
    print(
        "Goal:"
    )

    print(
        "  preserve V43's +PIP / -DIP load sharing"
    )

    print(
        "  remove always-on coupling during early motion"
    )

    print(
        "  reject transient low-RMSE models that later detach"
    )


    # --------------------------------------------------------
    # Freeze V38 source in experiment folder.
    # --------------------------------------------------------

    shutil.copy2(
        SOURCE,
        OUTDIR
        /
        "v38_source.xml",
    )


    # --------------------------------------------------------
    # Structural route selection BEFORE dynamics.
    # --------------------------------------------------------

    (
        route_name,
        route,
        probe_xml,
    ) = choose_route()


    probe_model = (
        mujoco.MjModel.from_xml_path(
            str(
                probe_xml
            )
        )
    )

    probe_data = (
        mujoco.MjData(
            probe_model
        )
    )

    tid = tendon_id(
        probe_model
    )


    rows = []

    traces = {}


    # --------------------------------------------------------
    # First benchmark the V43 fixed-coupling candidate.
    # --------------------------------------------------------

    v43_xml = (
        MJCF
        /
        "tmp_v43_k750_r20.xml"
    )


    if not v43_xml.exists():

        v43.build_candidate(
            750.0,
            20.0,
            v43_xml,
        )


    (
        baseline_row,
        bt,
        ba,
        bm,
        bp,
    ) = evaluate(
        "V43_FIXED_BASELINE",
        v43_xml,
        "fixed_abstract",
        750.0,
        0.0,
        0.0,
        0.0,
    )


    rows.append(
        baseline_row
    )

    traces[
        "V43_FIXED_BASELINE"
    ] = (
        bt,
        ba,
        bm,
        bp,
    )


    # --------------------------------------------------------
    # V44 physical staged cases.
    # --------------------------------------------------------

    case_index = 0


    for (
        ep,
        ed,
    ) in ENGAGE_CONFIGS:

        slack = tendon_length(
            probe_model,
            probe_data,
            tid,
            ep,
            ed,
        )


        open_length = tendon_length(
            probe_model,
            probe_data,
            tid,
            0.0,
            0.0,
        )


        target_length = tendon_length(
            probe_model,
            probe_data,
            tid,
            HW[
                1
            ],
            HW[
                2
            ],
        )


        print()
        print("=" * 180)

        print(
            f"ENGAGEMENT CONFIG "
            f"PIP/DIP = {ep:.1f}/{ed:.1f} deg"
        )

        print(
            "length open/slack/hardware-target = "
            f"{open_length*1000:.4f} / "
            f"{slack*1000:.4f} / "
            f"{target_length*1000:.4f} mm"
        )

        print(
            "available stretch at hardware target = "
            f"{(target_length-slack)*1000:.4f} mm"
        )

        print("=" * 180)


        if (
            open_length
            >
            slack
        ):

            print(
                "SKIP: coupling would already be taut open."
            )

            continue


        if (
            target_length
            <=
            slack
            +
            2e-5
        ):

            print(
                "SKIP: insufficient late engagement."
            )

            continue


        for K in K_VALUES:

            case_index += 1


            label = (
                f"V44_E{int(ep)}_{int(ed)}"
                f"_K{int(K)}"
            )


            xml = (
                OUTDIR
                /
                f"{label}.xml"
            )


            build_candidate(
                route,
                K,
                slack,
                xml,
            )


            (
                row,
                t,
                ang,
                mid,
                tip,
            ) = evaluate(
                label,
                xml,
                route_name,
                K,
                ep,
                ed,
                slack,
            )


            rows.append(
                row
            )


            traces[
                label
            ] = (
                t,
                ang,
                mid,
                tip,
            )


    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    with SUMMARY_CSV.open(
        "w",
        newline="",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    print()
    print("=" * 215)
    print("V44 SUMMARY")
    print("=" * 215)

    print(
        "label                 free    contact   "
        "both075  detach  drift   reported pose                "
        "terminal M/T   score"
    )

    print("-" * 215)


    for r in rows:

        print(
            f"{r['label']:<21} "
            f"{r['free_combined_rmse']:6.3f}  "
            f"{r['reported_contact_rmse']:7.3f}   "
            f"{r['both_fraction_075']:7.3f}  "
            f"{r['detach_events_075']:6d}  "
            f"{r['hold_drift_rmse']:6.3f}   "
            f"{r['reported_mcp']:6.2f} / "
            f"{r['reported_pip']:6.2f} / "
            f"{r['reported_dip']:6.2f}       "
            f"{r['terminal_mid']}/{r['terminal_tip']}        "
            f"{r['score']:7.3f}"
        )


    # Exclude V43 from V44 selection.
    v44_rows = [
        r
        for r in rows
        if r[
            "label"
        ].startswith(
            "V44_"
        )
    ]


    if not v44_rows:

        raise RuntimeError(
            "No V44 dynamics cases were valid."
        )


    best = min(
        v44_rows,
        key=lambda r:
            r[
                "score"
            ],
    )


    baseline = rows[
        0
    ]


    best_xml = (
        OUTDIR
        /
        f"{best['label']}.xml"
    )


    shutil.copy2(
        best_xml,
        SELECTED_XML,
    )


    (
        st,
        sa,
        sm,
        sp,
    ) = traces[
        best[
            "label"
        ]
    ]


    np.savez_compressed(
        SELECTED_TRACE,

        time=st,

        angles_deg=sa,

        middle_contact=sm,

        tip_contact=sp,
    )


    print()
    print("=" * 215)
    print("SELECTED V44")
    print("=" * 215)


    print(
        f"case = "
        f"{best['label']}"
    )

    print(
        f"route = "
        f"{best['route']}"
    )

    print(
        f"K = "
        f"{best['K_Npm']:.1f} N/m"
    )

    print(
        "engagement PIP/DIP = "
        f"{best['engage_pip_deg']:.1f} / "
        f"{best['engage_dip_deg']:.1f} deg"
    )


    print()
    print(
        "V43 -> V44"
    )

    print(
        "free RMSE: "
        f"{baseline['free_combined_rmse']:.3f} -> "
        f"{best['free_combined_rmse']:.3f}"
    )

    print(
        "reported contact RMSE: "
        f"{baseline['reported_contact_rmse']:.3f} -> "
        f"{best['reported_contact_rmse']:.3f}"
    )

    print(
        "BOTH-contact persistence over 0.75 s: "
        f"{baseline['both_fraction_075']:.3f} -> "
        f"{best['both_fraction_075']:.3f}"
    )

    print(
        "detach events over 0.75 s: "
        f"{baseline['detach_events_075']} -> "
        f"{best['detach_events_075']}"
    )

    print(
        "post-target pose drift: "
        f"{baseline['hold_drift_rmse']:.3f} -> "
        f"{best['hold_drift_rmse']:.3f} deg"
    )


    improved_stability = (
        best[
            "both_fraction_075"
        ]
        >
        baseline[
            "both_fraction_075"
        ]
        +
        0.15
        and
        best[
            "detach_events_075"
        ]
        <=
        baseline[
            "detach_events_075"
        ]
    )


    useful_accuracy = (
        best[
            "reported_contact_rmse"
        ]
        <=
        1.5
        and
        best[
            "free_combined_rmse"
        ]
        <=
        7.0
    )


    print()
    print("=" * 215)


    if (
        improved_stability
        and
        useful_accuracy
    ):

        print(
            "V44 DECISION: STRUCTURAL SUCCESS"
        )

        print()
        print(
            "The slack spatial coupling improved contact "
            "retention while preserving useful accuracy."
        )

        print(
            "NEXT: visually replay SELECTED V44, then freeze "
            "architecture and move to held-out validation."
        )


    elif useful_accuracy:

        print(
            "V44 DECISION: ACCURACY GOOD, STABILITY NOT YET PROVEN"
        )

        print()
        print(
            "Do NOT tune passives/contact friction."
        )

        print(
            "Inspect selected trace/contact mechanics only."
        )


    else:

        print(
            "V44 DECISION: REJECT THIS SPATIAL ROUTE"
        )

        print()
        print(
            "Do NOT start another K/passive scan."
        )

        print(
            "The next architecture must use a different "
            "physical return routing."
        )


    print()
    print(
        f"SELECTED XML:\n{SELECTED_XML}"
    )

    print(
        f"\nSELECTED TRACE:\n{SELECTED_TRACE}"
    )

    print(
        f"\nSUMMARY CSV:\n{SUMMARY_CSV}"
    )

    print("=" * 215)


if __name__ == "__main__":
    main()
