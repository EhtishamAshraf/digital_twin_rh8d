#!/usr/bin/env python3

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

import test_v43_pd_compliant_coupling as v43


# ============================================================
# PATHS
# ============================================================

WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

# Exact candidate that produced:
#
# 13.682 / 23.043 / 9.739 deg
# contact RMSE = 0.704 deg
#
CANDIDATE = (
    MJCF
    / "tmp_v43_k750_r20.xml"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v43_visual_monitor"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRACE_FILE = (
    OUTDIR
    / "v43_k750_r20_exact_contact_trace.npz"
)


# ============================================================
# PLAYBACK SETTINGS
# ============================================================

# 0.25 = quarter-speed playback.
PLAYBACK_SPEED = 0.25

# Simulation timestep is 1 ms.
# Show every fifth recorded state -> smooth ~50 FPS at 0.25x.
VIS_STRIDE = 5

# Pause when a contact first appears so it is easy to inspect.
CONTACT_PAUSE = 0.8


# ============================================================
# NAMES
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


# ============================================================
# ENSURE EXACT V43 CANDIDATE EXISTS
# ============================================================

if not CANDIDATE.exists():

    print(
        "V43 candidate XML not found; rebuilding exact "
        "K=750 N/m, rest=20 deg candidate."
    )

    v43.build_candidate(
        750.0,
        20.0,
        CANDIDATE,
    )


print()
print("=" * 150)
print("V43 EXACT VISUAL MONITOR")
print("=" * 150)

print()
print(
    f"Candidate:\n{CANDIDATE}"
)

print()
print(
    "First pass: exact physical simulation + trace recording."
)

print(
    "Second pass: slow visual playback of the recorded states."
)


# ============================================================
# HELPERS
# ============================================================

def joint_qpos_addr(
    model,
    joint_name,
):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if jid < 0:

        raise RuntimeError(
            f"Missing joint: {joint_name}"
        )

    return int(
        model.jnt_qposadr[
            jid
        ]
    )


def geom_id(
    model,
    name,
):

    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        name,
    )


def contact_between(
    data,
    fixture_id,
    target_ids,
):

    if fixture_id < 0:

        return False

    valid_targets = {
        x
        for x in target_ids
        if x >= 0
    }

    if not valid_targets:

        return False


    for i in range(
        data.ncon
    ):

        c = data.contact[
            i
        ]

        g1 = int(
            c.geom1
        )

        g2 = int(
            c.geom2
        )


        if (
            g1 == fixture_id
            and
            g2 in valid_targets
        ):

            return True


        if (
            g2 == fixture_id
            and
            g1 in valid_targets
        ):

            return True


    return False


# ============================================================
# RECORD EXACT V43 CONTACT RUN
#
# IMPORTANT:
#
# We wrap mj_step only to COPY the resulting state AFTER each
# physics step.
#
# test_prescribed_spool_v35p3.py still performs the measured
# spool prescription exactly as before.
#
# MCP / PIP / DIP remain dynamic.
# ============================================================

trace_t = []
trace_qpos = []
trace_qvel = []
trace_ctrl = []
trace_mid = []
trace_tip = []

cached_model_id = None
cached_ids = None


REAL_MJ_STEP = (
    mujoco.mj_step
)


def recording_mj_step(
    model,
    data,
    *args,
    **kwargs,
):

    global cached_model_id
    global cached_ids


    # This calls the real MuJoCo integration step.
    out = REAL_MJ_STEP(
        model,
        data,
        *args,
        **kwargs,
    )


    # --------------------------------------------------------
    # Only capture the index contact model.
    # --------------------------------------------------------

    jm = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        MCP,
    )

    jp = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        PIP,
    )

    jd = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        DIP,
    )


    if (
        jm < 0
        or
        jp < 0
        or
        jd < 0
    ):

        return out


    # If some internal setup resets simulation time, retain
    # only the final continuous simulation segment.
    if (
        trace_t
        and
        float(
            data.time
        )
        <
        trace_t[-1]
        -
        1e-10
    ):

        trace_t.clear()
        trace_qpos.clear()
        trace_qvel.clear()
        trace_ctrl.clear()
        trace_mid.clear()
        trace_tip.clear()


    if (
        cached_model_id
        !=
        id(model)
    ):

        cached_model_id = (
            id(model)
        )

        fixture_id = geom_id(
            model,
            FIXTURE,
        )

        mid_ids = [
            geom_id(
                model,
                n,
            )
            for n in MID_GEOMS
        ]

        tip_ids = [
            geom_id(
                model,
                n,
            )
            for n in TIP_GEOMS
        ]

        cached_ids = (
            fixture_id,
            mid_ids,
            tip_ids,
        )


    fixture_id, mid_ids, tip_ids = (
        cached_ids
    )


    mid_contact = contact_between(
        data,
        fixture_id,
        mid_ids,
    )

    tip_contact = contact_between(
        data,
        fixture_id,
        tip_ids,
    )


    trace_t.append(
        float(
            data.time
        )
    )

    trace_qpos.append(
        np.array(
            data.qpos,
            dtype=float,
            copy=True,
        )
    )

    trace_qvel.append(
        np.array(
            data.qvel,
            dtype=float,
            copy=True,
        )
    )

    trace_ctrl.append(
        np.array(
            data.ctrl,
            dtype=float,
            copy=True,
        )
    )

    trace_mid.append(
        bool(
            mid_contact
        )
    )

    trace_tip.append(
        bool(
            tip_contact
        )
    )


    return out


# Install recorder BEFORE p3/V43 installs its spool-step wrapper.
#
# The spool wrapper will therefore call this recorder as its
# underlying MuJoCo step.
mujoco.mj_step = (
    recording_mj_step
)


try:

    result = v43.run_contact(
        CANDIDATE,
        "V43_VISUAL_CAPTURE",
    )

finally:

    # Absolutely restore normal MuJoCo function.
    mujoco.mj_step = (
        REAL_MJ_STEP
    )


if len(
    trace_t
) < 10:

    raise RuntimeError(
        "Trace capture failed: fewer than 10 states recorded."
    )


trace_t = np.asarray(
    trace_t,
    dtype=float,
)

trace_qpos = np.asarray(
    trace_qpos,
    dtype=float,
)

trace_qvel = np.asarray(
    trace_qvel,
    dtype=float,
)

trace_ctrl = np.asarray(
    trace_ctrl,
    dtype=float,
)

trace_mid = np.asarray(
    trace_mid,
    dtype=bool,
)

trace_tip = np.asarray(
    trace_tip,
    dtype=bool,
)


np.savez_compressed(
    TRACE_FILE,

    time=trace_t,

    qpos=trace_qpos,

    qvel=trace_qvel,

    ctrl=trace_ctrl,

    middle_contact=trace_mid,

    tip_contact=trace_tip,
)


print()
print("=" * 150)
print("NUMERIC RESULT FROM THIS EXACT CAPTURE")
print("=" * 150)

print(
    "pose = "
    f"{result['mcp_deg']:.3f} / "
    f"{result['pip_deg']:.3f} / "
    f"{result['dip_deg']:.3f} deg"
)

print(
    f"contact RMSE = "
    f"{result['rmse_deg']:.3f} deg"
)

print(
    f"middle/tip event count = "
    f"{result['mid_on']} / "
    f"{result['tip_on']}"
)

print(
    "middle -> first tip:"
)

print(
    f"  dPIP = "
    f"{result['middle_to_tip_dPIP_deg']:+.3f} deg"
)

print(
    f"  dDIP = "
    f"{result['middle_to_tip_dDIP_deg']:+.3f} deg"
)

print()
print(
    f"Recorded states = "
    f"{len(trace_t)}"
)

print(
    f"Trace saved:\n{TRACE_FILE}"
)


# ============================================================
# PRINT CONTACT TRANSITIONS
#
# This is important because V43 reported 3/3.
# ============================================================

print()
print("=" * 150)
print("RECORDED CONTACT TRANSITIONS")
print("=" * 150)

previous_m = bool(
    trace_mid[0]
)

previous_t = bool(
    trace_tip[0]
)


print(
    f"t={trace_t[0]:.3f}  "
    f"M={'ON ' if previous_m else 'OFF'}  "
    f"T={'ON ' if previous_t else 'OFF'}"
)


for i in range(
    1,
    len(trace_t),
):

    m = bool(
        trace_mid[i]
    )

    t = bool(
        trace_tip[i]
    )


    if (
        m != previous_m
        or
        t != previous_t
    ):

        print(
            f"t={trace_t[i]:.3f}  "
            f"M={'ON ' if m else 'OFF'}  "
            f"T={'ON ' if t else 'OFF'}"
        )


    previous_m = m
    previous_t = t


# ============================================================
# VISUAL PLAYBACK
# ============================================================

print()
print("=" * 150)
print("OPENING MUJOCO VIEWER")
print("=" * 150)

print()
print(
    f"Playback speed = "
    f"{PLAYBACK_SPEED:.2f}x"
)

print(
    "The final pose will remain frozen until you close "
    "the viewer."
)

print()
print(
    "Mouse controls remain active: rotate / zoom / pan."
)


model = mujoco.MjModel.from_xml_path(
    str(
        CANDIDATE
    )
)

data = mujoco.MjData(
    model
)


mcp_adr = joint_qpos_addr(
    model,
    MCP,
)

pip_adr = joint_qpos_addr(
    model,
    PIP,
)

dip_adr = joint_qpos_addr(
    model,
    DIP,
)


if (
    trace_qpos.shape[1]
    !=
    model.nq
):

    raise RuntimeError(
        "Recorded qpos dimension does not match candidate XML."
    )


with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:


    # --------------------------------------------------------
    # Useful visual flags
    # --------------------------------------------------------

    for flag_name in [
        "mjVIS_CONTACTPOINT",
        "mjVIS_CONTACTFORCE",
        "mjVIS_TENDON",
    ]:

        if hasattr(
            mujoco.mjtVisFlag,
            flag_name,
        ):

            flag = getattr(
                mujoco.mjtVisFlag,
                flag_name,
            )

            viewer.opt.flags[
                flag
            ] = True


    # --------------------------------------------------------
    # Start from first recorded state.
    # --------------------------------------------------------

    data.qpos[:] = (
        trace_qpos[0]
    )

    data.qvel[:] = (
        trace_qvel[0]
    )

    if (
        data.ctrl.shape
        ==
        trace_ctrl[0].shape
    ):

        data.ctrl[:] = (
            trace_ctrl[0]
        )


    mujoco.mj_forward(
        model,
        data,
    )


    # --------------------------------------------------------
    # Camera centered on the contact cylinder / index finger.
    #
    # You can still freely adjust this with the mouse.
    # --------------------------------------------------------

    fixture_id = geom_id(
        model,
        FIXTURE,
    )


    if fixture_id >= 0:

        viewer.cam.lookat[:] = (
            data.geom_xpos[
                fixture_id
            ]
        )


    viewer.cam.distance = (
        0.14
    )

    viewer.cam.azimuth = (
        90
    )

    viewer.cam.elevation = (
        -15
    )


    viewer.sync()


    last_mid = bool(
        trace_mid[0]
    )

    last_tip = bool(
        trace_tip[0]
    )


    wall_start = (
        time.perf_counter()
    )

    sim_start = (
        trace_t[0]
    )


    for i in range(
        0,
        len(trace_t),
        VIS_STRIDE,
    ):

        if not viewer.is_running():

            break


        data.qpos[:] = (
            trace_qpos[i]
        )

        data.qvel[:] = (
            trace_qvel[i]
        )

        if (
            data.ctrl.shape
            ==
            trace_ctrl[i].shape
        ):

            data.ctrl[:] = (
                trace_ctrl[i]
            )


        data.time = float(
            trace_t[i]
        )


        mujoco.mj_forward(
            model,
            data,
        )


        viewer.sync()


        mcp_deg = np.degrees(
            data.qpos[
                mcp_adr
            ]
        )

        pip_deg = np.degrees(
            data.qpos[
                pip_adr
            ]
        )

        dip_deg = np.degrees(
            data.qpos[
                dip_adr
            ]
        )


        middle_now = bool(
            trace_mid[i]
        )

        tip_now = bool(
            trace_tip[i]
        )


        print(
            "\r"
            f"t={trace_t[i]:6.3f} s   "
            f"MCP={mcp_deg:7.2f}°   "
            f"PIP={pip_deg:7.2f}°   "
            f"DIP={dip_deg:7.2f}°   "
            f"M={'ON ' if middle_now else 'OFF'}   "
            f"T={'ON ' if tip_now else 'OFF'}",
            end="",
            flush=True,
        )


        # Pause on NEW middle/tip contact.
        if (
            middle_now
            and
            not last_mid
        ):

            print(
                "\n>>> MIDDLE CONTACT ON"
            )

            time.sleep(
                CONTACT_PAUSE
            )


        if (
            tip_now
            and
            not last_tip
        ):

            print(
                "\n>>> TIP CONTACT ON"
            )

            time.sleep(
                CONTACT_PAUSE
            )


        last_mid = (
            middle_now
        )

        last_tip = (
            tip_now
        )


        # ----------------------------------------------------
        # Real-time paced slow motion.
        # ----------------------------------------------------

        desired_wall = (
            wall_start
            +
            (
                trace_t[i]
                -
                sim_start
            )
            /
            PLAYBACK_SPEED
        )


        remaining = (
            desired_wall
            -
            time.perf_counter()
        )


        if remaining > 0:

            time.sleep(
                remaining
            )


    print()
    print()
    print("=" * 150)
    print("REPLAY COMPLETE — FINAL POSE HELD")
    print("=" * 150)


    # --------------------------------------------------------
    # Hold exact final state.
    # --------------------------------------------------------

    data.qpos[:] = (
        trace_qpos[-1]
    )

    data.qvel[:] = 0.0

    if (
        data.ctrl.shape
        ==
        trace_ctrl[-1].shape
    ):

        data.ctrl[:] = (
            trace_ctrl[-1]
        )


    data.time = float(
        trace_t[-1]
    )


    mujoco.mj_forward(
        model,
        data,
    )


    final_mcp = np.degrees(
        data.qpos[
            mcp_adr
        ]
    )

    final_pip = np.degrees(
        data.qpos[
            pip_adr
        ]
    )

    final_dip = np.degrees(
        data.qpos[
            dip_adr
        ]
    )


    print(
        "Final visual state:"
    )

    print(
        f"  MCP = "
        f"{final_mcp:.3f} deg"
    )

    print(
        f"  PIP = "
        f"{final_pip:.3f} deg"
    )

    print(
        f"  DIP = "
        f"{final_dip:.3f} deg"
    )

    print()
    print(
        "Close the MuJoCo window when you are finished inspecting."
    )


    while viewer.is_running():

        viewer.sync()

        time.sleep(
            0.03
        )


print()
print("Viewer closed.")
