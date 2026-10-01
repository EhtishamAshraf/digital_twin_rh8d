#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import math

import mujoco
import numpy as np

import scan_index_pip_passive_dynamic_v2 as base
import compare_index_configdep_replay_v6 as ab
import diagnose_index_guide_slip_v13 as tendon_tools
import scan_and_view_index_contact_friction_v18 as viewer_tools

from mujoco_viewer_style import make_index_finger_solid


# ============================================================
# V24 — LOAD-TRIGGERED EFFECTIVE TENDON REDISTRIBUTION
#
# V23 showed:
#
#   - mechanical breakaway can preserve free space
#   - middle -> breakaway -> tip sequence can appear
#   - but PIP reaches only ~6 deg
#
# V24 retains that mechanical trigger and adds a post-yield
# change in effective tendon moment arms.
#
# NO contact flag.
# NO bottle-dependent switching.
# Bottle registration remains frozen V5.
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
      "stateful_transmission_v24"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY = RESULT_DIR / "summary.csv"

SELECTED_MODEL = (
    MJCF_DIR
    / "ur5e_rh8d_index_stateful_transmission_v24_candidate.xml"
)


# ============================================================
# FROZEN MODEL
# ============================================================

SCALE = 5.10e-6
TENDON_K = 2100.0

CONTACT_MU = 0.05
CONTACT_K = 1000.0

BREAKAWAY_THRESHOLD = 0.015

# Best V23 released-state passive branch.
POST_SPRING_SCALE = 0.0
POST_FRICTION_SCALE = 0.25


MCP = "Index_Proximal--palm_assembly:1"
PIP = "Index_Middle--Index_Proximal"
DIP = "Index_Distal--Index_Middle"

SPOOL = "index_motor_spool"

TENDON = "idx_flex_spatial"


BASE_ARMS_MM = np.array(
    [5.0, 4.0, 1.0],
    dtype=float,
)


HW_CONTACT = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


FREE_GATE = 6.085


# ============================================================
# SMALL TARGETED GRID
# ============================================================

CASES = [

    {
        "name": "BASELINE_NO_YIELD",
        "yield": False,
        "arms_mm": [5.0, 4.0, 1.0],
    },

    {
        "name": "V23_CONTROL",
        "yield": True,
        "arms_mm": [5.0, 4.0, 1.0],
    },

    {
        "name": "R5_6_1",
        "yield": True,
        "arms_mm": [5.0, 6.0, 1.0],
    },

    {
        "name": "R4p5_6p5_1p2",
        "yield": True,
        "arms_mm": [4.5, 6.5, 1.2],
    },

    {
        "name": "R4p5_7_1p2",
        "yield": True,
        "arms_mm": [4.5, 7.0, 1.2],
    },

    {
        "name": "R4_7_1p5",
        "yield": True,
        "arms_mm": [4.0, 7.0, 1.5],
    },

    {
        "name": "R4_8_1p5",
        "yield": True,
        "arms_mm": [4.0, 8.0, 1.5],
    },

    {
        "name": "R3p5_8_1p5",
        "yield": True,
        "arms_mm": [3.5, 8.0, 1.5],
    },

    {
        "name": "R3p5_9_2",
        "yield": True,
        "arms_mm": [3.5, 9.0, 2.0],
    },
]


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(output):

    ab.replace_index_with_constant_tendon(
        ab.V5_SOURCE,
        output,
    )


# ============================================================
# STATEFUL TRANSMISSION LAW
# ============================================================

class StatefulTransmission:

    def __init__(self):

        self.original_step = mujoco.mj_step

        self.case = None
        self.phase = None

        self.cache = {}
        self.states = {}

        self.events = []

        self.styled_models = set()


    def configure(
        self,
        case,
        phase,
    ):

        self.case = case
        self.phase = phase


    def ids(
        self,
        model,
    ):

        key = id(model)

        if key in self.cache:
            return self.cache[key]


        def joint_data(name):

            jid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            if jid < 0:
                raise RuntimeError(
                    f"Missing joint {name}"
                )

            return {
                "jid": jid,
                "qadr": int(
                    model.jnt_qposadr[jid]
                ),
                "dof": int(
                    model.jnt_dofadr[jid]
                ),
            }


        try:

            tendon_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_TENDON,
                TENDON,
            )

            if tendon_id < 0:
                raise RuntimeError(
                    f"Missing tendon {TENDON}"
                )


            result = {
                "mcp": joint_data(MCP),
                "pip": joint_data(PIP),
                "dip": joint_data(DIP),
                "spool": joint_data(SPOOL),
                "tendon": tendon_id,
            }


        except RuntimeError:

            result = None


        self.cache[key] = result

        return result


    def state(
        self,
        model,
        ids,
    ):

        key = id(model)

        if key not in self.states:

            self.states[key] = {

                "yielded": False,

                "baseline_pip_k":
                    float(
                        model.jnt_stiffness[
                            ids["pip"]["jid"]
                        ]
                    ),

                "baseline_pip_f":
                    float(
                        model.dof_frictionloss[
                            ids["pip"]["dof"]
                        ]
                    ),
            }


        return self.states[key]


    def apply_visual_style(
        self,
        model,
    ):

        key = id(model)

        if key in self.styled_models:
            return

        make_index_finger_solid(
            model,
            alpha=1.0,
        )

        self.styled_models.add(key)


    def pre_step(
        self,
        model,
        data,
        ids,
        state,
    ):

        self.apply_visual_style(
            model
        )


        # Clear only forces owned by this experimental law.
        for key in [
            "mcp",
            "pip",
            "dip",
        ]:

            data.qfrc_applied[
                ids[key]["dof"]
            ] = 0.0


        # ----------------------------------------------------
        # BEFORE YIELD
        # ----------------------------------------------------

        if (
            not state["yielded"]
            or
            not self.case["yield"]
        ):

            model.jnt_stiffness[
                ids["pip"]["jid"]
            ] = (
                state[
                    "baseline_pip_k"
                ]
            )


            model.dof_frictionloss[
                ids["pip"]["dof"]
            ] = (
                state[
                    "baseline_pip_f"
                ]
            )


            return


        # ----------------------------------------------------
        # AFTER YIELD:
        #
        # 1. released PIP return branch from V23
        #
        # 2. load-dependent effective transmission change
        # ----------------------------------------------------

        model.jnt_stiffness[
            ids["pip"]["jid"]
        ] = (
            state["baseline_pip_k"]
            *
            POST_SPRING_SCALE
        )


        model.dof_frictionloss[
            ids["pip"]["dof"]
        ] = (
            state["baseline_pip_f"]
            *
            POST_FRICTION_SCALE
        )


        tendon = tendon_tools.tendon_state(
            model,
            data,
            ids["tendon"],
        )


        tension = max(
            0.0,
            float(
                tendon[
                    "tension_N"
                ]
            ),
        )


        target_arms = (
            np.asarray(
                self.case[
                    "arms_mm"
                ],
                dtype=float,
            )
            *
            1.0e-3
        )


        base_arms = (
            BASE_ARMS_MM
            *
            1.0e-3
        )


        delta = (
            target_arms
            -
            base_arms
        )


        # tau = T * r.
        #
        # The original fixed tendon already produces
        # T * BASE_ARMS.
        #
        # Add only the difference required to emulate the
        # post-yield effective transmission.
        additional_tau = (
            tension
            *
            delta
        )


        for key, tau in zip(
            [
                "mcp",
                "pip",
                "dip",
            ],
            additional_tau,
        ):

            data.qfrc_applied[
                ids[key]["dof"]
            ] += float(tau)


    def post_step(
        self,
        model,
        data,
        ids,
        state,
    ):

        if (
            not self.case["yield"]
            or
            state["yielded"]
        ):

            return


        load = abs(
            float(
                data.qfrc_constraint[
                    ids[
                        "pip"
                    ][
                        "dof"
                    ]
                ]
            )
        )


        if load < BREAKAWAY_THRESHOLD:
            return


        state["yielded"] = True


        encoder = (
            100.0
            +
            float(
                data.qpos[
                    ids[
                        "spool"
                    ][
                        "qadr"
                    ]
                ]
            )
            /
            SCALE
        )


        event = {

            "case":
                self.case["name"],

            "phase":
                self.phase,

            "time":
                float(
                    data.time
                ),

            "encoder":
                encoder,

            "pip_deg":
                math.degrees(
                    float(
                        data.qpos[
                            ids[
                                "pip"
                            ][
                                "qadr"
                            ]
                        ]
                    )
                ),

            "load_Nm":
                load,
        }


        self.events.append(event)


        print(
            "\n[TRANSMISSION BREAKAWAY] "
            f"{event['case']} "
            f"{event['phase']} "
            f"encoder={event['encoder']:.1f} "
            f"PIP={event['pip_deg']:.2f} deg "
            f"load={event['load_Nm']:.4f} Nm"
        )


    def install(self):

        original = self.original_step
        law = self


        def wrapped(
            model,
            data,
            *args,
            **kwargs,
        ):

            if law.case is None:

                return original(
                    model,
                    data,
                    *args,
                    **kwargs,
                )


            ids = law.ids(model)

            if ids is None:

                return original(
                    model,
                    data,
                    *args,
                    **kwargs,
                )


            state = law.state(
                model,
                ids,
            )


            law.pre_step(
                model,
                data,
                ids,
                state,
            )


            result = original(
                model,
                data,
                *args,
                **kwargs,
            )


            law.post_step(
                model,
                data,
                ids,
                state,
            )


            return result


        mujoco.mj_step = wrapped


    def restore(self):

        mujoco.mj_step = (
            self.original_step
        )


# ============================================================
# EVENT / CONTACT HELPERS
# ============================================================

def first_event(
    events,
    case,
    phase,
):

    for e in events:

        if (
            e["case"]
            == case
            and
            e["phase"]
            == phase
        ):

            return e

    return None


def first_contact(
    rows,
    prefix,
):

    if not rows:
        return None


    t0 = rows[0]["time"]


    for row in rows:

        if (
            row[
                f"{prefix}_rows"
            ] > 0
            and
            row[
                f"{prefix}_Fn_N"
            ] > 1.0e-4
        ):

            return float(
                row["time"]
                -
                t0
            )


    return None


def contact_duration(
    rows,
    prefix,
):

    if len(rows) < 2:
        return 0.0


    dt = float(
        np.median(
            np.diff(
                [
                    row["time"]
                    for row in rows
                ]
            )
        )
    )


    count = sum(
        1
        for row in rows
        if (
            row[
                f"{prefix}_rows"
            ] > 0
            and
            row[
                f"{prefix}_Fn_N"
            ] > 1.0e-4
        )
    )


    return count * dt


# ============================================================
# RUN ONE CASE
# ============================================================

def run_case(
    case,
    free_bag,
    contact_bag,
):

    label = (
        case["name"]
        .lower()
        .replace(".", "p")
    )


    contact_xml = (
        MJCF_DIR
        /
        f"tmp_v24_{label}_contact.xml"
    )


    free_xml = (
        MJCF_DIR
        /
        f"tmp_v24_{label}_free.xml"
    )


    build_model(
        contact_xml
    )


    ab.make_free_model(
        contact_xml,
        free_xml,
    )


    law = StatefulTransmission()
    law.install()


    old_scale = base.METERS_PER_TICK
    old_k = base.TENDON_STIFFNESS
    old_mu = base.CONTACT_MU
    old_contact_k = base.CONTACT_STIFFNESS


    try:

        base.METERS_PER_TICK = SCALE
        base.TENDON_STIFFNESS = TENDON_K

        base.CONTACT_MU = CONTACT_MU
        base.CONTACT_STIFFNESS = CONTACT_K


        # ----------------------------------------------------
        # FREE
        # ----------------------------------------------------

        law.configure(
            case,
            "FREE",
        )


        free_metrics, _ = base.run_free(
            free_xml,
            free_bag,
        )


        # ----------------------------------------------------
        # CONTACT
        # ----------------------------------------------------

        recorder = viewer_tools.ContactRecorder()
        recorder.install()


        try:

            law.configure(
                case,
                "CONTACT",
            )


            contact_metrics, _ = base.run_contact(
                contact_xml,
                contact_bag,
            )


        finally:

            recorder.restore()


    finally:

        law.restore()

        base.METERS_PER_TICK = old_scale
        base.TENDON_STIFFNESS = old_k

        base.CONTACT_MU = old_mu
        base.CONTACT_STIFFNESS = old_contact_k


    pose = np.asarray(
        contact_metrics["pose"],
        dtype=float,
    )


    rmse = float(
        np.sqrt(
            np.mean(
                (
                    pose
                    -
                    HW_CONTACT
                )
                ** 2
            )
        )
    )


    free_break = first_event(
        law.events,
        case["name"],
        "FREE",
    )


    contact_break = first_event(
        law.events,
        case["name"],
        "CONTACT",
    )


    middle_first = first_contact(
        recorder.rows,
        "mid",
    )


    tip_first = first_contact(
        recorder.rows,
        "tip",
    )


    middle_duration = contact_duration(
        recorder.rows,
        "mid",
    )


    tip_duration = contact_duration(
        recorder.rows,
        "tip",
    )


    if (
        contact_break is not None
        and
        recorder.rows
    ):

        t0 = recorder.rows[0]["time"]

        break_elapsed = (
            contact_break["time"]
            -
            t0
        )

    else:

        break_elapsed = None


    sequence_ok = (
        middle_first is not None
        and
        break_elapsed is not None
        and
        tip_first is not None
        and
        middle_first
        <=
        break_elapsed
        <=
        tip_first
    )


    result = {

        "name":
            case["name"],

        "post_mcp_mm":
            case["arms_mm"][0],

        "post_pip_mm":
            case["arms_mm"][1],

        "post_dip_mm":
            case["arms_mm"][2],

        "free_combined":
            free_metrics["combined"],

        "free_breakaway":
            free_break is not None,

        "contact_mcp":
            pose[0],

        "contact_pip":
            pose[1],

        "contact_dip":
            pose[2],

        "contact_rmse":
            rmse,

        "contact_breakaway":
            contact_break is not None,

        "middle_first_s":
            (
                middle_first
                if middle_first is not None
                else -1.0
            ),

        "breakaway_s":
            (
                break_elapsed
                if break_elapsed is not None
                else -1.0
            ),

        "tip_first_s":
            (
                tip_first
                if tip_first is not None
                else -1.0
            ),

        "middle_duration_s":
            middle_duration,

        "tip_duration_s":
            tip_duration,

        "sequence_ok":
            sequence_ok,
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
        SELECTED_MODEL
    )


    law = StatefulTransmission()
    law.install()


    old_scale = (
        viewer_tools.base.METERS_PER_TICK
    )

    old_k = (
        viewer_tools.base.TENDON_STIFFNESS
    )

    old_mu = (
        viewer_tools.base.CONTACT_MU
    )

    old_contact_k = (
        viewer_tools.base.CONTACT_STIFFNESS
    )


    try:

        viewer_tools.base.METERS_PER_TICK = SCALE
        viewer_tools.base.TENDON_STIFFNESS = TENDON_K

        viewer_tools.base.CONTACT_MU = CONTACT_MU
        viewer_tools.base.CONTACT_STIFFNESS = CONTACT_K


        law.configure(
            case,
            "CONTACT",
        )


        viewer_tools.visual_replay(
            SELECTED_MODEL,
            contact_bag,
        )


    finally:

        law.restore()

        viewer_tools.base.METERS_PER_TICK = old_scale
        viewer_tools.base.TENDON_STIFFNESS = old_k

        viewer_tools.base.CONTACT_MU = old_mu
        viewer_tools.base.CONTACT_STIFFNESS = old_contact_k


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


    if not free_bag.exists():
        raise RuntimeError(
            f"Missing free bag: {free_bag}"
        )


    if not contact_bag.exists():
        raise RuntimeError(
            f"Missing contact bag: {contact_bag}"
        )


    ab.patch_replay_modules()


    print()
    print("=" * 122)
    print(
        "RH8D INDEX — STATEFUL TENDON "
        "TRANSMISSION SCAN V24"
    )
    print("=" * 122)


    print()
    print(
        "Frozen:"
    )

    print(
        "  object pose = V5"
    )

    print(
        "  pre-yield transmission = 5 / 4 / 1 mm"
    )

    print(
        f"  breakaway threshold = "
        f"{BREAKAWAY_THRESHOLD:.3f} Nm"
    )

    print(
        f"  post-yield PIP spring scale = "
        f"{POST_SPRING_SCALE:.2f}"
    )

    print(
        f"  post-yield PIP friction scale = "
        f"{POST_FRICTION_SCALE:.2f}"
    )

    print(
        "  contact = mu 0.05, k 1000 N/m"
    )


    results = []


    for i, case in enumerate(
        CASES,
        start=1,
    ):

        print()
        print("=" * 122)

        print(
            f"[{i:02d}/{len(CASES):02d}] "
            f"{case['name']} "
            f"post arms = "
            f"{case['arms_mm']} mm"
        )

        print("=" * 122)


        result = run_case(
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
            f"{result['free_combined']:.3f} deg"
        )

        print(
            f"Free breakaway = "
            f"{result['free_breakaway']}"
        )


        print()
        print(
            "Contact MCP/PIP/DIP = "
            f"{result['contact_mcp']:.3f} / "
            f"{result['contact_pip']:.3f} / "
            f"{result['contact_dip']:.3f} deg"
        )

        print(
            f"Contact RMSE = "
            f"{result['contact_rmse']:.3f} deg"
        )


        print()
        print(
            "Sequence:"
        )

        print(
            f"  middle = "
            f"{result['middle_first_s']:.3f} s"
        )

        print(
            f"  yield  = "
            f"{result['breakaway_s']:.3f} s"
        )

        print(
            f"  tip    = "
            f"{result['tip_first_s']:.3f} s"
        )

        print(
            f"  tip duration = "
            f"{result['tip_duration_s']:.3f} s"
        )

        print(
            f"  sequence OK = "
            f"{result['sequence_ok']}"
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
                results[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # ========================================================
    # FREE-SPACE PRESERVATION
    # ========================================================

    valid = [
        r
        for r in results
        if (
            r["free_combined"]
            <= FREE_GATE
            and
            not r["free_breakaway"]
        )
    ]


    if not valid:

        raise RuntimeError(
            "No V24 candidate preserved free space."
        )


    sequence = [
        r
        for r in valid
        if r["sequence_ok"]
    ]


    pool = (
        sequence
        if sequence
        else valid
    )


    selected = min(
        pool,
        key=lambda r:
            r["contact_rmse"],
    )


    selected_case = next(
        c
        for c in CASES
        if (
            c["name"]
            ==
            selected["name"]
        )
    )


    baseline = next(
        r
        for r in results
        if (
            r["name"]
            ==
            "BASELINE_NO_YIELD"
        )
    )


    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 122)
    print(
        "V24 SELECTED STATEFUL TRANSMISSION"
    )
    print("=" * 122)


    print()
    print(
        f"Candidate = "
        f"{selected['name']}"
    )

    print(
        "Post-yield effective moment arms = "
        f"{selected['post_mcp_mm']:.2f} / "
        f"{selected['post_pip_mm']:.2f} / "
        f"{selected['post_dip_mm']:.2f} mm"
    )


    print()
    print(
        f"Free-space RMSE = "
        f"{selected['free_combined']:.3f} deg"
    )

    print(
        f"Free-space breakaway = "
        f"{selected['free_breakaway']}"
    )


    print()
    print(
        "Hardware contact target:"
    )

    print(
        "  14.536 / 23.811 / 9.327 deg"
    )


    print(
        "Prediction:"
    )

    print(
        "  "
        f"{selected['contact_mcp']:.3f} / "
        f"{selected['contact_pip']:.3f} / "
        f"{selected['contact_dip']:.3f} deg"
    )


    print(
        f"Contact RMSE = "
        f"{selected['contact_rmse']:.3f} deg"
    )


    print()
    print(
        f"Baseline RMSE = "
        f"{baseline['contact_rmse']:.3f} deg"
    )

    print(
        f"Improvement = "
        f"{baseline['contact_rmse'] - selected['contact_rmse']:+.3f} deg"
    )


    print()
    print(
        "Sequence:"
    )

    print(
        f"  middle contact = "
        f"{selected['middle_first_s']:.3f} s"
    )

    print(
        f"  transmission yield = "
        f"{selected['breakaway_s']:.3f} s"
    )

    print(
        f"  tip contact = "
        f"{selected['tip_first_s']:.3f} s"
    )

    print(
        f"  sequence OK = "
        f"{selected['sequence_ok']}"
    )


    print()
    print(
        "Summary:"
    )

    print(
        SUMMARY
    )


    # ========================================================
    # DECISION
    # ========================================================

    print()
    print("=" * 122)
    print(
        "V24 DECISION"
    )
    print("=" * 122)


    if (
        selected["contact_pip"]
        >= 18.0
        and
        selected["tip_duration_s"]
        > 0.0
        and
        selected["contact_rmse"]
        <= 6.0
    ):

        print()
        print(
            "MAJOR STRUCTURAL SUCCESS."
        )

        print(
            "We now have a model family that preserves free "
            "space and reproduces substantial PIP contact "
            "redistribution."
        )

        print()
        print(
            "STOP architecture searching."
        )

        print(
            "Next: narrow identification + held-out trial."
        )


    elif (
        selected["contact_pip"]
        >= 15.0
    ):

        print()
        print(
            "STRONG PARTIAL SUCCESS."
        )

        print(
            "Do one small local refinement around this "
            "post-yield transmission only."
        )


    else:

        print()
        print(
            "The post-yield transmission shift is still "
            "insufficient."
        )

        print(
            "Do not run another broad ratio scan."
        )

        print(
            "Next step should introduce an explicit "
            "load-sharing/internal tendon state."
        )


    # ========================================================
    # GUI — BEST ONLY
    # ========================================================

    if not args.no_viewer:

        print()
        print("=" * 122)
        print(
            "V24 AUTOMATIC BEST-CANDIDATE GUI"
        )
        print("=" * 122)


        print()
        print(
            f"Viewing {selected['name']}"
        )

        print(
            "Index visual alpha = 1.0"
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
        "V24 COMPLETE"
    )
    print("=" * 122)


if __name__ == "__main__":
    main()
