#!/usr/bin/env python3

from pathlib import Path
import csv
import math

import mujoco
import numpy as np

import fit_index_passive_contact_v35b2 as b2
import scan_index_zero_margin_damping_v35e2b as e2b


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MJCF = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
)

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v36a_segmented_zero_friction"
)

BASELINE = (
    MJCF
    / "ur5e_rh8d_index_cad_spatial_v35i1_best.xml"
)

V36A = (
    MJCF
    / "ur5e_rh8d_index_segmented_v36a.xml"
)

BAG = Path(
    "/home/ehtisham/rh8d_system_id/"
    "index_contact_20260911_164310"
)


BASELINE_K = 9000.0

CONTACT_K = 1000.0
CONTACT_DAMPING = 20.0
CONTACT_MU = 0.05


JOINTS = [
    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",
]

V36_INTERNAL = [
    "idx_v36_slip_mcp",
    "idx_v36_slip_pip",
    "idx_v36_ext1",
    "idx_v36_ext2",
    "idx_v36_ext3",
]


def qadr(model, name):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        return None

    return int(
        model.jnt_qposadr[
            jid
        ]
    )


class Recorder:

    def __init__(self):

        self.rows = []

        self.counter = 0

        self.final_internal = {}


    def inspect(
        self,
        model,
        data,
    ):

        self.counter += 1

        # 5 ms sampling is enough for trajectory equivalence.
        if (
            self.counter % 5
        ) != 0:
            return


        row = {
            "time":
                float(
                    data.time
                ),
        }


        for key, name in zip(
            [
                "mcp",
                "pip",
                "dip",
            ],
            JOINTS,
        ):

            qa = qadr(
                model,
                name,
            )

            if qa is not None:

                row[
                    key
                ] = math.degrees(
                    float(
                        data.qpos[
                            qa
                        ]
                    )
                )


        self.rows.append(
            row
        )


        for name in V36_INTERNAL:

            qa = qadr(
                model,
                name,
            )

            if qa is not None:

                self.final_internal[
                    name
                ] = float(
                    data.qpos[
                        qa
                    ]
                )


def run_case(
    xml,
    full_tendon_runtime_k,
    contact_replay,
):

    recorder = Recorder()

    real_step = mujoco.mj_step


    def wrapped_step(
        model,
        data,
        *args,
        **kwargs,
    ):

        out = real_step(
            model,
            data,
            *args,
            **kwargs,
        )

        recorder.inspect(
            model,
            data,
        )

        return out


    mujoco.mj_step = wrapped_step


    old_v35_k = float(
        b2.v35.TENDON_K
    )

    old_base_k = float(
        b2.v30.v24.base.TENDON_STIFFNESS
    )


    try:

        # V35:
        #   use 9000 N/m whole spatial tendon.
        #
        # V36:
        #   original whole tendon is visual only,
        #   therefore runtime stiffness MUST remain zero.
        b2.v35.TENDON_K = float(
            full_tendon_runtime_k
        )

        b2.v30.v24.base.TENDON_STIFFNESS = float(
            full_tendon_runtime_k
        )


        result = e2b.run_one(
            CONTACT_DAMPING,
            xml,
            BAG,
            contact_replay,
        )


    finally:

        mujoco.mj_step = real_step

        b2.v35.TENDON_K = old_v35_k

        b2.v30.v24.base.TENDON_STIFFNESS = (
            old_base_k
        )


    return (
        result,
        recorder,
    )


def trajectory_rmse(
    a,
    b,
):

    if not a.rows or not b.rows:

        return {
            "mcp": math.nan,
            "pip": math.nan,
            "dip": math.nan,
        }


    ta = np.array(
        [
            r["time"]
            for r in a.rows
        ],
        dtype=float,
    )

    tb = np.array(
        [
            r["time"]
            for r in b.rows
        ],
        dtype=float,
    )


    t0 = max(
        ta.min(),
        tb.min(),
    )

    t1 = min(
        ta.max(),
        tb.max(),
    )


    mask = (
        (ta >= t0)
        &
        (ta <= t1)
    )

    t = ta[
        mask
    ]


    out = {}


    for key in [
        "mcp",
        "pip",
        "dip",
    ]:

        ya = np.array(
            [
                r[key]
                for r in a.rows
            ],
            dtype=float,
        )[
            mask
        ]

        yb_raw = np.array(
            [
                r[key]
                for r in b.rows
            ],
            dtype=float,
        )

        yb = np.interp(
            t,
            tb,
            yb_raw,
        )


        out[
            key
        ] = float(
            np.sqrt(
                np.mean(
                    (
                        ya
                        -
                        yb
                    )
                    ** 2
                )
            )
        )


    return out


def save_trace(
    filename,
    rec,
):

    path = (
        OUTDIR
        /
        filename
    )


    if not rec.rows:
        return


    with path.open(
        "w",
        newline="",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "time",
                "mcp",
                "pip",
                "dip",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rec.rows
        )


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    if not BASELINE.exists():

        raise RuntimeError(
            f"Missing V35 baseline:\n{BASELINE}"
        )

    if not V36A.exists():

        raise RuntimeError(
            f"Missing V36A:\n{V36A}"
        )


    print()
    print("=" * 135)
    print("V36A MANDATORY ZERO-FRICTION EQUIVALENCE TEST")
    print("=" * 135)

    print(
        "Reference:"
    )

    print(
        "  V35 CAD spatial tendon"
    )

    print(
        f"  K = {BASELINE_K:.0f} N/m"
    )

    print()
    print(
        "Candidate:"
    )

    print(
        "  V36A 3 cable segments"
    )

    print(
        "  MCP guide friction = 0"
    )

    print(
        "  PIP guide friction = 0"
    )

    print(
        "  original full tendon mechanical K = 0"
    )


    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
    )


    old_cd = float(
        contact_replay.CONTACT_DAMPING
    )

    old_mu = float(
        b2.v35.CONTACT_MU
    )

    old_contact_k = float(
        b2.v35.CONTACT_K
    )


    contact_replay.CONTACT_DAMPING = (
        CONTACT_DAMPING
    )

    b2.v35.CONTACT_MU = (
        CONTACT_MU
    )

    b2.v35.CONTACT_K = (
        CONTACT_K
    )


    try:

        print()
        print(
            "Running V35 K=9000..."
        )

        r35, rec35 = run_case(
            BASELINE,
            BASELINE_K,
            contact_replay,
        )


        print()
        print(
            "Running V36A zero-friction segmented cable..."
        )

        r36, rec36 = run_case(
            V36A,
            0.0,
            contact_replay,
        )


    finally:

        contact_replay.CONTACT_DAMPING = old_cd
        b2.v35.CONTACT_MU = old_mu
        b2.v35.CONTACT_K = old_contact_k


    q35 = np.asarray(
        r35[
            "pose"
        ],
        dtype=float,
    )

    q36 = np.asarray(
        r36[
            "pose"
        ],
        dtype=float,
    )


    delta = (
        q36
        -
        q35
    )


    traj = trajectory_rmse(
        rec35,
        rec36,
    )


    print()
    print("=" * 135)
    print("V35 K=9000")
    print("=" * 135)

    print(
        "pose = "
        f"{q35[0]:.3f} / "
        f"{q35[1]:.3f} / "
        f"{q35[2]:.3f} deg"
    )

    print(
        f"hardware RMSE = "
        f"{r35['rmse']:.3f} deg"
    )

    print(
        "transitions M/T = "
        f"{r35.get('mid_transitions')} / "
        f"{r35.get('tip_transitions')}"
    )

    print(
        "gaps M/T = "
        f"{r35.get('target_mid_gap_mm'):+.3f} / "
        f"{r35.get('target_tip_gap_mm'):+.3f} mm"
    )


    print()
    print("=" * 135)
    print("V36A ZERO-FRICTION SEGMENTED")
    print("=" * 135)

    print(
        "pose = "
        f"{q36[0]:.3f} / "
        f"{q36[1]:.3f} / "
        f"{q36[2]:.3f} deg"
    )

    print(
        f"hardware RMSE = "
        f"{r36['rmse']:.3f} deg"
    )

    print(
        "transitions M/T = "
        f"{r36.get('mid_transitions')} / "
        f"{r36.get('tip_transitions')}"
    )

    print(
        "gaps M/T = "
        f"{r36.get('target_mid_gap_mm'):+.3f} / "
        f"{r36.get('target_tip_gap_mm'):+.3f} mm"
    )


    print()
    print("=" * 135)
    print("EQUIVALENCE ERROR")
    print("=" * 135)

    print(
        "target-pose V36 - V35 = "
        f"{delta[0]:+.3f} / "
        f"{delta[1]:+.3f} / "
        f"{delta[2]:+.3f} deg"
    )

    print(
        "whole-trajectory RMSE V36 vs V35:"
    )

    print(
        f"  MCP = "
        f"{traj['mcp']:.4f} deg"
    )

    print(
        f"  PIP = "
        f"{traj['pip']:.4f} deg"
    )

    print(
        f"  DIP = "
        f"{traj['dip']:.4f} deg"
    )


    print()
    print("=" * 135)
    print("V36 INTERNAL FINAL STATES")
    print("=" * 135)


    for name in V36_INTERNAL:

        if name in rec36.final_internal:

            value = rec36.final_internal[
                name
            ]

            print(
                f"{name:22s} = "
                f"{value * 1000.0:+.6f} mm"
            )


    save_trace(
        "v35_k9000_trace.csv",
        rec35,
    )

    save_trace(
        "v36a_trace.csv",
        rec36,
    )


    max_pose_diff = float(
        np.max(
            np.abs(
                delta
            )
        )
    )

    max_traj_rmse = max(
        traj.values()
    )


    print()
    print("=" * 135)

    if (
        max_pose_diff <= 0.75
        and
        max_traj_rmse <= 1.0
    ):

        print(
            "V36A EQUIVALENCE: PASS"
        )

        print(
            "Zero-friction segmented cable is close enough "
            "to proceed to guide-friction identification."
        )

    else:

        print(
            "V36A EQUIVALENCE: FAIL"
        )

        print(
            "Do NOT tune guide friction yet."
        )

        print(
            "First fix the segmented-cable implementation."
        )


    print()
    print(
        f"All outputs:\n{OUTDIR}"
    )

    print("=" * 135)


if __name__ == "__main__":
    main()
