#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

import fit_index_passive_contact_v35b2 as b2
import replay_index_zero_margin_v35e2a as e2a


# ================================================================
# V35E2B — ZERO-MARGIN DYNAMIC DAMPING SCAN
#
# VARIABLE ONLY:
#     direct-format contact damping
#
# FROZEN:
#     contact margin = 0
#     direct stiffness = 1000
#     mu = 0.05
#     CAD spatial tendon
#     V35B2 passives
#     H1 bottle pose
#     hardware encoder replay
#
# NO prescribed finger angles.
# NO contact-triggered mechanics.
# ================================================================


DAMPING_VALUES = [
    0.25,
    0.50,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
]

CONTACT_K = 1000.0
CONTACT_MU = 0.05

FORCE_THRESHOLD = 0.10

HW_FORCE = 0.14563

HW_HOLD_START = 38.051338
HW_HOLD_END = 40.074176


def transitions(
    flags,
):

    flags = np.asarray(
        flags,
        dtype=bool,
    )

    if flags.size < 2:
        return 0

    return int(
        np.sum(
            flags[1:]
            !=
            flags[:-1]
        )
    )


def longest_true(
    flags,
    dt,
):

    best = 0
    current = 0

    for value in flags:

        if value:
            current += 1
            best = max(
                best,
                current,
            )
        else:
            current = 0

    return (
        best
        *
        dt
    )


def run_one(
    damping,
    model_xml,
    bag,
    contact_replay,
):

    trace = []

    refs = {
        "model": None,
        "joints": None,
        "fixture": None,
        "mid_gid": None,
        "tip_gid": None,
    }

    original_step = mujoco.mj_step


    def capture_step(
        model,
        data,
        *args,
        **kwargs,
    ):

        out = original_step(
            model,
            data,
            *args,
            **kwargs,
        )

        local = e2a.replay_locals()

        if local is None:
            return out

        if refs["model"] is None:

            refs["model"] = model

            refs["joints"] = (
                e2a.joint_info(
                    model
                )
            )

            refs["fixture"] = (
                e2a.gid(
                    model,
                    e2a.FIXTURE_NAME,
                )
            )

            refs["mid_gid"] = (
                e2a.gid(
                    model,
                    "collision_idx_mid_mesh",
                )
            )

            refs["tip_gid"] = (
                e2a.gid(
                    model,
                    "collision_idx_tip",
                )
            )


        forces, active = (
            e2a.contact_forces(
                model,
                data,
                refs["fixture"],
            )
        )


        trace.append(
            {
                "time":
                    float(
                        local["time"]
                    ),

                "encoder":
                    float(
                        local["encoder"]
                    ),

                "q":
                    e2a.q_deg(
                        data,
                        refs["joints"],
                    ),

                "mid_force":
                    float(
                        forces[
                            "collision_idx_mid_mesh"
                        ]
                    ),

                "tip_force":
                    float(
                        forces[
                            "collision_idx_tip"
                        ]
                    ),

                "mid_contact":
                    bool(
                        active[
                            "collision_idx_mid_mesh"
                        ]
                    ),

                "tip_contact":
                    bool(
                        active[
                            "collision_idx_tip"
                        ]
                    ),

                "mid_gap":
                    e2a.physical_distance(
                        model,
                        data,
                        refs["mid_gid"],
                        refs["fixture"],
                    ),

                "tip_gap":
                    e2a.physical_distance(
                        model,
                        data,
                        refs["tip_gid"],
                        refs["fixture"],
                    ),
            }
        )

        return out


    contact_replay.CONTACT_DAMPING = (
        float(damping)
    )

    mujoco.mj_step = capture_step

    try:

        result = b2.run_candidate(
            model_xml,
            bag,
        )

    finally:

        mujoco.mj_step = (
            original_step
        )


    if not trace:

        raise RuntimeError(
            f"No trace for damping={damping}"
        )


    model = refs["model"]

    dt = float(
        model.opt.timestep
    )


    times = np.asarray(
        [
            row["time"]
            for row in trace
        ],
        dtype=float,
    )


    # ------------------------------------------------------------
    # Video target.
    # ------------------------------------------------------------

    target_time = float(
        contact_replay.TARGET_BAG_TIME
    )

    target_index = int(
        np.argmin(
            np.abs(
                times
                -
                target_time
            )
        )
    )

    target = trace[
        target_index
    ]


    # ------------------------------------------------------------
    # Entire trajectory topology.
    # ------------------------------------------------------------

    mid_flags = np.asarray(
        [
            row["mid_contact"]
            for row in trace
        ],
        dtype=bool,
    )

    tip_flags = np.asarray(
        [
            row["tip_contact"]
            for row in trace
        ],
        dtype=bool,
    )

    overlap_flags = (
        mid_flags
        &
        tip_flags
    )


    first_mid_indices = np.flatnonzero(
        mid_flags
    )

    first_tip_indices = np.flatnonzero(
        tip_flags
    )


    first_mid_encoder = (
        trace[
            int(
                first_mid_indices[0]
            )
        ]["encoder"]
        if first_mid_indices.size
        else np.nan
    )


    first_tip_encoder = (
        trace[
            int(
                first_tip_indices[0]
            )
        ]["encoder"]
        if first_tip_indices.size
        else np.nan
    )


    # ------------------------------------------------------------
    # Force-threshold onset, closer to hardware FTS onset.
    # ------------------------------------------------------------

    force_onset = next(
        (
            row
            for row in trace
            if (
                row["tip_force"]
                >=
                FORCE_THRESHOLD
            )
        ),
        None,
    )


    force_onset_encoder = (
        float(
            force_onset[
                "encoder"
            ]
        )
        if force_onset is not None
        else np.nan
    )


    # ------------------------------------------------------------
    # Hardware hold window.
    # ------------------------------------------------------------

    hold = [
        row
        for row in trace
        if (
            HW_HOLD_START
            <=
            row["time"]
            <=
            HW_HOLD_END
        )
    ]


    if hold:

        hold_tip_force = np.asarray(
            [
                row["tip_force"]
                for row in hold
            ],
            dtype=float,
        )

        hold_mid_flags = np.asarray(
            [
                row["mid_contact"]
                for row in hold
            ],
            dtype=bool,
        )

        hold_tip_flags = np.asarray(
            [
                row["tip_contact"]
                for row in hold
            ],
            dtype=bool,
        )

        hold_overlap = (
            hold_mid_flags
            &
            hold_tip_flags
        )

        positive_tip = (
            hold_tip_force[
                hold_tip_force > 0
            ]
        )


        positive_tip_median = (
            float(
                np.median(
                    positive_tip
                )
            )
            if positive_tip.size
            else np.nan
        )


        positive_tip_mean = (
            float(
                np.mean(
                    positive_tip
                )
            )
            if positive_tip.size
            else np.nan
        )


        max_tip_force = float(
            np.max(
                hold_tip_force
            )
        )


        hold_overlap_total = (
            float(
                np.sum(
                    hold_overlap
                )
            )
            *
            dt
        )


        hold_overlap_longest = (
            longest_true(
                hold_overlap,
                dt,
            )
        )


        hold_tip_transitions = (
            transitions(
                hold_tip_flags
            )
        )


        hold_mid_transitions = (
            transitions(
                hold_mid_flags
            )
        )

    else:

        positive_tip_median = np.nan
        positive_tip_mean = np.nan
        max_tip_force = np.nan

        hold_overlap_total = 0.0
        hold_overlap_longest = 0.0

        hold_tip_transitions = 0
        hold_mid_transitions = 0


    final = trace[-1]


    return {
        "damping":
            float(damping),

        "rmse":
            float(
                result["rmse"]
            ),

        "pose":
            np.asarray(
                result["pose"],
                dtype=float,
            ),

        "first_mid_encoder":
            float(
                first_mid_encoder
            ),

        "first_tip_encoder":
            float(
                first_tip_encoder
            ),

        "force_onset_encoder":
            float(
                force_onset_encoder
            ),

        "target_mid":
            bool(
                target["mid_contact"]
            ),

        "target_tip":
            bool(
                target["tip_contact"]
            ),

        "target_mid_gap_mm":
            1000.0
            *
            float(
                target["mid_gap"]
            ),

        "target_tip_gap_mm":
            1000.0
            *
            float(
                target["tip_gap"]
            ),

        "positive_tip_median":
            positive_tip_median,

        "positive_tip_mean":
            positive_tip_mean,

        "max_tip_force":
            max_tip_force,

        "force_error":
            (
                abs(
                    positive_tip_median
                    -
                    HW_FORCE
                )
                if np.isfinite(
                    positive_tip_median
                )
                else np.inf
            ),

        "overlap_total":
            hold_overlap_total,

        "overlap_longest":
            hold_overlap_longest,

        "tip_transitions":
            int(
                hold_tip_transitions
            ),

        "mid_transitions":
            int(
                hold_mid_transitions
            ),

        "final_mid_gap_mm":
            1000.0
            *
            float(
                final["mid_gap"]
            ),

        "final_tip_gap_mm":
            1000.0
            *
            float(
                final["tip_gap"]
            ),
    }


def fmt(
    value,
    width=8,
    precision=3,
):

    if not np.isfinite(
        value
    ):
        return (
            f"{'nan':>{width}s}"
        )

    return (
        f"{value:{width}.{precision}f}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    args = parser.parse_args()

    bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )

    if not bag.exists():

        raise RuntimeError(
            f"Missing bag:\n{bag}"
        )


    b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        b2.v30.v24.base.contact_replay
    )


    zero_margin_xml = (
        e2a.make_zero_margin_xml()
    )


    old_damping = float(
        contact_replay.CONTACT_DAMPING
    )

    old_mu = b2.v35.CONTACT_MU
    old_k = b2.v35.CONTACT_K


    b2.v35.CONTACT_MU = (
        CONTACT_MU
    )

    b2.v35.CONTACT_K = (
        CONTACT_K
    )


    rows = []


    try:

        print()
        print("=" * 150)
        print(
            "V35E2B — ZERO-MARGIN DYNAMIC CONTACT-DAMPING SCAN"
        )
        print("=" * 150)

        print()
        print(
            f"Model = "
            f"{zero_margin_xml}"
        )

        print()
        print(
            "FROZEN:"
        )

        print(
            "  contact margin = 0.000 mm"
        )

        print(
            f"  direct stiffness = "
            f"{CONTACT_K}"
        )

        print(
            f"  mu = "
            f"{CONTACT_MU}"
        )

        print(
            f"  hardware FTS target = "
            f"{HW_FORCE:.5f} N"
        )

        print()
        print(
            "VARIABLE ONLY:"
        )

        print(
            f"  damping = "
            f"{DAMPING_VALUES}"
        )

        print()


        for index, damping in enumerate(
            DAMPING_VALUES,
            start=1,
        ):

            print(
                f"[{index:02d}/"
                f"{len(DAMPING_VALUES):02d}] "
                f"damping={damping}"
            )

            row = run_one(
                damping,
                zero_margin_xml,
                bag,
                contact_replay,
            )

            rows.append(
                row
            )

            print(
                f"   RMSE="
                f"{row['rmse']:.3f} deg, "
                f"positive-tip-median="
                f"{row['positive_tip_median']:.4f} N, "
                f"longest-hold-overlap="
                f"{row['overlap_longest']:.3f} s"
            )


    finally:

        contact_replay.CONTACT_DAMPING = (
            old_damping
        )

        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_k
        )


    print()
    print("=" * 150)
    print(
        "V35E2B — COMPARISON TABLE"
    )
    print("=" * 150)

    print()
    print(
        " damp  RMSE   Ftip+med  Ferr   "
        "ovlpHold longest  midTr tipTr  "
        "E_mid  E_tip  E_F0.1   "
        "Tmid Ttip   finalMid finalTip"
    )

    print(
        "       [deg]    [N]     [N]     "
        "[s]      [s]                    "
        "[ticks] [ticks] [ticks]          "
        "        [mm]     [mm]"
    )

    print("-" * 150)


    for row in rows:

        print(
            f"{row['damping']:5.2f} "
            f"{row['rmse']:6.3f} "
            f"{fmt(row['positive_tip_median'],9,4)} "
            f"{fmt(row['force_error'],7,4)} "
            f"{row['overlap_total']:8.3f} "
            f"{row['overlap_longest']:7.3f} "
            f"{row['mid_transitions']:6d} "
            f"{row['tip_transitions']:5d} "
            f"{fmt(row['first_mid_encoder'],7,1)} "
            f"{fmt(row['first_tip_encoder'],7,1)} "
            f"{fmt(row['force_onset_encoder'],8,1)} "
            f"{str(row['target_mid']):>5s} "
            f"{str(row['target_tip']):>5s} "
            f"{row['final_mid_gap_mm']:+8.3f} "
            f"{row['final_tip_gap_mm']:+8.3f}"
        )


    print()
    print("=" * 150)
    print(
        "V35E2B — FORCE DETAILS"
    )
    print("=" * 150)

    print()

    for row in rows:

        print(
            f"d={row['damping']:5.2f}: "
            f"positive-contact tip "
            f"median={row['positive_tip_median']:.5f} N, "
            f"mean={row['positive_tip_mean']:.5f} N, "
            f"max={row['max_tip_force']:.5f} N"
        )


    print()
    print("=" * 150)
    print(
        "INTERPRETATION RULE"
    )
    print("=" * 150)

    print()
    print(
        "A useful damping candidate must improve several "
        "quantities together:"
    )

    print(
        "  1. tip force toward ~0.1456 N"
    )

    print(
        "  2. middle contact before distal contact"
    )

    print(
        "  3. larger sustained middle+tip overlap"
    )

    print(
        "  4. fewer contact transitions / less bounce"
    )

    print(
        "  5. target-time physical contact if supported "
        "by synchronized hardware video"
    )

    print(
        "  6. acceptable MCP/PIP/DIP pose RMSE"
    )

    print()
    print(
        "If no damping value achieves these together, "
        "close contact-damping tuning and move to the "
        "tendon/passive load-distribution model."
    )

    print()
    print("=" * 150)
    print(
        "V35E2B COMPLETE"
    )
    print("=" * 150)


if __name__ == "__main__":
    main()
