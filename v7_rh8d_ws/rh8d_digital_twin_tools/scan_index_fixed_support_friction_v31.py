#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv

import mujoco
import numpy as np

import scan_index_fixed_support_v30 as v30


# ============================================================
# V31 — FIXED-SUPPORT CONTACT FRICTION RETEST
#
# WHY REVISIT FRICTION?
#
# Old V19 friction test used:
#   - moving/recoiling UR5e support
#   - old constant transmission
#   - old PIP passive behavior
#
# V29/V30 corrected those confounders.
#
# Current corrected V30 model:
#
#   fixed UR5e/palm support
#   breakaway threshold = 0.015 Nm
#   post-yield transmission = 4 / 8 / 1.5 mm
#
# V31 changes ONLY runtime contact friction mu.
#
# Bottle position unchanged.
# Contact stiffness unchanged.
# Collision geometry unchanged.
# No pad.
# No mesh replacement.
# ============================================================


WS = v30.WS

RESULT_DIR = (
    WS
    / "rh8d_digital_twin_tools/results/"
      "fixed_support_friction_v31"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY = RESULT_DIR / "summary.csv"


CASE = {

    "name":
        "V30_SELECTED",

    "yield":
        True,

    "arms_mm":
        [4.0, 8.0, 1.5],
}


MU_VALUES = [
    0.05,
    0.10,
    0.20,
    0.40,
    0.80,
    1.20,
]


MIN_OVERLAP_S = 0.15


# ============================================================
# ACTUAL RUNTIME FRICTION VALIDATION
#
# V18 taught us not to trust requested values without checking
# the compiled/runtime contact.
#
# v30.run_case() creates v30.Recorder internally.
# We temporarily replace it with this subclass.
# ============================================================

OriginalRecorder = v30.Recorder


class MuRecorder(OriginalRecorder):

    last_mu_values = []


    def __init__(self):

        super().__init__()

        self.mu_values = []


    def record(
        self,
        model,
        data,
    ):

        ids = self.ids(
            model
        )


        if (
            ids["mid"] >= 0
            and
            ids["bottle"] >= 0
        ):

            for i in range(
                data.ncon
            ):

                contact = data.contact[i]


                pair = {
                    int(contact.geom1),
                    int(contact.geom2),
                }


                if pair != {
                    ids["mid"],
                    ids["bottle"],
                }:

                    continue


                force = np.zeros(
                    6,
                    dtype=float,
                )


                mujoco.mj_contactForce(
                    model,
                    data,
                    i,
                    force,
                )


                if (
                    abs(
                        float(
                            force[0]
                        )
                    )
                    <= 1.0e-4
                ):

                    continue


                try:

                    self.mu_values.append(
                        float(
                            contact.friction[0]
                        )
                    )

                except Exception:

                    pass


        super().record(
            model,
            data,
        )


    def restore(self):

        MuRecorder.last_mu_values = list(
            self.mu_values
        )

        super().restore()


# ============================================================
# RUN ONE MU
# ============================================================

def run_mu(
    mu,
    contact_bag,
):

    old_mu = float(
        v30.v24.CONTACT_MU
    )


    old_recorder = (
        v30.Recorder
    )


    MuRecorder.last_mu_values = []


    try:

        # v30.run_case() assigns:
        #
        # base.CONTACT_MU = v24.CONTACT_MU
        #
        # therefore this is the same validated runtime-global
        # route used after the V18 bug was discovered.
        v30.v24.CONTACT_MU = (
            float(mu)
        )


        v30.Recorder = (
            MuRecorder
        )


        result = v30.run_case(
            CASE,
            contact_bag,
        )


    finally:

        v30.Recorder = (
            old_recorder
        )

        v30.v24.CONTACT_MU = (
            old_mu
        )


    values = np.asarray(
        MuRecorder.last_mu_values,
        dtype=float,
    )


    if len(values) == 0:

        effective_mu = float(
            "nan"
        )

    else:

        effective_mu = float(
            np.median(
                values
            )
        )


    # ========================================================
    # HARD VALIDATION
    # ========================================================

    if (
        not np.isfinite(
            effective_mu
        )
        or
        abs(
            effective_mu
            -
            mu
        )
        > 0.005
    ):

        raise RuntimeError(
            "\n"
            "V31 RUNTIME FRICTION VALIDATION FAILED\n"
            f"  requested = {mu:.3f}\n"
            f"  effective = {effective_mu}\n"
            "\n"
            "Do not interpret this candidate."
        )


    result[
        "requested_mu"
    ] = float(
        mu
    )

    result[
        "effective_mu"
    ] = effective_mu


    return result


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


    v30.v24.ab.patch_replay_modules()


    print()
    print("=" * 122)
    print(
        "RH8D INDEX — FIXED-SUPPORT "
        "CONTACT FRICTION RETEST V31"
    )
    print("=" * 122)


    print()
    print(
        "FROZEN:"
    )

    print(
        "  UR5e/palm support = FIXED"
    )

    print(
        "  bottle position = V5"
    )

    print(
        "  transmission = 4 / 8 / 1.5 mm after yield"
    )

    print(
        "  breakaway threshold = 0.015 Nm"
    )

    print(
        "  contact stiffness = 1000 N/m"
    )

    print(
        "  collision geometry = baseline capsules"
    )

    print()
    print(
        "ONLY runtime friction mu changes."
    )

    print(
        MU_VALUES
    )


    results = []


    for index, mu in enumerate(
        MU_VALUES,
        start=1,
    ):

        print()
        print("=" * 122)

        print(
            f"[{index:02d}/{len(MU_VALUES):02d}] "
            f"mu = {mu:.3f}"
        )

        print("=" * 122)


        result = run_mu(
            mu,
            contact_bag,
        )


        results.append(
            result
        )


        print()
        print(
            "RUNTIME VALIDATION:"
        )

        print(
            f"  requested mu = "
            f"{result['requested_mu']:.3f}"
        )

        print(
            f"  effective mu = "
            f"{result['effective_mu']:.3f}"
        )


        print()
        print(
            "Pose MCP/PIP/DIP = "
            f"{result['mcp_deg']:.3f} / "
            f"{result['pip_deg']:.3f} / "
            f"{result['dip_deg']:.3f} deg"
        )

        print(
            f"RMSE = "
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
            f"  overlap total = "
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

        print(
            f"  support drift = "
            f"{result['max_support_drift_mm']:.6f} mm"
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


    baseline = min(
        results,
        key=lambda r:
            abs(
                r[
                    "requested_mu"
                ]
                -
                0.05
            ),
    )


    retained = [
        r
        for r in results
        if (
            r[
                "longest_overlap_s"
            ]
            >= MIN_OVERLAP_S
        )
    ]


    # ========================================================
    # SELECTION
    #
    # Contact topology now has priority.
    #
    # Once candidates retain middle+tip contact, choose the
    # lowest joint-angle RMSE among them.
    # ========================================================

    if retained:

        selected = min(
            retained,
            key=lambda r:
                r[
                    "rmse_deg"
                ],
        )

        mode = (
            "PERSISTENT MIDDLE+TIP WRAP"
        )


    else:

        selected = max(
            results,
            key=lambda r:
                (
                    r[
                        "longest_overlap_s"
                    ],
                    -r[
                        "rmse_deg"
                    ],
                ),
        )

        mode = (
            "NO MU PASSED RETENTION"
        )


    print()
    print("=" * 122)
    print(
        "V31 SELECTED RESULT"
    )
    print("=" * 122)


    print()
    print(
        f"Selection mode = "
        f"{mode}"
    )

    print(
        f"Requested/effective mu = "
        f"{selected['requested_mu']:.3f} / "
        f"{selected['effective_mu']:.3f}"
    )


    print()
    print(
        "Hardware target:"
    )

    print(
        "  14.536 / 23.811 / 9.327 deg"
    )


    print(
        "Prediction:"
    )

    print(
        "  "
        f"{selected['mcp_deg']:.3f} / "
        f"{selected['pip_deg']:.3f} / "
        f"{selected['dip_deg']:.3f} deg"
    )


    print(
        f"RMSE = "
        f"{selected['rmse_deg']:.3f} deg"
    )


    print()
    print(
        "Longest middle+tip overlap:"
    )

    print(
        f"  mu=0.05 control = "
        f"{baseline['longest_overlap_s']:.3f} s"
    )

    print(
        f"  selected = "
        f"{selected['longest_overlap_s']:.3f} s"
    )


    print(
        f"Middle after first tip = "
        f"{selected['middle_after_tip_s']:.3f} s"
    )


    print(
        f"Support drift = "
        f"{selected['max_support_drift_mm']:.6f} mm"
    )


    # ========================================================
    # DECISION
    # ========================================================

    print()
    print("=" * 122)
    print(
        "V31 DECISION"
    )
    print("=" * 122)


    improvement = (
        selected[
            "longest_overlap_s"
        ]
        -
        baseline[
            "longest_overlap_s"
        ]
    )


    if (
        selected[
            "longest_overlap_s"
        ]
        >= MIN_OVERLAP_S
    ):

        print()
        print(
            "FRICTION IS A KEY PART OF THE "
            "MISSING CONTACT-RETENTION MECHANISM."
        )

        print()
        print(
            "The corrected fixed-support architecture can "
            "maintain simultaneous middle/distal contact "
            "when the interface friction is increased."
        )


        if (
            selected[
                "rmse_deg"
            ]
            <= 3.0
        ):

            print()
            print(
                "MAJOR RESULT:"
            )

            print(
                "Both contact topology and joint-angle "
                "agreement are now reasonable."
            )

            print()
            print(
                "Do NOT add artificial contact structures."
            )

            print(
                "Next: narrow material/friction identification "
                "and held-out hardware validation."
            )


        else:

            print()
            print(
                "Wrap topology is improved, but transmission "
                "must be rebalanced slightly under the new "
                "friction value."
            )


    elif (
        improvement
        >= 0.05
    ):

        print()
        print(
            "FRICTION HELPS BUT IS NOT SUFFICIENT."
        )

        print()
        print(
            "Freeze the best physically plausible mu."
        )

        print(
            "Then add the smallest distributed-compliance "
            "model needed to obtain persistent wrapping."
        )


    else:

        print()
        print(
            "VERIFIED FRICTION CHANGE DOES NOT RECOVER "
            "PERSISTENT WRAPPING."
        )

        print()
        print(
            "This closes the friction hypothesis under the "
            "corrected fixed-support architecture."
        )

        print()
        print(
            "NEXT: distributed compliant surface mechanics."
        )

        print(
            "Do not revisit bottle pose, tendon ratio, "
            "support motion, or friction again."
        )


    print()
    print(
        "Summary:"
    )

    print(
        SUMMARY
    )


    # ========================================================
    # GUI — ONLY SELECTED MU
    # ========================================================

    if not args.no_viewer:

        print()
        print("=" * 122)
        print(
            "V31 AUTOMATIC SELECTED-MU GUI"
        )
        print("=" * 122)


        old_mu = float(
            v30.v24.CONTACT_MU
        )


        try:

            v30.v24.CONTACT_MU = float(
                selected[
                    "requested_mu"
                ]
            )


            print()
            print(
                f"Viewing mu = "
                f"{selected['requested_mu']:.3f}"
            )

            print(
                "Support = FIXED"
            )

            print(
                "Transmission = 4 / 8 / 1.5 mm"
            )

            print(
                "Bottle position = frozen V5"
            )

            print(
                "Index visual alpha = 1.0"
            )


            v30.visual_fixed(
                CASE,
                contact_bag,
            )


        finally:

            v30.v24.CONTACT_MU = (
                old_mu
            )


    print()
    print("=" * 122)
    print(
        "V31 COMPLETE"
    )
    print("=" * 122)


if __name__ == "__main__":
    main()
