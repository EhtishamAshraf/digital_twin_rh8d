#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import scan_index_contact_retention_v35c1 as c1
import fit_index_passive_contact_v35b2 as b2


# ============================================================
# V35C2-v2
#
# Scan ONLY MuJoCo direct-format contact damping.
#
# FROZEN:
#   CAD tendon
#   V35B2 passives
#   V33/H1 environment
#   bottle position
#   mu = 0.05
#   contact stiffness = 1000 N/m
#
# VARIABLE:
#   CONTACT_DAMPING only
#
# No contact-triggered rules.
# No prescribed joint angles.
# No transmission switching.
# ============================================================


FIXED_MU = 0.05
FIXED_K = 1000.0

DAMPING_VALUES = [
    0.25,
    0.50,
    1.00,   # existing baseline
    2.00,
    5.00,
    10.00,
    20.00,
]

BASELINE_RMSE = 2.323
RMSE_GATE = BASELINE_RMSE + 1.0


def unique_modules():

    modules = []

    # The actual MuJoCo contact law is one level below the
    # scan_index_pip_passive_dynamic_v2 wrapper:
    #
    #   scan_index_pip_passive_dynamic_v2.contact_replay
    #       -> replay_index_tabletop_hardware_contact_v1
    #
    # CONTACT_DAMPING lives there.

    for parent in (
        b2.v30.v24.base,
        b2.v30.v24.viewer_tools.base,
    ):

        module = getattr(
            parent,
            "contact_replay",
            None,
        )

        if module is None:

            raise RuntimeError(
                f"{parent.__name__} has no contact_replay module"
            )

        if not any(
            id(module) == id(x)
            for x in modules
        ):
            modules.append(module)

    return modules


def validate_contact_damping():

    modules = unique_modules()

    print()
    print("=" * 110)
    print("RUNTIME CONTACT-DAMPING IMPLEMENTATION")
    print("=" * 110)

    valid_any = False

    for module in modules:

        print()
        print("module :", module.__name__)
        print("file   :", getattr(module, "__file__", None))

        if hasattr(
            module,
            "CONTACT_DAMPING",
        ):

            print(
                "CONTACT_DAMPING =",
                module.CONTACT_DAMPING,
            )

        else:

            print(
                "CONTACT_DAMPING = NOT FOUND"
            )

            continue

        writers = []

        for name, fn in inspect.getmembers(
            module,
            inspect.isfunction,
        ):

            try:
                source = inspect.getsource(fn)
            except Exception:
                continue

            if (
                "geom_solref" in source
                and
                "CONTACT_DAMPING" in source
            ):

                writers.append(name)

        if writers:

            valid_any = True

            print(
                "functions using CONTACT_DAMPING "
                "to modify geom_solref:"
            )

            for name in writers:
                print("  ", name)

        else:

            print(
                "WARNING: no function source contains both "
                "geom_solref and CONTACT_DAMPING"
            )

    if not valid_any:

        raise RuntimeError(
            "\nCould not prove that the currently imported "
            "runtime contact implementation uses "
            "CONTACT_DAMPING.\n"
            "STOPPED before running simulations."
        )

    return modules


def set_damping(
    modules,
    damping,
):

    for module in modules:

        if hasattr(
            module,
            "CONTACT_DAMPING",
        ):

            module.CONTACT_DAMPING = float(
                damping
            )


def run_case(
    contact_bag,
    damping,
    modules,
):

    set_damping(
        modules,
        damping,
    )

    result = c1.run_one(
        contact_bag,
        FIXED_MU,
        FIXED_K,
    )

    result["damping"] = float(
        damping
    )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "contact_bag",
        type=Path,
    )

    args = parser.parse_args()

    contact_bag = (
        args.contact_bag
        .expanduser()
        .resolve()
    )

    if not contact_bag.exists():

        raise RuntimeError(
            f"Missing contact bag:\n{contact_bag}"
        )

    if not b2.SELECTED_MODEL.exists():

        raise RuntimeError(
            "Missing V35B2 selected model:\n"
            f"{b2.SELECTED_MODEL}"
        )

    # Same initialization used throughout V33/V35.
    b2.v30.v24.ab.patch_replay_modules()

    modules = validate_contact_damping()

    # Save globals.
    old_damping = {
        id(module):
            getattr(
                module,
                "CONTACT_DAMPING",
                None,
            )
        for module in modules
    }

    old_mu = b2.v35.CONTACT_MU
    old_k = b2.v35.CONTACT_K

    b2.v35.CONTACT_MU = FIXED_MU
    b2.v35.CONTACT_K = FIXED_K

    rows = []

    try:

        print()
        print("=" * 120)
        print(
            "V35C2-v2 — DIRECT CONTACT DAMPING SCAN"
        )
        print("=" * 120)

        print()
        print("FROZEN:")
        print("  CAD tendon")
        print("  V35B2 passive mechanics")
        print("  H1/V33 environment")
        print("  steel-cylinder geometry")
        print(f"  mu = {FIXED_MU}")
        print(f"  contact stiffness = {FIXED_K:.1f} N/m")

        print()
        print(
            "VARIABLE ONLY:"
        )

        print(
            "  CONTACT_DAMPING =",
            DAMPING_VALUES,
        )

        for i, damping in enumerate(
            DAMPING_VALUES,
            start=1,
        ):

            print()
            print("-" * 120)

            print(
                f"[{i:02d}/{len(DAMPING_VALUES):02d}] "
                f"CONTACT_DAMPING = {damping:.3f}"
            )

            print("-" * 120)

            result = run_case(
                contact_bag,
                damping,
                modules,
            )

            rows.append(
                result
            )

            q = result["pose"]

            print(
                "  q = "
                f"{q[0]:.3f} / "
                f"{q[1]:.3f} / "
                f"{q[2]:.3f} deg"
            )

            print(
                f"  RMSE = "
                f"{result['rmse']:.3f} deg"
            )

            print(
                "  contacts:"
            )

            print(
                f"    middle total       = "
                f"{result['middle']:.3f} s"
            )

            print(
                f"    tip total          = "
                f"{result['tip']:.3f} s"
            )

            print(
                f"    overlap total      = "
                f"{result['overlap']:.3f} s"
            )

            print(
                f"    LONGEST overlap    = "
                f"{result['longest_overlap']:.3f} s"
            )

            print(
                f"    middle after tip   = "
                f"{result['after_tip']:.3f} s"
            )

        # ----------------------------------------------------
        # Preserve reasonable joint calibration first.
        # ----------------------------------------------------

        admissible = [
            r
            for r in rows
            if r["rmse"] <= RMSE_GATE
        ]

        if not admissible:

            print()
            print(
                "WARNING: no damping candidate remained "
                "inside the +1 degree RMSE guard."
            )

            admissible = rows

        # ----------------------------------------------------
        # Primary objective:
        # sustained physical contact, not final angle.
        # ----------------------------------------------------

        admissible.sort(
            key=lambda r:
                (
                    -r["longest_overlap"],
                    -r["after_tip"],
                    r["rmse"],
                )
        )

        selected = admissible[0]

        ranked = sorted(
            rows,
            key=lambda r:
                (
                    -r["longest_overlap"],
                    -r["after_tip"],
                    r["rmse"],
                )
        )

        print()
        print("=" * 120)
        print(
            "V35C2-v2 RESULTS — SUSTAINED CONTACT RANKING"
        )
        print("=" * 120)

        for rank, r in enumerate(
            ranked,
            start=1,
        ):

            q = r["pose"]

            gate = (
                "PASS"
                if r["rmse"] <= RMSE_GATE
                else "FAIL"
            )

            print(
                f"{rank:02d}. "
                f"damping={r['damping']:6.2f} "
                f"RMSE={r['rmse']:6.3f} "
                f"gate={gate} "
                f"q="
                f"{q[0]:6.2f}/"
                f"{q[1]:6.2f}/"
                f"{q[2]:6.2f} "
                f"middle={r['middle']:.3f}s "
                f"tip={r['tip']:.3f}s "
                f"overlap={r['overlap']:.3f}s "
                f"LONGEST={r['longest_overlap']:.3f}s "
                f"after_tip={r['after_tip']:.3f}s"
            )

        print()
        print("=" * 120)
        print(
            "V35C2-v2 SELECTED"
        )
        print("=" * 120)

        print(
            f"CONTACT_DAMPING = "
            f"{selected['damping']:.3f}"
        )

        print(
            f"CONTACT_STIFFNESS = "
            f"{FIXED_K:.1f} N/m"
        )

        print(
            f"mu = "
            f"{FIXED_MU:.3f}"
        )

        print(
            f"RMSE = "
            f"{selected['rmse']:.3f} deg"
        )

        print(
            f"longest simultaneous contact = "
            f"{selected['longest_overlap']:.3f} s"
        )

        print(
            f"middle contact after first tip = "
            f"{selected['after_tip']:.3f} s"
        )

        # ----------------------------------------------------
        # GUI selected candidate only.
        # ----------------------------------------------------

        set_damping(
            modules,
            selected["damping"],
        )

        print()
        print("=" * 120)
        print(
            "V35C2-v2 SELECTED GUI"
        )
        print("=" * 120)

        print()
        print(
            "Watch the actual motion carefully."
        )

        print()
        print(
            "A = touch -> bounce -> touch -> bounce "
            "-> final gap"
        )

        print(
            "B = touch -> smaller bounce -> final gap"
        )

        print(
            "C = touch -> settles -> remains touching "
            "without full wrap"
        )

        print(
            "D = touch -> settles -> continued tendon pull "
            "naturally wraps finger"
        )

        b2.visual_replay(
            b2.SELECTED_MODEL,
            contact_bag,
        )

    finally:

        for module in modules:

            value = old_damping[
                id(module)
            ]

            if value is not None:

                module.CONTACT_DAMPING = value

        b2.v35.CONTACT_MU = old_mu
        b2.v35.CONTACT_K = old_k

    print()
    print("=" * 120)
    print(
        "V35C2-v2 COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
