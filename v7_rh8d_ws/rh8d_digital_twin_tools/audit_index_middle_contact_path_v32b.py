#!/usr/bin/env python3

from pathlib import Path
import argparse

import mujoco
import numpy as np

import scan_index_fixed_support_v30 as v30


# ============================================================
# V32B — ACTUAL MIDDLE-PHALANX CONTACT-PATH AUDIT
#
# Diagnostic only.
#
# V32 failed because none of the proposed skin elements ever
# contacted the bottle.
#
# This script records the REAL contact points from the trusted
# middle collision capsule and expresses them in Index_Middle
# local coordinates.
#
# No parameters change.
# No GUI: same V30 motion already inspected visually.
# ============================================================


CASE = {
    "name": "V30_SELECTED",
    "yield": True,
    "arms_mm": [4.0, 8.0, 1.5],
}


WS = v30.WS
MJCF_DIR = v30.v24.MJCF_DIR

MODEL_PATH = (
    MJCF_DIR
    / "tmp_v32b_contact_path.xml"
)


MID_BODY = "Index_Middle"
MID_GEOM = "collision_idx_mid"
BOTTLE = "index_contact_fixture"


# Original V32 proposed coverage.
V32_HALF_WIDTH_Y = 0.0055

V32_HALF_LENGTH_Z = 0.0038

V32_PATCH_Z = [
    -0.0082,
    -0.01294,
    -0.0177,
]


# ============================================================
# RECORDER
# ============================================================

class ContactPathRecorder:

    def __init__(self):

        self.original_step = mujoco.mj_step

        self.samples = []

        self.first_time = None

        self.cache = {}


    def ids(self, model):

        key = id(model)

        if key in self.cache:

            return self.cache[key]


        body = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            MID_BODY,
        )


        mid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            MID_GEOM,
        )


        bottle = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            BOTTLE,
        )


        if body < 0:
            raise RuntimeError(
                f"Missing body {MID_BODY}"
            )


        if mid < 0:
            raise RuntimeError(
                f"Missing geom {MID_GEOM}"
            )


        if bottle < 0:
            raise RuntimeError(
                f"Missing geom {BOTTLE}"
            )


        result = {
            "body": body,
            "mid": mid,
            "bottle": bottle,
        }


        self.cache[key] = result

        return result


    def record(self, model, data):

        ids = self.ids(model)


        if self.first_time is None:

            self.first_time = float(
                data.time
            )


        body_pos = np.asarray(
            data.xpos[
                ids["body"]
            ],
            dtype=float,
        )


        body_R = np.asarray(
            data.xmat[
                ids["body"]
            ],
            dtype=float,
        ).reshape(
            3,
            3,
        )


        for i in range(data.ncon):

            c = data.contact[i]


            pair = {
                int(c.geom1),
                int(c.geom2),
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


            Fn = abs(
                float(force[0])
            )


            if Fn <= 1.0e-4:

                continue


            world_point = np.asarray(
                c.pos,
                dtype=float,
            )


            # World -> Index_Middle local coordinates.
            local_point = (
                body_R.T
                @
                (
                    world_point
                    -
                    body_pos
                )
            )


            self.samples.append(
                {
                    "time":
                        float(data.time),

                    "elapsed":
                        float(
                            data.time
                            -
                            self.first_time
                        ),

                    "Fn":
                        Fn,

                    "x":
                        float(local_point[0]),

                    "y":
                        float(local_point[1]),

                    "z":
                        float(local_point[2]),
                }
            )


    def install(self):

        original = self.original_step

        recorder = self


        def wrapped(
            model,
            data,
            *args,
            **kwargs,
        ):

            result = original(
                model,
                data,
                *args,
                **kwargs,
            )


            recorder.record(
                model,
                data,
            )


            return result


        mujoco.mj_step = wrapped


    def restore(self):

        mujoco.mj_step = (
            self.original_step
        )


# ============================================================
# STATISTICS
# ============================================================

def print_stats(
    name,
    values,
    scale=1000.0,
    unit="mm",
):

    values = np.asarray(
        values,
        dtype=float,
    )


    print(
        f"{name}:"
    )

    print(
        f"  min    = "
        f"{scale*np.min(values):+.3f} {unit}"
    )

    print(
        f"  p05    = "
        f"{scale*np.percentile(values,5):+.3f} {unit}"
    )

    print(
        f"  median = "
        f"{scale*np.percentile(values,50):+.3f} {unit}"
    )

    print(
        f"  p95    = "
        f"{scale*np.percentile(values,95):+.3f} {unit}"
    )

    print(
        f"  max    = "
        f"{scale*np.max(values):+.3f} {unit}"
    )


def covered_by_old_v32(
    sample,
):

    # y must lie inside the proposed rectangular skin.
    if (
        abs(sample["y"])
        >
        V32_HALF_WIDTH_Y
    ):

        return False


    # z must lie inside at least one proposed patch.
    for center_z in V32_PATCH_Z:

        if (
            abs(
                sample["z"]
                -
                center_z
            )
            <=
            V32_HALF_LENGTH_Z
        ):

            return True


    return False


# ============================================================
# MAIN
# ============================================================

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
            f"Missing contact bag:\n"
            f"{contact_bag}"
        )


    v30.v24.ab.patch_replay_modules()


    v30.v24.build_model(
        MODEL_PATH
    )


    # --------------------------------------------------------
    # Correct V30 stateful tendon.
    # --------------------------------------------------------

    law = (
        v30.v24.StatefulTransmission()
    )

    law.install()


    # --------------------------------------------------------
    # Correct V30 fixed-support boundary condition.
    # --------------------------------------------------------

    clamp = (
        v30.FixedSupport()
    )

    clamp.install()


    # --------------------------------------------------------
    # Contact-path recorder.
    # --------------------------------------------------------

    recorder = (
        ContactPathRecorder()
    )

    recorder.install()


    old_scale = (
        v30.v24.base.METERS_PER_TICK
    )

    old_tendon = (
        v30.v24.base.TENDON_STIFFNESS
    )

    old_mu = (
        v30.v24.base.CONTACT_MU
    )

    old_contact_k = (
        v30.v24.base.CONTACT_STIFFNESS
    )


    try:

        v30.v24.base.METERS_PER_TICK = (
            v30.v24.SCALE
        )

        v30.v24.base.TENDON_STIFFNESS = (
            v30.v24.TENDON_K
        )

        v30.v24.base.CONTACT_MU = 0.05

        v30.v24.base.CONTACT_STIFFNESS = (
            v30.v24.CONTACT_K
        )


        law.configure(
            CASE,
            "CONTACT",
        )


        metrics, _ = (
            v30.v24.base.run_contact(
                MODEL_PATH,
                contact_bag,
            )
        )


    finally:

        recorder.restore()

        clamp.restore()

        law.restore()


        v30.v24.base.METERS_PER_TICK = (
            old_scale
        )

        v30.v24.base.TENDON_STIFFNESS = (
            old_tendon
        )

        v30.v24.base.CONTACT_MU = (
            old_mu
        )

        v30.v24.base.CONTACT_STIFFNESS = (
            old_contact_k
        )


    samples = recorder.samples


    print()
    print("=" * 116)
    print(
        "V32B — ACTUAL MIDDLE CONTACT PATH"
    )
    print("=" * 116)


    print()
    print(
        "Frozen corrected model:"
    )

    print(
        "  support = FIXED"
    )

    print(
        "  post-yield transmission = "
        "4 / 8 / 1.5 mm"
    )

    print(
        "  mu = 0.05"
    )

    print(
        "  bottle = V5 frozen"
    )


    print()
    print(
        "Replay final pose:"
    )

    print(
        "  "
        f"{metrics['pose'][0]:.3f} / "
        f"{metrics['pose'][1]:.3f} / "
        f"{metrics['pose'][2]:.3f} deg"
    )


    if not samples:

        raise RuntimeError(
            "No middle/bottle contacts were captured."
        )


    print()
    print(
        f"Captured active contact samples = "
        f"{len(samples)}"
    )


    # ========================================================
    # CONTACT PATH DISTRIBUTION
    # ========================================================

    print()
    print("=" * 116)
    print(
        "LOCAL CONTACT-POINT DISTRIBUTION "
        "IN Index_Middle FRAME"
    )
    print("=" * 116)


    print()

    print_stats(
        "local x",
        [
            s["x"]
            for s in samples
        ],
    )


    print()

    print_stats(
        "local y",
        [
            s["y"]
            for s in samples
        ],
    )


    print()

    print_stats(
        "local z",
        [
            s["z"]
            for s in samples
        ],
    )


    # ========================================================
    # IMPORTANT CONTACT STATES
    # ========================================================

    first = samples[0]

    peak = max(
        samples,
        key=lambda s:
            s["Fn"],
    )

    last = samples[-1]


    print()
    print("=" * 116)
    print(
        "KEY MIDDLE-CONTACT POINTS"
    )
    print("=" * 116)


    for title, sample in [
        (
            "FIRST",
            first,
        ),
        (
            "PEAK NORMAL FORCE",
            peak,
        ),
        (
            "LAST",
            last,
        ),
    ]:

        print()
        print(
            title
        )

        print(
            f"  elapsed = "
            f"{sample['elapsed']:.3f} s"
        )

        print(
            f"  Fn = "
            f"{sample['Fn']:.3f} N"
        )

        print(
            "  local xyz = "
            f"["
            f"{1000*sample['x']:+.3f}, "
            f"{1000*sample['y']:+.3f}, "
            f"{1000*sample['z']:+.3f}"
            f"] mm"
        )


    # ========================================================
    # WHY V32 PATCHES MISSED
    # ========================================================

    covered = np.asarray(
        [
            covered_by_old_v32(
                s
            )
            for s in samples
        ],
        dtype=bool,
    )


    coverage = (
        100.0
        *
        float(
            np.mean(
                covered
            )
        )
    )


    max_abs_y = max(
        abs(
            s["y"]
        )
        for s in samples
    )


    min_z = min(
        s["z"]
        for s in samples
    )

    max_z = max(
        s["z"]
        for s in samples
    )


    print()
    print("=" * 116)
    print(
        "V32 PATCH-COVERAGE AUDIT"
    )
    print("=" * 116)


    print()
    print(
        "Old V32 proposed coverage:"
    )

    print(
        f"  y = +/-"
        f"{1000*V32_HALF_WIDTH_Y:.3f} mm"
    )

    print(
        "  z centers = "
        f"{[1000*z for z in V32_PATCH_Z]} mm"
    )

    print(
        f"  each z half-length = "
        f"{1000*V32_HALF_LENGTH_Z:.3f} mm"
    )


    print()
    print(
        f"Actual contact samples falling inside "
        f"old V32 y/z coverage = "
        f"{coverage:.1f}%"
    )


    print()
    print(
        "Observed contact-path requirements:"
    )

    print(
        f"  maximum |local y| = "
        f"{1000*max_abs_y:.3f} mm"
    )

    print(
        f"  local z range = "
        f"[{1000*min_z:+.3f}, "
        f"{1000*max_z:+.3f}] mm"
    )


    # Add 1 mm safety margin.
    recommended_half_width = (
        max_abs_y
        +
        0.001
    )


    recommended_z_min = (
        min_z
        -
        0.001
    )

    recommended_z_max = (
        max_z
        +
        0.001
    )


    print()
    print(
        "Recommended minimum skin coverage "
        "(contact path + 1 mm margin):"
    )

    print(
        f"  half-width y >= "
        f"{1000*recommended_half_width:.3f} mm"
    )

    print(
        f"  z from "
        f"{1000*recommended_z_min:+.3f} "
        f"to "
        f"{1000*recommended_z_max:+.3f} mm"
    )


    # ========================================================
    # DECISION
    # ========================================================

    print()
    print("=" * 116)
    print(
        "V32B DECISION"
    )
    print("=" * 116)


    if coverage < 25.0:

        print()
        print(
            "V32 FAILED PRIMARILY BECAUSE THE PROPOSED "
            "SKIN PATCHES DID NOT COVER THE REAL CONTACT PATH."
        )

        print()
        print(
            "This is a geometry-placement failure, "
            "not evidence against distributed compliance."
        )

        print()
        print(
            "NEXT:"
        )

        print(
            "Rebuild the three compliant elements using "
            "the measured local contact-path coordinates "
            "above, then test only 2-3 stiffness values."
        )


    elif coverage < 80.0:

        print()
        print(
            "V32 COVERED ONLY PART OF THE ACTUAL "
            "MIDDLE-PHALANX CONTACT PATH."
        )

        print()
        print(
            "NEXT:"
        )

        print(
            "Expand/reposition the distributed skin to "
            "cover the measured path before interpreting "
            "compliance."
        )


    else:

        print()
        print(
            "V32 y/z placement already covered most of "
            "the baseline contact path."
        )

        print()
        print(
            "Then the missing contact indicates that the "
            "skin's radial placement/orientation is wrong."
        )

        print(
            "NEXT: use the measured local contact x "
            "distribution to place curved/radial elements."
        )


    try:
        MODEL_PATH.unlink()
    except FileNotFoundError:
        pass


    print()
    print("=" * 116)
    print(
        "V32B COMPLETE"
    )
    print("=" * 116)


if __name__ == "__main__":
    main()
