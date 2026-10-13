#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import fit_index_passive_contact_v35b2 as b2


# ============================================================
# V35D1
#
# STATIC GEOMETRY FEASIBILITY AUDIT
#
# IMPORTANT:
#
# The hardware MCP/PIP/DIP angles are NOT used during the
# dynamic replay.
#
# First:
#   run the normal identified V35B2 model with damping=20
#
# Then:
#   copy the resulting complete robot/environment state
#   and replace ONLY MCP/PIP/DIP with the measured H1 angles.
#
# Purpose:
#   determine whether current collision geometry can physically
#   support the measured hardware wrap configuration.
#
# This is a DIAGNOSTIC, not system identification.
# ============================================================


HW_DEG = np.array(
    [
        14.5355,
        23.8110,
        9.3270,
    ],
    dtype=float,
)


CONTACT_DAMPING = 20.0
CONTACT_MU = 0.05
CONTACT_K = 1000.0


FIXTURE_NAME = (
    "index_contact_fixture"
)


PREFERRED_INDEX_GEOMS = [
    "collision_idx_prox",
    "collision_idx_mid_mesh",
    "collision_idx_mid",
    "collision_idx_tip",
]


def oid(
    model,
    objtype,
    name,
):

    value = mujoco.mj_name2id(
        model,
        objtype,
        name,
    )

    if value < 0:

        return None

    return int(
        value
    )


def joint_qpos_address(
    model,
    name,
):

    jid = oid(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid is None:

        raise RuntimeError(
            f"Missing joint: {name}"
        )

    return int(
        model.jnt_qposadr[
            jid
        ]
    )


def get_index_geom_names(
    model,
):

    names = []


    for name in PREFERRED_INDEX_GEOMS:

        gid = oid(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )

        if gid is not None:

            names.append(
                name
            )


    # Include any additional index collision geoms too.
    for gid in range(
        model.ngeom
    ):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            gid,
        )

        if not name:

            continue

        if not name.startswith(
            "collision_idx"
        ):

            continue

        if name not in names:

            names.append(
                name
            )


    return names


def geom_distance(
    model,
    data,
    gid_a,
    gid_b,
):

    fromto = np.zeros(
        6,
        dtype=float,
    )


    distance = mujoco.mj_geomDistance(
        model,
        data,
        gid_a,
        gid_b,
        0.100,
        fromto,
    )


    return (
        float(
            distance
        ),
        fromto,
    )


def active_contacts_with_fixture(
    model,
    data,
    fixture_gid,
):

    rows = []


    for i in range(
        data.ncon
    ):

        contact = data.contact[
            i
        ]


        g1 = int(
            contact.geom1
        )

        g2 = int(
            contact.geom2
        )


        if (
            g1 != fixture_gid
            and
            g2 != fixture_gid
        ):

            continue


        other = (
            g2
            if g1 == fixture_gid
            else g1
        )


        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            other,
        )


        rows.append(
            (
                name,
                float(
                    contact.dist
                ),
            )
        )


    return rows


def print_geometry_audit(
    title,
    model,
    data,
    fixture_gid,
    geom_names,
):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


    contacts = active_contacts_with_fixture(
        model,
        data,
        fixture_gid,
    )


    if contacts:

        print()
        print(
            "ACTIVE MUJOCO CONTACTS:"
        )

        for name, distance in contacts:

            print(
                f"  {name:<28s} "
                f"contact.dist = "
                f"{1000.0*distance:+8.3f} mm"
            )

    else:

        print()
        print(
            "ACTIVE MUJOCO CONTACTS: NONE"
        )


    print()
    print(
        "EXACT COLLISION-SURFACE DISTANCES:"
    )


    for name in geom_names:

        gid = oid(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )


        distance, fromto = geom_distance(
            model,
            data,
            gid,
            fixture_gid,
        )


        margin_index = float(
            model.geom_margin[
                gid
            ]
        )

        margin_fixture = float(
            model.geom_margin[
                fixture_gid
            ]
        )

        margin_total = (
            margin_index
            +
            margin_fixture
        )


        contype = int(
            model.geom_contype[
                gid
            ]
        )

        conaff = int(
            model.geom_conaffinity[
                gid
            ]
        )


        if distance <= 0.0:

            status = "PENETRATING"

        elif distance <= margin_total:

            status = "INSIDE CONTACT MARGIN"

        else:

            status = "PHYSICAL GAP"


        print(
            f"  {name:<28s}"
            f"distance={1000.0*distance:+8.3f} mm   "
            f"margin={1000.0*margin_total:6.3f} mm   "
            f"contype={contype:<3d} "
            f"conaff={conaff:<3d} "
            f"{status}"
        )


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


    if not b2.SELECTED_MODEL.exists():

        raise RuntimeError(
            "Missing V35B2 selected model:\n"
            f"{b2.SELECTED_MODEL}"
        )


    # Same module patching used by previous experiments.
    b2.v30.v24.ab.patch_replay_modules()


    # ========================================================
    # Configure EXACT V35C2 selected contact law.
    # ========================================================

    parents = [
        b2.v30.v24.base,
        b2.v30.v24.viewer_tools.base,
    ]


    contact_modules = []


    for parent in parents:

        module = getattr(
            parent,
            "contact_replay",
            None,
        )

        if module is None:

            continue

        if not any(
            id(module) == id(x)
            for x in contact_modules
        ):

            contact_modules.append(
                module
            )


    if not contact_modules:

        raise RuntimeError(
            "Could not locate contact_replay module."
        )


    old_damping = {}


    for module in contact_modules:

        if not hasattr(
            module,
            "CONTACT_DAMPING",
        ):

            raise RuntimeError(
                f"{module.__name__} has no CONTACT_DAMPING"
            )

        old_damping[
            id(module)
        ] = (
            module.CONTACT_DAMPING
        )

        module.CONTACT_DAMPING = (
            CONTACT_DAMPING
        )


    old_mu = (
        b2.v35.CONTACT_MU
    )

    old_k = (
        b2.v35.CONTACT_K
    )


    b2.v35.CONTACT_MU = (
        CONTACT_MU
    )

    b2.v35.CONTACT_K = (
        CONTACT_K
    )


    # ========================================================
    # Capture the actual MjModel/MjData used by the normal
    # dynamic replay.
    # ========================================================

    captured = {}


    original_mj_step = (
        mujoco.mj_step
    )


    def capture_step(
        model,
        data,
        *step_args,
        **step_kwargs,
    ):

        result = original_mj_step(
            model,
            data,
            *step_args,
            **step_kwargs,
        )

        captured[
            "model"
        ] = model

        captured[
            "data"
        ] = data

        return result


    mujoco.mj_step = (
        capture_step
    )


    try:

        print()
        print("=" * 120)
        print(
            "V35D1 — RUNNING NORMAL V35C2 DYNAMIC REPLAY"
        )
        print("=" * 120)


        result = b2.run_candidate(
            b2.SELECTED_MODEL,
            contact_bag,
        )


    finally:

        mujoco.mj_step = (
            original_mj_step
        )


        for module in contact_modules:

            module.CONTACT_DAMPING = (
                old_damping[
                    id(module)
                ]
            )


        b2.v35.CONTACT_MU = (
            old_mu
        )

        b2.v35.CONTACT_K = (
            old_k
        )


    if (
        "model" not in captured
        or
        "data" not in captured
    ):

        raise RuntimeError(
            "Could not capture replay model/data."
        )


    model = captured[
        "model"
    ]

    data = captured[
        "data"
    ]


    fixture_gid = oid(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FIXTURE_NAME,
    )


    if fixture_gid is None:

        raise RuntimeError(
            f"Missing fixture geom: "
            f"{FIXTURE_NAME}"
        )


    geom_names = get_index_geom_names(
        model
    )


    print()
    print(
        "Dynamic replay reported:"
    )

    print(
        "  MCP/PIP/DIP = "
        f"{result['pose'][0]:.3f} / "
        f"{result['pose'][1]:.3f} / "
        f"{result['pose'][2]:.3f} deg"
    )

    print(
        f"  RMSE = "
        f"{result['rmse']:.3f} deg"
    )


    # ========================================================
    # Determine qpos sign convention automatically.
    # ========================================================

    joint_names = [
        b2.v35.MCP,
        b2.v35.PIP,
        b2.v35.DIP,
    ]


    qadr = [
        joint_qpos_address(
            model,
            name,
        )
        for name in joint_names
    ]


    raw_deg = np.degrees(
        np.array(
            [
                data.qpos[
                    adr
                ]
                for adr in qadr
            ]
        )
    )


    expected_deg = np.asarray(
        result[
            "pose"
        ],
        dtype=float,
    )


    signs = np.ones(
        3,
        dtype=float,
    )


    for i in range(
        3
    ):

        positive_error = abs(
            raw_deg[
                i
            ]
            -
            expected_deg[
                i
            ]
        )

        negative_error = abs(
            -raw_deg[
                i
            ]
            -
            expected_deg[
                i
            ]
        )

        if negative_error < positive_error:

            signs[
                i
            ] = -1.0


    interpreted_deg = (
        signs
        *
        raw_deg
    )


    print()
    print(
        "Captured final qpos:"
    )

    print(
        "  raw       = "
        f"{raw_deg[0]:.3f} / "
        f"{raw_deg[1]:.3f} / "
        f"{raw_deg[2]:.3f} deg"
    )

    print(
        "  interpreted= "
        f"{interpreted_deg[0]:.3f} / "
        f"{interpreted_deg[1]:.3f} / "
        f"{interpreted_deg[2]:.3f} deg"
    )

    print(
        "  signs     = "
        f"{signs.astype(int).tolist()}"
    )


    # Save exact dynamic-final state.
    dynamic_qpos = (
        data.qpos.copy()
    )

    dynamic_qvel = (
        data.qvel.copy()
    )


    mujoco.mj_forward(
        model,
        data,
    )


    print_geometry_audit(
        "V35D1-A — ACTUAL DYNAMIC FINAL CONFIGURATION",
        model,
        data,
        fixture_gid,
        geom_names,
    )


    # ========================================================
    # STATIC HARDWARE-TARGET GEOMETRY TEST
    #
    # This is NOT a simulated solution.
    # We prescribe measured q ONLY to test geometric
    # compatibility.
    # ========================================================

    data.qpos[:] = (
        dynamic_qpos
    )

    data.qvel[:] = 0.0


    for i, adr in enumerate(
        qadr
    ):

        data.qpos[
            adr
        ] = (
            signs[
                i
            ]
            *
            np.deg2rad(
                HW_DEG[
                    i
                ]
            )
        )


    mujoco.mj_forward(
        model,
        data,
    )


    print()
    print("=" * 120)
    print(
        "STATIC HARDWARE TARGET"
    )
    print("=" * 120)

    print(
        "MCP/PIP/DIP prescribed ONLY for geometry audit:"
    )

    print(
        f"  {HW_DEG[0]:.4f} / "
        f"{HW_DEG[1]:.4f} / "
        f"{HW_DEG[2]:.4f} deg"
    )


    print_geometry_audit(
        "V35D1-B — MEASURED HARDWARE CONFIGURATION",
        model,
        data,
        fixture_gid,
        geom_names,
    )


    print()
    print("=" * 120)
    print(
        "V35D1 INTERPRETATION"
    )
    print("=" * 120)

    print()
    print(
        "If the measured hardware pose still shows "
        "millimetre-scale middle/tip gaps:"
    )

    print(
        "  => collision geometry / object registration "
        "is incompatible with the real wrap."
    )

    print()
    print(
        "If the measured hardware pose gives middle + tip "
        "contact/penetration:"
    )

    print(
        "  => geometry CAN support the wrap; the remaining "
        "failure is dynamic equilibrium/compliance."
    )


    print()
    print("=" * 120)
    print(
        "STATIC HARDWARE-TARGET GUI"
    )
    print("=" * 120)

    print()
    print(
        "NOTE: this GUI is a geometry diagnostic."
    )

    print(
        "The measured hardware angles are prescribed here;"
    )

    print(
        "this is NOT being claimed as simulated behavior."
    )


    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        while viewer.is_running():

            viewer.sync()

            time.sleep(
                0.02
            )


if __name__ == "__main__":
    main()
