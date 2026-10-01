#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import numpy as np
import mujoco

import fit_index_contact_v1 as base


WS = base.WS

FREE_MODEL = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_identified.xml"
)

CONTACT_MODEL = (
    WS
    / "src/ur5e_mujoco_description/mjcf/"
      "ur5e_rh8d_index_contact_v1.xml"
)


OPEN_ENCODER = 100.0
METERS_PER_TICK = 6.0e-6

TENDON_NAME = "idx_flex_spatial"
SPOOL_NAME = "index_motor_spool"
ACTUATOR_NAME = "pos_index_tendon"

MCP_NAME = "Index_Proximal--palm_assembly:1"
PIP_NAME = "Index_Middle--Index_Proximal"
DIP_NAME = "Index_Distal--Index_Middle"


# Same physical sequence used for free-space validation.
FREE_SEQUENCE = [
    ("closing", 100.0),
    ("closing", 500.0),
    ("closing", 1000.0),
    ("closing", 1500.0),
    ("closing", 2000.0),
    ("closing", 2466.0),
    ("closing", 2961.0),
    ("closing", 3466.0),

    ("opening", 3000.0),
    ("opening", 2500.0),
    ("opening", 2000.0),
    ("opening", 1500.0),
    ("opening", 1000.0),
    ("opening", 500.0),
    ("opening", 100.0),
]


def obj_id(model, obj_type, name):

    idx = mujoco.mj_name2id(
        model,
        obj_type,
        name,
    )

    if idx < 0:
        raise RuntimeError(
            f"Could not find {name}"
        )

    return idx


def setup_model(path):

    model = mujoco.MjModel.from_xml_path(
        str(path)
    )

    data = mujoco.MjData(model)

    model.opt.gravity[:] = 0.0

    tendon_id = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    spool_jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_NAME,
    )

    actuator_id = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAME,
    )

    mcp_jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        MCP_NAME,
    )

    pip_jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        PIP_NAME,
    )

    dip_jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        DIP_NAME,
    )

    return {
        "model": model,
        "data": data,

        "tendon": tendon_id,

        "spool_q":
            model.jnt_qposadr[spool_jid],

        "spool_dof":
            model.jnt_dofadr[spool_jid],

        "actuator": actuator_id,

        "mcp_q":
            model.jnt_qposadr[mcp_jid],

        "pip_q":
            model.jnt_qposadr[pip_jid],

        "dip_q":
            model.jnt_qposadr[dip_jid],
    }


def impose_encoder(
    ctx,
    encoder,
):

    model = ctx["model"]
    data = ctx["data"]

    spool = max(
        0.0,
        (
            encoder - OPEN_ENCODER
        ) * METERS_PER_TICK,
    )

    data.qpos[
        ctx["spool_q"]
    ] = spool

    data.qvel[
        ctx["spool_dof"]
    ] = 0.0

    data.ctrl[
        ctx["actuator"]
    ] = spool

    return spool


def settle_open(ctx):

    model = ctx["model"]
    data = ctx["data"]

    mujoco.mj_resetData(
        model,
        data,
    )

    data.ctrl[:] = 0.0

    for _ in range(
        int(
            1.5
            / model.opt.timestep
        )
    ):

        impose_encoder(
            ctx,
            OPEN_ENCODER,
        )

        mujoco.mj_step(
            model,
            data,
        )


def print_state(
    label,
    encoder,
    ctx,
):

    data = ctx["data"]

    tendon_length = data.ten_length[
        ctx["tendon"]
    ]

    tendon_velocity = data.ten_velocity[
        ctx["tendon"]
    ]

    print(
        f"{label:18s} "
        f"enc={encoder:7.1f}  "
        f"L={tendon_length*1000:8.4f} mm  "
        f"dL={tendon_velocity*1000:9.4f} mm/s  "
        f"MCP={math.degrees(data.qpos[ctx['mcp_q']]):7.2f}°  "
        f"PIP={math.degrees(data.qpos[ctx['pip_q']]):7.2f}°  "
        f"DIP={math.degrees(data.qpos[ctx['dip_q']]):7.2f}°"
    )


def audit_free_space():

    ctx = setup_model(
        FREE_MODEL
    )

    model = ctx["model"]
    data = ctx["data"]

    settle_open(
        ctx
    )

    print()
    print("=" * 100)
    print("FREE-SPACE TENDON COORDINATE")
    print("=" * 100)

    max_length = -1e9

    for phase, encoder in FREE_SEQUENCE:

        for _ in range(
            int(
                2.0
                / model.opt.timestep
            )
        ):

            impose_encoder(
                ctx,
                encoder,
            )

            mujoco.mj_step(
                model,
                data,
            )

        L = float(
            data.ten_length[
                ctx["tendon"]
            ]
        )

        max_length = max(
            max_length,
            L,
        )

        print_state(
            phase,
            encoder,
            ctx,
        )

    print()
    print(
        f"MAX FREE-SPACE tendon coordinate = "
        f"{max_length*1000:.4f} mm"
    )

    return max_length


def audit_contact(
    bag,
):

    (
        states,
        force_data,
        encoder_data,
    ) = base.read_bag(
        bag
    )

    transitions = (
        base.compress_states(
            states
        )
    )

    (
        approach_start,
        regulation_start,
        hold_start,
        hold_end,
    ) = base.find_intervals(
        transitions
    )

    encoder_time = (
        encoder_data[:, 0]
    )

    encoder_position = (
        base.median_filter_5(
            encoder_data[:, 1]
        )
    )


    ctx = setup_model(
        CONTACT_MODEL
    )

    model = ctx["model"]
    data = ctx["data"]

    fixture_gid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        base.FIXTURE_GEOM,
    )


    # Use a reasonably damped contact law only for this
    # diagnostic. We are NOT identifying it here.
    model.geom_solref[
        fixture_gid,
        0
    ] = -32000.0

    model.geom_solref[
        fixture_gid,
        1
    ] = -320.0


    settle_open(
        ctx
    )


    print()
    print("=" * 100)
    print("CONTACT TENDON COORDINATE")
    print("=" * 100)

    print()
    print(
        f"regulation start = "
        f"{regulation_start:.6f} s"
    )

    print(
        f"hold start       = "
        f"{hold_start:.6f} s"
    )

    print(
        f"hold end         = "
        f"{hold_end:.6f} s"
    )


    dt = model.opt.timestep

    duration = (
        hold_end
        - approach_start
    )

    steps = int(
        math.ceil(
            duration / dt
        )
    )


    sample_times = {
        "CONTACT/REG START":
            regulation_start,

        "HOLD START":
            hold_start,

        "HOLD MIDDLE":
            0.5
            * (
                hold_start
                + hold_end
            ),

        "HOLD END":
            hold_end,
    }


    printed = set()

    contact_lengths = []

    precontact_lengths = []


    for step in range(
        steps + 1
    ):

        t = (
            approach_start
            + step * dt
        )

        encoder = float(
            np.interp(
                t,
                encoder_time,
                encoder_position,
            )
        )

        impose_encoder(
            ctx,
            encoder,
        )

        mujoco.mj_step(
            model,
            data,
        )

        L = float(
            data.ten_length[
                ctx["tendon"]
            ]
        )


        if t < regulation_start:

            precontact_lengths.append(
                L
            )

        if (
            hold_start
            <= t
            <= hold_end
        ):

            contact_lengths.append(
                L
            )


        for label, requested_t in (
            sample_times.items()
        ):

            if (
                label not in printed
                and t >= requested_t
            ):

                print_state(
                    label,
                    encoder,
                    ctx,
                )

                printed.add(
                    label
                )


    pre = np.asarray(
        precontact_lengths
    )

    hold = np.asarray(
        contact_lengths
    )


    print()
    print(
        "Pre-contact tendon coordinate:"
    )

    print(
        f"  max    = "
        f"{np.max(pre)*1000:.4f} mm"
    )

    print(
        f"  p95    = "
        f"{np.percentile(pre,95)*1000:.4f} mm"
    )


    print()
    print(
        "0.15-N hold-region tendon coordinate:"
    )

    print(
        f"  min    = "
        f"{np.min(hold)*1000:.4f} mm"
    )

    print(
        f"  median = "
        f"{np.median(hold)*1000:.4f} mm"
    )

    print(
        f"  max    = "
        f"{np.max(hold)*1000:.4f} mm"
    )


def main():

    if len(
        sys.argv
    ) != 2:

        print(
            "Usage: python3 "
            "audit_index_tendon_loading.py "
            "/path/to/contact_rosbag_directory"
        )

        sys.exit(1)


    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    print()
    print("=" * 100)
    print(
        "RH8D INDEX TENDON LOAD-STATE AUDIT"
    )
    print("=" * 100)

    print()
    print(
        "Primary tendon stiffness remains "
        "1200 N/m."
    )


    max_free = audit_free_space()

    audit_contact(
        bag
    )


    print()
    print("=" * 100)
    print("WHY THIS AUDIT MATTERS")
    print("=" * 100)

    print()
    print(
        "If the contact-hold tendon coordinate "
        "occupies a range that is larger than the "
        "free-space coordinate, we can introduce a "
        "secondary passive tendon spring that is "
        "inactive below that threshold."
    )

    print()
    print(
        "That would preserve the validated "
        "1200-N/m free-space behavior while adding "
        "stiffness only when the transmission becomes "
        "loaded."
    )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
