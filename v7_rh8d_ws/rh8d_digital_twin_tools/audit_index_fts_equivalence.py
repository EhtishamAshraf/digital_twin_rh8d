#!/usr/bin/env python3

from pathlib import Path
import math

import mujoco
import numpy as np


WS = Path.home() / (
    "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
)

MODEL_PATH = (
    WS /
    "src/ur5e_mujoco_description/mjcf/"
    "ur5e_rh8d_index_contact_cylinder_v1.xml"
)

OPEN_TICKS = 100.0
END_TICKS = 1120.0

METERS_PER_TICK = 6.0e-6
CLOSING_RATE_TICKS_S = 200.0

SPOOL_JOINT = "index_motor_spool"
ACTUATOR_NAME = "pos_index_tendon"

SENSOR_NAME = "index_tip_force"

TIP_GEOM = "collision_idx_tip"
FIXTURE_GEOM = "index_contact_fixture"


def object_id(model, object_type, name):

    idx = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if idx < 0:
        raise RuntimeError(
            f"Could not find MuJoCo object: {name}"
        )

    return idx


def set_motor_ticks(
    model,
    data,
    spool_qadr,
    spool_dofadr,
    actuator_id,
    ticks,
):

    q = max(
        0.0,
        (ticks - OPEN_TICKS)
        * METERS_PER_TICK,
    )

    data.qpos[spool_qadr] = q
    data.qvel[spool_dofadr] = 0.0
    data.ctrl[actuator_id] = q


def relevant_contact_force_world(
    model,
    data,
    tip_gid,
    fixture_gid,
):

    resultant_world = np.zeros(3)
    number_contacts = 0

    for contact_index in range(data.ncon):

        c = data.contact[contact_index]

        pair_is_correct = (
            (
                c.geom1 == tip_gid
                and
                c.geom2 == fixture_gid
            )
            or
            (
                c.geom2 == tip_gid
                and
                c.geom1 == fixture_gid
            )
        )

        if not pair_is_correct:
            continue

        wrench_contact = np.zeros(6)

        mujoco.mj_contactForce(
            model,
            data,
            contact_index,
            wrench_contact,
        )

        # contact.frame stores the contact-frame axes.
        #
        # mj_contactForce() gives force in that local contact frame.
        # Rotating it into world coordinates lets us vector-sum multiple
        # simultaneous contacts before taking the magnitude.
        contact_R = (
            np.asarray(c.frame)
            .reshape(3, 3)
        )

        force_world = (
            contact_R.T
            @ wrench_contact[:3]
        )

        resultant_world += force_world
        number_contacts += 1

    return (
        resultant_world,
        number_contacts,
    )


def main():

    print()
    print("=" * 92)
    print("RH8D INDEX — SIMULATED FTS EQUIVALENCE AUDIT")
    print("=" * 92)

    print()
    print(f"Loading:")
    print(MODEL_PATH)

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    # Keep identical local identification conditions.
    model.opt.gravity[:] = 0.0


    spool_jid = object_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        SPOOL_JOINT,
    )

    spool_qadr = model.jnt_qposadr[
        spool_jid
    ]

    spool_dofadr = model.jnt_dofadr[
        spool_jid
    ]


    actuator_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAME,
    )


    sensor_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        SENSOR_NAME,
    )

    sensor_adr = model.sensor_adr[
        sensor_id
    ]

    sensor_dim = model.sensor_dim[
        sensor_id
    ]

    if sensor_dim != 3:

        raise RuntimeError(
            f"{SENSOR_NAME} dimension is "
            f"{sensor_dim}, expected 3"
        )


    tip_gid = object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        TIP_GEOM,
    )

    fixture_gid = object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FIXTURE_GEOM,
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    # ---------------------------------------------------------
    # Settle open finger.
    # ---------------------------------------------------------

    for _ in range(
        int(
            1.5
            / model.opt.timestep
        )
    ):

        set_motor_ticks(
            model,
            data,
            spool_qadr,
            spool_dofadr,
            actuator_id,
            OPEN_TICKS,
        )

        mujoco.mj_step(
            model,
            data,
        )


    rows = []

    ticks = OPEN_TICKS

    first_contact_ticks = None
    first_sensor_001_ticks = None
    first_sensor_010_ticks = None


    while ticks <= END_TICKS:

        set_motor_ticks(
            model,
            data,
            spool_qadr,
            spool_dofadr,
            actuator_id,
            ticks,
        )

        mujoco.mj_step(
            model,
            data,
        )


        sensor_xyz = (
            data.sensordata[
                sensor_adr:
                sensor_adr + 3
            ].copy()
        )

        sensor_norm = float(
            np.linalg.norm(
                sensor_xyz
            )
        )


        contact_world, ncon = (
            relevant_contact_force_world(
                model,
                data,
                tip_gid,
                fixture_gid,
            )
        )

        contact_norm = float(
            np.linalg.norm(
                contact_world
            )
        )


        if (
            first_contact_ticks is None
            and
            ncon > 0
        ):
            first_contact_ticks = ticks


        if (
            first_sensor_001_ticks is None
            and
            sensor_norm >= 0.01
        ):
            first_sensor_001_ticks = ticks


        if (
            first_sensor_010_ticks is None
            and
            sensor_norm >= 0.10
        ):
            first_sensor_010_ticks = ticks


        if (
            ncon > 0
            or
            sensor_norm > 1e-6
        ):

            diff = (
                sensor_norm
                -
                contact_norm
            )

            ratio = (
                sensor_norm
                / contact_norm
                if contact_norm > 1e-9
                else math.nan
            )

            rows.append(
                (
                    ticks,
                    sensor_xyz.copy(),
                    sensor_norm,
                    contact_norm,
                    diff,
                    ratio,
                    ncon,
                )
            )


        ticks += (
            CLOSING_RATE_TICKS_S
            * model.opt.timestep
        )


    print()
    print("=" * 92)
    print("ONSET")
    print("=" * 92)

    print(
        f"First geometric contact       = "
        f"{first_contact_ticks}"
    )

    print(
        f"First simulated FTS >=0.01 N = "
        f"{first_sensor_001_ticks}"
    )

    print(
        f"First simulated FTS >=0.10 N = "
        f"{first_sensor_010_ticks}"
    )


    print()
    print("=" * 92)
    print("SELECTED SAMPLES")
    print("=" * 92)

    print(
        " ticks    FTS norm [N]   contact resultant [N]   "
        "difference [N]   ratio   contacts"
    )


    targets = [
        1010,
        1020,
        1030,
        1050,
        1075,
        1100,
    ]


    for target in targets:

        if not rows:
            break

        row = min(
            rows,
            key=lambda r:
                abs(r[0] - target),
        )

        (
            ticks_r,
            sensor_xyz,
            sensor_norm,
            contact_norm,
            diff,
            ratio,
            ncon,
        ) = row

        print(
            f"{ticks_r:7.1f}"
            f"   {sensor_norm:12.6f}"
            f"   {contact_norm:20.6f}"
            f"   {diff:14.6f}"
            f"   {ratio:7.4f}"
            f"   {ncon:3d}"
        )


    valid = [
        r
        for r in rows
        if (
            r[3] > 0.01
            and
            r[2] > 0.01
        )
    ]


    print()
    print("=" * 92)
    print("EQUIVALENCE STATISTICS")
    print("=" * 92)


    if not valid:

        print(
            "No samples with both forces "
            "above 0.01 N."
        )

        return


    sensor = np.array(
        [
            r[2]
            for r in valid
        ]
    )

    contact = np.array(
        [
            r[3]
            for r in valid
        ]
    )

    error = (
        sensor - contact
    )

    ratio = (
        sensor / contact
    )


    rmse = math.sqrt(
        float(
            np.mean(
                error ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(error)
        )
    )


    print(
        f"Samples                  = "
        f"{len(valid)}"
    )

    print(
        f"FTS/contact median ratio = "
        f"{np.median(ratio):.6f}"
    )

    print(
        f"FTS/contact mean ratio   = "
        f"{np.mean(ratio):.6f}"
    )

    print(
        f"Force RMSE               = "
        f"{rmse:.8f} N"
    )

    print(
        f"Force MAE                = "
        f"{mae:.8f} N"
    )

    print(
        f"Maximum absolute error   = "
        f"{np.max(np.abs(error)):.8f} N"
    )


    print()
    print("=" * 92)

    if (
        abs(
            np.median(ratio) - 1.0
        ) < 0.02
        and
        rmse < 0.01
    ):

        print(
            "RESULT: simulated FTS magnitude and "
            "contact resultant are effectively equivalent."
        )

    else:

        print(
            "RESULT: simulated FTS and direct contact "
            "resultant are NOT numerically equivalent."
        )

        print(
            "We must use the actual FTS sensor output "
            "during system identification."
        )

    print("=" * 92)


if __name__ == "__main__":
    main()
