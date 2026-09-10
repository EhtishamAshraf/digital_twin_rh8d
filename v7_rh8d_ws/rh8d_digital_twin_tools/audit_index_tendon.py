#!/usr/bin/env python3

from pathlib import Path
import mujoco
import numpy as np


WS = Path.home() / (
    "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
)

MODEL_PATH = (
    WS
    / "src/ur5e_mujoco_description/mjcf/ur5e_rh8d_scene.xml"
)

TENDON_NAME = "idx_flex_spatial"

JOINTS = [
    "Index_Proximal--palm_assembly:1",
    "Index_Middle--Index_Proximal",
    "Index_Distal--Index_Middle",
]

SITES = [
    "palm_flexor_origin_idx",
    "idx_guide_p",
    "idx_guide_m",
    "idx_guide_d",
    "idx_anchor",
]


def name_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Could not find: {name}")
    return idx


def main():
    print(f"Loading:\n{MODEL_PATH}\n")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    tendon_id = name_id(
        model,
        mujoco.mjtObj.mjOBJ_TENDON,
        TENDON_NAME,
    )

    print("=" * 70)
    print("INDEX TENDON AUDIT")
    print("=" * 70)

    open_length = data.ten_length[tendon_id]

    print(f"\nOpen tendon length:")
    print(f"  {open_length:.8f} m")
    print(f"  {open_length * 1000.0:.3f} mm")

    print("\nTendon routing sites in WORLD coordinates:")
    print("------------------------------------------------")

    for site_name in SITES:
        sid = name_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        xyz = data.site_xpos[sid]

        print(
            f"{site_name:28s}"
            f" x={xyz[0]*1000:8.3f} mm"
            f" y={xyz[1]*1000:8.3f} mm"
            f" z={xyz[2]*1000:8.3f} mm"
        )

    print("\nLocal tendon sensitivity at OPEN configuration:")
    print("------------------------------------------------")
    print("dL/dq is approximately the tendon moment arm.")
    print("Computed numerically so it works with dense or sparse MuJoCo Jacobians.")
    print()

    qpos_reference = data.qpos.copy()
    eps = 1.0e-5  # rad

    for joint_name in JOINTS:

        jid = name_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        dof_adr = model.jnt_dofadr[jid]
        qpos_adr = model.jnt_qposadr[jid]

        # Baseline configuration
        data.qpos[:] = qpos_reference
        mujoco.mj_forward(model, data)
        length_0 = data.ten_length[tendon_id]

        # Very small positive joint perturbation.
        # Positive is used because the open finger is near its lower joint limit.
        data.qpos[qpos_adr] += eps
        mujoco.mj_forward(model, data)
        length_1 = data.ten_length[tendon_id]

        jac = (length_1 - length_0) / eps

        print(f"{joint_name}")
        print(f"  qpos address : {qpos_adr}")
        print(f"  dof address  : {dof_adr}")
        print(f"  dL/dq        : {jac:.8f} m/rad")
        print(f"  moment arm   : {jac * 1000.0:.3f} mm")
        print()

    # Restore original configuration
    data.qpos[:] = qpos_reference
    mujoco.mj_forward(model, data)

    print("\nFinite-difference check using +0.10 rad:")
    print("------------------------------------------------")

    qpos_original = data.qpos.copy()

    for joint_name in JOINTS:

        data.qpos[:] = qpos_original
        mujoco.mj_forward(model, data)

        baseline = data.ten_length[tendon_id]

        jid = name_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qadr = model.jnt_qposadr[jid]

        data.qpos[qadr] += 0.10

        mujoco.mj_forward(model, data)

        new_length = data.ten_length[tendon_id]

        delta = new_length - baseline

        print(f"{joint_name}")
        print(f"  L(0)         = {baseline * 1000.0:.3f} mm")
        print(f"  L(+0.10 rad) = {new_length * 1000.0:.3f} mm")
        print(f"  delta L      = {delta * 1000.0:.3f} mm")
        print(
            f"  deltaL/dq    = "
            f"{delta / 0.10 * 1000.0:.3f} mm/rad"
        )
        print()

    data.qpos[:] = qpos_original
    mujoco.mj_forward(model, data)

    print("=" * 70)
    print("AUDIT FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()
