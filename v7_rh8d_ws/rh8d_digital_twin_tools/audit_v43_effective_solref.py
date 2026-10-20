from pathlib import Path
import mujoco
import test_v43_pd_compliant_coupling as v43

WS = Path.home() / (
    "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/"
    "2-MuJoCo/digital_twin/v7_rh8d_ws"
)
MJCF = WS / "src/ur5e_mujoco_description/mjcf"

CASES = [
    ("A_ORIGINAL", MJCF / "tmp_v43_k750_r20.xml"),
    ("C_HALF_STIFFNESS",
     MJCF / "tmp_v43_control_mesh_half_fixture_stiffness.xml"),
]

for label, xml in CASES:
    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)

    # Check the XML as compiled independently.
    disk_model = mujoco.MjModel.from_xml_path(str(xml))
    gid = mujoco.mj_name2id(
        disk_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "index_contact_fixture",
    )
    print("DISK compiled solref:", disk_model.geom_solref[gid])

    original_step = mujoco.mj_step
    seen = set()

    def inspect_step(model, data, *args, **kwargs):
        result = original_step(model, data, *args, **kwargs)

        for checkpoint in (6.191, 7.335):
            if checkpoint in seen or data.time < checkpoint:
                continue
            seen.add(checkpoint)

            fix = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM,
                "index_contact_fixture"
            )

            print(f"\nACTUAL REPLAY at t={data.time:.3f}")
            print("Runtime fixture solref:", model.geom_solref[fix])
            print("Runtime fixture priority:", model.geom_priority[fix])
            print("Contact override flags:", model.opt.enableflags)

            matches = 0
            for k in range(data.ncon):
                c = data.contact[k]
                a, b = int(c.geom1), int(c.geom2)

                if fix not in (a, b):
                    continue

                partner = b if a == fix else a
                name = mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    partner,
                )

                if name not in (
                    "collision_idx_mid_mesh",
                    "collision_idx_mid",
                    "collision_idx_tip",
                ):
                    continue

                matches += 1
                print(
                    f"  {name}: "
                    f"solref={list(c.solref)} "
                    f"dist={c.dist*1000:+.4f}mm "
                    f"efc_address={c.efc_address}"
                )

            print("Matching contacts:", matches)

        return result

    mujoco.mj_step = inspect_step
    try:
        result = v43.run_contact(xml, label + "_SOLREF_AUDIT")
        print("\nReported RMSE:", result["rmse_deg"])
    finally:
        mujoco.mj_step = original_step

print("\nAUDIT COMPLETE — no model changes.")
