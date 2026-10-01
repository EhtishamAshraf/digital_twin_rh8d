#!/usr/bin/env python3

import mujoco


_STYLED = set()


def make_index_finger_solid(model, alpha=1.0):
    """
    Make visual geoms belonging to Index bodies opaque.

    Collision geoms are deliberately ignored.
    Rendering only: no collision/dynamics parameters are changed.
    """

    key = id(model)

    if key in _STYLED:
        return

    changed_geoms = 0
    changed_materials = set()

    for gid in range(model.ngeom):

        gname = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            gid,
        )

        gname = gname or ""

        # Keep diagnostic collision capsules unchanged.
        if "collision" in gname.lower():
            continue

        bid = int(
            model.geom_bodyid[gid]
        )

        belongs_to_index = False

        current = bid

        while current >= 0:

            bname = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                current,
            )

            bname = bname or ""

            lower = bname.lower()

            if (
                "index" in lower
                or "idx" in lower
            ):
                belongs_to_index = True
                break

            parent = int(
                model.body_parentid[current]
            )

            if parent == current:
                break

            current = parent

        if not belongs_to_index:
            continue

        model.geom_rgba[gid, 3] = alpha
        changed_geoms += 1

        matid = int(
            model.geom_matid[gid]
        )

        if matid >= 0:

            model.mat_rgba[
                matid,
                3,
            ] = alpha

            changed_materials.add(
                matid
            )

    _STYLED.add(key)

    print(
        f"[VIEWER] Solid index finger: "
        f"{changed_geoms} visual geoms, "
        f"{len(changed_materials)} materials, "
        f"alpha={alpha:.2f}"
    )
