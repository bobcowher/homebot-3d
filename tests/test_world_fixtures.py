import numpy as np
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import build_mjcf, tile_center

def test_each_fixture_becomes_a_body():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    for name in m.fixtures:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"fixture_{name}")
        assert bid != -1, f"missing body fixture_{name}"

def test_fixture_body_at_tile_center():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fixture_fridge")
    cx, cy = tile_center(*m.fixtures["fridge"])
    np.testing.assert_allclose(model.body_pos[bid][:2], [cx, cy], atol=1e-6)

def test_fixtures_have_multiple_geoms_all_prefixed():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    # recliner is modelled as seat + back → at least 2 geoms, all fixture_-prefixed
    recliner_geoms = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        for gid in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
        .startswith("fixture_recliner")
    ]
    assert len(recliner_geoms) >= 2


def test_furniture_geoms_are_reachable_sized():
    from homebot3d.constants import REACH_RADIUS, ROBOT_RADIUS
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    margin = REACH_RADIUS - ROBOT_RADIUS  # robot can reach within this of centre
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("fixture_"):
            for k in (0, 1):  # x, y: |offset from centre| + half-extent <= margin
                surface = abs(model.geom_pos[gid][k]) + model.geom_size[gid][k]
                assert surface <= margin + 1e-9, f"{name} axis {k} surface {surface}"
