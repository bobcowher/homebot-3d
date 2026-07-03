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
