import numpy as np
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import build_mjcf, tile_center
from homebot3d.constants import TILE, WALL_HEIGHT

def test_tile_center():
    x, y = tile_center(0, 0)
    assert x == 0.25 and y == 0.25          # (0+0.5)*0.5

def test_mjcf_compiles_in_mujoco():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    assert model is not None

def test_wall_count_matches_grid():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    n_walls = int((m.tiles == 1).sum())
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
             for i in range(model.ngeom)]
    wall_names = [n for n in names if n and n.startswith("wall_")]
    assert len(wall_names) == n_walls

def test_wall_geom_position_and_size():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_0_0")
    assert gid != -1
    np.testing.assert_allclose(model.geom_size[gid],
                               [TILE/2, TILE/2, WALL_HEIGHT/2], atol=1e-6)
    cx, cy = tile_center(0, 0)
    np.testing.assert_allclose(model.geom_pos[gid], [cx, cy, WALL_HEIGHT/2], atol=1e-6)

def test_has_floor_plane():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert gid != -1
    assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE
