import numpy as np
from homebot3d.maps import MAP_REGISTRY, FLOOR, WALL, LAWN, DefaultHouseMap

def test_default_map_registered():
    assert MAP_REGISTRY["default"] is DefaultHouseMap

def test_default_map_shape_and_border_walls():
    m = DefaultHouseMap()
    assert m.tiles.shape == (18, 27)
    # Border is wall everywhere except the east lawn strip (cols 24-26), which
    # is open ground beyond the front door and carries no wall panels.
    assert (m.tiles[0, :24] == WALL).all()       # top border wall (interior)
    assert (m.tiles[-1, :24] == WALL).all()      # bottom border wall (interior)
    assert (m.tiles[:, 24:] == LAWN).all()       # east strip is lawn, not wall

def test_fixtures_are_on_valid_coordinates():
    m = DefaultHouseMap()
    for name in ("fridge", "recliner", "door"):
        col, row = m.fixtures[name]
        assert 0 <= col < m.tiles.shape[1]
        assert 0 <= row < m.tiles.shape[0]

def test_robot_start_is_a_floor_tile():
    m = DefaultHouseMap()
    col, row = m.robot_start_tile
    assert m.tiles[row, col] == FLOOR

def test_valid_floor_tiles_excludes_walls():
    m = DefaultHouseMap()
    for col, row in m.valid_floor_tiles():
        assert m.tiles[row, col] == FLOOR

def test_spawn_trash_is_deterministic():
    m = DefaultHouseMap()
    a = m.spawn_trash(3, np.random.default_rng(0))
    b = m.spawn_trash(3, np.random.default_rng(0))
    assert a == b
    assert len(a) == 3
