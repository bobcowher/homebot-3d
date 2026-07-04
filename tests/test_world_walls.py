import numpy as np
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import compile_model, tile_center
from homebot3d.constants import TILE, WALL_HEIGHT, WALL_THICK


def _model():
    return compile_model(DefaultHouseMap())


def _geom_names(model, prefix):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
             for i in range(model.ngeom)]
    return [n for n in names if n and n.startswith(prefix)]


def test_tile_center():
    x, y = tile_center(0, 0)
    assert x == 0.25 and y == 0.25


def test_mjcf_compiles_in_mujoco():
    assert _model() is not None


def test_walls_exist_and_are_thin():
    model = _model()
    walls = _geom_names(model, "wall_")
    assert len(walls) > 0
    for name in walls:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        sx, sy, sz = model.geom_size[gid]
        # One planar axis is thin (a wall panel or a square jamb post).
        assert min(sx, sy) <= WALL_THICK + 1e-6
        np.testing.assert_allclose(sz, WALL_HEIGHT / 2, atol=1e-6)


def test_door_frames_present():
    model = _model()
    m = DefaultHouseMap()
    posts = _geom_names(model, "wall_frame_")
    assert len(posts) == 2 * len(m.doorways)


def test_robot_drives_through_doorway_without_stalling():
    # Living<->hallway doorway at row 8, cols 4-6. Tile row 8 sits at LARGER y
    # than the start row 7 (y=(row+0.5)*TILE), so the robot must drive +y to
    # pass through. Command world-frame +y velocity directly (heading-agnostic)
    # so the test isolates wall geometry, not the controller.
    m = DefaultHouseMap()
    model = compile_model(m, robot_start=(5, 7))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    from homebot3d.robot import Robot
    r = Robot(model, data)
    a_vy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vy")
    y0 = r.y
    for _ in range(400):                 # 4 s at dt=0.01
        data.ctrl[a_vy] = 1.0            # +y toward and through the doorway
        mujoco.mj_step(model, data)
    # Robot should have travelled well past the doorway line into the hallway.
    assert r.y > y0 + 0.9, f"stalled: y0={y0:.3f} y={r.y:.3f}"
