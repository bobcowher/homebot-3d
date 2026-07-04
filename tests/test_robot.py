import numpy as np
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import compile_model, tile_center
from homebot3d.robot import Robot

def _sim():
    m = DefaultHouseMap()
    model = compile_model(m)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return m, model, data

def test_initial_pose_matches_start_tile():
    m, model, data = _sim()
    r = Robot(model, data)
    cx, cy = tile_center(*m.robot_start_tile)
    assert abs(r.x - cx) < 1e-6
    assert abs(r.y - cy) < 1e-6

def test_forward_action_moves_along_heading():
    m, model, data = _sim()
    r = Robot(model, data)
    x0, y0 = r.x, r.y
    for _ in range(50):                      # 0.5 s at dt=0.01
        r.apply(np.array([1.0, 0.0]))
        mujoco.mj_step(model, data)
    assert r.x > x0 + 0.05                    # heading 0 => +x motion
    assert abs(r.y - y0) < 0.05

def test_turn_action_changes_heading():
    m, model, data = _sim()
    r = Robot(model, data)
    h0 = r.heading
    for _ in range(50):
        r.apply(np.array([0.0, 1.0]))
        mujoco.mj_step(model, data)
    assert r.heading > h0 + 0.1

def test_collided_false_in_open_space():
    m, model, data = _sim()
    r = Robot(model, data)
    assert r.collided() is False
