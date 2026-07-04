import numpy as np
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import compile_model
from homebot3d.robot import Robot
from homebot3d.sensors import Camera, privileged

def _sim():
    m = DefaultHouseMap()
    model = compile_model(m)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return m, model, data

def test_rgb_shape_and_dtype():
    m, model, data = _sim()
    cam = Camera(model, width=128, height=128)
    frame = cam.render_rgb(data)
    assert frame.shape == (128, 128, 3)
    assert frame.dtype == np.uint8
    cam.close()

def test_rgb_deterministic_for_same_state():
    m, model, data = _sim()
    cam = Camera(model)
    a = cam.render_rgb(data)
    b = cam.render_rgb(data)
    np.testing.assert_array_equal(a, b)
    cam.close()

def test_depth_shape():
    m, model, data = _sim()
    cam = Camera(model)
    d = cam.render_depth(data)
    assert d.shape == (128, 128)
    assert d.dtype == np.float32
    cam.close()

def test_privileged_pose_and_goal_vec():
    m, model, data = _sim()
    r = Robot(model, data)
    p = privileged(r, goal_xy=(r.x + 1.0, r.y + 2.0))
    np.testing.assert_allclose(p["pose"][:2], [r.x, r.y], atol=1e-6)
    np.testing.assert_allclose(p["goal_vec"], [1.0, 2.0], atol=1e-6)
