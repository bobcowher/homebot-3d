import numpy as np
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import build_mjcf, tile_center
from homebot3d.constants import ROBOT_HALFHEIGHT

def _model(robot_start=None):
    m = DefaultHouseMap()
    return m, mujoco.MjModel.from_xml_string(build_mjcf(m, robot_start=robot_start))

def test_robot_body_exists_at_start_tile():
    m, model = _model()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    assert bid != -1
    cx, cy = tile_center(*m.robot_start_tile)
    np.testing.assert_allclose(model.body_pos[bid], [cx, cy, ROBOT_HALFHEIGHT], atol=1e-6)

def test_robot_has_three_joints():
    _, model = _model()
    for j in ("slide_x", "slide_y", "yaw"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) != -1

def test_robot_has_ego_camera():
    _, model = _model()
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego") != -1

def test_robot_has_velocity_actuators():
    _, model = _model()
    for a in ("vx", "vy", "wz"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) != -1

def test_robot_start_override():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m, robot_start=(9, 9)))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    cx, cy = tile_center(9, 9)
    np.testing.assert_allclose(model.body_pos[bid][:2], [cx, cy], atol=1e-6)
