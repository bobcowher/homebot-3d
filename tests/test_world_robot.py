import numpy as np
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import compile_model, tile_center
from homebot3d.constants import ROBOT_HALFHEIGHT, CAMERA_HEIGHT

def _model(robot_start=None):
    m = DefaultHouseMap()
    return m, compile_model(m, robot_start=robot_start)

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
    model = compile_model(m, robot_start=(9, 9))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    cx, cy = tile_center(9, 9)
    np.testing.assert_allclose(model.body_pos[bid][:2], [cx, cy], atol=1e-6)

def test_ego_camera_at_sensor_height():
    _, model = _model()
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    # cam_pos is relative to the robot body origin, which sits at ROBOT_HALFHEIGHT.
    world_z = model.cam_pos[cid][2] + ROBOT_HALFHEIGHT
    np.testing.assert_allclose(world_z, CAMERA_HEIGHT, atol=1e-6)

def test_robot_has_visual_detail_geoms():
    _, model = _model()
    for g in ("robot_torso", "robot_mast", "robot_head",
              "robot_wheel_l", "robot_wheel_r"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) != -1

def test_visual_geoms_do_not_collide():
    _, model = _model()
    for g in ("robot_torso", "robot_mast", "robot_head", "robot_wedge",
              "robot_wheel_l", "robot_wheel_r", "robot_caster"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        assert model.geom_contype[gid] == 0
        assert model.geom_conaffinity[gid] == 0
