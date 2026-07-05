"""Guard the drive against the floor-contact stick-slip regression.

The robot has no vertical DOF, so the floor plane must not collide with it; an
intermittent floor contact injects friction that makes forward speed surge and
stall. These tests pin the floor as non-colliding and confirm a straight drive
never touches it while still moving."""
import math
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import compile_model
from homebot3d.env import HomeBot3DEnv


def test_floor_is_non_colliding():
    model = compile_model(DefaultHouseMap())
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert model.geom_contype[fid] == 0
    assert model.geom_conaffinity[fid] == 0


def test_forward_drive_never_contacts_floor_and_moves():
    env = HomeBot3DEnv(goals=("drink",), max_steps=10**6)
    env.reset_world(seed=0)
    r = env._robot
    fid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    x0, y0 = r.x, r.y
    floor_contacts = 0
    for _ in range(60):                      # short drive: stays clear of walls
        env.step_physics([1.0, 0.0])
        d = env.data
        for k in range(d.ncon):
            c = d.contact[k]
            if r._body_geom in (c.geom1, c.geom2) and fid in (c.geom1, c.geom2):
                floor_contacts += 1
    assert floor_contacts == 0, "robot must never contact the floor plane"
    assert math.hypot(r.x - x0, r.y - y0) > 0.2, "robot should have driven forward"
    env.close()
