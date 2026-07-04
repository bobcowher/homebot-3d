import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import compile_model
from homebot3d.env import HomeBot3DEnv


def _gid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


# --- human prop in the recliner ---

def test_human_geoms_present_and_collidable():
    model = compile_model(DefaultHouseMap())
    for g in ("fixture_human_head", "fixture_human_torso", "fixture_human_lap",
              "fixture_human_arm_l", "fixture_human_arm_r"):
        gid = _gid(model, g)
        assert gid != -1, f"missing {g}"
        # fixture_-prefixed and solid, so Robot.collided() treats it as an obstacle
        assert model.geom_contype[gid] == 1
        assert model.geom_conaffinity[gid] == 1


def test_drink_goal_still_reachable_with_human():
    # The human sits at the recliner (drink goal). Robot must still be able to
    # come within REACH_RADIUS of the recliner centre without being walled off.
    from homebot3d.constants import REACH_RADIUS, ROBOT_RADIUS
    m = DefaultHouseMap()
    model = compile_model(m)
    # geom_pos is relative to the recliner body (its tile centre). Every human
    # geom stays close enough to centre that the robot fits alongside to reach it.
    for name in ("fixture_human_head", "fixture_human_torso", "fixture_human_lap"):
        gid = _gid(model, name)
        px, py = model.geom_pos[gid][0], model.geom_pos[gid][1]
        sx, sy = model.geom_size[gid][0], model.geom_size[gid][1]
        assert abs(px) + sx <= REACH_RADIUS - ROBOT_RADIUS + 1e-6
        assert abs(py) + sy <= REACH_RADIUS - ROBOT_RADIUS + 1e-6


# --- rendered trash ---

def test_trash_geoms_rendered_and_non_colliding():
    model = compile_model(DefaultHouseMap(), trash=[(2, 2), (3, 3)])
    for i in (0, 1):
        gid = _gid(model, f"trash_{i}")
        assert gid != -1
        assert model.geom_contype[gid] == 0
        assert model.geom_conaffinity[gid] == 0


def test_no_trash_geoms_when_none():
    model = compile_model(DefaultHouseMap())
    assert _gid(model, "trash_0") == -1


def test_env_renders_trash_matching_task_positions():
    env = HomeBot3DEnv(goals=("trash",), n_trash=2)
    env.reset_world(seed=0)
    assert len(env._tasks.trash_positions) == 2
    for i in range(2):
        assert _gid(env.model, f"trash_{i}") != -1
    # no drink/package fixtures targeted as goals here, but trash geoms exist
    env.close()


def test_env_no_trash_geoms_without_trash_goal():
    env = HomeBot3DEnv(goals=("drink",))
    env.reset_world(seed=0)
    assert env._tasks.trash_positions == []
    assert _gid(env.model, "trash_0") == -1
    env.close()
