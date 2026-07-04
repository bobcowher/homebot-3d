import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import compile_model
from homebot3d.env import HomeBot3DEnv


def _gid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _geoms_with_prefix(model, prefix):
    return [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
            .startswith(prefix)]


# --- human prop in the recliner ---

def test_human_geoms_present_and_collidable():
    model = compile_model(DefaultHouseMap())
    for g in ("fixture_human_head", "fixture_human_torso", "fixture_human_thigh_l",
              "fixture_human_uarm_l", "fixture_human_foot_l"):
        gid = _gid(model, g)
        assert gid != -1, f"missing {g}"
        # fixture_-prefixed and solid, so Robot.collided() treats it as an obstacle
        assert model.geom_contype[gid] == 1
        assert model.geom_conaffinity[gid] == 1


def test_drink_goal_still_reachable_with_human():
    # The human sits at the recliner (drink goal). Every human geom's centre must
    # stay within REACH_RADIUS - ROBOT_RADIUS of the recliner centre so the human
    # never sprawls far enough to wall the robot off from the goal.
    from homebot3d.constants import REACH_RADIUS, ROBOT_RADIUS
    model = compile_model(DefaultHouseMap())
    limit = REACH_RADIUS - ROBOT_RADIUS      # 0.57 m
    for g in _geoms_with_prefix(model, "fixture_human_"):
        px, py = model.geom_pos[g][0], model.geom_pos[g][1]
        assert (px * px + py * py) ** 0.5 <= limit, \
            f"human geom {g} too far from centre"


# --- rendered trash ---

def test_trash_piles_rendered_and_non_colliding():
    model = compile_model(DefaultHouseMap(), trash=[(2, 2), (3, 3)])
    for i in (0, 1):
        pile = _geoms_with_prefix(model, f"trash_{i}_")
        assert len(pile) > 1, f"trash pile {i} should have multiple pieces"
        for gid in pile:
            assert model.geom_contype[gid] == 0
            assert model.geom_conaffinity[gid] == 0


def test_no_trash_geoms_when_none():
    model = compile_model(DefaultHouseMap())
    assert _geoms_with_prefix(model, "trash_") == []


def test_env_renders_trash_matching_task_positions():
    env = HomeBot3DEnv(goals=("trash",), n_trash=2)
    env.reset_world(seed=0)
    assert len(env._tasks.trash_positions) == 2
    for i in range(2):
        assert _geoms_with_prefix(env.model, f"trash_{i}_")
    env.close()


def test_env_no_trash_geoms_without_trash_goal():
    env = HomeBot3DEnv(goals=("drink",))
    env.reset_world(seed=0)
    assert env._tasks.trash_positions == []
    assert _geoms_with_prefix(env.model, "trash_") == []
    env.close()
