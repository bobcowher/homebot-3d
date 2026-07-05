import mujoco
from homebot3d.env import HomeBot3DEnv
from homebot3d.world import tile_center


def _teleport(env, col, row):
    """Place the robot's body at a tile centre and refresh derived state.

    slide_x / slide_y are displacement joints relative to the body's compiled
    home tile, so the world target must be offset by that home position.
    """
    m, d = env.model, env.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot")
    hx, hy = float(m.body_pos[bid, 0]), float(m.body_pos[bid, 1])
    ax = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "slide_x")]
    ay = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "slide_y")]
    x, y = tile_center(col, row)
    d.qpos[ax] = x - hx
    d.qpos[ay] = y - hy
    mujoco.mj_forward(m, d)
    return x, y


def _cargo_alpha(env, name):
    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return float(env.model.geom_rgba[gid, 3])


def test_drink_pickup_then_deliver_rewards_and_cargo_toggle():
    env = HomeBot3DEnv(goals=("drink",), n_trash=0)
    env.reset_world(seed=0)
    m = env._map

    # Before pickup the objective is the source (fridge) and the cup is hidden.
    assert env._goal_xy() == tile_center(*m.pickup_tiles["drink"])
    assert _cargo_alpha(env, "cargo_cup") == 0.0

    # Reach the source → +1, carrying, cup visible, objective flips to target.
    _teleport(env, *m.pickup_tiles["drink"])
    reward, terminated, _, _ = env.step_physics([0.0, 0.0])
    assert reward == 1.0
    assert not terminated
    assert _cargo_alpha(env, "cargo_cup") == 1.0
    assert _cargo_alpha(env, "cargo_cup_arm") == 1.0   # the holding arm shows too
    assert env._goal_xy() == tile_center(*m.dropoff_tiles["drink"])

    # Reach the target → +1, delivered, cup+arm hidden again, done.
    _teleport(env, *m.dropoff_tiles["drink"])
    reward, terminated, _, _ = env.step_physics([0.0, 0.0])
    assert reward == 1.0
    assert terminated
    assert _cargo_alpha(env, "cargo_cup") == 0.0
    assert _cargo_alpha(env, "cargo_cup_arm") == 0.0
    env.close()


def test_package_pickup_then_deliver():
    env = HomeBot3DEnv(goals=("package",), n_trash=0)
    env.reset_world(seed=0)
    m = env._map

    _teleport(env, *m.pickup_tiles["package"])
    reward, _, _, _ = env.step_physics([0.0, 0.0])
    assert reward == 1.0
    assert _cargo_alpha(env, "cargo_box") == 1.0

    _teleport(env, *m.dropoff_tiles["package"])
    reward, terminated, _, _ = env.step_physics([0.0, 0.0])
    assert reward == 1.0
    assert terminated
    assert _cargo_alpha(env, "cargo_box") == 0.0
    env.close()


def test_collected_trash_pile_is_hidden():
    env = HomeBot3DEnv(goals=("trash",), n_trash=2)
    env.reset_world(seed=0)
    tile = tuple(env._tasks.trash_positions[0])
    gids = env._trash_gids[tile]
    assert gids and all(env.model.geom_rgba[g, 3] == 1.0 for g in gids)  # visible first
    _teleport(env, *tile)
    reward, _, _, _ = env.step_physics([0.0, 0.0])
    assert reward == 1.0
    assert tile not in set(map(tuple, env._tasks.trash_positions))
    assert all(env.model.geom_rgba[g, 3] == 0.0 for g in gids), "collected pile must hide"
    env.close()


def test_reset_clears_cargo_alpha():
    env = HomeBot3DEnv(goals=("drink",), n_trash=0)
    env.reset_world(seed=0)
    # Force a carried state, then reset should clear the visual.
    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cargo_cup")
    env.model.geom_rgba[gid, 3] = 1.0
    env.reset_world(seed=1)
    assert _cargo_alpha(env, "cargo_cup") == 0.0
    env.close()
