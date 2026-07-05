"""The front-door yard: lawn, stoop, open door leaf, collidable gate, and the
doorstep parcel that shows until the robot picks the package up."""
import mujoco
from homebot3d.maps import DefaultHouseMap, LAWN
from homebot3d.world import compile_model
from homebot3d.constants import WALL_HEIGHT
from homebot3d.env import HomeBot3DEnv
from tests.test_pickup_deliver import _teleport


def _gid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _geoms_with_prefix(model, prefix):
    return [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
            .startswith(prefix)]


# --- yard scenery (visual only) ---

def test_yard_scenery_present_and_non_colliding():
    model = compile_model(DefaultHouseMap())
    for name in ("yard_lawn", "yard_stoop", "yard_door_leaf"):
        gid = _gid(model, name)
        assert gid != -1, f"missing {name}"
        # yard_-prefixed scenery is view-only, so it must not collide.
        assert model.geom_contype[gid] == 0
        assert model.geom_conaffinity[gid] == 0


# --- the gate keeps the robot inside (view-only lawn) ---

def test_gate_is_collidable_low_barrier():
    model = compile_model(DefaultHouseMap())
    gid = _gid(model, "wall_gate")
    assert gid != -1, "missing wall_gate"
    # wall_-prefixed → Robot.collided() treats it as an obstacle.
    assert model.geom_contype[gid] == 1
    assert model.geom_conaffinity[gid] == 1
    # Low enough to see the yard over, tall enough to stop the floor robot.
    assert model.geom_size[gid][2] < WALL_HEIGHT / 2


def test_lawn_tiles_never_spawn_the_robot_or_trash():
    m = DefaultHouseMap()
    for col, row in m.valid_floor_tiles():
        assert m.tiles[row, col] != LAWN, "lawn is not a walkable floor tile"


# --- doorstep parcel ---

def test_package_parcel_geoms_present_and_non_colliding():
    model = compile_model(DefaultHouseMap())
    parcel = _geoms_with_prefix(model, "package_")
    assert parcel, "doorstep parcel should exist"
    for gid in parcel:
        assert model.geom_contype[gid] == 0
        assert model.geom_conaffinity[gid] == 0


def test_parcel_shows_until_pickup_then_hides():
    env = HomeBot3DEnv(goals=("package",), n_trash=0)
    env.reset_world(seed=0)
    parcel = _geoms_with_prefix(env.model, "package_")
    assert parcel
    # Awaiting pickup → the parcel is on the stoop.
    assert all(env.model.geom_rgba[g, 3] == 1.0 for g in parcel)
    # Reach the source (the door) → package is picked up, parcel disappears.
    _teleport(env, *env._map.pickup_tiles["package"])
    env.step_physics([0.0, 0.0])
    assert all(env.model.geom_rgba[g, 3] == 0.0 for g in parcel), \
        "parcel must vanish once carried"
    env.close()


def test_no_parcel_shown_without_package_goal():
    env = HomeBot3DEnv(goals=("drink",), n_trash=0)
    env.reset_world(seed=0)
    parcel = _geoms_with_prefix(env.model, "package_")
    # Geoms exist in the model but are hidden when the package isn't a goal.
    assert all(env.model.geom_rgba[g, 3] == 0.0 for g in parcel)
    env.close()
