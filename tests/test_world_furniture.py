import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import compile_model
from homebot3d.constants import REACH_RADIUS, ROBOT_RADIUS


def _model():
    return compile_model(DefaultHouseMap())


def _geoms(model, prefix):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
            .startswith(prefix)]


def test_each_furniture_kind_present():
    m = DefaultHouseMap()
    model = _model()
    for kind, _, _ in m.furniture:
        assert _geoms(model, f"fixture_{kind}_"), f"missing furniture {kind}"


def test_furniture_geoms_all_fixture_prefixed_and_reachable():
    m = DefaultHouseMap()
    model = _model()
    margin = REACH_RADIUS - ROBOT_RADIUS
    kinds = {k for k, _, _ in m.furniture}
    for kind in kinds:
        for name in _geoms(model, f"fixture_{kind}_"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for k in (0, 1):
                surface = abs(model.geom_pos[gid][k]) + model.geom_size[gid][k]
                assert surface <= margin + 1e-9, f"{name} axis {k} = {surface}"


def test_furniture_bodies_placed_at_tiles():
    from homebot3d.world import tile_center
    m = DefaultHouseMap()
    model = _model()
    for idx, (kind, col, row) in enumerate(m.furniture):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                f"fixture_{kind}_{idx}")
        assert bid != -1
        cx, cy = tile_center(col, row)
        assert abs(model.body_pos[bid][0] - cx) < 1e-6
        assert abs(model.body_pos[bid][1] - cy) < 1e-6
