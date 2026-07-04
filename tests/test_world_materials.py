import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import compile_model


def _model():
    return compile_model(DefaultHouseMap())


def test_materials_exist():
    model = _model()
    for mat in ("floormat", "tilemat", "wallmat", "woodmat", "fabricmat"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat) != -1


def test_floor_uses_material():
    model = _model()
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert model.geom_matid[fid] != -1


def test_model_still_builds_and_steps():
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)  # must not raise
