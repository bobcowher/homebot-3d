import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import texture_assets, compile_model

EXPECTED = {"floor_wood.png", "floor_tile.png", "wall_paint.png",
            "wood.png", "fabric.png"}


def test_texture_assets_returns_named_bytes():
    assets = texture_assets()
    assert set(assets) == EXPECTED
    for name, data in assets.items():
        assert isinstance(data, bytes) and len(data) > 0


def test_compile_model_builds_and_steps():
    model = compile_model(DefaultHouseMap())
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)                     # must not raise


def test_expected_materials_assigned():
    model = compile_model(DefaultHouseMap())
    for mat in ("floormat", "tilemat", "wallmat", "woodmat", "fabricmat"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat) != -1
