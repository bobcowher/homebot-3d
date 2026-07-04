# HomeBot-3D Embodiment, House Dressing & POV Teleop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give HomeBot-3D a recognizable primitive robot body, a house that reads as a house, and a custom GLFW teleop driver with true hold-to-move controls and a first-person POV camera.

**Architecture:** All world geometry is still generated as MJCF strings from the 2D grid map (`world.py`). The robot gains visual detail as massless, non-colliding child geoms so drive dynamics and collision behavior are unchanged. Materials/textures use MuJoCo built-in procedural generators (no asset files). A render-less physics seam (`reset_world` / `step_physics`) on the env lets a GLFW/GLX teleop window run without touching the EGL offscreen renderer. `play.py` becomes a custom GLFW app.

**Tech Stack:** Python 3.11, MuJoCo 3.10 (`mujoco`), `glfw` 3.4 (already installed), NumPy, Gymnasium, pytest. conda env `homebot-3d`.

## Global Constraints

- No new third-party dependencies (`glfw` and `OpenGL` already present via mujoco).
- Determinism preserved: materials/textures/lighting/visual geoms must not affect physics or RNG.
- Drive interface unchanged: action is `[linear, angular]` in `[-1, 1]`; joints `slide_x`/`slide_y`/`yaw`; actuators `vx`/`vy`/`wz`.
- The collision geom `robot_body` (name + default contype/conaffinity + default density) is unchanged so the `kv=20` actuator tuning and `Robot.collided()` behavior stay valid.
- All furniture collision geoms are named with the `fixture_` prefix so `Robot.collided()` still detects them.
- Furniture half-extents in x/y stay ≤ `0.6 * TILE` (0.3 m) so a goal at a fixture center remains reachable within `REACH_RADIUS` (0.75 m).
- Run tests with: `conda run -n homebot-3d python -m pytest -q`
- The MuJoCo GL env default is `egl` (set in `homebot3d/__init__.py`); teleop uses its own GLFW window context and must not create the EGL `Camera`.

---

### Task 1: Robot dimension constants

**Files:**
- Modify: `homebot3d/constants.py`
- Test: `tests/test_smoke.py` (add one assertion) — or a new `tests/test_constants.py`

**Interfaces:**
- Produces: module constants `CAMERA_HEIGHT`, `ROBOT_BODY_HALF`, `ROBOT_BODY_HALFHEIGHT`, `WHEEL_RADIUS` (all `float`, metres).

- [ ] **Step 1: Write the failing test**

Create `tests/test_constants.py`:

```python
from homebot3d import constants


def test_robot_embodiment_constants_present():
    assert constants.CAMERA_HEIGHT == 0.6
    assert constants.ROBOT_BODY_HALF == 0.12
    assert constants.ROBOT_BODY_HALFHEIGHT == 0.09
    assert constants.WHEEL_RADIUS == 0.06
    # Camera must sit above the base cylinder top.
    assert constants.CAMERA_HEIGHT > 2 * constants.ROBOT_HALFHEIGHT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n homebot-3d python -m pytest tests/test_constants.py -q`
Expected: FAIL with `AttributeError: module 'homebot3d.constants' has no attribute 'CAMERA_HEIGHT'`

- [ ] **Step 3: Add the constants**

Append to `homebot3d/constants.py`:

```python
CAMERA_HEIGHT = 0.6         # metres above floor — ego camera / sensor head height
ROBOT_BODY_HALF = 0.12      # torso box half-extent in x and y (metres)
ROBOT_BODY_HALFHEIGHT = 0.09  # torso box half-height (metres)
WHEEL_RADIUS = 0.06         # cosmetic wheel radius (metres)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n homebot-3d python -m pytest tests/test_constants.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add homebot3d/constants.py tests/test_constants.py
git commit -m "feat(constants): robot embodiment dimensions"
```

---

### Task 2: Robot embodiment in the MJCF

**Files:**
- Modify: `homebot3d/world.py` (function `_robot_body`)
- Test: `tests/test_world_robot.py` (add camera-height test; existing tests must stay green)

**Interfaces:**
- Consumes: `CAMERA_HEIGHT`, `ROBOT_BODY_HALF`, `ROBOT_BODY_HALFHEIGHT`, `WHEEL_RADIUS`, `ROBOT_RADIUS`, `ROBOT_HALFHEIGHT` from `constants`.
- Produces: `robot` body with unchanged joints/actuators/collision geom `robot_body`; added massless non-colliding visual geoms `robot_torso`, `robot_mast`, `robot_head`, `robot_wedge`, `robot_wheel_l`, `robot_wheel_r`, `robot_caster`; `ego` camera at world height `CAMERA_HEIGHT`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_world_robot.py`:

```python
from homebot3d.constants import CAMERA_HEIGHT

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n homebot-3d python -m pytest tests/test_world_robot.py -q`
Expected: FAIL (`test_ego_camera_at_sensor_height` — camera currently at ~0.22 m; visual-geom tests — geoms absent)

- [ ] **Step 3: Replace `_robot_body`**

In `homebot3d/world.py`, update the import line and replace the whole `_robot_body` function:

```python
from homebot3d.constants import (
    TILE, WALL_HEIGHT, ROBOT_RADIUS, ROBOT_HALFHEIGHT,
    CAMERA_HEIGHT, ROBOT_BODY_HALF, ROBOT_BODY_HALFHEIGHT, WHEEL_RADIUS,
)


def _robot_body(map: Map, robot_start) -> str:
    col, row = robot_start if robot_start is not None else map.robot_start_tile
    cx, cy = tile_center(col, row)
    z = ROBOT_HALFHEIGHT

    base_top = ROBOT_HALFHEIGHT                      # local z of base cylinder top
    torso_z = base_top + ROBOT_BODY_HALFHEIGHT       # torso box centre (local z)
    head_z = CAMERA_HEIGHT - ROBOT_HALFHEIGHT        # camera / head centre (local z)
    mast_bottom = torso_z + ROBOT_BODY_HALFHEIGHT
    mast_cz = (mast_bottom + head_z) / 2
    mast_hz = max((head_z - mast_bottom) / 2, 0.01)
    wheel_z = -ROBOT_HALFHEIGHT + WHEEL_RADIUS
    fwd = ROBOT_RADIUS - 0.03
    vis = 'contype="0" conaffinity="0" density="0"'  # visual-only: no mass, no collision

    return f"""
    <body name="robot" pos="{cx} {cy} {z}">
      <joint name="slide_x" type="slide" axis="1 0 0"/>
      <joint name="slide_y" type="slide" axis="0 1 0"/>
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom name="robot_body" type="cylinder"
            size="{ROBOT_RADIUS} {ROBOT_HALFHEIGHT}" rgba="0.25 0.5 0.75 1"/>
      <geom name="robot_torso" type="box" {vis}
            size="{ROBOT_BODY_HALF} {ROBOT_BODY_HALF} {ROBOT_BODY_HALFHEIGHT}"
            pos="0 0 {torso_z}" rgba="0.9 0.9 0.92 1"/>
      <geom name="robot_mast" type="box" {vis}
            size="0.02 0.02 {mast_hz}" pos="0 0 {mast_cz}" rgba="0.3 0.3 0.32 1"/>
      <geom name="robot_head" type="box" {vis}
            size="0.05 0.06 0.04" pos="0 0 {head_z}" rgba="0.15 0.15 0.17 1"/>
      <geom name="robot_wedge" type="box" {vis}
            size="0.05 0.03 0.03" pos="{fwd} 0 {base_top}" rgba="0.95 0.55 0.1 1"/>
      <geom name="robot_wheel_l" type="cylinder" {vis} zaxis="0 1 0"
            size="{WHEEL_RADIUS} 0.02" pos="0 {ROBOT_RADIUS} {wheel_z}"
            rgba="0.1 0.1 0.1 1"/>
      <geom name="robot_wheel_r" type="cylinder" {vis} zaxis="0 1 0"
            size="{WHEEL_RADIUS} 0.02" pos="0 {-ROBOT_RADIUS} {wheel_z}"
            rgba="0.1 0.1 0.1 1"/>
      <geom name="robot_caster" type="sphere" {vis} size="0.03"
            pos="{fwd} 0 {-ROBOT_HALFHEIGHT + 0.03}" rgba="0.1 0.1 0.1 1"/>
      <camera name="ego" pos="{ROBOT_RADIUS} 0 {head_z}" xyaxes="0 -1 0 0 0 1"/>
    </body>"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n homebot-3d python -m pytest tests/test_world_robot.py tests/test_robot.py tests/test_sensors.py -q`
Expected: PASS (new camera/visual tests pass; existing robot/collision/sensor tests still green — proves dynamics & collision unchanged)

- [ ] **Step 5: Visual check + commit**

```bash
PYTHONPATH=. MUJOCO_GL=egl conda run -n homebot-3d python scripts/render_dev.py --goal drink --steps 0 --out _dev_render
git add homebot3d/world.py tests/test_world_robot.py
git commit -m "feat(world): recognizable primitive robot body + raised camera"
```
(Reviewer/author inspects `_dev_render/overview_000.png` and `ego_000.png` — robot should read as a mobile robot; POV should be at desk height, not floor.)

---

### Task 3: House materials, skybox & lighting

**Files:**
- Modify: `homebot3d/world.py` (add `_asset_block`, apply materials to floor/walls, replace light + add headlight)
- Test: `tests/test_world_materials.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `<asset>` with textures `sky`, `floortex` and materials `floormat`, `wallmat`; floor geom uses `material="floormat"`; wall geoms use `material="wallmat"`; a directional light + headlight ambient.

- [ ] **Step 1: Write the failing test**

Create `tests/test_world_materials.py`:

```python
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import build_mjcf


def _model():
    return mujoco.MjModel.from_xml_string(build_mjcf(DefaultHouseMap()))


def test_materials_exist():
    model = _model()
    for mat in ("floormat", "wallmat"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat) != -1


def test_floor_uses_material():
    model = _model()
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert model.geom_matid[fid] != -1


def test_model_still_builds_and_steps():
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n homebot-3d python -m pytest tests/test_world_materials.py -q`
Expected: FAIL (`test_materials_exist` — materials absent)

- [ ] **Step 3: Add assets, apply materials, fix lighting**

In `homebot3d/world.py` add the helper:

```python
def _asset_block() -> str:
    return """
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.55 0.7 0.9" rgb2="0.1 0.12 0.2" width="256" height="256"/>
    <texture name="floortex" type="2d" builtin="checker"
             rgb1="0.82 0.80 0.76" rgb2="0.70 0.68 0.64" width="512" height="512"/>
    <material name="floormat" texture="floortex" texrepeat="8 8" reflectance="0.1"/>
    <material name="wallmat" rgba="0.86 0.86 0.88 1" reflectance="0.05"/>
  </asset>"""
```

In `_wall_geoms`, change the wall geom string to use the material instead of rgba:

```python
                parts.append(
                    f'<geom name="wall_{r}_{c}" type="box" '
                    f'size="{hx} {hy} {hz}" pos="{cx} {cy} {hz}" '
                    f'material="wallmat"/>'
                )
```

In `build_mjcf`, insert `_asset_block()` after the `<mujoco ...>`/`<option>` lines, add a `<visual>` headlight, replace the point light with a directional one, and give the floor a material:

```python
def build_mjcf(map: Map, robot_start=None) -> str:
    rows, cols = map.tiles.shape
    fx = cols * TILE
    fy = rows * TILE
    return f"""
<mujoco model="homebot3d">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0 0 0"/>
  </visual>
{_asset_block()}
  <worldbody>
    <light directional="true" pos="0 0 5" dir="0.2 0.3 -1" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" pos="{fx/2} {fy/2} 0"
          size="{fx/2} {fy/2} 0.1" material="floormat"/>
{_wall_geoms(map)}
{_fixture_bodies(map)}
{_robot_body(map, robot_start)}
  </worldbody>
  <actuator>
    <velocity name="vx" joint="slide_x" kv="20" ctrlrange="-2 2"/>
    <velocity name="vy" joint="slide_y" kv="20" ctrlrange="-2 2"/>
    <velocity name="wz" joint="yaw" kv="2" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n homebot-3d python -m pytest tests/test_world_materials.py tests/test_world_walls.py -q`
Expected: PASS

- [ ] **Step 5: Visual check + commit**

```bash
PYTHONPATH=. MUJOCO_GL=egl conda run -n homebot-3d python scripts/render_dev.py --goal drink --steps 0 --out _dev_render
git add homebot3d/world.py tests/test_world_materials.py
git commit -m "feat(world): procedural materials, skybox and even lighting"
```
(Inspect `_dev_render/overview_000.png` — floor checkered, no blown-out hotspot, walls read cleanly.)

---

### Task 4: Furniture-shaped fixtures

**Files:**
- Modify: `homebot3d/world.py` (function `_fixture_bodies`)
- Test: `tests/test_world_fixtures.py` (add furniture-geom + collision-prefix tests; existing tests stay green)

**Interfaces:**
- Consumes: `TILE`, `tile_center`.
- Produces: each `fixture_{name}` body sits at `(cx, cy, 0)`; furniture built from multiple geoms all named `fixture_{name}_*`; all furniture geom half-extents in x/y ≤ 0.3 m.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_world_fixtures.py`:

```python
def test_fixtures_have_multiple_geoms_all_prefixed():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    # recliner is modelled as seat + back → at least 2 geoms, all fixture_-prefixed
    recliner_geoms = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        for gid in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
        .startswith("fixture_recliner")
    ]
    assert len(recliner_geoms) >= 2


def test_furniture_geoms_are_reachable_sized():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("fixture_"):
            assert model.geom_size[gid][0] <= 0.3 + 1e-9
            assert model.geom_size[gid][1] <= 0.3 + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n homebot-3d python -m pytest tests/test_world_fixtures.py -q`
Expected: FAIL (`test_fixtures_have_multiple_geoms_all_prefixed` — recliner is a single geom today)

- [ ] **Step 3: Replace `_fixture_bodies`**

In `homebot3d/world.py`, replace the whole `_fixture_bodies` function:

```python
def _furniture_geoms(name: str) -> str:
    """Multi-geom procedural furniture, all geoms named fixture_{name}_*.

    Positions are relative to the fixture body origin at floor level (z=0).
    All x/y half-extents stay <= 0.3 m so a goal at the fixture centre is
    reachable within REACH_RADIUS.
    """
    if name == "fridge":
        return (
            f'<geom name="fixture_fridge_body" type="box" size="0.22 0.22 0.45" '
            f'pos="0 0 0.45" rgba="0.72 0.78 0.85 1"/>'
            f'<geom name="fixture_fridge_door" type="box" size="0.02 0.20 0.4" '
            f'pos="0.22 0 0.5" rgba="0.6 0.66 0.74 1"/>'
        )
    if name == "recliner":
        return (
            f'<geom name="fixture_recliner_seat" type="box" size="0.28 0.28 0.12" '
            f'pos="0 0 0.12" rgba="0.5 0.3 0.2 1"/>'
            f'<geom name="fixture_recliner_back" type="box" size="0.28 0.06 0.20" '
            f'pos="0 -0.22 0.32" rgba="0.45 0.27 0.18 1"/>'
        )
    if name == "door":
        return (
            f'<geom name="fixture_door_panel" type="box" size="0.28 0.05 0.5" '
            f'pos="0 0 0.5" rgba="0.32 0.22 0.12 1"/>'
            f'<geom name="fixture_door_knob" type="sphere" size="0.03" '
            f'pos="0.18 0.06 0.5" rgba="0.85 0.7 0.2 1"/>'
        )
    # Fallback: a plain box for any other fixture name.
    return (
        f'<geom name="fixture_{name}_box" type="box" size="0.28 0.28 0.3" '
        f'pos="0 0 0.3" rgba="0.5 0.5 0.5 1"/>'
    )


def _fixture_bodies(map: Map) -> str:
    parts = []
    for name, (col, row) in map.fixtures.items():
        cx, cy = tile_center(col, row)
        parts.append(
            f'<body name="fixture_{name}" pos="{cx} {cy} 0">'
            f'{_furniture_geoms(name)}</body>'
        )
    return "\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n homebot-3d python -m pytest tests/test_world_fixtures.py tests/test_robot.py tests/test_integration.py -q`
Expected: PASS (furniture tests pass; collision + integration still green — furniture blocks but goals remain reachable)

- [ ] **Step 5: Visual check + commit**

```bash
PYTHONPATH=. MUJOCO_GL=egl conda run -n homebot-3d python scripts/render_dev.py --goal drink --steps 800 --every 200 --out _dev_render
git add homebot3d/world.py tests/test_world_fixtures.py
git commit -m "feat(world): furniture-shaped fixtures"
```
(Inspect frames — furniture reads as fridge/recliner/door; confirm the drive still reaches the recliner and scores `+1`.)

---

### Task 5: Render-less physics seam on the env

**Files:**
- Modify: `homebot3d/env.py` (add `reset_world`, `step_physics`; refactor `reset`, `step`)
- Test: `tests/test_env.py` (add render-less tests; existing env/goal-env/integration tests stay green)

**Interfaces:**
- Consumes: existing env internals.
- Produces: `HomeBot3DEnv.reset_world(seed=None) -> dict` (rebuilds model/data/robot/tasks, no camera, returns info); `HomeBot3DEnv.step_physics(action) -> (reward: float, terminated: bool, truncated: bool, info: dict)` (no obs render). Gym `reset`/`step` behavior unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_env.py`:

```python
def test_reset_world_creates_no_camera():
    env = HomeBot3DEnv(goals=("drink",))
    info = env.reset_world(seed=0)
    assert env._camera is None
    assert "privileged" in info


def test_step_physics_no_camera_and_matches_types():
    env = HomeBot3DEnv(goals=("drink",))
    env.reset_world(seed=0)
    reward, term, trunc, info = env.step_physics([0.0, 0.0])
    assert env._camera is None
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)


def test_gym_step_still_returns_obs():
    env = HomeBot3DEnv(goals=("drink",))
    obs, info = env.reset(seed=0)
    assert obs.shape == (84, 84, 3)
    obs2, r, term, trunc, info2 = env.step([0.0, 0.0])
    assert obs2.shape == (84, 84, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n homebot-3d python -m pytest tests/test_env.py -q`
Expected: FAIL (`reset_world`/`step_physics` don't exist)

- [ ] **Step 3: Refactor env**

In `homebot3d/env.py`, replace the `reset` and `step` methods of `HomeBot3DEnv` with:

```python
    def reset_world(self, seed=None):
        rng = np.random.default_rng(seed)
        start = self._sample_start_tile(rng)
        xml = build_mjcf(self._map, robot_start=start)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._robot = Robot(self.model, self.data)
        self._tasks.reset(self._map, self.n_trash, rng)
        self._steps = 0
        return self._info()

    def step_physics(self, action):
        self._robot.apply(np.asarray(action, dtype=np.float32))
        mujoco.mj_step(self.model, self.data)
        self._steps += 1
        reward = float(self._tasks.step(self._robot))
        terminated = bool(self._tasks.is_done())
        truncated = bool(self._steps >= self.max_steps)
        return reward, terminated, truncated, self._info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info = self.reset_world(seed)
        if self._camera is not None:
            self._camera.close()
        self._camera = Camera(self.model, width=self.width, height=self.height)
        return self._obs(), info

    def step(self, action):
        reward, terminated, truncated, info = self.step_physics(action)
        return self._obs(), reward, terminated, truncated, info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n homebot-3d python -m pytest tests/test_env.py tests/test_goal_env.py tests/test_integration.py -q`
Expected: PASS (new render-less tests pass; Gym API + HER + integration unchanged)

- [ ] **Step 5: Commit**

```bash
git add homebot3d/env.py tests/test_env.py
git commit -m "feat(env): render-less reset_world/step_physics seam"
```

---

### Task 6: Teleop input mapping (pure, testable)

**Files:**
- Create: `homebot3d/teleop_input.py`
- Test: `tests/test_teleop_input.py`

**Interfaces:**
- Produces: `keys_to_action(held: set[str]) -> np.ndarray` — maps a set of lowercase key chars (`{"w","a","s","d"}`) to `np.array([linear, angular], dtype=float32)`, each component clamped to `[-1, 1]`. `w`=+lin, `s`=-lin, `d`=+ang, `a`=-ang.

- [ ] **Step 1: Write the failing test**

Create `tests/test_teleop_input.py`:

```python
import numpy as np
from homebot3d.teleop_input import keys_to_action


def test_empty_is_zero():
    np.testing.assert_array_equal(keys_to_action(set()), [0.0, 0.0])


def test_single_keys():
    np.testing.assert_array_equal(keys_to_action({"w"}), [1.0, 0.0])
    np.testing.assert_array_equal(keys_to_action({"s"}), [-1.0, 0.0])
    np.testing.assert_array_equal(keys_to_action({"d"}), [0.0, 1.0])
    np.testing.assert_array_equal(keys_to_action({"a"}), [0.0, -1.0])


def test_opposing_keys_cancel():
    np.testing.assert_array_equal(keys_to_action({"w", "s"}), [0.0, 0.0])
    np.testing.assert_array_equal(keys_to_action({"a", "d"}), [0.0, 0.0])


def test_combined_drive_and_turn():
    np.testing.assert_array_equal(keys_to_action({"w", "d"}), [1.0, 1.0])


def test_returns_float32():
    assert keys_to_action({"w"}).dtype == np.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n homebot-3d python -m pytest tests/test_teleop_input.py -q`
Expected: FAIL (`ModuleNotFoundError: homebot3d.teleop_input`)

- [ ] **Step 3: Implement**

Create `homebot3d/teleop_input.py`:

```python
import numpy as np


def keys_to_action(held) -> np.ndarray:
    """Map a set of held key chars to a [linear, angular] action in [-1, 1]."""
    linear = (1.0 if "w" in held else 0.0) - (1.0 if "s" in held else 0.0)
    angular = (1.0 if "d" in held else 0.0) - (1.0 if "a" in held else 0.0)
    return np.array([np.clip(linear, -1.0, 1.0),
                     np.clip(angular, -1.0, 1.0)], dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n homebot-3d python -m pytest tests/test_teleop_input.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add homebot3d/teleop_input.py tests/test_teleop_input.py
git commit -m "feat(teleop): pure keys_to_action mapping"
```

---

### Task 7: Custom GLFW teleop driver

**Files:**
- Rewrite: `play.py`
- Test: manual (interactive GUI — no automated test) + `py_compile`

**Interfaces:**
- Consumes: `HomeBot3DEnv.reset_world`, `HomeBot3DEnv.step_physics`, `keys_to_action`, `homebot3d` gym-id registration is not needed (constructs env directly).
- Produces: an interactive driver. WASD hold-to-move, `R` reset, `V` toggle POV/overview camera, `Esc` quit.

- [ ] **Step 1: Rewrite `play.py`**

Replace the entire contents of `play.py`:

```python
"""Custom GLFW teleop for HomeBot3DEnv — first-person POV with hold-to-move.

Controls: W/S drive fwd/back (held), A/D turn (held), V toggle POV/overview,
R reset, Esc quit.

Uses its own GLFW/GLX window + MjrContext and steps physics via the env's
render-less seam (reset_world / step_physics), so it never touches the EGL
offscreen renderer used for observations.
"""
import argparse
import time
import glfw
import mujoco
from homebot3d.env import HomeBot3DEnv
from homebot3d.teleop_input import keys_to_action

_KEYMAP = {glfw.KEY_W: "w", glfw.KEY_A: "a", glfw.KEY_S: "s", glfw.KEY_D: "d"}


def _overview_camera(model, cam):
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = model.geom_pos[fid]
    cam.distance = 1.5 * model.stat.extent
    cam.azimuth = 90.0
    cam.elevation = -55.0


def _pov_camera(model, cam):
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--goals", nargs="+", default=["trash", "drink", "package"])
    p.add_argument("--map", default="default")
    p.add_argument("--random-start", action="store_true")
    args = p.parse_args()

    env = HomeBot3DEnv(goals=tuple(args.goals), map_name=args.map,
                       random_start=args.random_start)
    env.reset_world(seed=None)

    held = set()
    state = {"reset": False, "pov": True}

    def on_key(window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_R and action == glfw.PRESS:
            state["reset"] = True
        elif key == glfw.KEY_V and action == glfw.PRESS:
            state["pov"] = not state["pov"]
        elif key in _KEYMAP:
            if action == glfw.PRESS:
                held.add(_KEYMAP[key])
            elif action == glfw.RELEASE:
                held.discard(_KEYMAP[key])

    if not glfw.init():
        raise SystemExit("Failed to init GLFW")
    window = glfw.create_window(1200, 800, "HomeBot3D teleop", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0)  # no vsync; the timestep sleep below paces the loop
    glfw.set_key_callback(window, on_key)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(env.model, maxgeom=10000)
    context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def apply_camera():
        (_pov_camera if state["pov"] else _overview_camera)(env.model, cam)

    apply_camera()
    timestep = env.model.opt.timestep

    while not glfw.window_should_close(window):
        step_start = time.time()

        reward, term, trunc, _ = env.step_physics(keys_to_action(held))
        if reward > 0:
            print(f"+{reward}")

        if state["reset"] or term or trunc:
            env.reset_world(seed=None)
            # Model instance changed — rebuild scene/context and re-apply camera.
            scene = mujoco.MjvScene(env.model, maxgeom=10000)
            context = mujoco.MjrContext(env.model,
                                        mujoco.mjtFontScale.mjFONTSCALE_150)
            apply_camera()
            state["reset"] = False
            held.clear()
        else:
            apply_camera()

        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(env.model, env.data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

        dt = timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)

    glfw.terminate()
    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Compile check**

Run: `conda run -n homebot-3d python -m py_compile play.py`
Expected: no output (success)

- [ ] **Step 3: Import/smoke check (no window)**

Run: `conda run -n homebot-3d python -m pytest tests/test_teleop_input.py -q`
Expected: PASS (confirms the action mapping `play.py` depends on is intact)

- [ ] **Step 4: Manual interactive verification**

Run: `conda run -n homebot-3d python play.py`
Expected: a window opens showing first-person POV; **W** drives forward smoothly while held and stops on release; **A/D** turn; **V** toggles to the top-down overview and back; **R** resets; **Esc** quits. Driving into the recliner prints `+1.0`.

- [ ] **Step 5: Commit**

```bash
git add play.py
git commit -m "feat(play): custom GLFW POV teleop with hold-to-move"
```

---

## Self-Review Notes

- **Spec coverage:** robot embodiment (T1–T2), camera height (T2), house materials/skybox/lighting (T3), furniture-shaped fixtures (T4), render-less EGL/GLX seam (T5), teleop input (T6), GLFW POV teleop + hold-to-move + camera toggle (T7). All spec sections mapped.
- **Determinism:** visual geoms are `density=0 contype=0 conaffinity=0`; materials/lighting are visual-only; RNG path in `reset_world` matches the old `reset`. Existing determinism test in `test_env.py` guards this.
- **Type consistency:** `reset_world(seed) -> info(dict)`, `step_physics(action) -> (float, bool, bool, dict)`, `keys_to_action(set) -> np.ndarray` used identically in T5/T6/T7.
- **Collision preserved:** `robot_body` unchanged; furniture geoms all `fixture_`-prefixed (T4) so `Robot.collided()` semantics hold.
```
