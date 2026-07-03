# HomeBot-3D Navigation Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-env MuJoCo Gymnasium environment where a diff-drive robot navigates a 3D house extruded from 2D grid maps, observed through an egocentric RGB camera, rewarded for reaching task locations.

**Architecture:** Standard `mujoco` Python bindings (CPU physics + EGL GPU render). A `world.py` module compiles a 2D tile grid into an MJCF string (walls→boxes, floor→plane, fixtures→bodies, robot→diff-drive base). `env.py` wraps it as a first-class single-env Gymnasium env, with a `GoalEnv`/HER variant. A privileged seam exposes pose/depth/goal-vector for training-time supervision without entering the deployed RGB observation.

**Tech Stack:** Python 3.11+, `mujoco` (DeepMind bindings), `gymnasium`, `numpy`, `pytest`. No JAX/MJX. PyTorch only for downstream consumers (not this package).

## Global Constraints

- **Substrate:** standard `mujoco` bindings only. No `mjx`, no `jax`, no `brax`.
- **Render backend:** EGL. Set `os.environ.setdefault("MUJOCO_GL", "egl")` before importing `mujoco`, in the package `__init__.py`. Vendor-neutral — must not use CUDA-specific render paths.
- **Single-env first:** no assumption of vectorization anywhere in `env.py`.
- **Determinism:** every `reset(seed=s)` produces identical spawn and identical episode. All randomness flows through a single `np.random.Generator` seeded in `reset`.
- **Deployed observation is RGB only.** Pose/depth/goal-vector are available ONLY through the `privileged` seam, never in the returned `obs` of `HomeBot3DEnv`.
- **World units:** metres. Grid conversion constant `TILE = 0.5` (m per tile). Wall height `WALL_HEIGHT = 1.2`. Robot body radius `ROBOT_RADIUS = 0.18`, height `ROBOT_HALFHEIGHT = 0.12`.
- **Coordinate mapping:** grid `(col, row)` → world `(x, y) = (col * TILE, row * TILE)`, z up. Tile centre offset uses the tile's own index (cell centre = `(col + 0.5) * TILE`, `(row + 0.5) * TILE`) — see Task 3, applied consistently everywhere.
- **Action:** continuous `Box([-1,-1], [1,1])` = `[linear, angular]`, scaled by `MAX_LIN = 1.0` m/s and `MAX_ANG = 2.0` rad/s. No discrete mode.
- **Reach radius:** `REACH_RADIUS = 0.75` (m) — a goal counts as reached when the robot body centre is within this planar distance of the target.
- Commit after every task. Commit messages: no `Co-Authored-By` trailer.

---

## File Structure

| File | Responsibility |
|---|---|
| `homebot3d/__init__.py` | Set `MUJOCO_GL=egl`; export `HomeBot3DEnv`, `HomeBot3DGoalEnv`; register Gymnasium IDs. |
| `homebot3d/maps.py` | Grid map classes + `MAP_REGISTRY` (ported/trimmed from 2D). |
| `homebot3d/world.py` | `build_mjcf(map, robot_start, fixture_present)` → MJCF XML string. The keystone. |
| `homebot3d/robot.py` | `Robot` wrapper: apply `[linear, angular]`, read pose/heading, detect collision. |
| `homebot3d/sensors.py` | `Camera` (egocentric RGB via `mujoco.Renderer`) + `privileged()` (pose/depth/goal-vec). |
| `homebot3d/goals.py` | Goal registry + `goal_to_coordinates` (world metres). |
| `homebot3d/tasks.py` | `TaskManager`: reach detection, reward, termination, goal gating. |
| `homebot3d/env.py` | `_HomeBotCore`, `HomeBot3DEnv`, `HomeBot3DGoalEnv`. |
| `play.py` | Human keyboard play wrapper. |
| `tests/…` | One test file per module. |

---

## Task 1: Project scaffold + MuJoCo/EGL smoke test

De-risks the render path on the actual GPU (Blackwell) before any real code is built.

**Files:**
- Create: `pyproject.toml`, `setup.py`, `requirements.txt`, `homebot3d/__init__.py`, `homebot3d/constants.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces: constants `TILE, WALL_HEIGHT, ROBOT_RADIUS, ROBOT_HALFHEIGHT, MAX_LIN, MAX_ANG, REACH_RADIUS` in `homebot3d/constants.py`.

- [ ] **Step 1: Write `homebot3d/constants.py`**

```python
TILE = 0.5              # metres per grid tile
WALL_HEIGHT = 1.2       # metres
ROBOT_RADIUS = 0.18     # metres (cylinder radius)
ROBOT_HALFHEIGHT = 0.12 # metres (cylinder half-height)
MAX_LIN = 1.0           # m/s at action = 1.0
MAX_ANG = 2.0           # rad/s at action = 1.0
REACH_RADIUS = 0.75     # metres — goal reached when body centre within this planar distance
```

- [ ] **Step 2: Write `homebot3d/__init__.py`**

```python
import os
os.environ.setdefault("MUJOCO_GL", "egl")  # vendor-neutral EGL render path
```

- [ ] **Step 3: Write packaging files**

`requirements.txt`:
```
mujoco>=3.1
gymnasium>=0.29
numpy>=1.24
pytest>=7.0
```

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
```

`setup.py`:
```python
from setuptools import setup, find_packages
setup(name="gym-homebot-3d", version="0.1.0", packages=find_packages(),
      install_requires=["mujoco>=3.1", "gymnasium>=0.29", "numpy>=1.24"])
```

- [ ] **Step 4: Write the failing smoke test**

```python
# tests/test_smoke.py
import numpy as np
import homebot3d  # sets MUJOCO_GL
import mujoco

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="box" pos="0 0 0.5">
      <geom type="box" size="0.25 0.25 0.25" rgba="1 0 0 1"/>
    </body>
    <camera name="cam" pos="0 -3 1.5" xyaxes="1 0 0 0 0.5 1"/>
  </worldbody>
</mujoco>
"""

def test_mujoco_egl_renders_a_frame():
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=84, width=84)
    renderer.update_scene(data, camera="cam")
    frame = renderer.render()
    assert frame.shape == (84, 84, 3)
    assert frame.dtype == np.uint8
    assert frame.sum() > 0          # not an all-black frame
    renderer.close()
```

- [ ] **Step 5: Run it — expect PASS (this is the environment probe, not TDD)**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: PASS. If it fails with an EGL/GL error, STOP — the render backend must be fixed before any further task (try `MUJOCO_GL=egl` env, verify `libEGL`, check the Blackwell driver). This is the §11 render-path risk; catch it here.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml setup.py requirements.txt homebot3d/ tests/test_smoke.py
git commit -m "chore: scaffold gym-homebot-3d + MuJoCo EGL smoke test"
```

---

## Task 2: Maps module

**Files:**
- Create: `homebot3d/maps.py`
- Test: `tests/test_maps.py`

**Interfaces:**
- Produces:
  - Constants `FLOOR = 0`, `WALL = 1`.
  - `class Map` with attrs: `tiles: np.ndarray (rows, cols)`, `fixtures: dict[str, tuple[int,int]]` as `(col, row)`, `robot_start_tile: tuple[int,int]` as `(col, row)`, `door_tiles: list[tuple[int,int]]`.
  - Method `valid_floor_tiles() -> list[tuple[int,int]]` returning `(col, row)` FLOOR tiles.
  - Method `spawn_trash(n, rng, exclude=None) -> list[tuple[int,int]]`.
  - `MAP_REGISTRY: dict[str, type[Map]]` with key `"default"` → `DefaultHouseMap`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_maps.py
import numpy as np
from homebot3d.maps import MAP_REGISTRY, FLOOR, WALL, DefaultHouseMap

def test_default_map_registered():
    assert MAP_REGISTRY["default"] is DefaultHouseMap

def test_default_map_shape_and_border_walls():
    m = DefaultHouseMap()
    assert m.tiles.shape == (18, 27)
    assert (m.tiles[0, :] == WALL).all()       # top border all wall
    assert (m.tiles[-1, :] == WALL).all()       # bottom border all wall

def test_fixtures_are_on_valid_coordinates():
    m = DefaultHouseMap()
    for name in ("fridge", "recliner", "door"):
        col, row = m.fixtures[name]
        assert 0 <= col < m.tiles.shape[1]
        assert 0 <= row < m.tiles.shape[0]

def test_robot_start_is_a_floor_tile():
    m = DefaultHouseMap()
    col, row = m.robot_start_tile
    assert m.tiles[row, col] == FLOOR

def test_valid_floor_tiles_excludes_walls():
    m = DefaultHouseMap()
    for col, row in m.valid_floor_tiles():
        assert m.tiles[row, col] == FLOOR

def test_spawn_trash_is_deterministic():
    m = DefaultHouseMap()
    a = m.spawn_trash(3, np.random.default_rng(0))
    b = m.spawn_trash(3, np.random.default_rng(0))
    assert a == b
    assert len(a) == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_maps.py -v`
Expected: FAIL with `ModuleNotFoundError: homebot3d.maps`.

- [ ] **Step 3: Write `homebot3d/maps.py`**

Port the 2D layout, trimmed to what the nav spine needs (no sprite sizes, no pixel rects, no LAWN — walls are enough). Keep the `default` map identical in topology to the 2D `DefaultHouseMap`.

```python
import numpy as np
from typing import Optional, Type

FLOOR = 0
WALL = 1


class Map:
    tiles: np.ndarray
    fixtures: dict           # {name: (col, row)}
    robot_start_tile: tuple  # (col, row)
    door_tiles: list

    def _finalize(self):
        mask = self.tiles == FLOOR
        rows, cols = np.where(mask)
        self._floor_tiles = list(zip(cols.tolist(), rows.tolist()))

    def valid_floor_tiles(self) -> list[tuple[int, int]]:
        return list(self._floor_tiles)

    def spawn_trash(self, n, rng, exclude: Optional[list[tuple]] = None):
        candidates = self.valid_floor_tiles()
        if exclude:
            ex = set(exclude)
            candidates = [t for t in candidates if t not in ex]
        n = min(n, len(candidates))
        idxs = rng.choice(len(candidates), size=n, replace=False)
        return [candidates[i] for i in idxs]


class DefaultHouseMap(Map):
    def __init__(self):
        COLS, ROWS = 27, 18
        t = np.ones((ROWS, COLS), dtype=np.uint8) * WALL
        t[1:8, 1:11] = FLOOR       # living room
        t[1:8, 12:23] = FLOOR      # kitchen
        t[4:6, 11] = FLOOR         # living<->kitchen doorway
        t[9:11, 1:23] = FLOOR      # hallway
        t[8, 4:7] = FLOOR          # living->hallway
        t[8, 14:17] = FLOOR        # kitchen->hallway
        t[11, 10:13] = FLOOR       # hallway->bedroom
        t[12:17, 1:23] = FLOOR     # bedroom
        t[9:11, 23] = FLOOR        # east doorway to outside
        self.tiles = t
        self.fixtures = {
            "fridge":   (19, 1),
            "recliner": (5,  5),
            "door":     (23, 9),
        }
        self.door_tiles = [(23, 9), (23, 10)]
        self.robot_start_tile = (8, 4)
        self._finalize()


MAP_REGISTRY: dict[str, Type[Map]] = {
    "default": DefaultHouseMap,
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_maps.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add homebot3d/maps.py tests/test_maps.py
git commit -m "feat: 3D maps module — grid + fixtures + MAP_REGISTRY"
```

---

## Task 3: World compiler — floor + walls → MJCF

The keystone module. This task does floor + walls only; fixtures and robot come in Tasks 4–5.

**Files:**
- Create: `homebot3d/world.py`
- Test: `tests/test_world_walls.py`

**Interfaces:**
- Produces:
  - `tile_center(col, row) -> tuple[float, float]` → `((col + 0.5) * TILE, (row + 0.5) * TILE)`.
  - `build_mjcf(map) -> str` (extended in later tasks with more kwargs). Emits a `<mujoco>` doc with a floor plane and one box geom per WALL tile. Wall boxes are named `wall_{row}_{col}`.
  - Every wall box geom: `type="box"`, half-sizes `(TILE/2, TILE/2, WALL_HEIGHT/2)`, centre at `(tile_center(col,row)[0], tile_center(col,row)[1], WALL_HEIGHT/2)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_world_walls.py
import numpy as np
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import build_mjcf, tile_center
from homebot3d.constants import TILE, WALL_HEIGHT

def test_tile_center():
    x, y = tile_center(0, 0)
    assert x == 0.25 and y == 0.25          # (0+0.5)*0.5

def test_mjcf_compiles_in_mujoco():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    assert model is not None

def test_wall_count_matches_grid():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    n_walls = int((m.tiles == 1).sum())
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
             for i in range(model.ngeom)]
    wall_names = [n for n in names if n and n.startswith("wall_")]
    assert len(wall_names) == n_walls

def test_wall_geom_position_and_size():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_0_0")
    assert gid != -1
    np.testing.assert_allclose(model.geom_size[gid],
                               [TILE/2, TILE/2, WALL_HEIGHT/2], atol=1e-6)
    cx, cy = tile_center(0, 0)
    np.testing.assert_allclose(model.geom_pos[gid], [cx, cy, WALL_HEIGHT/2], atol=1e-6)

def test_has_floor_plane():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert gid != -1
    assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_world_walls.py -v`
Expected: FAIL with `ModuleNotFoundError: homebot3d.world`.

- [ ] **Step 3: Write `homebot3d/world.py`**

```python
import numpy as np
from homebot3d.maps import Map, WALL
from homebot3d.constants import TILE, WALL_HEIGHT


def tile_center(col: int, row: int) -> tuple[float, float]:
    return (col + 0.5) * TILE, (row + 0.5) * TILE


def _wall_geoms(map: Map) -> str:
    rows, cols = map.tiles.shape
    parts = []
    hx = hy = TILE / 2
    hz = WALL_HEIGHT / 2
    for r in range(rows):
        for c in range(cols):
            if map.tiles[r, c] == WALL:
                cx, cy = tile_center(c, r)
                parts.append(
                    f'<geom name="wall_{r}_{c}" type="box" '
                    f'size="{hx} {hy} {hz}" pos="{cx} {cy} {hz}" '
                    f'rgba="0.7 0.7 0.72 1"/>'
                )
    return "\n".join(parts)


def build_mjcf(map: Map) -> str:
    rows, cols = map.tiles.shape
    fx = cols * TILE
    fy = rows * TILE
    return f"""
<mujoco model="homebot3d">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="{fx/2} {fy/2} 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" pos="{fx/2} {fy/2} 0"
          size="{fx/2} {fy/2} 0.1" rgba="0.85 0.82 0.78 1"/>
{_wall_geoms(map)}
  </worldbody>
</mujoco>
"""
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_world_walls.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add homebot3d/world.py tests/test_world_walls.py
git commit -m "feat: world compiler — grid floor + walls to MJCF"
```

---

## Task 4: World compiler — fixtures → MJCF bodies

**Files:**
- Modify: `homebot3d/world.py`
- Test: `tests/test_world_fixtures.py`

**Interfaces:**
- Consumes: `build_mjcf(map)` from Task 3.
- Produces: `build_mjcf(map)` now also emits one body per fixture in `map.fixtures`, named `fixture_{name}`, with a box geom named `fixture_{name}_geom`, standing at `tile_center` with fixed half-size `(TILE*0.6, TILE*0.6, 0.35)` centred at z = 0.35. Fixtures are visual/collidable landmarks (nav targets), not articulated.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_world_fixtures.py
import numpy as np
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import build_mjcf, tile_center

def test_each_fixture_becomes_a_body():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    for name in m.fixtures:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"fixture_{name}")
        assert bid != -1, f"missing body fixture_{name}"

def test_fixture_body_at_tile_center():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fixture_fridge")
    cx, cy = tile_center(*m.fixtures["fridge"])
    np.testing.assert_allclose(model.body_pos[bid][:2], [cx, cy], atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_world_fixtures.py -v`
Expected: FAIL — `mj_name2id` returns -1 (bodies not emitted yet).

- [ ] **Step 3: Extend `homebot3d/world.py`**

Add a `_fixture_bodies` helper and call it inside `build_mjcf`:

```python
def _fixture_bodies(map: Map) -> str:
    hx = hy = TILE * 0.6
    hz = 0.35
    colors = {"fridge": "0.6 0.7 0.9 1", "recliner": "0.5 0.3 0.2 1",
              "door": "0.3 0.2 0.1 1"}
    parts = []
    for name, (col, row) in map.fixtures.items():
        cx, cy = tile_center(col, row)
        rgba = colors.get(name, "0.5 0.5 0.5 1")
        parts.append(
            f'<body name="fixture_{name}" pos="{cx} {cy} {hz}">'
            f'<geom name="fixture_{name}_geom" type="box" '
            f'size="{hx} {hy} {hz}" rgba="{rgba}"/></body>'
        )
    return "\n".join(parts)
```

In `build_mjcf`, insert `{_fixture_bodies(map)}` in the `<worldbody>` right after the wall geoms line:

```python
{_wall_geoms(map)}
{_fixture_bodies(map)}
  </worldbody>
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_world_fixtures.py -v`
Expected: PASS (2 tests). Re-run Task 3 tests: `python -m pytest tests/test_world_walls.py -v` — still PASS.

- [ ] **Step 5: Commit**

```bash
git add homebot3d/world.py tests/test_world_fixtures.py
git commit -m "feat: world compiler — fixtures as landmark bodies"
```

---

## Task 5: World compiler — diff-drive robot body

**Files:**
- Modify: `homebot3d/world.py`
- Test: `tests/test_world_robot.py`

**Interfaces:**
- Consumes: `build_mjcf` from Task 4.
- Produces: `build_mjcf(map, robot_start=None)`. When `robot_start` is `None`, uses `map.robot_start_tile`. Emits a `robot` body with:
  - three joints: `slide_x` (slide, axis 1 0 0), `slide_y` (slide, axis 0 1 0), `yaw` (hinge, axis 0 0 1).
  - a cylinder geom named `robot_body`, size `(ROBOT_RADIUS, ROBOT_HALFHEIGHT)`.
  - an egocentric camera named `ego`, mounted forward, `pos="ROBOT_RADIUS 0 0.1"`, `xyaxes="0 -1 0 0 0 1"` (looks along body +x, up = +z).
  - three velocity actuators named `vx` (joint slide_x), `vy` (joint slide_y), `wz` (joint yaw).
  - Robot body placed at `tile_center(robot_start)` with z = `ROBOT_HALFHEIGHT`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_world_robot.py
import numpy as np
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import build_mjcf, tile_center
from homebot3d.constants import ROBOT_HALFHEIGHT

def _model(robot_start=None):
    m = DefaultHouseMap()
    return m, mujoco.MjModel.from_xml_string(build_mjcf(m, robot_start=robot_start))

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
    model = mujoco.MjModel.from_xml_string(build_mjcf(m, robot_start=(9, 9)))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    cx, cy = tile_center(9, 9)
    np.testing.assert_allclose(model.body_pos[bid][:2], [cx, cy], atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_world_robot.py -v`
Expected: FAIL — `build_mjcf()` has no `robot_start` kwarg / robot body missing.

- [ ] **Step 3: Extend `homebot3d/world.py`**

Add imports and a `_robot_body` helper, and thread `robot_start` + an `<actuator>` block through `build_mjcf`:

```python
from homebot3d.constants import (TILE, WALL_HEIGHT, ROBOT_RADIUS,
                                 ROBOT_HALFHEIGHT)

def _robot_body(map, robot_start) -> str:
    col, row = robot_start if robot_start is not None else map.robot_start_tile
    cx, cy = tile_center(col, row)
    z = ROBOT_HALFHEIGHT
    return f"""
    <body name="robot" pos="{cx} {cy} {z}">
      <joint name="slide_x" type="slide" axis="1 0 0"/>
      <joint name="slide_y" type="slide" axis="0 1 0"/>
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom name="robot_body" type="cylinder"
            size="{ROBOT_RADIUS} {ROBOT_HALFHEIGHT}" rgba="0.2 0.6 0.3 1"/>
      <camera name="ego" pos="{ROBOT_RADIUS} 0 0.1" xyaxes="0 -1 0 0 0 1"/>
    </body>"""
```

Change the signature to `def build_mjcf(map: Map, robot_start=None) -> str:` and insert the robot body plus an actuator block:

```python
{_fixture_bodies(map)}
{_robot_body(map, robot_start)}
  </worldbody>
  <actuator>
    <velocity name="vx" joint="slide_x" kv="8" ctrlrange="-2 2"/>
    <velocity name="vy" joint="slide_y" kv="8" ctrlrange="-2 2"/>
    <velocity name="wz" joint="yaw" kv="2" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_world_robot.py -v`
Expected: PASS (5 tests). Re-run Tasks 3–4 tests — still PASS.

- [ ] **Step 5: Commit**

```bash
git add homebot3d/world.py tests/test_world_robot.py
git commit -m "feat: world compiler — diff-drive robot body + ego camera"
```

---

## Task 6: Robot control wrapper

**Files:**
- Create: `homebot3d/robot.py`
- Test: `tests/test_robot.py`

**Interfaces:**
- Consumes: a compiled `mujoco.MjModel` + `mujoco.MjData` with joints `slide_x/slide_y/yaw` and actuators `vx/vy/wz`.
- Produces:
  - `class Robot(model, data)`.
  - Property `x -> float`, `y -> float` (world position of robot body centre, from `data.qpos` of slide joints).
  - Property `heading -> float` (yaw radians, from `data.qpos` of `yaw` joint).
  - `apply(action: np.ndarray)` — `action = [linear, angular]` in `[-1,1]`; sets `data.ctrl` so the base drives along its current heading: `ctrl[vx] = linear*MAX_LIN*cos(heading)`, `ctrl[vy] = linear*MAX_LIN*sin(heading)`, `ctrl[wz] = angular*MAX_ANG`.
  - `collided() -> bool` — True if any active contact involves the `robot_body` geom and a wall/fixture geom.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_robot.py
import numpy as np
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import build_mjcf, tile_center
from homebot3d.robot import Robot

def _sim():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return m, model, data

def test_initial_pose_matches_start_tile():
    m, model, data = _sim()
    r = Robot(model, data)
    cx, cy = tile_center(*m.robot_start_tile)
    assert abs(r.x - cx) < 1e-6
    assert abs(r.y - cy) < 1e-6

def test_forward_action_moves_along_heading():
    m, model, data = _sim()
    r = Robot(model, data)
    x0, y0 = r.x, r.y
    for _ in range(50):                      # 0.5 s at dt=0.01
        r.apply(np.array([1.0, 0.0]))
        mujoco.mj_step(model, data)
    assert r.x > x0 + 0.05                    # heading 0 => +x motion
    assert abs(r.y - y0) < 0.05

def test_turn_action_changes_heading():
    m, model, data = _sim()
    r = Robot(model, data)
    h0 = r.heading
    for _ in range(50):
        r.apply(np.array([0.0, 1.0]))
        mujoco.mj_step(model, data)
    assert r.heading > h0 + 0.1

def test_collided_false_in_open_space():
    m, model, data = _sim()
    r = Robot(model, data)
    assert r.collided() is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_robot.py -v`
Expected: FAIL with `ModuleNotFoundError: homebot3d.robot`.

- [ ] **Step 3: Write `homebot3d/robot.py`**

```python
import numpy as np
import mujoco
from homebot3d.constants import MAX_LIN, MAX_ANG


class Robot:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self._qx = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_x")]
        self._qy = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_y")]
        self._qyaw = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")]
        self._a_vx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vx")
        self._a_vy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vy")
        self._a_wz = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wz")
        self._body_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_body")

    @property
    def x(self) -> float:
        return float(self.data.qpos[self._qx])

    @property
    def y(self) -> float:
        return float(self.data.qpos[self._qy])

    @property
    def heading(self) -> float:
        return float(self.data.qpos[self._qyaw])

    def apply(self, action):
        linear = float(np.clip(action[0], -1.0, 1.0)) * MAX_LIN
        angular = float(np.clip(action[1], -1.0, 1.0)) * MAX_ANG
        h = self.heading
        self.data.ctrl[self._a_vx] = linear * np.cos(h)
        self.data.ctrl[self._a_vy] = linear * np.sin(h)
        self.data.ctrl[self._a_wz] = angular

    def collided(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self._body_geom in (c.geom1, c.geom2):
                return True
        return False
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_robot.py -v`
Expected: PASS (4 tests). If `test_forward_action_moves_along_heading` under-moves, raise actuator `kv` in `world.py` (Task 5) and re-run — this is the documented tuning point.

- [ ] **Step 5: Commit**

```bash
git add homebot3d/robot.py tests/test_robot.py
git commit -m "feat: Robot control wrapper — velocity drive + collision read"
```

---

## Task 7: Sensors — egocentric RGB + privileged seam

**Files:**
- Create: `homebot3d/sensors.py`
- Test: `tests/test_sensors.py`

**Interfaces:**
- Consumes: compiled `model`/`data` with camera `ego`; a `Robot` for pose.
- Produces:
  - `class Camera(model, width=84, height=84)` with `render_rgb(data) -> np.ndarray (H,W,3) uint8` and `render_depth(data) -> np.ndarray (H,W) float32`; `close()`.
  - `privileged(robot, goal_xy) -> dict` returning `{"pose": np.array([x, y, heading], float32), "goal_vec": np.array([gx-x, gy-y], float32)}`. Depth is fetched separately via `Camera.render_depth` so the seam stays cheap when unused.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sensors.py
import numpy as np
import mujoco
from homebot3d.maps import DefaultHouseMap
from homebot3d.world import build_mjcf
from homebot3d.robot import Robot
from homebot3d.sensors import Camera, privileged

def _sim():
    m = DefaultHouseMap()
    model = mujoco.MjModel.from_xml_string(build_mjcf(m))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return m, model, data

def test_rgb_shape_and_dtype():
    m, model, data = _sim()
    cam = Camera(model, width=84, height=84)
    frame = cam.render_rgb(data)
    assert frame.shape == (84, 84, 3)
    assert frame.dtype == np.uint8
    cam.close()

def test_rgb_deterministic_for_same_state():
    m, model, data = _sim()
    cam = Camera(model)
    a = cam.render_rgb(data)
    b = cam.render_rgb(data)
    np.testing.assert_array_equal(a, b)
    cam.close()

def test_depth_shape():
    m, model, data = _sim()
    cam = Camera(model)
    d = cam.render_depth(data)
    assert d.shape == (84, 84)
    assert d.dtype == np.float32
    cam.close()

def test_privileged_pose_and_goal_vec():
    m, model, data = _sim()
    r = Robot(model, data)
    p = privileged(r, goal_xy=(r.x + 1.0, r.y + 2.0))
    np.testing.assert_allclose(p["pose"][:2], [r.x, r.y], atol=1e-6)
    np.testing.assert_allclose(p["goal_vec"], [1.0, 2.0], atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_sensors.py -v`
Expected: FAIL with `ModuleNotFoundError: homebot3d.sensors`.

- [ ] **Step 3: Write `homebot3d/sensors.py`**

```python
import numpy as np
import mujoco


class Camera:
    def __init__(self, model, width=84, height=84):
        self.width = width
        self.height = height
        self._rgb = mujoco.Renderer(model, height=height, width=width)
        self._depth = mujoco.Renderer(model, height=height, width=width)
        self._depth.enable_depth_rendering()

    def render_rgb(self, data) -> np.ndarray:
        self._rgb.update_scene(data, camera="ego")
        return self._rgb.render()

    def render_depth(self, data) -> np.ndarray:
        self._depth.update_scene(data, camera="ego")
        return self._depth.render().astype(np.float32)

    def close(self):
        self._rgb.close()
        self._depth.close()


def privileged(robot, goal_xy) -> dict:
    return {
        "pose": np.array([robot.x, robot.y, robot.heading], dtype=np.float32),
        "goal_vec": np.array([goal_xy[0] - robot.x, goal_xy[1] - robot.y],
                             dtype=np.float32),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_sensors.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add homebot3d/sensors.py tests/test_sensors.py
git commit -m "feat: sensors — egocentric RGB/depth render + privileged seam"
```

---

## Task 8: Goals + TaskManager (reach semantics)

**Files:**
- Create: `homebot3d/goals.py`, `homebot3d/tasks.py`
- Test: `tests/test_tasks.py`

**Interfaces:**
- Produces (`goals.py`):
  - `GOAL_TARGETS: dict[str, str]` = `{"trash": "trash", "drink": "recliner", "package": "door"}` (nav-spine mapping: each high-level goal reduces to reaching one location; drink/package deliver-to-recliner and reach-door respectively, carrying deferred).
  - `goal_to_coordinates(goal, map, trash_positions=None, rng=None) -> tuple[float,float]` in world metres via `tile_center`.
- Produces (`tasks.py`):
  - `class TaskManager(goals: list[str])`.
  - `reset(map, n_trash, rng)` — samples trash tiles if `"trash"` in goals; clears delivered flags.
  - `step(robot) -> float` — for each active goal, +1.0 the first time the robot centre comes within `REACH_RADIUS` of its target; removes trash targets as reached.
  - `is_done() -> bool` — all active goals reached.
  - `get_info(robot) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tasks.py
import numpy as np
from homebot3d.maps import DefaultHouseMap
from homebot3d.goals import goal_to_coordinates, GOAL_TARGETS
from homebot3d.tasks import TaskManager
from homebot3d.world import tile_center

class FakeRobot:
    def __init__(self, x, y): self.x, self.y = x, y

def test_goal_targets_mapping():
    assert GOAL_TARGETS == {"trash": "trash", "drink": "recliner", "package": "door"}

def test_goal_to_coordinates_maps_drink_to_recliner():
    m = DefaultHouseMap()
    assert goal_to_coordinates("drink", m) == tile_center(*m.fixtures["recliner"])

def test_reaching_recliner_rewards_once_and_completes():
    m = DefaultHouseMap()
    tm = TaskManager(goals=["drink"])
    tm.reset(m, n_trash=0, rng=np.random.default_rng(0))
    rx, ry = tile_center(*m.fixtures["recliner"])
    assert tm.step(FakeRobot(rx, ry)) == 1.0
    assert tm.step(FakeRobot(rx, ry)) == 0.0      # no double reward
    assert tm.is_done() is True

def test_far_away_gives_no_reward():
    m = DefaultHouseMap()
    tm = TaskManager(goals=["drink"])
    tm.reset(m, n_trash=0, rng=np.random.default_rng(0))
    assert tm.step(FakeRobot(-100.0, -100.0)) == 0.0
    assert tm.is_done() is False

def test_trash_targets_reached_individually():
    m = DefaultHouseMap()
    tm = TaskManager(goals=["trash"])
    tm.reset(m, n_trash=2, rng=np.random.default_rng(1))
    assert len(tm.trash_positions) == 2
    (c0, r0) = tm.trash_positions[0]
    x0, y0 = tile_center(c0, r0)
    assert tm.step(FakeRobot(x0, y0)) == 1.0
    assert len(tm.trash_positions) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_tasks.py -v`
Expected: FAIL with `ModuleNotFoundError: homebot3d.goals`.

- [ ] **Step 3: Write `homebot3d/goals.py`**

```python
import numpy as np
from typing import Optional
from homebot3d.maps import Map
from homebot3d.world import tile_center

GOAL_TARGETS = {"trash": "trash", "drink": "recliner", "package": "door"}


def goal_to_coordinates(goal, map: Map, trash_positions: Optional[list] = None,
                        rng=None) -> tuple[float, float]:
    target = GOAL_TARGETS[goal]
    if target == "trash":
        if not trash_positions:
            raise ValueError("trash goal requires at least one trash position")
        idx = int(rng.integers(0, len(trash_positions))) if rng is not None else 0
        col, row = trash_positions[idx]
    else:
        col, row = map.fixtures[target]
    return tile_center(col, row)
```

- [ ] **Step 4: Write `homebot3d/tasks.py`**

```python
import math
import numpy as np
from homebot3d.maps import Map
from homebot3d.world import tile_center
from homebot3d.constants import REACH_RADIUS


def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


class TaskManager:
    def __init__(self, goals):
        self.goals = set(goals)
        self.trash_positions = []
        self.drink_done = False
        self.package_done = False

    def reset(self, map: Map, n_trash, rng):
        self._map = map
        self.trash_positions = (
            map.spawn_trash(n_trash, rng, exclude=list(map.fixtures.values()))
            if "trash" in self.goals else []
        )
        self.drink_done = False
        self.package_done = False

    def step(self, robot) -> float:
        reward = 0.0
        if "trash" in self.goals and self.trash_positions:
            remaining = []
            for (c, r) in self.trash_positions:
                tx, ty = tile_center(c, r)
                if _dist(robot.x, robot.y, tx, ty) <= REACH_RADIUS:
                    reward += 1.0
                else:
                    remaining.append((c, r))
            self.trash_positions = remaining
        if "drink" in self.goals and not self.drink_done:
            gx, gy = tile_center(*self._map.fixtures["recliner"])
            if _dist(robot.x, robot.y, gx, gy) <= REACH_RADIUS:
                self.drink_done = True
                reward += 1.0
        if "package" in self.goals and not self.package_done:
            gx, gy = tile_center(*self._map.fixtures["door"])
            if _dist(robot.x, robot.y, gx, gy) <= REACH_RADIUS:
                self.package_done = True
                reward += 1.0
        return reward

    def is_done(self) -> bool:
        trash = "trash" not in self.goals or not self.trash_positions
        drink = "drink" not in self.goals or self.drink_done
        pkg = "package" not in self.goals or self.package_done
        return trash and drink and pkg

    def get_info(self, robot) -> dict:
        return {
            "trash_remaining": len(self.trash_positions),
            "drink_done": self.drink_done,
            "package_done": self.package_done,
        }
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_tasks.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add homebot3d/goals.py homebot3d/tasks.py tests/test_tasks.py
git commit -m "feat: goals + TaskManager with reach semantics"
```

---

## Task 9: `HomeBot3DEnv` (Gymnasium) + determinism + random_start

**Files:**
- Create: `homebot3d/env.py`
- Modify: `homebot3d/__init__.py` (export + register)
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: `maps`, `world.build_mjcf`, `Robot`, `Camera`, `TaskManager`, constants.
- Produces:
  - `class HomeBot3DEnv(gymnasium.Env)` with `__init__(goals=("trash","drink","package"), n_trash=2, max_steps=2000, map_name="default", random_start=False, width=84, height=84, render_mode=None)`.
  - `observation_space = Box(0, 255, (height, width, 3), uint8)`.
  - `action_space = Box(-1, 1, (2,), float32)`.
  - `reset(seed=None, options=None) -> (obs, info)`; `info` includes `map_name` and a `privileged` dict (pose/goal_vec).
  - `step(action) -> (obs, reward, terminated, truncated, info)`; `terminated` when `TaskManager.is_done()`, `truncated` at `max_steps`.
  - `_sample_start_tile(rng)` — valid floor tile ≥ `2*ROBOT_RADIUS/TILE + 1` tiles from every fixture; seed-deterministic.
  - Registers Gymnasium id `"HomeBot3D-V1"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env.py
import numpy as np
from homebot3d.env import HomeBot3DEnv

def test_spaces():
    env = HomeBot3DEnv(width=84, height=84)
    assert env.observation_space.shape == (84, 84, 3)
    assert env.action_space.shape == (2,)
    env.close()

def test_reset_returns_rgb_obs_only():
    env = HomeBot3DEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (84, 84, 3) and obs.dtype == np.uint8
    assert "privileged" in info               # seam lives in info, not obs
    assert "pose" in info["privileged"]
    env.close()

def test_step_contract():
    env = HomeBot3DEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (84, 84, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    env.close()

def test_determinism_same_seed_same_frames():
    e1 = HomeBot3DEnv(); o1, _ = e1.reset(seed=42)
    e2 = HomeBot3DEnv(); o2, _ = e2.reset(seed=42)
    np.testing.assert_array_equal(o1, o2)
    a = np.array([0.5, 0.2], dtype=np.float32)
    f1 = e1.step(a)[0]; f2 = e2.step(a)[0]
    np.testing.assert_array_equal(f1, f2)
    e1.close(); e2.close()

def test_truncation_at_max_steps():
    env = HomeBot3DEnv(goals=("drink",), max_steps=3)
    env.reset(seed=0)
    truncated = False
    for _ in range(3):
        truncated = env.step(np.array([0.0, 0.0], dtype=np.float32))[3]
    assert truncated is True
    env.close()

def test_random_start_changes_spawn_but_is_seed_stable():
    e = HomeBot3DEnv(random_start=True)
    o_a, i_a = e.reset(seed=7)
    o_b, i_b = e.reset(seed=7)
    np.testing.assert_allclose(i_a["privileged"]["pose"], i_b["privileged"]["pose"])
    e.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_env.py -v`
Expected: FAIL with `ModuleNotFoundError: homebot3d.env`.

- [ ] **Step 3: Write `homebot3d/env.py`**

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from homebot3d.maps import MAP_REGISTRY
from homebot3d.world import build_mjcf, tile_center
from homebot3d.robot import Robot
from homebot3d.sensors import Camera, privileged
from homebot3d.tasks import TaskManager
from homebot3d.goals import GOAL_TARGETS
from homebot3d.constants import TILE, ROBOT_RADIUS, REACH_RADIUS


class HomeBot3DEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, goals=("trash", "drink", "package"), n_trash=2,
                 max_steps=2000, map_name="default", random_start=False,
                 width=84, height=84, render_mode=None):
        super().__init__()
        self.goals = list(goals)
        self.n_trash = n_trash
        self.max_steps = max_steps
        self.map_name = map_name
        self.random_start = random_start
        self.width, self.height = width, height
        self.render_mode = render_mode
        self.observation_space = spaces.Box(0, 255, (height, width, 3), np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._map = MAP_REGISTRY[map_name]()
        self._tasks = TaskManager(self.goals)
        self._camera = None
        self._steps = 0

    def _min_tile_gap(self):
        return int(np.ceil(2 * ROBOT_RADIUS / TILE)) + 1

    def _sample_start_tile(self, rng):
        if not self.random_start:
            return self._map.robot_start_tile
        gap = self._min_tile_gap()
        fixtures = list(self._map.fixtures.values())
        candidates = []
        for (c, r) in self._map.valid_floor_tiles():
            if all(abs(c - fc) + abs(r - fr) >= gap for (fc, fr) in fixtures):
                candidates.append((c, r))
        if not candidates:
            return self._map.robot_start_tile
        return candidates[int(rng.integers(0, len(candidates)))]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        start = self._sample_start_tile(rng)
        xml = build_mjcf(self._map, robot_start=start)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._robot = Robot(self.model, self.data)
        if self._camera is not None:
            self._camera.close()
        self._camera = Camera(self.model, width=self.width, height=self.height)
        self._tasks.reset(self._map, self.n_trash, rng)
        self._steps = 0
        return self._obs(), self._info()

    def step(self, action):
        self._robot.apply(np.asarray(action, dtype=np.float32))
        mujoco.mj_step(self.model, self.data)
        self._steps += 1
        reward = float(self._tasks.step(self._robot))
        terminated = self._tasks.is_done()
        truncated = self._steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, self._info()

    def _goal_xy(self):
        for g in ("drink", "package", "trash"):
            if g in self.goals:
                target = GOAL_TARGETS[g]
                if target == "trash" and self._tasks.trash_positions:
                    return tile_center(*self._tasks.trash_positions[0])
                if target != "trash":
                    return tile_center(*self._map.fixtures[target])
        return (self._robot.x, self._robot.y)

    def _obs(self):
        return self._camera.render_rgb(self.data)

    def _info(self):
        info = self._tasks.get_info(self._robot)
        info["map_name"] = self.map_name
        info["privileged"] = privileged(self._robot, self._goal_xy())
        return info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._camera.render_rgb(self.data)

    def close(self):
        if self._camera is not None:
            self._camera.close()
            self._camera = None
```

- [ ] **Step 4: Register the env in `homebot3d/__init__.py`**

Append below the `MUJOCO_GL` line:

```python
from gymnasium.envs.registration import register
from homebot3d.env import HomeBot3DEnv

register(id="HomeBot3D-V1", entry_point="homebot3d.env:HomeBot3DEnv")
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_env.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add homebot3d/env.py homebot3d/__init__.py tests/test_env.py
git commit -m "feat: HomeBot3DEnv — single-env Gymnasium + determinism + random_start"
```

---

## Task 10: `HomeBot3DGoalEnv` (GoalEnv / HER)

**Files:**
- Modify: `homebot3d/env.py`, `homebot3d/__init__.py`
- Test: `tests/test_goal_env.py`

**Interfaces:**
- Consumes: `HomeBot3DEnv`.
- Produces:
  - `class HomeBot3DGoalEnv(HomeBot3DEnv)` with `observation_space = Dict({"observation": Box RGB, "achieved_goal": Box(2,), "desired_goal": Box(2,)})`.
  - `achieved_goal` = robot planar position `[x, y]`; `desired_goal` = active goal `[gx, gy]`.
  - `compute_reward(achieved_goal, desired_goal, info) -> reward` (vectorised-safe): `0.0` if within `REACH_RADIUS`, else `-1.0`. Sparse, HER-relabel friendly.
  - Registers Gymnasium id `"HomeBot3D-Goal-V1"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goal_env.py
import numpy as np
from homebot3d.env import HomeBot3DGoalEnv
from homebot3d.constants import REACH_RADIUS

def test_dict_obs_structure():
    env = HomeBot3DGoalEnv()
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    assert obs["observation"].shape == (84, 84, 3)
    assert obs["achieved_goal"].shape == (2,)
    assert obs["desired_goal"].shape == (2,)
    env.close()

def test_compute_reward_scalar():
    env = HomeBot3DGoalEnv()
    env.reset(seed=0)
    ag = np.array([0.0, 0.0]); dg = np.array([0.0, 0.0])
    assert env.compute_reward(ag, dg, {}) == 0.0
    far = np.array([10.0, 10.0])
    assert env.compute_reward(far, dg, {}) == -1.0
    env.close()

def test_compute_reward_vectorised():
    env = HomeBot3DGoalEnv()
    env.reset(seed=0)
    ag = np.array([[0.0, 0.0], [10.0, 10.0]])
    dg = np.array([[0.0, 0.0], [0.0, 0.0]])
    out = env.compute_reward(ag, dg, [{}, {}])
    np.testing.assert_array_equal(out, [0.0, -1.0])
    env.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_goal_env.py -v`
Expected: FAIL — `HomeBot3DGoalEnv` not defined.

- [ ] **Step 3: Add `HomeBot3DGoalEnv` to `homebot3d/env.py`**

```python
class HomeBot3DGoalEnv(HomeBot3DEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rgb = self.observation_space
        self.observation_space = spaces.Dict({
            "observation": rgb,
            "achieved_goal": spaces.Box(-np.inf, np.inf, (2,), np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, (2,), np.float32),
        })

    def _dict_obs(self, rgb):
        return {
            "observation": rgb,
            "achieved_goal": np.array([self._robot.x, self._robot.y], np.float32),
            "desired_goal": np.array(self._goal_xy(), np.float32),
        }

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self._dict_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._dict_obs(obs), reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        return np.where(d <= REACH_RADIUS, 0.0, -1.0)
```

- [ ] **Step 4: Register in `homebot3d/__init__.py`**

```python
from homebot3d.env import HomeBot3DGoalEnv
register(id="HomeBot3D-Goal-V1", entry_point="homebot3d.env:HomeBot3DGoalEnv")
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_goal_env.py -v`
Expected: PASS (3 tests). Note: `compute_reward` returns a numpy scalar for scalar input — the `== 0.0` comparisons hold.

- [ ] **Step 6: Commit**

```bash
git add homebot3d/env.py homebot3d/__init__.py tests/test_goal_env.py
git commit -m "feat: HomeBot3DGoalEnv — GoalEnv dict obs + HER compute_reward"
```

---

## Task 11: Human play wrapper + full-suite verification

**Files:**
- Create: `play.py`
- Test: `tests/test_integration.py`

**Interfaces:**
- Consumes: `HomeBot3DEnv`.
- Produces: `play.py` CLI that steps the env from arrow/WASD keys using MuJoCo's passive viewer; an integration test that runs a short rollout headless and asserts frames render and an episode can complete.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_integration.py
import numpy as np
from homebot3d.env import HomeBot3DEnv

def test_short_rollout_headless():
    env = HomeBot3DEnv(goals=("drink",), max_steps=50)
    obs, info = env.reset(seed=0)
    frames = 0
    for _ in range(50):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert obs.shape == (84, 84, 3)
        frames += 1
        if term or trunc:
            break
    assert frames > 0
    env.close()

def test_driving_toward_recliner_eventually_rewards():
    # Scripted controller: steer straight at the recliner using the privileged seam.
    env = HomeBot3DEnv(goals=("drink",), max_steps=1500, random_start=False)
    _, info = env.reset(seed=0)
    got_reward = False
    for _ in range(1500):
        gx, gy = info["privileged"]["goal_vec"]
        heading_err = np.arctan2(gy, gx)
        action = np.array([0.6, np.clip(heading_err, -1, 1)], dtype=np.float32)
        _, reward, term, trunc, info = env.step(action)
        if reward > 0:
            got_reward = True
            break
        if trunc:
            break
    assert got_reward, "scripted seek-recliner controller never reached the goal"
    env.close()
```

- [ ] **Step 2: Run to verify it fails/then passes**

Run: `python -m pytest tests/test_integration.py -v`
Expected: `test_short_rollout_headless` PASSES immediately (env exists). `test_driving_toward_recliner_eventually_rewards` must PASS — if it does not, the diff-drive tuning (actuator `kv` in Task 5) or `REACH_RADIUS` is wrong; fix and re-run. This test is the end-to-end proof the nav loop actually works.

- [ ] **Step 3: Write `play.py`**

```python
"""Human play wrapper for HomeBot3DEnv (MuJoCo passive viewer).

Controls: W/S forward/back, A/D turn, R reset, Esc quit.
"""
import argparse
import numpy as np
import mujoco.viewer
from homebot3d.env import HomeBot3DEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--goals", nargs="+", default=["trash", "drink", "package"])
    p.add_argument("--map", default="default")
    p.add_argument("--random-start", action="store_true")
    args = p.parse_args()

    env = HomeBot3DEnv(goals=tuple(args.goals), map_name=args.map,
                       random_start=args.random_start)
    env.reset(seed=0)
    action = np.zeros(2, dtype=np.float32)

    def key_cb(keycode):
        nonlocal action
        c = chr(keycode) if 0 < keycode < 0x110000 else ""
        if c in "Ww": action[:] = [1.0, 0.0]
        elif c in "Ss": action[:] = [-1.0, 0.0]
        elif c in "Aa": action[:] = [0.0, -1.0]
        elif c in "Dd": action[:] = [0.0, 1.0]
        elif c in "Rr": env.reset(seed=0); action[:] = 0
        else: action[:] = 0

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_cb) as v:
        while v.is_running():
            _, reward, term, trunc, _ = env.step(action)
            if reward > 0:
                print(f"+{reward}")
            if term or trunc:
                env.reset(seed=0)
            v.sync()
    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -v`
Expected: ALL tests across all files PASS.

- [ ] **Step 5: Commit**

```bash
git add play.py tests/test_integration.py
git commit -m "feat: human play wrapper + end-to-end nav integration tests"
```

---

## Self-Review Notes (author)

- **Spec §5 privileged seam:** implemented in `sensors.privileged` + exposed via `info["privileged"]`, never in `obs` — satisfies "deployed observation is RGB only." Depth available via `Camera.render_depth` for training wrappers.
- **Spec §6 grid extrusion:** Tasks 3–5. **§7 RGB obs 84×84:** Task 7/9. **§8 diff-drive continuous action:** Tasks 5–6. **§9 tasks/reward/GoalEnv/HER:** Tasks 8/10. **§10 determinism + headless render:** Tasks 9/11.
- **AMD/EGL (§3, global constraints):** EGL set in `__init__`, verified in Task 1; no JAX anywhere.
- **Deferred (spec §4 out-of-scope):** arm, domain randomization, deployed depth, multi-floor — no tasks, correct.
- **Known tuning points flagged inline:** actuator `kv` (Task 6/5), reach radius (Task 8). Both have an end-to-end guard in Task 11.
