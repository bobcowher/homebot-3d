# Pickup → Carry → Deliver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Turn the drink/package goals into a two-phase pickup→deliver task with a visible carried item on the robot.

**Architecture:** `TaskManager` becomes a small per-goal state machine (`seek_source → seek_target → done`). The env toggles carried-item geom alpha each step and reports the phase-aware current objective through the existing privileged/HER `_goal_xy` seam. Trash stays reach-based and unchanged.

**Tech Stack:** MuJoCo MJCF (procedural geoms), Gymnasium env, NumPy. No new deps.

## Global Constraints
- Single-env only; parallelism is an anti-goal. No new dependencies.
- Privileged info (pose/goal_vec) stays in `info["privileged"]`, never in obs.
- Collision-prefix contract unchanged: `wall_`/`fixture_` = collision; `trash_`/`cargo_` are ignored by `Robot.collided()`.
- Reward: **+1 on pickup (reach source), +1 on delivery (reach target)**. No shaping term.
- `REACH_RADIUS = 0.75` gates both pickup and delivery (reuse `_dist <= REACH_RADIUS`).
- Run tests with `PYTHONPATH="$(pwd)" python3 -m pytest -q` (conda env `homebot-3d`). Never inline `python3 -c`.

## Locked Design Decisions (from user)
- **Drink:** source = fridge tile `(19,1)`; target = human/recliner tile `(5,5)`.
- **Package ("bring in"):** source = front door tile `(23,9)`; target = kitchen table tile `(16,4)`.
- **Carried visual:** a cup shows while carrying drink, a box while carrying package; hidden otherwise. Toggle via `model.geom_rgba[gid,3]` at runtime (no rebuild).
- Carrying is per-goal independent (robot may carry both cup and box); no single-carry constraint.

---

### Task 1: Map source/target tiles

**Files:** Modify `homebot3d/maps.py`; Test `tests/test_tasks.py`

**Interfaces produced:** `map.pickup_tiles: dict[str,tuple]`, `map.dropoff_tiles: dict[str,tuple]`.

- [ ] Add base-class annotations under the existing ones in `class Map`:
```python
    pickup_tiles: dict     # {carry_goal: (col,row)} where the item is picked up
    dropoff_tiles: dict    # {carry_goal: (col,row)} where it is delivered
```
- [ ] In `DefaultHouseMap.__init__`, before `self._finalize()`:
```python
        # Two-phase delivery goals: pick up at source, deliver at target.
        self.pickup_tiles = {
            "drink":   self.fixtures["fridge"],    # get a drink from the kitchen
            "package": self.fixtures["door"],      # a parcel arrives at the door
        }
        self.dropoff_tiles = {
            "drink":   self.fixtures["recliner"],  # hand it to the seated human
            "package": (16, 4),                    # bring it in to the kitchen table
        }
```
- [ ] Commit.

### Task 2: TaskManager state machine

**Files:** Rewrite `homebot3d/tasks.py`; Test `tests/test_tasks.py`

**Interfaces produced:** `TaskManager.phase: dict`, `.carrying: set`, `.current_goal_xy(goal)->(x,y)`; `reset(map,n_trash,rng,trash=None)`, `step(robot)->float`, `is_done()->bool`, `get_info(robot)->dict`.

- [ ] Replace the `drink_done`/`package_done` booleans with a phase machine. Full file body:
```python
import math
from homebot3d.maps import Map
from homebot3d.world import tile_center
from homebot3d.constants import REACH_RADIUS


def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


class TaskManager:
    CARRY_GOALS = ("drink", "package")

    def __init__(self, goals):
        self.goals = set(goals)
        self.trash_positions = []
        self.phase = {}        # carry_goal -> "seek_source" | "seek_target" | "done"
        self.carrying = set()  # carry_goals currently held (drives the cargo visual)

    def reset(self, map: Map, n_trash, rng, trash=None):
        self._map = map
        if "trash" not in self.goals:
            self.trash_positions = []
        elif trash is not None:
            self.trash_positions = list(trash)
        else:
            self.trash_positions = map.spawn_trash(
                n_trash, rng, exclude=list(map.fixtures.values()))
        self.phase = {g: "seek_source" for g in self.CARRY_GOALS if g in self.goals}
        self.carrying = set()

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
        for g in self.CARRY_GOALS:
            if g not in self.goals:
                continue
            if self.phase[g] == "seek_source":
                sx, sy = tile_center(*self._map.pickup_tiles[g])
                if _dist(robot.x, robot.y, sx, sy) <= REACH_RADIUS:
                    self.phase[g] = "seek_target"
                    self.carrying.add(g)
                    reward += 1.0
            elif self.phase[g] == "seek_target":
                tx, ty = tile_center(*self._map.dropoff_tiles[g])
                if _dist(robot.x, robot.y, tx, ty) <= REACH_RADIUS:
                    self.phase[g] = "done"
                    self.carrying.discard(g)
                    reward += 1.0
        return reward

    def is_done(self) -> bool:
        trash = "trash" not in self.goals or not self.trash_positions
        carry = all(self.phase.get(g) == "done"
                    for g in self.CARRY_GOALS if g in self.goals)
        return trash and carry

    def current_goal_xy(self, goal):
        """Phase-aware objective for a carry goal: source until picked up, else target."""
        if self.phase.get(goal) == "seek_source":
            return tile_center(*self._map.pickup_tiles[goal])
        return tile_center(*self._map.dropoff_tiles[goal])

    def get_info(self, robot) -> dict:
        return {
            "trash_remaining": len(self.trash_positions),
            "carrying": sorted(self.carrying),
            "drink_phase": self.phase.get("drink"),
            "package_phase": self.phase.get("package"),
        }
```
- [ ] Rewrite `tests/test_tasks.py` for the two-phase flow (drop the old `test_goal_targets_mapping` GOAL_TARGETS dict assert; see Task 3). New cases:
  - drink: reaching recliner FIRST (no pickup) → `0.0`, not done; reaching fridge → `1.0` and `"drink" in tm.carrying`; then reaching recliner → `1.0` and `is_done()`.
  - package: reaching door (source) → `1.0`, carrying; reaching `(16,4)` → `1.0`, done.
  - far away → `0.0`, not done.
  - trash: unchanged (`test_trash_targets_reached_individually`).
- [ ] Run `pytest tests/test_tasks.py -q`; commit.

### Task 3: goals.py — source/target coordinates

**Files:** Modify `homebot3d/goals.py`, `homebot3d/env.py` (import); Test `tests/test_tasks.py`

**Interfaces produced:** `goal_to_coordinates(goal,map,trash_positions=None,rng=None)` returns the DELIVERY target (drink→recliner, package→(16,4)); `GOAL_SOURCES` dict.

- [ ] Replace `GOAL_TARGETS`/`goal_to_coordinates` with map-driven lookups:
```python
from typing import Optional
from homebot3d.maps import Map
from homebot3d.world import tile_center

# Which carry goals exist and their human-readable source/target are defined by
# the map (pickup_tiles / dropoff_tiles). Trash is handled separately.
def goal_to_coordinates(goal, map: Map, trash_positions: Optional[list] = None,
                        rng=None) -> tuple[float, float]:
    """Delivery-target coordinates for a goal (a trash tile for the trash goal)."""
    if goal == "trash":
        if not trash_positions:
            raise ValueError("trash goal requires at least one trash position")
        idx = int(rng.integers(0, len(trash_positions))) if rng is not None else 0
        return tile_center(*trash_positions[idx])
    return tile_center(*map.dropoff_tiles[goal])
```
- [ ] Remove the `GOAL_TARGETS` import from `env.py` (Task 5 stops using it).
- [ ] Update `tests/test_tasks.py`: drop the exact-dict assert; keep `test_goal_to_coordinates_maps_drink_to_recliner` (still true — drink target is recliner).
- [ ] Commit.

### Task 4: Carried-item geoms on the robot

**Files:** Modify `homebot3d/world.py` (`_robot_body`); Test `tests/test_world_robot.py`

**Interfaces produced:** geoms `cargo_cup`, `cargo_box` on the robot body, visual-only, alpha 0 by default.

- [ ] In `_robot_body`, add just before the `<camera name="ego" .../>` line (they mount on top of the torso; `torso_z` is already computed there). Place the cup forward-right, the box centred, sitting above the torso top (`torso_z + ROBOT_BODY_HALFHEIGHT`):
```python
      <geom name="cargo_cup" type="cylinder" size="0.04 0.055"
            pos="0.09 0.05 {torso_z + ROBOT_BODY_HALFHEIGHT + 0.06}"
            contype="0" conaffinity="0" rgba="0.90 0.90 0.95 0"/>
      <geom name="cargo_box" type="box" size="0.10 0.10 0.08"
            pos="0 0 {torso_z + ROBOT_BODY_HALFHEIGHT + 0.09}"
            contype="0" conaffinity="0" rgba="0.72 0.55 0.35 0"/>
```
  (alpha `0` = hidden; env sets it to `1` while carrying.)
- [ ] Add `tests/test_world_robot.py::test_cargo_geoms_present_hidden_noncolliding`: both geoms exist, `geom_contype==0`, `geom_rgba[...,3]==0.0`.
- [ ] Run `pytest tests/test_world_robot.py -q`; commit.

### Task 5: Env wiring — phase-aware goal + cargo toggle

**Files:** Modify `homebot3d/env.py`; Test `tests/test_env.py` (or new `tests/test_pickup_deliver.py`)

**Interfaces consumed:** `TaskManager.current_goal_xy`, `.carrying`; geoms `cargo_cup`/`cargo_box`.

- [ ] In `reset_world`, after `self._robot = Robot(...)`, cache cargo geom ids:
```python
        self._cargo_gid = {
            "drink": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cargo_cup"),
            "package": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cargo_box"),
        }
```
- [ ] In `step_physics`, after `reward = float(self._tasks.step(self._robot))`, toggle visibility:
```python
        for g, gid in self._cargo_gid.items():
            self.model.geom_rgba[gid, 3] = 1.0 if g in self._tasks.carrying else 0.0
```
  (Also set them once in `reset_world` so a reset clears any leftover alpha.)
- [ ] Rewrite `_goal_xy` to be phase-aware and drop `GOAL_TARGETS`:
```python
    def _goal_xy(self):
        for g in ("drink", "package", "trash"):
            if g not in self.goals:
                continue
            if g == "trash":
                if self._tasks.trash_positions:
                    return tile_center(*self._tasks.trash_positions[0])
                continue
            return self._tasks.current_goal_xy(g)
        return (self._robot.x, self._robot.y)
```
- [ ] Add tests (`tests/test_pickup_deliver.py`): drive the env by teleporting the robot (`data.qpos` on slide joints, `mj_forward`) to source then target and assert reward sequence `+1` (pickup) then `+1` (deliver); assert `cargo_cup` alpha is `1.0` after pickup and `0.0` after delivery; assert `_goal_xy()` equals the source before pickup and the target after.
- [ ] Run full suite `pytest -q`; commit.

### Task 6: Visual verification (manual, not a test)

- [ ] Render with the robot near the fridge carrying a cup: reuse the scratch free-camera render pattern (see `scripts/render_dev.py::write_png`). Confirm the cup/box appears on the robot only while carrying. Drive-check with `PYTHONPATH="$(pwd)" python3 play.py --goals drink package`.

## Self-Review Notes
- Old `test_reaching_recliner_rewards_once_and_completes` MUST change — reaching the recliner without a pickup now yields 0. This is the main behavioral break; update it in Task 2.
- HER GoalEnv: `desired_goal` now moves once at pickup (source→target). `compute_reward` is distance-based so this is fine; the single-goal constraint is unchanged.
- `_min_tile_gap`/random-start exclusion uses `fixtures.values()` — sources/targets that are fixtures already excluded; `(16,4)` kitchen-table target is furniture (already an obstacle), fine.
- Keep the "collected trash stays rendered until reset" caveat from the prior merge; out of scope here.
