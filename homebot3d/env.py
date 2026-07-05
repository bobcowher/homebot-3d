import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from homebot3d.maps import MAP_REGISTRY
from homebot3d.world import compile_model, tile_center
from homebot3d.robot import Robot
from homebot3d.sensors import Camera, privileged
from homebot3d.tasks import TaskManager
from homebot3d.constants import TILE, ROBOT_RADIUS, REACH_RADIUS, OBS_SIZE


class HomeBot3DEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, goals=("trash", "drink", "package"), n_trash=2,
                 max_steps=2000, map_name="default", random_start=False,
                 width=OBS_SIZE, height=OBS_SIZE, render_mode=None):
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

    def reset_world(self, seed=None):
        rng = np.random.default_rng(seed)
        start = self._sample_start_tile(rng)
        # Spawn trash before compiling so the pieces render in the model, then hand
        # the same positions to the task manager so rewards match what's on screen.
        trash = (self._map.spawn_trash(
                    self.n_trash, rng, exclude=list(self._map.fixtures.values()))
                 if "trash" in self.goals else [])
        self.model = compile_model(self._map, robot_start=start, trash=trash)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._robot = Robot(self.model, self.data)
        self._cargo_gid = {
            "drink": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cargo_cup"),
            "package": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cargo_box"),
        }
        self._tasks.reset(self._map, self.n_trash, rng, trash=trash)
        # Map each spawned pile's tile to its geom ids so a collected pile can be
        # hidden (visual feedback that it was picked up).
        self._trash_gids = {}
        for i, tile in enumerate(trash):
            prefix = f"trash_{i}_"
            gids = [g for g in range(self.model.ngeom)
                    if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                        or "").startswith(prefix)]
            self._trash_gids[tuple(tile)] = gids
        # Clear any leftover cargo visibility so a reset starts with nothing carried.
        for gid in self._cargo_gid.values():
            self.model.geom_rgba[gid, 3] = 0.0
        self._steps = 0
        return self._info()

    def step_physics(self, action):
        self._robot.apply(np.asarray(action, dtype=np.float32))
        mujoco.mj_step(self.model, self.data)
        self._steps += 1
        reward = float(self._tasks.step(self._robot))
        for g, gid in self._cargo_gid.items():
            self.model.geom_rgba[gid, 3] = 1.0 if g in self._tasks.carrying else 0.0
        # Hide any pile that's been collected (no longer in the remaining set).
        remaining = set(map(tuple, self._tasks.trash_positions))
        for tile, gids in self._trash_gids.items():
            if tile not in remaining:
                for gid in gids:
                    self.model.geom_rgba[gid, 3] = 0.0
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


class HomeBot3DGoalEnv(HomeBot3DEnv):
    """GoalEnv wrapper for HER training.

    Supports single-goal HER only.  Multi-goal ``desired_goal`` advancement
    (i.e. automatically cycling through multiple sub-goals as each is reached)
    is deferred and not implemented; pass exactly one goal name.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.goals) != 1:
            raise ValueError(
                f"HomeBot3DGoalEnv supports single-goal HER only; got goals={self.goals}. "
                "Multi-goal desired_goal advancement is not implemented."
            )
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
