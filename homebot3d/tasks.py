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
        # trash positions are spawned by the env before model compile (so they can
        # be rendered) and passed in; fall back to spawning here if not provided.
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
