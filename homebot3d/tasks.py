import math
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
