import math
import numpy as np
from typing import Optional, Type

FLOOR = 0
WALL = 1
LAWN = 2   # outside ground beyond the front door: rendered as grass, no wall
           # panels, and not a valid floor tile (robot/trash never spawn there)


class Map:
    tiles: np.ndarray
    fixtures: dict           # {name: (col, row)}
    robot_start_tile: tuple  # (col, row)
    door_tiles: list
    doorways: list           # [(axis, line, lo, hi)] — drives door-frame posts
    furniture: list          # [(kind, col, row)] — non-goal obstacle pieces
    pickup_tiles: dict       # {carry_goal: (col,row)} where the item is picked up
    dropoff_tiles: dict      # {carry_goal: (col,row)} where it is delivered
    fixture_euler: dict = {}  # {name: yaw_radians} — rotate a fixture body about z

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
        t[:, 24:27] = LAWN         # outside lawn beyond the front door (view-only)
        self.tiles = t
        self.fixtures = {
            "fridge":   (19, 1),
            "recliner": (5,  5),
            "door":     (23, 9),
        }
        self.door_tiles = [(23, 9), (23, 10)]
        # (axis, line, lo, hi): "h" = horizontal wall line at row=line spanning
        # columns lo..hi; "v" = vertical wall line at col=line spanning rows lo..hi.
        self.doorways = [
            ("v", 11, 4, 5),    # living <-> kitchen
            ("h", 8, 4, 6),     # living  -> hallway
            ("h", 8, 14, 16),   # kitchen -> hallway
            ("h", 11, 10, 12),  # hallway -> bedroom
            ("v", 23, 9, 10),   # east exit
        ]
        self.robot_start_tile = (8, 4)
        # The recliner+human are rotated 180° to face -y (north), toward a TV on
        # the north wall — a bachelor watching a movie. Keeping the setup against
        # the north wall leaves the south (row-8) doorway clear. No sofa.
        self.fixture_euler = {"recliner": math.pi}
        # (kind, col, row) — non-goal obstacle furniture, one-body each. Layout
        # mirrors the 2D DefaultHouseMap: TV and side table in the living room,
        # sink+counter+fridge along the kitchen's north wall. The bed and
        # nightstand furnish the bedroom (a 3D-only touch the 2D map lacks).
        self.furniture = [
            ("tv",            5,  1),   # living — north wall, faces the human (+y)
            ("coffee_table",  3,  5),   # living — side table west of the recliner
            ("sink",         15,  1),   # kitchen — north wall
            ("counter",      17,  1),   # kitchen — north wall, sink <-> fridge
            ("bed",           4, 14),   # bedroom
            ("nightstand",    2, 14),   # bedroom
        ]
        # Two-phase delivery goals: pick up at source, deliver at target.
        self.pickup_tiles = {
            "drink":   self.fixtures["fridge"],    # get a drink from the kitchen
            "package": self.fixtures["door"],      # a parcel arrives at the door
        }
        self.dropoff_tiles = {
            "drink":   self.fixtures["recliner"],  # hand it to the seated human
            "package": self.fixtures["recliner"],  # bring the parcel in to the human too
        }
        self._finalize()


MAP_REGISTRY: dict[str, Type[Map]] = {
    "default": DefaultHouseMap,
}
