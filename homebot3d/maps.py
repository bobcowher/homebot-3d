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
        # (kind, col, row) — non-goal obstacle furniture, one-body each.
        self.furniture = [
            ("sofa",          3, 3),    # living
            ("coffee_table",  5, 3),    # living
            ("counter",      20, 3),    # kitchen
            ("kitchen_table",16, 4),    # kitchen
            ("bed",           4, 14),   # bedroom
            ("nightstand",    2, 14),   # bedroom
        ]
        self._finalize()


MAP_REGISTRY: dict[str, Type[Map]] = {
    "default": DefaultHouseMap,
}
