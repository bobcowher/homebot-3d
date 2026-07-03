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
