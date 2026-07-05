from typing import Optional
from homebot3d.maps import Map
from homebot3d.world import tile_center


# Which carry goals exist and their source/target tiles are defined by the map
# (pickup_tiles / dropoff_tiles). Trash is handled separately.
def goal_to_coordinates(goal, map: Map, trash_positions: Optional[list] = None,
                        rng=None) -> tuple[float, float]:
    """Delivery-target coordinates for a goal (a trash tile for the trash goal)."""
    if goal == "trash":
        if not trash_positions:
            raise ValueError("trash goal requires at least one trash position")
        idx = int(rng.integers(0, len(trash_positions))) if rng is not None else 0
        return tile_center(*trash_positions[idx])
    return tile_center(*map.dropoff_tiles[goal])
