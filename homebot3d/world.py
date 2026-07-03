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
