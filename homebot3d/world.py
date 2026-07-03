import numpy as np
from homebot3d.maps import Map, WALL
from homebot3d.constants import TILE, WALL_HEIGHT, ROBOT_RADIUS, ROBOT_HALFHEIGHT


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


def _robot_body(map: Map, robot_start) -> str:
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


def build_mjcf(map: Map, robot_start=None) -> str:
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
{_fixture_bodies(map)}
{_robot_body(map, robot_start)}
  </worldbody>
  <actuator>
    <velocity name="vx" joint="slide_x" kv="20" ctrlrange="-2 2"/>
    <velocity name="vy" joint="slide_y" kv="20" ctrlrange="-2 2"/>
    <velocity name="wz" joint="yaw" kv="2" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""
