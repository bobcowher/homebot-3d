from homebot3d.maps import Map, WALL
from homebot3d.constants import (
    TILE, WALL_HEIGHT, ROBOT_RADIUS, ROBOT_HALFHEIGHT,
    CAMERA_HEIGHT, ROBOT_BODY_HALF, ROBOT_BODY_HALFHEIGHT, WHEEL_RADIUS,
)


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

    base_top = ROBOT_HALFHEIGHT                      # local z of base cylinder top
    torso_z = base_top + ROBOT_BODY_HALFHEIGHT       # torso box centre (local z)
    head_z = CAMERA_HEIGHT - ROBOT_HALFHEIGHT        # camera / head centre (local z)
    mast_bottom = torso_z + ROBOT_BODY_HALFHEIGHT
    mast_cz = (mast_bottom + head_z) / 2
    mast_hz = max((head_z - mast_bottom) / 2, 0.01)
    wheel_z = -ROBOT_HALFHEIGHT + WHEEL_RADIUS
    fwd = ROBOT_RADIUS - 0.03
    vis = 'contype="0" conaffinity="0" density="0"'  # visual-only: no mass, no collision

    return f"""
    <body name="robot" pos="{cx} {cy} {z}">
      <joint name="slide_x" type="slide" axis="1 0 0"/>
      <joint name="slide_y" type="slide" axis="0 1 0"/>
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom name="robot_body" type="cylinder"
            size="{ROBOT_RADIUS} {ROBOT_HALFHEIGHT}" rgba="0.25 0.5 0.75 1"/>
      <geom name="robot_torso" type="box" {vis}
            size="{ROBOT_BODY_HALF} {ROBOT_BODY_HALF} {ROBOT_BODY_HALFHEIGHT}"
            pos="0 0 {torso_z}" rgba="0.9 0.9 0.92 1"/>
      <geom name="robot_mast" type="box" {vis}
            size="0.02 0.02 {mast_hz}" pos="0 0 {mast_cz}" rgba="0.3 0.3 0.32 1"/>
      <geom name="robot_head" type="box" {vis}
            size="0.05 0.06 0.04" pos="0 0 {head_z}" rgba="0.15 0.15 0.17 1"/>
      <geom name="robot_wedge" type="box" {vis}
            size="0.05 0.03 0.03" pos="{fwd} 0 {base_top}" rgba="0.95 0.55 0.1 1"/>
      <geom name="robot_wheel_l" type="cylinder" {vis} zaxis="0 1 0"
            size="{WHEEL_RADIUS} 0.02" pos="0 {ROBOT_RADIUS} {wheel_z}"
            rgba="0.1 0.1 0.1 1"/>
      <geom name="robot_wheel_r" type="cylinder" {vis} zaxis="0 1 0"
            size="{WHEEL_RADIUS} 0.02" pos="0 {-ROBOT_RADIUS} {wheel_z}"
            rgba="0.1 0.1 0.1 1"/>
      <geom name="robot_caster" type="sphere" {vis} size="0.03"
            pos="{fwd} 0 {-ROBOT_HALFHEIGHT + 0.03}" rgba="0.1 0.1 0.1 1"/>
      <camera name="ego" pos="{ROBOT_RADIUS} 0 {head_z}" xyaxes="0 -1 0 0 0 1"/>
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
