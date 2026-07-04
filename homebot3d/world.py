from homebot3d.maps import Map, WALL
from homebot3d.constants import (
    TILE, WALL_HEIGHT, ROBOT_RADIUS, ROBOT_HALFHEIGHT,
    CAMERA_HEIGHT, ROBOT_BODY_HALF, ROBOT_BODY_HALFHEIGHT, WHEEL_RADIUS,
)


def tile_center(col: int, row: int) -> tuple[float, float]:
    return (col + 0.5) * TILE, (row + 0.5) * TILE


def _asset_block() -> str:
    return """
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.55 0.7 0.9" rgb2="0.1 0.12 0.2" width="256" height="256"/>
    <texture name="floortex" type="2d" builtin="checker"
             rgb1="0.82 0.80 0.76" rgb2="0.70 0.68 0.64" width="512" height="512"/>
    <material name="floormat" texture="floortex" texrepeat="8 8" reflectance="0.1"/>
    <material name="wallmat" rgba="0.86 0.86 0.88 1" reflectance="0.05"/>
  </asset>"""


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
                    f'material="wallmat"/>'
                )
    return "\n".join(parts)


def _furniture_geoms(name: str) -> str:
    """Multi-geom procedural furniture, all geoms named fixture_{name}_*.

    Positions are relative to the fixture body origin at floor level (z=0).
    All x/y half-extents stay <= 0.3 m so a goal at the fixture centre is
    reachable within REACH_RADIUS.
    """
    if name == "fridge":
        return (
            f'<geom name="fixture_fridge_body" type="box" size="0.22 0.22 0.45" '
            f'pos="0 0 0.45" rgba="0.72 0.78 0.85 1"/>'
            f'<geom name="fixture_fridge_door" type="box" size="0.02 0.20 0.4" '
            f'pos="0.22 0 0.5" rgba="0.6 0.66 0.74 1"/>'
        )
    if name == "recliner":
        return (
            f'<geom name="fixture_recliner_seat" type="box" size="0.28 0.28 0.12" '
            f'pos="0 0 0.12" rgba="0.5 0.3 0.2 1"/>'
            f'<geom name="fixture_recliner_back" type="box" size="0.28 0.06 0.20" '
            f'pos="0 -0.22 0.32" rgba="0.45 0.27 0.18 1"/>'
        )
    if name == "door":
        return (
            f'<geom name="fixture_door_panel" type="box" size="0.28 0.05 0.5" '
            f'pos="0 0 0.5" rgba="0.32 0.22 0.12 1"/>'
            f'<geom name="fixture_door_knob" type="sphere" size="0.03" '
            f'pos="0.18 0.06 0.5" rgba="0.85 0.7 0.2 1"/>'
        )
    # Fallback: a plain box for any other fixture name.
    return (
        f'<geom name="fixture_{name}_box" type="box" size="0.28 0.28 0.3" '
        f'pos="0 0 0.3" rgba="0.5 0.5 0.5 1"/>'
    )


def _fixture_bodies(map: Map) -> str:
    parts = []
    for name, (col, row) in map.fixtures.items():
        cx, cy = tile_center(col, row)
        parts.append(
            f'<body name="fixture_{name}" pos="{cx} {cy} 0">'
            f'{_furniture_geoms(name)}</body>'
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
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0 0 0"/>
  </visual>
{_asset_block()}
  <worldbody>
    <light directional="true" pos="0 0 5" dir="0.2 0.3 -1" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" pos="{fx/2} {fy/2} 0"
          size="{fx/2} {fy/2} 0.1" material="floormat"/>
{_wall_geoms(map)}
{_fixture_bodies(map)}
{_robot_body(map, robot_start)}
  </worldbody>
  <actuator>
    <!-- kv tuned so the velocity servo tracks commanded speed promptly:
         tau = mass/kv ~= 24.4/80 ~= 0.3s (was kv=20 -> 1.2s, felt sluggish). -->
    <velocity name="vx" joint="slide_x" kv="80" ctrlrange="-2 2"/>
    <velocity name="vy" joint="slide_y" kv="80" ctrlrange="-2 2"/>
    <velocity name="wz" joint="yaw" kv="2" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""
