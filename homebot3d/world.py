from functools import lru_cache
from pathlib import Path

import mujoco

from homebot3d.maps import Map, WALL, FLOOR
from homebot3d.constants import (
    TILE, WALL_HEIGHT, WALL_THICK, ROBOT_RADIUS, ROBOT_HALFHEIGHT,
    CAMERA_HEIGHT, EGO_FOVY, ROBOT_BODY_HALF, ROBOT_BODY_HALFHEIGHT, WHEEL_RADIUS,
)

_TEX_DIR = Path(__file__).parent / "assets" / "textures"
_TEX_FILES = ("floor_wood.png", "floor_tile.png", "wall_paint.png",
              "wood.png", "fabric.png")


def tile_center(col: int, row: int) -> tuple[float, float]:
    return (col + 0.5) * TILE, (row + 0.5) * TILE


def _asset_block() -> str:
    return """
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.55 0.7 0.9" rgb2="0.1 0.12 0.2" width="256" height="256"/>
    <texture name="floor_wood" type="2d" file="floor_wood.png"/>
    <texture name="floor_tile" type="2d" file="floor_tile.png"/>
    <texture name="wall_paint" type="2d" file="wall_paint.png"/>
    <texture name="wood" type="2d" file="wood.png"/>
    <texture name="fabric" type="2d" file="fabric.png"/>
    <material name="floormat" texture="floor_wood" texrepeat="6 4" reflectance="0.05"/>
    <material name="tilemat"  texture="floor_tile" texrepeat="6 4" reflectance="0.1"/>
    <material name="wallmat"  texture="wall_paint" texrepeat="4 2" reflectance="0.02"/>
    <material name="woodmat"  texture="wood"       texrepeat="2 2" reflectance="0.05"/>
    <material name="fabricmat" texture="fabric"    texrepeat="2 2" reflectance="0.02"/>
  </asset>"""


def _wall_geoms(map: Map) -> str:
    """Thin boundary wall panels.

    For each floor cell, emit a thin panel on every edge shared with a wall
    cell or the grid border. Panels are WALL_THICK thick on the boundary-normal
    axis and slightly over-long on their tangent axis so perpendicular panels
    overlap and close corner gaps. All named wall_{i} so Robot.collided()
    detects them.
    """
    rows, cols = map.tiles.shape
    hz = WALL_HEIGHT / 2
    thin = WALL_THICK / 2
    long = TILE / 2 + WALL_THICK / 2      # tangent half-extent with corner overlap
    parts = []
    i = 0

    def is_wall(r, c):
        return (r < 0 or r >= rows or c < 0 or c >= cols
                or map.tiles[r, c] == WALL)

    for r in range(rows):
        for c in range(cols):
            if map.tiles[r, c] != FLOOR:
                continue
            cx, cy = tile_center(c, r)
            edges = []
            if is_wall(r - 1, c):                    # north (smaller y)
                edges.append((cx, r * TILE, long, thin))
            if is_wall(r + 1, c):                    # south (larger y)
                edges.append((cx, (r + 1) * TILE, long, thin))
            if is_wall(r, c - 1):                    # west (smaller x)
                edges.append((c * TILE, cy, thin, long))
            if is_wall(r, c + 1):                    # east (larger x)
                edges.append(((c + 1) * TILE, cy, thin, long))
            for (px, py, hx, hy) in edges:
                parts.append(
                    f'<geom name="wall_{i}" type="box" '
                    f'size="{hx} {hy} {hz}" pos="{px} {py} {hz}" '
                    f'material="wallmat"/>'
                )
                i += 1
    return "\n".join(parts)


def _door_frames(map: Map) -> str:
    """Cosmetic-but-collidable jamb posts flanking each doorway opening."""
    hz = WALL_HEIGHT / 2
    post = WALL_THICK                     # chunkier square footprint than a panel
    parts = []
    for i, (axis, line, lo, hi) in enumerate(getattr(map, "doorways", [])):
        if axis == "h":                   # horizontal wall line, opening cols lo..hi
            y = (line + 0.5) * TILE
            centres = [(lo * TILE, y), ((hi + 1) * TILE, y)]
        else:                             # vertical wall line, opening rows lo..hi
            x = (line + 0.5) * TILE
            centres = [(x, lo * TILE), (x, (hi + 1) * TILE)]
        for j, (px, py) in enumerate(centres):
            parts.append(
                f'<geom name="wall_frame_{i}_{j}" type="box" '
                f'size="{post} {post} {hz}" pos="{px} {py} {hz}" '
                f'rgba="0.30 0.22 0.14 1"/>'
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
            f'pos="0 0 0.12" material="fabricmat"/>'
            f'<geom name="fixture_recliner_back" type="box" size="0.28 0.06 0.20" '
            f'pos="0 -0.22 0.32" material="fabricmat"/>'
        )
    if name == "door":
        return (
            f'<geom name="fixture_door_panel" type="box" size="0.28 0.05 0.5" '
            f'pos="0 0 0.5" material="woodmat"/>'
            f'<geom name="fixture_door_knob" type="sphere" size="0.03" '
            f'pos="0.18 0.06 0.5" rgba="0.85 0.7 0.2 1"/>'
        )
    # Fallback: a plain box for any other fixture name.
    return (
        f'<geom name="fixture_{name}_box" type="box" size="0.28 0.28 0.3" '
        f'pos="0 0 0.3" rgba="0.5 0.5 0.5 1"/>'
    )


def _furniture_piece(kind: str, idx: int) -> str:
    """Textured multi-geom furniture, geoms local to a body at floor level.

    Every geom's |offset| + half-extent stays <= REACH_RADIUS - ROBOT_RADIUS
    (0.57 m) on x and y so a goal at the piece centre remains reachable and the
    piece never walls off a room.
    """
    p = f"fixture_{kind}_{idx}"
    if kind == "sofa":
        return (
            f'<geom name="{p}_seat" type="box" size="0.5 0.22 0.12" '
            f'pos="0 0 0.12" material="fabricmat"/>'
            f'<geom name="{p}_back" type="box" size="0.5 0.07 0.18" '
            f'pos="0 -0.18 0.28" material="fabricmat"/>'
        )
    if kind == "coffee_table":
        return (
            f'<geom name="{p}_top" type="box" size="0.28 0.18 0.03" '
            f'pos="0 0 0.22" material="woodmat"/>'
            f'<geom name="{p}_leg" type="box" size="0.24 0.14 0.11" '
            f'pos="0 0 0.11" material="woodmat"/>'
        )
    if kind == "counter":
        return (
            f'<geom name="{p}_base" type="box" size="0.5 0.22 0.40" '
            f'pos="0 0 0.40" material="woodmat"/>'
            f'<geom name="{p}_top" type="box" size="0.5 0.22 0.02" '
            f'pos="0 0 0.42" material="tilemat"/>'
        )
    if kind == "kitchen_table":
        return (
            f'<geom name="{p}_top" type="box" size="0.35 0.35 0.03" '
            f'pos="0 0 0.35" material="woodmat"/>'
            f'<geom name="{p}_leg" type="box" size="0.30 0.30 0.16" '
            f'pos="0 0 0.16" material="woodmat"/>'
        )
    if kind == "bed":
        return (
            f'<geom name="{p}_mattress" type="box" size="0.5 0.5 0.12" '
            f'pos="0 0 0.16" material="fabricmat"/>'
            f'<geom name="{p}_headboard" type="box" size="0.5 0.06 0.22" '
            f'pos="0 -0.45 0.22" material="woodmat"/>'
        )
    if kind == "nightstand":
        return (
            f'<geom name="{p}_box" type="box" size="0.15 0.15 0.25" '
            f'pos="0 0 0.25" material="woodmat"/>'
        )
    # Fallback: a plain textured box.
    return (
        f'<geom name="{p}_box" type="box" size="0.28 0.28 0.28" '
        f'pos="0 0 0.28" material="woodmat"/>'
    )


def _furniture_bodies(map: Map) -> str:
    parts = []
    for idx, (kind, col, row) in enumerate(getattr(map, "furniture", [])):
        cx, cy = tile_center(col, row)
        parts.append(
            f'<body name="fixture_{kind}_{idx}" pos="{cx} {cy} 0">'
            f'{_furniture_piece(kind, idx)}</body>'
        )
    return "\n".join(parts)


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
      <camera name="ego" pos="{ROBOT_RADIUS} 0 {head_z}" xyaxes="0 -1 0 0 0 1" fovy="{EGO_FOVY}"/>
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
{_door_frames(map)}
{_fixture_bodies(map)}
{_furniture_bodies(map)}
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


@lru_cache(maxsize=1)
def texture_assets() -> dict[str, bytes]:
    """Committed texture PNGs as filename -> bytes, for MjModel asset injection.

    Cached: the PNGs are static committed bytes, so read them once and reuse the
    dict across every compile_model call (reset_world compiles on each episode
    reset — a reset-heavy RL loop would otherwise re-read ~285 KB every reset).
    """
    return {name: (_TEX_DIR / name).read_bytes() for name in _TEX_FILES}


def compile_model(map: Map, robot_start=None) -> mujoco.MjModel:
    """Compile the house MJCF with texture bytes supplied inline (no fs lookup)."""
    return mujoco.MjModel.from_xml_string(
        build_mjcf(map, robot_start), texture_assets())
