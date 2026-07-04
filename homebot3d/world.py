from functools import lru_cache
from pathlib import Path

import mujoco

from homebot3d.maps import Map, WALL, FLOOR
from homebot3d.constants import (
    TILE, WALL_HEIGHT, WALL_THICK, ROBOT_RADIUS, ROBOT_HALFHEIGHT,
    CAMERA_HEIGHT, EGO_FOVY, EGO_CAM_BACK, EGO_CAM_RAISE,
    ROBOT_BODY_HALF, ROBOT_BODY_HALFHEIGHT, WHEEL_RADIUS,
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


def _merge_runs(indices) -> list[tuple[int, int]]:
    """Collapse a set of ints into (start, end) inclusive contiguous runs."""
    runs: list[list[int]] = []
    for k in sorted(indices):
        if runs and k == runs[-1][1] + 1:
            runs[-1][1] = k
        else:
            runs.append([k, k])
    return [(a, b) for a, b in runs]


def _wall_box(i: int, px: float, py: float, hx: float, hy: float, hz: float) -> str:
    return (f'<geom name="wall_{i}" type="box" '
            f'size="{hx} {hy} {hz}" pos="{px} {py} {hz}" material="wallmat"/>')


def _wall_geoms(map: Map) -> str:
    """Thin single-panel walls on each wall tile's centreline.

    A 1-tile-thick wall line becomes ONE thin panel centred on the tile rather
    than two panels on its floor-facing faces (which read as a 0.5 m-thick wall
    and z-fought where adjacent cells' panels overlapped). Each wall tile that
    borders floor is classified horizontal- and/or vertical-type; collinear
    tiles merge into runs so no two panels share a coplanar face.

    Run ends are joined so corners and doorways are clean: at a perpendicular
    wall (a corner/junction) the panel extends to that wall's centreline (they
    meet exactly, no gap, no overshoot); at an opening (floor beyond the end)
    it extends to the tile edge — the doorjamb the frame post sits on. Panels
    are WALL_THICK thick, named wall_{i} so Robot.collided() detects them.
    """
    rows, cols = map.tiles.shape
    hz = WALL_HEIGHT / 2
    thin = WALL_THICK / 2

    def is_floor(r, c):
        return 0 <= r < rows and 0 <= c < cols and map.tiles[r, c] == FLOOR

    def is_wall(r, c):                     # out-of-bounds (border) counts as wall
        return not (0 <= r < rows and 0 <= c < cols) or map.tiles[r, c] == WALL

    h_tiles: dict[int, set] = {}          # row -> cols carrying a horizontal panel
    v_tiles: dict[int, set] = {}          # col -> rows carrying a vertical panel
    for r in range(rows):
        for c in range(cols):
            if map.tiles[r, c] != WALL:
                continue
            if is_floor(r - 1, c) or is_floor(r + 1, c):
                h_tiles.setdefault(r, set()).add(c)
            if is_floor(r, c - 1) or is_floor(r, c + 1):
                v_tiles.setdefault(c, set()).add(r)

    parts = []
    i = 0
    for r, cset in sorted(h_tiles.items()):
        cy = (r + 0.5) * TILE
        for c0, c1 in _merge_runs(cset):
            # perpendicular wall -> meet its centreline; opening -> tile edge
            x0 = (c0 - 0.5) * TILE if is_wall(r, c0 - 1) else c0 * TILE
            x1 = (c1 + 1.5) * TILE if is_wall(r, c1 + 1) else (c1 + 1) * TILE
            parts.append(_wall_box(i, (x0 + x1) / 2, cy, (x1 - x0) / 2, thin, hz))
            i += 1
    for c, rset in sorted(v_tiles.items()):
        cx = (c + 0.5) * TILE
        for r0, r1 in _merge_runs(rset):
            y0 = (r0 - 0.5) * TILE if is_wall(r0 - 1, c) else r0 * TILE
            y1 = (r1 + 1.5) * TILE if is_wall(r1 + 1, c) else (r1 + 1) * TILE
            parts.append(_wall_box(i, cx, (y0 + y1) / 2, thin, (y1 - y0) / 2, hz))
            i += 1
    return "\n".join(parts)


def _door_frames(map: Map) -> str:
    """Slim dark-wood jamb posts framing each doorway opening.

    Posts match the wall thickness (half = WALL_THICK/2) so they sit flush with
    the wall faces — no jutting. Each post is offset by its half-width into the
    opening so it abuts the wall's end face rather than overlapping it; an
    overlap would put coplanar dark/white faces at the same depth and z-fight.
    """
    hz = WALL_HEIGHT / 2
    half = WALL_THICK / 2                  # flush with a wall panel's thickness
    parts = []
    for i, (axis, line, lo, hi) in enumerate(getattr(map, "doorways", [])):
        if axis == "h":                   # horizontal wall line, opening cols lo..hi
            y = (line + 0.5) * TILE
            centres = [(lo * TILE + half, y), ((hi + 1) * TILE - half, y)]
        else:                             # vertical wall line, opening rows lo..hi
            x = (line + 0.5) * TILE
            centres = [(x, lo * TILE + half), (x, (hi + 1) * TILE - half)]
        for j, (px, py) in enumerate(centres):
            parts.append(
                f'<geom name="wall_frame_{i}_{j}" type="box" '
                f'size="{half} {half} {hz}" pos="{px} {py} {hz}" '
                f'rgba="0.30 0.22 0.14 1"/>'
            )
    return "\n".join(parts)


def _seated_human() -> str:
    """A capsule-based seated human, geoms local to the recliner body (facing +y).

    Rounded torso, jointed arms resting on the lap, thighs/shins to the floor, and
    feet. All geoms fixture_human_* (collidable) and within ~0.4 m of centre.
    """
    skin = "0.82 0.62 0.50 1"
    shirt = "0.24 0.44 0.62 1"
    pants = "0.26 0.26 0.32 1"
    shoe = "0.15 0.12 0.10 1"

    def cap(n, x1, y1, z1, x2, y2, z2, rad, rgba):
        return (f'<geom name="fixture_human_{n}" type="capsule" '
                f'fromto="{x1} {y1} {z1} {x2} {y2} {z2}" size="{rad}" rgba="{rgba}"/>')

    def sph(n, x, y, z, rad, rgba):
        return (f'<geom name="fixture_human_{n}" type="sphere" size="{rad}" '
                f'pos="{x} {y} {z}" rgba="{rgba}"/>')

    def box(n, x, y, z, sx, sy, sz, rgba):
        return (f'<geom name="fixture_human_{n}" type="box" size="{sx} {sy} {sz}" '
                f'pos="{x} {y} {z}" rgba="{rgba}"/>')

    return "".join([
        cap("torso",     0, -0.04, 0.30,  0, -0.09, 0.60, 0.13, shirt),
        cap("shoulders", -0.14, -0.07, 0.585,  0.14, -0.07, 0.585, 0.05, shirt),
        cap("neck",      0, -0.06, 0.60,  0, -0.05, 0.68, 0.04, skin),
        sph("head",      0, -0.04, 0.78, 0.10, skin),
        cap("uarm_l",    0.14, -0.07, 0.585,  0.17, 0.00, 0.42, 0.045, shirt),
        cap("farm_l",    0.17, 0.00, 0.42,  0.13, 0.17, 0.33, 0.04, skin),
        cap("uarm_r",    -0.14, -0.07, 0.585,  -0.17, 0.00, 0.42, 0.045, shirt),
        cap("farm_r",    -0.17, 0.00, 0.42,  -0.13, 0.17, 0.33, 0.04, skin),
        # legs drape forward over the seat front (y>0.28) so knees/shins/feet show
        cap("thigh_l",   0.07, 0.02, 0.30,  0.08, 0.31, 0.25, 0.06, pants),
        cap("thigh_r",   -0.07, 0.02, 0.30,  -0.08, 0.31, 0.25, 0.06, pants),
        cap("shin_l",    0.08, 0.31, 0.25,  0.09, 0.34, 0.07, 0.05, pants),
        cap("shin_r",    -0.08, 0.31, 0.25,  -0.09, 0.34, 0.07, 0.05, pants),
        box("foot_l",    0.08, 0.39, 0.03, 0.05, 0.08, 0.03, shoe),
        box("foot_r",    -0.08, 0.39, 0.03, 0.05, 0.08, 0.03, shoe),
    ])


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
        # Recliner (seat + back) with a seated human the robot serves, facing +y
        # (away from the back at -y). The human's geoms are fixture_human_* so
        # they collide and register like any fixture; all stay within ~0.4 m of
        # centre so the drink goal is still reachable.
        return (
            f'<geom name="fixture_recliner_seat" type="box" size="0.28 0.28 0.12" '
            f'pos="0 0 0.12" material="fabricmat"/>'
            f'<geom name="fixture_recliner_back" type="box" size="0.28 0.06 0.20" '
            f'pos="0 -0.22 0.32" material="fabricmat"/>'
            + _seated_human()
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


# A messy little litter pile: several varied bits (crumpled paper, wrappers, a
# tipped can) at fixed offsets so each trash tile reads as a pile, not a dot.
# (dx, dy, z, type, size, rgba, euler) — offsets stay within ~0.13 m of centre.
_TRASH_PILE = [
    (0.00,  0.00, 0.055, "sphere",   "0.06",              "0.88 0.86 0.80 1", ""),        # crumpled paper
    (0.10,  0.05, 0.035, "box",      "0.055 0.035 0.035", "0.74 0.28 0.22 1", "0 0 0.5"), # red wrapper
    (-0.09, -0.06, 0.03, "box",      "0.045 0.045 0.03",  "0.52 0.58 0.32 1", "0 0 -0.7"),# green scrap
    (0.03, -0.10, 0.03,  "cylinder", "0.03 0.05",         "0.30 0.52 0.72 1", "1.4 0 0"), # tipped can
    (-0.07, 0.08, 0.03,  "sphere",   "0.038",             "0.62 0.60 0.54 1", ""),        # small wad
    (0.07, -0.02, 0.025, "box",      "0.03 0.05 0.025",   "0.80 0.74 0.40 1", "0 0 1.1"), # yellow flyer
]


def _trash_geoms(trash) -> str:
    """A visual litter pile on the floor at each trash tile.

    Visual-only (contype/conaffinity 0) so the robot drives onto a pile to
    "collect" it (reward is reach-based). Named trash_* — NOT the wall_/fixture_
    prefixes, so Robot.collided() ignores them.
    """
    parts = []
    for i, (col, row) in enumerate(trash or []):
        cx, cy = tile_center(col, row)
        for j, (dx, dy, z, typ, size, rgba, euler) in enumerate(_TRASH_PILE):
            e = f' euler="{euler}"' if euler else ""
            parts.append(
                f'<geom name="trash_{i}_{j}" type="{typ}" size="{size}" '
                f'pos="{cx + dx} {cy + dy} {z}"{e} contype="0" conaffinity="0" '
                f'rgba="{rgba}"/>'
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
      <!-- ego: chase view set back/above the robot so its own body shows at the
           frame bottom. xyaxes tilts the view down ~12deg (up vector 0.21 0 0.98). -->
      <camera name="ego" pos="{-EGO_CAM_BACK} 0 {head_z + EGO_CAM_RAISE}" xyaxes="0 -1 0 0.21 0 0.98" fovy="{EGO_FOVY}"/>
    </body>"""


def build_mjcf(map: Map, robot_start=None, trash=None) -> str:
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
{_trash_geoms(trash)}
{_robot_body(map, robot_start)}
  </worldbody>
  <actuator>
    <!-- kv tuned so the velocity servo tracks commanded speed promptly:
         tau = mass/kv ~= 24.4/80 ~= 0.3s (was kv=20 -> 1.2s, felt sluggish). -->
    <velocity name="vx" joint="slide_x" kv="80" ctrlrange="-2 2"/>
    <velocity name="vy" joint="slide_y" kv="80" ctrlrange="-2 2"/>
    <!-- wz kv=20 (was 2): a stiff yaw servo holds heading against contact
         torque. With kv=2 a glancing bump on a doorframe delivered enough
         angular impulse to spin the robot past 90deg so "forward" drove it
         into the wall corner — a permanent wedge ("sticking at stop"). kv=20
         absorbs the impulse (doorway pass-through verified) and also sharpens
         turning (steady rate ~1.95 vs ~1.1 rad/s); kv>=50 goes unstable. -->
    <velocity name="wz" joint="yaw" kv="20" ctrlrange="-3 3"/>
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


def compile_model(map: Map, robot_start=None, trash=None) -> mujoco.MjModel:
    """Compile the house MJCF with texture bytes supplied inline (no fs lookup)."""
    return mujoco.MjModel.from_xml_string(
        build_mjcf(map, robot_start, trash), texture_assets())
