# HomeBot-3D — Embodiment, House Dressing & POV Teleop

**Date:** 2026-07-03
**Branch:** nav-spine
**Status:** Design — awaiting review

## Context

The nav spine works end-to-end (perception → action → reward, verified). The
world is currently visually skeletal: the robot is a bare green cylinder, the
"house" is grey boxes on a flat plane, the ego camera sits at ~0.22 m (ankle
height), and the only interactive driver (`play.py`) uses MuJoCo's passive
viewer, which cannot deliver key-release events — so teleop is a sticky
"go-until-stopped" experience rather than hold-to-drive.

This change makes the sim **look and feel like a robot in a house**, driven in
first person:

1. A recognizable primitive **robot body** with the camera on a mast.
2. A **house** that reads as a house — materials, lighting, furniture-shaped
   fixtures.
3. A custom **GLFW teleop** driver with true hold-to-move controls and a POV /
   overview camera toggle.

## Goals

- Robot reads as a mobile robot (TurtleBot/Roomba-class), built from MuJoCo
  primitives only — no meshes, no external assets, fully deterministic.
- Camera raised to a realistic sensor height (~0.6 m) so POV sees over furniture.
- House uses MuJoCo built-in procedural materials/textures + improved lighting;
  fixtures become furniture-shaped obstacles driven by the existing 2D grid map.
- `play.py` becomes a custom GLFW window with hold-to-move WASD, a camera-mode
  toggle (POV default ↔ overview), reset, and quit.
- No new third-party dependencies (`glfw` 3.4 already present via mujoco).
- Determinism and the existing action interface preserved.

## Non-Goals (explicit scope boundaries)

- **No change to drive dynamics.** The abstract diff-drive (slide_x, slide_y,
  yaw + velocity actuators) stays. Visible wheels are cosmetic; we do NOT switch
  to real wheel-contact/slip physics. True wheeled dynamics is a separate
  sim-to-real phase.
- **No real-robot import.** The primitive body is dimensioned so it can later be
  re-fitted to a real target platform, but importing a URDF/MJCF is out of scope.
- **No ceiling** (would darken the POV interior; open-top house keeps lighting
  simple).
- Headless EGL rendering (`scripts/render_dev.py`, training obs) is untouched.

## Component 1 — Robot embodiment

Primitive "Mid" build (~8 geoms), a nested body hierarchy under the existing
`robot` body so it all moves as one rigid unit driven by the same joints:

- **Drive base** — cylinder, radius `ROBOT_RADIUS`, low profile.
- **Body** — box torso on the base.
- **Sensor mast** — thin vertical box/cylinder.
- **Camera head** — small box at mast top; the `ego` camera rides here at
  `CAMERA_HEIGHT` (~0.6 m), still facing +x.
- **Heading wedge** — a small colored geom at the front so orientation is legible
  in both POV and overview.
- **Wheels** — two cosmetic side cylinders + a front caster sphere. Visual only;
  no wheel joints (drive stays abstract).

New constants in `constants.py`: `CAMERA_HEIGHT` and the robot part dimensions
(mast height, body/wheel sizes) so the body is defined in one place and can be
re-dimensioned for a real platform later.

Collision note: `collided()` filters to `wall_`/`fixture_` prefixes, so the added
robot child geoms do not register as collisions. The camera geom must not clip
the view (camera sits slightly forward of the head geom).

## Component 2 — House dressing

- **Materials/textures** via a MuJoCo `<asset>` block using **built-in procedural
  textures** (`builtin="checker"` floor, a wall material, `type="skybox"
  builtin="gradient"` sky). No external files → still zero-dep and deterministic.
- **Lighting** — add ambient and/or a second light and tune to kill the current
  blown-out overhead hotspot; verify via rendered frames.
- **Furniture-shaped fixtures** — the fridge/recliner/door fixtures become
  multi-geom furniture (e.g. recliner = seat + back) instead of plain boxes,
  still procedural, still positioned from the grid map, still `fixture_`-named so
  task/collision logic is unchanged. Furniture half-extents stay ≤ ~0.6·TILE so a
  goal at a fixture center remains reachable within `REACH_RADIUS` (0.75 m)
  without the robot needing to pass through the furniture.

## Component 3 — Render-less physics seam (env)

To let the GLFW (GLX) teleop coexist with EGL offscreen rendering without GL
context conflict, split rendering out of the physics step:

- `reset_world(seed) -> info` — rebuild model/data/robot/tasks, sample start,
  reset tasks. No camera creation, no obs.
- `step_physics(action) -> (reward, terminated, truncated, info)` — apply action,
  `mj_step`, compute reward/termination. No obs render.
- Gym `reset()` = `reset_world()` + create EGL `Camera` + return obs.
- Gym `step()` = `step_physics()` + render obs.

Teleop calls `reset_world` / `step_physics` and renders the window itself via its
own GLFW context + `MjrContext`, never creating the EGL `Camera`. Existing Gym
API behavior is unchanged (pure refactor; covered by existing env tests).

## Component 4 — Custom GLFW teleop (`play.py` rewrite)

A minimal interactive driver (desktop/display only):

- Creates a GLFW window + GL context, `MjrContext`, `MjvScene`, `MjvOption`, and a
  `MjvCamera`.
- **Controls (true hold-to-move):** GLFW key press/release maintains a held-keys
  set. **W/S** forward/back, **A/D** turn — additive while held, zero when
  released. **R** reset, **V** toggle camera mode, **Esc** quit.
- **Camera modes:** POV (camera fixed to the `ego` camera) as default; **V**
  cycles to an overview free-cam framed on the house. (Optional third mode: a
  third-person chase cam — include only if trivial.)
- Real-time paced loop (sleep off the remainder of one physics timestep, as in
  the current pacing fix).
- **Testable seam:** a pure function `keys_to_action(held: set[str]) -> np.ndarray`
  maps the held-keys set to the `[linear, angular]` action, unit-tested without a
  window. The GLFW loop is a thin shell around it.

## Determinism & collision

- Materials/textures/lighting are visual only — no effect on physics or RNG.
- Furniture adds static collision geometry; robot start sampling
  (`_sample_start_tile`) and goal reachability are re-verified after furniture is
  added (a scripted-drive frame check + the existing integration test).
- The env refactor is behavior-preserving for the Gym API.

## Testing strategy

- `keys_to_action` — unit tests (each key, combinations, empty set, clamping).
- Env refactor — existing env/goal-env/integration tests must stay green;
  add a test that `step_physics` matches `step`'s reward/termination without obs.
- World — update `test_world_*` for the new robot hierarchy, materials, and
  furniture geoms; add a test asserting the `ego` camera height ≈ `CAMERA_HEIGHT`.
- Determinism — existing determinism test must stay green.
- Visual verification — `scripts/render_dev.py` frames (ego + overview) inspected
  to confirm the robot, furniture, lighting, and POV height look right. Teleop
  window itself is verified by the user interactively (no automated GUI test).

## Risks

- **GL context conflict (EGL vs GLX)** — mitigated by Component 3 (teleop never
  creates the EGL camera).
- **Furniture traps the robot or blocks goals** — mitigated by size bound + start
  sampling + reachability re-check.
- **Teleop feel** — if GLFW hold-to-move still feels off, tuning lives in
  `keys_to_action` / velocity scaling, not the render loop.

## Future (out of scope, noted)

- Real-robot MJCF/URDF import matched to a chosen sim-to-real platform.
- True wheeled diff-drive dynamics (wheel joints + contact friction).
- Depth/segmentation overlays in the teleop HUD.
