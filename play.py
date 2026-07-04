"""Custom GLFW teleop for HomeBot3DEnv — first-person POV with hold-to-move.

Controls: W/S drive fwd/back (held), A/D turn left/right (held),
V toggle POV/overview, R reset, Esc quit.

Movement keys are polled from physical key state each frame (glfw.get_key),
not reconstructed from press/release events — X11 key auto-repeat delivers
spurious release/press pairs while a key is held, which would otherwise make
held-to-move stutter or stall. Discrete actions (R/V/Esc) stay on the key
callback, where edge-triggering is what we want.

Uses its own GLFW/GLX window + MjrContext and steps physics via the env's
render-less seam (reset_world / step_physics), so it never touches the EGL
offscreen renderer used for observations.
"""
import argparse
import time
import glfw
import mujoco
from homebot3d.env import HomeBot3DEnv
from homebot3d.teleop_input import keys_to_action

_KEYMAP = {glfw.KEY_W: "w", glfw.KEY_A: "a", glfw.KEY_S: "s", glfw.KEY_D: "d"}


def _overview_camera(model, cam):
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = model.geom_pos[fid]
    cam.distance = 1.5 * model.stat.extent
    cam.azimuth = 90.0
    cam.elevation = -55.0


def _pov_camera(model, cam):
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--goals", nargs="+", default=["trash", "drink", "package"])
    p.add_argument("--map", default="default")
    p.add_argument("--random-start", action="store_true")
    p.add_argument("--max-steps", type=int, default=100000,
                   help="episode truncation cap; high so free-driving isn't cut short")
    args = p.parse_args()

    env = HomeBot3DEnv(goals=tuple(args.goals), map_name=args.map,
                       random_start=args.random_start, max_steps=args.max_steps)
    env.reset_world(seed=None)

    state = {"reset": False, "pov": True}

    def on_key(window, key, scancode, action, mods):
        # Discrete, edge-triggered actions only; movement is polled in the loop.
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_R:
            state["reset"] = True
        elif key == glfw.KEY_V:
            state["pov"] = not state["pov"]

    if not glfw.init():
        raise SystemExit("Failed to init GLFW")
    window = glfw.create_window(1200, 800, "HomeBot3D teleop", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("Failed to create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # vsync on: cap the loop to display refresh and show
    # whole frames. Without it an RTX-class GPU renders ~750 fps to a 60 Hz panel
    # and the monitor samples mid-swap — tearing that reads as jittery motion.
    glfw.set_key_callback(window, on_key)

    # One physics step per vsynced frame. Because swap_interval(1) pins the frame
    # period to the display refresh, a single fixed timestep per frame advances
    # the camera by an EQUAL increment every frame — smooth translation. A
    # wall-time accumulator instead does 1 step some frames and 2 on others (10 ms
    # steps vs a ~16.7 ms frame), and each double-step is a visible forward lurch
    # ("occasional pulses") that rotation hides but forward motion exposes. Match
    # the timestep to the refresh so one step per frame also runs ~real-time.
    mode = glfw.get_video_mode(glfw.get_primary_monitor())
    refresh = mode.refresh_rate if mode and mode.refresh_rate else 60
    teleop_dt = 1.0 / refresh
    env.model.opt.timestep = teleop_dt

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(env.model, maxgeom=10000)
    context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def apply_camera():
        (_pov_camera if state["pov"] else _overview_camera)(env.model, cam)

    apply_camera()
    fps_mark, fps_frames = time.time(), 0

    while not glfw.window_should_close(window):
        glfw.poll_events()
        # Poll physical key state for continuous movement (auto-repeat safe).
        held = {ch for key, ch in _KEYMAP.items()
                if glfw.get_key(window, key) == glfw.PRESS}
        action = keys_to_action(held)

        reward, term, _, _ = env.step_physics(action)
        if reward > 0:
            print(f"+{reward}")
        # Reset only on manual R or genuine termination (all goals reached) —
        # NOT on truncation, which would cut a manual drive short.
        if state["reset"] or term:
            env.reset_world(seed=None)
            # reset_world compiles a fresh model (MJCF timestep 0.01) — re-apply
            # the refresh-matched teleop timestep, then rebuild the GL context and
            # scene against the new model instance.
            env.model.opt.timestep = teleop_dt
            context.free()
            scene = mujoco.MjvScene(env.model, maxgeom=10000)
            context = mujoco.MjrContext(env.model,
                                        mujoco.mjtFontScale.mjFONTSCALE_150)
            state["reset"] = False
        apply_camera()

        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(env.model, env.data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)   # blocks until vblank: paces the loop, yields CPU

        # Effective render FPS once a second: should read ~refresh with vsync on;
        # a much lower number means a render bottleneck (software GL / heavy scene).
        fps_frames += 1
        now = time.time()
        if now - fps_mark >= 1.0:
            print(f"render {fps_frames / (now - fps_mark):.0f} fps")
            fps_mark, fps_frames = now, 0

    glfw.terminate()
    env.close()


if __name__ == "__main__":
    main()
