"""Headless smoke for the play.py render/step/reset path.

Creates a HIDDEN GLFW window, runs the fixed-timestep accumulator loop for a
short spell (including a forced reset to rebuild scene/context), and renders
each frame. Exits 0 on success. Requires a GL-capable display/driver; if GLFW
cannot init here that's environmental, not a code fault.
"""
import time
import glfw
import mujoco
from homebot3d.env import HomeBot3DEnv
from homebot3d.teleop_input import keys_to_action

if not glfw.init():
    raise SystemExit("glfw.init failed (no display) — environmental, skipping")

glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(320, 240, "smoke", None, None)
if not window:
    glfw.terminate()
    raise SystemExit("create_window failed (no display) — environmental, skipping")
glfw.make_context_current(window)

env = HomeBot3DEnv(goals=("drink",), max_steps=100000)
env.reset_world(seed=0)

cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(env.model, maxgeom=10000)
context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")

timestep = env.model.opt.timestep
prev = time.time()
accumulator = 0.0
forced_reset_done = False

for frame in range(30):
    now = time.time()
    accumulator += min(now - prev, 0.25)
    prev = now
    action = keys_to_action({"w"})
    while accumulator >= timestep:
        env.step_physics(action)
        accumulator -= timestep
        if frame == 15 and not forced_reset_done:      # exercise reset-rebuild
            env.reset_world(seed=1)
            context.free()
            scene = mujoco.MjvScene(env.model, maxgeom=10000)
            context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            forced_reset_done = True
            accumulator = 0.0
            break
    w, h = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, w, h)
    mujoco.mjv_updateScene(env.model, env.data, opt, None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    accumulator += timestep      # ensure the step loop runs next frame headlessly

glfw.terminate()
env.close()
print(f"GL smoke OK: 30 frames rendered, reset-rebuild={forced_reset_done}")
