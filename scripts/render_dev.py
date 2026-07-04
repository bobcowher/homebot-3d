"""Headless dev-render harness for HomeBot3DEnv.

Renders what the agent sees (ego camera) plus a top-down overview to PNG, and
optionally drives a scripted controller toward a goal so the frames show motion.
This is the way to *look at* the sim without the interactive viewer — the PNGs
can be inspected directly.

Usage:
    python scripts/render_dev.py                 # single frame, default goal
    python scripts/render_dev.py --steps 300     # drive 300 steps toward goal
    python scripts/render_dev.py --goal drink --out /tmp/frames

Outputs (in --out, default ./_dev_render):
    overview_000.png, ego_000.png, ...   one pair per captured frame
"""
import argparse
import math
import os
import struct
import zlib
import numpy as np
import mujoco

from homebot3d.env import HomeBot3DEnv


def write_png(path, arr):
    """Write an (H, W, 3) uint8 array to a PNG. Stdlib only — no Pillow."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, _ = arr.shape
    # Each scanline is prefixed with a filter-type byte (0 = none).
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def top_down_camera(model):
    """Free camera looking straight down at the whole scene."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Frame the floor plane: its center is (fx/2, fy/2). Recover from model
    # extents via the floor geom.
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    cx, cy, _ = model.geom_pos[fid]
    cam.lookat[:] = [cx, cy, 0.0]
    cam.distance = 1.4 * max(model.geom_size[fid][0], model.geom_size[fid][1]) * 2
    cam.elevation = -89.0
    cam.azimuth = 90.0
    return cam


def heading_error(robot, gx, gy):
    bearing = math.atan2(gy - robot.y, gx - robot.x)
    return (bearing - robot.heading + math.pi) % (2 * math.pi) - math.pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--goal", default="drink", choices=["trash", "drink", "package"])
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--every", type=int, default=60, help="capture cadence in steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="./_dev_render")
    p.add_argument("--size", type=int, default=480)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = HomeBot3DEnv(goals=(args.goal,), map_name="default")
    env.reset(seed=args.seed)

    overview_cam = top_down_camera(env.model)
    ov = mujoco.Renderer(env.model, height=args.size, width=args.size)
    ego = mujoco.Renderer(env.model, height=args.size, width=args.size)

    frame = 0

    def capture():
        nonlocal frame
        ov.update_scene(env.data, camera=overview_cam)
        write_png(f"{args.out}/overview_{frame:03d}.png", ov.render())
        ego.update_scene(env.data, camera="ego")
        write_png(f"{args.out}/ego_{frame:03d}.png", ego.render())
        gx, gy = env._goal_xy()
        d = math.hypot(gx - env._robot.x, gy - env._robot.y)
        print(f"frame {frame}: step={env._steps} "
              f"pose=({env._robot.x:.2f},{env._robot.y:.2f},{env._robot.heading:.2f}) "
              f"goal=({gx:.2f},{gy:.2f}) dist={d:.2f}")
        frame += 1

    capture()

    for i in range(args.steps):
        gx, gy = env._goal_xy()
        err = heading_error(env._robot, gx, gy)
        linear = 1.0 if abs(err) < 0.5 else 0.0
        angular = float(np.clip(2.0 * err, -1.0, 1.0))
        _, reward, term, trunc, _ = env.step(np.array([linear, angular], np.float32))
        if reward > 0:
            print(f"  +{reward} at step {env._steps}")
        if (i + 1) % args.every == 0 or term or trunc:
            capture()
        if term or trunc:
            print(f"episode end: term={term} trunc={trunc}")
            break

    ov.close()
    ego.close()
    env.close()
    print(f"wrote {frame} frame pair(s) to {args.out}")


if __name__ == "__main__":
    main()
