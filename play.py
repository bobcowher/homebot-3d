"""Human play wrapper for HomeBot3DEnv (MuJoCo passive viewer).

Controls: W/S forward/back, A/D turn, R reset, Esc quit.
"""
import argparse
import numpy as np
import mujoco.viewer
from homebot3d.env import HomeBot3DEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--goals", nargs="+", default=["trash", "drink", "package"])
    p.add_argument("--map", default="default")
    p.add_argument("--random-start", action="store_true")
    args = p.parse_args()

    env = HomeBot3DEnv(goals=tuple(args.goals), map_name=args.map,
                       random_start=args.random_start)
    env.reset(seed=0)
    action = np.zeros(2, dtype=np.float32)

    def key_cb(keycode):
        nonlocal action
        c = chr(keycode) if 0 < keycode < 0x110000 else ""
        if c in "Ww": action[:] = [1.0, 0.0]
        elif c in "Ss": action[:] = [-1.0, 0.0]
        elif c in "Aa": action[:] = [0.0, -1.0]
        elif c in "Dd": action[:] = [0.0, 1.0]
        elif c in "Rr": env.reset(seed=0); action[:] = 0
        else: action[:] = 0

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_cb) as v:
        while v.is_running():
            _, reward, term, trunc, _ = env.step(action)
            if reward > 0:
                print(f"+{reward}")
            if term or trunc:
                env.reset(seed=0)
            v.sync()
    env.close()


if __name__ == "__main__":
    main()
