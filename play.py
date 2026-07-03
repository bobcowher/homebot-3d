"""Human play wrapper for HomeBot3DEnv (MuJoCo passive viewer).

Controls: W/S forward/back, A/D turn, R reset, Esc quit.

The viewer is re-launched against the current env.model / env.data after
every reset so MuJoCo's passive viewer never renders stale handles.
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

    # Outer loop: one iteration per episode / viewer session.
    # We break out to relaunch the viewer whenever the episode ends or the
    # user presses R; we exit entirely when the viewer window is closed.
    quit_requested = False

    while not quit_requested:
        env.reset(seed=None)
        action = np.zeros(2, dtype=np.float32)
        reset_requested = False

        def key_cb(keycode):
            nonlocal action, reset_requested, quit_requested
            c = chr(keycode) if 0 < keycode < 0x110000 else ""
            if c in "Ww":
                action[:] = [1.0, 0.0]
            elif c in "Ss":
                action[:] = [-1.0, 0.0]
            elif c in "Aa":
                action[:] = [0.0, -1.0]
            elif c in "Dd":
                action[:] = [0.0, 1.0]
            elif c in "Rr":
                reset_requested = True
                action[:] = 0
            else:
                action[:] = 0

        with mujoco.viewer.launch_passive(env.model, env.data,
                                          key_callback=key_cb) as v:
            while v.is_running():
                if reset_requested:
                    # Break to outer loop so viewer is relaunched with fresh
                    # model/data handles after env.reset().
                    break

                _, reward, term, trunc, _ = env.step(action)
                if reward > 0:
                    print(f"+{reward}")
                if term or trunc:
                    # Episode over — relaunch viewer on next reset.
                    break

                v.sync()

            # If viewer was closed by the user (not by a reset/episode-end
            # break), signal the outer loop to exit.
            if not v.is_running() and not reset_requested and not (term or trunc):
                quit_requested = True

        # If the viewer was closed while an episode was still running, exit.
        if not reset_requested and not quit_requested:
            # episode ended naturally; loop back to reset and relaunch
            pass

    env.close()


if __name__ == "__main__":
    main()
