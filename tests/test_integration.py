import numpy as np
from homebot3d.env import HomeBot3DEnv


def test_short_rollout_headless():
    env = HomeBot3DEnv(goals=("drink",), max_steps=50)
    obs, info = env.reset(seed=0)
    frames = 0
    for _ in range(50):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert obs.shape == (128, 128, 3)
        frames += 1
        if term or trunc:
            break
    assert frames > 0
    env.close()


def test_driving_toward_recliner_eventually_rewards():
    # Scripted controller: steer straight at the recliner using the privileged seam.
    # Control law: compute heading error as (goal_bearing - robot_heading), normalised
    # to [-pi, pi], so the robot converges to pointing at the goal rather than spinning
    # at a constant angular velocity proportional to the raw world-frame bearing.
    env = HomeBot3DEnv(goals=("drink",), max_steps=1500, random_start=False)
    _, info = env.reset(seed=0)
    got_reward = False
    for _ in range(1500):
        gx, gy = info["privileged"]["goal_vec"]
        goal_bearing = np.arctan2(gy, gx)
        robot_heading = info["privileged"]["pose"][2]
        heading_err = goal_bearing - robot_heading
        # Normalise to [-pi, pi]
        heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
        action = np.array([0.6, np.clip(heading_err, -1, 1)], dtype=np.float32)
        _, reward, term, trunc, info = env.step(action)
        if reward > 0:
            got_reward = True
            break
        if trunc:
            break
    assert got_reward, "scripted seek-recliner controller never reached the goal"
    env.close()
