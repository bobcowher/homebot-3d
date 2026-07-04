import numpy as np
import pytest
from homebot3d.env import HomeBot3DGoalEnv
from homebot3d.constants import REACH_RADIUS

def test_dict_obs_structure():
    env = HomeBot3DGoalEnv(goals=("drink",))
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    assert obs["observation"].shape == (128, 128, 3)
    assert obs["achieved_goal"].shape == (2,)
    assert obs["desired_goal"].shape == (2,)
    env.close()

def test_compute_reward_scalar():
    env = HomeBot3DGoalEnv(goals=("drink",))
    env.reset(seed=0)
    ag = np.array([0.0, 0.0]); dg = np.array([0.0, 0.0])
    assert env.compute_reward(ag, dg, {}) == 0.0
    far = np.array([10.0, 10.0])
    assert env.compute_reward(far, dg, {}) == -1.0
    env.close()

def test_compute_reward_vectorised():
    env = HomeBot3DGoalEnv(goals=("drink",))
    env.reset(seed=0)
    ag = np.array([[0.0, 0.0], [10.0, 10.0]])
    dg = np.array([[0.0, 0.0], [0.0, 0.0]])
    out = env.compute_reward(ag, dg, [{}, {}])
    np.testing.assert_array_equal(out, [0.0, -1.0])
    env.close()

def test_multi_goal_raises():
    with pytest.raises(ValueError):
        HomeBot3DGoalEnv(goals=("drink", "trash"))
