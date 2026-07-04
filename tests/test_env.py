import numpy as np
from homebot3d.env import HomeBot3DEnv

def test_spaces():
    env = HomeBot3DEnv(width=84, height=84)
    assert env.observation_space.shape == (84, 84, 3)
    assert env.action_space.shape == (2,)
    env.close()

def test_reset_returns_rgb_obs_only():
    env = HomeBot3DEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (84, 84, 3) and obs.dtype == np.uint8
    assert "privileged" in info               # seam lives in info, not obs
    assert "pose" in info["privileged"]
    env.close()

def test_step_contract():
    env = HomeBot3DEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (84, 84, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    env.close()

def test_determinism_same_seed_same_frames():
    e1 = HomeBot3DEnv(); o1, _ = e1.reset(seed=42)
    e2 = HomeBot3DEnv(); o2, _ = e2.reset(seed=42)
    np.testing.assert_array_equal(o1, o2)
    a = np.array([0.5, 0.2], dtype=np.float32)
    f1 = e1.step(a)[0]; f2 = e2.step(a)[0]
    np.testing.assert_array_equal(f1, f2)
    e1.close(); e2.close()

def test_truncation_at_max_steps():
    env = HomeBot3DEnv(goals=("drink",), max_steps=3)
    env.reset(seed=0)
    truncated = False
    for _ in range(3):
        truncated = env.step(np.array([0.0, 0.0], dtype=np.float32))[3]
    assert truncated is True
    env.close()

def test_random_start_changes_spawn_but_is_seed_stable():
    e = HomeBot3DEnv(random_start=True)
    o_a, i_a = e.reset(seed=7)
    o_b, i_b = e.reset(seed=7)
    np.testing.assert_allclose(i_a["privileged"]["pose"], i_b["privileged"]["pose"])
    e.close()

def test_reset_world_creates_no_camera():
    env = HomeBot3DEnv(goals=("drink",))
    info = env.reset_world(seed=0)
    assert env._camera is None
    assert "privileged" in info

def test_step_physics_no_camera_and_matches_types():
    env = HomeBot3DEnv(goals=("drink",))
    env.reset_world(seed=0)
    reward, term, trunc, info = env.step_physics([0.0, 0.0])
    assert env._camera is None
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)

def test_gym_step_still_returns_obs():
    env = HomeBot3DEnv(goals=("drink",))
    obs, info = env.reset(seed=0)
    assert obs.shape == (84, 84, 3)
    obs2, r, term, trunc, info2 = env.step([0.0, 0.0])
    assert obs2.shape == (84, 84, 3)
