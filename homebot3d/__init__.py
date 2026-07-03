import os
os.environ.setdefault("MUJOCO_GL", "egl")  # vendor-neutral EGL render path

from gymnasium.envs.registration import register
from homebot3d.env import HomeBot3DEnv
from homebot3d.env import HomeBot3DGoalEnv

register(id="HomeBot3D-V1", entry_point="homebot3d.env:HomeBot3DEnv")
register(id="HomeBot3D-Goal-V1", entry_point="homebot3d.env:HomeBot3DGoalEnv")
