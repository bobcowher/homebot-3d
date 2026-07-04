import numpy as np
import mujoco

from homebot3d.constants import OBS_SIZE


class Camera:
    def __init__(self, model, width=OBS_SIZE, height=OBS_SIZE):
        self.width = width
        self.height = height
        self._rgb = mujoco.Renderer(model, height=height, width=width)
        self._depth = mujoco.Renderer(model, height=height, width=width)
        self._depth.enable_depth_rendering()

    def render_rgb(self, data) -> np.ndarray:
        self._rgb.update_scene(data, camera="ego")
        return self._rgb.render()

    def render_depth(self, data) -> np.ndarray:
        self._depth.update_scene(data, camera="ego")
        return self._depth.render().astype(np.float32)

    def close(self):
        self._rgb.close()
        self._depth.close()


def privileged(robot, goal_xy) -> dict:
    return {
        "pose": np.array([robot.x, robot.y, robot.heading], dtype=np.float32),
        "goal_vec": np.array([goal_xy[0] - robot.x, goal_xy[1] - robot.y],
                             dtype=np.float32),
    }
