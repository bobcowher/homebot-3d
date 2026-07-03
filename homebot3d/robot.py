import numpy as np
import mujoco
from homebot3d.constants import MAX_LIN, MAX_ANG


class Robot:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        self._qyaw = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")]
        self._a_vx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vx")
        self._a_vy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vy")
        self._a_wz = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wz")
        self._body_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_body")

    @property
    def x(self) -> float:
        return float(self.data.xpos[self._body_id, 0])

    @property
    def y(self) -> float:
        return float(self.data.xpos[self._body_id, 1])

    @property
    def heading(self) -> float:
        return float(self.data.qpos[self._qyaw])

    def apply(self, action):
        linear = float(np.clip(action[0], -1.0, 1.0)) * MAX_LIN
        angular = float(np.clip(action[1], -1.0, 1.0)) * MAX_ANG
        h = self.heading
        self.data.ctrl[self._a_vx] = linear * np.cos(h)
        self.data.ctrl[self._a_vy] = linear * np.sin(h)
        self.data.ctrl[self._a_wz] = angular

    def collided(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self._body_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self._body_geom else c.geom1
                other_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                if other_name and (
                    other_name.startswith("wall_") or other_name.startswith("fixture_")
                ):
                    return True
        return False
