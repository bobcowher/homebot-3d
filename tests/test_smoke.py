import numpy as np
import homebot3d  # sets MUJOCO_GL
import mujoco

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="box" pos="0 0 0.5">
      <geom type="box" size="0.25 0.25 0.25" rgba="1 0 0 1"/>
    </body>
    <camera name="cam" pos="0 -3 1.5" xyaxes="1 0 0 0 0.5 1"/>
  </worldbody>
</mujoco>
"""

def test_mujoco_egl_renders_a_frame():
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=84, width=84)
    renderer.update_scene(data, camera="cam")
    frame = renderer.render()
    assert frame.shape == (84, 84, 3)
    assert frame.dtype == np.uint8
    assert frame.sum() > 0          # not an all-black frame
    renderer.close()
