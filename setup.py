from setuptools import setup, find_packages
setup(name="gym-homebot-3d", version="0.1.1", packages=find_packages(),
      # Ship the MuJoCo texture PNGs inside the wheel. Without this a non-editable
      # pip install (e.g. git+ssh on a training host) gets the .py modules but not
      # assets/textures/*.png, and world.texture_assets() FileNotFounds on first
      # env.reset(). Editable installs happen to work because they run from the
      # source tree, which masks the gap.
      package_data={"homebot3d": ["assets/textures/*.png"]},
      install_requires=["mujoco>=3.1", "gymnasium>=0.29", "numpy>=1.24"])
