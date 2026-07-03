from setuptools import setup, find_packages
setup(name="gym-homebot-3d", version="0.1.0", packages=find_packages(),
      install_requires=["mujoco>=3.1", "gymnasium>=0.29", "numpy>=1.24"])
