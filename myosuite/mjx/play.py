import jax
import jax.numpy as jp
from jax.random import PRNGKey
import mujoco 
from mujoco import mjx
import mediapy


# Load Mujoco Model
mj_model = mujoco.MjModel.from_xml_path('myosuite/simhive/myo_sim/hand/myohand.xml')
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(m=mj_model, d=mj_data)

print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())

# Disable body skeleton rendering by setting transparency
mj_model.geom_rgba[:10, 3] = 0.0  # Make all geoms invisible

# Renderer setup
renderer = mujoco.Renderer(mj_model)

# Scene Option: Disable body visualization and scene visualization
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True  # Keep joint visualization
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False  # Disable transparency
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False  # Disable contact points
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = False  # Disable constraints

# Camera setup: Change camera angle
camera = mujoco.MjvCamera()
camera.lookat[:] = [0.2, -0.2, 1]  # Adjust lookat point
camera.azimuth = 30  # Adjust azimuth (horizontal rotation)
camera.elevation = -45  # Adjust elevation (vertical angle)
camera.distance = 1.1  # Adjust distance from the scene

# Simulation parameters
duration = 3.8  # (seconds)
framerate = 60  # (Hz)

frames = []
mujoco.mj_resetData(mj_model, mj_data)
while mj_data.time < duration:
    mujoco.mj_step(mj_model, mj_data)
    if len(frames) < mj_data.time * framerate:
        renderer.update_scene(mj_data, scene_option=scene_option, camera=camera)
        pixels = renderer.render()
        frames.append(pixels)

mediapy.write_video("myohand_mjx.mp4", frames, fps=framerate)

