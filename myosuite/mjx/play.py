import time
import jax
import mujoco
from mujoco import mjx
import mediapy


# Load Mujoco Model
mj_model = mujoco.MjModel.from_xml_path(
    "myosuite/envs/myo/assets/hand/myohand_object_mjx1740458008.183464_processed.xml"
)
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
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False  # Keep joint visualization
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False  # Disable transparency
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = (
    False  # Disable contact points
)
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = False  # Disable constraints

# Camera setup: Change camera angle
camera = mujoco.MjvCamera()
camera.lookat[:] = [0.0, 0.0, 0.0]  # Adjust lookat point
camera.azimuth = 0  # Adjust azimuth (horizontal rotation)
camera.elevation = -45  # Adjust elevation (vertical angle)
camera.distance = 3.0  # Adjust distance from the scene

# Simulation parameters
duration = 1  # (seconds)
framerate = 60  # (Hz)

jit_step = jax.jit(mjx.step)

frames = []
mujoco.mj_resetData(mj_model, mj_data)
mjx_data = mjx.put_data(mj_model, mj_data)

while mjx_data.time < duration:
    start_time = time.time()
    mjx_data = jit_step(mjx_model, mjx_data)
    if len(frames) < mjx_data.time * framerate:
        mj_data = mjx.get_data(mj_model, mjx_data)
        renderer.update_scene(mj_data, scene_option=scene_option, camera=camera)
        pixels = renderer.render()
        frames.append(pixels)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

mediapy.write_video("myohand_mjx.mp4", frames, fps=framerate)
