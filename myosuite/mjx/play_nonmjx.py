import mujoco
import mediapy


# Load Mujoco Model
mj_model = mujoco.MjModel.from_xml_path(
    "myosuite/envs/myo/assets/hand/myohand_object_mjx1740458008.183464_processed.xml"
)
mj_data = mujoco.MjData(mj_model)


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


frames = []
mujoco.mj_resetData(mj_model, mj_data)

while mj_data.time < duration:
    # start_time = time.time()
    mujoco.mj_step(mj_model, mj_data)
    if len(frames) < mj_data.time * framerate:
        renderer.update_scene(mj_data, scene_option=scene_option, camera=camera)
        pixels = renderer.render()
        frames.append(pixels)
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")

mediapy.write_video("myohand_nonmjx.mp4", frames, fps=framerate)
