import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
from orbax import checkpoint as ocp


import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model


class ElbowPose1D6MRandom(PipelineEnv):
    def __init__(self, **kwargs):
        path = "myosuite/envs/myo/assets/elbow/myoelbow_1dof6muscles.xml"
        mj_model = mujoco.MjModel.from_xml_path(path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        print (mj_model.actuator_biastype)

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

        self.target_jnt_range = {'r_elbow_flex':(0, 2.27),}
        self.pose_thd = .175
        self.weighted_reward_keys = {
            "pose": 1.0,
            "bonus": 4.0,
            "act_reg": 1.0,
            "penalty": 50,
        }

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=low, maxval=hi
        )

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            'pose': zero,
            'bonus': zero,
            'penalty': zero,
            'act_reg': zero,
            'sparse': zero
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        pose_dist = jp.linalg.norm(self.target_jnt_value - data.data.qpos, axis=-1)
        if self.sys.na > 0:
            act_mag = jp.linalg.norm(data.act.copy(), axis=-1) / self.sys.na
        else:
            act_mag = jp.linalg.norm(jp.zeros_like(data.data.qpos))
        if self.sys.na !=0: act_mag= act_mag/self.sys.na
        far_th = 4*jp.pi/2

        sub_rewards = {
            "pose": -1.*pose_dist, 
            "bonus": 1.*(pose_dist<self.pose_thd) + 1.*(pose_dist<1.5*self.pose_thd), 
            "penalty": -1.*(pose_dist>far_th),
            "act_reg": -1.*act_mag,
            "sparse": -1.0*pose_dist
        }

        state.metrics.update(**sub_rewards)

        reward = jp.sum([wt*sub_rewards[key] for key, wt in self.reward_weight_dict.items()], axis=0)
        terminated = jp.where((pose_dist < self.pose_thd) | (pose_dist > far_th), 1.0, 0.0)

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )

    def _get_obs(
        self, data: mjx.Data, action: jp.ndarray
    ) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""

        # external_contact_forces are excluded
        return jp.concatenate([
            data.qpos,
            data.qvel
        ])


from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# Function to update the animation
def update(img):
    plt.clf()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.imshow(img)
    plt.axis('off')


env = ElbowPose1D6MRandom()
# Reset the environment to get initial state
mj_model, mj_data = env.mj_model, env.mj_data 
renderer = mujoco.Renderer(mj_model)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

def get_image():
    mujoco.mj_resetData(mj_model, mj_data)
    while True:
        mujoco.mj_step(mj_model, mj_data)
        renderer.update_scene(mj_data, scene_option=scene_option)
        img = renderer.render()

        yield img

fig = plt.figure(figsize=(6, 6))
ani = animation.FuncAnimation(fig, update, frames=get_image(), interval=20)
plt.show()
