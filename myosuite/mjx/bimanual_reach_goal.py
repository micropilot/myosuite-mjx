import jax
from jax import numpy as jp
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


class BimanualReachGoal(PipelineEnv):
    def __init__(self, **kwargs):
        path = "myosuite/envs/myo/assets/arm/myoarm_bionic_bimanual_reach.xml"
        mj_model = mujoco.MjModel.from_xml_path(path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self.proximity_th = 0.005

        self.arm_start = jp.array([-0.4, -0.25, 1.05])
        self.mpl_start = jp.array([0.4, -0.25, 1.05])
        self.goal_center = jp.array([0.0, 0.15, 0.95])

        self.arm_start_shifts = jp.array([0.055, 0.055, 0])
        self.mpl_start_shifts = jp.array([0.055, 0.055, 0])
        self.goal_shifts = jp.array([0.098, 0.098, 0])
        self.PILLAR_HEIGHT = 1.09

        self.start_left_bid = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "start_left"
        )
        self.start_right_bid = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "start_right"
        )
        self.goal_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal")

        # arm
        self.palm_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "S_grasp")
        self.fin0 = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "THtip")
        self.fin1 = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "IFtip")
        self.fin2 = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "MFtip")
        self.fin3 = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "RFtip")
        self.fin4 = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "LFtip")

        # mpl
        self.Rpalm1_sid = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SITE, "prosthesis/palm_thumb"
        )
        self.Rpalm2_sid = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SITE, "prosthesis/palm_pinky"
        )

        self.arm_start_pos = self.arm_start
        self.mpl_start_pos = self.mpl_start
        self.goal_pos = self.goal_center

        self._reset_noise_scale = 1e-2

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {"goal_dist": zero}
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics, fully JAX-compatible."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        obs = self._get_obs(data)

        arm_goal_dist = jp.abs(
            jp.linalg.norm(data.qpos[self.fin0] - self.goal_pos, axis=-1)
        )[0][0]
        mpl_goal_dist = jp.abs(
            jp.linalg.norm(data.qpos[self.Rpalm1_sid] - self.goal_pos, axis=-1)
        )[0][0]

        reward = -(0.6 * arm_goal_dist + 0.4 * mpl_goal_dist)
        done = jp.where(arm_goal_dist < self.proximity_th, 1.0, 0.0)
        done = jp.where(mpl_goal_dist < self.proximity_th, 1.0, done)

        state.metrics.update(goal_dist=arm_goal_dist + mpl_goal_dist)

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        myohand_qpos = data.qpos[self.myo_joint_range]
        myohand_qvel = data.qvel[self.myo_dof_range]

        pros_hand_qpos = data.qpos[self.prosth_joint_range]
        pros_hand_qvel = data.qvel[self.prosth_dof_range]

        # external_contact_forces are excluded
        return jp.concatenate(
            [myohand_qpos, myohand_qvel, pros_hand_qpos, pros_hand_qvel]
        )


rng = jax.random.PRNGKey(0)
env = BimanualReachGoal()
env.reset(rng)
