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


class ElbowPose1D6MRandom(PipelineEnv):
    def __init__(self, **kwargs):
        path = "myosuite/envs/myo/assets/elbow/myoelbow_1dof6muscles.xml"
        mj_model = mujoco.MjModel.from_xml_path(path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        target_jnt_range = {
            "r_elbow_flex": (0, 2.27),
        }
        self.pose_thd = 0.175
        self.reward_weight_dict = {
            "pose": 1.0,
            "bonus": 4.0,
            "act_reg": 1.0,
            "penalty": 50,
        }

        self.target_jnt_ids = []
        self.target_jnt_range = []
        for jnt_name, jnt_range in target_jnt_range.items():
            self.target_jnt_ids.append(
                mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            )
            self.target_jnt_range.append(jnt_range)
        self.target_jnt_range = jp.array(self.target_jnt_range)
        self.target_jnt_value = jp.mean(
            self.target_jnt_range, axis=1
        )  # pseudo targets for init

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""

        qpos = self.sys.qpos0
        qvel = jp.zeros(self.sys.nv)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "pose": zero,
            "bonus": zero,
            "penalty": zero,
            "act_reg": zero,
            "sparse": zero,
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics, fully JAX-compatible."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        obs = self._get_obs(data)

        # Calculate pose distance
        pose_dist = jp.linalg.norm(self.target_jnt_value - data.qpos, axis=-1)

        # Calculate action magnitude with conditional check for self.sys.na
        act_mag = jp.where(
            self.sys.na > 0,
            jp.linalg.norm(data.act.copy(), axis=-1) / self.sys.na,
            jp.linalg.norm(jp.zeros_like(data.qpos)),
        )

        # Define threshold for far distances
        far_th = 2 * jp.pi  # Equivalent to 4 * pi / 2

        # Compute sub-rewards
        sub_rewards = {
            "pose": -pose_dist,
            "bonus": jp.where(pose_dist < self.pose_thd, 1.0, 0.0)
            + jp.where(pose_dist < 1.5 * self.pose_thd, 1.0, 0.0),
            "penalty": jp.where(pose_dist > far_th, -1.0, 0.0),
            "act_reg": -act_mag,
            "sparse": -pose_dist,
        }

        # Update metrics with sub-rewards
        state.metrics.update(**sub_rewards)

        # Calculate total reward using weighted sum of sub-rewards
        reward = jp.sum(
            jp.array(
                [
                    self.reward_weight_dict[key] * sub_rewards[key]
                    for key in self.reward_weight_dict.keys()
                ]
            ),
            axis=0,
        )

        # Determine if the episode is done based on the pose distance thresholds
        done = jp.where((pose_dist < self.pose_thd) | (pose_dist > far_th), 1.0, 0.0)

        # Return updated state
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""

        # external_contact_forces are excluded
        return jp.concatenate([data.qpos, data.qvel])


import functools
from datetime import datetime

env = ElbowPose1D6MRandom()
# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

print("It has been jitted")

# initialize the state
state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]

print("State initialized")

# grab a trajectory
for i in range(10):
    ctrl = -0.1 * jp.ones(env.sys.nu)
    state = jit_step(state, ctrl)
    rollout.append(state.pipeline_state)

print("Trajectory grabbed")

train_fn = functools.partial(
    ppo.train,
    num_timesteps=1e6,
    num_evals=5,
    reward_scaling=0.1,
    episode_length=100,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=24,
    num_updates_per_batch=8,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=1024,
    batch_size=128,
    seed=0,
)


# Initialize tracking lists for metrics
times, x_data, y_data, ydataerr = [], [], [], []
max_y, min_y = 13000, 0


def progress(num_steps, metrics):
    # Append data points for tracking
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    ydataerr.append(metrics["eval/episode_reward_std"])

    # Print progress details
    print(f"\n[Progress Update]")
    print(
        f"Environment steps: {num_steps}/{int(train_fn.keywords['num_timesteps'] * 1.25)}"
    )
    print(f"Reward per episode: {y_data[-1]:.3f}")
    print(f"Reward std deviation: {ydataerr[-1]:.3f}")
    print(f"Time elapsed: {times[-1] - times[0] if len(times) > 1 else 'N/A'}")

    # Optionally, print min and max limits for rewards if helpful
    print(f"Expected reward range: [{min_y}, {max_y}]")


make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")


# @title Save Model
model_path = "mjx_brax_policy"
model.save_params(model_path, params)

# @title Load Model and Define Inference Function
params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)
