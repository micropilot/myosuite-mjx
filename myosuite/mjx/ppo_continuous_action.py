import os

os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
import functools

import jax

from datetime import datetime
from jax import numpy as jp


import brax

import flax
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac

from myosuite.mjx.myodm_v0 import TrackEnv


dof_robot = 29
model_path = "/../envs/myo/assets/hand/myohand_object.xml"
object_name = "airplane"
reference = {
    "time": jp.array((0.0, 4.0)),
    "robot": jp.zeros((2, dof_robot)),
    "robot_vel": jp.zeros((2, dof_robot)),
    "object_init": jp.array((0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0)),
    "object": jp.array(
        [[-0.2, -0.2, 0.1, 1.0, 0.0, 0.0, -1.0], [0.2, 0.2, 0.1, 1.0, 0.0, 0.0, 1.0]]
    ),
}

env = TrackEnv(
    model_path=model_path,
    object_name=object_name,
    reference=reference,
)


times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    print(f"time_spent: {times[-2] - times[-1]}")
    print(metrics)


train_fn = functools.partial(
    ppo.train,
    num_timesteps=2_000_000,
    num_evals=20,
    reward_scaling=5,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=4,
    unroll_length=50,
    num_minibatches=32,
    num_updates_per_batch=8,
    discounting=0.95,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=128,
    batch_size=256,
    max_devices_per_host=8,
    seed=1,
)


make_inference_fn, params, _ = train_fn(environment=env)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
