import os

os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
import functools

from datetime import datetime
from jax import numpy as jp

from brax import envs
from brax.training.agents.ppo import train as ppo

from myosuite.mjx.reach_v0 import ReachEnvV0


model_path = "myosuite/simhive/myo_sim/finger/myofinger_v0.xml"
target_reach_range = {
            "IFtip": jp.array([[0.2, 0.05, 0.20], [0.2, 0.05, 0.20]]),
        }


envs.register_environment('reach_v0', ReachEnvV0)

env_name = 'reach_v0'
env = envs.get_environment(env_name, 
                           model_path=model_path,
                           target_reach_range=target_reach_range)


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
    episode_length=100,
    normalize_observations=True,
    action_repeat=4,
    unroll_length=50,
    num_minibatches=32,
    num_updates_per_batch=8,
    discounting=0.95,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=2048,
    batch_size=1024,
    max_devices_per_host=8,
    seed=1,
)


make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
