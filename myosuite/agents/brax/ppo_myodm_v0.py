import os

os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
import functools
import wandb 
import mediapy as media

from datetime import datetime
import jax
from jax import numpy as jp

from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo

from myosuite.envs.myo.myodm.myodm_v0_mjx import TrackEnv


jax.config.update('jax_default_matmul_precision', 'highest')

model_path = "/../assets/hand/myohand_object_mjx.xml"
object_name = "airplane"
reference = {
            "time": (0.0, 4.0),
            "robot": jp.zeros((2, 29)),
            "robot_vel": jp.zeros((2, 29)),
            "object_init": jp.array((0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0)),
            "object": jp.array(
                [
                    [-0.2, -0.2, 0.1, 1.0, 0.0, 0.0, -1.0],
                    [0.2, 0.2, 0.1, 1.0, 0.0, 0.0, 1.0],
                ]
            ),
        }
obs_keys = ["qp", "qv", "hand_qpos_err", "hand_qvel_err", "obj_com_err"]
weighted_reward_keys = {
    "pose": 0.0,
    "object": 1.0,
    "bonus": 1.0,
    "penalty": -2,
}

envs.register_environment("MyoHandAirplaneFly-v0", TrackEnv)

env_name = "MyoHandAirplaneFly-v0"
env = envs.get_environment(
    env_name, 
    model_path=model_path, 
    object_name=object_name,
    reference=reference,
    obs_keys=obs_keys,
    weighted_reward_keys=weighted_reward_keys,
)


times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    print(f"Time spent: {times[-1] - times[-2]}")
    print(f"Progress at step {num_steps}:")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    wandb.log(step=num_steps, data=metrics)


# Define a configuration dictionary
config = {
    "num_timesteps": 100_000_000,
    "num_evals": 1000,
    "reward_scaling": 1,
    "episode_length": 32,
    "normalize_observations": True,
    "action_repeat": 1,
    "unroll_length": 5,
    "num_minibatches": 32,
    "num_updates_per_batch": 4,
    "discounting": 0.97,
    "learning_rate": 3e-4,
    "entropy_cost": 1e-2,
    "num_envs": 1024,
    "batch_size": 512,
    "seed": 1,
}

run = wandb.init(
    project="MyoHandAirplaneFly-v0",
    config=config,
    name="brax_ppo_v0",
)

# Use the config dictionary in functools.partial
train_fn = functools.partial(ppo.train, **config)


make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

if not os.path.exists("policies"):
    os.makedirs("policies", exist_ok=True)

model.save_params('policies/myohand_airplane_v0_brax_ppo', params)