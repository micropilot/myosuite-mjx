import os

os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
import functools
# import wandb 
import mediapy as media

from datetime import datetime
import jax
from jax import numpy as jp

from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo

from myosuite.envs.myo.myobase.reach_v0_mjx import ReachEnvV0


jax.config.update('jax_default_matmul_precision', 'highest')

model_path = "myosuite/simhive/myo_sim/finger/myofinger_v0.xml"
# target_reach_range = {
#     "IFtip": jp.array([[0.2, 0.05, 0.20], [0.2, 0.05, 0.20]]),
# }

target_reach_range = {
    "IFtip": jp.array([[0.1, -0.1, 0.1], [0.27, 0.1, 0.3]]),
}

envs.register_environment("reach_v0", ReachEnvV0)

env_name = "reach_v0"
env = envs.get_environment(
    env_name, model_path=model_path, target_reach_range=target_reach_range
)

print (env.action_size)

times = [datetime.now()]
name = "brax_ppo_reach_random_v0"

def progress(num_steps, metrics):
    times.append(datetime.now())
    print(f"Time spent: {times[-1] - times[-2]}")
    print(f"Progress at step {num_steps}:")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # wandb.log(step=num_steps, data=metrics)

    

def policy_params(current_step, make_policy, params):
    # Save the model with the specified filename format
    model_filename = f"{name}_brax_ppo_{current_step}"
    model.save_params(f'policies/{model_filename}', params)



# Define a configuration dictionary
config = {
    "num_timesteps": 100_000,
    "num_evals": 10,
    "reward_scaling": 1,
    "episode_length": 32,
    "normalize_observations": True,
    "action_repeat": 1,
    "unroll_length": 5,
    "num_minibatches": 8,
    "num_updates_per_batch": 4,
    "discounting": 0.97,
    "learning_rate": 3e-4,
    "entropy_cost": 1e-2,
    "num_envs": 32,
    "batch_size": 16,
    "seed": 1,
}

# run = wandb.init(
#     project="myoFingerReachRandom-V0",
#     config=config,
#     name=name,
# )

# Use the config dictionary in functools.partial
train_fn = functools.partial(ppo.train, **config)


make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress, policy_params_fn=policy_params)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

if not os.path.exists("policies"):
    os.makedirs("policies", exist_ok=True)

model.save_params(f"policies/{name}_brax_ppo_final", params)