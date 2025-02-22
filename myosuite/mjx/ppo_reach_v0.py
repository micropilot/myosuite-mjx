import os

os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
import functools

from datetime import datetime
from jax import numpy as jp

from brax import envs
from brax.training.agents.ppo import train as ppo

from myosuite.mjx.reach_v0 import ReachEnvV0


model_path = "myosuite/simhive/myo_sim/finger/myofinger_v0.xml"
# target_reach_range = {
        #     "IFtip": jp.array([[0.2, 0.05, 0.20], [0.2, 0.05, 0.20]]),
        # }

# Time spent: 0:00:01.639262
# Progress at step 3112960:
# Metrics:
#   eval/walltime: 29.420922994613647
#   training/sps: 160742.83952329747
#   training/walltime: 40.617655515670776
#   training/entropy_loss: -0.001987969968467951
#   training/policy_loss: -0.02683013305068016
#   training/total_loss: 5.092004776000977
#   training/v_loss: 5.120822906494141
#   eval/episode_bonus: 23.578125
#   eval/episode_penalty: 0.0
#   eval/episode_reach: -0.9515132904052734
#   eval/episode_reward: 93.36099243164062
#   eval/episode_bonus_std: 2.306950569152832
#   eval/episode_penalty_std: 0.0
#   eval/episode_reach_std: 0.044229600578546524
#   eval/episode_reward_std: 9.247026443481445
#   eval/avg_episode_length: 32.0
#   eval/epoch_eval_time: 0.2388324737548828
#   eval/sps: 17150.09661627415
# time to jit: 0:00:39.146235
# time to train: 0:00:52.407653

target_reach_range = {
            "IFtip": ((0.1, -0.1, 0.1), (0.27, 0.1, 0.3)),
        }

envs.register_environment('reach_v0', ReachEnvV0)

env_name = 'reach_v0'
env = envs.get_environment(env_name, 
                           model_path=model_path,
                           target_reach_range=target_reach_range)


times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    print(f"Time spent: {times[-1] - times[-2]}")
    print(f"Progress at step {num_steps}:")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


train_fn = functools.partial(
    ppo.train,
    num_timesteps=2_000_000, 
    num_evals=20, 
    reward_scaling=1, 
    episode_length=32, 
    normalize_observations=True, 
    action_repeat=1, 
    unroll_length=5, 
    num_minibatches=32, 
    num_updates_per_batch=4, 
    discounting=0.97, 
    learning_rate=3e-4, 
    entropy_cost=1e-2, 
    num_envs=2048, 
    batch_size=1024, 
    seed=1
)


make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
