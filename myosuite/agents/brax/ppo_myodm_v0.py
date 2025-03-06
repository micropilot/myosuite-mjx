import os

os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
import functools

from datetime import datetime
from jax import numpy as jp

from brax import envs
from brax.training.agents.ppo import train as ppo

from myosuite.envs.myo.myodm.myodm_v0_mjx import TrackEnv


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

envs.register_environment("myodm_airplane_v0", TrackEnv)

env_name = "myodm_airplane_v0"
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
    seed=1,
)


make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
