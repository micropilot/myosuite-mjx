import jax 
import jax.numpy as jp
import mujoco
from brax import base
from brax.envs.base import State

from myosuite.mjx.base_v0 import BaseV0


class ReachEnvV0(BaseV0):
    DEFAULT_OBS_KEYS = ["qpos", "qvel", "tip_pos", "reach_err"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50,
    }
    
    def __init__(self, 
                 model_path: str, 
                 obs_keys: list = DEFAULT_OBS_KEYS, 
                 weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
                 target_reach_range: dict = None,
                 far_th: float = 0.35,
                 **kwargs):
        super().__init__(model_path, **kwargs)
        
        self.target_reach_range = target_reach_range
        self.far_th = far_th
        

    def reset(self, rng: jax.Array = None):
        key, subkey = jax.random.split(rng)
        for site, span in self.target_reach_range.items():
            sid = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, site + "_target")
            self.sys.site_pos[sid] = jax.random.uniform(subkey, span)
        
        state = super().reset(subkey)

        return state
    
    def compute_reward(self, pipeline_state: State) -> dict:
        tip_pos = pipeline_state.qpos[self.tip_sids]
        target_pos = pipeline_state.qpos[self.target_sids]

        reach_dist = jp.linalg.norm(tip_pos - target_pos, axis=-1)
        
        far_th = (
            self.far_th * len(self.tip_sids)
            if jp.squeeze(self.sys.time) > 2 * self.sys.dt
            else jp.inf
        )

        near_th = len(self.tip_sids) * 0.0125

        done = reach_dist > far_th or reach_dist < near_th

        metrics = {
            "reach": -1.0 * reach_dist,
            "bonus": 1.0 * (reach_dist < 2 * near_th) + 1.0 * (reach_dist < near_th),
            "penalty": -1.0 * (reach_dist > far_th)
        }

        reward = jp.sum([metrics[k] * v for k, v in self.weighted_reward_keys.items()])

        return reward, done, metrics
    
    def get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        position = pipeline_state.q[2:]
        velocity = pipeline_state.qd
        tip_pos = pipeline_state.qpos[self.tip_sids]
        target_pos = pipeline_state.qpos[self.target_sids]
        reach_err = target_pos - tip_pos
        
        obs = jax.concatenate([position, velocity, tip_pos, reach_err], axis=-1)
        return obs
    
    
    
    
    