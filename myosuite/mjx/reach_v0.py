import jax
import jax.numpy as jp
import mujoco
from brax import base
from brax.envs.base import State
from dataclasses import replace

from myosuite.mjx.base_v0 import BaseV0


class ReachEnvV0(BaseV0):

    def __init__(
        self,
        model_path: str,
        obs_keys: list = ["qpos", "qvel", "tip_pos", "reach_err"],
        weighted_reward_keys: dict = {
            "reach": 1.0,
            "bonus": 4.0,
            "penalty": 50,
        },
        target_reach_range: dict = None,
        far_th: float = 0.35,
        **kwargs
    ):

        super().__init__(
            model_path,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            sites=target_reach_range.keys(),
            **kwargs
        )

        self.target_reach_range = target_reach_range
        self.far_th = far_th

    def reset(self, rng: jax.Array = None):
        key, subkey = jax.random.split(rng)
        new_site_pos = (
            self.sys.site_pos.copy()
        )  # Assuming site_pos is a mutable type like a list or numpy array
        for site, span in self.target_reach_range.items():
            sid = mujoco.mj_name2id(
                self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, site + "_target"
            )
            new_site_pos = new_site_pos.at[sid].set(
                jax.random.uniform(
                    subkey, shape=span[0].shape, minval=span[0], maxval=span[1]
                )
            )

        # Create a new instance of the dataclass with the updated site_pos
        self.sys = replace(self.sys, site_pos=new_site_pos)

        state = super().reset(subkey)

        return state

    def compute_reward(self, pipeline_state: State) -> dict:
        tip_pos = pipeline_state.site_xpos[self.tip_sids]
        target_pos = pipeline_state.site_xpos[self.target_sids]

        reach_dist = jp.linalg.norm(tip_pos - target_pos, axis=-1)

        far_th = (
            self.far_th * len(self.tip_sids)
            if jp.squeeze(pipeline_state.time) > 2 * self.dt
            else jp.inf
        )

        near_th = len(self.tip_sids) * 0.0125

        done = jp.logical_or(reach_dist > far_th, reach_dist < near_th)

        metrics = {
            "reach": -1.0 * reach_dist,
            "bonus": 1.0 * jp.where(reach_dist < 2 * near_th, 1.0, 0.0)
            + 1.0 * jp.where(reach_dist < near_th, 1.0, 0.0),
            "penalty": -1.0 * jp.where(reach_dist > far_th, 1.0, 0.0),
        }

        reward = jp.sum(
            jp.array([metrics[k] * v for k, v in self.weighted_reward_keys.items()])
        )

        return reward, done, metrics

    def get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        position = pipeline_state.qpos
        velocity = pipeline_state.qvel * pipeline_state.time
        tip_pos = pipeline_state.site_xpos[self.tip_sids]
        target_pos = pipeline_state.site_xpos[self.target_sids]

        reach_err = target_pos - tip_pos

        if self.sys.na > 0:
            obs = jp.concatenate(
                [
                    position,
                    velocity,
                    tip_pos.flatten(),
                    reach_err.flatten(),
                    pipeline_state.act,
                ]
            )
        else:
            obs = jp.concatenate(
                [position, velocity, tip_pos.flatten(), reach_err.flatten()]
            )

        return obs
