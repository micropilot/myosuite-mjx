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
        self.iftip_min = target_reach_range["IFtip"][0]
        self.iftip_max = target_reach_range["IFtip"][1]
        self.far_th = far_th

    def reset(self, rng: jax.Array = None) -> State:
        super().reset(rng)
        key, subkey = jax.random.split(rng)

        qpos = self.sys.qpos0
        qvel = jp.zeros(qpos.shape)

        info = {}
        for site, span in self.target_reach_range.items():
            sid = mujoco.mj_name2id(
                        self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, site + "_target"
                    )
            target_pos = jax.random.uniform(
                key, 
                shape=span[0].shape, 
                minval=self.target_reach_range[site][0], 
                maxval=self.target_reach_range[site][1]
            )
            info[site] = target_pos

        reward, done, zero = jp.zeros(3)
        data = self.pipeline_init(qpos, qvel)

        obs = self.get_obs(data, jp.zeros(self.sys.act_size()), info)
        metrics = {k: jp.array(zero)for k in self.weighted_reward_keys.keys()}
        metrics['reward'] = reward
        
        state = State(
            pipeline_state=data, 
            obs=obs, 
            reward=reward, 
            done=done, 
            metrics=metrics,
            info=info
        )

        return state


    def compute_reward(self, pipeline_state: base.State, info: dict) -> dict:
        tip_pos = pipeline_state.site_xpos[self.tip_sids]
        # Initialize an empty list to store target positions
        target_pos_list = []

        # Iterate over the keys in target_reach_range
        for site in self.target_reach_range.keys():
            # Check if the site is in info and append its value to the list
            if site in info:
                target_pos_list.append(info[site])

        # Concatenate all the target positions into a single vector
        target_pos = jp.concatenate(target_pos_list) 

        reach_dist = jp.linalg.norm(tip_pos - target_pos)

        far_th = jax.lax.cond(
            jp.squeeze(pipeline_state.time) > 2 * self.dt,
            lambda _: self.far_th * len(self.tip_sids),
            lambda _: jp.inf,
            operand=None
        )

        near_th = len(self.tip_sids) * 0.0125

        # Convert boolean to float: 1.0 for True, 0.0 for False
        done = jp.where(jp.logical_or(reach_dist > far_th, reach_dist < near_th), 1.0, 0.0)

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

    def get_obs(
            self, 
            pipeline_state: base.State, 
            action: jax.Array,
            info: dict
        ) -> jax.Array:

        position = pipeline_state.qpos
        velocity = pipeline_state.qvel * pipeline_state.time
        tip_pos = pipeline_state.site_xpos[self.tip_sids]

        # Initialize an empty list to store target positions
        target_pos_list = []

        # Iterate over the keys in target_reach_range
        for site in self.target_reach_range.keys():
            # Check if the site is in info and append its value to the list
            if site in info:
                target_pos_list.append(info[site])

        # Concatenate all the target positions into a single vector
        target_pos = jp.concatenate(target_pos_list)

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
