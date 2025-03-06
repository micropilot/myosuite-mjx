import jax
import jax.numpy as jp
import mujoco
from brax import base
from brax.envs.base import State

from myosuite.envs.myo.base_v0_mjx import BaseV0


class ReachEnvV0(BaseV0):

    def __init__(self, model_path: str, frame_skip: int = 10, **kwargs):
        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
        )

        self._setup(**kwargs)

    def _setup(
        self,
        target_reach_range: dict,
        far_th: float = 0.35,
        obs_keys: list = ["qpos", "qvel", "tip_pos", "reach_err"],
        weighted_reward_keys: dict = {
            "reach": 1.0,
            "bonus": 4.0,
            "penalty": 50,
        },
        **kwargs
    ):
        self.far_th = far_th
        self.target_reach_range = target_reach_range
        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            sites=self.target_reach_range.keys(),
            **kwargs
        )

    def get_info(self, pipeline_state: base.State, info: dict) -> dict:

        info["tip_pos"] = pipeline_state.site_xpos[self.tip_sids]
        if "target_pos" not in info:
            info["target_pos"] = pipeline_state.site_xpos[self.target_sids]
        info["reach_err"] = info["target_pos"] - info["tip_pos"]

        return info

    def get_obs(self, pipeline_state: base.State, info: dict) -> jax.Array:
        position = pipeline_state.qpos
        velocity = pipeline_state.qvel * pipeline_state.time
        tip_pos = info["tip_pos"]
        reach_err = info["reach_err"]

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

    def compute_reward(self, pipeline_state: base.State, info: dict) -> dict:
        reach_dist = jp.linalg.norm(info["reach_err"], axis=-1)[0]

        # TODO: act_mag not implemented

        far_th = jax.lax.cond(
            jp.squeeze(pipeline_state.time) > 2 * self.dt,
            lambda _: self.far_th * len(self.tip_sids),
            lambda _: jp.inf,
            operand=None,
        )

        near_th = len(self.tip_sids) * 0.0125

        # Convert boolean to float: 1.0 for True, 0.0 for False
        done = jp.where(
            jp.logical_or(reach_dist > far_th, reach_dist < near_th), 1.0, 0.0
        )

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

    def reset(self, rng: jax.Array = None) -> State:
        key, *subkey = jax.random.split(rng, len(self.target_reach_range.items()) + 1)

        # We cannot update mj_model.site or sys.site_pos with brax. 
        # So we first get the state reset and then
        target_pos = []
        for idx, (site, span) in enumerate(self.target_reach_range.items()):
            sid = mujoco.mj_name2id(
                self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, site + "_target"
            )

            target_pos.append(jax.random.uniform(
                subkey[idx],
                shape=span[0].shape,
                minval=self.target_reach_range[site][0],
                maxval=self.target_reach_range[site][1],
            ))

        info = {"target_pos": jp.concatenate(target_pos)}

        return super().reset(key, info=info)


if __name__ == "__main__":
    model_path = "myosuite/simhive/myo_sim/finger/myofinger_v0.xml"
    target_reach_range_jax = {
            "IFtip": jp.array([[0.2, 0.05, 0.20], [0.2, 0.05, 0.20]])
        }
    env = ReachEnvV0(
        model_path=model_path, 
        target_reach_range=target_reach_range_jax
    )
    state = env.reset(rng=jax.random.PRNGKey(0))
