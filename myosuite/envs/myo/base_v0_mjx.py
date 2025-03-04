from brax import base
from brax.envs.base import PipelineEnv, State
import jax
from jax import numpy as jp
import mujoco

from myosuite.envs.env_base_mjx import EnvBaseMJX
from myosuite.envs.myo.fatigue_jax import CumulativeFatigue


class BaseV0(EnvBaseMJX):
    def _setup(
        self,
        obs_keys: list,
        weighted_reward_keys: dict,
        sites: list = None,
        muscle_condition="",
        fatigue_reset_vec=None,
        fatigue_reset_random=False,
        **kwargs
    ):  
        if self.sys.na > 0 and "act" not in obs_keys:
            obs_keys = obs_keys.copy()
            obs_keys.append("act")

        # Initialize as empty lists
        tip_sids_list = []
        target_sids_list = []

        if sites:
            for site in sites:
                tip_sids_list.append(
                    mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, site)
                )
                target_sids_list.append(
                    mujoco.mj_name2id(
                        self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, site + "_target"
                    )
                )

        # Convert lists to JAX arrays
        self.tip_sids = jp.array(tip_sids_list)
        self.target_sids = jp.array(target_sids_list)

        self.muscle_condition = muscle_condition
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random
        self.initializeConditions()

        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            **kwargs
        )

        # TODO: setup viewer later

    def initializeConditions(self):
        # for muscle weakness we assume that a weaker muscle has a
        # reduced maximum force
        if self.muscle_condition == "sarcopenia":
            for mus_idx in range(self.sys.model.actuator_gainprm.shape[0]):
                self.sys.model.actuator_gainprm[mus_idx, 2] = (
                    0.5 * self.sys.model.actuator_gainprm[mus_idx, 2].copy()
                )

        # for muscle fatigue we used the 3CC-r model
        elif self.muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(
                self.sys.model, self.frame_skip, seed=self.get_input_seed()
            )

        # tendon transfer to redirect EIP --> EPL
        # https://www.assh.org/handcare/condition/tendon-transfer-surgery
        elif self.muscle_condition == "reafferentation":
            self.EPLpos = mujoco.mj_name2id(
                self.sys.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "EPL"
            )
            self.EIPpos = mujoco.mj_name2id(
                self.sys.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "EIP"
            )

    def compute_reward(
            self, 
            pipeline_state: base.State,
            info: dict
        ) -> dict:
        # implemented in task subclass
        raise NotImplementedError

    # step the simulation forward
    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        muscle_act_ind = (
            self.sys.mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        )

        # Explicitely project normalized space (-1,1) to actuator space (0,1) if muscles
        if self.sys.na and self.normalize_act:
            action = action.at[muscle_act_ind].set(
                1.0 / (1.0 + jp.exp(-5.0 * (action[muscle_act_ind] - 0.5)))
            )

        # implement abnormalities
        if self.muscle_condition == "fatigue":
            action[muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                action[muscle_act_ind]
            )
        elif self.muscle_condition == "reafferentation":
            # redirect EIP --> EPL
            action[self.EPLpos] = action[self.EIPpos].copy()
            # Set EIP to 0
            action[self.EIPpos] = 0

        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        info = self.get_info(pipeline_state)
        obs = self.get_obs(pipeline_state, info)

        reward, done, metrics = self.compute_reward(pipeline_state, info)
        metrics['reward'] = reward

        state.metrics.update(**metrics)

        return state.replace(
            pipeline_state=pipeline_state, 
            obs=obs, 
            reward=reward, 
            done=done,
            metrics=metrics,
            info=info
        )
        

    def reset(
            self, 
            rng: jax.Array = None, 
            fatigue_reset: bool = True, 
        ) -> State:
        if fatigue_reset:
            if self.muscle_condition == "fatigue":
                self.muscle_fatigue.reset(
                    fatigue_reset_vec=self.fatigue_reset_vec,
                    fatigue_reset_random=self.fatigue_reset_random,
                )
            else:
                pass

        return super().reset(rng)


    def get_obs(
            self, 
            pipeline_state: base.State, 
            info: dict
        )-> jax.Array:
        raise NotImplementedError

    def get_info(
            self,
            pipeline_state: base.State
        ) -> dict:
        raise NotImplementedError