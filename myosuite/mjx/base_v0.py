from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco

from myosuite.mjx.fatigue import CumulativeFatigue


class BaseV0(PipelineEnv):
    def __init__(
            self,
            model_path: str,
            obs_keys: list, 
            weighted_reward_keys: dict, 
            sites: list = None,
            frame_skip=10,
            muscle_condition="",
            fatigue_reset_vec=None,
            fatigue_reset_random=False,
            **kwargs
        ):

        sys = mjcf.load(model_path)

        n_frames = 5
        sys = sys.tree_replace({
            'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
            'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
            'opt.iterations': 1,
            'opt.ls_iterations': 4,
        })

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend='mjx', **kwargs)

        if self.sys.na > 0 and "act" not in obs_keys:
            obs_keys = obs_keys.copy()
            obs_keys.append("act")

        # ids
        self.tip_sids = []
        self.target_sids = []
        if sites:
            for site in sites:
                self.tip_sids.append(mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, site))
                self.target_sids.append(mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, site + "_target"))

        self.muscle_condition = muscle_condition
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random
        self.frame_skip = frame_skip
        self.weighted_reward_keys = weighted_reward_keys
        self.initializeConditions()
        
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
            self.EPLpos = mujoco.mj_name2id(self.sys.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "EPL")
            self.EIPpos = mujoco.mj_name2id(self.sys.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "EIP")

    def compute_reward(self, data: base.State) -> dict:
        # implemented in task subclass
        raise NotImplementedError
    
    # step the simulation forward
    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        muscle_act_ind = self.sys.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        
        # Explicitely project normalized space (-1,1) to actuator space (0,1) if muscles
        if self.sys.model.na and self.normalize_act:
            # find muscle actuators
            action[muscle_act_ind] = 1.0 / (
                1.0 + jp.exp(-5.0 * (action[muscle_act_ind] - 0.5))
            )
            # TODO: actuator space may not always be (0,1) for muscle or (-1, 1) for others
            isNormalized = (
                False  # refuse internal reprojection as we explicitly did it here
            )
        else:
            isNormalized = self.normalize_act  # accept requested reprojection

        # implement abnormalities
        if self.muscle_condition == "fatigue":
            # import ipdb; ipdb.set_trace()
            action[muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                action[muscle_act_ind]
            )
        elif self.muscle_condition == "reafferentation":
            # redirect EIP --> EPL
            action[self.EPLpos] = action[self.EIPpos].copy()
            # Set EIP to 0
            action[self.EIPpos] = 0
        
        # step forward
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = jp.clip(action, action_min, action_max)
        
        pipeline_state = self.pipeline_step(pipeline_state, action)
        obs = self._get_obs(pipeline_state, action)

        reward, done, metrics = self.compute_reward(pipeline_state)

        state.metrics.update(
            reward=reward
        )
        state.metrics.update(**metrics)

        return state.replace(
            pipeline_state=pipeline_state, 
            obs=obs, 
            reward=reward, 
            done=done
        )
    
    def reset(self, fatigue_reset: bool = True, rng: jax.Array = None) -> State:
        if fatigue_reset:
            if self.muscle_condition == "fatigue":
                self.muscle_fatigue.reset(
                    fatigue_reset_vec=self.fatigue_reset_vec,
                    fatigue_reset_random=self.fatigue_reset_random
                )
            else:
                pass 
        
        qpos = self.sys.init_q
        qvel = jp.zeros(qpos.shape)

        reward, done, zero = jp.zeros(3)
        data = self.pipeline_init(
            qpos,
            qvel
        )

        obs = self.get_obs(data.data, jp.zeros(self.sys.act_size()))
        metrics = {k: zero for k in self.weighted_reward_keys.keys()}
        state = State(
            data,
            obs,
            reward,
            done,
            metrics
        )
        return state
        
    def get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        raise NotImplementedError
    
    