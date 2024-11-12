import jax 
from jax import numpy as jp 

import mujoco
from mujoco import mjx

from brax.base import State
from brax.envs.base import PipelineEnv, State
from brax.training.agents.ppo import train as ppo
from brax.io import mjcf, model


class MyoHandCubeLiftEnv(PipelineEnv):
    def __init__(self, **kwargs):
        path = "myosuite/mjx/myo/assets/hand/myohand_cubesmall.xml"
        mj_model = mujoco.MjModel.from_xml_path(path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

        self.reward_weight_dict = {
            "pose": 1.0,
            "bonus": 1.0,
            "object": 1.0,
            "penalty": -2,
        }


    def reset(self, rng: jp.ndarray) -> State:
        qpos = self.sys.qpos0
        qvel = self.sys.nv 

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data)

        reward, done, zero = jp.zeros(3)

        metrics = {
            'pose': zero,
            'bonus': zero,
            'penalty': zero,
            'act_reg': zero,
            'sparse': zero
        }
        return State(data, obs, reward, done, metrics)
    
    def step(self, state: State, action: jp.ndarray) -> State:
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        obs = self._get_obs(data)

        reward = jp.zeros(1)
        done = jp.bool_(False)

        # Return updated state
        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )
    
    def _get_obs(
        self, data: mjx.Data
    ) -> jp.ndarray:
        """Returns the observation."""
        return jp.concatenate([data.qpos, data.qvel]
    )

env = MyoHandCubeLiftEnv()