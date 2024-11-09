import os 
import jax 
import mujoco 
from jax import numpy as jp 

from brax.envs.base import Env, MjxEnv, State 
from myosuite.envs.myo.fatigue_jax import CumulativeFatigueJAX

class BaseJax(MjxEnv):

    def __init__(
        self, path, reward_weights_dict=None, **kwargs
    ):

        mj_model = mujoco.MjModel.from_xml_path(path)

        self.tip_sids = []
        self.target_sids = []
        for i in range(mj_model.nsite):
            site_name = mj_model.site_name[i]
            self.tip_sids.append(i)
            self.target_sids.append(i)

        super().__init__(model=mj_model, **kwargs)


    def step(self, state: State, action: jp.ndarray) -> State:
        return state

    def reset(self, rng: jp.ndarray) -> State:
        return State()


# base = BaseJax(path='myosuite/simhive/myo_sim/finger/motorfinger_v0.xml')