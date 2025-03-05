from brax.envs.base import PipelineEnv, State
import jax
from jax import numpy as jp
import mujoco
from brax.io import mjcf


class EnvBaseMJX(PipelineEnv):
    def __init__(
        self, 
        model_path: str,
        frame_skip: int = 10,
    ):  
        model = mujoco.MjModel.from_xml_path(model_path)
        sys = mjcf.load_model(model)

        sys = sys.tree_replace(
            {
                "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                "opt.iterations": 1,
                "opt.ls_iterations": 4,
            }
        )


        super().__init__(sys=sys, backend="mjx", n_frames=frame_skip)

    def _setup(
            self,
            obs_keys: list,
            weighted_reward_keys: dict,
            normalize_act: bool = True,
            **kwargs
    ):  
        self.obs_keys = obs_keys
        self.weighted_reward_keys = weighted_reward_keys
        self.normalize_act = normalize_act
        action_range = self.sys.actuator_ctrlrange
        self.low_action = jp.array(action_range[:, 0])
        self.high_action = jp.array(action_range[:, 1])

        self.init_qpos = self.sys.qpos0
        self.init_qvel = jp.zeros(self.init_qpos.shape)

        # reward, done, zero = jp.zeros(3)
        # pipeline_state = self.pipeline_init(self.init_qpos, self.init_qvel)
        
        # state = State(
        #     pipeline_state=pipeline_state, 
        #     obs=None, 
        #     reward=reward, 
        #     done=done, 
        #     metrics={},
        #     info={}
        # )
        # state = self.step(state, jp.zeros(self.sys.nu))

    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, self.low_action, self.high_action)

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
    
    def reset(self, rng: jax.Array = None) -> State:
        qpos = self.sys.qpos0   
        qvel = jp.zeros(qpos.shape)
        
        reward, done, zero = jp.zeros(3)
        pipeline_state = self.pipeline_init(qpos, qvel)

        info = self.get_info(pipeline_state)
        obs = self.get_obs(pipeline_state, info)
        metrics = {k: jp.array(zero)for k in self.weighted_reward_keys.keys()}
        metrics['reward'] = reward
        
        state = State(
            pipeline_state=pipeline_state, 
            obs=obs, 
            reward=reward, 
            done=done, 
            metrics=metrics,
            info=info
        )

        return state
    
    def get_obs(self, state: State) -> jax.Array:
        raise NotImplementedError
    
    def get_info(self, state: State) -> dict:
        raise NotImplementedError
    
    def compute_reward(self, state: State, info: dict) -> dict:
        raise NotImplementedError
    
    
    
    
    
        