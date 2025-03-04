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
        print (self.sys.qpos0)

    def _setup(
            self,
            obs_keys: list,
            weighted_reward_keys: dict,
            normalize_act: bool = True,
            **kwargs
    ):
        action_range = self.sys.actuator_ctrlrange
        self.low_action = jp.array(action_range[:, 0])
        self.high_action = jp.array(action_range[:, 1])

        self.init_qpos = self.sys.qpos0   
        self.init_qvel = jp.zeros(self.init_qpos.shape)

        # TODO: env_base if self.normalize_act implementation

        pipeline_state = self.pipeline_init(self.init_qpos, self.init_qvel) 
        pipeline_state = self.step(pipeline_state, jp.zeros(self.sys.nu))

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

        raise NotImplementedError
    
    def reset(self, rng: jax.Array = None) -> State:
        raise NotImplementedError
    
    def get_obs(self, state: State) -> jax.Array:
        raise NotImplementedError
    
    def get_info(self, state: State) -> dict:
        raise NotImplementedError
    
    def compute_reward(self, state: State, info: dict) -> dict:
        raise NotImplementedError
    
    
    
    
    
        