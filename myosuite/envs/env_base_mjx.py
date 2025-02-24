from brax.envs.base import PipelineEnv, State
import jax
import mujoco
from brax.io import mjcf


class EnvBaseMJX(PipelineEnv):
    def __init__(
        self, 
        model_path: str,
        obs_keys: list, 
        weighted_reward_keys: dict, 
        frame_skip: int = 10, 
        **kwargs
    ):  
        model = mujoco.MjModel.from_xml_path(model_path)
        sys = mjcf.load_model(model)

        n_frames = 10
        sys = sys.tree_replace(
            {
                "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                "opt.iterations": 1,
                "opt.ls_iterations": 4,
            }
        )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend="mjx", **kwargs)
        
    def step(self, state: State, action: jax.Array) -> State:
        raise NotImplementedError
    
    def reset(self, rng: jax.Array = None) -> State:
        raise NotImplementedError
    
    def get_obs(self, state: State) -> jax.Array:
        raise NotImplementedError
    
    
    
        