import jax 
import mujoco 
from jax import numpy as jp 

from brax.envs.base import Env, MjxEnv, State 
from myosuite.mjx.utils import perturbed_pipeline_step

class MyoFingerPoseFixedJAX(MjxEnv):

    def __init__(
        self, **kwargs
    ):  
        model_path = 'myosuite/simhive/myo_sim/finger/motorfinger_v0.xml'
        target_jnt_range = {'IFadb':(0, 0),
                            'IFmcp':(0, 0),
                            'IFpip':(.75, .75),
                            'IFdip':(.75, .75)
                            }
        viz_site_targets = ('IFtip',)
        reward_weight_dict = {
            "pose": 1.0,
            "bonus": 4.0,
            "act_reg": 1.0,
            "penalty": 50,
        }
        self.normalize_act = True,

        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.pose_thd = 0.35
        self.weight_range = None 

        if target_jnt_range:
            self.target_jnt_ids = []
            self.target_jnt_range = []
            for jnt_name, jnt_range in target_jnt_range.items():
                self.target_jnt_ids.append(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name))
                self.target_jnt_range.append(jnt_range)
            self.target_jnt_range = jp.array(self.target_jnt_range)
            self.target_jnt_value = jp.mean(self.target_jnt_range, axis=1)
        else:
            self.target_jnt_value = None

        self.tip_sids = []
        self.target_sids = []
        if viz_site_targets:
            for site in viz_site_targets:
                self.tip_sids.append(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site))
                self.target_sids.append(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site + "_target"))

        kwargs['n_frames'] = kwargs.get('n_frames', 5)

        super().__init__(model=self.mj_model, **kwargs)

        self.q_pos_init = jp.array([0.0, 0.0, 0.0, 0.0])  # for reaching
        self.q_vel_init = jp.zeros(self.sys.nv)

        action_range = self.sys.actuator_ctrlrange
        self.low_action = -jp.ones(self.sys.nu) if self.normalize_act else jp.array(action_range[:, 0])
        self.high_action = jp.ones(self.sys.nu) if self.normalize_act else jp.array(action_range[:, 1])
        data = self.pipeline_init(
            self.q_pos_init,
            self.q_vel_init,
        )

        # normalize_act has more steps for qpos and qvel

        self.state_dim = self._get_obs(data.data).shape[-1]
        self.action_dim = self.sys.nu

        assert reward_weight_dict is not None
        self.reward_weight_dict = reward_weight_dict

    def compute_reward(self, data):
        pose_dist = jp.linalg.norm(self.target_jnt_value - data.data.qpos, axis=-1)
        # Compute act_mag based on the dimensionality
        if self.sys.na > 0:
            act_mag = jp.linalg.norm(data.act.copy(), axis=-1) / self.sys.na
        else:
            act_mag = jp.linalg.norm(jp.zeros_like(data.data.qpos))
        if self.sys.na !=0: act_mag= act_mag/self.sys.na
        far_th = 4*jp.pi/2

        # Assuming pose_dist, self.pose_thd, and far_th are defined as JAX arrays
        reward = jp.where(jp.isnan(-1.0 * pose_dist), 0.0, -1.0 * pose_dist)

        # Define the termination condition based on both criteria
        terminated = jp.where((pose_dist < self.pose_thd) | (pose_dist > far_th), 1.0, 0.0)

        sub_rewards = {
            "pose": -1.*pose_dist, 
            "bonus": 1.*(pose_dist<self.pose_thd) + 1.*(pose_dist<1.5*self.pose_thd), 
            "penalty": -1.*(pose_dist>far_th),
            "act_reg": -1.*act_mag,
        }

        return reward, terminated, sub_rewards

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""

        apply_every = 1
        hold_for = 1
        magnitude = 1

        # Reset the applied forces every 200 steps
        rng, subkey = jax.random.split(state.info['rng'])
        xfrc_applied = jp.zeros((self.sys.nbody, 6))
        xfrc_applied = jax.lax.cond(
            state.info['step_counter'] % apply_every == 0,
            lambda _: jax.random.normal(subkey, shape=(self.sys.nbody, 6)) * magnitude,
            lambda _: state.info['last_xfrc_applied'], operand=None)
        # Reset to 0 every 50 steps
        perturb = jax.lax.cond(
            state.info['step_counter'] % apply_every < hold_for, lambda _: 1, lambda _: 0, operand=None)
        xfrc_applied = xfrc_applied * perturb

        action = self.unnorm_action(action)

        # Run dynamics with perturbed step
        data = perturbed_pipeline_step(self.sys, state.pipeline_state, action, xfrc_applied, self._n_frames)
        observation = self._get_obs(data.data)

        # Compute reward based on new data
        reward, terminated, sub_rewards = self.compute_reward(data)

        # Update `state.info` consistently
        state.info.update(
            rng=rng,
            step_counter=state.info['step_counter'] + 1,
            last_xfrc_applied=xfrc_applied,
        )
        state.info.update(**sub_rewards)

        return state.replace(
            pipeline_state=data,
            obs=observation,
            reward=reward,
            done=terminated
     )

    def reset(self, rng):
        step_counter = 0 

        qpos = self.q_pos_init.copy()
        qvel = self.q_vel_init.copy()
        
        reward, done, zero = jp.zeros(3)
        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data.data)
        state = State(data, obs, reward, done, 
                      {'reward': zero},
                      {'rng': rng, 
                       'step_counter': step_counter,
                       'last_xfrc_applied': jp.zeros((self.sys.nbody, 6))
                       })
        
        return state
    
    def unnorm_action(self, action):
        return (action + 1) / 2 * (self.high_action - self.low_action) + self.low_action
    
    def _get_obs(
            self, data
    ) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        return jp.concatenate(
            (   data.qpos,
                data.qvel
            )
        )


# from PIL import Image
# from matplotlib import pyplot as plt
# import matplotlib.animation as animation

# # Function to update the animation
# def update(img):
#     plt.clf()
#     plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
#     plt.margins(0,0)
#     plt.imshow(img)
#     plt.axis('off')


# env = MyoFingerPoseFixedJAX()
# # Reset the environment to get initial state
# mj_model, mj_data = env.mj_model, env.mj_data 
# renderer = mujoco.Renderer(mj_model)

# # enable joint visualization option:
# scene_option = mujoco.MjvOption()
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# def get_image():
#     mujoco.mj_resetData(mj_model, mj_data)
#     while True:
#         mujoco.mj_step(mj_model, mj_data)
#         renderer.update_scene(mj_data, scene_option=scene_option)
#         img = renderer.render()

#         yield img

# fig = plt.figure(figsize=(6, 6))
# ani = animation.FuncAnimation(fig, update, frames=get_image(), interval=20)
# plt.show()
    

