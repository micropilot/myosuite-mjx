import jax
from jax import numpy as jp
from matplotlib import pyplot as plt

import mujoco
from mujoco import mjx

from brax.envs.base import Env, MjxEnv, State


class PoseEnvV0(MjxEnv):
    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'pose_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 1.0,
        "bonus": 4.0,
        "act_reg": 1.0,
        "penalty": 50,
    }

    def __init__(self, 
                viz_site_targets:tuple = None,
                target_jnt_range:dict = None,
                target_jnt_value:list = None,
                reset_type = "init",
                target_type = "generate",
                obs_keys:list = DEFAULT_OBS_KEYS,
                weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
                pose_thd = 0.35,
                weight_bodyname = None,
                weight_range = None,
                **kwargs):
        
        # Load mj model and setup simulation
        path = "myosuite/envs/myo/assets/hand/myohand_pose.xml"
        mj_model = mujoco.MjModel.from_xml_path(path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6

        physics_steps_per_control_step = 5
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(model=mj_model, **kwargs)

        self.pose_thd = pose_thd
        self.reward_weight_dict = weighted_reward_keys

        if target_jnt_range:
            self.target_jnt_ids = []
            self.target_jnt_range = []
            for jnt_name, jnt_range in target_jnt_range.items():
                self.target_jnt_ids.append(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name))
                self.target_jnt_range.append(jnt_range)
            self.target_jnt_range = jp.array(self.target_jnt_range)
            self.target_jnt_value = jp.mean(self.target_jnt_range, axis=1)  # pseudo targets for init
        else:
            self.target_jnt_value = target_jnt_value        

        self.tip_sids = []
        self.target_sids = []
        if viz_site_targets:
            for site in viz_site_targets:
                self.tip_sids.append(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site))
                self.target_sids.append(mujoco.mj_name2id(mj_model,  mujoco.mjtObj.mjOBJ_SITE, site + "_target"))

        # TODO: Ignore normalize act for now

        self.q_pos_init = jp.array(
            [0.] * 23)
        self.q_vel_init = jp.zeros(self.sys.nv)

        action_range = self.sys.actuator_ctrlrange
        self.low_action = jp.array(action_range[:, 0])
        self.high_action = jp.array(action_range[:, 1])

        data = self.pipeline_init(
            self.q_pos_init,
            self.q_vel_init,
        )

        self.state_dim = self._get_obs(data.data, jp.array([0, 0, 0]), jp.array([0, 0, 0])).shape[-1]
        self.action_dim = self.sys.nu

    def reset(self, rng):
        # implemented in task subclass
        raise NotImplementedError

    def unnorm_action(self, action):
        return (action + 1) / 2 * (self.high_action - self.low_action) + self.low_action

    def compute_reward(self, data, info):
        # implemented in task subclass
        raise NotImplementedError
    
    def get_info(self, state, data):
        left_hand_pos = data.data.site_xpos[4]
        right_hand_pos = data.data.site_xpos[5]
        
        target_pos_left = state.info['target_left']
        target_dist_left = jp.sqrt(jp.square(left_hand_pos - target_pos_left).sum())

        target_pos_right = state.info['target_right']
        target_dist_right = jp.sqrt(jp.square(right_hand_pos - target_pos_right).sum())

        angle_targets = jp.arctan2(target_pos_right[1] - target_pos_left[1], target_pos_right[0] - target_pos_left[0])
        
        reached = jp.logical_and(target_dist_left < 0.05, target_dist_right < 0.05)
        
        max_joint_vel = jp.max(jp.abs(data.data.qvel[self.body_vel_idxs]))
        
        success = jp.where(reached, 1.0, 0.0)
        success_left = jp.where(target_dist_left < 0.05, 1.0, 0.0)
        
        total_successes = state.info['total_successes'] + success

        return {
            'hand_pos': left_hand_pos,
            'angle_targets': angle_targets,
            'target_dist_left': target_dist_left,
            'target_dist_right': target_dist_right,
            'max_joint_vel': max_joint_vel,
            'success': success,
            'success_left': success_left,
            'total_successes': total_successes  # TODO Use final and change goal after reaching
        }
    
    def _resample_target(self, state, log_info):
        # By default we don't resample target
        return state

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

        data = perturbed_pipeline_step(self.sys, state.pipeline_state, action, xfrc_applied, self._n_frames)
        observation = self._get_obs(data.data, state.info['target_left'], state.info['target_right'])

        log_info = self.get_info(state, data)
        state = self._resample_target(state, log_info)
        reward, terminated = self.compute_reward(data, log_info)

        state.metrics.update(
            reward=reward
        )
        state.info.update(
            rng=rng,
            step_counter=state.info['step_counter'] + 1,
            last_xfrc_applied=xfrc_applied,
        )
        state.info.update(**log_info)

        return state.replace(
            pipeline_state=data, obs=observation, reward=reward, done=terminated
        )

    def _get_obs(
            self, data, target_left, target_right=None
    ) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""

        raise NotImplementedError