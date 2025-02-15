import os 
import time 
import jax
from jax import numpy as jp

import mujoco

from brax.io import mjcf
from brax.envs.base import PipelineEnv, State

from myosuite.mjx.reference_motion import ReferenceMotion
from myosuite.mjx.quat_math import euler2quat, quat2euler, quatDiff2Vel, mat2quat
from myosuite.mjx.utils import perturbed_pipeline_step


class TrackEnv(PipelineEnv):
    DEFAULT_OBS_KEYS = ["qp", "qv", "hand_qpos_err", "hand_qvel_err", "obj_com_err"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 0.0,  # 1.0,
        "object": 1.0,
        "bonus": 1.0,
        "penalty": -2,
    }

    def __init__(self, 
                model_path:str = None, 
                object_name:str = None,
                reference: dict = None,
                motion_start_time: float = 0,
                motion_extrapolation: bool = True,
                terminate_obj_fail: bool = True,
                terminate_pose_fail: bool = False,
                seed: int = None,
                **kwargs):
        
        # Load model and setup simulation
        processed_model_path = self.__process_path(object_name, model_path)
        mj_model = mujoco.MjModel.from_xml_path(processed_model_path)
        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys=sys, **kwargs)
        
        self.reward_weights_dict = self.DEFAULT_RWD_KEYS_AND_WEIGHTS

        self.init_qpos = self.sys.init_q
        
        self._load_reference_motion(mj_model, 
                                    object_name, 
                                    reference, 
                                    motion_start_time, 
                                    motion_extrapolation, 
                                    terminate_obj_fail, 
                                    terminate_pose_fail, 
                                    seed)
        
    def __process_path(self, object_name, model_path):
        # Load mj model and setup simulation
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.object_name = object_name
        time_stamp = str(time.time())

        # Process model_path to import the right object
        with open(curr_dir + model_path, "r") as file:
            processed_xml = file.read()
            processed_xml = processed_xml.replace("OBJECT_NAME", object_name)
        processed_model_path = (
            curr_dir + model_path[:-4] + time_stamp + "_processed.xml"
        )
        with open(processed_model_path, "w") as file:
            file.write(processed_xml)

        return processed_model_path
        
    def _load_reference_motion(self, 
                               mj_model,
                               object_name,
                               reference, 
                               motion_start_time, 
                               motion_extrapolation, 
                               terminate_obj_fail,
                               terminate_pose_fail,    
                               seed):
        self.ref = ReferenceMotion(
            reference_data=reference,
            motion_extrapolation=motion_extrapolation,
            rng_key=seed,
        )

        self.motion_start_time = motion_start_time
        self.target_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "target")

        ##########################################
        self.lift_bonus_thresh = 0.02
        ### PRE-GRASP
        self.obj_err_scale = 50
        self.base_err_scale = 40
        self.lift_bonus_mag = 1  # 2.5

        ### DEEPMIMIC
        self.qpos_reward_weight = 0.35
        self.qpos_err_scale = 5.0

        self.qvel_reward_weight = 0.05
        self.qvel_err_scale = 0.1

        # TERMINATIONS FOR OBJ TRACK
        self.obj_fail_thresh = 0.25
        # TERMINATIONS FOR HAND-OBJ DISTANCE
        self.base_fail_thresh = 0.25
        self.TermObj = terminate_obj_fail

        # TERMINATIONS FOR MIMIC
        self.qpos_fail_thresh = 0.75
        self.TermPose = terminate_pose_fail
        ##########################################

        self.object_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        self.wrist_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "lunate")

        # Disable body skeleton rendering by setting transparency
        mj_model.geom_rgba[self.object_bid, 3] = 0.0  # Make all geoms invisible

        ipos = self.sys.mj_model.body_ipos[self.object_bid]
        pos = self.sys.mj_model.body_pos[self.object_bid]
        self.lift_z = (ipos + pos)[2] + self.lift_bonus_thresh
 
        if motion_extrapolation == False:
            self.spec.max_episode_steps = self.ref.horizon

        robot_init, object_init = self.ref.get_init()
        if robot_init is not None:
            self.init_qpos = self.init_qpos.at[: self.ref.robot_dim].set(robot_init)
        if object_init is not None:
            self.init_qpos = self.init_qpos.at[self.ref.robot_dim : self.ref.robot_dim + 3].set(object_init[:3])
            self.init_qpos = self.init_qpos.at[-3:].set(quat2euler(object_init[3:]))

    
    def reset(self, rng):
        # qpos and qvel contain both hand and object pose and vel
        qpos = self.init_qpos
        qvel = jp.zeros(self.sys.qd_size())
        
        rng, rng1, rng2 = jax.random.split(rng, 3)
        self.ref.reset()

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jp.zeros(3)

        metrics = {
                    'pose': zero, 
                    'object': zero, 
                    'bonus': zero,
                    'penalty': zero,
                }

        state = State(pipeline_state, obs, reward, done, metrics)
        
        return state

    def norm2(self, x):
        return jp.sum(jp.square(x))
    
    def rotation_distance(self, q1, q2, euler=True):
        if euler:
            q1 = euler2quat(q1)
            q2 = euler2quat(q2)

        return jp.abs(quatDiff2Vel(q2, q1, 1)[0])
    
    def update_reference_insim(self, curr_ref):
        if curr_ref.object is not None:
            # Create a new instance of the system with updated site_pos
            new_site_pos = self.sys.site_pos.at[self.target_sid].set(curr_ref.object[:3])
            
            # Assuming self.sys is a dataclass, create a new instance with updated site_pos
            self.sys = self.sys.replace(site_pos=new_site_pos)
    
    def compute_reward(self, curr_ref,data):
        # get current hand pose + vel 
        curr_hand_qpos = data.q[:-6].copy()
        curr_hand_qvel = data.qd[:-6].copy()

        # get target hand pose + vel 
        targ_hand_qpos = curr_ref.robot  
        targ_hand_qvel = jp.array([0]) if curr_ref.robot_vel is None else curr_ref.robot_vel

        # get targets from reference object
        targ_obj_com = curr_ref.object[:3]
        targ_obj_rot = curr_ref.object[3:]

        # get real values from physics object
        curr_obj_com = data.xipos[self.object_bid].copy()
        curr_obj_rot = mat2quat(data.ximat[self.object_bid])

        # calculate both object "matching"
        obj_com_err = jp.sqrt(self.norm2(targ_obj_com - curr_obj_com))
        obj_rot_err = self.rotation_distance(curr_obj_rot, targ_obj_rot, False) / jp.pi 
        obj_reward = jp.exp(-self.obj_err_scale * obj_com_err) * jp.exp(-self.obj_err_scale * obj_rot_err)

        # calculate lif bonus
        lift_bonus = (targ_obj_com[2] >= self.lift_z) * (curr_obj_com[2] >= self.lift_z)

        # calculate reward terms 
        hand_qpos_err = curr_hand_qpos - targ_hand_qpos
        hand_qvel_err = jp.array([0]) if curr_ref.robot_vel is None else (curr_hand_qvel - targ_hand_qvel)
        qpos_reward = jp.exp(-self.qpos_err_scale * self.norm2(hand_qpos_err))
        qvel_reward = jp.array([0]) if hand_qvel_err is None else jp.exp(-self.qvel_err_scale * self.norm2(hand_qvel_err))

        # weight and sum individual reward terms 
        pose_reward = self.qpos_reward_weight * qpos_reward 
        vel_reward = self.qvel_reward_weight * qvel_reward 

        base_error = curr_obj_com - data.xipos[self.wrist_bid].copy()
        base_error = jp.sqrt(self.norm2(base_error))
        base_reward = jp.exp(-self.base_err_scale * base_error)

        # check for termination 
        obj_term, qpos_term, base_term = False, False, False
        if self.TermObj:
            # object too far from reference
            obj_term = self.norm2(obj_com_err) >= self.obj_fail_thresh**2
            obj_term = jp.where(obj_term, 1, 0)  # Convert to integer
            # wrist too far from object 
            base_term = self.norm2(base_error) >= self.base_fail_thresh**2
            base_term = jp.where(base_term, 1, 0)  # Convert to integer
        
        if self.TermPose:
            # termination on posture 
            qpos_term = self.norm2(hand_qpos_err) >= self.qpos_fail_thresh
            qpos_term = jp.where(qpos_term, 1, 0)  # Convert to integer

        done = (obj_term + qpos_term + base_term) > 0

        metrics = { 'pose': jp.float32(pose_reward + vel_reward), 
                    'object': jp.float32(obj_reward + base_reward), 
                    'bonus': jp.float32(self.lift_bonus_mag * lift_bonus),
                    'penalty': jp.float32(done),
                    }
        
        reward = jp.sum(
            jp.array([wt * metrics[key] for key, wt in self.reward_weights_dict.items()]), axis=0
        )

        return reward, done, metrics

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""

        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator_ctrlrange[:, 0]
        action_max = self.sys.actuator_ctrlrange[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        # get reference for current time (returns a named tuple)
        pipeline_state0 = state.pipeline_state
        curr_ref = self.ref.get_reference(pipeline_state0.time + self.motion_start_time)
        self.update_reference_insim(curr_ref)

        pipeline_state = self.pipeline_step(pipeline_state0, action)
        obs = self._get_obs(pipeline_state)

        reward, done, metrics = self.compute_reward(curr_ref, pipeline_state)

        state.metrics.update(
            **metrics
        )
        return state.replace(pipeline_state=pipeline_state, 
                             obs=obs, 
                             reward=reward,
                             done=done)

    def _get_obs(
            self, data
    ) -> jp.ndarray:
        # ToDo add time for cyclic tasks
        return jp.concatenate(
            (   data.qpos,
                data.qvel,
            )
        )



# dof_robot = 29
# model_path = '/../envs/myo/assets/hand/myohand_object.xml'
# object_name = 'airplane'
# reference =  {'time':(0.0, 4.0),
#             'robot':jp.zeros((2, dof_robot)),
#             'robot_vel':jp.zeros((2, dof_robot)),
#             'object_init':jp.array((0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0)),
#             'object':jp.array([ [-.2, -.2, 0.1, 1.0, 0.0, 0.0, -1.0],
#                                 [0.2, 0.2, 0.1, 1.0, 0.0, 0.0, 1.0]])
#             }

# env = TrackEnv(model_path=model_path, 
#                object_name=object_name, 
#                reference=reference,)
# jit_reset = jax.jit(env.reset)
# jit_step = jax.jit(env.step)

# print ("It has been jitted")

# # initialize the state
# state = jit_reset(jax.random.PRNGKey(0))
# rollout = [state.pipeline_state]

# print ("State initialized")

# # grab a trajectory
# for i in range(10):
#   ctrl = -0.1 * jp.ones(env.sys.nu)
#   state = jit_step(state, ctrl)
#   rollout.append(state.pipeline_state)

# print ("Trajectory grabbed")