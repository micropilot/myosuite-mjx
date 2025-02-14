import os 
import time 
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt

import mujoco
from mujoco import mjx

from brax import base
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
                obsd_model_path:str = None,
                reference: dict = None,
                motion_start_time: float = 0,
                motion_extrapolation: bool = True,
                terminate_obj_fail: bool = True,
                terminate_pose_fail: bool = False,
                seed: int = None,
                **kwargs):
        
        # Load model and setup simulation
        processed_model_path = self.__process_path(
                                                    object_name, 
                                                    model_path, 
                                                    obsd_model_path
                                                )
        mj_model = mujoco.MjModel.from_xml_path(processed_model_path)
        sys = mjcf.load_model(mj_model)

        n_frames = 5 

        sys = sys.tree_replace({
            'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
            'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
            'opt.iterations': 1,
            'opt.ls_iterations': 4,
        })

        super().__init__(sys=sys, backend='mjx', **kwargs)

        
        self.init_qpos = jp.zeros(self.sys.nq)
        self.init_qvel = jp.zeros(self.sys.nv)

        action_range = self.sys.actuator_ctrlrange
        self.low_action = jp.array(action_range[:, 0])
        self.high_action = jp.array(action_range[:, 1])

        data = self.pipeline_init(
            self.init_qpos,
            self.init_qvel,
        )

        self.state_dim = self._get_obs(data).shape[-1]
        
        self.reward_weights_dict = self.DEFAULT_RWD_KEYS_AND_WEIGHTS

        self._load_reference_motion(mj_model, 
                                    object_name, 
                                    reference, 
                                    motion_start_time, 
                                    motion_extrapolation, 
                                    terminate_obj_fail, 
                                    terminate_pose_fail, 
                                    seed)
        
    def __process_path(self, object_name, model_path, obsd_model_path):
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

        # hack because in the super()._setup the initial posture is set to the average qpos and when a step is called, it ends in a `done` state
        self.initialized_pos = True
        # if self.sim.model.nkey>0:
        # self.init_qpos[:] = self.sim.model.key_qpos[0,:]
    
    def update_reference_insim(self, curr_ref):
        if curr_ref.object is not None:
            self.sys.data.data.site_pos[self.target_sid][:] = curr_ref.object[:3]
    
    def reset(self, rng):
        step_counter = 0 

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        rng, subkey = jax.random.split(rng)
        self.ref.reset()

        reward, done, zero = jp.zeros(3)
        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data)

        state = State(
                        data, 
                        obs,  
                        done, 
                        {'reward': zero}, 
                        {
                            'rng': rng,
                            'step_counter': step_counter,
                            'last_xfrc_applied': jp.zeros((self.sys.nbody, 6)),
                            'success': 0.,
                            'success_left': 0.,
                            'total_successes': 0.
                        }
                    )
        
        # state.info.update(**info)
        return state

    def unnorm_action(self, action):
        return (action + 1) / 2 * (self.high_action - self.low_action) + self.low_action

    def norm2(self, x):
        return jp.sum(jp.square(x))
    
    def rotation_distance(self, q1, q2, euler=True):
        if euler:
            q1 = euler2quat(q1)
            q2 = euler2quat(q2)

        return jp.abs(quatDiff2Vel(q2, q1, 1)[0])
    
    def compute_reward(self, data):
        # get reference for current time (returns a named tuple)
        curr_ref = self.ref.get_reference(data.time + self.motion_start_time)
        self.update_reference_insim(curr_ref)

        # get current hand pose + vel 
        curr_hand_qpos = data.data.qpos[:-6].copy()
        curr_hand_qvel = data.data.qvel[:-6].copy()

        # get target hand pose + vel 
        targ_hand_qpos = curr_ref.robot  
        targ_hand_qvel = jp.array([0]) if curr_ref.robot_vel is None else curr_ref.robot_vel

        # get targets from reference object
        targ_obj_com = curr_ref.object[:3]
        targ_obj_rot = curr_ref.object[3:]

        # get real values from physics object
        curr_obj_com = data.data.xipos[self.object_bid].copy()
        curr_obj_rot = mat2quat(jp.reshape(data.data.ximat[self.object_bid], (3, 3)))

        # calculate both object "matching"
        obj_com_err = jp.sqrt(self.norm2(targ_obj_com - curr_obj_com))
        obj_rot_err = self.rotation_distance(curr_obj_rot, targ_obj_rot, False) / jp.pi 
        obj_reward = jp.exp(-self.obj_err_scale * obj_com_err) * jp.exp(-self.obj_err_scale * obj_rot_err)

        # calculate lif bonus
        lift_bonus = (targ_obj_com[2] >= self.lift_z) and (curr_obj_com[2] >= self.lift_z)

        # calculate reward terms 
        hand_qpos_err = curr_hand_qpos - targ_hand_qpos
        hand_qvel_err = jp.array([0]) if curr_ref.robot_vel is None else (curr_hand_qvel - targ_hand_qvel)
        qpos_reward = jp.exp(-self.qpos_err_scale * self.norm2(hand_qpos_err))
        qvel_reward = jp.array([0]) if hand_qvel_err is None else np.exp(-self.qvel_err_scale * self.norm2(hand_qvel_err))

        # weight and sum individual reward terms 
        pose_reward = self.qpos_reward_weight * qpos_reward 
        vel_reward = self.qvel_reward_weight * qvel_reward 

        base_error = curr_obj_com - data.data.xipos[self.wrist_bid].copy()
        base_error = jp.sqrt(self.norm2(base_error))
        base_reward = jp.exp(-self.base_err_scale * base_error)

        # check for termination 
        obj_term, qpos_term, base_term = False, False, False
        if self.TermObj:
            # object too far from reference
            obj_term = (True if self.norm2(obj_com_err) >= self.obj_fail_thresh**2 else False)
            # wrist too far from object 
            base_term = (True if self.norm2(base_error) >= self.base_fail_thresh**2 else False) 
        
        if self.TermPose:
            # termination on posture 
            qpos_term = (True if self.norm2(hand_qpos_err) >= self.qpos_fail_thresh else False)

        terminated = obj_term or qpos_term or base_term

        rwd_dict = { 'pose': pose_reward + vel_reward, 
                    'object': obj_reward + base_reward, 
                    'bonus': self.lift_bonus_mag * lift_bonus,
                    'penalty': terminated,
                    'sparse': jp.array([0]),
                    'solved': jp.array([0]),
                    'done': self.initialized_pos and terminated}
        
        rwd_dict["dense"] = jp.sum(
            [wt * rwd_dict[key] for key, wt in self.reward_weights_dict.items()], axis=0
        )

        return rwd_dict["dense"], terminated, rwd_dict

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
        observation = self._get_obs(data, state.info['target_left'], state.info['target_right'])

        reward, terminated, rwd_dict = self.compute_reward(data)

        state.metrics.update(
            reward=reward
        )
        state.info.update(
            rng=rng,
            step_counter=state.info['step_counter'] + 1,
            last_xfrc_applied=xfrc_applied,
        )
        state.info.update(**rwd_dict)

        return state.replace(
            pipeline_state=data, obs=observation, reward=reward, done=terminated
        )

    def _get_obs(
            self, data
    ) -> jp.ndarray:
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