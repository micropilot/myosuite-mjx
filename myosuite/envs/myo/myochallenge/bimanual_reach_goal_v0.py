import collections
import enum
import os, time

from dm_control.utils import rewards
from scipy.spatial.transform import Rotation as R
import mujoco
from myosuite.utils import gym
import numpy as np
from myosuite.utils.quat_math import mat2euler, euler2quat
from typing import List

from myosuite.envs.myo.base_v0 import BaseV0

CONTACT_TRAJ_MIN_LENGTH = 100
GOAL_CONTACT = 10
MAX_TIME = 10.0


class BimanualReachGoalV0(BaseV0):
    DEFAULT_OBS_KEYS = ["time", "myohand_qpos", "myohand_qvel", "pros_hand_qpos", "pros_hand_qvel", "touching_body"]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "act": 0,
        "arm_goal_dist": -0.6,  # Penalizes distance from the goal
        "mpl_goal_dist": -0.4,  # Penalizes distance from the goal
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    def _setup(self,
               frame_skip: int = 10,
               arm_start=np.array([-0.4, -0.25, 1.05]),  # Start and goal centers, pos = center + shift * [0, 1]
               mpl_start=np.array([0.4, -0.25, 1.05]),
               goal_center=np.array([0.0, 0.15, 0.95]),
               proximity_th=0.005,  # Proximity threshold for success

               arm_start_shifts=np.array([0.055, 0.055, 0]),
               mpl_start_shifts=np.array([0.055, 0.055, 0]),
               goal_shifts=np.array([0.098, 0.098, 0]),

               task_choice='fixed',  # fixed/ random
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
               ):

        # user parameters
        self.task_choice = task_choice
        self.proximity_th = proximity_th

        # start position centers (before changes)
        self.arm_start = arm_start
        self.mpl_start = mpl_start
        self.goal_center = goal_center

        self.arm_start_shifts = arm_start_shifts
        self.mpl_start_shifts = mpl_start_shifts
        self.goal_shifts = goal_shifts
        self.PILLAR_HEIGHT = 1.09

        self.id_info = IdInfo(self.sim.model)

        self.start_left_bid = self.id_info.start_left_id
        self.start_right_bid = self.id_info.start_right_id
        self.goal_bid = self.id_info.goal_id

        # define the palm and tip site id.
        # arm
        self.palm_sid = self.sim.model.site_name2id('S_grasp')
        self.init_palm_z = self.sim.data.site_xpos[self.palm_sid][-1]
        
        self.fin0 = self.sim.model.site_name2id("THtip")
        self.fin1 = self.sim.model.site_name2id("IFtip")
        self.fin2 = self.sim.model.site_name2id("MFtip")
        self.fin3 = self.sim.model.site_name2id("RFtip")
        self.fin4 = self.sim.model.site_name2id("LFtip")

        # mpl
        self.Rpalm1_sid = self.sim.model.site_name2id('prosthesis/palm_thumb')
        self.Rpalm2_sid = self.sim.model.site_name2id('prosthesis/palm_pinky')

        self.arm_start_pos = self.arm_start
        self.mpl_start_pos = self.mpl_start
        self.goal_pos = self.goal_center

        print (self.sim.model.qpos0.shape, self.sim.model.body_pos.shape, self.start_left_bid)
        self.sim.model.body_pos[self.start_left_bid] = self.arm_start_pos
        self.sim.model.body_pos[self.start_right_bid] = self.mpl_start_pos
        self.sim.model.body_pos[self.goal_bid] = self.goal_pos

        # check whether the object experience force over max force
        self.TARGET_GOAL_TOUCH = GOAL_CONTACT


        self.touch_history = []

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       frame_skip=frame_skip,
                       **kwargs,
                       )

        self.init_qpos[:] = self.sim.model.key_qpos[2].copy()
        # adding random disturbance to start and goal positions, coefficients might need to be adaptable
        self.initialized_pos = False

    def _obj_label_to_obs(self, touching_body):
        # Function to convert touching body set to a binary observation vector
        # order follows the definition in enum class
        obs_vec = np.array([0, 0, 0, 0, 0])
        for i in touching_body:
            if i == ObjLabels.MYO:
                obs_vec[0] += 1
            elif i == ObjLabels.PROSTH:
                obs_vec[1] += 1
            elif i == ObjLabels.ARM_START:
                obs_vec[2] += 1
            elif i == ObjLabels.MPL_START:
                obs_vec[3] += 1
            else:
                obs_vec[4] += 1

        return obs_vec

    def get_obs_dict(self, sim):
        obs_dict = {}

        obs_dict["time"] = np.array([self.sim.data.time])
        obs_dict["qp"] = sim.data.qpos.copy()
        obs_dict["qv"] = sim.data.qvel.copy()

        # MyoHand data
        obs_dict["myohand_qpos"] = sim.data.qpos[self.id_info.myo_joint_range].copy()
        obs_dict["myohand_qvel"] = sim.data.qvel[self.id_info.myo_dof_range].copy()

        # Prosthetic hand data and velocity
        obs_dict["pros_hand_qpos"] = sim.data.qpos[self.id_info.prosth_joint_range].copy()
        obs_dict["pros_hand_qvel"] = sim.data.qvel[self.id_info.prosth_dof_range].copy()

        # One more joint for qpos due to </freejoint>
        obs_dict["arm_start_pos"] = self.arm_start_pos
        obs_dict["mpl_start_pos"] = self.mpl_start_pos
        obs_dict["goal_pos"] = self.goal_pos
        obs_dict["elbow_fle"] = self.sim.data.joint('elbow_flexion').qpos.copy()

        this_model = sim.model
        this_data = sim.data

        # Get touching object in terms of binary encoding
        touching_objects = set(get_touching_objects(this_model, this_data, self.id_info))
        self.touch_history.append(touching_objects)

        obs_vec = self._obj_label_to_obs(touching_objects)
        obs_dict["touching_body"] = obs_vec
        obs_dict["palm_pos"] = sim.data.site_xpos[self.palm_sid]
        obs_dict['fin0'] = sim.data.site_xpos[self.fin0]
        obs_dict['fin1'] = sim.data.site_xpos[self.fin1]
        obs_dict['fin2'] = sim.data.site_xpos[self.fin2]
        obs_dict['fin3'] = sim.data.site_xpos[self.fin3]
        obs_dict['fin4'] = sim.data.site_xpos[self.fin4]

        obs_dict["Rpalm_pos"] = (sim.data.site_xpos[self.Rpalm1_sid] + sim.data.site_xpos[self.Rpalm2_sid]) / 2
        obs_dict["Rpalm_thumb"] = sim.data.site_xpos[self.Rpalm1_sid]
        obs_dict["Rpalm_pinky"] = sim.data.site_xpos[self.Rpalm2_sid]

        obs_dict['MPL_ori'] = mat2euler(np.reshape(self.sim.data.site_xmat[self.Rpalm1_sid], (3, 3)))
        obs_dict['MPL_ori_err'] = obs_dict['MPL_ori'] - np.array([np.pi, 0, np.pi])

        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):

        act = np.linalg.norm(obs_dict['act'], axis=-1)[0][0] / self.sim.model.na if self.sim.model.na != 0 else 0

        arm_goal_dist = np.abs(np.linalg.norm(obs_dict["fin0"] - obs_dict["goal_pos"], axis=-1))[0][0]
        mpl_goal_dist = np.abs(np.linalg.norm(obs_dict["Rpalm_thumb"] - obs_dict["goal_pos"], axis=-1))[0][0]
        
        # elbow_err = 5 * np.exp(-10 * (obs_dict['elbow_fle'][0] - 1.) ** 2) - 5

        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("act", act),
                ("arm_goal_dist", arm_goal_dist),
                ("mpl_goal_dist", mpl_goal_dist),
                # ("lift_bonus", elbow_err),
                # Must keys
                ("sparse", 0),
                ("goal_dist", arm_goal_dist + mpl_goal_dist),
                ("solved", (arm_goal_dist < self.proximity_th) and (mpl_goal_dist < self.proximity_th)),
                ("done", self._get_done(arm_goal_dist, mpl_goal_dist)),
            )
        )

        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        return rwd_dict

    def _get_done(self, z1, z2):
        if self.obs_dict['time'] > MAX_TIME:
            return 1  
        elif (z1 < self.proximity_th) and (z2 < self.proximity_th):
            self.obs_dict['time'] = MAX_TIME
            return 1
        elif self.rwd_dict and self.rwd_dict['solved']:
            return 1
        return 0

    def step(self, a, **kwargs):
        # We unnormalize robotic actuators, muscle ones are handled in the parent implementation
        processed_controls = a.copy()
        if self.normalize_act:
            robotic_act_ind = self.sim.model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE
            processed_controls[robotic_act_ind] = (np.mean(self.sim.model.actuator_ctrlrange[robotic_act_ind], axis=-1)
                                                   + processed_controls[robotic_act_ind]
                                                   * (self.sim.model.actuator_ctrlrange[robotic_act_ind, 1]
                                                      - self.sim.model.actuator_ctrlrange[robotic_act_ind, 0]) / 2.0)
        return super().step(processed_controls, **kwargs)

    def get_metrics(self, paths, successful_steps=5):
        """
        Evaluate paths and report metrics
        """
        num_success = 0
        num_paths = len(paths)

        # average sucess over entire env horizon
        for path in paths:
            # record success if solved for provided successful_steps, check how the path is stored
            if np.sum(path['env_infos']['rwd_dict']['solved'] * 1.0) > successful_steps and evaluate_contact_trajectory(
                    path['env_infos']['touch_history']) is None:
                num_success += 1
        score = num_success / num_paths

        times = np.mean([np.round(p['env_infos']['obs_dict']['time'][-1], 5) for p in paths])
        max_force = np.mean([np.round(p['env_infos']['obs_dict']['max_force'][-1], 5) for p in paths])
        goal_dist = np.mean([np.mean(p['env_infos']['rwd_dict']['goal_dist']) for p in paths])

        # average activations over entire trajectory (can be shorter than horizon, if done) realized
        effort = 1.0 * np.mean([np.mean(p['env_infos']['rwd_dict']['act']) for p in paths])

        metrics = {
            'score': score,
            'time': times,
            'effort': effort,
            'peak force': max_force,
            'goal dist': goal_dist, 
        }
        return metrics

    def reset(self, **kwargs):
        self.arm_start_pos = self.arm_start + self.arm_start_shifts * (2 * self.np_random.random(3) - 1)
        self.mpl_start_pos = self.mpl_start + self.mpl_start_shifts * (2 * self.np_random.random(3) - 1)
        self.goal_pos = self.goal_center + self.goal_shifts * (2 * self.np_random.random(3) - 1)
        #
        self.sim.model.body_pos[self.start_left_bid] = self.arm_start_pos
        self.sim.model.body_pos[self.start_right_bid] = self.mpl_start_pos
        self.touch_history = []
        self.over_max = False
        self.goal_touch = 0

        self.sim.forward()

        self.init_qpos[:] = self.sim.model.key_qpos[2].copy()
        # self.init_qpos[:-14] *= 0 # Use fully open as init pos

        obs = super().reset(
            reset_qpos=self.init_qpos, reset_qvel=self.init_qvel, **kwargs
        )
        self.init_palm_z = self.sim.data.site_xpos[self.palm_sid][-1]
        return obs


class ObjLabels(enum.Enum):
    MYO = 0
    PROSTH = 1
    ARM_START = 2
    MPL_START = 3
    GOAL = 4
    ENV = 5

class ContactTrajIssue(enum.Enum):
    MYO_SHORT = 0
    PROSTH_SHORT = 1
    NO_GOAL = 2  # Maybe can enforce implicitly, and only declare success is sufficient consecutive frames with only
    # goal contact.
    ENV_CONTACT = 3

class IdInfo:
    def __init__(self, model: mujoco.MjModel):
        myo_bodies = [model.body(i).id for i in range(model.nbody)
                      if not model.body(i).name.startswith("prosthesis")
                      and not model.body(i).name in ["start_left", "start_right"]]
        self.myo_body_range = (min(myo_bodies), max(myo_bodies))

        prosth_bodies = [model.body(i).id for i in range(model.nbody) if model.body(i).name.startswith("prosthesis/")]
        self.prosth_body_range = (min(prosth_bodies), max(prosth_bodies))

        self.myo_joint_range = np.concatenate([model.joint(i).qposadr for i in range(model.njnt)
                                               if not model.joint(i).name.startswith("prosthesis")])

        self.myo_dof_range = np.concatenate([model.joint(i).dofadr for i in range(model.njnt)
                                             if not model.joint(i).name.startswith("prosthesis")])

        self.prosth_joint_range = np.concatenate([model.joint(i).qposadr for i in range(model.njnt)
                                                  if model.joint(i).name.startswith("prosthesis")])

        self.prosth_dof_range = np.concatenate([model.joint(i).dofadr for i in range(model.njnt)
                                                if model.joint(i).name.startswith("prosthesis")])


        self.start_left_id = model.body("start_left").id
        self.start_right_id = model.body("start_right").id
        self.goal_id = model.body("goal").id


def get_touching_objects(model: mujoco.MjModel, data: mujoco.MjData, id_info: IdInfo):
    for con in data.contact:
        if model.geom(con.geom1).bodyid == id_info.start_left_id:
            yield body_id_to_label(model.geom(con.geom2).bodyid, id_info)
        elif model.geom(con.geom2).bodyid == id_info.start_right_id:
            yield body_id_to_label(model.geom(con.geom1).bodyid, id_info)


def body_id_to_label(body_id, id_info: IdInfo):
    if id_info.myo_body_range[0] <= body_id <= id_info.myo_body_range[1]:
        return ObjLabels.MYO
    elif id_info.prosth_body_range[0] <= body_id <= id_info.prosth_body_range[1]:
        return ObjLabels.PROSTH
    elif body_id == id_info.start_left_id:
        return ObjLabels.ARM_START
    elif body_id == id_info.start_right_id:
        return ObjLabels.MPL_START
    elif body_id == id_info.goal_id:
        return ObjLabels.GOAL
    else:
        return ObjLabels.ENV


def evaluate_contact_trajectory(contact_trajectory: List[set]):
    for s in contact_trajectory:
        if ObjLabels.ENV in s:
            return ContactTrajIssue.ENV_CONTACT

    myo_frames = np.nonzero([ObjLabels.MYO in s for s in contact_trajectory])[0]
    prosth_frames = np.nonzero([ObjLabels.PROSTH in s for s in contact_trajectory])[0]

    if len(myo_frames) < CONTACT_TRAJ_MIN_LENGTH:
        return ContactTrajIssue.MYO_SHORT
    elif len(prosth_frames) < CONTACT_TRAJ_MIN_LENGTH:
        return ContactTrajIssue.PROSTH_SHORT

    # Check if only goal was touching object for the last CONTACT_TRAJ_MIN_LENGTH frames
    elif len([s for s in contact_trajectory if ObjLabels.ARM_START in s and ObjLabels.MPL_START in s]) < GOAL_CONTACT:
        return ContactTrajIssue.NO_GOAL
