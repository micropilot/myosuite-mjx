import os 
import jax 
import mujoco 
from jax import numpy as jp 

from brax.envs.base import Env, MjxEnv, State 
from myosuite.envs.myo.fatigue_jax import CumulativeFatigueJAX

class MyoFingerPoseFixedJAX(MjxEnv):

    def __init__(
        self, model_path, **kwargs
    ):
        target_jnt_range = {'IFadb':(0, 0),
                            'IFmcp':(0, 0),
                            'IFpip':(.75, .75),
                            'IFdip':(.75, .75)
                            }
        viz_site_targets = ('IFtip',)
        self.normalize_act = True,

        mj_model = mujoco.MjModel.from_xml_path(model_path)

        self.reset_type = "init"
        self.target_type = "generate"
        self.pose_thd = 0.35
        self.weight_bodyname = {
                            "pose": 1.0,
                            "bonus": 4.0,
                            "act_reg": 1.0,
                            "penalty": 50,
                        }
        self.weight_range = None 

        if target_jnt_range:
            self.target_jnt_ids = []
            self.target_jnt_range = []
            for jnt_name, jnt_range in target_jnt_range.items():
                self.target_jnt_ids.append(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name))
                self.target_jnt_range.append(jnt_range)
            self.target_jnt_range = jp.array(self.target_jnt_range)
            self.target_jnt_value = jp.mean(self.target_jnt_range, axis=1)
        else:
            self.target_jnt_value = target_jnt_value

        self.tip_sids = []
        self.target_sids = []
        if viz_site_targets:
            for site in viz_site_targets:
                self.tip_sids.append(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site))
                self.target_sids.append(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site + "_target"))

        self.muscle_condition = ""
        self.fatigue_reset_vec = None
        self.fatigue_reset_random = None
        self.frame_skip = 10
        self.initializeConditions()

        super().__init__(model=mj_model, **kwargs)

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

        assert self.weight_bodyname is not None
        self.reward_weight_dict = self.weight_bodyname

    def initializeConditions(self):
        # for muscle weakness we assume that a weaker muscle has a
        # reduced maximum force
        if self.muscle_condition == "sarcopenia":
            for mus_idx in range(self.sim.model.actuator_gainprm.shape[0]):
                self.sim.model.actuator_gainprm[mus_idx, 2] = (
                    0.5 * self.sim.model.actuator_gainprm[mus_idx, 2].copy()
                )

        # for muscle fatigue we used the 3CC-r model
        elif self.muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(
                self.sim.model, self.frame_skip, seed=self.get_input_seed()
            )

        # Tendon transfer to redirect EIP --> EPL
        # https://www.assh.org/handcare/condition/tendon-transfer-surgery
        elif self.muscle_condition == "reafferentation":
            self.EPLpos = self.sim.model.actuator_name2id("EPL")
            self.EIPpos = self.sim.model.actuator_name2id("EIP")

    def step(self, state: State, action: jp.ndarray) -> State:
        muscle_a = action.copy()
        muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        # Explicitely project normalized space (-1,1) to actuator space (0,1) if muscles
        if self.model.na and self.normalize_act:
            # find muscle actuators
            muscle_a[muscle_act_ind] = 1.0 / (
                1.0 + np.exp(-5.0 * (muscle_a[muscle_act_ind] - 0.5))
            )
            # TODO: actuator space may not always be (0,1) for muscle or (-1, 1) for others
            isNormalized = (
                False  # refuse internal reprojection as we explicitly did it here
            )
        else:
            isNormalized = self.normalize_act  # accept requested reprojection

        # implement abnormalities
        if self.muscle_condition == "fatigue":
            # import ipdb; ipdb.set_trace()
            muscle_a[muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                muscle_a[muscle_act_ind]
            )
        elif self.muscle_condition == "reafferentation":
            # redirect EIP --> EPL
            muscle_a[self.EPLpos] = muscle_a[self.EIPpos].copy()
            # Set EIP to 0
            muscle_a[self.EIPpos] = 0
        
        self.last_ctrl = self.robot.step(
            ctrl_desired=muscle_a,
            ctrl_normalized=isNormalized,
            step_duration=self.dt,
            realTimeSim=self.mujoco_render_frames,
            render_cbk=self.mj_render if self.mujoco_render_frames else None,
        )

    def reset(self, **kwargs):

        # udpate wegith
        if self.weight_bodyname is not None:
            bid = self.sim.model.body_name2id(self.weight_bodyname)
            gid = self.sim.model.body_geomadr[bid]
            weight = self.np_random.uniform(low=self.weight_range[0], high=self.weight_range[1])
            self.sim.model.body_mass[bid] = weight
            self.sim_obsd.model.body_mass[bid] = weight
            # self.sim_obsd.model.geom_size[gid] = self.sim.model.geom_size[gid] * weight/10
            self.sim.model.geom_size[gid][0] = 0.01 + 2.5*weight/100
            # self.sim_obsd.model.geom_size[gid][0] = weight/10

        # update target
        if self.target_type == "generate":
            # use target_jnt_range to generate targets
            self.update_target(restore_sim=True)
        elif self.target_type == "switch":
            # switch between given target choices
            # TODO: Remove hard-coded numbers
            if self.target_jnt_value[0] != -0.145125:
                self.target_jnt_value = np.array([-0.145125, 0.92524251, 1.08978337, 1.39425813, -0.78286243, -0.77179383, -0.15042819, 0.64445902])
                self.sim.model.site_pos[self.target_sids[0]] = np.array([-0.11000209, -0.01753063, 0.20817679])
                self.sim.model.site_pos[self.target_sids[1]] = np.array([-0.1825131, 0.07417956, 0.11407256])
                self.sim.forward()
            else:
                self.target_jnt_value = np.array([-0.12756566, 0.06741454, 1.51352705, 0.91777418, -0.63884237, 0.22452487, 0.42103326, 0.4139465])
                self.sim.model.site_pos[self.target_sids[0]] = np.array([-0.11647777, -0.05180014, 0.19044284])
                self.sim.model.site_pos[self.target_sids[1]] = np.array([-0.17728016, 0.01489491, 0.17953786])
        elif self.target_type == "fixed":
            self.update_target(restore_sim=True)
        else:
            print("{} Target Type not found ".format(self.target_type))

        # update init state
        if self.reset_type is None or self.reset_type == "none":
            # no reset; use last state
            ## NOTE: fatigue is also not reset in this case!
            obs = self.get_obs()
        elif self.reset_type == "init":
            # reset to init state
            obs = super().reset(**kwargs)
        elif self.reset_type == "random":
            # reset to random state
            jnt_init = self.np_random.uniform(high=self.sim.model.jnt_range[:,1], low=self.sim.model.jnt_range[:,0])
            obs = super().reset(reset_qpos=jnt_init, **kwargs)
        else:
            print("Reset Type not found")

        return obs

    def _get_obs(
            self, data
    ) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        pose_err = self.target_jnt_value - data.qpos
        act = data.act.copy() if self.sys.na > 0 else jp.zeros_like(data.qpos)

        return jp.concatenate(
            (   data.qpos,
                data.qvel,
                pose_err,
                act
            )
        )


base = MyoFingerPoseFixedJAX(model_path='myosuite/simhive/myo_sim/finger/motorfinger_v0.xml')