from jax import numpy as jp
import mujoco
import collections
from brax.envs.base import MjxEnv, State

from myosuite.logger.reference_motion_jax import ReferenceMotion
from myosuite.utils.quat_math_jax import euler2quat, mat2quat, quatDiff2Vel
from myosuite.utils.fatigue import CumulativeFatigueJAX
from myosuite.utils.pipeline import perturbed_pipeline_step
from myosuite.utils.tolerance import tolerance


class MyoHandAirplaneV0(MjxEnv):
    DEFAULT_OBS_KEYS = ["qp", "qv", "hand_qpos_err", "hand_qvel_err", "obj_com_err"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 0.0,
        "object": 1.0,
        "bonus": 1.0,
        "penalty": -2,
    }

    def __init__(
        self,
        model_path,
        weighted_reward_keys={},
        frame_skip=10,
        muscle_condition="",
        **kwargs
    ):
        mj_model = mujoco.MjModel.from_xml_path(model_path)

        # Additional environment setup
        self.frame_skip = frame_skip
        self.muscle_condition = muscle_condition

        # Fatigue setup if needed
        self.fatigue_model = None
        if self.muscle_condition == "fatigue":
            self.fatigue_model = CumulativeFatigueJAX(
                mj_model, frame_skip, seed=kwargs.get("seed", 0)
            )

        # Initialize position and velocity arrays
        self.q_pos_init = jp.zeros(mj_model.nq)
        self.q_vel_init = jp.zeros(mj_model.nv)

        # For JAX, define control limits
        action_range = mj_model.actuator_ctrlrange
        self.low_action = jp.array(action_range[:, 0])
        self.high_action = jp.array(action_range[:, 1])

        super().__init__(model=mj_model, **kwargs)
        self._setup(**kwargs)

    def _setup(
        self,
        reference,
        motion_start_time=0,
        motion_extrapolation=True,
        obs_keys=None,
        weighted_reward_keys=None,
        Termimate_obj_fail=True,
        Termimate_pose_fail=False,
        **kwargs
    ):
        self.ref = ReferenceMotion(reference, motion_extrapolation=motion_extrapolation)
        self.motion_start_time = motion_start_time

        self.lift_bonus_thresh = 0.02
        self.obj_err_scale = 50
        self.base_err_scale = 40
        self.lift_bonus_mag = 1

        self.qpos_reward_weight = 0.35
        self.qpos_err_scale = 5.0
        self.qvel_reward_weight = 0.05
        self.qvel_err_scale = 0.1

        self.obj_fail_thresh = 0.25
        self.base_fail_thresh = 0.25
        self.TermObj = Termimate_obj_fail
        self.qpos_fail_thresh = 0.75
        self.TermPose = Termimate_pose_fail

        # Initialize observation keys and reward weights
        obs_keys = obs_keys or self.DEFAULT_OBS_KEYS
        weighted_reward_keys = weighted_reward_keys or self.DEFAULT_RWD_KEYS_AND_WEIGHTS
        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys)

    def get_obs_dict(self, data):
        obs_dict = {}
        curr_ref = self.ref.get_reference(data.time + self.motion_start_time)

        # Update target positions in simulation
        obs_dict["time"] = jp.array([data.time])
        obs_dict["qp"] = data.qpos.copy()
        obs_dict["qv"] = data.qvel.copy()
        obs_dict["robot_err"] = obs_dict["qp"][:-6] - curr_ref.robot

        obs_dict["curr_hand_qpos"] = data.qpos[:-6]
        obs_dict["curr_hand_qvel"] = data.qvel[:-6]
        obs_dict["targ_hand_qpos"] = curr_ref.robot
        obs_dict["targ_hand_qvel"] = (
            jp.array([0]) if curr_ref.robot_vel is None else curr_ref.robot_vel
        )

        # Object and wrist errors
        obs_dict["curr_obj_com"] = data.xipos[self.object_bid]
        obs_dict["curr_obj_rot"] = mat2quat(
            jp.reshape(data.ximat[self.object_bid], (3, 3))
        )
        obs_dict["wrist_err"] = data.xipos[self.wrist_bid]
        obs_dict["base_error"] = obs_dict["curr_obj_com"] - obs_dict["wrist_err"]

        # Target and error calculations
        obs_dict["targ_obj_com"] = curr_ref.object[:3]
        obs_dict["targ_obj_rot"] = curr_ref.object[3:]
        obs_dict["hand_qpos_err"] = (
            obs_dict["curr_hand_qpos"] - obs_dict["targ_hand_qpos"]
        )
        obs_dict["hand_qvel_err"] = (
            jp.array([0])
            if curr_ref.robot_vel is None
            else obs_dict["curr_hand_qvel"] - obs_dict["targ_hand_qvel"]
        )
        obs_dict["obj_com_err"] = obs_dict["curr_obj_com"] - obs_dict["targ_obj_com"]
        return obs_dict

    def get_reward_dict(self, obs_dict):
        obj_com_err = jp.sqrt(
            self.norm2(obs_dict["targ_obj_com"] - obs_dict["curr_obj_com"])
        )
        obj_rot_err = (
            self.rotation_distance(obs_dict["curr_obj_rot"], obs_dict["targ_obj_rot"])
            / jp.pi
        )
        obj_reward = jp.exp(-self.obj_err_scale * (obj_com_err + 0.1 * obj_rot_err))

        lift_bonus = (obs_dict["targ_obj_com"][2] >= self._lift_z) & (
            obs_dict["curr_obj_com"][2] >= self._lift_z
        )

        qpos_reward = jp.exp(
            -self.qpos_err_scale * self.norm2(obs_dict["hand_qpos_err"])
        )
        qvel_reward = (
            jp.array([0])
            if obs_dict["hand_qvel_err"] is None
            else jp.exp(-self.qvel_err_scale * self.norm2(obs_dict["hand_qvel_err"]))
        )

        pose_reward = self.qpos_reward_weight * qpos_reward
        vel_reward = self.qvel_reward_weight * qvel_reward

        base_error = jp.sqrt(self.norm2(obs_dict["base_error"]))
        base_reward = jp.exp(-self.base_err_scale * base_error)

        rwd_dict = collections.OrderedDict(
            (
                ("pose", float(pose_reward + vel_reward)),
                ("object", float(obj_reward + base_reward)),
                ("bonus", self.lift_bonus_mag * float(lift_bonus)),
                ("penalty", float(self.check_termination(obs_dict))),
                ("sparse", 0),
                ("solved", 0),
                ("done", self.check_termination(obs_dict)),
            )
        )
        rwd_dict["dense"] = jp.sum(
            jp.array(
                [self.rwd_keys_wt[key] * rwd_dict[key] for key in self.rwd_keys_wt]
            )
        )
        return rwd_dict

    def rotation_distance(self, q1, q2):
        q1 = euler2quat(q1) if q1.ndim == 1 else q1
        q2 = euler2quat(q2) if q2.ndim == 1 else q2
        return jp.abs(quatDiff2Vel(q2, q1, 1)[0])

    def check_termination(self, obs_dict):
        obj_term = self.TermObj and jp.where(
            self.norm2(obs_dict["obj_com_err"]) >= self.obj_fail_thresh**2, 1, 0
        )
        qpos_term = self.TermPose and jp.where(
            self.norm2(obs_dict["hand_qpos_err"]) >= self.qpos_fail_thresh, 1, 0
        )
        base_term = self.TermObj and jp.where(
            self.norm2(obs_dict["base_error"]) >= self.base_fail_thresh**2, 1, 0
        )
        return obj_term | qpos_term | base_term

    def norm2(self, x):
        return jp.sum(x**2)

    def reset(self, rng):
        """Resets the environment to initial state."""
        qpos = self.q_pos_init.copy()
        qvel = self.q_vel_init.copy()

        # Initialize fatigue model if applicable
        if self.fatigue_model:
            self.fatigue_model.reset()

        state = State(
            pipeline_state=None,
            obs=self._get_obs(),
            reward=jp.zeros(()),
            done=jp.zeros(()),
            info={"rng": rng},
        )
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        # Normalize and process actions if muscles are involved
        muscle_a = action.copy()
        if self.muscle_condition == "fatigue" and self.fatigue_model:
            muscle_a, _, _ = self.fatigue_model.compute_act(muscle_a)

        # Unnormalize actions for the control range
        action = self.unnorm_action(action)

        # Perturb and apply dynamics through JAX-compatible step function
        data = perturbed_pipeline_step(
            self.sys,
            state.pipeline_state,
            action,
            xfrc_applied=jp.zeros((self.sys.nbody, 6)),
            frames=self.frame_skip,
        )

        # Compute observations and rewards
        obs = self._get_obs(data)
        reward, terminated = self.compute_reward(data)

        # Update state information
        new_state = state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=terminated
        )
        return new_state

    def _get_obs(self, data) -> jp.ndarray:
        """Return observations based on current model state."""
        return jp.concatenate([data.qpos, data.qvel])

    def compute_reward(self, data):
        """Computes reward based on the state of the model."""
        # Example reward for standing upright
        standing = tolerance(
            data.qpos[2], bounds=(self.q_pos_init[2], float("inf")), margin=0.4
        )
        reward = standing  # Placeholder reward logic
        terminated = jp.where(data.qpos[2] < 0.2, 1.0, 0.0)

        return reward, terminated

    def unnorm_action(self, action):
        """Scale normalized actions to actuator control range."""
        return (action + 1) / 2 * (self.high_action - self.low_action) + self.low_action
