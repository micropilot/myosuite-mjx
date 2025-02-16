import jax
from jax import numpy as jp
import jax.numpy as jnp

import mujoco
from mujoco import mjx

from brax.base import State
from brax.envs.base import PipelineEnv, State
from brax.training.agents.ppo import train as ppo
from brax.io import mjcf, model

from myosuite.mjx.reference_motion import ReferenceMotion
from myosuite.mjx.quat_math import euler2quat, mat2quat, quat2euler, quatDiff2Vel


class MyoHandCubeLiftEnv(PipelineEnv):
    def __init__(self, **kwargs):
        path = "myosuite/mjx/myo/assets/hand/myohand_cubesmall.xml"
        mj_model = mujoco.MjModel.from_xml_path(path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self.reward_weight_dict = {
            "pose": 1.0,
            "object": 1.0,
            "bonus": 1.0,
            "penalty": -2,
        }

        reference_data = "myosuite/envs/myo/myodm/data/MyoHand_cubesmall_lift.npz"
        self.ref = ReferenceMotion(reference_data, motion_extrapolation=0)
        self.motion_start_time = 0  # Define default value
        self.target_sid = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SITE, "target"
        )

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
        self.TermObj = True

        # TERMINATIONS FOR MIMIC
        self.qpos_fail_thresh = 0.75
        self.TermPose = False
        ##########################################

        self.object_bid = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "cubesmall"
        )
        self.wrist_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "lunate")

        # turn off the body skeleton rendering
        self.sim.model.geom_rgba[
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "body"), 3
        ] = 0.0

        self._lift_z = self.sys.xipos[self.object_bid][2] + self.lift_bonus_thresh

        # Adjust horizon if not motion_extrapolation
        if motion_extrapolation == False:
            self.spec.max_episode_steps = self.ref.horizon  # doesn't work always. WIP

        # Adjust init as per the specified key
        robot_init, object_init = self.ref.get_init()
        if robot_init is not None:
            self.init_qpos[: self.ref.robot_dim] = robot_init
        if object_init is not None:
            self.init_qpos[self.ref.robot_dim : self.ref.robot_dim + 3] = object_init[
                :3
            ]
            self.init_qpos[-3:] = quat2euler(object_init[3:])

        # hack because in the super()._setup the initial posture is set to the average qpos and when a step is called, it ends in a `done` state
        self.initialized_pos = True
        # if self.sim.model.nkey>0:
        # self.init_qpos[:] = self.sim.model.key_qpos[0,:]

    def rotation_distance(self, q1, q2, euler=True):
        if euler:
            q1 = euler2quat(q1)  # Assuming euler2quat is compatible with JAX
            q2 = euler2quat(
                q2
            )  # Ensure euler2quat is JAX-compatible (returns jnp.array)

        # Assuming quatDiff2Vel returns a jax.numpy array, ensuring compatibility
        return jnp.abs(quatDiff2Vel(q2, q1, 1)[0])

    def update_reference_insim(self, curr_ref):
        if curr_ref.object is not None:
            # Assuming `self.sim.model.site_pos` and `self.sim_obsd.model.site_pos` are mutable jax arrays
            self.sim.model.site_pos = self.sim.model.site_pos.at[self.target_sid].set(
                curr_ref.object[:3]
            )
            self.sim_obsd.model.site_pos = self.sim_obsd.model.site_pos.at[
                self.target_sid
            ].set(curr_ref.object[:3])
            self.sim.forward()  # Make sure sim.forward is compatible with JAX (i.e., side-effects do not conflict with JAX tracing)

    def norm2(self, x):
        # JAX-compatible version of np.sum(np.square(x))
        return jnp.sum(jnp.square(x))

    def reset(self, rng: jp.ndarray) -> State:
        qpos = self.sys.qpos0
        qvel = self.sys.nv

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data)

        reward, done, zero = jp.zeros(3)

        metrics = {
            "pose": zero,
            "bonus": zero,
            "penalty": zero,
            "act_reg": zero,
            "sparse": zero,
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        obs = self._get_obs(data)

        reward = jp.zeros(1)
        done = jp.bool_(False)

        # Return updated state
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:
        """Returns the observation."""
        return jp.concatenate([data.qpos, data.qvel])


env = MyoHandCubeLiftEnv()
