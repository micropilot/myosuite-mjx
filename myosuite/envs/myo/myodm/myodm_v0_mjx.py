import os
import time
import jax
import jax.numpy as jp
import mujoco
from brax import base
from brax.envs.base import State

from myosuite.envs.myo.base_v0_mjx import BaseV0

from myosuite.logger.reference_motion_jax import ReferenceMotion
from myosuite.utils.quat_math_jax import euler2quat, quat2euler, quatDiff2Vel, mat2quat


class TrackEnv(BaseV0):

    def __init__(
        self, model_path: str, object_name: str, frame_skip: int = 10, **kwargs
    ):
        # Load model and setup simulation
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

        super().__init__(
            model_path=processed_model_path,
            frame_skip=frame_skip,
        )
        os.remove(processed_model_path)

        self._setup(**kwargs)

    def _setup(
        self,
        reference: dict,
        motion_start_time: float = 0,
        motion_extrapolation: bool = True,
        obs_keys: list = ["qp", "qv", "hand_qpos_err", "hand_qvel_err", "obj_com_err"],
        weighted_reward_keys: dict = {
            "pose": 1.0,
            "object": 1.0,
            "bonus": 1.0,
            "penalty": -2,
        },
        terminate_obj_fail: bool = True,
        terminate_pose_fail: bool = False,
        **kwargs
    ):

        self.ref = ReferenceMotion(
            reference_data=reference,
            motion_extrapolation=motion_extrapolation,
        )

        self.motion_start_time = motion_start_time
        self.target_sid = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, "target"
        )

        ##########################################
        self.lift_bonus_thresh = 0.02
        # PRE-GRASP
        self.obj_err_scale = 50
        self.base_err_scale = 40
        self.lift_bonus_mag = 2.5

        # DEEPMIMIC
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

        self.object_bid = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.object_name
        )
        self.wrist_bid = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, "lunate"
        )

        # Disable body skeleton rendering by setting transparency
        self.sys.mj_model.geom_rgba[self.object_bid, 3] = (
            0.0  # Make all geoms invisible
        )

        ipos = self.sys.mj_model.body_ipos[self.object_bid]
        pos = self.sys.mj_model.body_pos[self.object_bid]
        self.lift_z = (ipos + pos)[2] + self.lift_bonus_thresh

        super()._setup(
            obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, **kwargs
        )

        if not motion_extrapolation:
            self.spec.max_episode_steps = self.ref.horizon

        robot_init, object_init = self.ref.get_init()
        if robot_init is not None:
            self.init_qpos = self.init_qpos.at[: self.ref.robot_dim].set(robot_init)
        if object_init is not None:
            self.init_qpos = self.init_qpos.at[
                self.ref.robot_dim : self.ref.robot_dim + 3
            ].set(object_init[:3])
            self.init_qpos = self.init_qpos.at[-3:].set(quat2euler(object_init[3:]))

    def reset(self, rng: jax.Array = None) -> State:
        self.ref.reset()
        super().reset(rng, info={})
        key, subkey = jax.random.split(rng)

        # qpos and qvel contain both hand and object pose and vel
        reward, done, zero = jp.zeros(3)
        pipeline_state = self.pipeline_init(self.init_qpos, self.init_qvel)

        info = self.get_info(pipeline_state, {})
        obs = self.get_obs(pipeline_state, info)
        metrics = {k: jp.array(zero) for k in self.weighted_reward_keys.keys()}
        metrics["reward"] = reward

        state = State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

        return state

    def norm2(self, x):
        result = jp.sum(jp.square(x))
        return result

    def rotation_distance(self, q1, q2, euler=True):
        if euler:
            q1 = euler2quat(q1)
            q2 = euler2quat(q2)

        result = jp.abs(quatDiff2Vel(q2, q1, 1)[0])

        return result

    def compute_reward(self, pipeline_state: base.State, info: dict) -> dict:
        # get targets from reference object
        tgt_obj_com = info["targ_obj_com"].flatten()
        tgt_obj_rot = info["targ_obj_rot"].flatten()

        # get real values from physics object
        obj_com = info["curr_obj_com"].flatten()
        obj_rot = info["curr_obj_rot"].flatten()

        # calculate both object "matching"
        obj_com_err = jp.sqrt(self.norm2(tgt_obj_com - obj_com))
        obj_rot_err = self.rotation_distance(obj_rot, tgt_obj_rot, False) / jp.pi
        obj_reward = jp.exp(-self.obj_err_scale * (obj_com_err + 0.1 * obj_rot_err))

        # calculate lift bonus
        lift_bonus = jp.where(
            jp.logical_and(
                jp.greater_equal(tgt_obj_com[2], self.lift_z),
                jp.greater_equal(obj_com[2], self.lift_z),
            ),
            1.0,
            0.0,
        )

        # calculate reward terms
        qpos_reward = jp.exp(-self.qpos_err_scale * self.norm2(info["hand_qpos_err"]))
        qvel_reward = jp.where(
            info["hand_qvel_err"] is None,
            0.0,
            jp.exp(-self.qvel_err_scale * self.norm2(info["hand_qvel_err"])),
        )

        # weight and sum individual reward terms
        pose_reward = self.qpos_reward_weight * qpos_reward
        vel_reward = self.qvel_reward_weight * qvel_reward

        base_error = jp.sqrt(self.norm2(info["base_error"]))
        base_reward = jp.exp(-self.base_err_scale * base_error)

        obj_term = jp.where(
            self.TermObj & (self.norm2(info["obj_com_err"]) >= self.obj_fail_thresh**2),
            1.0,
            0.0,
        )
        base_term = jp.where(
            self.TermObj & (self.norm2(info["base_error"]) >= self.base_fail_thresh**2),
            1.0,
            0.0,
        )
        qpos_term = jp.where(
            self.TermPose
            & (self.norm2(info["hand_qpos_err"]) >= self.qpos_fail_thresh),
            1.0,
            0.0,
        )

        done = jp.where(
            jp.logical_or(jp.logical_or(obj_term, qpos_term), base_term), 1.0, 0.0
        )

        metrics = {
            "pose": pose_reward + vel_reward,
            "object": obj_reward + base_reward,
            "bonus": float(self.lift_bonus_mag) * lift_bonus,
            "penalty": done,
        }

        reward = jp.sum(
            jp.array([metrics[k] * v for k, v in self.weighted_reward_keys.items()])
        )

        return reward, done, metrics

    def get_obs(self, pipeline_state: base.State, info: dict) -> jp.ndarray:
        position = pipeline_state.qpos
        velocity = pipeline_state.qvel
        hand_qpos_err = info["hand_qpos_err"]
        hand_qvel_err = info["hand_qvel_err"]
        obj_com_err = info["obj_com_err"]

        if self.sys.na > 0:
            obs = jp.concatenate(
                [
                    position,
                    velocity,
                    hand_qpos_err.flatten(),
                    hand_qvel_err.flatten(),
                    obj_com_err.flatten(),
                    pipeline_state.act,
                ]
            )
        else:
            obs = jp.concatenate(
                [
                    position,
                    velocity,
                    hand_qpos_err.flatten(),
                    hand_qvel_err.flatten(),
                    obj_com_err.flatten(),
                ]
            )
        return obs

    def get_info(self, pipeline_state: base.State, info: dict) -> dict:
        curr_ref = self.ref.get_reference(pipeline_state.time + self.motion_start_time)
        # update reference in sim
        # qpos = pipeline_state.qpos
        # jax.debug.print("MJX qpos {}", qpos)
        # qpos = qpos.at[:3].set(curr_ref.object[:3])
        # data = self.pipeline_init(qpos, pipeline_state.qvel)
        # jax.debug.print("MJX qpos after update{}", data.qpos)

        # get current hand pose + vel
        info["curr_hand_qpos"] = pipeline_state.q[:-6].copy()
        info["curr_hand_qvel"] = pipeline_state.qd[:-6].copy()

        # get targets from reference object
        info["targ_hand_qpos"] = curr_ref.robot
        info["targ_hand_qvel"] = (
            jp.array([0]) if curr_ref.robot_vel is None else curr_ref.robot_vel
        )

        # get real values from physics object
        # info["curr_obj_com"] = pipeline_state.xipos[self.object_bid].copy()
        info["curr_obj_com"] = pipeline_state.qpos[self.ref.robot_dim : self.ref.robot_dim + 3]
        info["curr_obj_rot"] = mat2quat(pipeline_state.ximat[self.object_bid])

        info["wrist_err"] = pipeline_state.xipos[self.wrist_bid].copy()

        info["base_error"] = info["curr_obj_com"] - info["wrist_err"]

        info["targ_obj_com"] = curr_ref.object[:3]
        info["targ_obj_rot"] = curr_ref.object[3:]

        # Errors
        info["hand_qpos_err"] = info["curr_hand_qpos"] - info["targ_hand_qpos"]
        info["hand_qvel_err"] = (
            jp.array([0])
            if curr_ref.robot_vel is None
            else (info["curr_hand_qvel"] - info["targ_hand_qvel"])
        )

        info["obj_com_err"] = info["curr_obj_com"] - info["targ_obj_com"]

        return info


# jax.config.update("jax_platform_name", "cpu")

# model_path = "/../assets/hand/myohand_object_mjx.xml"
# object_name = "airplane"
# reference = {
#             "time": (0.0, 4.0),
#             "robot": jp.zeros((2, 29)),
#             "robot_vel": jp.zeros((2, 29)),
#             "object_init": jp.array((0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0)),
#             "object": jp.array(
#                 [
#                     [-0.2, -0.2, 0.1, 1.0, 0.0, 0.0, -1.0],
#                     [0.2, 0.2, 0.1, 1.0, 0.0, 0.0, 1.0],
#                 ]
#             ),
#         }
# obs_keys = ["qp", "qv", "hand_qpos_err", "hand_qvel_err", "obj_com_err"]
# weighted_reward_keys = {
#             "pose": 0.0,
#             "object": 1.0,
#             "bonus": 1.0,
#             "penalty": -2,
#         }

# jax_env = TrackEnv(
#             model_path=model_path,
#             object_name=object_name,
#             reference=reference,
#             obs_keys=obs_keys,
#             weighted_reward_keys=weighted_reward_keys,
#         )

# key = jax.random.PRNGKey(0)
# start_time = time.time()
# jax_state = jax_env.reset(rng=key)
# end_time = time.time()
# print(f"Time taken: {end_time - start_time} seconds")
