import unittest
import numpy as np
import jax
import jax.numpy as jp
import time 

from myosuite.envs.myo.myodm.myodm_v0 import TrackEnv as MujocoTrackEnv
from myosuite.envs.myo.myodm.myodm_v0_mjx import TrackEnv as JaxTrackEnv

# Configure JAX to use CPU for consistent testing
jax.config.update("jax_platform_name", "cpu")


class TestTrackEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and model"""
        cls.model_path = "/../assets/hand/myohand_object_mjx.xml"
        cls.object_name = "airplane"
        cls.reference = {
            "time": (0.0, 4.0),
            "robot": np.zeros((2, 29)),
            "robot_vel": np.zeros((2, 29)),
            "object_init": np.array((0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0)),
            "object": np.array(
                [
                    [-0.2, -0.2, 0.1, 1.0, 0.0, 0.0, -1.0],
                    [0.2, 0.2, 0.1, 1.0, 0.0, 0.0, 1.0],
                ]
            ),
        }
        cls.obs_keys = ["qp", "qv", "hand_qpos_err", "hand_qvel_err", "obj_com_err"]
        cls.weighted_reward_keys = {
            "pose": 0.0,
            "object": 1.0,
            "bonus": 1.0,
            "penalty": -2,
        }

    def test_initialization(self):
        """Test that both implementations initialize similarly"""
        mujoco_env = MujocoTrackEnv(
            model_path=self.model_path,
            object_name=self.object_name,
            reference=self.reference,
            obs_keys=self.obs_keys,
            weighted_reward_keys=self.weighted_reward_keys,
        )

        jax_env = JaxTrackEnv(
            model_path=self.model_path,
            object_name=self.object_name,
            reference=self.reference,
            obs_keys=self.obs_keys,
            weighted_reward_keys=self.weighted_reward_keys,
        )

        # Compare relevant attributes
        self.assertEqual(mujoco_env.frame_skip, jax_env.frame_skip)
        self.assertEqual(mujoco_env.object_name, jax_env.object_name)
        self.assertEqual(mujoco_env.lift_bonus_thresh, jax_env.lift_bonus_thresh)
        self.assertEqual(mujoco_env.obj_err_scale, jax_env.obj_err_scale)

    def test_reset(self):
        """Test reset behavior"""
        mujoco_env = MujocoTrackEnv(
            model_path=self.model_path,
            object_name=self.object_name,
            reference=self.reference,
            obs_keys=self.obs_keys,
            weighted_reward_keys=self.weighted_reward_keys,
        )
        print ("Is it doing it before this?")

        jax_env = JaxTrackEnv(
            model_path=self.model_path,
            object_name=self.object_name,
            reference=self.reference,
            obs_keys=self.obs_keys,
            weighted_reward_keys=self.weighted_reward_keys,
        )

        # Reset with same RNG seed
        key = jax.random.PRNGKey(0)
        mujoco_env.seed(0)

        jit_reset = jax.jit(jax_env.reset)

        mujoco_obs = mujoco_env.reset()
        jax_state = jit_reset(rng=key)

        # Compare position
        np.testing.assert_allclose(
            mujoco_obs[0][:35],
            np.array(jax_state.obs[:35]),
            rtol=1e-5,
            err_msg="Position mismatch after reset",
        )

        # Compare velocity
        np.testing.assert_allclose(
            mujoco_obs[0][35:70],
            np.array(jax_state.obs[35:70]),
            rtol=1e-5,
            err_msg="Velocity mismatch after reset",
        )

        # Compare hand qpos error
        np.testing.assert_allclose(
            mujoco_obs[0][70:99],
            np.array(jax_state.obs[70:99]),
            rtol=1e-5,
            err_msg="Hand Qpos Err mismatch after reset",
        )

        # Compare hand qvel error
        np.testing.assert_allclose(
            mujoco_obs[0][99:128],
            np.array(jax_state.obs[99:128]),
            rtol=1e-5,
            err_msg="Hand Qvel Err mismatch after reset",
        )

    # def test_step(self):
    #     """Test stepping behavior"""
    #     mujoco_env = MujocoTrackEnv(
    #         model_path=self.model_path,
    #         object_name=self.object_name,
    #         reference=self.reference,
    #     )

    #     jax_env = JaxTrackEnv(
    #         model_path=self.model_path,
    #         object_name=self.object_name,
    #         reference=self.reference,
    #     )

    #     # Reset environments with same seed
    #     key = jax.random.PRNGKey(0)
    #     mujoco_env.seed(0)

    #     mujoco_obs = mujoco_env.reset()
    #     jax_state = jax_env.reset(rng=key)

    #     # Test sequence of actions
    #     test_actions = [
    #         np.zeros(30, dtype=np.float32),
    #         np.ones(30, dtype=np.float32) * 0.1,
    #         np.random.uniform(-0.1, 0.1, 30).astype(np.float32),
    #     ]

    #     for action in test_actions:
    #         # Step both environments
    #         mujoco_next_obs, mujoco_reward, mujoco_done, _, mujoco_info = (
    #             mujoco_env.step(action)
    #         )
    #         jax_next_state = jax_env.step(jax_state, jp.array(action))

    #         # Compare observations
    #         np.testing.assert_allclose(
    #             mujoco_next_obs,
    #             np.array(jax_next_state.obs),
    #             rtol=1e-4,
    #             err_msg=f"Observation mismatch for action {action}",
    #         )

    #         # Compare rewards
    #         np.testing.assert_allclose(
    #             mujoco_reward,
    #             float(jax_next_state.reward),
    #             rtol=1e-4,
    #             err_msg=f"Reward mismatch for action {action}",
    #         )

    #         # Compare done flags
    #         self.assertEqual(
    #             mujoco_done,
    #             bool(jax_next_state.done),
    #             f"Done flag mismatch for action {action}",
    #         )

    #         # Update jax state
    #         jax_state = jax_next_state

    # def test_reward_computation(self):
    #     """Test reward computation"""
    #     mujoco_env = MujocoTrackEnv(
    #         model_path=self.model_path,
    #         object_name=self.object_name,
    #         reference=self.reference,
    #     )

    #     jax_env = JaxTrackEnv(
    #         model_path=self.model_path,
    #         object_name=self.object_name,
    #         reference=self.reference,
    #     )

    #     # Reset with same seed
    #     key = jax.random.PRNGKey(0)
    #     mujoco_env.seed(0)

    #     mujoco_obs = mujoco_env.reset()
    #     jax_state = jax_env.reset(rng=key)

    #     # Test reward components
    #     mujoco_reward_dict = mujoco_env.get_reward_dict(
    #         mujoco_env.get_obs_dict(mujoco_env.sim)
    #     )
    #     jax_reward, _, jax_metrics = jax_env.compute_reward(
    #         jax_state.pipeline_state,
    #         jax_state.info
    #     )

    #     # Compare reward components
    #     for key in ["pose", "object", "bonus", "penalty"]:
    #         np.testing.assert_allclose(
    #             mujoco_reward_dict[key],
    #             float(jax_metrics[key]),
    #             rtol=1e-4,
    #             err_msg=f"Reward component {key} mismatch",
    #         )

    # def test_observation_space(self):
    #     """Test observation space consistency"""
    #     mujoco_env = MujocoTrackEnv(
    #         model_path=self.model_path,
    #         object_name=self.object_name,
    #         reference=self.reference,
    #     )

    #     jax_env = JaxTrackEnv(
    #         model_path=self.model_path,
    #         object_name=self.object_name,
    #         reference=self.reference,
    #     )

    #     # Reset environments
    #     key = jax.random.PRNGKey(0)
    #     mujoco_env.seed(0)

    #     mujoco_obs = mujoco_env.reset()
    #     jax_state = jax_env.reset(rng=key)

    #     # Compare observation dimensions
    #     self.assertEqual(
    #         mujoco_obs.shape,
    #         jax_state.obs.shape,
    #         "Observation space dimension mismatch",
    #     )

    #     # Verify observation components
    #     mujoco_obs_dict = mujoco_env.get_obs_dict(mujoco_env.sim)
    #     jax_obs = jax_env.get_obs(
    #         jax_state.pipeline_state, 
    #         jp.zeros(jax_env.sys.act_size()),
    #         jax_state.info
    #     )

    #     # Compare each observation component
    #     start_idx = 0
    #     for key in mujoco_env.DEFAULT_OBS_KEYS:
    #         if key in mujoco_obs_dict:
    #             component_size = mujoco_obs_dict[key].size
    #             np.testing.assert_allclose(
    #                 mujoco_obs_dict[key],
    #                 np.array(jax_obs[start_idx : start_idx + component_size]),
    #                 rtol=1e-4,
    #                 err_msg=f"Observation component {key} mismatch",
    #             )
    #             start_idx += component_size


if __name__ == "__main__":
    unittest.main() 