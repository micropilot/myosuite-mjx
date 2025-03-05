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
        model_path = "/../assets/hand/myohand_object_mjx.xml"
        object_name = "airplane"
        reference = {
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
        obs_keys = ["qp", "qv", "hand_qpos_err", "hand_qvel_err", "obj_com_err"]
        weighted_reward_keys = {
            "pose": 0.0,
            "object": 1.0,
            "bonus": 1.0,
            "penalty": -2,
        }

        cls.mujoco_env = MujocoTrackEnv(
            model_path=model_path,
            object_name=object_name,
            reference=reference,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            seed=0,
        )

        cls.jax_env = JaxTrackEnv(
            model_path=model_path,
            object_name=object_name,
            reference=reference,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
        )

    def test_initialization(self):
        """Test that both implementations initialize similarly"""
        # Compare relevant attributes
        self.assertEqual(self.mujoco_env.object_name, self.jax_env.object_name)
        self.assertEqual(self.mujoco_env.lift_bonus_thresh, self.jax_env.lift_bonus_thresh)
        self.assertEqual(self.mujoco_env.obj_err_scale, self.jax_env.obj_err_scale)

    def test_reset(self):
        """Test reset behavior"""

        # Reset with same RNG seed
        mujoco_obs = self.mujoco_env.reset()
        key = jax.random.PRNGKey(0)
        jit_reset = jax.jit(self.jax_env.reset)
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

        # Compare object com error
        # Cannot use this for testing because of randomization
        # np.testing.assert_allclose(
        #     mujoco_obs[0][128:131],
        #     np.array(jax_state.obs[128:131]),
        #     rtol=1e-5,
        #     err_msg="Object Com Err mismatch after reset",
        # )
        
        # Compare action
        np.testing.assert_allclose(
            mujoco_obs[0][131:],
            np.array(jax_state.obs[131:]),
            rtol=1e-5,
            err_msg="Action mismatch after reset",
        )

    def test_step(self):
        """Test stepping behavior"""
        # Reset environments with same seed
        key = jax.random.PRNGKey(0)
        mujoco_obs = self.mujoco_env.reset()
        jax_state = self.jax_env.reset(rng=key)

        # Test sequence of actions
        test_actions = [
            np.zeros(45, dtype=np.float32),
            np.ones(45, dtype=np.float32) * 0.1,
            np.random.uniform(-0.1, 0.1, 45).astype(np.float32),
        ]

        jit_step = jax.jit(self.jax_env.step)

        for action in test_actions:
            # Step both environments
            mujoco_obs, mujoco_reward, mujoco_done, _, mujoco_info = (
                self.mujoco_env.step(action)
            )
            jax_state = jit_step(jax_state, jp.array(action))
            # jax_state = self.jax_env.step(jax_state, jp.array(action))


            # Compare position
            position_diff_norm = np.linalg.norm(
                mujoco_obs[:35] - np.array(jax_state.obs[:35])
            )
            
            # Assert that the norm is below the threshold
            self.assertLessEqual(
                position_diff_norm,
                0.01,
                "Position mismatch after reset: Norm of difference is too large"
            )

            # Compare velocity
            velocity_diff_norm = np.linalg.norm(
                mujoco_obs[35:70] - np.array(jax_state.obs[35:70])
            )
            self.assertLessEqual(
                velocity_diff_norm,
                0.5,
                "Velocity mismatch after reset: Norm of difference is too large"
            )

            # Compare hand qpos error
            hand_qpos_err_diff_norm = np.linalg.norm(
                mujoco_obs[70:99] - np.array(jax_state.obs[70:99])
            )
            self.assertLessEqual(
                hand_qpos_err_diff_norm,
                0.01,
                "Hand Qpos Err mismatch after reset: Norm of difference is too large"
            )

            # Compare hand qvel error
            hand_qvel_err_diff_norm = np.linalg.norm(
                mujoco_obs[99:128] - np.array(jax_state.obs[99:128])
            )
            self.assertLessEqual(
                hand_qvel_err_diff_norm,
                0.5,
                "Hand Qvel Err mismatch after reset: Norm of difference is too large"
            )

            # Compare object com error
            object_com_err_diff_norm = np.linalg.norm(
                mujoco_obs[128:131] - np.array(jax_state.obs[128:131])
            )
            self.assertLessEqual(
                object_com_err_diff_norm,
                0.5,
                "Object Com Err mismatch after reset: Norm of difference is too large"
            )

            # Compare action
            action_diff_norm = np.linalg.norm(
                mujoco_obs[131:] - np.array(jax_state.obs[131:])
            )
            self.assertLessEqual(
                action_diff_norm,
                0.01,
                "Action mismatch after reset: Norm of difference is too large"
            )

            # Compare rewards
            np.testing.assert_allclose(
                mujoco_reward,
                float(jax_state.reward),
                rtol=1e-2,
                err_msg=f"Reward mismatch for action {action}",
            )

            # Compare done flags
            self.assertEqual(
                mujoco_done,
                bool(jax_state.done),
                f"Done flag mismatch for action {action}",
            )

    def test_reward_computation(self):
        """Test reward computation"""
        # Reset with same seed
        key = jax.random.PRNGKey(0)

        mujoco_obs =self.mujoco_env.reset()
        jax_state = self.jax_env.reset(rng=key)

        # Test reward components
        mujoco_reward_dict = self.mujoco_env.get_reward_dict(
            self.mujoco_env.get_obs_dict(self.mujoco_env.sim)
        )
        jax_reward, _, jax_metrics = self.jax_env.compute_reward(
            jax_state.pipeline_state,
            jax_state.info
        )

        # Compare reward components
        for key in ["pose", "object", "bonus", "penalty"]:
            assert (mujoco_reward_dict[key] - float(jax_metrics[key])) < 0.01


if __name__ == "__main__":
    unittest.main() 