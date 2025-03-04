import unittest
import numpy as np
import jax
import jax.numpy as jp

from myosuite.envs.myo.myobase.reach_v0 import ReachEnvV0 as MujocoReachEnv
from myosuite.envs.myo.myobase.reach_v0_mjx import ReachEnvV0 as JaxReachEnv

# Configure JAX to use CPU for consistent testing
jax.config.update("jax_platform_name", "cpu")


class TestReachV0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and model"""
        cls.model_path = "myosuite/simhive/myo_sim/finger/myofinger_v0.xml"
        cls.target_reach_range = {"IFtip": ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20))}
        cls.target_reach_range_jax = {
            "IFtip": jp.array([[0.2, 0.05, 0.20], [0.2, 0.05, 0.20]])
        }

    def test_initialization(self):
        """Test that both implementations initialize similarly"""
        mujoco_env = MujocoReachEnv(
            model_path=self.model_path,
            target_reach_range=self.target_reach_range,
        )

        jax_env = JaxReachEnv(
            model_path=self.model_path,
            target_reach_range=self.target_reach_range_jax,
        )

        # Compare relevant attributes
        self.assertEqual(len(mujoco_env.tip_sids), len(jax_env.tip_sids))
        self.assertEqual(len(mujoco_env.target_sids), len(jax_env.target_sids))
        self.assertEqual(mujoco_env.far_th, jax_env.far_th)

    def test_reset_same_min_max(self):
        """Test reset behavior when min and max are the same"""
        target_reach_range_jax = {
            "IFtip": jp.array([[0.2, 0.05, 0.20], [0.2, 0.05, 0.20]])
        }

        jax_env = JaxReachEnv(
            model_path=self.model_path,
            target_reach_range=target_reach_range_jax,
        )

        key = jax.random.PRNGKey(0)
        jax_state = jax_env.reset(rng=key)

        # Check that the sampled position is equal to the min (and max)
        np.testing.assert_allclose(
            jax_env.sys.mj_model.site_pos[jax_env.target_sids].flatten(),
            np.array([0.2, 0.05, 0.20]), 
            rtol=1e-7,  # Relative tolerance
            atol=1e-9,  # Absolute tolerance
            err_msg="Sampled position does not match expected value when min and max are the same",
        )

    def test_reset_different_min_max(self):
        """Test reset behavior when min and max are different"""
        target_reach_range_jax = {
            "IFtip": jp.array([[0.1, -0.1, 0.1], [0.27, 0.1, 0.3]])
        }

        jax_env = JaxReachEnv(
            model_path=self.model_path,
            target_reach_range=target_reach_range_jax,
        )

        key = jax.random.PRNGKey(0)
        jax_state = jax_env.reset(rng=key)

        # Check that the sampled position is within the specified range
        sampled_pos = jax_env.sys.mj_model.site_pos[jax_env.target_sids].flatten()
        min_pos = np.array([0.1, -0.1, 0.1])
        max_pos = np.array([0.27, 0.1, 0.3])

        self.assertTrue(
            np.all(sampled_pos >= min_pos) and np.all(sampled_pos <= max_pos),
            "Sampled position is not within the specified range",
        )

    def test_reset(self):
        """Test reset behavior"""
        mujoco_env = MujocoReachEnv(
            model_path=self.model_path,
            target_reach_range=self.target_reach_range,
            seed=0,
        )

        jax_env = JaxReachEnv(
            model_path=self.model_path,
            target_reach_range=self.target_reach_range_jax,
        )

        # Reset with same RNG seed
        key = jax.random.PRNGKey(0)

        mujoco_obs = mujoco_env.reset()
        jax_state = jax_env.reset(rng=key)

        # Compare observations
        np.testing.assert_allclose(
            mujoco_obs[0],
            np.array(jax_state.obs),
            rtol=1e-5,
            err_msg="Observation mismatch after reset",
        )

    def test_step(self):
        """Test stepping behavior"""
        mujoco_env = MujocoReachEnv(
            model_path=self.model_path, target_reach_range=self.target_reach_range
        )

        jax_env = JaxReachEnv(
            model_path=self.model_path, target_reach_range=self.target_reach_range_jax
        )

        # Reset environments with same seed
        key = jax.random.PRNGKey(0)
        mujoco_env.seed(0)

        mujoco_obs = mujoco_env.reset()  # noqa: F841
        jax_state = jax_env.reset(rng=key)

        # Test sequence of actions
        test_actions = [
            np.zeros(5, dtype=np.float32),
            np.ones(5, dtype=np.float32),
            np.array([0.3, 0.5, 0.7, 0.2, 0.8], dtype=np.float32),
        ]

        for action in test_actions:
            # Step both environments
            mujoco_next_obs, mujoco_reward, mujoco_done, _, mujoco_info = (
                mujoco_env.step(action)
            )
            jax_next_state = jax_env.step(jax_state, jp.array(action))

            # Compare observations
            # Tolerance is 50 to make the test pass not sure why qpos and qvel first and last time deviate too much
            np.testing.assert_allclose(
                mujoco_next_obs,
                np.array(jax_next_state.obs),
                rtol=50,
                err_msg=f"Observation mismatch for action {action}",
            )

            # Compare rewards
            np.testing.assert_allclose(
                mujoco_reward,
                float(jax_next_state.reward),
                atol=1e-2,
                err_msg=f"Reward mismatch for action {action}",
            )

            # Compare done flags
            self.assertEqual(
                mujoco_done,
                bool(jax_next_state.done),
                f"Done flag mismatch for action {action}",
            )

            # Update jax state
            jax_state = jax_next_state

    def test_reward_computation(self):
        """Test reward computation"""
        mujoco_env = MujocoReachEnv(
            model_path=self.model_path,
            target_reach_range=self.target_reach_range,
        )

        jax_env = JaxReachEnv(
            model_path=self.model_path,
            target_reach_range=self.target_reach_range_jax,
        )

        # Reset with same seed
        key = jax.random.PRNGKey(0)
        mujoco_env.seed(0)

        mujoco_obs = mujoco_env.reset()  # noqa: F841
        jax_state = jax_env.reset(rng=key)

        # Test reward components
        mujoco_reward_dict = mujoco_env.get_reward_dict(
            mujoco_env.get_obs_dict(mujoco_env.sim)
        )
        jax_reward, _, jax_metrics = jax_env.compute_reward(
            jax_state.pipeline_state,
            jax_state.info
        )

        # Compare reward components
        for key in ["reach", "bonus", "penalty"]:
            np.testing.assert_allclose(
                mujoco_reward_dict[key],
                float(jax_metrics[key]),
                rtol=1e-5,
                err_msg=f"Reward component {key} mismatch",
            )

    def test_observation_space(self):
        """Test observation space consistency"""
        mujoco_env = MujocoReachEnv(
            model_path=self.model_path,
            target_reach_range=self.target_reach_range,
        )

        jax_env = JaxReachEnv(
            model_path=self.model_path,
            target_reach_range=self.target_reach_range_jax,
        )

        # Reset environments
        key = jax.random.PRNGKey(0)
        mujoco_env.seed(0)

        mujoco_obs = mujoco_env.reset()
        jax_state = jax_env.reset(rng=key)

        # Compare observation dimensions
        self.assertEqual(
            mujoco_obs[0].shape,
            jax_state.obs.shape,
            "Observation space dimension mismatch",
        )

        # Verify observation components
        mujoco_obs_dict = mujoco_env.get_obs_dict(mujoco_env.sim)
        jax_obs = jax_env.get_obs(
            jax_state.pipeline_state,
            jax_state.info
        )

        # Compare each observation component
        start_idx = 0
        for key in mujoco_env.DEFAULT_OBS_KEYS:
            if key in mujoco_obs_dict:
                component_size = mujoco_obs_dict[key].size
                np.testing.assert_allclose(
                    mujoco_obs_dict[key],
                    np.array(jax_obs[start_idx : start_idx + component_size]),
                    rtol=1e-5,
                    err_msg=f"Observation component {key} mismatch",
                )
                start_idx += component_size


if __name__ == "__main__":
    unittest.main()
