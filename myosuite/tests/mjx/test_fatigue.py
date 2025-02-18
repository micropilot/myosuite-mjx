import unittest
import numpy as np
import jax
import jax.numpy as jp
import mujoco

from myosuite.envs.myo.fatigue import CumulativeFatigue as NumpyCumulativeFatigue
from myosuite.mjx.fatigue import CumulativeFatigue as JaxCumulativeFatigue

# Configure JAX to use CPU for consistent testing
jax.config.update('jax_platform_name', 'cpu')

class TestFatigue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and model"""
        # Create a minimal mujoco model with muscle actuators
        cls.model = mujoco.MjModel.from_xml_path('myosuite/simhive/myo_sim/finger/myofinger_v0.xml')
        
        # Test parameters
        cls.frame_skip = 5
        cls.test_act = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        cls.test_fatigue_vec = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    def setUp(self):
        """Create fresh instances for each test"""
        self.numpy_fatigue = NumpyCumulativeFatigue(self.model, frame_skip=self.frame_skip)
        self.jax_fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip)

    def test_initialization(self):
        """Test that both implementations initialize the same way"""
        # Check dimensions
        self.assertEqual(self.jax_fatigue.na, self.numpy_fatigue.na)
        
        # Check parameters
        np.testing.assert_allclose(
            np.array(self.jax_fatigue.F),
            self.numpy_fatigue.F,
            rtol=1e-5
        )
        np.testing.assert_allclose(
            np.array(self.jax_fatigue.R),
            self.numpy_fatigue.R,
            rtol=1e-5
        )
        np.testing.assert_allclose(
            np.array(self.jax_fatigue.r),
            self.numpy_fatigue.r,
            rtol=1e-5
        )
        
        # Check initial states
        np.testing.assert_allclose(
            np.array(self.jax_fatigue.MA),
            self.numpy_fatigue.MA,
            rtol=1e-5
        )
        np.testing.assert_allclose(
            np.array(self.jax_fatigue.MR),
            self.numpy_fatigue.MR,
            rtol=1e-5
        )
        np.testing.assert_allclose(
            np.array(self.jax_fatigue.MF),
            self.numpy_fatigue.MF,
            rtol=1e-5
        )

    def test_compute_act(self):
        """Test activation computation matches between implementations"""
        # Test with different activation levels
        test_acts = [
            np.zeros(5, dtype=np.float32),  # Zero activation
            np.ones(5, dtype=np.float32),   # Full activation
            np.array([0.5]*5, dtype=np.float32),  # Mixed activation
            np.array([0.5]*5, dtype=np.float32)   # Equal activation
        ]
        
        for act in test_acts:
            # Compute activations
            numpy_MA, numpy_MR, numpy_MF = self.numpy_fatigue.compute_act(act)
            jax_MA, jax_MR, jax_MF = self.jax_fatigue.compute_act(act)
            
            # Compare results
            np.testing.assert_allclose(
                np.array(jax_MA),
                numpy_MA,
                rtol=1e-5,
                err_msg=f"MA mismatch for activation {act}"
            )
            np.testing.assert_allclose(
                np.array(jax_MR),
                numpy_MR,
                rtol=1e-5,
                err_msg=f"MR mismatch for activation {act}"
            )
            np.testing.assert_allclose(
                np.array(jax_MF),
                numpy_MF,
                rtol=1e-5,
                err_msg=f"MF mismatch for activation {act}"
            )

    def test_reset(self):
        """Test reset behavior matches between implementations"""
        # Test normal reset
        self.numpy_fatigue.reset()
        self.jax_fatigue.reset()
        
        np.testing.assert_allclose(
            np.array(self.jax_fatigue.MA),
            self.numpy_fatigue.MA,
            rtol=1e-5
        )
        
        # Test reset with fatigue vector
        self.numpy_fatigue.reset(fatigue_reset_vec=self.test_fatigue_vec)
        self.jax_fatigue.reset(fatigue_reset_vec=self.test_fatigue_vec)
        
        np.testing.assert_allclose(
            np.array(self.jax_fatigue.MF),
            self.numpy_fatigue.MF,
            rtol=1e-5
        )

    def test_get_effort(self):
        """Test effort calculation matches between implementations"""
        # Set same activation
        self.numpy_fatigue.compute_act(self.test_act)
        self.jax_fatigue.compute_act(self.test_act)
        
        # Compare effort
        np.testing.assert_allclose(
            float(self.jax_fatigue.get_effort()),
            self.numpy_fatigue.get_effort(),
            rtol=1e-5
        )

    def test_parameter_updates(self):
        """Test parameter updates behave the same"""
        # Test fatigue coefficient update
        new_F = 0.02
        self.numpy_fatigue.set_FatigueCoefficient(new_F)
        self.jax_fatigue.set_FatigueCoefficient(new_F)
        np.testing.assert_allclose(
            np.array(self.jax_fatigue.F),
            self.numpy_fatigue.F,
            rtol=1e-5
        )
        
        # Test recovery coefficient update
        new_R = 0.003
        self.numpy_fatigue.set_RecoveryCoefficient(new_R)
        self.jax_fatigue.set_RecoveryCoefficient(new_R)
        np.testing.assert_allclose(
            np.array(self.jax_fatigue.R),
            self.numpy_fatigue.R,
            rtol=1e-5
        )
        
        # Test recovery multiplier update
        new_r = 12
        self.numpy_fatigue.set_RecoveryMultiplier(new_r)
        self.jax_fatigue.set_RecoveryMultiplier(new_r)
        np.testing.assert_allclose(
            np.array(self.jax_fatigue.r),
            self.numpy_fatigue.r,
            rtol=1e-5
        )

    def test_sequential_activations(self):
        """Test behavior over sequential activations matches"""
        activation_sequence = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
            np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
            np.array([0.]*5, dtype=np.float32),
            np.array([1.]*5, dtype=np.float32),
            np.array([0.5]*5, dtype=np.float32),
            np.array([-1.0]*5, dtype=np.float32)
        ]
        
        for act in activation_sequence:
            numpy_MA, _, _ = self.numpy_fatigue.compute_act(act)
            jax_MA, _, _ = self.jax_fatigue.compute_act(act)
            
            np.testing.assert_allclose(
                np.array(jax_MA),
                numpy_MA,
                rtol=1e-5,
                err_msg=f"Sequential activation mismatch for {act}"
            )

if __name__ == '__main__':
    unittest.main()
