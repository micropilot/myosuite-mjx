import pytest 
import unittest
import numpy as np
import jax
import jax.numpy as jp
import mujoco
from functools import partial
from jax import vmap, jit, tree_util

import cProfile
import pstats

from myosuite.envs.myo.fatigue import CumulativeFatigue as NumpyCumulativeFatigue
from myosuite.mjx.fatigue import CumulativeFatigue as JaxCumulativeFatigue

# Configure JAX to use CPU for consistent testing
jax.config.update('jax_platform_name', 'cpu')

class TestFatigue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and model"""
        cls.model = mujoco.MjModel.from_xml_path('myosuite/simhive/myo_sim/finger/myofinger_v0.xml')
        cls.frame_skip = 5
        cls.test_act = np.array([0.5]*5, dtype=np.float32)
        cls.test_fatigue_vec = np.array([0.5]*5, dtype=np.float32)
        cls.key = jax.random.PRNGKey(0)

    def test_pytree_structure(self):
        """Test that the class works correctly as a PyTree"""
        fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip, key=self.key)
        
        # Test flattening and unflattening
        flat, treedef = tree_util.tree_flatten(fatigue)
        restored = tree_util.tree_unflatten(treedef, flat)
        
        # Check that all attributes are preserved
        np.testing.assert_allclose(restored.MA, fatigue.MA)
        np.testing.assert_allclose(restored.MR, fatigue.MR)
        np.testing.assert_allclose(restored.MF, fatigue.MF)
        self.assertEqual(restored.na, fatigue.na)
        
        # Test that the restored object works correctly
        MA1, _, _ = fatigue.compute_act(self.test_act)
        MA2, _, _ = restored.compute_act(self.test_act)
        np.testing.assert_allclose(MA1, MA2)

    def test_jit_compilation(self):
        """Test that methods can be JIT compiled"""
        fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip, key=self.key)
        
        # Test compute_act
        jitted_compute = jax.jit(lambda f, a: f.compute_act(a))
        MA1, MR1, MF1 = fatigue.compute_act(self.test_act)
        MA2, MR2, MF2 = jitted_compute(fatigue, self.test_act)
        
        np.testing.assert_allclose(MA1, MA2)
        np.testing.assert_allclose(MR1, MR2)
        np.testing.assert_allclose(MF1, MF2)
        
        # Test get_effort
        jitted_effort = jax.jit(lambda f: f.get_effort())
        effort1 = fatigue.get_effort()
        effort2 = jitted_effort(fatigue)
        
        np.testing.assert_allclose(effort1, effort2)

    def test_vmap_compatibility(self):
        """Test that the class works with vmap"""
        profiler = cProfile.Profile()
        profiler.enable()    
        batch_size = 2  # Reduced batch size
        keys = jax.random.split(self.key, batch_size)
        
        # Create batch of actions
        batch_acts = jp.stack([self.test_act] * batch_size)
        
        #Define batch computation without JIT
        @jax.jit
        def batch_compute(keys, acts):
            def single_compute(key, act):
                fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip, key=key)
                return fatigue.compute_act(act)
            return vmap(single_compute)(keys, acts)
        
        # Run batch computation
        batch_MA, batch_MR, batch_MF = batch_compute(keys, batch_acts)
        
    
        # # Verify shapes
        # self.assertEqual(batch_MA.shape, (batch_size, 5))
        # self.assertEqual(batch_MR.shape, (batch_size, 5))
        # self.assertEqual(batch_MF.shape, (batch_size, 5))
        
        # Verify against sequential computation
        # for i in range(batch_size):
        #     fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip, key=keys[i])
        #     MA, MR, MF = fatigue.compute_act(batch_acts[i])
            
            # np.testing.assert_allclose(MA, batch_MA[i])
            # np.testing.assert_allclose(MR, batch_MR[i])
            # np.testing.assert_allclose(MF, batch_MF[i])

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)

    # def test_numpy_compatibility(self):
    #     """Test compatibility with numpy implementation"""
    #     numpy_fatigue = NumpyCumulativeFatigue(self.model, frame_skip=self.frame_skip)
    #     jax_fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip, key=self.key)
        
    #     # Test compute_act
    #     numpy_MA, numpy_MR, numpy_MF = numpy_fatigue.compute_act(self.test_act)
    #     jax_MA, jax_MR, jax_MF = jax_fatigue.compute_act(self.test_act)
        
    #     np.testing.assert_allclose(numpy_MA, np.array(jax_MA), rtol=1e-5)
    #     np.testing.assert_allclose(numpy_MR, np.array(jax_MR), rtol=1e-5)
    #     np.testing.assert_allclose(numpy_MF, np.array(jax_MF), rtol=1e-5)
        
    #     # Test effort calculation
    #     numpy_effort = numpy_fatigue.get_effort()
    #     jax_effort = float(jax_fatigue.get_effort())
        
    #     np.testing.assert_allclose(numpy_effort, jax_effort, rtol=1e-5)

    # def test_state_updates(self):
    #     """Test that state updates work correctly through PyTree transformations"""
    #     fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip, key=self.key)
        
    #     # Define a sequence of actions
    #     actions = [
    #         np.array([0.1]*5, dtype=np.float32),
    #         np.array([0.5]*5, dtype=np.float32),
    #         np.array([0.9]*5, dtype=np.float32)
    #     ]
        
    #     # Run sequence through JIT
    #     @jax.jit
    #     def run_sequence(f, acts):
    #         for act in acts:
    #             f.compute_act(act)
    #         return f
        
    #     updated_fatigue = run_sequence(fatigue, actions)
        
    #     # Verify against sequential computation
    #     reference_fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip, key=self.key)
    #     for act in actions:
    #         reference_fatigue.compute_act(act)
        
    #     np.testing.assert_allclose(updated_fatigue.MA, reference_fatigue.MA)
    #     np.testing.assert_allclose(updated_fatigue.MR, reference_fatigue.MR)
    #     np.testing.assert_allclose(updated_fatigue.MF, reference_fatigue.MF)

if __name__ == '__main__':
    unittest.main()
