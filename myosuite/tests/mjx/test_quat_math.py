import unittest
import numpy as np
import jax
import jax.numpy as jp

# Configure JAX to use CPU to avoid potential GPU-related issues
jax.config.update('jax_platform_name', 'cpu')

from myosuite.utils.quat_math import mulQuat as np_mulQuat, negQuat as np_negQuat, quat2Vel as np_quat2Vel, diffQuat as np_diffQuat
from myosuite.mjx.quat_math import mulQuat as jax_mulQuat, negQuat as jax_negQuat, quat2Vel as jax_quat2Vel, diffQuat as jax_diffQuat


class TestQuatMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize JAX"""
        # Ensure JAX is initialized on CPU
        cls.device = jax.devices('cpu')[0]
        
    def setUp(self):
        # Define some test quaternions with explicit dtype
        self.test_cases = [
            # Identity quaternion
            (np.array([1., 0., 0., 0.], dtype=np.float32), 
             np.array([0.7071, 0.7071, 0., 0.], dtype=np.float32)),
            # 90-degree rotations around different axes
            (np.array([0.7071, 0.7071, 0., 0.], dtype=np.float32), 
             np.array([0.7071, 0., 0.7071, 0.], dtype=np.float32)),
            # Arbitrary quaternions
            (np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), 
             np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)),
            # Zero rotation
            (np.array([1., 0., 0., 0.], dtype=np.float32), 
             np.array([1., 0., 0., 0.], dtype=np.float32)),
        ]
        
        # Additional test cases specifically for negQuat
        self.neg_test_cases = [
            np.array([1., 0., 0., 0.], dtype=np.float32),  # Identity
            np.array([0., 1., 0., 0.], dtype=np.float32),  # Pure i
            np.array([0., 0., 1., 0.], dtype=np.float32),  # Pure j
            np.array([0., 0., 0., 1.], dtype=np.float32),  # Pure k
            np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),  # Equal components
            np.array([0.7071, 0.7071, 0., 0.], dtype=np.float32),  # 90-degree rotation
        ]

        # Add test cases for quat2Vel
        self.vel_test_cases = [
            # No rotation (identity quaternion)
            np.array([1., 0., 0., 0.], dtype=np.float32),
            
            # 90-degree rotation around x-axis
            np.array([0.7071067811865476, 0.7071067811865476, 0., 0.], dtype=np.float32),
            
            # 90-degree rotation around y-axis
            np.array([0.7071067811865476, 0., 0.7071067811865476, 0.], dtype=np.float32),
            
            # 90-degree rotation around z-axis
            np.array([0.7071067811865476, 0., 0., 0.7071067811865476], dtype=np.float32),
            
            # 45-degree rotation around x-axis
            np.array([0.9238795325112867, 0.3826834323650898, 0., 0.], dtype=np.float32),
            
            # Arbitrary rotation
            np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        ]
        
        # Expected results for dt=1.0
        self.vel_expected_results = [
            # For identity quaternion: no rotation
            (0.0, np.array([0., 0., 0.], dtype=np.float32)),
            
            # For 90-degree rotation around x: pi/2 speed, [1,0,0] axis
            (np.pi/2, np.array([1., 0., 0.], dtype=np.float32)),
            
            # For 90-degree rotation around y: pi/2 speed, [0,1,0] axis
            (np.pi/2, np.array([0., 1., 0.], dtype=np.float32)),
            
            # For 90-degree rotation around z: pi/2 speed, [0,0,1] axis
            (np.pi/2, np.array([0., 0., 1.], dtype=np.float32)),
            
            # For 45-degree rotation around x: pi/4 speed, [1,0,0] axis
            (np.pi/4, np.array([1., 0., 0.], dtype=np.float32)),
            
            # For arbitrary rotation: specific values
            (2*np.arccos(0.5), np.array([1., 1., 1.]/np.sqrt(3), dtype=np.float32)),
        ]

        # Add test cases for diffQuat
        self.diff_test_cases = [
            # Same quaternions (should result in identity)
            (np.array([1., 0., 0., 0.], dtype=np.float32),  # Identity
             np.array([1., 0., 0., 0.], dtype=np.float32)),
            
            # 90-degree difference around x-axis
            (np.array([1., 0., 0., 0.], dtype=np.float32),  # Identity
             np.array([0.7071067811865476, 0.7071067811865476, 0., 0.], dtype=np.float32)),
            
            # 90-degree difference around y-axis
            (np.array([1., 0., 0., 0.], dtype=np.float32),  # Identity
             np.array([0.7071067811865476, 0., 0.7071067811865476, 0.], dtype=np.float32)),
            
            # 180-degree difference around z-axis
            (np.array([1., 0., 0., 0.], dtype=np.float32),  # Identity
             np.array([0., 0., 0., 1.], dtype=np.float32)),
            
            # Arbitrary rotations
            (np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
             np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)),
        ]

        # Expected results for diffQuat
        self.diff_expected_results = [
            np.array([1., 0., 0., 0.], dtype=np.float32),  # Identity (no difference)
            np.array([0.7071067811865476, 0.7071067811865476, 0., 0.], dtype=np.float32),  # 90-deg x
            np.array([0.7071067811865476, 0., 0.7071067811865476, 0.], dtype=np.float32),  # 90-deg y
            np.array([0., 0., 0., 1.], dtype=np.float32),  # 180-deg z
            np.array([0., -1., 0., 0.], dtype=np.float32),  # Result for arbitrary case
        ]

    def test_mulQuat_implementations_match(self):
        """Test that JAX and NumPy implementations give the same results"""
        try:
            for qa, qb in self.test_cases:
                # Convert inputs to the appropriate type
                qa_jax = jp.array(qa, dtype=jp.float32)
                qb_jax = jp.array(qb, dtype=jp.float32)
                
                # Compute results from both implementations
                result_np = np_mulQuat(qa, qb)
                result_jax = jax_mulQuat(qa_jax, qb_jax)
                
                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)
                
                # Compare results
                np.testing.assert_allclose(
                    result_np, 
                    result_jax, 
                    rtol=1e-5, 
                    atol=1e-5,
                    err_msg=f"Results don't match for qa={qa}, qb={qb}"
                )
        except Exception as e:
            print(f"Error in mulQuat test: {str(e)}")
            raise

    def test_negQuat_implementations_match(self):
        """Test that JAX and NumPy implementations of negQuat give the same results"""
        try:
            for q in self.neg_test_cases:
                # Convert input to JAX array
                print(f"Testing negQuat with input: {q} (type: {q.dtype})")
                q_jax = jp.array(q, dtype=jp.float32)
                print(f"JAX array: {q_jax} (type: {q_jax.dtype})")
                
                # Compute results from both implementations
                result_np = np_negQuat(q)
                print(f"NumPy result: {result_np}")
                
                result_jax = jax_negQuat(q_jax)
                print(f"JAX result: {result_jax}")
                
                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)
                
                # Compare results
                np.testing.assert_allclose(
                    result_np, 
                    result_jax, 
                    rtol=1e-5, 
                    atol=1e-5,
                    err_msg=f"Results don't match for q={q}"
                )
        except Exception as e:
            print(f"Error in negQuat test: {str(e)}")
            raise

    def test_negQuat_properties(self):
        """Test mathematical properties of quaternion negation"""
        for q in self.neg_test_cases:
            q_jax = jp.array(q)
            
            # Test double negation returns original quaternion
            neg_neg_q = jax_negQuat(jax_negQuat(q_jax))
            np.testing.assert_allclose(
                np.array(neg_neg_q),
                q,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Double negation failed for q={q}"
            )
            
            # Test that negation preserves norm
            neg_q = jax_negQuat(q_jax)
            orig_norm = jp.sqrt(jp.sum(q_jax * q_jax))
            neg_norm = jp.sqrt(jp.sum(neg_q * neg_q))
            np.testing.assert_allclose(
                float(orig_norm),
                float(neg_norm),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Norm not preserved for q={q}"
            )

    def test_mulQuat_properties(self):
        """Test mathematical properties of quaternion multiplication"""
        
        # Test identity quaternion property
        identity = np.array([1., 0., 0., 0.])
        identity_jax = jp.array(identity)
        
        for qa, _ in self.test_cases:
            qa_jax = jp.array(qa)
            
            # Identity * q = q
            result_jax = jax_mulQuat(identity_jax, qa_jax)
            np.testing.assert_allclose(
                np.array(result_jax), 
                qa, 
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Identity property failed for q={qa}"
            )
            
            # q * Identity = q
            result_jax = jax_mulQuat(qa_jax, identity_jax)
            np.testing.assert_allclose(
                np.array(result_jax), 
                qa, 
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Identity property failed for q={qa}"
            )

    def test_mulQuat_norm_preservation(self):
        """Test that quaternion multiplication preserves norm"""
        for qa, qb in self.test_cases:
            qa_jax = jp.array(qa)
            qb_jax = jp.array(qb)
            
            # Compute result
            result = jax_mulQuat(qa_jax, qb_jax)
            
            # Check that the result is still a unit quaternion
            norm = jp.sqrt(jp.sum(result * result))
            np.testing.assert_allclose(
                float(norm), 
                1.0, 
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Norm not preserved for qa={qa}, qb={qb}"
            )

    def test_quat2Vel_implementations_match(self):
        """Test that JAX and NumPy implementations of quat2Vel give the same results"""
        try:
            for q, (expected_speed, expected_axis) in zip(self.vel_test_cases, self.vel_expected_results):
                # Convert input to JAX array
                print(f"\nTesting quat2Vel with input: {q} (type: {q.dtype})")
                q_jax = jp.array(q, dtype=jp.float32)
                
                # Test with different dt values
                for dt in [1.0, 0.5, 0.1]:
                    # Compute results from both implementations
                    speed_np, axis_np = np_quat2Vel(q, dt)
                    speed_jax, axis_jax = jax_quat2Vel(q_jax, dt)
                    
                    print(f"dt={dt}")
                    print(f"NumPy result: speed={speed_np}, axis={axis_np}")
                    print(f"JAX result: speed={speed_jax}, axis={axis_jax}")
                    
                    # Convert JAX results to numpy for comparison
                    speed_jax = float(speed_jax)
                    axis_jax = np.array(axis_jax)
                    
                    # Compare results
                    np.testing.assert_allclose(
                        speed_np, 
                        speed_jax, 
                        rtol=1e-5, 
                        atol=1e-5,
                        err_msg=f"Speed doesn't match for q={q}, dt={dt}"
                    )
                    np.testing.assert_allclose(
                        axis_np, 
                        axis_jax, 
                        rtol=1e-5, 
                        atol=1e-5,
                        err_msg=f"Axis doesn't match for q={q}, dt={dt}"
                    )
                    
                    # For dt=1.0, also check against expected results
                    if dt == 1.0:
                        np.testing.assert_allclose(
                            speed_jax, 
                            expected_speed, 
                            rtol=1e-5, 
                            atol=1e-5,
                            err_msg=f"Speed doesn't match expected for q={q}"
                        )
                        np.testing.assert_allclose(
                            np.abs(axis_jax), 
                            np.abs(expected_axis), 
                            rtol=1e-5, 
                            atol=1e-5,
                            err_msg=f"Axis doesn't match expected for q={q}"
                        )
                    
        except Exception as e:
            print(f"Error in quat2Vel test: {str(e)}")
            raise

    def test_diffQuat_implementations_match(self):
        """Test that JAX and NumPy implementations of diffQuat give the same results"""
        try:
            for (q1, q2), expected_result in zip(self.diff_test_cases, self.diff_expected_results):
                # Convert inputs to JAX arrays
                print(f"\nTesting diffQuat with inputs: q1={q1}, q2={q2}")
                q1_jax = jp.array(q1, dtype=jp.float32)
                q2_jax = jp.array(q2, dtype=jp.float32)
                
                # Compute results from both implementations
                result_np = np_diffQuat(q1, q2)
                result_jax = jax_diffQuat(q1_jax, q2_jax)
                
                print(f"NumPy result: {result_np}")
                print(f"JAX result: {result_jax}")
                print(f"Expected result: {expected_result}")
                
                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)
                
                # Compare results
                np.testing.assert_allclose(
                    result_np, 
                    result_jax, 
                    rtol=1e-5, 
                    atol=1e-5,
                    err_msg=f"Results don't match for q1={q1}, q2={q2}"
                )
                
                # Compare with expected results
                np.testing.assert_allclose(
                    np.abs(result_jax), 
                    np.abs(expected_result), 
                    rtol=1e-5, 
                    atol=1e-5,
                    err_msg=f"Result doesn't match expected for q1={q1}, q2={q2}"
                )
                
        except Exception as e:
            print(f"Error in diffQuat test: {str(e)}")
            raise

    def test_diffQuat_properties(self):
        """Test mathematical properties of quaternion difference"""
        try:
            for q1, q2 in self.diff_test_cases:
                q1_jax = jp.array(q1, dtype=jp.float32)
                q2_jax = jp.array(q2, dtype=jp.float32)
                
                # Property 1: diff(q, q) should be identity quaternion
                result = jax_diffQuat(q1_jax, q1_jax)
                identity = jp.array([1., 0., 0., 0.], dtype=jp.float32)
                np.testing.assert_allclose(
                    np.array(result),
                    identity,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Self-difference not identity for q={q1}"
                )
                
                # Property 2: Norm preservation
                result = jax_diffQuat(q1_jax, q2_jax)
                norm = jp.sqrt(jp.sum(result * result))
                np.testing.assert_allclose(
                    float(norm),
                    1.0,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Norm not preserved for q1={q1}, q2={q2}"
                )
                
                # Property 3: diff(q1,q2) * q1 = q2 (up to numerical precision)
                diff_result = jax_diffQuat(q1_jax, q2_jax)
                reconstructed = jax_mulQuat(diff_result, q1_jax)
                np.testing.assert_allclose(
                    np.abs(np.array(reconstructed)),
                    np.abs(q2),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Reconstruction failed for q1={q1}, q2={q2}"
                )
                
        except Exception as e:
            print(f"Error in diffQuat properties test: {str(e)}")
            raise


if __name__ == '__main__':
    try:
        unittest.main()
    except Exception as e:
        print(f"Test failed with error: {str(e)}")