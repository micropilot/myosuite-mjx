import unittest
import numpy as np
import jax.numpy as jp
import glob
import pickle

from myosuite.mjx.reference_motion import ReferenceMotion as JaxReferenceMotion
from myosuite.logger.reference_motion import ReferenceMotion as NumpyReferenceMotion
from myosuite.logger.reference_motion import ReferenceType

class TestReferenceMotion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Path to test data
        cls.data_dir = "./myosuite/envs/myo/myodm/data/"
        cls.reference_files = glob.glob(f"{cls.data_dir}*.npz")
        cls.ignore_files = [
            f'{cls.data_dir}MyoHand_cylindersmall_lift.npz',
            f"{cls.data_dir}MyoHand_fryingpan_cook2.npz",
            f"{cls.data_dir}MyoHand_hand_pass1.npz",
            f"{cls.data_dir}MyoHand_knife_lift.npz",
            f"{cls.data_dir}MyoHand_wineglass_drink1.npz"
        ]
        cls.fixed_ref_data = {
                "time": (0.0, 4.0),
                "robot": np.zeros((1, 29)),
                "robot_vel": np.zeros((1, 29)),
                "object_init": np.array((0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0)),
                "object": np.array(
                        [[-0.2, -0.2, 0.1, 1.0, 0.0, 0.0, -1.0]]
                ),
            }

        cls.random_ref_data = {
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
        cls.file_path = './myosuite/envs/myo/myodm/data/MyoHand_airplane_fly1.npz'
        
    def test_load_npz_file(self):
        """Test loading .npz files in both implementations"""
        for file_path in self.reference_files:
            if file_path in self.ignore_files:
                print(f"Skipping {file_path} because it is in the ignore list")
                continue

            # Load with both implementations
            numpy_ref = NumpyReferenceMotion(file_path)
            jax_ref = JaxReferenceMotion(file_path)
            
            # Compare basic properties
            self.assertEqual(jax_ref.horizon, numpy_ref.horizon)
            self.assertEqual(jax_ref.robot_dim, numpy_ref.robot_dim)
            self.assertEqual(jax_ref.object_dim, numpy_ref.object_dim)
            self.assertEqual(jax_ref.type.value, numpy_ref.type.value)
            
            # Compare reference data
            np.testing.assert_allclose(
                np.array(jax_ref.reference["time"]),
                numpy_ref.reference["time"],
                rtol=1e-5
            )
            if jax_ref.reference["robot"] is not None:
                np.testing.assert_allclose(
                    np.array(jax_ref.reference["robot"]),
                    numpy_ref.reference["robot"],
                    rtol=1e-5
                )
            if jax_ref.reference["object"] is not None:
                np.testing.assert_allclose(
                    np.array(jax_ref.reference["object"]),
                    numpy_ref.reference["object"],
                    rtol=1e-5
                )

    def test_fixed_reference(self):
        """Test fixed reference type in both implementations"""
        # Create references
        
        jax_ref = JaxReferenceMotion(self.fixed_ref_data)
        numpy_ref = NumpyReferenceMotion(self.fixed_ref_data)
        
        # Check type
        self.assertEqual(jax_ref.type.value, ReferenceType.FIXED.value)
        self.assertEqual(numpy_ref.type.value, ReferenceType.FIXED.value    )
        
        # Check initialization
        robot_init_jax, object_init_jax = jax_ref.get_init()
        robot_init_numpy, object_init_numpy = numpy_ref.get_init()
        
        np.testing.assert_allclose(np.array(robot_init_jax), robot_init_numpy)
        np.testing.assert_allclose(np.array(object_init_jax), object_init_numpy)

    def test_random_reference(self):
        """Test random reference type in both implementations"""
        # Create references
        jax_ref = JaxReferenceMotion(self.random_ref_data)
        numpy_ref = NumpyReferenceMotion(self.random_ref_data)
        
        # Check type
        self.assertEqual(jax_ref.type.value, ReferenceType.RANDOM.value)
        self.assertEqual(numpy_ref.type.value, ReferenceType.RANDOM.value)
        
        # Check initialization
        robot_init_jax, object_init_jax = jax_ref.get_init()
        robot_init_numpy, object_init_numpy = numpy_ref.get_init()
        
        np.testing.assert_allclose(np.array(robot_init_jax), robot_init_numpy)
        np.testing.assert_allclose(np.array(object_init_jax), object_init_numpy)

    def test_track_reference(self):
        """Test tracking reference type using actual motion files"""
        
        # Create references
        jax_ref = JaxReferenceMotion(self.file_path)
        numpy_ref = NumpyReferenceMotion(self.file_path)
        
        # Check type
        self.assertEqual(jax_ref.type.value, ReferenceType.TRACK.value)
        self.assertEqual(numpy_ref.type.value, ReferenceType.TRACK.value)
        
        # Test time slot finding
        test_times = [0.0, 0.1, 0.5, 1.0]
        for time in test_times:
            jax_indices = jax_ref.find_timeslot_in_reference(time)
            numpy_indices = numpy_ref.find_timeslot_in_reference(time)
            self.assertEqual(jax_indices, numpy_indices)

    def test_reset(self):
        """Test reset functionality"""
        
        # Create references
        jax_ref = JaxReferenceMotion(self.file_path)
        numpy_ref = NumpyReferenceMotion(self.file_path)
        
        # Advance index cache
        _ = jax_ref.find_timeslot_in_reference(0.5)
        _ = numpy_ref.find_timeslot_in_reference(0.5)
        
        # Reset
        jax_ref.reset()
        numpy_ref.reset()
        
        # Check if index cache is reset
        self.assertEqual(jax_ref.index_cache, 0)
        self.assertEqual(numpy_ref.index_cache, 0)

    def test_error_handling(self):
        """Test error handling in both implementations"""
        # Test invalid reference data
        invalid_ref = {
            "time": np.array([0.0]),
            "robot": np.array([0.0, 0.1, 0.2])  # Wrong shape
        }
        
        with self.assertRaises(AssertionError):
            _ = JaxReferenceMotion(invalid_ref)
        with self.assertRaises(AssertionError):
            _ = NumpyReferenceMotion(invalid_ref)
        
        # Test missing time key
        invalid_ref = {
            "robot": np.array([[0.0, 0.1, 0.2]])
        }
        
        with self.assertRaises(AssertionError):
            _ = JaxReferenceMotion(invalid_ref)
        with self.assertRaises(AssertionError):
            _ = NumpyReferenceMotion(invalid_ref)

if __name__ == '__main__':
    unittest.main()
