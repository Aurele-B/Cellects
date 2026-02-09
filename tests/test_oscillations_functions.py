#!/usr/bin/env python3
"""
This script contains all unit tests of the oscillations functions script
"""
import unittest
import os
from tests._base import CellectsUnitTest, rgb_video_test, binary_video_test
from cellects.image_analysis.oscillations_functions import *

class TestOscillationsFunctions(CellectsUnitTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.chdir(cls.path_output)
        cls.converted_video = rgb_video_test
        cls.binary = binary_video_test
        cls.arena_label = 1

    def test_detect_oscillations_dynamics(self):
        oscillations_video = detect_oscillations_dynamics(self.converted_video, self.binary, self.arena_label, starting_time=0,
                                     expected_oscillation_period=1, time_interval=1, minimal_oscillating_cluster_size=1)
        self.assertTrue(oscillations_video.any())

    def test_detect_oscillations_dynamics_with_low_memory(self):
        oscillations_video = detect_oscillations_dynamics(self.converted_video, self.binary, self.arena_label, starting_time=0,
                                     expected_oscillation_period=1, time_interval=1, minimal_oscillating_cluster_size=1,
                                     lose_accuracy_to_save_memory=True)
        self.assertTrue(oscillations_video.any())

    def tearDown(self):
        """Remove all written files."""
        dims = self.converted_video.shape
        if os.path.isfile(f"coord_thickening{self.arena_label}_t{dims[0]}_y{dims[1]}_x{dims[2]}.h5"):
            os.remove(f"coord_thickening{self.arena_label}_t{dims[0]}_y{dims[1]}_x{dims[2]}.h5")
        if os.path.isfile(f"coord_slimming{self.arena_label}_t{dims[0]}_y{dims[1]}_x{dims[2]}.h5"):
            os.remove(f"coord_slimming{self.arena_label}_t{dims[0]}_y{dims[1]}_x{dims[2]}.h5")


if __name__ == '__main__':
    unittest.main()