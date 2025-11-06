#!/usr/bin/env python3
"""
This script contains all unit tests of the cell leaving detection script
"""
import unittest
from tests._base import CellectsUnitTest
from cellects.image_analysis.cell_leaving_detection import *
from cellects.image_analysis.morphological_operations import cross_33


class TestCellLeavingDetection(CellectsUnitTest):
    """Test suite for ProgressivelyAddDistantShapes class."""
    def setUp(self):
        # Basic test arrays
        self.new_shape = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                                        [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
                                        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.uint8)
        self.previous_binary = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]], dtype=np.uint8)
        self.covering_intensity = np.full(self.new_shape.shape, 150,dtype=np.uint8)
        self.covering_intensity *= self.previous_binary
        self.greyscale_image = np.array([[0, 0, 0, 20, 200, 200, 20, 0, 0, 0],
                                               [0, 200, 200, 200, 200, 200, 200, 200, 200, 0],
                                               [0, 200, 200, 200, 200, 200, 200, 200, 200, 200],
                                               [0, 200, 200, 200, 200, 200, 200, 200, 200, 200],
                                               [200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
                                               [20, 200, 200, 200, 200, 200, 200, 20, 200, 20],
                                               [0, 200, 200, 200, 200, 200, 20, 20, 200, 0],
                                               [0, 200, 200, 200, 200, 20, 20, 20, 20, 0],
                                               [20, 200, 200, 200, 200, 20, 20, 20, 0, 0],
                                               [0, 20, 200, 200, 200, 20, 0, 0, 0, 0]], dtype=np.uint8)
        self.fading_coefficient = 0.1
        self.lighter_background = False
        self.several_blob_per_arena = True
        self.erodila_disk = cross_33
        self.protect_from_fading = np.zeros_like(self.new_shape)
        self.add_to_fading = np.zeros_like(self.new_shape)
        self.expected = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.uint8)

    def test_cell_leaving_detection(self):
        """Test cell_leaving_detection main functionality."""
        new_shape, covering_intensity = cell_leaving_detection(self.new_shape, self.covering_intensity,
                                                               self.previous_binary, self.greyscale_image,
                                                               self.fading_coefficient, self.lighter_background,
                                                               self.several_blob_per_arena, self.erodila_disk,
                                                               self.protect_from_fading, self.add_to_fading)
        self.assertTrue(np.array_equal(new_shape, self.expected))

    def test_cell_leaving_detection_with_several_blob(self):
        """Test cell_leaving_detection with several blob."""
        self.several_blob_per_arena = False
        new_shape, covering_intensity = cell_leaving_detection(self.new_shape, self.covering_intensity,
                                                               self.previous_binary, self.greyscale_image,
                                                               self.fading_coefficient, self.lighter_background,
                                                               self.several_blob_per_arena, self.erodila_disk,
                                                               self.protect_from_fading, self.add_to_fading)
        self.assertTrue(np.array_equal(new_shape, self.expected))

    def test_cell_leaving_detection_with_lighter_background(self):
        """Test cell_leaving_detection with lighter background."""
        self.lighter_background = True
        self.greyscale_image[self.greyscale_image == 0] = 255
        self.greyscale_image[self.greyscale_image == 20] = 230
        self.greyscale_image[self.greyscale_image == 200] = 30
        new_shape, covering_intensity = cell_leaving_detection(self.new_shape, self.covering_intensity,
                                                               self.previous_binary, self.greyscale_image,
                                                               self.fading_coefficient, self.lighter_background,
                                                               self.several_blob_per_arena, self.erodila_disk,
                                                               self.protect_from_fading, self.add_to_fading)
        self.assertTrue(np.array_equal(new_shape, self.expected))


if __name__ == '__main__':
    unittest.main()