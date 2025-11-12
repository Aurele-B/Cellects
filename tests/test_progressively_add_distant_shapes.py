#!/usr/bin/env python3
"""
This script contains all unit tests of the progressively add distant shapes script
"""
import unittest
from tests._base import CellectsUnitTest, binary_video_test
from cellects.image_analysis.progressively_add_distant_shapes import *
from cellects.image_analysis.morphological_operations import *


class TestProgressivelyAddDistantShapes(CellectsUnitTest):
    """Test suite for ProgressivelyAddDistantShapes class."""

    def setUp(self):
        # Basic test arrays
        self.previous_shape = np.array([[0, 1, 0],
                                        [1, 0, 0],
                                        [0, 0, 0]], dtype=np.uint8)
        self.new_potentials = np.array([[0, 1, 0],
                                        [1, 0, 0],
                                        [0, 0, 1]], dtype=np.uint8)
        self.expected_order = np.array([[0, 1, 0],
                                            [1, 0, 0],
                                            [0, 0, 2]], dtype=np.uint8)
        self.pads1 = ProgressivelyAddDistantShapes(self.new_potentials, self.previous_shape, max_distance=3)
        self.binary = binary_video_test
        self.expected_connection = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
                                    [0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]], dtype=np.uint8)
        self.pads2 = ProgressivelyAddDistantShapes(self.binary[5, ...], self.binary[4, ...], max_distance=7)

    def test_main_shape(self):
        """Test that ProgressivelyAddDistantShapes detects the main shape correctly."""
        self.assertTrue(np.array_equal(self.pads1.main_shape, self.previous_shape))

    def test_main_shape_wrong_label(self):
        """Test that ProgressivelyAddDistantShapes detects the main shape correctly."""
        previous_shape = np.array([[0, 1, 0, 0, 0],
                                          [1, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0]], dtype=np.uint8)
        new_potentials = np.array([[0, 1, 0, 0, 0],
                                          [1, 0, 0, 1, 1],
                                          [0, 0, 0, 1, 0],
                                          [0, 1, 0, 0, 0],
                                          [0, 1, 0, 0, 1]], dtype=np.uint8)
        pads = ProgressivelyAddDistantShapes(previous_shape, new_potentials, 5)
        pads.new_order = np.array([[0, 5, 0, 0, 0],
                                          [2, 0, 0, 3, 3],
                                          [0, 0, 0, 3, 0],
                                          [0, 1, 0, 0, 0],
                                          [0, 1, 0, 0, 4]], dtype=np.uint8)
        pads._check_main_shape_label(previous_shape)
        self.assertTrue(np.any(pads.new_order * previous_shape == 1))

    def test_init_with_empty_shapes(self):
        """Test initialization with empty shapes."""
        empty_shape = np.zeros((3, 3), dtype=np.uint8)
        pads = ProgressivelyAddDistantShapes(empty_shape, empty_shape, 5)
        self.assertTrue(np.array_equal(pads.new_order, empty_shape))
        self.assertTrue(np.array_equal(pads.main_shape, empty_shape))

    def test_check_main_shape_label_no_overlap(self):
        """Test check_main_shape_label when previous shape doesn't overlap with new_order."""
        non_overlapping = np.array([[0, 0, 1],
                                    [0, 1, 0],
                                    [1, 0, 0]], dtype=np.uint8)
        pads = ProgressivelyAddDistantShapes(self.previous_shape, non_overlapping, 5)
        expected = np.array([[0, 1, 1],
                                    [1, 1, 0],
                                    [1, 0, 0]], dtype=np.uint8)
        # Should set main_shape to the union of both shapes
        self.assertTrue(np.array_equal(pads.main_shape, expected))

    def test_consider_shapes_sizes_basic(self):
        """Test basic functionality of consider_shapes_sizes."""
        self.pads1.consider_shapes_sizes()
        self.assertTrue(np.array_equal(self.pads1.new_order, self.expected_order))

    def test_consider_shapes_sizes_no_max_distance(self):
        """Test functionality of consider_shapes_sizes with no max_distance."""
        pads = ProgressivelyAddDistantShapes(self.new_potentials, self.previous_shape, 0)
        pads.consider_shapes_sizes()
        self.assertTrue(np.array_equal(pads.new_order, self.expected_order))

    def test_consider_shapes_size_min(self):
        """Test functionality of consider_shapes_sizes with minimum size."""
        self.pads1.consider_shapes_sizes(min_shape_size=1)
        self.assertTrue(np.array_equal(self.pads1.new_order, self.expected_order))

    def test_consider_shapes_size_max(self):
        """Test functionality of consider_shapes_sizes with maximum size."""
        self.pads1.consider_shapes_sizes(max_shape_size=2)
        self.assertTrue(np.array_equal(self.pads1.new_order, self.expected_order))

    def test_consider_shapes_size_min_and_max(self):
        """Test functionality of consider_shapes_sizes with maximum size."""
        self.pads1.consider_shapes_sizes(min_shape_size=1, max_shape_size=2)
        self.assertTrue(np.array_equal(self.pads1.new_order, self.expected_order))

    def test_connect_shapes(self):
        """Test functionality of connect_shapes."""
        self.pads2.connect_shapes(only_keep_connected_shapes=False, rank_connecting_pixels=False, intensity_valley=None)
        self.assertIsInstance(self.pads2.expanded_shape, np.ndarray)

    def test_connect_shapes_fails(self):
        """Test functionality of connect_shapes fails."""
        self.pads1.connect_shapes(only_keep_connected_shapes=False, rank_connecting_pixels=False, intensity_valley=None)
        self.assertTrue(np.array_equal(self.pads1.expanded_shape, self.previous_shape))

    def test_connect_shapes_without_unconnected(self):
        """Test functionality of connect_shapes without unconnected shapes."""
        self.pads2.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=False, intensity_valley=None)
        self.assertIsInstance(self.pads2.expanded_shape, np.ndarray)

    def test_connect_shapes_with_pixel_ranking(self):
        """Test functionality of connect_shapes with pixel ranking."""
        self.pads2.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=True, intensity_valley=None)
        self.assertIsInstance(self.pads2.expanded_shape, np.ndarray)

    def test_connect_shapes_with_intensity_valley(self):
        """Test functionality of connect_shapes with intensity_valley."""
        intensity_valley = np.arange(self.binary[0, ...].size)[::-1].reshape(self.binary[0, ...].shape)
        self.pads2.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=False,
                                  intensity_valley=intensity_valley)
        self.assertIsInstance(self.pads2.expanded_shape, np.ndarray)

    def test_modify_past_analysis(self):
        """Test functionality of modify_past_analysis."""
        self.pads2.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=True)
        result = self.pads2.modify_past_analysis(self.binary, self.binary)

        self.assertTrue(result.shape == (6, 10, 10))


if __name__ == '__main__':
    unittest.main()