#!/usr/bin/env python3
"""
This script contains all unit tests of the image segmentation script
"""

import unittest
from tests._base import CellectsUnitTest
from cellects.image_analysis.image_segmentation import *
import numpy as np
from numba.typed import Dict, List
import cv2


class TestApplyFilter(CellectsUnitTest):
    """Test suite for apply_filter function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple 3x3 test image
        self.test_image = np.zeros((3, 3), dtype=np.float32)
        self.test_image[1, 1] = 1.0

    def test_gaussian_filter_valid_input(self):
        """Test Gaussian filter with valid input parameters."""
        expected_shape = (3, 3)
        result = apply_filter(self.test_image, "Gaussian", [1.0])
        self.assertEqual(result.shape, expected_shape)
        # Check that the result is not identical to input (filter was applied)
        self.assertFalse(np.array_equal(result, self.test_image))

    def test_median_filter_valid_input(self):
        """Test Median filter with valid input."""
        expected_shape = (3, 3)
        result = apply_filter(self.test_image, "Median", [])
        self.assertEqual(result.shape, expected_shape)

    def test_butterworth_filter_valid_input(self):
        """Test Butterworth filter with valid parameters."""
        result = apply_filter(self.test_image, "Butterworth", [0.005, 2])
        self.assertEqual(result.shape, (3, 3))

    def test_frangi_filter_valid_input(self):
        """Test Frangi filter with valid parameters."""
        result = apply_filter(self.test_image, "Frangi", [1.0, 3.0])
        self.assertEqual(result.shape, (3, 3))

    def test_laplace_filter_valid_input(self):
        """Test Laplace filter with valid parameter."""
        result = apply_filter(self.test_image, "Laplace", [3])
        self.assertEqual(result.shape, (3, 3))

    def test_median_no_params(self):
        """Test Median filter with no parameters."""
        result = apply_filter(self.test_image, "Median", [])
        self.assertEqual(result.shape, (3, 3))

    def test_farid_no_params(self):
        """Test Farid filter with no parameters."""
        result = apply_filter(self.test_image, "Farid", [])
        self.assertEqual(result.shape, (3, 3))

    def test_rescale_to_uint8(self):
        """Test rescaling to uint8 when input is float32."""
        # Create a test image with float values outside 0-255
        float_image = np.array([[1.5, 2.5], [0.0, 3.0]], dtype=np.float32)
        result = apply_filter(float_image, "Gaussian", [1.0], rescale_to_uint8=True)
        self.assertEqual(result.dtype, np.uint8)


class TestGetColorSpaces(CellectsUnitTest):

    def test_typed_dict(self):
        # Create a BGR image for testing
        self.bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Call the get_color_spaces function
        self.result = get_color_spaces(self.bgr_image)
        # Add assertions to verify the expected behavior
        self.assertIsInstance(self.result, Dict)

    def test_complete_dict(self):
        # Create a BGR image for testing
        self.bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Call the get_color_spaces function
        self.result = get_color_spaces(self.bgr_image)

        self.assertIn('bgr', self.result)
        self.assertIn('lab', self.result)
        self.assertIn('hsv', self.result)
        self.assertIn('luv', self.result)
        self.assertIn('hls', self.result)
        self.assertIn('yuv', self.result)

        self.assertEqual(self.result['bgr'].size, self.bgr_image.size)
        self.assertEqual(self.result['lab'].size, self.bgr_image.size)
        self.assertEqual(self.result['hsv'].size, self.bgr_image.size)
        self.assertEqual(self.result['luv'].size, self.bgr_image.size)
        self.assertEqual(self.result['hls'].size, self.bgr_image.size)
        self.assertEqual(self.result['yuv'].size, self.bgr_image.size)


class TestCombineColorSpaces(CellectsUnitTest):
    """Test suite for combine_color_spaces function."""

    def test_basic_combination(self):
        """Test basic combination of color spaces without background subtraction."""
        c_space_dict = Dict()
        c_space_dict['bgr'] = np.array([1.0, 0.5, 0.2])
        c_space_dict['hsv'] = np.array([0.3, 0.8, 0.1])
        all_c_spaces = Dict()
        all_c_spaces['bgr'] = np.ones((5, 5, 3))
        all_c_spaces['hsv'] = np.full((5, 5, 3), 0.5)

        result = combine_color_spaces(c_space_dict, all_c_spaces)

        expected_shape = (5, 5)
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(result.max() < 256)
        self.assertTrue(result.min() > -1)


    def test_background_subtraction_larger_image(self):
        """Test background subtraction when image sum > background sum."""
        c_space_dict = Dict()
        c_space_dict['bgr'] = np.array([1.0, 0.5, 0.2])
        c_space_dict['hsv'] = np.array([0.3, 0.8, 0.1])
        all_c_spaces = Dict()
        all_c_spaces['bgr'] = np.random.rand(5, 5, 3)
        all_c_spaces['hsv'] = np.random.rand(5, 5, 3)

        background = np.full((5, 5), .01)
        result = combine_color_spaces(c_space_dict, all_c_spaces, background)

        self.assertTrue(np.all(result >= 0))
        self.assertTrue(result.max() < 256)
        self.assertTrue(result.min() > -1)

    def test_negative_coefficients(self):
        """Test with negative coefficients."""
        c_space_dict = Dict()
        c_space_dict['bgr'] = np.array([-1.0, 0.5, -2.0])
        c_space_dict['hsv'] = np.array([-1.0, 0.5, -2.0])
        all_c_spaces = Dict()
        all_c_spaces['bgr'] = np.random.rand(5, 5, 3)
        all_c_spaces['hsv'] = np.random.rand(5, 5, 3)

        result = combine_color_spaces(c_space_dict, all_c_spaces)
        expected_shape = (5, 5)
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.all(result >= 0))


class TestGenerateColorSpaceCombination(CellectsUnitTest):
    def test_generate_color_space_combination(self):

        # Create a BGR image for testing
        self.bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
        c_space_dict = Dict()
        c_space_dict['bgr'] = np.array((1, 0, 1), np.uint8)
        c_space_dict['hsv'] = np.array((0.5, 5, 0.5), np.uint8)
        c_spaces = List(['bgr', 'hsv'])

        subtract_background = np.zeros((100, 100), dtype=np.float64)
        expected_result = np.zeros((100, 100), dtype=np.float64)

        result, _ = generate_color_space_combination(self.bgr_image, c_spaces, c_space_dict, background=subtract_background)
        self.assertEqual(result.shape, (100, 100))
        self.assertTrue(np.allclose(result, expected_result))


class TestOtsuThreshold(CellectsUnitTest):
    def test_get_otsu_threshold(self):
        image = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint8)

        expected_threshold = 5

        threshold = get_otsu_threshold(image)
        self.assertAlmostEqual(threshold, expected_threshold, places=1)


class TestOtsuThresholding(CellectsUnitTest):
    def test_otsu_thresholding(self):
        image = np.array([[[-100, 150.54752, 200287, 250, 300], [-100, -150, 200, 250, 300], [100, 150.5735, 200, 250, 300]], [[100.56, 15054, 200, 250, 300], [100, 150, 200.548, 250, 300], [100, 150, 200, 250, 300]]], dtype=np.float64)
        expected_binary_image = np.zeros(image.shape, np.uint8)
        expected_binary_image[0, 0, 2] = 1
        binary_image = otsu_thresholding(image)
        self.assertTrue(np.array_equal(binary_image, expected_binary_image))


class TestSegmentWithLumValue(CellectsUnitTest):
    def test_segment_with_lum_value_lighter_background(self):
        converted_video = np.array([[[100, 120], [130, 140]], [[160, 170], [180, 200]]], dtype=np.uint8)
        basic_bckgrnd_values = np.array([110, 150], dtype=np.uint8)
        l_threshold = 180
        lighter_background = True
        expected_segmentation = np.array([[[1, 1], [1, 0]], [[1, 1], [0, 0]]], dtype=np.uint8)
        expected_l_threshold_over_time = np.array([140, 180], dtype=np.int64)

        segmentation, l_threshold_over_time = segment_with_lum_value(converted_video, basic_bckgrnd_values,
                                                                     l_threshold, lighter_background)
        self.assertTrue(np.array_equal(segmentation, expected_segmentation))
        self.assertTrue(np.array_equal(l_threshold_over_time, expected_l_threshold_over_time))

    def test_segment_with_lum_value_darker_background(self):
        converted_video = np.array([[[100, 120], [130, 140]], [[160, 170], [180, 200]]], dtype=np.uint8)
        basic_bckgrnd_values = np.array([110, 130], dtype=np.uint8)
        l_threshold = 150
        lighter_background = False
        expected_segmentation = np.array([[[0, 0], [0, 1]], [[1, 1], [1, 1]]], dtype=np.uint8)
        expected_l_threshold_over_time = np.array([130, 150], dtype=np.int64)

        segmentation, l_threshold_over_time = segment_with_lum_value(converted_video, basic_bckgrnd_values,
                                                                     l_threshold, lighter_background)
        self.assertTrue(np.array_equal(segmentation, expected_segmentation))
        self.assertTrue(np.array_equal(l_threshold_over_time, expected_l_threshold_over_time))


class TestRollingWindowSegmentation(CellectsUnitTest):
    """Test suite for rolling_window_segmentation function."""

    def test_rolling_window_with_normal_input(self):
        """Test normal operation with valid input."""
        # Setup test data
        greyscale_image = np.array([[1, 2, 1, 1], [1, 3, 4, 1], [2, 4, 3, 1], [2, 1, 2, 1]])
        possibly_filled_pixels = greyscale_image > 1
        patch_size = (2, 2)

        # Execute function
        result = rolling_window_segmentation(greyscale_image, possibly_filled_pixels, patch_size)

        # Expected output based on the example in docstring
        expected = np.array([[0, 1, 0, 0],
                             [0, 1, 1, 0],
                             [0, 1, 1, 0],
                             [0, 0, 1, 0]], dtype=np.uint8)

        # Verify result
        self.assertTrue(np.array_equal(result, expected))

    def test_rolling_window_with_single_pixel(self):
        """Test with single pixel image."""
        greyscale_image = np.array([[42]], dtype=np.uint8)
        possibly_filled_pixels = np.array([[1]], dtype=np.uint8)
        patch_size = (2, 2)

        result = rolling_window_segmentation(greyscale_image, possibly_filled_pixels, patch_size)
        self.assertEqual(result.shape, (1, 1))
        # Should be all zeros since patch is larger than image
        self.assertEqual(result[0, 0], 0)

    def test_rolling_window_with_image_smaller_than_patch(self):
        """Test with image smaller than patch size."""
        greyscale_image = np.ones((2, 2), dtype=np.uint8)
        possibly_filled_pixels = np.ones((2, 2), dtype=np.uint8)
        patch_size = (5, 5)

        result = rolling_window_segmentation(greyscale_image, possibly_filled_pixels, patch_size)
        expected = np.zeros((2, 2), dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected))

    def test_rolling_window_with_zero_variation_in_patches(self):
        """Test with patches that have zero variation."""
        greyscale_image = np.array([[5, 5], [5, 5]], dtype=np.uint8)
        possibly_filled_pixels = np.ones_like(greyscale_image, dtype=np.uint8)
        patch_size = (2, 2)

        result = rolling_window_segmentation(greyscale_image, possibly_filled_pixels, patch_size)
        expected = np.zeros((2, 2), dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected))

    def test_rolling_window_with_all_zeros_in_possibly_filled(self):
        """Test with all zeros in possibly_filled_pixels."""
        greyscale_image = np.ones((2, 2), dtype=np.uint8)
        possibly_filled_pixels = np.zeros((2, 2), dtype=np.uint8)
        patch_size = (2, 2)

        result = rolling_window_segmentation(greyscale_image, possibly_filled_pixels, patch_size)
        expected = np.zeros((2, 2), dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected))

    def test_rolling_window_with_non_uint8_inputs(self):
        """Test with non-uint8 inputs."""
        greyscale_image = np.ones((2, 2), dtype=np.float32)
        possibly_filled_pixels = np.ones((2, 2), dtype=np.float32)
        patch_size = (2, 2)

        result = rolling_window_segmentation(greyscale_image, possibly_filled_pixels, patch_size)
        self.assertTrue(result.dtype == "uint8")


class TestBinaryQualityIndex(CellectsUnitTest):
    """Test suite for binary_quality_index function."""

    def test_binary_quality_empty_image(self):
        """Test that binary_quality_index returns 0 for an empty image."""
        # Setup test data
        binary_img = np.zeros((5, 5), dtype=np.uint8)

        # Execute function
        result = binary_quality_index(binary_img)

        # Verify result
        self.assertEqual(result, 0.0)

    def test_binary_quality_single_component(self):
        """Test binary quality index for an image with single connected component."""
        # Setup test data - 2x2 square
        binary_img = np.zeros((5, 5), dtype=np.uint8)
        binary_img[1:3, 1:3] = 1

        # Execute function
        result = binary_quality_index(binary_img)

        # The perimeter of a 2x2 square is 4
        expected = np.square(4) / binary_img.sum()
        self.assertEqual(result, expected)

    def test_binary_quality_multiple_components(self):
        """Test binary quality index for an image with multiple connected components."""
        # Setup test data - two 2x2 squares
        binary_img = np.zeros((5, 5), dtype=np.uint8)
        binary_img[1:3, 1:3] = 1  # First square
        binary_img[1:3, 4:6] = 1  # Second square (out of bounds, will be smaller)

        # Execute function
        result = binary_quality_index(binary_img)

        # The largest component is the first square
        expected = np.square(4) / binary_img.sum()
        self.assertEqual(result, expected)

    def test_binary_quality_full_image(self):
        """Test binary quality index for a fully white image."""
        # Setup test data
        binary_img = np.ones((5, 5), dtype=np.uint8)

        # Execute function
        result = binary_quality_index(binary_img)

        # For a full 5x5 image, the perimeter is 5*2 + 3*2 = 16
        expected = np.square(16) / binary_img.sum()
        self.assertEqual(result, expected)

    def test_binary_quality_non_binary_values(self):
        """Test that function handles non-binary values appropriately."""
        # Setup test data with intermediate values
        binary_img = np.array([[0, 50, 100, 150, 200],
                               [25, 75, 125, 175, 230],
                               [50, 100, 150, 200, 255],
                               [75, 125, 175, 230, 0],
                               [100, 150, 200, 255, 50]], dtype=np.uint8)

        # Execute function - this should still work as the function doesn't explicitly
        # check for binary values, just uses the sum and connected components
        result = binary_quality_index(binary_img)

        # Verify it doesn't raise an exception and returns a value
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)


class TestFindThresholdGivenMask(CellectsUnitTest):
    """Test suite for find_threshold_given_mask function."""

    def test_normal_operation(self):
        """Test normal operation with clear separation between regions."""
        greyscale = np.array([[255, 128, 54], [0, 64, 20]], dtype=np.uint8)
        mask = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        expected = 54
        result = find_threshold_given_mask(greyscale, mask)
        self.assertEqual(result, expected)

    def test_empty_region_a(self):
        """Test when region A has no pixels."""
        greyscale = np.array([[0, 64, 20]], dtype=np.uint8)
        mask = np.array([[0, 0, 0]], dtype=np.uint8)  # All zeros
        expected = 255  # Should return maximum since region A is empty
        result = find_threshold_given_mask(greyscale, mask)
        self.assertEqual(result, expected)

    def test_empty_region_b(self):
        """Test when region B has no pixels."""
        greyscale = np.array([[255, 128, 54]], dtype=np.uint8)
        mask = np.array([[1, 1, 1]], dtype=np.uint8)  # All ones
        expected = 0  # Should return minimum since region B is empty
        result = find_threshold_given_mask(greyscale, mask)
        self.assertEqual(result, expected)

    def test_single_value_regions(self):
        """Test when both regions have the same single value."""
        greyscale = np.array([[50, 50], [150, 150]], dtype=np.uint8)
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        expected = 75.0  # Average value
        result = find_threshold_given_mask(greyscale, mask)
        self.assertEqual(result, expected)

    def test_single_value_regions(self):
        """Test when both regions have the same single value."""
        greyscale = np.array([[50, 50], [100, 100]], dtype=np.uint8)
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        expected = 75.0  # Average value
        result = find_threshold_given_mask(greyscale, mask)
        self.assertEqual(result, expected)

    def test_min_threshold(self):
        """Test with non-zero minimum threshold."""
        greyscale = np.array([[255, 128, 54], [0, 64, 20]], dtype=np.uint8)
        mask = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        expected = 54
        result = find_threshold_given_mask(greyscale, mask, min_threshold=30)
        self.assertEqual(result, expected)

    def test_all_max_values(self):
        """Test when all values are maximum."""
        greyscale = np.full((2, 2), 255, dtype=np.uint8)
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        expected = 255  # Should return maximum since all values are equal
        result = find_threshold_given_mask(greyscale, mask)
        self.assertEqual(result, expected)

    def test_mixed_values_with_min_threshold(self):
        """Test with mixed values and non-zero minimum threshold."""
        greyscale = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        expected = 25.0
        result = find_threshold_given_mask(greyscale, mask, min_threshold=15)
        self.assertEqual(result, expected)

    def test_edge_case_min_threshold_equal_max(self):
        """Test when min_threshold equals the maximum possible value."""
        greyscale = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        expected = 25.0  # Average
        result = find_threshold_given_mask(greyscale, mask, min_threshold=255)
        self.assertEqual(result, expected)

    def test_large_image(self):
        """Test with larger image to ensure performance."""
        size = 100
        greyscale = np.random.randint(0, 256, size=(size, size), dtype=np.uint8)
        mask = np.random.choice([0, 1], size=(size, size))
        result = find_threshold_given_mask(greyscale, mask)
        # Just verify it returns a value in range
        self.assertTrue(0 <= result <= 255)


if __name__ == '__main__':
    unittest.main()
