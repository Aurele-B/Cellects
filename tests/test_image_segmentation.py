#!/usr/bin/env python3
"""
This script contains all unit tests of the image_analysis directory
"""

import unittest
from tests._base import CellectsUnitTest
from cellects.image_analysis.image_segmentation import *
import numpy as np
from numba.typed import Dict, List
import cv2


class TestGetAllColorSpaces(CellectsUnitTest):

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


class TestFilterMexicanHat(CellectsUnitTest):

    def test_filter_mexican_hat(self):
        # Create a test image for filtering
        image = np.zeros((100, 100), dtype=np.uint8)

        # Call the filter_mexican_hat function
        result = filter_mexican_hat(image)

        # Add assertions to verify the expected behavior
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, image.shape)


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


if __name__ == '__main__':
    unittest.main()
