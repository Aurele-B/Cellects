#!/usr/bin/env python3
"""
This script contains all unit tests of the image_analysis directory
"""

import unittest
from cellects.test.cellects_unit_test import CellectsUnitTest
from cellects.image_analysis.image_segmentation import *
from numpy import ndarray, allclose, array_equal, int64, min, max, all, any, zeros_like, argmin, logical_and, pi, square, mean, median, float32, histogram, cumsum, logical_not, float64, array, zeros, std, sum, uint8, round, isin, append, delete, argmax, diff, argsort, argwhere, logical_or, unique, nonzero
from numba.typed import Dict as TDict
from cv2 import TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, kmeans, KMEANS_RANDOM_CENTERS, filter2D, cvtColor, COLOR_BGR2LAB, COLOR_BGR2HSV, COLOR_BGR2LUV, COLOR_BGR2HLS, COLOR_BGR2YUV, connectedComponents, connectedComponentsWithStats



class TestGetAllColorSpaces(CellectsUnitTest):

    def test_typed_dict(self):
        # Create a BGR image for testing
        self.bgr_image = zeros((100, 100, 3), dtype=uint8)

        # Call the get_all_color_spaces function
        self.result = get_all_color_spaces(self.bgr_image)
        # Add assertions to verify the expected behavior
        self.assertIsInstance(self.result, TDict)

    def test_complete_dict(self):
        # Create a BGR image for testing
        self.bgr_image = zeros((100, 100, 3), dtype=uint8)

        # Call the get_all_color_spaces function
        self.result = get_all_color_spaces(self.bgr_image)

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


class TestGetSomeColorSpaces(CellectsUnitTest):
    # Create a BGR image for testing
    bgr_image = zeros((100, 100, 3), dtype=uint8)
    spaces = TDict()
    spaces['bgr'] = array((0, 0, 1), uint8)
    spaces['lab'] = array((0, 0, 1), uint8)
    spaces['hsv'] = array((0, 0, 1), uint8)
    spaces['luv'] = array((0, 0, 1), uint8)
    spaces['hls'] = array((0, 0, 1), uint8)
    spaces['yuv'] = array((0, 0, 1), uint8)
    def test_typed_dict(self):
        # Call the get_all_color_spaces function
        self.result = get_some_color_spaces(self.bgr_image, self.spaces)
        # Add assertions to verify the expected behavior
        self.assertIsInstance(self.result, TDict)

    def test_complete_dict(self):
        # Call the get_all_color_spaces function
        # Create a BGR image for testing
        self.bgr_image = zeros((100, 100, 3), dtype=uint8)
        self.result = get_some_color_spaces(self.bgr_image, self.spaces)

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
        image = zeros((100, 100), dtype=uint8)

        # Call the filter_mexican_hat function
        result = filter_mexican_hat(image)

        # Add assertions to verify the expected behavior
        self.assertIsInstance(result, ndarray)
        self.assertEqual(result.dtype, uint8)
        self.assertEqual(result.shape, image.shape)


class TestGenerateColorSpaceCombination(CellectsUnitTest):
    def test_generate_color_space_combination(self):

        # Create a BGR image for testing
        self.bgr_image = zeros((100, 100, 3), dtype=uint8)
        c_space_dict = TDict()
        c_space_dict['bgr'] = array((1, 0, 1), uint8)
        c_space_dict['hsv'] = array((0.5, 5, 0.5), uint8)
        all_c_spaces = get_all_color_spaces(self.bgr_image)

        subtract_background = zeros((100, 100), dtype=float64)
        expected_result = zeros((100, 100), dtype=float64)

        result = generate_color_space_combination(c_space_dict, all_c_spaces, subtract_background)
        self.assertEqual(result.shape, (100, 100))
        self.assertTrue(allclose(result, expected_result))


class TestOtsuThreshold(CellectsUnitTest):
    def test_get_otsu_threshold(self):
        image = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=uint8)

        expected_threshold = 5

        threshold = get_otsu_threshold(image)
        self.assertAlmostEqual(threshold, expected_threshold, places=1)


class TestOtsuThresholding(CellectsUnitTest):
    def test_otsu_thresholding(self):
        image = array([[[-100, 150.54752, 200287, 250, 300], [-100, -150, 200, 250, 300], [100, 150.5735, 200, 250, 300]], [[100.56, 15054, 200, 250, 300], [100, 150, 200.548, 250, 300], [100, 150, 200, 250, 300]]], dtype=float64)
        expected_binary_image = zeros(image.shape, uint8)
        expected_binary_image[0, 0, 2] = 1
        binary_image = otsu_thresholding(image)
        self.assertTrue(array_equal(binary_image, expected_binary_image))


class TestSegmentWithLumValue(CellectsUnitTest):
    def test_segment_with_lum_value_lighter_background(self):
        converted_video = array([[[100, 120], [130, 140]], [[160, 170], [180, 200]]], dtype=uint8)
        basic_bckgrnd_values = array([110, 150], dtype=uint8)
        l_threshold = 180
        lighter_background = True
        expected_segmentation = array([[[1, 1], [1, 0]], [[1, 1], [0, 0]]], dtype=uint8)
        expected_l_threshold_over_time = array([140, 180], dtype=int64)

        segmentation, l_threshold_over_time = segment_with_lum_value(converted_video, basic_bckgrnd_values,
                                                                     l_threshold, lighter_background)
        self.assertTrue(array_equal(segmentation, expected_segmentation))
        self.assertTrue(array_equal(l_threshold_over_time, expected_l_threshold_over_time))

    def test_segment_with_lum_value_darker_background(self):
        converted_video = array([[[100, 120], [130, 140]], [[160, 170], [180, 200]]], dtype=uint8)
        basic_bckgrnd_values = array([110, 130], dtype=uint8)
        l_threshold = 150
        lighter_background = False
        expected_segmentation = array([[[0, 0], [0, 1]], [[1, 1], [1, 1]]], dtype=uint8)
        expected_l_threshold_over_time = array([130, 150], dtype=int64)

        segmentation, l_threshold_over_time = segment_with_lum_value(converted_video, basic_bckgrnd_values,
                                                                     l_threshold, lighter_background)
        self.assertTrue(array_equal(segmentation, expected_segmentation))
        self.assertTrue(array_equal(l_threshold_over_time, expected_l_threshold_over_time))


if __name__ == '__main__':
    unittest.main()
