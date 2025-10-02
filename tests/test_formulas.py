#!/usr/bin/env python3
"""
This script contains all unit tests of the formulas script
14 tests
"""
import unittest

import cv2
import numpy as np
from cellects.utils.formulas import *
from cellects.utils.utilitarian import translate_dict
from tests._base import CellectsUnitTest


class TestSumOfAbsDifferences(CellectsUnitTest):

    def test_sum_of_abs_differences(self):
        # Test case 1: Arrays with same values
        array1 = np.array([1, 2, 3])
        array2 = np.array([1, 2, 3])
        result = sum_of_abs_differences(array1, array2)
        expected_result = 0
        self.assertEqual(result, expected_result)

        # Test case 2: Arrays with different values
        array1 = np.array([1, 2, 3])
        array2 = np.array([4, 5, 6])
        result = sum_of_abs_differences(array1, array2)
        expected_result = 9
        self.assertEqual(result, expected_result)

        # Test case 3: Arrays with negative values
        array1 = np.array([-1, -2, -3])
        array2 = np.array([1, 2, 3])
        result = sum_of_abs_differences(array1, array2)
        expected_result = np.sum(np.abs(array1 - array2))
        self.assertEqual(result, expected_result)


class TestToUint8(CellectsUnitTest):

    def test_to_uint8(self):
        # Test case 1: Array with positive values
        an_array = np.array([1.5, 2.7, 3.9])
        result = to_uint8(an_array)
        expected_result = np.array([2, 3, 4], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_result)

        # Test case 2: Array with negative values
        an_array = np.array([-1.4, -2.7, -3.9])
        result = to_uint8(an_array)
        expected_result = np.array([255, 253, 252], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_result)

        # Test case 3: Array with zeros
        an_array = np.zeros(3)
        result = to_uint8(an_array)
        expected_result = np.zeros(3, dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_result)


class TestBracketToUint8ImageContrast(CellectsUnitTest):

    def test_zeros_with_one(self):
        image = np.zeros((3, 3))
        image[1, 1] = 1
        result = bracket_to_uint8_image_contrast(image)
        expected_result = np.array([[0,   0,   0],
                                 [0, 255,   0],
                                 [0,   0,   0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_result)

    def test_zeros_with_twofivefive(self):
        image = np.zeros((3, 3))
        image[1, 1] = 255
        result = bracket_to_uint8_image_contrast(image)
        expected_result = np.array([[0,   0,   0],
                                 [0, 255,   0],
                                 [0,   0,   0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_result)

    def test_ones_with_zero(self):
        image = np.ones((3, 3))
        image[1, 1] = 0
        result = bracket_to_uint8_image_contrast(image)
        expected_result = np.array([[255, 255, 255],
                                 [255,   0, 255],
                                 [255, 255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_result)

    def test_ones_with_twofivefive(self):
        image = np.ones((3, 3))
        image[1, 1] = 255
        result = bracket_to_uint8_image_contrast(image)
        expected_result = np.array([[0,   0,   0],
                                 [0, 255,   0],
                                 [0,   0,   0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_result)

    def test_negative_with_higher(self):
        image = - np.ones((3, 3)) * 3
        image[1, 1] = 600
        image[2, 1] = 300
        result = bracket_to_uint8_image_contrast(image)
        expected_result = np.array([[0,   0,   0],
                                 [0, 255,   0],
                                 [0, 128,   0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_result)

    def test_complex(self):
        image = - np.ones((3, 3)) * 0.54
        image[1, 1] = -60
        image[2, 1] = 300
        image[2, 2] = 120
        result = bracket_to_uint8_image_contrast(image)
        expected_result = np.array([[ 42,  42,  42],
                                 [ 42,   0,  42],
                                 [ 42, 255, 128]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_result)


class TestLinearModel(CellectsUnitTest):

    def test_linear_model(self):
        # Test case 1: Positive slope and intercept
        x = np.array([1, 2, 3])
        a = 2
        b = 3
        result = linear_model(x, a, b)
        expected_result = np.array([5, 7, 9])
        np.testing.assert_array_equal(result, expected_result)

        # Test case 2: Negative slope and positive intercept
        x = np.array([4, 5, 6])
        a = -1.5
        b = 2.5
        result = linear_model(x, a, b)
        expected_result = np.array([-3.5, -5, -6.5])
        np.testing.assert_array_equal(result, expected_result)

        # Test case 3: Zero slope and intercept
        x = np.array([-1, 0, 1])
        a = 0
        b = 0
        result = linear_model(x, a, b)
        expected_result = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected_result)


class TestGetPowerDists(CellectsUnitTest):

    def test_get_power_dists(self):
        binary_image = np.array([[1, 1, 0, 0, 0],
                                 [1, 1, 1, 0, 0],
                                 [0, 1, 1, 1, 0],
                                 [0, 0, 1, 1, 1]], dtype=np.uint8)
        cx = 2
        cy = 1
        n = 2
        xn, yn = get_power_dists(binary_image, cx, cy, n)
        expected_xn = np.array([4, 1, 0, 1, 4])
        expected_yn = np.array([1, 0, 1, 4])

        np.testing.assert_array_equal(xn, expected_xn)
        np.testing.assert_array_equal(yn, expected_yn)


class TestGetStandardDeviations(CellectsUnitTest):

    def test_get_standard_deviations(self):
        binary_image = np.array([[1, 1, 0, 0, 0],
                                 [1, 1, 1, 0, 0],
                                 [0, 1, 1, 1, 0],
                                 [0, 0, 1, 1, 1]], dtype=np.uint8)
        mo = translate_dict(cv2.moments(binary_image))
        nb, output, stats, centroids = cv2.connectedComponentsWithStats(binary_image, cv2.CV_16U)
        cy = centroids[1, 0]
        cx = centroids[1, 1]

        x2, y2 = get_power_dists(binary_image, cx, cy, 2)
        X2, Y2 = np.meshgrid(x2, y2)
        expected_vx, expected_vy = get_var(mo, binary_image, X2, Y2)
        expected_std_x = np.sqrt(expected_vx)
        expected_std_y = np.sqrt(expected_vy)

        std_x, std_y = get_standard_deviations(mo, binary_image, cx, cy)

        self.assertEqual(std_x, expected_std_x)
        self.assertEqual(std_y, expected_std_y)


class TestGetSkewness(CellectsUnitTest):
    def test_get_skewness(self):
        binary_image = np.array([[1, 1, 0, 0, 0],
                              [1, 1, 1, 0, 0],
                              [0, 1, 1, 1, 0],
                              [0, 0, 1, 1, 1]], dtype=np.uint8)
        mo = translate_dict(cv2.moments(binary_image))
        nb, output, stats, centroids = cv2.connectedComponentsWithStats(binary_image, cv2.CV_16U)
        cy = centroids[1, 0]
        cx = centroids[1, 1]

        sx, sy = get_standard_deviations(mo, binary_image, cx, cy)

        x3, y3 = get_power_dists(binary_image, cx, cy, 3)
        X3, Y3 = np.meshgrid(x3, y3)
        expected_m3x, expected_m3y = get_var(mo, binary_image, X3, Y3)

        expected_x_skewness, expected_y_skewness = get_skewness_kurtosis(expected_m3x, expected_m3y, sx, sy, 3)

        x_skewness, y_skewness = get_skewness(mo, binary_image, cx, cy, sx, sy)

        self.assertEqual(x_skewness, expected_x_skewness)
        self.assertEqual(y_skewness, expected_y_skewness)


class TestGetKurtosis(CellectsUnitTest):

    def test_get_kurtosis(self):
        binary_image = np.array([[1, 1, 0, 0, 0],
                                 [1, 1, 1, 0, 0],
                                 [0, 1, 1, 1, 0],
                                 [0, 0, 1, 1, 1]], dtype=np.uint8)
        mo = translate_dict(cv2.moments(binary_image))
        nb, output, stats, centroids = cv2.connectedComponentsWithStats(binary_image, cv2.CV_16U)
        cy = centroids[1, 0]
        cx = centroids[1, 1]
        sx, sy = get_standard_deviations(mo, binary_image, cx, cy)

        x4, y4 = get_power_dists(binary_image, cx, cy, 4)
        X4, Y4 = np.meshgrid(x4, y4)
        expected_m4x, expected_m4y = get_var(mo, binary_image, X4, Y4)

        expected_x_kurtosis, expected_y_kurtosis = get_skewness_kurtosis(expected_m4x, expected_m4y, sx, sy, 4)

        x_kurtosis, y_kurtosis = get_kurtosis(mo, binary_image, cx, cy, sx, sy)

        self.assertEqual(x_kurtosis, expected_x_kurtosis)
        self.assertEqual(y_kurtosis, expected_y_kurtosis)


class TestGetInertiaAxes(CellectsUnitTest):
    def test_get_inertia_axes(self):
        binary_image = np.array([[1, 1, 0, 0, 0],
                              [1, 1, 1, 0, 0],
                              [0, 1, 1, 1, 0],
                              [0, 0, 1, 1, 1]], dtype=np.uint8)
        mo = translate_dict(cv2.moments(binary_image))

        cx, cy, major_axis_len, minor_axis_len, axes_orientation = get_inertia_axes(mo)

        # Calculate expected values
        c20 = (mo["m20"] / mo["m00"]) - np.square(cx)
        c02 = (mo["m02"] / mo["m00"]) - np.square(cy)
        c11 = (mo["m11"] / mo["m00"]) - (cx * cy)
        expected_major_axis_len = np.sqrt(6 * (c20 + c02 + np.sqrt(np.square(2 * c11) + np.square(c20 - c02))))
        expected_minor_axis_len = np.sqrt(6 * (c20 + c02 - np.sqrt(np.square(2 * c11) + np.square(c20 - c02))))
        if (c20 - c02) != 0:
            expected_axes_orientation = (0.5 * np.arctan((2 * c11) / (c20 - c02))) + ((c20 < c02) * (np.pi / 2))
        else:
            expected_axes_orientation = 0.0

        self.assertEqual(major_axis_len, expected_major_axis_len)
        self.assertEqual(minor_axis_len, expected_minor_axis_len)
        self.assertEqual(axes_orientation, expected_axes_orientation)


class TestEuclideanDistance(CellectsUnitTest):
    def test_eudist(self):
        v1 = [1, 2, 3]  # Coordinates of point 1
        v2 = [4, 5, 6]  # Coordinates of point 2

        expected_dist = np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2)
        actual_dist = eudist(v1, v2)

        self.assertEqual(actual_dist, expected_dist)


class TestCartesianToPolar(CellectsUnitTest):
    def test_cart2pol(self):
        x = 3  # Coordinate over the x axis
        y = 4  # Coordinate over the y axis

        expected_rho = np.sqrt(x ** 2 + y ** 2)
        expected_phi = np.arctan2(y, x)
        actual_rho, actual_phi = cart2pol(x, y)

        self.assertEqual(actual_rho, expected_rho)
        self.assertEqual(actual_phi, expected_phi)


class TestPolarToCartesian(CellectsUnitTest):
    def test_pol2cart(self):
        rho = 5  # Distance
        phi = 1.2  # Angle

        expected_x = rho * np.cos(phi)
        expected_y = rho * np.sin(phi)
        actual_x, actual_y = pol2cart(rho, phi)

        self.assertEqual(actual_x, expected_x)
        self.assertEqual(actual_y, expected_y)


class TestCohenD(CellectsUnitTest):
    def test_cohen_d(self):
        vector_1 = [1, 2, 3, 4, 5]  # Sample vector 1
        vector_2 = [2, 4, 6, 8, 10]  # Sample vector 2

        expected_d = (np.mean(vector_2) - np.mean(vector_1)) / np.sqrt(
            ((len(vector_2) - 1) * np.std(vector_2) ** 2 + (len(vector_1) - 1) * np.std(vector_1) ** 2) / (
                        len(vector_1) + len(vector_2) - 2))
        actual_d = cohen_d(vector_1, vector_2)

        self.assertEqual(actual_d, expected_d)


class TestMovingAverage(CellectsUnitTest):
    def test_moving_average(self):
        vector = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Sample vector
        step = 3  # Step/window size

        expected_result = np.array([1. ,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5, 10])
        actual_result = moving_average(vector, step)

        self.assertEqual(actual_result.tolist(), expected_result.tolist())


if __name__ == '__main__':
    unittest.main()
