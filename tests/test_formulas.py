#!/usr/bin/env python3
"""Test suite for statistical and geometric analysis tools used in numerical arrays.

This test module verifies functions from `cellects.utils.formulas` related to image processing,
statistical modeling, and coordinate transformations. Tests focus on basic functionality with edge cases
for accuracy validation across computer vision and scientific computing scenarios.

Each class includes unit tests covering typical and boundary conditions like negative values,
zero inputs, and non-uniform distributions.
"""

import cv2
import unittest
from cellects.utils.formulas import *
from cellects.utils.utilitarian import translate_dict
from tests._base import CellectsUnitTest


class TestSumOfAbsDifferences(CellectsUnitTest):
    """Test suite for sum_of_abs_differences function."""
    def test_sum_of_abs_differences(self):
        """Test basic functionality."""
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
    """Test suite for to_uint8 function."""
    def test_to_uint8(self):
        """Test basic functionality."""
        # Test case 1: Array with positive values
        an_array = np.array([1.5, 2.7, 3.9])
        result = to_uint8(an_array)
        expected_result = np.array([2, 3, 4], dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected_result))

        # Test case 2: Array with negative values
        an_array = np.array([-1.4, -2.7, -3.9])
        result = to_uint8(an_array)
        expected_result = np.array([255, 253, 252], dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected_result))

        # Test case 3: Array with zeros
        an_array = np.zeros(3)
        result = to_uint8(an_array)
        expected_result = np.zeros(3, dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected_result))


class TestBracketToUint8ImageContrast(CellectsUnitTest):
    """Test suite for bracket_to_uint8_image_contrast function."""
    def test_zeros_with_one(self):
        """Test only zeros except one 1."""
        image = np.zeros((3, 3))
        image[1, 1] = 1
        result = bracket_to_uint8_image_contrast(image)
        expected_result = np.array([[0,   0,   0],
                                 [0, 255,   0],
                                 [0,   0,   0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_zeros_with_twofivefive(self):
        """Test only zeros except one 255."""
        image = np.zeros((3, 3))
        image[1, 1] = 255
        result = bracket_to_uint8_image_contrast(image)
        expected_result = np.array([[0,   0,   0],
                                 [0, 255,   0],
                                 [0,   0,   0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_ones_with_zero(self):
        """Test only ones except one 0."""
        image = np.ones((3, 3))
        image[1, 1] = 0
        result = bracket_to_uint8_image_contrast(image)
        expected_result = np.array([[255, 255, 255],
                                 [255,   0, 255],
                                 [255, 255, 255]], dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_ones_with_twofivefive(self):
        """Test only ones except one 255."""
        image = np.ones((3, 3))
        image[1, 1] = 255
        result = bracket_to_uint8_image_contrast(image)
        expected_result = np.array([[0,   0,   0],
                                 [0, 255,   0],
                                 [0,   0,   0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_negative_with_higher(self):
        """Test higher values than 255."""
        image = - np.ones((3, 3)) * 3
        image[1, 1] = 600
        image[2, 1] = 300
        result = bracket_to_uint8_image_contrast(image)
        expected_result = np.array([[0,   0,   0],
                                 [0, 255,   0],
                                 [0, 128,   0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_complex(self):
        """Test float, negative and high values."""
        image = - np.ones((3, 3)) * 0.54
        image[1, 1] = -60
        image[2, 1] = 300
        image[2, 2] = 120
        result = bracket_to_uint8_image_contrast(image)
        expected_result = np.array([[ 42,  42,  42],
                                 [ 42,   0,  42],
                                 [ 42, 255, 128]], dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected_result))


class TestLinearModel(CellectsUnitTest):
    """Test suite for linear_model function."""
    def test_linear_model(self):
        """Test basic functionality."""
        # Test case 1: Positive slope and intercept
        x = np.array([1, 2, 3])
        a = 2
        b = 3
        result = linear_model(x, a, b)
        expected_result = np.array([5, 7, 9])
        self.assertTrue(np.array_equal(result, expected_result))

        # Test case 2: Negative slope and positive intercept
        x = np.array([4, 5, 6])
        a = -1.5
        b = 2.5
        result = linear_model(x, a, b)
        expected_result = np.array([-3.5, -5, -6.5])
        self.assertTrue(np.array_equal(result, expected_result))

        # Test case 3: Zero slope and intercept
        x = np.array([-1, 0, 1])
        a = 0
        b = 0
        result = linear_model(x, a, b)
        expected_result = np.array([0, 0, 0])
        self.assertTrue(np.array_equal(result, expected_result))


class TestGetPowerDists(CellectsUnitTest):
    """Test suite for get_power_dists function."""
    def test_get_power_dists(self):
        """Test basic functionality."""
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

        self.assertTrue(np.array_equal(xn, expected_xn))
        self.assertTrue(np.array_equal(yn, expected_yn))


class TestGetStandardDeviations(CellectsUnitTest):
    """Test suite for get_kurtosis function."""
    def test_get_standard_deviations(self):
        """Test basic functionality."""
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
    """Test suite for get_skewness function."""
    def test_get_skewness(self):
        """Test basic functionality."""
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
    """Test suite for get_kurtosis function."""
    def test_get_kurtosis(self):
        """Test basic functionality."""
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
    """Test suite for eudist function."""
    def test_get_inertia_axes(self):
        """Test basic functionality."""
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
    """Test suite for eudist function."""
    def test_eudist(self):
        """Test basic functionality."""
        v1 = [1, 2, 3]  # Coordinates of point 1
        v2 = [4, 5, 6]  # Coordinates of point 2

        expected_dist = np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2)
        actual_dist = eudist(v1, v2)

        self.assertEqual(actual_dist, expected_dist)


class TestMovingAverage(CellectsUnitTest):
    """Test suite for moving_average function."""
    def test_moving_average(self):
        """Test basic functionality."""
        vector = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Sample vector
        step = 3  # Step/window size

        expected_result = np.array([1. ,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5, 10])
        actual_result = moving_average(vector, step)

        self.assertEqual(actual_result.tolist(), expected_result.tolist())


class TestFindCommonCoord(CellectsUnitTest):
    """Test suite for find_common_coord function."""

    def test_find_common_coord_basic_case(self):
        """Test basic functionality with one matching row."""
        array1 = np.array([[1, 2], [3, 4]])
        array2 = np.array([[5, 6], [1, 2]])

        expected = np.array([True, False])
        result = find_common_coord(array1, array2)

        self.assertTrue(np.array_equal(result, expected))

    def test_find_common_coord_no_matches(self):
        """Test when there are no matching rows."""
        array1 = np.array([[1, 2], [3, 4]])
        array2 = np.array([[5, 6], [7, 8]])

        expected = np.array([False, False])
        result = find_common_coord(array1, array2)

        self.assertTrue(np.array_equal(result, expected))

    def test_find_common_coord_all_match(self):
        """Test when all rows match."""
        array1 = np.array([[1, 2], [3, 4]])
        array2 = np.array([[1, 2], [3, 4]])

        expected = np.array([True, True])
        result = find_common_coord(array1, array2)

        self.assertTrue(np.array_equal(result, expected))

    def test_find_common_coord_single_row(self):
        """Test with single row in one array."""
        array1 = np.array([[1, 2]])
        array2 = np.array([[5, 6], [1, 2]])

        expected = np.array([True])
        result = find_common_coord(array1, array2)

        self.assertTrue(np.array_equal(result, expected))

    def test_find_common_coord_larger_arrays(self):
        """Test with larger arrays having partial matches."""
        array1 = np.array([[1, 2], [3, 4], [5, 6]])
        array2 = np.array([[1, 2], [7, 8], [3, 4]])

        expected = np.array([True,  True, False])
        result = find_common_coord(array1, array2)

        self.assertTrue(np.array_equal(result, expected))


class TestFindDuplicatesCoord(CellectsUnitTest):
    """Test suite for find_duplicates_coord function."""

    def test_find_duplicates_with_mixed_duplicates(self):
        """Test that find_duplicates_coord correctly identifies mixed duplicates."""
        input_array = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
        expected = np.array([True, False, True, False])
        result = find_duplicates_coord(input_array)
        self.assertTrue(np.array_equal(result, expected))

    def test_find_duplicates_all_unique(self):
        """Test that find_duplicates_coord returns all False for unique rows."""
        input_array = np.array([[1, 2], [3, 4], [5, 6]])
        expected = np.array([False, False, False])
        result = find_duplicates_coord(input_array)
        self.assertTrue(np.array_equal(result, expected))

    def test_find_duplicates_all_duplicates(self):
        """Test that find_duplicates_coord returns all True for duplicate rows."""
        input_array = np.array([[1, 2], [1, 2], [1, 2]])
        expected = np.array([True, True, True])
        result = find_duplicates_coord(input_array)
        self.assertTrue(np.array_equal(result, expected))

    def test_find_duplicates_empty_array(self):
        """Test that find_duplicates_coord handles empty arrays."""
        input_array = np.array([])
        expected = np.array([], dtype=bool)
        result = find_duplicates_coord(input_array)
        self.assertTrue(np.array_equal(result, expected))

    def test_find_duplicates_single_row(self):
        """Test that find_duplicates_coord handles single row arrays."""
        input_array = np.array([[1, 2]])
        expected = np.array([False])
        result = find_duplicates_coord(input_array)
        self.assertTrue(np.array_equal(result, expected))

    def test_find_duplicates_different_shapes(self):
        """Test that find_duplicates_coord works with different row shapes."""
        input_array = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
        expected = np.array([True, False, True])
        result = find_duplicates_coord(input_array)
        self.assertTrue(np.array_equal(result, expected))


if __name__ == '__main__':
    unittest.main()