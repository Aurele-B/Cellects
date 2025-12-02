#!/usr/bin/env python3
"""
Test suite for utility functions in utilitarian module.

This test suite covers core utilitarian functions including array comparison operations,
dictionary conversion, path abbreviation, nearest value lookup, and progress tracking.
Tests include unit tests and edge cases validation across different input types.
"""
import os
import unittest
from pathlib import Path

import numpy as np
from cellects.utils.utilitarian import *
from tests._base import CellectsUnitTest


class TestGreaterAlongFirstAxis(CellectsUnitTest):
    """Test suite for greater_along_first_axis function."""
    def test_greater_along_first_axis(self):
        """Test basic functionality."""
        test_cases = [
            {
                'array_in_1': np.array([[1, 2, 3],
                                        [4, 5, 6],
                                        [7, 8, 9]]),
                'array_in_2': np.array([3, 6, 9]),
                'expected_result': np.array([[False, False, False],
                                             [False, False, False],
                                             [False, False, False]])
            },
            {
                'array_in_1': np.array([[1, 2, 3],
                                        [4, 5, 6],
                                        [7, 8, 9]]),
                'array_in_2': np.array([0, 4, 7]),
                'expected_result': np.array([[True, True, True],
                                             [False, True, True],
                                             [False, True, True]])
            },
            {
                'array_in_1': np.array([[1, 2, 3],
                                        [4, 5, 6],
                                        [7, 8, 9]]),
                'array_in_2': np.array([0, 0, 0]),
                'expected_result': np.array([[True, True, True],
                                             [True, True, True],
                                             [True, True, True]])
            },
            {
                'array_in_1': np.array([[1.5, 2.7],
                                        [3.2, 4.8]]),
                'array_in_2': np.array([2.2, 4.4]),
                'expected_result': np.array([[False, True],
                                             [False, True]])
            }
        ]
        for test_case in test_cases:
            array_in_1 = test_case['array_in_1']
            array_in_2 = test_case['array_in_2']
            expected_result = test_case['expected_result']
            actual_result = greater_along_first_axis(array_in_1, array_in_2)
            self.assertEqual(actual_result.tolist(), expected_result.tolist())


class TestLessAlongFirstAxis(CellectsUnitTest):
    """Test suite for less_along_first_axis function."""
    def test_less_along_first_axis(self):
        """Test basic functionality."""
        test_cases = [
            {
                'array_in_1': np.array([[1, 2, 3],
                                        [4, 5, 6],
                                        [7, 8, 9]]),
                'array_in_2': np.array([3, 6, 9]),
                'expected_result': np.array([[True, True, False],
                                             [True, True, False],
                                             [True, True, False]])
            },
            {
                'array_in_1': np.array([[1, 2, 3],
                                        [4, 5, 6],
                                        [7, 8, 9]]),
                'array_in_2': np.array([0, 0, 0]),
                'expected_result': np.array([[False, False, False],
                                             [False, False, False],
                                             [False, False, False]])
            },
            {
                'array_in_1': np.array([[1, 2, 3],
                                        [4, 5, 6],
                                        [7, 8, 9]]),
                'array_in_2': np.array([10, 10, 10]),
                'expected_result': np.array([[True, True, True],
                                             [True, True, True],
                                             [True, True, True]])
            },
            {
                'array_in_1': np.array([[1.5, 2.7],
                                        [3.2, 4.8]]),
                'array_in_2': np.array([2.2, 4.4]),
                'expected_result': np.array([[True, False],
                                             [True, False]])
            }
        ]

        for test_case in test_cases:
            array_in_1 = test_case['array_in_1']
            array_in_2 = test_case['array_in_2']
            expected_result = test_case['expected_result']

            actual_result = less_along_first_axis(array_in_1, array_in_2)

            self.assertEqual(actual_result.tolist(), expected_result.tolist())


class TestTranslateDict(CellectsUnitTest):
    """Test suite for translate_dict function."""
    def test_translate_dict(self):
        """Test basic functionality."""
        old_dict = {'key1': 1, 'key2': 2, 'key3': 3}
        typed_dict = translate_dict(old_dict)
        self.assertEqual(len(typed_dict), len(old_dict))

    def test_translate_dict_wrong_value(self):
        """Test functionality with one str as value."""
        old_dict = {'key1': 1, 'key2': "2", 'key3': 3}

        typed_dict = translate_dict(old_dict)
        self.assertEqual(len(typed_dict), len(old_dict) - 1)


class TestReducePathLen(CellectsUnitTest):
    """Test suite for reduce_path_len function."""
    def test_reduce_path_len(self):
        """Test basic functionality."""
        pathway = "some/long/path/to/a/file.txt"
        to_start = 15
        from_end = 8

        result = reduce_path_len(pathway, to_start, from_end)

        self.assertIsInstance(result, str)
        self.assertLessEqual(len(result), to_start + from_end + 3)

        if len(pathway) > to_start + from_end + 3:
            expected_result = pathway[:to_start] + "..." + pathway[-from_end:]
        else:
            expected_result = pathway

        self.assertEqual(result, expected_result)
    def test_reduce_path_len_no_str(self):
        """Test basic functionality."""
        pathway = Path("some/long/path/to/a/file.txt")
        to_start = 15
        from_end = 8

        result = reduce_path_len(pathway, to_start, from_end)

        self.assertIsInstance(result, str)
        self.assertLessEqual(len(result), to_start + from_end + 3)


class TestFindNearest(CellectsUnitTest):
    """Test suite for find_nearest function."""
    def test_find_nearest(self):
        """Test basic functionality."""
        arr = [1, 4, 6, 9, 12]
        value = 7

        result = find_nearest(arr, value)

        self.assertEqual(result, 6)

        arr = np.array((0.5, 1.5, 2.5, 3.5, 4.5))
        value = 2.2

        result = find_nearest(arr, value)

        self.assertAlmostEqual(result, 2.5, places=6)


class TestPercentAndTimeTracker(CellectsUnitTest):
    """Test suite for PercentAndTimeTracker class."""
    def test_get_progress(self):
        """Test basic functionality."""
        total_iterations = 100
        tracker = PercentAndTimeTracker(total_iterations)
        progress, eta = tracker.get_progress()
        self.assertEqual(progress, 1)

    def test_get_progress_with_step_and_elements(self):
        """Test with with_elements_number and step."""
        total_iterations = 100
        tracker_with_elements_and_step = PercentAndTimeTracker(total_iterations, compute_with_elements_number=True)

        element = np.arange(100)
        for i in np.arange(total_iterations):
            progress, eta = tracker_with_elements_and_step.get_progress(step=i, element_number=element[i])
            if i == 0:
                self.assertEqual(progress, 0)
        self.assertEqual(progress, 100)


    def test_get_progress_with_step(self):
        """Test by setting step."""
        total_iterations = 100
        tracker_with_elements_and_step = PercentAndTimeTracker(total_iterations)

        for i in np.arange(total_iterations):
            progress, eta = tracker_with_elements_and_step.get_progress(step=i)
            if i == 0:
                self.assertEqual(progress, 1)
        self.assertEqual(progress, 100)

    def test_get_progress_with_elements(self):
        """Test with with_elements_number."""
        total_iterations = 100
        tracker_with_elements_and_step = PercentAndTimeTracker(total_iterations, compute_with_elements_number=True)

        element = np.arange(100)
        for i in np.arange(total_iterations):
            progress, eta = tracker_with_elements_and_step.get_progress(element_number=element[i])
            if i == 0:
                self.assertEqual(progress, 0)
        self.assertEqual(progress, 100)


class TestInsensitiveGlob(CellectsUnitTest):
    """Test suite for insensitive_glob function."""
    def test_insensitive_glob_prefix(self):
        """Test basic functionality."""
        os.chdir(self.path_input)
        result = insensitive_glob("Test_v" + "*")
        expected_result = ['test_vstack.h5']
        self.assertCountEqual(result, expected_result)

    def test_insensitive_glob_suffix(self):
        """Test with caps extension."""
        os.chdir(self.path_input)
        result = insensitive_glob("*.mP4")
        expected_result = ['test_read_video.mp4']
        self.assertCountEqual(result, expected_result)


class TestSmallestMemoryArray(CellectsUnitTest):
    """Test suite for smallest_memory_array function."""

    def test_smallest_memory_array_2d_list_uint8(self):
        """Test 2D list that fits in uint8."""
        input_array = [[1, 2], [3, 4]]
        expected_dtype = np.uint8
        result = smallest_memory_array(input_array)
        self.assertEqual(result.dtype, expected_dtype)
        self.assertTrue(np.array_equal(result, np.array([[1, 2], [3, 4]], dtype=np.uint8)))

    def test_smallest_memory_array_2d_list_uint16(self):
        """Test 2D list that fits in uint16."""
        input_array = [[1000, 2000], [3000, 4000]]
        expected_dtype = np.uint16
        result = smallest_memory_array(input_array)
        self.assertEqual(result.dtype, expected_dtype)
        self.assertTrue(np.array_equal(result, np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)))

    def test_smallest_memory_array_2d_list_uint32(self):
        """Test 2D list that fits in uint32."""
        input_array = [[100000, 200000], [300000, 400000]]
        expected_dtype = np.uint32
        result = smallest_memory_array(input_array)
        self.assertEqual(result.dtype, expected_dtype)
        self.assertTrue(np.array_equal(result, np.array([[100000, 200000], [300000, 400000]], dtype=np.uint32)))

    def test_smallest_memory_array_2d_list_uint64(self):
        """Test 2D list that fits in uint64."""
        input_array = [[2**31, 2**32], [2**33, 2**34]]
        expected_dtype = np.uint64
        result = smallest_memory_array(input_array)
        self.assertEqual(result.dtype, expected_dtype)
        self.assertTrue(np.array_equal(result,
                                      np.array([[2**31, 2**32], [2**33, 2**34]], dtype=np.uint64)))

    def test_smallest_memory_array_single_element(self):
        """Test single element array."""
        input_array = [[42]]
        expected_dtype = np.uint8
        result = smallest_memory_array(input_array)
        self.assertEqual(result.dtype, expected_dtype)
        self.assertTrue(np.array_equal(result, np.array([[42]], dtype=np.uint8)))

    def test_smallest_memory_array_numpy_ndarray_input(self):
        """Test with numpy ndarray input."""
        input_array = np.array([[100, 200], [300, 400]], dtype=np.int64)
        expected_dtype = np.uint16
        result = smallest_memory_array(input_array)
        self.assertEqual(result.dtype, expected_dtype)
        self.assertTrue(np.array_equal(result, np.array([[100, 200], [300, 400]], dtype=np.uint16)))

    def test_smallest_memory_array_large_values(self):
        """Test with values at the edge of uint64."""
        input_array = [[np.iinfo(np.uint64).max - 1, np.iinfo(np.uint64).max]]
        expected_dtype = np.uint64
        result = smallest_memory_array(input_array)
        self.assertEqual(result.dtype, expected_dtype)
        self.assertTrue(np.array_equal(result,
                                      np.array([[np.iinfo(np.uint64).max - 1, np.iinfo(np.uint64).max]],
                                               dtype=np.uint64)))

    def test_smallest_memory_array_zero_values(self):
        """Test with zero values."""
        input_array = [[0, 0], [0, 0]]
        expected_dtype = np.uint8
        result = smallest_memory_array(input_array)
        self.assertEqual(result.dtype, expected_dtype)
        self.assertTrue(np.array_equal(result, np.array([[0, 0], [0, 0]], dtype=np.uint8)))

    def test_smallest_memory_array_tuple(self):
        """Test with zero values."""
        img = np.array([[0,1,0],[0,1,0],[0,1,0]], dtype=np.uint8)
        array_object = np.nonzero(img)
        result = smallest_memory_array(array_object)
        expected_dtype = np.uint8
        self.assertEqual(result.dtype, expected_dtype)
        self.assertTrue(np.array_equal(result, np.array(array_object, dtype=np.uint8)))

    def test_smallest_memory_array_empty_tuple(self):
        """Test with zero values."""
        img = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=np.uint8)
        array_object = np.nonzero(img)
        result = smallest_memory_array(array_object)
        expected_dtype = np.uint8
        self.assertEqual(result.dtype, expected_dtype)
        self.assertTrue(np.array_equal(result, np.array(array_object, dtype=np.uint8)))


class TestRemoveCoordinates(CellectsUnitTest):
    """Test suite for remove_coordinates function."""

    def test_remove_coordinates_normal_case(self):
        """Test normal operation with some matching coordinates."""
        arr1 = np.array([[0, 0], [1, 2], [3, 4]])
        arr2 = np.array([[1, 2], [5, 6]])
        expected = np.array([[0, 0], [3, 4]])

        result = remove_coordinates(arr1, arr2)
        self.assertTrue(np.array_equal(result, expected))

    def test_remove_coordinates_no_matches(self):
        """Test when there are no matching coordinates."""
        arr1 = np.array([[0, 0], [1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])
        expected = np.array([[0, 0], [1, 2], [3, 4]])

        result = remove_coordinates(arr1, arr2)
        self.assertTrue(np.array_equal(result, expected))

    def test_remove_coordinates_all_match(self):
        """Test when all coordinates in arr1 match coordinates in arr2."""
        arr1 = np.array([[0, 0], [1, 2], [3, 4]])
        arr2 = np.array([[0, 0], [1, 2], [3, 4]])

        result = remove_coordinates(arr1, arr2)
        self.assertTrue(result.shape[0] == 0)

    def test_remove_coordinates_duplicates_in_arr1(self):
        """Test with duplicate coordinates in arr1."""
        arr1 = np.array([[0, 0], [1, 2], [1, 2], [3, 4]])
        arr2 = np.array([[1, 2], [5, 6]])
        expected = np.array([[0, 0], [3, 4]])

        result = remove_coordinates(arr1, arr2)
        self.assertTrue(np.array_equal(result, expected))

    def test_remove_coordinates_duplicates_in_arr2(self):
        """Test with duplicate coordinates in arr2."""
        arr1 = np.array([[0, 0], [1, 2], [3, 4]])
        arr2 = np.array([[1, 2], [1, 2], [5, 6]])
        expected = np.array([[0, 0], [3, 4]])

        result = remove_coordinates(arr1, arr2)
        self.assertTrue(np.array_equal(result, expected))

    def test_remove_coordinates_mixed_data_types(self):
        """Test with mixed data types (int and float)."""
        arr1 = np.array([[0, 0], [1.5, 2.3], [3, 4]])
        arr2 = np.array([[1.5, 2.3], [5, 6]])
        expected = np.array([[0, 0], [3, 4]])

        result = remove_coordinates(arr1, arr2)
        self.assertTrue(np.array_equal(result, expected))
    def test_remove_coordinates_wrong_shapes(self):
        """Test with mixed data types (int and float)."""
        arr1 = np.array([[0, 0, 0], [1.5, 2.3, 0], [3, 4, 0]])
        arr2 = np.array([[1.5, 2.3], [5, 6]])

        with self.assertRaises(ValueError):
            result = remove_coordinates(arr1, arr2)



if __name__ == '__main__':
    unittest.main()