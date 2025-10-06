#!/usr/bin/env python3
"""
This script contains all unit tests of the utilitarian script
10 tests
"""
import os
import unittest
from tests._base import CellectsUnitTest
from cellects.utils.utilitarian import *
from numpy import zeros, uint8, float32, random, array, testing, array_equal, allclose
from cv2 import imwrite, rotate, ROTATE_90_COUNTERCLOCKWISE, ROTATE_90_CLOCKWISE, VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS
from numba import types


class TestGreaterAlongFirstAxis(CellectsUnitTest):
    def test_greater_along_first_axis(self):
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
            }
        ]

        for test_case in test_cases:
            array_in_1 = test_case['array_in_1']
            array_in_2 = test_case['array_in_2']
            expected_result = test_case['expected_result']

            actual_result = greater_along_first_axis(array_in_1, array_in_2)

            self.assertEqual(actual_result.tolist(), expected_result.tolist())



class TestLessAlongFirstAxis(CellectsUnitTest):
    def test_less_along_first_axis(self):
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
            }
        ]

        for test_case in test_cases:
            array_in_1 = test_case['array_in_1']
            array_in_2 = test_case['array_in_2']
            expected_result = test_case['expected_result']

            actual_result = less_along_first_axis(array_in_1, array_in_2)

            self.assertEqual(actual_result.tolist(), expected_result.tolist())


class TestTranslateDict(CellectsUnitTest):

    def test_translate_dict(self):
        old_dict = {'key1': 1, 'key2': 2, 'key3': 3}

        typed_dict = translate_dict(old_dict)
        self.assertEqual(len(typed_dict), len(old_dict))

        for key, value in old_dict.items():
            self.assertIn(key, typed_dict)
            self.assertEqual(typed_dict[key], value)


class TestReducePathLen(CellectsUnitTest):
    def test_reduce_path_len(self):
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


class TestFindNearest(CellectsUnitTest):
    def test_find_nearest(self):
        arr = [1, 4, 6, 9, 12]
        value = 7

        result = find_nearest(arr, value)

        self.assertEqual(result, 6)

        arr = np.array((0.5, 1.5, 2.5, 3.5, 4.5))
        value = 2.2

        result = find_nearest(arr, value)

        self.assertAlmostEqual(result, 2.5, places=6)


class TestPercentAndTimeTracker(CellectsUnitTest):
    def test_get_progress(self):
        total_iterations = 100
        tracker = PercentAndTimeTracker(total_iterations)
        progress, eta = tracker.get_progress()
        self.assertEqual(progress, 1)

    def test_get_progress_with_step_and_elements(self):
        total_iterations = 100
        tracker_with_elements_and_step = PercentAndTimeTracker(total_iterations, compute_with_elements_number=True)

        element = np.arange(100)
        for i in np.arange(total_iterations):
            progress, eta = tracker_with_elements_and_step.get_progress(step=i, element_number=element[i])
            if i == 0:
                self.assertEqual(progress, 0)
        self.assertEqual(progress, 100)


    def test_get_progress_with_step(self):
        total_iterations = 100
        tracker_with_elements_and_step = PercentAndTimeTracker(total_iterations)

        for i in np.arange(total_iterations):
            progress, eta = tracker_with_elements_and_step.get_progress(step=i)
            if i == 0:
                self.assertEqual(progress, 1)
        self.assertEqual(progress, 100)

    def test_get_progress_with_elements(self):
        total_iterations = 100
        tracker_with_elements_and_step = PercentAndTimeTracker(total_iterations, compute_with_elements_number=True)

        element = np.arange(100)
        for i in np.arange(total_iterations):
            progress, eta = tracker_with_elements_and_step.get_progress(element_number=element[i])
            if i == 0:
                self.assertEqual(progress, 0)
        self.assertEqual(progress, 100)


class TestInsensitiveGlob(CellectsUnitTest):
    def test_insensitive_glob_prefix(self):
        os.chdir(self.path_input)
        result = insensitive_glob("Last" + "*")
        expected_result = ['last_binary_img.tif', 'last_original_img.tif']
        self.assertCountEqual(result, expected_result)

    def test_insensitive_glob_suffix(self):
        os.chdir(self.path_input)
        result = insensitive_glob("*.TIF")
        expected_result = ['last_binary_img.tif', 'last_original_img.tif']
        self.assertCountEqual(result, expected_result)





if __name__ == '__main__':
    unittest.main()