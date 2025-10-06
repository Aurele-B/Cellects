#!/usr/bin/env python3
"""
This script contains all unit tests of the morphological_operations script
"""

import unittest

import numpy as np

from tests._base import CellectsUnitTest
from cellects.core.program_organizer import ProgramOrganizer
from cellects.core.one_video_per_blob import OneVideoPerBlob
from cellects.utils.load_display_save import PickleRick
from cellects.image_analysis.morphological_operations import *
from numpy import zeros, uint8, float32, random, array, testing, array_equal, allclose, int32, int64, diff, concatenate
from cv2 import imread, imwrite, rotate, ROTATE_90_COUNTERCLOCKWISE, ROTATE_90_CLOCKWISE, VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS
from numba import types


class TestCompareNeighborsWithValue(CellectsUnitTest):
    # Create a sample matrix for testing
    matrix = np.array([[9, 0, 4, 6],
                   [4, 9, 1, 3],
                   [7, 2, 1, 4],
                   [9, 0, 8, 5]], dtype=np.int8)

    def test_is_equal_connectivity_4(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_equal(1)
        expected_result = np.array([[0, 0, 1, 0],
                                 [0, 1, 1, 1],
                                 [0, 1, 1, 1],
                                 [0, 0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)
        comparer.is_equal(2)
        expected_result = np.array([[0, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [1, 0, 1, 0],
                                 [0, 1, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)

    def test_is_equal_connectivity_8(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_equal(1)
        expected_result = np.array([[0, 1, 1, 1],
                                 [0, 2, 1, 2],
                                 [0, 2, 1, 2],
                                 [0, 1, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)
        comparer.is_equal(2)
        expected_result = np.array([[0, 0, 0, 0],
                                 [1, 1, 1, 0],
                                 [1, 0, 1, 0],
                                 [1, 1, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)

    def test_is_equal_with_itself_connectivity_4(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_equal(1, and_itself=True)
        expected_result = np.array([[0, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)
        comparer.is_equal(2, and_itself=True)
        expected_result = np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)

    def test_is_equal_with_itself_connectivity_8(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_equal(1, and_itself=True)
        expected_result = np.array([[0, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)
        comparer.is_equal(2, and_itself=True)
        expected_result = np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)

    def test_is_sup_connectivity_4(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_sup(5)
        expected_result = np.array([[2, 2, 1, 2],
                                 [3, 0, 1, 1],
                                 [2, 2, 1, 0],
                                 [3, 2, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.sup_neighbor_nb, expected_result)

    def test_is_sup_connectivity_8(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_sup(5)
        expected_result = np.array([[4, 3, 3, 3],
                                 [5, 2, 2, 2],
                                 [4, 4, 2, 1],
                                 [5, 5, 1, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.sup_neighbor_nb, expected_result)

    def test_is_sup_with_itself_connectivity_4(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_sup(5, and_itself=True)
        expected_result = np.array([[2, 0, 0, 2],
                                 [0, 0, 0, 0],
                                 [2, 0, 0, 0],
                                 [3, 0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.sup_neighbor_nb, expected_result)
        comparer.is_sup(3, and_itself=True)
        expected_result = np.array([[3, 0, 2, 3],
                                 [4, 1, 0, 0],
                                 [3, 0, 0, 2],
                                 [3, 0, 2, 4]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.sup_neighbor_nb, expected_result)

    def test_is_sup_with_itself_connectivity_8(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_sup(5, and_itself=True)
        expected_result = np.array([[4, 0, 0, 3],
                                 [0, 2, 0, 0],
                                 [4, 0, 0, 0],
                                 [5, 0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.sup_neighbor_nb, expected_result)

    def test_is_inf_connectivity_4(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_inf(5)
        expected_result = np.array([[2, 2, 3, 2],
                                 [1, 4, 3, 3],
                                 [2, 2, 3, 3],
                                 [1, 2, 2, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.inf_neighbor_nb, expected_result)

    def test_is_inf_connectivity_8(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_inf(5)
        expected_result = np.array([[4, 5, 5, 5],
                                 [3, 6, 6, 6],
                                 [4, 4, 5, 5],
                                 [3, 3, 5, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.inf_neighbor_nb, expected_result)

    def test_is_inf_with_itself_connectivity_4(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_inf(5, and_itself=True)
        expected_result = np.array([[0, 2, 3, 0],
                                 [1, 0, 3, 3],
                                 [0, 2, 3, 3],
                                 [0, 2, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.inf_neighbor_nb, expected_result)

    def test_is_inf_with_itself_connectivity_8(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_inf(5, and_itself=True)
        expected_result = np.array([[0, 5, 5, 0],
                                 [3, 0, 6, 6],
                                 [0, 4, 5, 5],
                                 [0, 3, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(comparer.inf_neighbor_nb, expected_result)


class TestCC(CellectsUnitTest):
    binary_img = np.array([[1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1],
                        [0, 1, 1, 1, 1],
                        [0, 1, 1, 0, 0]], dtype=np.uint8)
    def test_cc_ordering(self):
        expected_order = np.array([[2, 2, 2, 2, 0],
                                [2, 0, 0, 0, 0],
                                [0, 0, 1, 0, 1],
                                [0, 1, 1, 1, 1],
                                [0, 1, 1, 0, 0]], dtype=np.uint8)
        expected_stats = np.array([[ 0,  0,  5,  5, 12],
                                [ 1,  2,  5,  5,  8],
                                [ 0,  0,  4,  2,  5]], dtype=int32)
        new_order, stats, centers = cc(self.binary_img)
        np.testing.assert_array_equal(new_order, expected_order)
        np.testing.assert_array_equal(stats, expected_stats)


class TestMakeGravityField(CellectsUnitTest):
    original_shape = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    def test_make_gravity_field_no_erosion_no_max_distance(self):
        expected_field = np.array([[1, 2, 3, 4, 3, 2, 1],
                                [2, 3, 4, 5, 4, 3, 2],
                                [3, 4, 5, 0, 5, 4, 3],
                                [4, 5, 0, 0, 0, 5, 4],
                                [3, 4, 5, 0, 5, 4, 3],
                                [2, 3, 4, 5, 4, 3, 2],
                                [1, 2, 3, 4, 3, 2, 1]], dtype=np.uint32)
        field = make_gravity_field(self.original_shape)
        np.testing.assert_array_equal(field, expected_field)

    def test_make_gravity_field_with_erosion(self):
        expected_field = np.array([[1, 2, 3, 4, 3, 2, 1],
                                [2, 3, 4, 5, 4, 3, 2],
                                [3, 4, 5, 6, 5, 4, 3],
                                [4, 5, 6, 0, 6, 5, 4],
                                [3, 4, 5, 6, 5, 4, 3],
                                [2, 3, 4, 5, 4, 3, 2],
                                [1, 2, 3, 4, 3, 2, 1]], dtype=np.uint8)
        field = make_gravity_field(self.original_shape, with_erosion=1)
        np.testing.assert_array_equal(field, expected_field)

    def test_make_gravity_field_with_max_distance(self):
        expected_field = np.array([[0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 2, 1, 0, 0],
                                [0, 1, 2, 0, 2, 1, 0],
                                [1, 2, 0, 0, 0, 2, 1],
                                [0, 1, 2, 0, 2, 1, 0],
                                [0, 0, 1, 2, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)
        field = make_gravity_field(self.original_shape, max_distance=2)
        np.testing.assert_array_equal(field, expected_field)


class TestGetEveryCoordBetween2Points(CellectsUnitTest):
    def test_get_every_coord_between_2_points_vertical(self):
        point_A = (0, 0)
        point_B = (4, 0)
        expected_segment = np.array([[0, 1, 2, 3, 4],
                                  [0, 0, 0, 0, 0]], dtype =np.uint64)
        segment = get_every_coord_between_2_points(point_A, point_B)
        np.testing.assert_array_equal(segment, expected_segment)

    def test_get_every_coord_between_2_points_horizontal(self):
        point_A = (0, 0)
        point_B = (0, 4)
        expected_segment = np.array([[0, 0, 0, 0, 0],
                                  [0, 1, 2, 3, 4]], dtype =np.uint64)
        segment = get_every_coord_between_2_points(point_A, point_B)
        np.testing.assert_array_equal(segment, expected_segment)

    def test_get_every_coord_between_2_points_diagonal(self):
        point_A = (0, 0)
        point_B = (4, 4)
        expected_segment = np.array([[0, 1, 2, 3, 4],
                                  [0, 1, 2, 3, 4]], dtype =np.uint64)
        segment = get_every_coord_between_2_points(point_A, point_B)
        np.testing.assert_array_equal(segment, expected_segment)

    def test_get_every_coord_between_2_points_reversed_diagonal(self):
        point_A = (0, 4)
        point_B = (4, 0)

        expected_segment = np.array([[0, 1, 2, 3, 4],
                                  [4, 3, 2, 1, 0]], dtype =np.uint64)
        segment = get_every_coord_between_2_points(point_A, point_B)
        np.testing.assert_array_equal(segment, expected_segment)
        

class TestDrawMeASun(CellectsUnitTest):
    main_shape = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    cross_33 = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=np.uint8)

    def test_draw_me_a_sun(self):
        expected_rays = np.arange(2, 14, dtype=np.uint32)
        expected_sun = np.array([[0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0],
                              [0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0],
                              [0,  0,  0,  3,  0,  2,  0,  4,  0,  0,  0],
                              [0,  0,  5,  0,  3,  0,  4,  0,  6,  0,  0],
                              [0,  0,  0,  5,  0,  0,  0,  6,  0,  0,  0],
                              [7,  7,  7,  0,  0,  0,  0,  0,  8,  8,  8],
                              [0,  0,  0,  9,  0,  0,  0, 10,  0,  0,  0],
                              [0,  0,  9,  0, 11,  0, 12,  0, 10,  0,  0],
                              [0,  0,  0, 11,  0, 13,  0, 12,  0,  0,  0],
                              [0,  0,  0,  0,  0, 13,  0,  0,  0,  0,  0],
                              [0,  0,  0,  0,  0, 13,  0,  0,  0,  0,  0]], dtype=np.uint32)
        rays, sun = draw_me_a_sun(self.main_shape, self.cross_33)
        np.testing.assert_array_equal(rays, expected_rays)
        np.testing.assert_array_equal(sun, expected_sun)



class TestFindMedianShape(CellectsUnitTest):
    def test_find_median_shape(self):
        binary_3d_matrix = np.array([[[1, 1, 0],
                                   [0, 1, 0],
                                   [1, 0, 1]],
                                  [[1, 0, 0],
                                   [0, 0, 1],
                                   [0, 1, 0]],
                                  [[0, 1, 0],
                                   [0, 0, 1],
                                   [0, 0, 0]],
                                  [[0, 1, 0],
                                   [1, 1, 1],
                                   [0, 1, 0]]], dtype=np.uint8)
        expected_median_shape = np.array([[1, 1, 0],
                                       [0, 1, 1],
                                       [0, 1, 0]], dtype=np.uint8)
        median_shape = find_median_shape(binary_3d_matrix)
        np.testing.assert_array_equal(median_shape, expected_median_shape)


class TestReduceImageSizeForSpeed(CellectsUnitTest):
    def test_reduce_image_size_for_speed(self):
        image_of_2_shapes = np.array([[1, 0, 1, 1],
                                   [2, 0, 2, 2],
                                   [1, 0, 1, 1],
                                   [1, 0, 2, 2]], dtype=np.uint8)
        expected_shape1_idx = (np.array([0, 0, 0, 2, 2, 2, 3], dtype=np.int64), np.array([0, 2, 3, 0, 2, 3, 0], dtype=np.int64))
        expected_shape2_idx = (np.array([1, 1, 1, 3, 3], dtype=np.int64), np.array([0, 2, 3, 2, 3], dtype=np.int64))
        shape1_idx, shape2_idx = reduce_image_size_for_speed(image_of_2_shapes)
        np.testing.assert_array_equal(shape1_idx, expected_shape1_idx)
        np.testing.assert_array_equal(shape2_idx, expected_shape2_idx)


class TestMinimalDistance(CellectsUnitTest):
    def test_minimal_distance(self):
        image_of_2_shapes = np.array([[0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0],
                                   [0, 0, 0, 2, 0],
                                   [0, 0, 0, 0, 0]])
        distance = get_minimal_distance_between_2_shapes(image_of_2_shapes, increase_speed=False)
        self.assertAlmostEqual(distance, 2.23606, places=3)

        image_of_2_shapes = np.array([[0, 0, 0, 0, 0],
                                   [0, 1, 0, 2, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0]])
        distance = get_minimal_distance_between_2_shapes(image_of_2_shapes, increase_speed=False)
        self.assertEqual(distance, 2.0)

        image_of_2_shapes = np.array([[0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 2]])
        distance = get_minimal_distance_between_2_shapes(image_of_2_shapes, increase_speed=False)
        self.assertAlmostEqual(distance, 3.605551, places=3)

    def test_minimal_distance_with_speed_increase(self):
        image_of_2_shapes = np.array([[0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0],
                                      [0, 0, 0, 2, 0],
                                      [0, 0, 0, 0, 0]])
        distance = get_minimal_distance_between_2_shapes(image_of_2_shapes, increase_speed=True)
        self.assertAlmostEqual(distance, 2.23606, places=3)

        image_of_2_shapes = np.array([[0, 0, 0, 0, 0],
                                   [0, 1, 0, 2, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0]])
        distance = get_minimal_distance_between_2_shapes(image_of_2_shapes, increase_speed=True)
        self.assertEqual(distance, 2.0)

        image_of_2_shapes = np.array([[0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 2]])
        distance = get_minimal_distance_between_2_shapes(image_of_2_shapes, increase_speed=True)
        self.assertAlmostEqual(distance, 3.605551, places=3)


class TestFindMajorIncline(CellectsUnitTest):
    def test_find_major_incline_failure(self):
        vector = np.concatenate((np.repeat(10, 50),array((20, 20)),array((40, 40)), np.repeat(50, 50)))
        natural_noise = 10
        left, right = find_major_incline(vector, natural_noise)
        self.assertEqual(left, 0)
        self.assertEqual(right, 1)


class TestRankFromTopToBottomFromLeftToRight(CellectsUnitTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.binary_image = np.array([[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                          [0,  0,  1,  0,  0,  0,  1,  0,  0,  1,  0],
                                          [0,  1,  1,  1,  0,  0,  1,  0,  1,  0,  0],
                                          [0,  0,  1,  0,  0,  1,  1,  0,  1,  1,  0],
                                          [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
                                          [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                          [0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0],
                                          [0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0],
                                          [0,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0],
                                          [0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0],
                                          [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=np.uint8)
        cls.y_boundaries = np.array([0,  1,  0,  0,  -1,  0,  1,  0,  0,  -1,  0], dtype=np.int8)

    def test_rank_from_top_to_bottom_from_left_to_right(self):
        ordered_stats, ordered_centroids, ordered_image = rank_from_top_to_bottom_from_left_to_right(self.binary_image, self.y_boundaries, get_ordered_image=True)
        self.assertTrue(len(np.unique(ordered_image)) == 7)
        self.assertTrue(ordered_centroids.shape[0] == 6)
        self.assertTrue(ordered_stats[:, 4].sum() == self.binary_image.sum())
        self.assertTrue(ordered_image[2, 2] == 1)
        self.assertTrue(ordered_image[2, 6] == 2)
        self.assertTrue(ordered_image[2, 8] == 3)
        self.assertTrue(ordered_image[6, 1] == 4)
        self.assertTrue(ordered_image[7, 5] == 5)
        self.assertTrue(ordered_image[7, 8] == 6)


class TestExpandUntilNeighborCenterGetsNearerThanOwn(CellectsUnitTest):
    def test_no_expansion(self):
        shape_to_expand = np.zeros((9, 9), dtype=np.uint8)
        shape_to_expand[5:8, 5:8] = 1
        without_shape_i = np.zeros((9, 9), dtype=np.uint8)
        without_shape_i[1:4, 1:4] = 1
        without_shape_i[1:4, 5:8] = 1
        without_shape_i[5:8, 1:4] = 1
        shape_original_centroid = [6, 6]
        ref_centroids = np.array([[2, 2], [6, 2], [2, 6]], dtype=int32)
        kernel = np.ones((3, 3), dtype=np.uint8)
        expanded_shape = expand_until_neighbor_center_gets_nearer_than_own(
            shape_to_expand, without_shape_i, shape_original_centroid, ref_centroids, kernel
        )
        expected_result = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8
        )
        self.assertTrue(np.array_equal(expanded_shape, expected_result))

    def test_expansion(self):
        shape_to_expand = np.zeros((10, 10), dtype=np.uint8)
        shape_to_expand[5:8, 6] = 1
        shape_to_expand[6, 5:8] = 1
        without_shape_i = np.zeros((10, 10), dtype=np.uint8)
        without_shape_i[1:4, 2] = 1
        without_shape_i[2, 1:4] = 1
        shape_original_centroid = [6, 6]
        ref_centroids = np.array([[4, 4], [2, 2]], dtype=int32)
        kernel = np.ones((3, 3), dtype=np.uint8)
        expanded_shape = expand_until_neighbor_center_gets_nearer_than_own(
            shape_to_expand, without_shape_i, shape_original_centroid, ref_centroids, kernel
        )
        expected_result = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8
        )
        self.assertTrue(np.array_equal(expanded_shape, expected_result))


class TestImageBorders(CellectsUnitTest):
    def test_image_borders(self):
        # Test 1: Verify borders for a 3x3 image
        dimensions = (3, 3)
        borders = image_borders(dimensions)
        expected_result = np.array(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]], dtype=uint8
        )
        self.assertTrue(np.array_equal(borders, expected_result))

    def test_image_borders_large(self):
        # Test 2: Verify borders for a larger image
        dimensions = (5, 7)
        borders = image_borders(dimensions)
        expected_result = np.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]], dtype=uint8
        )
        self.assertTrue(np.array_equal(borders, expected_result))


class TestGetRadiusDistance(CellectsUnitTest):
    def test_get_radius_distance(self):
        # Test: Verify radius distance for a large binary video
        binary_video = np.array(
            [[[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]],

             [[0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 1, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0]],

             [[0, 0, 1, 0, 0],
              [0, 1, 1, 1, 0],
              [1, 1, 1, 1, 1],
              [0, 1, 1, 1, 0],
              [0, 0, 1, 0, 0]]], dtype=uint8
        )
        field = make_gravity_field(binary_video[0, ...],  2)
        distance_against_time, time_start, time_end = get_radius_distance_against_time(binary_video, field)
        expected_distance = np.array([2., 1.], dtype=np.float32)
        expected_start = 1
        expected_end = 2
        self.assertTrue(np.array_equal(distance_against_time, expected_distance))
        self.assertEqual(time_start, expected_start)
        self.assertEqual(time_end, expected_end)


class TestExpandToFillHoles(CellectsUnitTest):
    def test_expand_to_fill_holes(self):
        # Test 1: Verify expansion to fill holes
        binary_video = np.array(
            [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
              [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
              [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
              [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
              [0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
              [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
              [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
              [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
              [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=uint8
        )
        holes = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8
        )
        cross_33 = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], dtype=uint8
        )
        expanded_video, holes_time_end, distance_against_time = expand_to_fill_holes(binary_video, holes)
        expected_expanded_video = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                                          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                          [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                          [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                          [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=np.uint8)
        expected_holes_time_end = 1
        expected_distance_against_time = np.array([2.], dtype=np.float32)
        self.assertTrue(np.array_equal(expanded_video, expected_expanded_video))
        self.assertEqual(holes_time_end, expected_holes_time_end)
        self.assertTrue(np.array_equal(distance_against_time, expected_distance_against_time))


class TestEllipse(unittest.TestCase):
    def test_create_circle(self):
        # Test 1: Verify the correctness of ellipse_fun to create a circle
        sizes = [9, 9]
        ellipse = Ellipse(sizes).create()
        expected_result = np.array([[False, False, False, False,  True, False, False, False, False],
                                 [False, False,  True,  True,  True,  True,  True, False, False],
                                 [False,  True,  True,  True,  True,  True,  True,  True, False],
                                 [False,  True,  True,  True,  True,  True,  True,  True, False],
                                 [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                                 [False,  True,  True,  True,  True,  True,  True,  True, False],
                                 [False,  True,  True,  True,  True,  True,  True,  True, False],
                                 [False, False,  True,  True,  True,  True,  True, False, False],
                                 [False, False, False, False,  True, False, False, False, False]])
        self.assertTrue(np.array_equal(ellipse, expected_result))

    def test_create_circle(self):
        # Test 2: Verify the correctness of ellipse_fun to create an ellipse
        sizes = [5, 7]
        ellipse = Ellipse(sizes).create()
        expected_result = np.array([[False, False, False,  True, False, False, False],
                                 [False,  True,  True,  True,  True,  True, False],
                                 [True,  True,  True,  True,  True,  True,  True],
                                 [False,  True,  True,  True,  True,  True, False],
                                 [False, False, False,  True, False, False, False]])
        self.assertTrue(np.array_equal(ellipse, expected_result))


if __name__ == '__main__':
    unittest.main()