#!/usr/bin/env python3
"""
This script contains all unit tests of the morphological_operations script
"""

import unittest
from cellects.test.cellects_unit_test import CellectsUnitTest
from cellects.core.program_organizer import ProgramOrganizer
from cellects.core.one_video_per_blob import OneVideoPerBlob
from cellects.core.cellects_paths import TEST_DIR
from cellects.utils.load_display_save import PickleRick
from cellects.image_analysis.morphological_operations import *
from numpy import zeros, uint8, float32, random, array, testing, array_equal, allclose, int32, int64, diff, concatenate
from cv2 import imread, imwrite, rotate, ROTATE_90_COUNTERCLOCKWISE, ROTATE_90_CLOCKWISE, VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS
from numba import types


class TestCompareNeighborsWithValue(CellectsUnitTest):
    # Create a sample matrix for testing
    matrix = array([[9, 0, 4, 6],
                   [4, 9, 1, 3],
                   [7, 2, 1, 4],
                   [9, 0, 8, 5]], dtype=int8)

    def test_is_equal_connectivity_4(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_equal(1)
        expected_result = array([[0, 0, 1, 0],
                                 [0, 1, 1, 1],
                                 [0, 1, 1, 1],
                                 [0, 0, 1, 0]], dtype=uint8)
        testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)
        comparer.is_equal(2)
        expected_result = array([[0, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [1, 0, 1, 0],
                                 [0, 1, 0, 0]], dtype=uint8)
        testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)

    def test_is_equal_connectivity_8(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_equal(1)
        expected_result = array([[0, 1, 1, 1],
                                 [0, 2, 1, 2],
                                 [0, 2, 1, 2],
                                 [0, 1, 1, 1]], dtype=uint8)
        testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)
        comparer.is_equal(2)
        expected_result = array([[0, 0, 0, 0],
                                 [1, 1, 1, 0],
                                 [1, 0, 1, 0],
                                 [1, 1, 1, 0]], dtype=uint8)
        testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)

    def test_is_equal_with_itself_connectivity_4(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_equal(1, and_itself=True)
        expected_result = array([[0, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 0]], dtype=uint8)
        testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)
        comparer.is_equal(2, and_itself=True)
        expected_result = array([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]], dtype=uint8)
        testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)

    def test_is_equal_with_itself_connectivity_8(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_equal(1, and_itself=True)
        expected_result = array([[0, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 0]], dtype=uint8)
        testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)
        comparer.is_equal(2, and_itself=True)
        expected_result = array([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]], dtype=uint8)
        testing.assert_array_equal(comparer.equal_neighbor_nb, expected_result)

    def test_is_sup_connectivity_4(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_sup(5)
        expected_result = array([[2, 2, 1, 2],
                                 [3, 0, 1, 1],
                                 [2, 2, 1, 0],
                                 [3, 2, 1, 1]], dtype=uint8)
        testing.assert_array_equal(comparer.sup_neighbor_nb, expected_result)

    def test_is_sup_connectivity_8(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_sup(5)
        expected_result = array([[4, 3, 3, 3],
                                 [5, 2, 2, 2],
                                 [4, 4, 2, 1],
                                 [5, 5, 1, 2]], dtype=uint8)
        testing.assert_array_equal(comparer.sup_neighbor_nb, expected_result)

    def test_is_sup_with_itself_connectivity_4(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_sup(5, and_itself=True)
        expected_result = array([[2, 0, 0, 2],
                                 [0, 0, 0, 0],
                                 [2, 0, 0, 0],
                                 [3, 0, 1, 0]], dtype=uint8)
        testing.assert_array_equal(comparer.sup_neighbor_nb, expected_result)
        comparer.is_sup(3, and_itself=True)
        expected_result = array([[3, 0, 2, 3],
                                 [4, 1, 0, 0],
                                 [3, 0, 0, 2],
                                 [3, 0, 2, 4]], dtype=uint8)
        testing.assert_array_equal(comparer.sup_neighbor_nb, expected_result)

    def test_is_sup_with_itself_connectivity_8(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_sup(5, and_itself=True)
        expected_result = array([[4, 0, 0, 3],
                                 [0, 2, 0, 0],
                                 [4, 0, 0, 0],
                                 [5, 0, 1, 0]], dtype=uint8)
        testing.assert_array_equal(comparer.sup_neighbor_nb, expected_result)

    def test_is_inf_connectivity_4(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_inf(5)
        expected_result = array([[2, 2, 3, 2],
                                 [1, 4, 3, 3],
                                 [2, 2, 3, 3],
                                 [1, 2, 2, 1]], dtype=uint8)
        testing.assert_array_equal(comparer.inf_neighbor_nb, expected_result)

    def test_is_inf_connectivity_8(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_inf(5)
        expected_result = array([[4, 5, 5, 5],
                                 [3, 6, 6, 6],
                                 [4, 4, 5, 5],
                                 [3, 3, 5, 3]], dtype=uint8)
        testing.assert_array_equal(comparer.inf_neighbor_nb, expected_result)

    def test_is_inf_with_itself_connectivity_4(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_inf(5, and_itself=True)
        expected_result = array([[0, 2, 3, 0],
                                 [1, 0, 3, 3],
                                 [0, 2, 3, 3],
                                 [0, 2, 0, 0]], dtype=uint8)
        testing.assert_array_equal(comparer.inf_neighbor_nb, expected_result)

    def test_is_inf_with_itself_connectivity_8(self):
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_inf(5, and_itself=True)
        expected_result = array([[0, 5, 5, 0],
                                 [3, 0, 6, 6],
                                 [0, 4, 5, 5],
                                 [0, 3, 0, 0]], dtype=uint8)
        testing.assert_array_equal(comparer.inf_neighbor_nb, expected_result)


class TestCC(CellectsUnitTest):
    binary_img = array([[1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1],
                        [0, 1, 1, 1, 1],
                        [0, 1, 1, 0, 0]], dtype=uint8)
    def test_cc_ordering(self):
        expected_order = array([[2, 2, 2, 2, 0],
                                [2, 0, 0, 0, 0],
                                [0, 0, 1, 0, 1],
                                [0, 1, 1, 1, 1],
                                [0, 1, 1, 0, 0]], dtype=uint8)
        expected_stats = array([[ 0,  0,  5,  5, 12],
                                [ 1,  2,  5,  5,  8],
                                [ 0,  0,  4,  2,  5]], dtype=int32)
        new_order, stats, centers = cc(self.binary_img)
        testing.assert_array_equal(new_order, expected_order)
        testing.assert_array_equal(stats, expected_stats)


class TestMakeGravityField(CellectsUnitTest):
    original_shape = array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    def test_make_gravity_field_no_erosion_no_max_distance(self):
        expected_field = array([[1, 2, 3, 4, 3, 2, 1],
                                [2, 3, 4, 5, 4, 3, 2],
                                [3, 4, 5, 0, 5, 4, 3],
                                [4, 5, 0, 0, 0, 5, 4],
                                [3, 4, 5, 0, 5, 4, 3],
                                [2, 3, 4, 5, 4, 3, 2],
                                [1, 2, 3, 4, 3, 2, 1]], dtype=uint32)
        field = make_gravity_field(self.original_shape)
        testing.assert_array_equal(field, expected_field)

    def test_make_gravity_field_with_erosion(self):
        expected_field = array([[1, 2, 3, 4, 3, 2, 1],
                                [2, 3, 4, 5, 4, 3, 2],
                                [3, 4, 5, 6, 5, 4, 3],
                                [4, 5, 6, 0, 6, 5, 4],
                                [3, 4, 5, 6, 5, 4, 3],
                                [2, 3, 4, 5, 4, 3, 2],
                                [1, 2, 3, 4, 3, 2, 1]], dtype=uint8)
        field = make_gravity_field(self.original_shape, with_erosion=1)
        testing.assert_array_equal(field, expected_field)

    def test_make_gravity_field_with_max_distance(self):
        expected_field = array([[0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 2, 1, 0, 0],
                                [0, 1, 2, 0, 2, 1, 0],
                                [1, 2, 0, 0, 0, 2, 1],
                                [0, 1, 2, 0, 2, 1, 0],
                                [0, 0, 1, 2, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0]], dtype=uint8)
        field = make_gravity_field(self.original_shape, max_distance=2)
        testing.assert_array_equal(field, expected_field)


class TestGetEveryCoordBetween2Points(CellectsUnitTest):
    def test_get_every_coord_between_2_points_vertical(self):
        point_A = (0, 0)
        point_B = (4, 0)
        expected_segment = array([[0, 1, 2, 3, 4],
                                  [0, 0, 0, 0, 0]], dtype=uint64)
        segment = get_every_coord_between_2_points(point_A, point_B)
        testing.assert_array_equal(segment, expected_segment)

    def test_get_every_coord_between_2_points_horizontal(self):
        point_A = (0, 0)
        point_B = (0, 4)
        expected_segment = array([[0, 0, 0, 0, 0],
                                  [0, 1, 2, 3, 4]], dtype=uint64)
        segment = get_every_coord_between_2_points(point_A, point_B)
        testing.assert_array_equal(segment, expected_segment)

    def test_get_every_coord_between_2_points_diagonal(self):
        point_A = (0, 0)
        point_B = (4, 4)
        expected_segment = array([[0, 1, 2, 3, 4],
                                  [0, 1, 2, 3, 4]], dtype=uint64)
        segment = get_every_coord_between_2_points(point_A, point_B)
        testing.assert_array_equal(segment, expected_segment)

    def test_get_every_coord_between_2_points_reversed_diagonal(self):
        point_A = (0, 4)
        point_B = (4, 0)

        expected_segment = array([[0, 1, 2, 3, 4],
                                  [4, 3, 2, 1, 0]], dtype=uint64)
        segment = get_every_coord_between_2_points(point_A, point_B)
        testing.assert_array_equal(segment, expected_segment)
        

class TestDrawMeASun(CellectsUnitTest):
    main_shape = array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    cross_33 = array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=uint8)

    def test_draw_me_a_sun(self):
        expected_rays = arange(2, 14, dtype=uint32)
        expected_sun = array([[0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0],
                              [0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0],
                              [0,  0,  0,  3,  0,  2,  0,  4,  0,  0,  0],
                              [0,  0,  5,  0,  3,  0,  4,  0,  6,  0,  0],
                              [0,  0,  0,  5,  0,  0,  0,  6,  0,  0,  0],
                              [7,  7,  7,  0,  0,  0,  0,  0,  8,  8,  8],
                              [0,  0,  0,  9,  0,  0,  0, 10,  0,  0,  0],
                              [0,  0,  9,  0, 11,  0, 12,  0, 10,  0,  0],
                              [0,  0,  0, 11,  0, 13,  0, 12,  0,  0,  0],
                              [0,  0,  0,  0,  0, 13,  0,  0,  0,  0,  0],
                              [0,  0,  0,  0,  0, 13,  0,  0,  0,  0,  0]], dtype=uint32)
        rays, sun = draw_me_a_sun(self.main_shape, self.cross_33)
        testing.assert_array_equal(rays, expected_rays)
        testing.assert_array_equal(sun, expected_sun)



class TestFindMedianShape(CellectsUnitTest):
    def test_find_median_shape(self):
        binary_3d_matrix = array([[[1, 1, 0],
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
                                   [0, 1, 0]]], dtype=uint8)
        expected_median_shape = array([[1, 1, 0],
                                       [0, 1, 1],
                                       [0, 1, 0]], dtype=uint8)
        median_shape = find_median_shape(binary_3d_matrix)
        testing.assert_array_equal(median_shape, expected_median_shape)


class TestReduceImageSizeForSpeed(CellectsUnitTest):
    def test_reduce_image_size_for_speed(self):
        image_of_2_shapes = array([[1, 0, 1, 1],
                                   [2, 0, 2, 2],
                                   [1, 0, 1, 1],
                                   [1, 0, 2, 2]], dtype=uint8)
        expected_shape1_idx = (array([0, 0, 0, 2, 2, 2, 3], dtype=int64), array([0, 2, 3, 0, 2, 3, 0], dtype=int64))
        expected_shape2_idx = (array([1, 1, 1, 3, 3], dtype=int64), array([0, 2, 3, 2, 3], dtype=int64))
        shape1_idx, shape2_idx = reduce_image_size_for_speed(image_of_2_shapes)
        testing.assert_array_equal(shape1_idx, expected_shape1_idx)
        testing.assert_array_equal(shape2_idx, expected_shape2_idx)


class TestMinimalDistance(CellectsUnitTest):
    def test_minimal_distance(self):
        image_of_2_shapes = array([[0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0],
                                   [0, 0, 0, 2, 0],
                                   [0, 0, 0, 0, 0]])
        distance = get_minimal_distance_between_2_shapes(image_of_2_shapes, increase_speed=False)
        self.assertAlmostEqual(distance, 2.23606, places=3)

        image_of_2_shapes = array([[0, 0, 0, 0, 0],
                                   [0, 1, 0, 2, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0]])
        distance = get_minimal_distance_between_2_shapes(image_of_2_shapes, increase_speed=False)
        self.assertEqual(distance, 2.0)

        image_of_2_shapes = array([[0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 2]])
        distance = get_minimal_distance_between_2_shapes(image_of_2_shapes, increase_speed=False)
        self.assertAlmostEqual(distance, 3.605551, places=3)

    def test_minimal_distance_with_speed_increase(self):
        image_of_2_shapes = array([[0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0],
                                      [0, 0, 0, 2, 0],
                                      [0, 0, 0, 0, 0]])
        distance = get_minimal_distance_between_2_shapes(image_of_2_shapes, increase_speed=True)
        self.assertAlmostEqual(distance, 2.23606, places=3)

        image_of_2_shapes = array([[0, 0, 0, 0, 0],
                                   [0, 1, 0, 2, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0]])
        distance = get_minimal_distance_between_2_shapes(image_of_2_shapes, increase_speed=True)
        self.assertEqual(distance, 2.0)

        image_of_2_shapes = array([[0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 2]])
        distance = get_minimal_distance_between_2_shapes(image_of_2_shapes, increase_speed=True)
        self.assertAlmostEqual(distance, 3.605551, places=3)


class TestFindMajorIncline(CellectsUnitTest):
    def test_find_major_incline_failure(self):
        vector = concatenate((repeat(10, 50),array((20, 20)),array((40, 40)), repeat(50, 50)))
        natural_noise = 10
        left, right = find_major_incline(vector, natural_noise)
        self.assertEqual(left, 0)
        self.assertEqual(right, 1)


class TestRankFromTopToBottomFromLeftToRight(CellectsUnitTest):
    po = ProgramOrganizer()
    po.load_variable_dict()
    po.all['global_pathway'] = TEST_DIR / "experiment"
    po.all['first_folder_sample_number'] = 100
    po.look_for_data()
    po.load_data_to_run_cellects_quickly()
    po.update_output_list()

    def test_with_y_boundaries(self):
        self.po.get_first_image()
        backmask = zeros(self.po.first_im.shape[:2], uint8)
        backmask[-30:, :] = 1
        backmask = nonzero(backmask)
        self.po.vars['convert_for_origin'] = {'lab': array([0, 0, 1], dtype=int8), 'logical': 'None'}
        self.po.vars['convert_for_motion'] = {'lab': array([0, 0, 1], dtype=int8), 'logical': 'None'}
        self.po.fast_image_segmentation(True, backmask=backmask)
        self.po.all['automatically_crop'] = True
        self.po.cropping(is_first_image=True)
        self.po.all['scale_with_image_or_cells'] = 1
        self.po.all['starting_blob_hsize_in_mm'] = 15
        self.po.get_average_pixel_size()
        self.po.videos = OneVideoPerBlob(self.po.first_image, self.po.starting_blob_hsize_in_pixels,
                                         self.po.all['raw_images'])
        self.po.videos.first_image.shape_number = self.po.sample_number

        self.po.videos.big_kernel = Ellipse((self.po.videos.k_size, self.po.videos.k_size)).create()  # fromfunction(self.circle_fun, (self.k_size, self.k_size))
        self.po.videos.big_kernel = self.po.videos.big_kernel.astype(uint8)
        self.po.videos.small_kernel = array(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=uint8)
        self.po.videos.ordered_stats, ordered_centroids, self.po.videos.ordered_first_image = rank_from_top_to_bottom_from_left_to_right(
            self.po.videos.first_image.validated_shapes, self.po.videos.first_image.y_boundaries, get_ordered_image=True)
        self.assertEqual(ordered_centroids.shape[0], self.po.sample_number)


        self.assertTrue(array_equal(unique(self.po.videos.ordered_first_image), arange(self.po.sample_number + 1)))

        self.po.videos.ordered_stats, ordered_centroids, without_y_boundaries = rank_from_top_to_bottom_from_left_to_right(
            self.po.videos.first_image.validated_shapes, None,
            get_ordered_image=True)

        self.assertTrue(not array_equal(self.po.videos.ordered_first_image, without_y_boundaries))
        self.assertTrue(array_equal(self.po.videos.ordered_first_image > 0, without_y_boundaries > 0))


class TestExpandUntilNeighborCenterGetsNearerThanOwn(CellectsUnitTest):
    def test_no_expansion(self):
        shape_to_expand = zeros((9, 9), dtype=uint8)
        shape_to_expand[5:8, 5:8] = 1
        without_shape_i = zeros((9, 9), dtype=uint8)
        without_shape_i[1:4, 1:4] = 1
        without_shape_i[1:4, 5:8] = 1
        without_shape_i[5:8, 1:4] = 1
        shape_original_centroid = [6, 6]
        ref_centroids = array([[2, 2], [6, 2], [2, 6]], dtype=int32)
        kernel = ones((3, 3), dtype=uint8)
        expanded_shape = expand_until_neighbor_center_gets_nearer_than_own(
            shape_to_expand, without_shape_i, shape_original_centroid, ref_centroids, kernel
        )
        expected_result = array(
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
        self.assertTrue(array_equal(expanded_shape, expected_result))

    def test_expansion(self):
        shape_to_expand = zeros((10, 10), dtype=uint8)
        shape_to_expand[5:8, 6] = 1
        shape_to_expand[6, 5:8] = 1
        without_shape_i = zeros((10, 10), dtype=uint8)
        without_shape_i[1:4, 2] = 1
        without_shape_i[2, 1:4] = 1
        shape_original_centroid = [6, 6]
        ref_centroids = array([[4, 4], [2, 2]], dtype=int32)
        kernel = ones((3, 3), dtype=uint8)
        expanded_shape = expand_until_neighbor_center_gets_nearer_than_own(
            shape_to_expand, without_shape_i, shape_original_centroid, ref_centroids, kernel
        )
        expected_result = array(
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
        self.assertTrue(array_equal(expanded_shape, expected_result))


class TestImageBorders(CellectsUnitTest):
    def test_image_borders(self):
        # Test 1: Verify borders for a 3x3 image
        dimensions = (3, 3)
        borders = image_borders(dimensions)
        expected_result = array(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]], dtype=uint8
        )
        self.assertTrue(array_equal(borders, expected_result))

    def test_image_borders_large(self):
        # Test 2: Verify borders for a larger image
        dimensions = (5, 7)
        borders = image_borders(dimensions)
        expected_result = array(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]], dtype=uint8
        )
        self.assertTrue(array_equal(borders, expected_result))


class TestGetRadiusDistance(CellectsUnitTest):
    def test_get_radius_distance(self):
        # Test: Verify radius distance for a large binary video
        binary_video = array(
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
        expected_distance = array([2., 1.], dtype=float32)
        expected_start = 1
        expected_end = 2
        self.assertTrue(array_equal(distance_against_time, expected_distance))
        self.assertEqual(time_start, expected_start)
        self.assertEqual(time_end, expected_end)


class TestExpandToFillHoles(CellectsUnitTest):
    def test_expand_to_fill_holes(self):
        # Test 1: Verify expansion to fill holes
        binary_video = array(
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
        holes = array(
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
        cross_33 = array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], dtype=uint8
        )
        expanded_video, holes_time_end, distance_against_time = expand_to_fill_holes(binary_video, holes, cross_33)
        expected_expanded_video = array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=uint8)
        expected_holes_time_end = 1
        expected_distance_against_time = array([2.], dtype=float32)
        self.assertTrue(array_equal(expanded_video, expected_expanded_video))
        self.assertEqual(holes_time_end, expected_holes_time_end)
        self.assertTrue(array_equal(distance_against_time, expected_distance_against_time))


class TestEllipse(unittest.TestCase):
    def test_create_circle(self):
        # Test 1: Verify the correctness of ellipse_fun to create a circle
        sizes = [9, 9]
        ellipse = Ellipse(sizes).create()
        expected_result = array([[False, False, False, False,  True, False, False, False, False],
                                 [False, False,  True,  True,  True,  True,  True, False, False],
                                 [False,  True,  True,  True,  True,  True,  True,  True, False],
                                 [False,  True,  True,  True,  True,  True,  True,  True, False],
                                 [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                                 [False,  True,  True,  True,  True,  True,  True,  True, False],
                                 [False,  True,  True,  True,  True,  True,  True,  True, False],
                                 [False, False,  True,  True,  True,  True,  True, False, False],
                                 [False, False, False, False,  True, False, False, False, False]])
        self.assertTrue(array_equal(ellipse, expected_result))

    def test_create_circle(self):
        # Test 2: Verify the correctness of ellipse_fun to create an ellipse
        sizes = [5, 7]
        ellipse = Ellipse(sizes).create()
        expected_result = array([[False, False, False,  True, False, False, False],
                                 [False,  True,  True,  True,  True,  True, False],
                                 [True,  True,  True,  True,  True,  True,  True],
                                 [False,  True,  True,  True,  True,  True, False],
                                 [False, False, False,  True, False, False, False]])
        self.assertTrue(array_equal(ellipse, expected_result))


if __name__ == '__main__':
    unittest.main()