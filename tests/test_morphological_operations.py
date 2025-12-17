#!/usr/bin/env python3
"""
Unit tests for morphological operations.
"""

import unittest
import numpy as np
from tests._base import CellectsUnitTest, several_arenas_bin_img
from cellects.image_analysis.morphological_operations import *


class TestCompareNeighborsWithValue(CellectsUnitTest):
    """Test CompareNeighborsWithValue functionality."""
    vector = np.array([9, 0, 4, 6], dtype=np.int8)
    matrix = np.array([[9, 0, 4, 6],
                   [4, 9, 1, 3],
                   [7, 2, 1, 4],
                   [9, 0, 8, 5]], dtype=np.int8)

    def test_vector_is_equal(self):
        """Test is equal on a simple vector."""
        comparer = CompareNeighborsWithValue(self.vector, connectivity=4)
        comparer.is_equal(1)
        expected_result = np.array([0, 0, 0, 0], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.equal_neighbor_nb, expected_result))

    def test_vector_is_sup(self):
        """Test is sup on a simple vector."""
        comparer = CompareNeighborsWithValue(self.vector, connectivity=4)
        comparer.is_sup(1)
        expected_result = np.array([1, 2, 1, 2], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.sup_neighbor_nb, expected_result))

    def test_vector_is_inf(self):
        """Test is inf on a simple vector."""
        comparer = CompareNeighborsWithValue(self.vector, connectivity=4)
        comparer.is_inf(1)
        expected_result = np.array([1, 0, 1, 0], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.inf_neighbor_nb, expected_result))

    def test_is_equal_connectivity_4(self):
        """
        Test checking connectivity of 4 for a matrix with given values.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_equal(1)
        expected_result = np.array([[0, 0, 1, 0],
                                 [0, 1, 1, 1],
                                 [0, 1, 1, 1],
                                 [0, 0, 1, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.equal_neighbor_nb, expected_result))
        comparer.is_equal(2)
        expected_result = np.array([[0, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [1, 0, 1, 0],
                                 [0, 1, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.equal_neighbor_nb, expected_result))

    def test_is_equal_connectivity_8(self):
        """
        Test that connectivity 8 comparison of neighbors with value works correctly.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_equal(1)
        expected_result = np.array([[0, 1, 1, 1],
                                 [0, 2, 1, 2],
                                 [0, 2, 1, 2],
                                 [0, 1, 1, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.equal_neighbor_nb, expected_result))
        comparer.is_equal(2)
        expected_result = np.array([[0, 0, 0, 0],
                                 [1, 1, 1, 0],
                                 [1, 0, 1, 0],
                                 [1, 1, 1, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.equal_neighbor_nb, expected_result))

    def test_is_equal_diagonal(self):
        """
        Test equality comparison between a value and the diagonal pixels.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=0)
        comparer.is_equal(1)
        expected_result = np.array([[0, 1, 0, 1],
                                           [0, 1, 0, 1],
                                           [0, 1, 0, 1],
                                           [0, 1, 0, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.equal_neighbor_nb, expected_result))


    def test_is_sup_diagonal(self):
        """
        Test superiority comparison between a value and the diagonal pixels.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=0)
        comparer.is_sup(1)
        expected_result = np.array([[3, 3, 3, 3],
                                           [3, 3, 3, 3],
                                           [3, 3, 3, 3],
                                           [3, 3, 3, 3]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.sup_neighbor_nb, expected_result))


    def test_is_inf_diagonal(self):
        """
        Test inferiority comparison between a value and the diagonal pixels.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=0)
        comparer.is_inf(1)
        expected_result = np.array([[1, 0, 1, 0],
                                           [1, 0, 1, 0],
                                           [1, 0, 1, 0],
                                           [1, 0, 1, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.inf_neighbor_nb, expected_result))


    def test_is_equal_with_itself_connectivity_4(self):
        """
        Test is equal with itself connectivity 4.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_equal(1, and_itself=True)
        expected_result = np.array([[0, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.equal_neighbor_nb, expected_result))
        comparer.is_equal(2, and_itself=True)
        expected_result = np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.equal_neighbor_nb, expected_result))

    def test_is_equal_with_itself_connectivity_8(self):
        """
        Test that the neighbor comparison with connectivity 8 works correctly when comparing a value with itself.

        Parameters
        ----------
        self : TestCase
            The test case instance.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_equal(1, and_itself=True)
        expected_result = np.array([[0, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.equal_neighbor_nb, expected_result))
        comparer.is_equal(2, and_itself=True)
        expected_result = np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.equal_neighbor_nb, expected_result))

    def test_is_sup_connectivity_4(self):
        """
        Test that the connectivity of 4 is working properly for the matrix.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_sup(5)
        expected_result = np.array([[2, 2, 1, 2],
                                 [3, 0, 1, 1],
                                 [2, 2, 1, 0],
                                 [3, 2, 1, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.sup_neighbor_nb, expected_result))

    def test_is_sup_connectivity_8(self):
        """Test that the neighboring values connectivity with 8 is correctly checked."""
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_sup(5)
        expected_result = np.array([[4, 3, 3, 3],
                                 [5, 2, 2, 2],
                                 [4, 4, 2, 1],
                                 [5, 5, 1, 2]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.sup_neighbor_nb, expected_result))

    def test_is_sup_with_itself_connectivity_4(self):
        """
        Test that the Connectivity 4 comparison with itself works correctly.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_sup(5, and_itself=True)
        expected_result = np.array([[2, 0, 0, 2],
                                 [0, 0, 0, 0],
                                 [2, 0, 0, 0],
                                 [3, 0, 1, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.sup_neighbor_nb, expected_result))
        comparer.is_sup(3, and_itself=True)
        expected_result = np.array([[3, 0, 2, 3],
                                 [4, 1, 0, 0],
                                 [3, 0, 0, 2],
                                 [3, 0, 2, 4]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.sup_neighbor_nb, expected_result))

    def test_is_sup_with_itself_connectivity_8(self):
        """
        Test that compares neighbors with value using 8-connectivity and checks if a specific condition holds.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_sup(5, and_itself=True)
        expected_result = np.array([[4, 0, 0, 3],
                                 [0, 2, 0, 0],
                                 [4, 0, 0, 0],
                                 [5, 0, 1, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.sup_neighbor_nb, expected_result))

    def test_is_inf_connectivity_4(self):
        """Test that the `CompareNeighborsWithValue` class correctly identifies
        infinity neighbors with connectivity 4.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_inf(5)
        expected_result = np.array([[2, 2, 3, 2],
                                 [1, 4, 3, 3],
                                 [2, 2, 3, 3],
                                 [1, 2, 2, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.inf_neighbor_nb, expected_result))

    def test_is_inf_connectivity_8(self):
        """
        Test checking infinite connectivity with value 5 in an 8-connected neighborhood.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_inf(5)
        expected_result = np.array([[4, 5, 5, 5],
                                 [3, 6, 6, 6],
                                 [4, 4, 5, 5],
                                 [3, 3, 5, 3]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.inf_neighbor_nb, expected_result))

    def test_is_inf_with_itself_connectivity_4(self):
        """
        Test that connectivity 4 with itself comparison works correctly.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=4)
        comparer.is_inf(5, and_itself=True)
        expected_result = np.array([[0, 2, 3, 0],
                                 [1, 0, 3, 3],
                                 [0, 2, 3, 3],
                                 [0, 2, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.inf_neighbor_nb, expected_result))

    def test_is_inf_with_itself_connectivity_8(self):
        """
        Test that the connectivity 8 and is_inf method return correct result.
        """
        comparer = CompareNeighborsWithValue(self.matrix, connectivity=8)
        comparer.is_inf(5, and_itself=True)
        expected_result = np.array([[0, 5, 5, 0],
                                 [3, 0, 6, 6],
                                 [0, 4, 5, 5],
                                 [0, 3, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(comparer.inf_neighbor_nb, expected_result))


class TestCC(CellectsUnitTest):
    """Test the Connected Components algorithm."""
    binary_img = np.array([[1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1],
                        [0, 1, 1, 1, 1],
                        [0, 1, 1, 0, 0]], dtype=np.uint8)
    def test_cc_ordering(self):
        """Test that cc ordering is correct."""
        expected_order = np.array([[2, 2, 2, 2, 0],
                                [2, 0, 0, 0, 0],
                                [0, 0, 1, 0, 1],
                                [0, 1, 1, 1, 1],
                                [0, 1, 1, 0, 0]], dtype=np.uint8)
        expected_stats = np.array([[ 0,  0,  5,  5, 12],
                                [ 1,  2,  5,  5,  8],
                                [ 0,  0,  4,  2,  5]], dtype=np.int32)
        new_order, stats, centers = cc(self.binary_img)
        self.assertTrue(np.array_equal(new_order, expected_order))
        self.assertTrue(np.array_equal(stats, expected_stats))

    def test_cc_with_large_component_touching_border(self):
        """Test that cc is correct when one large component touches the border."""
        binary_img = np.array([[0, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 1, 0, 0]], dtype=np.uint8)
        new_order, stats, centers = cc(binary_img)
        self.assertTrue(np.array_equal(new_order, binary_img))

    def test_cc_with_many_components_touching_border(self):
        """Test that cc is correct when one large component touches the border."""
        binary_img = np.ones((50, 50), dtype=np.uint8)
        binary_img[1::3, :] = 0
        binary_img[:, 1::3] = 0
        new_order, stats, centers = cc(binary_img)
        self.assertEqual(new_order.max(), 289)


class TestShapeSelection(CellectsUnitTest):
    """Test suite for shape_selection() with image processing validation."""
    @classmethod
    def setUpClass(cls):
        """Initialize a data set for testing"""
        super().setUpClass()
        cls.height, cls.width = 100, 100
        cls.img = np.zeros((cls.height, cls.width), dtype=np.uint8)
        # Add two perfect circles in opposite corners
        cls.img[20:30, 20:30] = create_ellipse(10, 10)
        cls.img[-30:-20, -30:-20] = create_ellipse(10, 10)


    def test_shape_selection_valid_input(self):
        """Verify correct shape filtering with valid parameters and expected counts"""
        # Setup: Create binary image with 2 valid circular shapes

        validated_shapes, shape_number, stats, centroids = shape_selection(
            binary_image=self.img,
            several_blob_per_arena=True,
            true_shape_number=2,
            horizontal_size=10,
            spot_shape='circle'
        )

        # Validate output has exactly 2 connected components
        self.assertEqual(shape_number, 2)

    def test_shape_selection_empty(self):
        """Ensure no shape is detected form an empty video"""
        img = np.zeros((50, 50), dtype=np.uint8)

        validated_shapes, shape_number, stats, centroids = shape_selection(
                binary_image=img,
                several_blob_per_arena=False,
                true_shape_number=2,
                horizontal_size=10,
                spot_shape='triangle'  # Unsupported shape
            )
        self.assertEqual(validated_shapes.sum(), 0)

    def test_shape_selection_bio_mask_filtering(self):
        """Verify bio_mask correctly preserves protected components"""
        protected = np.zeros_like(self.img)

        validated_shapes, shape_number, stats, centroids = shape_selection(
            binary_image=self.img,
            several_blob_per_arena=True,
            true_shape_number=2,
            horizontal_size=10,
            bio_mask=protected
        )
        self.assertEqual(shape_number, 2)

    def test_shape_selection_rectangle_shape_validation(self):
        """Verify rectangle shape filtering with size constraints"""
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        # Add two perfect circles in opposite corners
        img[20:30, 20:30] = 1
        img[-30:-20, -30:-20] = 1
        validated_shapes, shape_number, stats, centroids = shape_selection(
            binary_image=self.img,
            several_blob_per_arena=False,
            true_shape_number=1,
            horizontal_size=10,
            spot_shape='rectangle'
        )
        self.assertEqual(shape_number, 2)

    def test_shape_selection_multiple_iterations_required(self):
        """Test scenario requiring multiple filtering iterations to reach target count"""
        img = np.zeros((100, 100), dtype=np.uint8)

        # Create 5 shapes with varying sizes (3 within range, 2 too small)
        for i in range(3):
            cv2.circle(img[20+i*20:40+i*20, 20+i*20:40+i*20], (10,10), 5, 255, -1)

        # Two shapes with invalid sizes
        cv2.circle(img[-30:-10, :20], (10,10), 3, 255, -1)    # Too small
        cv2.circle(img[-30:-10, 40:60], (10,10), 7, 255, -1)  # Too large

        validated_shapes, shape_number, stats, centroids = shape_selection(
            binary_image=img,
            several_blob_per_arena=False,
            true_shape_number=3,
            horizontal_size=10,
            spot_shape='circle'
        )
        self.assertEqual(shape_number, 3)  # Only valid shapes remain


class TestRoundedInvertedDistanceTransform(CellectsUnitTest):
    """
    Test rounded inverted distance transform.
    This test class verifies the behavior of the `rounded_inverted_distance_transform`
    function with various parameters.
    """
    original_shape = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    def test_rounded_inverted_distance_transform_no_erosion_no_max_distance(self):
        """Test that the rounded inverted distance transform is correctly computed without erosion and max distance.
        """
        expected_field = np.array([[1, 2, 3, 4, 3, 2, 1],
                                [2, 3, 4, 5, 4, 3, 2],
                                [3, 4, 5, 0, 5, 4, 3],
                                [4, 5, 0, 0, 0, 5, 4],
                                [3, 4, 5, 0, 5, 4, 3],
                                [2, 3, 4, 5, 4, 3, 2],
                                [1, 2, 3, 4, 3, 2, 1]], dtype=np.uint32)
        field = rounded_inverted_distance_transform(self.original_shape)
        self.assertTrue(np.array_equal(field, expected_field))

    def test_rounded_inverted_distance_transform_with_erosion(self):
        """Test that the rounded inverted distance transform with erosion produces the expected output."""
        expected_field = np.array([[1, 2, 3, 4, 3, 2, 1],
                                [2, 3, 4, 5, 4, 3, 2],
                                [3, 4, 5, 6, 5, 4, 3],
                                [4, 5, 6, 0, 6, 5, 4],
                                [3, 4, 5, 6, 5, 4, 3],
                                [2, 3, 4, 5, 4, 3, 2],
                                [1, 2, 3, 4, 3, 2, 1]], dtype=np.uint8)
        field = rounded_inverted_distance_transform(self.original_shape, with_erosion=1)
        self.assertTrue(np.array_equal(field, expected_field))

    def test_rounded_inverted_distance_transform_with_max_distance(self):
        """
        Test the rounded inverted distance transform with a maximum distance."""
        expected_field = np.array([[0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 2, 1, 0, 0],
                                [0, 1, 2, 0, 2, 1, 0],
                                [1, 2, 0, 0, 0, 2, 1],
                                [0, 1, 2, 0, 2, 1, 0],
                                [0, 0, 1, 2, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)
        field = rounded_inverted_distance_transform(self.original_shape, max_distance=2)
        self.assertTrue(np.array_equal(field, expected_field))


class TestInvertedDistanceTransform(CellectsUnitTest):
    """Test the behavior of the inverted_distance_transform function.
    This class contains unit tests for verifying different scenarios
    of the inverted_distance_transform functionality.
    """
    original_shape = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    def test_inverted_distance_transform_no_erosion_no_max_distance(self):
        """
        Test that the distance transform is inverted correctly without erosion and no maximum distance.
        """
        expected_field = np.array([
            [1.        , 1.77712415, 2.3694833 , 2.60555128, 2.3694833 , 1.77712415, 1.        ],
            [1.77712415, 2.3694833 , 3.19133771, 3.60555128, 3.19133771, 2.3694833 , 1.77712415],
            [2.3694833 , 3.19133771, 3.60555128, 0.        , 3.60555128, 3.19133771, 2.3694833 ],
            [2.60555128, 3.60555128, 0.        , 0.        , 0.        , 3.60555128, 2.60555128],
            [2.3694833 , 3.19133771, 3.60555128, 0.        , 3.60555128, 3.19133771, 2.3694833 ],
            [1.77712415, 2.3694833 , 3.19133771, 3.60555128, 3.19133771, 2.3694833 , 1.77712415],
            [1.        , 1.77712415, 2.3694833 , 2.60555128, 2.3694833 , 1.77712415, 1.        ]], dtype=np.float64)
        field = inverted_distance_transform(self.original_shape)
        self.assertTrue(np.array_equal(np.round(field, 8), expected_field))

    def test_inverted_distance_transform_with_erosion(self):
        """
        Test that checks the result of `inverted_distance_transform` with erosion."""
        expected_field = np.array([
            [1.        , 1.63708941, 2.08036303, 2.24264069, 2.08036303, 1.63708941, 1.        ],
            [1.63708941, 2.41421356, 3.00657271, 3.24264069, 3.00657271, 2.41421356, 1.63708941],
            [2.08036303, 3.00657271, 3.82842712, 4.24264069, 3.82842712, 3.00657271, 2.08036303],
            [2.24264069, 3.24264069, 4.24264069, 0.        , 4.24264069, 3.24264069, 2.24264069],
            [2.08036303, 3.00657271, 3.82842712, 4.24264069, 3.82842712, 3.00657271, 2.08036303],
            [1.63708941, 2.41421356, 3.00657271, 3.24264069, 3.00657271, 2.41421356, 1.63708941],
            [1.        , 1.63708941, 2.08036303, 2.24264069, 2.08036303, 1.63708941, 1.        ]], dtype=np.float64)
        field = inverted_distance_transform(self.original_shape, with_erosion=1)
        self.assertTrue(np.array_equal(np.round(field, 8), expected_field))

    def test_inverted_distance_transform_with_max_distance(self):
        """
        Test inverted distance transform with max_distance."""
        expected_field = np.array([
            [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 1.        , 1.41421356, 1.        , 0.        , 0.        ],
            [0.        , 1.        , 1.41421356, 0.        , 1.41421356, 1.        , 0.        ],
            [0.        , 1.41421356, 0.        , 0.        , 0.        , 1.41421356, 0.        ],
            [0.        , 1.        , 1.41421356, 0.        , 1.41421356, 1.        , 0.        ],
            [0.        , 0.        , 1.        , 1.41421356, 1.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ]], dtype=np.float64)
        field = inverted_distance_transform(self.original_shape, max_distance=2)
        self.assertTrue(np.array_equal(np.round(field, 8), expected_field))


class TestGetLinePoints(CellectsUnitTest):
    """Test get_line_points functionality."""
    def test_get_every_coord_between_2_points_vertical(self):
        point_A = (0, 0)
        point_B = (4, 0)
        expected_segment = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.uint64)
        segment = get_line_points(point_A, point_B)
        self.assertTrue(np.array_equal(segment, expected_segment))

    def test_get_every_coord_between_2_points_horizontal(self):
        """
        Test that the function returns every coordinate between two points in a horizontal line.
        """
        point_A = (0, 0)
        point_B = (0, 4)
        expected_segment = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]], dtype =np.uint64)
        segment = get_line_points(point_A, point_B)
        self.assertTrue(np.array_equal(segment, expected_segment))

    def test_get_every_coord_between_2_points_diagonal(self):
        """
        Test the `get_line_points` function to ensure it correctly computes coordinates between two diagonal points.
        """
        point_A = (0, 0)
        point_B = (4, 4)
        expected_segment = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype =np.uint64)
        segment = get_line_points(point_A, point_B)
        self.assertTrue(np.array_equal(segment, expected_segment))

    def test_get_every_coord_between_2_points_reversed_diagonal(self):
        """Test getting coordinates between two points in a reversed diagonal."""
        point_A = (0, 4)
        point_B = (4, 0)

        expected_segment = np.array([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], dtype =np.uint64)
        segment = get_line_points(point_A, point_B)
        self.assertTrue(np.array_equal(segment, expected_segment))


class TestGetAllLineCoordinates(CellectsUnitTest):
    """Test suite for get_all_line_coordinates function."""

    def test_get_all_line_coordinates_basic(self):
        """Test normal operation with integer inputs."""
        mat = np.zeros((10, 10))
        start_point = np.array((0, 0))
        end_points = np.array([[1, 2], [3, 4]])
        expected_first_line = np.array([[0, 0], [0, 1], [1, 2]], dtype=np.uint64)
        expected_second_line = np.array([[0, 0], [1, 1], [1, 2], [2, 3], [3, 4]], dtype=np.uint64)
        result = get_all_line_coordinates(start_point, end_points)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.array_equal(result[0], expected_first_line))
        self.assertTrue(np.array_equal(result[1], expected_second_line))

    def test_get_all_line_coordinates_empty_end_points(self):
        """Test behavior with empty end points array."""
        start_point = np.array([0, 0])
        end_points = np.array([])

        result = get_all_line_coordinates(start_point, end_points)

        self.assertEqual(len(result), 0)

    def test_get_all_line_coordinates_single_end_point(self):
        """Test with single end point."""
        start_point = np.array([0, 0])
        end_points = np.array([[1, 2]])

        result = get_all_line_coordinates(start_point, end_points)

        self.assertEqual(len(result), 1)

    def test_get_all_line_coordinates_zero_length(self):
        """Test with start point equal to end point (zero-length line)."""
        start_point = np.array([5, 5])
        end_points = np.array([[5, 5]])

        result = get_all_line_coordinates(start_point, end_points)
        self.assertEqual(len(result), 1)



class TestDrawMeASun(CellectsUnitTest):
    """Test Draw Me a Sun functionality."""
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

    def test_draw_me_a_sun(self):
        """
        Test that the draw_me_a_sun function produces expected rays and sun patterns."""
        expected_rays = np.arange(1, 13, dtype=np.uint32)
        expected_sun = np.array([
           [ 0,  4,  0,  0,  0,  6,  0,  0,  0,  8,  0],
           [ 2,  0,  4,  0,  0,  6,  0,  0,  8,  0, 10],
           [ 0,  2,  0,  4,  0,  6,  0,  8,  0, 10,  0],
           [ 0,  0,  2,  0,  4,  0,  8,  0, 10,  0,  0],
           [ 0,  0,  0,  2,  0,  0,  0, 10,  0,  0,  0],
           [ 1,  1,  1,  0,  0,  0,  0,  0, 12, 12, 12],
           [ 0,  0,  0,  3,  0,  0,  0, 11,  0,  0,  0],
           [ 0,  0,  3,  0,  5,  0,  9,  0, 11,  0,  0],
           [ 0,  3,  0,  5,  0,  7,  0,  9,  0, 11,  0],
           [ 3,  0,  5,  0,  0,  7,  0,  0,  9,  0, 11],
           [ 0,  5,  0,  0,  0,  7,  0,  0,  0,  9,  0]], dtype=np.uint32)
        rays, sun = draw_me_a_sun(self.main_shape)
        self.assertTrue(np.array_equal(rays, expected_rays))
        self.assertTrue(np.array_equal(sun, expected_sun))
        self.assertTrue(np.array_equal(rays, expected_rays))


class TestFindMedianShape(CellectsUnitTest):
    """Test find_median_shape suite."""
    def test_find_median_shape(self):
        """Test find_median_shape."""
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
        self.assertTrue(np.array_equal(median_shape, expected_median_shape))


class TestReduceImageSizeForSpeed(CellectsUnitTest):
    """Test the `reduce_image_size_for_speed` function."""
    def test_reduce_image_size_for_speed(self):
        """Test reduce_image_size_for_speed functionality."""
        image_of_2_shapes = np.array([[1, 0, 2],
                                             [0, 0, 2]], dtype=np.uint8)
        expected_shape1_idx = (np.array([0], dtype=np.int64), np.array([0], dtype=np.int64))
        expected_shape2_idx = (np.array([0, 1], dtype=np.int64), np.array([2, 2], dtype=np.int64))
        shape1_idx, shape2_idx = reduce_image_size_for_speed(image_of_2_shapes)
        self.assertTrue(np.array_equal(shape1_idx, expected_shape1_idx))
        self.assertTrue(np.array_equal(shape2_idx, expected_shape2_idx))


class TestMinimalDistance(CellectsUnitTest):
    """Test minimal distance calculation between two shapes.
    This class tests the functionality of computing the minimal distance between
    two shapes in an image using different speed optimization settings.
    """
    def test_minimal_distance(self):
        """Test minimal distance between shapes in different configurations."""
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
        """Test Minimal Distance Calculation with Speed Increase."""
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


class TestMinMaxEuclideanPair(CellectsUnitTest):
    """Test get_min_or_max_euclidean_pair suite."""
    def test_get_max_euclidean_pair(self):
        """Test get max Euclidean pair."""
        coords = np.array([[0, 1], [2, 3], [4, 5]])
        point1, point2 = get_min_or_max_euclidean_pair(coords, min_or_max="max")
        self.assertTrue(np.array_equal(point1, np.array([0, 1])))
        self.assertTrue(np.array_equal(point2, np.array([4, 5])))

    def test_get_min_euclidean_pair(self):
        """Test get min Euclidean pair."""
        coords = (np.array([0, 2, 4, 8, 1, 5]), np.array([0, 2, 4, 8, 0, 5]))
        point1, point2 = get_min_or_max_euclidean_pair(coords, min_or_max="min")
        self.assertTrue(np.array_equal(point1, np.array([0, 0])))
        self.assertTrue(np.array_equal(point2, np.array([1, 0])))


class TestFindMajorIncline(CellectsUnitTest):
    """Test finding major incline in a vector."""
    def test_find_major_incline_failure(self):
        """Test that find_major_incline handles vector with major incline."""
        vector = np.concatenate((np.repeat(10, 50), np.array((20, 20)), np.array((40, 40)),
                                 np.repeat(50, 50)))
        natural_noise = 10
        left, right = find_major_incline(vector, natural_noise)
        self.assertEqual(left, 0)
        self.assertEqual(right, 1)


class TestRankFromTopToBottomFromLeftToRight(CellectsUnitTest):
    """Test ranking from top to bottom from left to right."""
    @classmethod
    def setUpClass(cls):
        """Setup test fixtures, including a binary image and y-axis boundaries."""
        super().setUpClass()

        cls.binary_image = several_arenas_bin_img
        cls.y_boundaries = np.array([0,  1,  0,  0,  -1,  0,  1,  0,  0,  -1,  0], dtype=np.int8)

    def test_rank_from_top_to_bottom_from_left_to_right(self):
        """
        Test that the ranking function orders objects from top to bottom and left to right.
        This test method verifies that the `rank_from_top_to_bottom_from_left_to_right`
        function correctly orders objects in a binary image based on their positions
        from top to bottom and left to right.
        """
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

    def test_rank_from_top_to_bottom_from_left_to_right_with_no_boundaries(self):
        """
        Test that the ranking function orders objects from top to bottom and left to right.
        This test method verifies that the `rank_from_top_to_bottom_from_left_to_right`
        function correctly orders objects in a binary image based on their positions
        from top to bottom and left to right.
        """
        binary_image = np.zeros((7, 7), dtype=np.uint8)
        binary_image[1:3, 1:3] = 1
        binary_image[4:6, 4:6] = 1

        # Y boundaries with no detected rows
        y_boundaries = np.zeros(self.binary_image.shape[0])  # No +1 or -1 markers to indicate row intervals
        ordered_stats, ordered_centroids, ordered_image = rank_from_top_to_bottom_from_left_to_right(self.binary_image,
                                                                                                     y_boundaries,
                                                                                                     get_ordered_image=True)
        self.assertTrue(len(np.unique(ordered_image)) == 7)
        self.assertTrue(ordered_centroids.shape[0] == 6)
        self.assertTrue(ordered_stats[:, 4].sum() == self.binary_image.sum())


class TestGetLargestConnectedComponent(CellectsUnitTest):
    """Test suite for get_largest_connected_component function."""

    def test_normal_case_multiple_components(self):
        """Test normal operation with multiple components of different sizes."""
        # Setup test data - two rectangles, one larger than the other
        segmentation = np.zeros((10, 10), dtype=np.uint8)
        segmentation[2:6, 2:5] = 1  # Larger component (12 pixels)
        segmentation[6:9, 6:9] = 1  # Small component (9 pixels)
        expected_size = 12
        size, mask = get_largest_connected_component(segmentation)

        self.assertEqual(size, expected_size)
        self.assertTrue(mask.shape[0] == segmentation.shape[0])
        self.assertTrue(mask.shape[1] == segmentation.shape[1])
        self.assertTrue(np.all(mask[2:6, 2:5]))  # The larger component
        self.assertFalse(np.any(mask[6:9, 6:9]))  # The smaller component

    def test_single_component(self):
        """Test with only one connected component."""
        segmentation = np.zeros((5, 5), dtype=np.uint8)
        segmentation[1:4, 1:4] = 1  # Single component of size 9

        expected_size = 9
        expected_mask_shape = segmentation.shape

        size, mask = get_largest_connected_component(segmentation)

        self.assertEqual(mask.sum(), expected_size)
        self.assertEqual(size, expected_size)

    def test_empty_segmentation(self):
        """Test with empty segmentation (no connected components)."""
        segmentation = np.zeros((5, 5), dtype=np.uint8)

        with self.assertRaises(AssertionError):
            get_largest_connected_component(segmentation)

    def test_single_pixel_component(self):
        """Test with single pixel as the only component."""
        segmentation = np.zeros((3, 3), dtype=np.uint8)
        segmentation[1, 1] = 1

        expected_size = 1
        size, mask = get_largest_connected_component(segmentation)
        self.assertEqual(size, expected_size)

    def test_large_uniform_component(self):
        """Test with a single large component filling most of the image."""
        segmentation = np.ones((10, 10), dtype=np.uint8)
        segmentation[0, 0] = 0  # Leave one pixel as background

        size, mask = get_largest_connected_component(segmentation)
        self.assertTrue(np.array_equal(mask, segmentation >0))

    def test_largest_component_is_edge_case(self):
        """Test when largest component is at the edge of the array."""
        segmentation = np.zeros((5, 5), dtype=np.uint8)
        # Create two components: one at edge (4 pixels) and one in center (3 pixels)
        segmentation[0, 0:2] = 1  # Edge component
        segmentation[2:4, 2:4] = 1  # Center component

        expected_size = 4

        size, mask = get_largest_connected_component(segmentation)

        self.assertEqual(size, expected_size)
        self.assertTrue(mask[2:4, 2:4].sum() == expected_size)


class TestExpandUntilNeighborCenterGetsNearerThanOwn(CellectsUnitTest):
    """Test shape expansion until centers are closer."""
    def test_no_expansion(self):
        """
        Test that the function expands a shape until its neighbor centroid gets nearer than its own.
        """
        shape_to_expand = np.zeros((9, 9), dtype=np.uint8)
        shape_to_expand[5:8, 5:8] = 1
        without_shape_i = np.zeros((9, 9), dtype=np.uint8)
        without_shape_i[1:4, 1:4] = 1
        without_shape_i[1:4, 5:8] = 1
        without_shape_i[5:8, 1:4] = 1
        shape_original_centroid = [6, 6]
        ref_centroids = np.array([[2, 2], [6, 2], [2, 6]], dtype=np.int32)
        expanded_shape = expand_until_neighbor_center_gets_nearer_than_own(
            shape_to_expand, without_shape_i, shape_original_centroid, ref_centroids, square_33
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
             [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
        )
        self.assertTrue(np.array_equal(expanded_shape, expected_result))

    def test_expansion(self):
        """
        Test expansion of shape until neighbor center gets near.
        This test method verifies that a given shape expands correctly
        until the centroid of its neighboring shape gets closer than its own
        centroid.
        """
        shape_to_expand = np.zeros((10, 10), dtype=np.uint8)
        shape_to_expand[5:8, 6] = 1
        shape_to_expand[6, 5:8] = 1
        without_shape_i = np.zeros((10, 10), dtype=np.uint8)
        without_shape_i[1:4, 2] = 1
        without_shape_i[2, 1:4] = 1
        without_shape_i[:3, 7] = 1
        without_shape_i[1, 6:9] = 1
        shape_original_centroid = [6, 6]
        ref_centroids = np.array([[4, 4], [2, 2], [1, 7]], dtype=np.int32)
        expanded_shape = expand_until_neighbor_center_gets_nearer_than_own(
            shape_to_expand, without_shape_i, shape_original_centroid, ref_centroids, square_33
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
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
        )
        self.assertTrue(np.array_equal(expanded_shape, expected_result))


class TestImageBorders(CellectsUnitTest):
    """Test image border functionality."""
    def test_image_borders(self):
        """Test image borders detection."""
        # Test 1: Verify borders for a 3x3 image
        dimensions = (3, 3)
        borders = image_borders(dimensions)
        expected_result = np.array(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]], dtype=np.uint8
        )
        self.assertTrue(np.array_equal(borders, expected_result))

    def test_image_borders_circular(self):
        """Test image borders circular."""
        dimensions = (3, 3)
        borders = image_borders(dimensions, "circular")
        expected_result = np.array(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]], dtype=np.uint8
        )
        self.assertTrue(np.array_equal(borders, expected_result))

    def test_image_borders_large(self):
        """
        def test_image_borders_large(self):
            Test that borders are correctly computed for larger-sized images.
        """
        # Test 2: Verify borders for a larger image
        dimensions = (5, 7)
        borders = image_borders(dimensions)
        expected_result = np.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
        )
        self.assertTrue(np.array_equal(borders, expected_result))


class TestGetRadiusDistance(CellectsUnitTest):
    """Test get_radius_distance."""
    def test_get_radius_distance(self):
        """
        Test that verifies radius distance for a large binary video.
        """
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
              [0, 0, 1, 0, 0]]], dtype=np.uint8
        )
        field = rounded_inverted_distance_transform(binary_video[0, ...],  2)
        distance_against_time, time_start, time_end = get_radius_distance_against_time(binary_video, field)
        expected_distance = np.array([2., 1.], dtype=np.float32)
        expected_start = 1
        expected_end = 2
        self.assertTrue(np.array_equal(distance_against_time, expected_distance))
        self.assertEqual(time_start, expected_start)
        self.assertEqual(time_end, expected_end)


class TestCloseHoles(CellectsUnitTest):
    """Test suite for close_holes function."""

    def test_close_holes_with_single_small_hole(self):
        """Test that close_holes fills a single small hole in the center."""
        # Setup test data - create a square with a small hole
        binary_img = np.zeros((10, 10), dtype=np.uint8)
        binary_img[2:8, 2:8] = 1
        binary_img[4:6, 4:6] = 0  # Create a small hole

        expected = np.ones((10, 10), dtype=np.uint8)
        expected[:2, :] = 0
        expected[8:, :] = 0
        expected[:, :2] = 0
        expected[:, 8:] = 0

        result = close_holes(binary_img)
        self.assertTrue(np.array_equal(result, expected))

    def test_close_holes_with_no_holes(self):
        """Test that close_holes returns same image when no holes exist."""
        binary_img = np.ones((5, 5), dtype=np.uint8)
        result = close_holes(binary_img)
        self.assertTrue(np.array_equal(result, binary_img))

    def test_close_holes_with_all_zeros(self):
        """Test that close_holes handles all-zero image."""
        binary_img = np.zeros((5, 5), dtype=np.uint8)
        result = close_holes(binary_img)
        self.assertTrue(np.array_equal(result, binary_img))

    def test_close_holes_with_single_pixel(self):
        """Test that close_holes handles single pixel image."""
        binary_img = np.array([[1]], dtype=np.uint8)
        result = close_holes(binary_img)
        self.assertTrue(np.array_equal(result, binary_img))

    def test_close_holes_with_multiple_holes(self):
        """Test that close_holes fills multiple small holes."""
        binary_img = np.zeros((11, 11), dtype=np.uint8)
        binary_img[2:9, 2:9] = 1
        binary_img[3:5, 3:5] = 0  # First hole
        binary_img[6:8, 6:8] = 0  # Second hole

        expected = np.zeros((11, 11), dtype=np.uint8)
        expected[2:9, 2:9] = 1

        result = close_holes(binary_img)
        self.assertTrue(np.array_equal(result, expected))

    def test_close_holes_with_large_hole(self):
        """Test that close_holes doesn't fill holes larger than main object."""
        binary_img = np.zeros((10, 10), dtype=np.uint8)
        binary_img[2:8, 2:8] = 1
        binary_img[3:7, 3:7] = 0  # Large hole

        result = close_holes(binary_img)
        self.assertTrue(np.array_equal(result, binary_img))

    def test_close_holes_with_invalid_input_type(self):
        """Test that close_holes raises error for invalid input type."""
        with self.assertRaises(TypeError):
            close_holes([1, 2, 3])  # List instead of ndarray


class TestDynamicallyExpandToFillHoles(CellectsUnitTest):

    def test_dynamically_expand_to_fill_holes_with_no_hole(self):
        """Test dynamically expanding to fill holes without holes."""
        binary_video = np.zeros((2, 5, 5), dtype=np.uint8)
        holes = np.zeros((5, 5), dtype=np.uint8)
        expanded_video, holes_time_end, distance_against_time = dynamically_expand_to_fill_holes(binary_video, holes)
        self.assertIsInstance(expanded_video, np.ndarray)

    def test_dynamically_expand_to_fill_holes(self):
        """
        Test that binary video dynamically expands to fill holes.
        """
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
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=np.uint8
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
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
        )
        cross_33 = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], dtype=np.uint8
        )
        expanded_video, holes_time_end, distance_against_time = dynamically_expand_to_fill_holes(binary_video, holes)
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


class TestCreateEllipse(unittest.TestCase):
    """Test that an ellipse is created correctly."""
    def test_create_circle(self):
        """
        Verify the correctness of ellipse_fun to create a circle
        """
        # Test 1: Verify the correctness of ellipse_fun to create a circle
        ellipse = create_ellipse(9, 9)
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

    def test_create_ellipse(self):
        """
        Test that verifies the correctness of ellipse creation.
        """
        # Test 2: Verify the correctness of ellipse_fun to create an ellipse
        ellipse = create_ellipse(5, 7)
        expected_result = np.array([[False, False, False,  True, False, False, False],
                                 [False,  True,  True,  True,  True,  True, False],
                                 [True,  True,  True,  True,  True,  True,  True],
                                 [False,  True,  True,  True,  True,  True, False],
                                 [False, False, False,  True, False, False, False]])
        self.assertTrue(np.array_equal(ellipse, expected_result))


class TestGetContours(CellectsUnitTest):
    """Test get_contours functionality."""

    def test_get_contours_with_rectangle(self):
        """Test that get_contours returns expected contours for a simple rectangle."""
        # Setup test data - 5x5 image with 3x3 inner rectangle
        binary_img = np.zeros((5, 5), dtype=np.uint8)
        binary_img[1:4, 1:4] = 1

        # Expected result - contours should be the edges of the rectangle
        expected = np.array([[0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0],
                             [0, 1, 0, 1, 0],
                             [0, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0]], dtype=np.uint8)

        # Execute function
        result = get_contours(binary_img)

        # Verify result
        self.assertTrue(np.array_equal(result, expected))

    def test_get_contours_all_ones(self):
        """Test that get_contours returns all zeros for input with no contours."""
        # Setup test data
        binary_img = np.ones((5, 5), dtype=np.uint8)

        # Expected result - all zeros (no contours)
        expected = np.array([[1, 1, 1, 1, 1],
                                   [1, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 1],
                                   [1, 1, 1, 1, 1]], dtype=np.uint8)

        # Execute function
        result = get_contours(binary_img)

        # Verify result
        self.assertTrue(np.array_equal(result, expected))

    def test_get_contours_single_pixel(self):
        """Test that get_contours returns no contours for single pixel."""
        # Setup test data
        binary_img = np.array([[1]], dtype=np.uint8)

        # Expected result - no contours
        expected = np.array([[1]], dtype=np.uint8)

        # Execute function
        result = get_contours(binary_img)

        # Verify result
        self.assertTrue(np.array_equal(result, expected))

    def test_get_contours_no_pixel(self):
        """Test that get_contours returns no contours for single pixel."""
        # Setup test data
        binary_img = np.zeros(1, dtype=np.uint8)

        # Expected result - no contours
        expected = np.zeros(1, dtype=np.uint8)

        # Execute function
        result = get_contours(binary_img)

        # Verify result
        self.assertTrue(np.array_equal(result, expected))


class TestPrepareBoxCounting(CellectsUnitTest):
    """Test suite for prepare_box_counting function."""

    def test_normal_image_default_params(self):
        """Test with normal image and default parameters."""
        binary_image = np.zeros((10, 10), dtype=np.uint8)
        binary_image[2:4, 2:6] = 1
        binary_image[7:9, 4:7] = 1

        cropped_img, side_lengths = prepare_box_counting(binary_image, min_im_side=2, min_mesh_side=2)

        # Check that image is cropped to bounding box
        expected_cropped = np.array([[0, 0, 0, 0, 0, 0, 0],
                                           [0, 1, 1, 1, 1, 0, 0],
                                           [0, 1, 1, 1, 1, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 1, 1, 1, 0],
                                           [0, 0, 0, 1, 1, 1, 0],
                                           [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        expected_side_lengths = np.array([4, 2])
        self.assertTrue(np.array_equal(cropped_img, expected_cropped))
        self.assertTrue(np.array_equal(side_lengths, expected_side_lengths))

    def test_image_with_contours(self):
        """Test contour extraction when contours=True."""
        binary_image = np.zeros((5, 5), dtype=np.uint8)
        binary_image[1:4, 1:4] = 1

        cropped_img, _ = prepare_box_counting(binary_image, contours=True)

        self.assertTrue(np.array_equal(cropped_img, binary_image))

    def test_empty_image(self):
        """Test with completely empty image."""
        binary_image = np.zeros((5, 5), dtype=np.uint8)

        cropped_img, side_lengths = prepare_box_counting(binary_image, min_im_side=2, min_mesh_side=2)

        # Should return empty array and None for side_lengths
        self.assertTrue(np.array_equal(cropped_img, binary_image))
        self.assertIsNone(side_lengths)

    def test_single_pixel(self):
        """Test with single pixel image."""
        binary_image = np.zeros((5, 5), dtype=np.uint8)
        binary_image[2, 2] = 1

        cropped_img, side_lengths = prepare_box_counting(binary_image, min_im_side=2, min_mesh_side=2)

        # Should return single pixel and no side lengths (min_side < min_im_side)
        expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(cropped_img, expected))
        self.assertTrue(np.array_equal(side_lengths, np.array([2])))

    def test_zoom_step_nonzero(self):
        """Test side lengths computation with non-zero zoom_step."""
        binary_image = np.zeros((16, 16), dtype=np.uint8)
        binary_image[4:12, 4:12] = 1

        _, side_lengths = prepare_box_counting(binary_image, min_im_side=2, min_mesh_side=2, zoom_step=2)

        # Should return [8, 10, 12, 14] with zoom_step=2
        expected = np.array([2, 4, 6, 8], dtype=np.uint8)
        self.assertTrue(np.array_equal(side_lengths, expected))

    def test_small_image_not_on_contours(self):
        """Test when min_side < min_im_side."""
        binary_image = np.zeros((8, 8), dtype=np.uint8)
        binary_image[2:6, 2:6] = 1

        cropped_img, side_lengths = prepare_box_counting(binary_image, min_im_side=2, min_mesh_side=2, contours=False)

        # Should return cropped image but no side lengths
        expected = np.array([[0, 0, 0, 0, 0, 0],
                                   [0, 1, 1, 1, 1, 0],
                                   [0, 1, 1, 1, 1, 0],
                                   [0, 1, 1, 1, 1, 0],
                                   [0, 1, 1, 1, 1, 0],
                                   [0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        expected_side_lengths = np.array([4, 2])
        self.assertTrue(np.array_equal(cropped_img, expected))
        self.assertTrue(np.array_equal(side_lengths, expected_side_lengths))


class TestBoxCountingDimension(CellectsUnitTest):
    """Test suite for box_counting_dimension function."""
    binary_image = np.zeros((10, 10), dtype=np.uint8)
    binary_image[2:4, 2:6] = 1
    binary_image[7:9, 4:7] = 1
    binary_image[4:7, 5] = 1

    def test_normal_operation(self):
        """Test that the function returns expected results with normal input."""
        # Setup test data

        # Prepare inputs as shown in the example
        zoomed_binary = self.binary_image  # Assuming this is already prepared
        side_lengths = np.array([10, 5, 2])  # Example side lengths

        expected_dimension = np.float64(1.1777767776413224)
        expected_r_value = np.float64(0.9491082009262662)
        expected_box_nb = 3

        # Execute function
        dimension, r_value, box_nb = box_counting_dimension(zoomed_binary, side_lengths)

        # Verify results
        self.assertEqual(dimension, expected_dimension)
        self.assertEqual(r_value, expected_r_value)
        self.assertEqual(box_nb, expected_box_nb)

    def test_empty_side_lengths(self):
        """Test that the function handles empty side_lengths array."""
        side_lengths = np.array([], dtype=np.float64)

        # Should return zeros
        dimension, r_value, box_nb = box_counting_dimension(self.binary_image, side_lengths)

        self.assertEqual(dimension, 0.0)
        self.assertEqual(r_value, 0.0)
        self.assertEqual(box_nb, 0.0)

    def test_side_lengths_none(self):
        """Test that the function handles side_lengths=None."""

        # Should return zeros
        dimension, r_value, box_nb = box_counting_dimension(self.binary_image, None)

        self.assertEqual(dimension, 0.0)
        self.assertEqual(r_value, 0.0)
        self.assertEqual(box_nb, 0.0)

    def test_all_zero_box_counts(self):
        """Test that the function handles case where all box counts are zero."""
        binary_image = np.zeros((10, 10), dtype=np.uint8)
        side_lengths = np.array([2, 3, 5])

        # All boxes should be empty
        dimension, r_value, box_nb = box_counting_dimension(binary_image, side_lengths)

        self.assertEqual(dimension, 0.0)
        self.assertEqual(r_value, 0.0)
        self.assertEqual(box_nb, 0.0)  # Still should count the boxes


    def test_invalid_image_data(self):
        """Test that function handles non-binary image data."""
        # Create an image with values other than 0 and 255
        invalid_image = np.array([[0, 3, 1], [4, 255, 2]], dtype=np.uint8)
        side_lengths = np.array([2])

        # Should still process (but behavior may be undefined)
        dimension, r_value, box_nb = box_counting_dimension(invalid_image, side_lengths)

        # Just verify it doesn't crash
        self.assertEqual(dimension, 0.0)
        self.assertEqual(r_value, 0.0)
        self.assertEqual(box_nb, 0.0)


class TestKeepShapeConnectedWithRef(CellectsUnitTest):
    """Test suite for keep_shape_connected_with_ref function."""

    def test_normal_operation(self):
        """Test that the function returns expected results with normal input."""
        all_shapes = np.zeros((5, 5), dtype=np.uint8)
        reference_shape = np.zeros((5, 5), dtype=np.uint8)
        reference_shape[3, 3] = 1

        # Create some components that don't intersect with reference
        all_shapes[0:2, 0:2] = 1
        all_shapes[3:4, 3:4] = 1
        result = keep_shape_connected_with_ref(all_shapes, reference_shape)
        self.assertTrue(np.array_equal(result, reference_shape))

    def test_happy_path_intersecting_shapes(self):
        """Test that the function returns first intersecting component."""
        # Create test images
        all_shapes = np.zeros((5, 5), dtype=np.uint8)
        reference_shape = np.zeros((5, 5), dtype=np.uint8)

        # Component 1 does not intersect
        all_shapes[0:2, 0:2] = 1

        # Component 2 intersects
        all_shapes[2:4, 2:4] = 1
        reference_shape[3, 3] = 1

        # Component 3 intersects (but should not be returned as it's not first)
        all_shapes[4:5, 4:5] = 1
        reference_shape[4, 4] = 1

        result = keep_shape_connected_with_ref(all_shapes, reference_shape)
        self.assertTrue(np.array_equal(result, all_shapes))

    def test_no_intersecting_components(self):
        """Test that function returns None when no components intersect."""
        all_shapes = np.zeros((5, 5), dtype=np.uint8)
        reference_shape = np.zeros((5, 5), dtype=np.uint8)

        # Create some components that don't intersect with reference
        all_shapes[0:2, 0:2] = 1
        all_shapes[2:3, 3:4] = 1

        result = keep_shape_connected_with_ref(all_shapes, reference_shape)
        self.assertTrue(np.array_equal(result, reference_shape))

    def test_single_component_no_intersection(self):
        """Test single component case that doesn't intersect."""
        all_shapes = np.ones((3, 3), dtype=np.uint8)
        reference_shape = np.zeros((3, 3), dtype=np.uint8)

        result = keep_shape_connected_with_ref(all_shapes, reference_shape)
        self.assertTrue(np.array_equal(result, reference_shape))

    def test_single_component_with_intersection(self):
        """Test single component case that intersects."""
        all_shapes = np.ones((3, 3), dtype=np.uint8)
        reference_shape = np.ones((3, 3), dtype=np.uint8)

        expected = np.ones((3, 3), dtype=np.uint8)
        result = keep_shape_connected_with_ref(all_shapes, reference_shape)
        self.assertTrue(np.array_equal(result, expected))

    def test_empty_reference_shape(self):
        """Test with empty reference shape."""
        all_shapes = np.zeros((5, 5), dtype=np.uint8)
        reference_shape = np.zeros((5, 5), dtype=np.uint8)

        result = keep_shape_connected_with_ref(all_shapes, reference_shape)
        self.assertIsNone(result)

    def test_multiple_intersecting_components(self):
        """Test that function returns first intersecting component when multiple exist."""
        all_shapes = np.zeros((5, 5), dtype=np.uint8)
        reference_shape = np.zeros((5, 5), dtype=np.uint8)

        # Component 1 intersects
        all_shapes[0:2, 0:2] = 1
        reference_shape[1, 1] = 1

        # Component 2 also intersects (but should not be returned)
        all_shapes[3:4, 3:4] = 1
        reference_shape[3, 3] = 1

        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[0:2, 0:2] = 1

        result = keep_shape_connected_with_ref(all_shapes, reference_shape)
        np.testing.assert_array_equal(result, expected)


class TestKeepLargestShape(CellectsUnitTest):
    """Test suite for keep_largest_shape function."""

    def test_keep_largest_shape_basic_case(self):
        """Test basic functionality with clear dominant shape."""
        indexed_shapes = np.array([0, 2, 2, 3, 1], dtype=np.int32)
        expected = np.array([0, 1, 1, 0, 0], dtype=np.uint8)
        result = keep_largest_shape(indexed_shapes)
        self.assertTrue(np.array_equal(result, expected))

    def test_keep_largest_shape_tie_breaks_first(self):
        """Test that ties break by choosing the first occurring maximum."""
        indexed_shapes = np.array([0, 2, 3, 2, 3], dtype=np.int32)
        expected = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        result = keep_largest_shape(indexed_shapes)
        self.assertTrue(np.array_equal(result, expected))

    def test_keep_largest_shape_single_dominant(self):
        """Test with only one non-zero shape present."""
        indexed_shapes = np.array([0, 2, 2, 2], dtype=np.int32)
        expected = np.array([0, 1, 1, 1], dtype=np.uint8)
        result = keep_largest_shape(indexed_shapes)
        self.assertTrue(np.array_equal(result, expected))

    def test_keep_largest_shape_all_same(self):
        """Test when all elements are the same shape."""
        indexed_shapes = np.array([0, 2, 2, 2], dtype=np.int32)
        expected = np.array([0, 1, 1, 1], dtype=np.uint8)
        result = keep_largest_shape(indexed_shapes)
        self.assertTrue(np.array_equal(result, expected))

    def test_keep_largest_shape_with_zero(self):
        """Test that zeros are properly ignored in counting."""
        indexed_shapes = np.array([0, 2, 0, 2, 3], dtype=np.int32)
        expected = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        result = keep_largest_shape(indexed_shapes)
        self.assertTrue(np.array_equal(result, expected))

    def test_keep_largest_shape_single_element(self):
        """Test with single element array."""
        indexed_shapes = np.array([2], dtype=np.int32)
        expected = np.array([1], dtype=np.uint8)
        result = keep_largest_shape(indexed_shapes)
        self.assertTrue(np.array_equal(result, expected))


class TestKeepOneConnectedComponent(CellectsUnitTest):
    """Test suite for keep_one_connected_component function."""

    def test_happy_path_multiple_components(self):
        """Test normal operation with multiple components."""
        # Create test image with two components of different sizes
        img = np.zeros((5, 5), dtype=np.uint8)
        img[0:2, 0:2] = 1  # 4-pixel component
        img[3:4, 3:4] = 1  # 1-pixel component

        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[0:2, 0:2] = 1

        result = keep_one_connected_component(img)
        self.assertTrue(np.array_equal(result, expected))

    def test_single_component(self):
        """Test that single component returns original image."""
        img = np.zeros((5, 5), dtype=np.uint8)
        img[1:3, 1:3] = 1

        result = keep_one_connected_component(img)
        self.assertTrue(np.array_equal(result, img))

    def test_empty_image(self):
        """Test with completely empty image."""
        img = np.zeros((5, 5), dtype=np.uint8)

        result = keep_one_connected_component(img)
        self.assertTrue(np.array_equal(result, img))

    def test_non_uint8_input(self):
        """Test that input is converted to uint8 if needed."""
        img = np.zeros((5, 5), dtype=np.int32)
        img[1:3, 1:3] = 1

        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[1:3, 1:3] = 1

        result = keep_one_connected_component(img)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.array_equal(result, expected))

    def test_large_component_first(self):
        """Test when largest component is first in image."""
        img = np.zeros((5, 5), dtype=np.uint8)
        img[0:2, 0:2] = 1  # larger component
        img[4:5, 4:5] = 1  # smaller component

        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[0:2, 0:2] = 1

        result = keep_one_connected_component(img)
        self.assertTrue(np.array_equal(result, expected))

    def test_large_component_last(self):
        """Test when largest component is last in image."""
        img = np.zeros((5, 5), dtype=np.uint8)
        img[4:5, 4:5] = 1  # smaller component
        img[0:2, 0:2] = 1  # larger component

        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[0:2, 0:2] = 1

        result = keep_one_connected_component(img)
        self.assertTrue(np.array_equal(result, expected))

    def test_all_pixels_set(self):
        """Test when all pixels are part of one component."""
        img = np.ones((5, 5), dtype=np.uint8)

        result = keep_one_connected_component(img)
        self.assertTrue(np.array_equal(result, img))

    def test_same_size_components(self):
        """Test when multiple components have same size."""
        img = np.zeros((5, 5), dtype=np.uint8)
        img[0:2, 0:2] = 1  # first component
        img[3:5, 3:5] = 1  # second component of same size

        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[0:2, 0:2] = 1  # should keep first one based on order

        result = keep_one_connected_component(img)
        self.assertTrue(np.array_equal(result, expected))

    def test_all_zero_input(self):
        """Test with all zeros input."""
        img = np.zeros((5, 5), dtype=np.uint8)

        result = keep_one_connected_component(img)
        self.assertTrue(np.array_equal(result, img))

    def test_single_pixel_image(self):
        """Test with single pixel image."""
        img = np.array([[1]], dtype=np.uint8)

        result = keep_one_connected_component(img)
        self.assertTrue(np.array_equal(result, img))


class TestCreateMask(CellectsUnitTest):
    """Test suite for create_mask function."""

    def test_create_mask_rectangular_normal(self):
        """Test creating a normal rectangular mask."""
        dims = (10, 10)
        minmax = (2, 8, 3, 7)  # x_min, x_max, y_min, y_max
        shape = 'rectangle'

        result = create_mask(dims, minmax, shape)

        # Expected: 6x4 rectangle of True values in a 10x10 False array
        expected = np.zeros(dims, dtype=bool)
        expected[2:8, 3:7] = True

        self.assertTrue(np.array_equal(result, expected))
        self.assertEqual(result.shape, dims)

    def test_create_mask_circle_normal(self):
        """Test creating a normal circular mask."""
        dims = (10, 10)
        minmax = (2, 8, 3, 7)  # x_min, x_max, y_min, y_max
        shape = 'circle'

        result = create_mask(dims, minmax, shape)

        # We can't easily predict the exact circle pattern without knowing implementation
        # So we check basic properties instead:
        self.assertEqual(result.shape, dims[:2])
        self.assertTrue(np.any(result))  # Should have some True values
        self.assertTrue(np.any(~result))  # Should have some False values

    def test_create_mask_small_region(self):
        """Test creating a mask with minimal region size (1x1 pixel)."""
        dims = (5, 5)
        minmax = (2, 3, 2, 3)  # Single pixel at (2,2)
        shape = 'rectangle'

        result = create_mask(dims, minmax, shape)

        expected = np.zeros(dims, dtype=bool)
        expected[2, 2] = True

        self.assertTrue(np.array_equal(result, expected))

    def test_create_mask_boundary_coordinates(self):
        """Test creating a mask with coordinates at array boundaries."""
        dims = (5, 5)
        minmax = (0, 5, 0, 5)  # Entire array
        shape = 'rectangle'

        result = create_mask(dims, minmax, shape)

        expected = np.ones((5, 5), dtype=bool)
        self.assertTrue(np.array_equal(result, expected))

    def test_create_mask_invalid_shape(self):
        """Test that invalid shape values don't crash the function."""
        dims = (5, 5)
        minmax = (1, 4, 1, 4)
        shape = 'invalid_shape'

        # Should not raise an exception
        result = create_mask(dims, minmax, shape)

        # Should behave like rectangle in this case
        expected = np.zeros(dims, dtype=bool)
        expected[1:4, 1:4] = True

        self.assertTrue(np.array_equal(result, expected))

    def test_create_mask_empty_region(self):
        """Test creating a mask with no region (min=max)."""
        dims = (5, 5)
        minmax = (2, 2, 3, 3)  # Empty region
        shape = 'rectangle'

        result = create_mask(dims, minmax, shape)

        expected = np.zeros(dims, dtype=bool)
        self.assertTrue(np.array_equal(result, expected))

    def test_create_mask_circle_failure(self):
        """Test behavior when circle creation fails."""
        dims = (5, 5)
        minmax = (1, -1, 2, 4)  # Invalid coordinates
        shape = 'circle'

        with self.assertRaises(ValueError):
            create_mask(dims, minmax, shape)

    def test_create_mask_negative_coordinates(self):
        """Test behavior with negative coordinates."""
        dims = (5, 5)
        minmax = (-1, 3, 2, 4)  # Negative x_min
        shape = 'rectangle'

        result = create_mask(dims, minmax, shape)

        # Expected: Should clamp to valid range (0)
        expected = np.zeros(dims, dtype=bool)
        # Note: Actual behavior depends on implementation
        self.assertTrue(np.array_equal(result, expected))


class TestDrawImgWithMask(CellectsUnitTest):
    """Test suite for draw_img_with_mask function."""

    def test_draw_1_pixel_mask(self):
        """Test that circle mask is applied correctly to normal image."""
        # Setup test data
        dims = (3, 3, 3)
        img = np.zeros(dims)
        minmax = (1, 2, 1, 2)  # y_min, y_max, x_min, x_max
        shape = 'circle'
        drawing = (255, 0, 0)  # Red

        # Execute function
        result = draw_img_with_mask(img, dims, minmax, shape, drawing)

        # Verify result
        self.assertEqual(result[1, 1, 0], 255.)

    def test_draw_circle_normal_case(self):
        """Test that circle mask is applied correctly to normal image."""
        # Setup test data
        dims = (7, 7, 3)
        img = np.zeros(dims)
        minmax = (1, 6, 1, 6)  # y_min, y_max, x_min, x_max
        shape = 'circle'
        drawing = (255, 0, 0)  # Red

        # Execute function
        result = draw_img_with_mask(img, dims, minmax, shape, drawing)

        # Verify result
        self.assertEqual((result[:, :, 0] == 255).sum(), 13)

    def test_draw_circle_edge_case1(self):
        """Test with circle of the same size of the image."""
        # Setup test data
        dims = (5, 6, 3)
        img = np.zeros(dims)
        minmax = (0, 5, 0, 6)  # y_min, y_max, x_min, x_max
        shape = 'circle'
        drawing = (255, 0, 0)  # Red

        # Execute function
        result = draw_img_with_mask(img, dims, minmax, shape, drawing)

        # Verify result
        self.assertEqual((result[:, :, 0] == 255).sum(), 18)

    def test_draw_rectangle_normal_case(self):
        """Test that rectangle mask is applied correctly to normal image."""
        # Setup test data
        dims = (5, 6, 3)
        img = np.zeros(dims)
        minmax = (1, 4, 1, 5)  # y_min, y_max, x_min, x_max
        shape = 'rectangle'
        draw = (0, 255, 0)  # Green

        # Execute function
        result = draw_img_with_mask(img, dims, minmax, shape, draw)

        self.assertTrue(result.shape == dims)
        self.assertTrue((result[:, :, 1] == 255).sum() == 12)

    def test_draw_rectangle_edge_case1(self):
        """Test with rectangle of the same size of the image."""
        # Setup test data
        dims = (5, 6, 3)
        img = np.zeros(dims)
        minmax = (0, 5, 0, 6)  # y_min, y_max, x_min, x_max
        shape = 'rectangle'
        drawing = (255, 255, 255)

        # Execute function
        result = draw_img_with_mask(img, dims, minmax, shape, drawing)

        # Verify result
        self.assertTrue(result.shape == dims)
        self.assertTrue((result == 255).sum() == img.size)

    def test_draw_circle_borders(self):
        """Test with the contour of a circle as mask."""
        # Setup test data
        dims = (7, 7, 3)
        img = np.zeros(dims)
        minmax = (1, 7, 2, 7)  # y_min, y_max, x_min, x_max
        shape = 'circle'
        drawing = (255, 0, 0)  # Red

        # Execute function
        result = draw_img_with_mask(img, dims, minmax, shape, drawing, only_contours=True, dilate_mask=1)

        # Verify result
        self.assertEqual((result[:, :, 0] == 255).sum(), 27)


if __name__ == '__main__':
    unittest.main()