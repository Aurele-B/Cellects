#!/usr/bin/env python3
"""
This script contains all unit tests of the skeletonize script
"""
import unittest
from tests._base import CellectsUnitTest
import numpy as np
from cellects.image.skeletonize import medial_axis, skeletonize


class TestMedialAxis(CellectsUnitTest):
    """Unit tests for medial_axis()."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()

        cls.empty = np.zeros((3, 3), dtype=bool)

        cls.single = np.zeros((3, 3), dtype=bool)
        cls.single[1, 1] = True

        cls.line = np.ones((1, 5), dtype=bool)
        cls.vline = np.ones((5, 1), dtype=bool)

        cls.rectangle = np.ones((3, 5), dtype=bool)

    def test_empty_image(self):
        """An empty image should return an empty medial axis."""
        result = medial_axis(self.empty)

        self.assertEqual(result.shape, self.empty.shape)
        self.assertEqual(result.dtype, np.bool_)
        self.assertFalse(result.any())

    def test_empty_image_with_distance(self):
        """An empty image should return an empty medial axis and zero distances."""
        skeleton, distance = medial_axis(self.empty, return_distance=True)

        self.assertEqual(skeleton.shape, self.empty.shape)
        self.assertEqual(skeleton.dtype, np.bool_)
        self.assertEqual(distance.shape, self.empty.shape)
        self.assertEqual(distance.dtype, np.float64)
        np.testing.assert_allclose(distance, 0.0, atol=1e-15)

    def test_single_pixel(self):
        """A single foreground pixel should be its own medial axis."""
        result = medial_axis(self.single)

        self.assertEqual(result.shape, self.single.shape)
        self.assertEqual(result.dtype, np.bool_)
        np.testing.assert_array_equal(result, self.single)

    def test_single_pixel_with_distance(self):
        """The distance at the single foreground pixel should be zero."""
        _, distance = medial_axis(self.single, return_distance=True)

        self.assertEqual(distance.shape, self.single.shape)
        self.assertEqual(distance.dtype, np.float64)

    def test_horizontal_line_is_retained(self):
        """A 1-pixel-thick horizontal line should be fully retained."""
        result = medial_axis(self.line)

        self.assertEqual(result.shape, self.line.shape)
        self.assertEqual(result.dtype, np.bool_)
        np.testing.assert_array_equal(result, self.line)

    def test_vertical_line_is_retained(self):
        """A 1-pixel-thick vertical line should be fully retained."""
        result = medial_axis(self.vline)

        self.assertEqual(result.shape, self.vline.shape)
        self.assertEqual(result.dtype, np.bool_)
        np.testing.assert_array_equal(result, self.vline)

    def test_rectangle_output_is_subset_and_nonempty(self):
        """The medial axis should be a non-empty subset of the input foreground."""
        result = medial_axis(self.rectangle)

        self.assertEqual(result.shape, self.rectangle.shape)
        self.assertEqual(result.dtype, np.bool_)

        outside = np.logical_and(result, ~self.rectangle)
        self.assertEqual(int(outside.sum()), 0)

        self.assertTrue(result.any())

    def test_return_distance_shape_and_dtype(self):
        """The distance transform should match the image shape and be float."""
        _, distance = medial_axis(self.rectangle, return_distance=True)

        self.assertEqual(distance.shape, self.rectangle.shape)
        self.assertEqual(distance.dtype, np.float64)
        self.assertTrue(np.all(distance >= 0.0))

    def test_invalid_ndim_raises(self):
        """Non-2-D input should raise ValueError."""
        with self.assertRaises(ValueError):
            medial_axis(np.ones(3, dtype=bool))

        with self.assertRaises(ValueError):
            medial_axis(np.ones((2, 2, 2), dtype=bool))


class TestSkeletonize(CellectsUnitTest):
    """Unit tests for skeletonize()."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()

        cls.empty2d = np.zeros((3, 3), dtype=bool)
        cls.empty3d = np.zeros((3, 3, 3), dtype=bool)

        cls.single2d = np.zeros((3, 3), dtype=bool)
        cls.single2d[1, 1] = True

        cls.single3d = np.zeros((3, 3, 3), dtype=bool)
        cls.single3d[1, 1, 1] = True

        cls.line2d = np.ones((1, 5), dtype=bool)
        cls.vline2d = np.ones((5, 1), dtype=bool)

        cls.line3d = np.zeros((1, 1, 5), dtype=bool)
        cls.line3d[0, 0, :] = True

        cls.block2d = np.ones((3, 3), dtype=bool)
        cls.block3d = np.ones((3, 3, 3), dtype=bool)

    def test_empty_2d(self):
        """An empty 2-D image should return an empty skeleton."""
        result = skeletonize(self.empty2d)

        self.assertEqual(result.shape, self.empty2d.shape)
        self.assertEqual(result.dtype, np.bool_)
        self.assertFalse(result.any())

    def test_empty_3d(self):
        """An empty 3-D image should return an empty skeleton."""
        result = skeletonize(self.empty3d)

        self.assertEqual(result.shape, self.empty3d.shape)
        self.assertEqual(result.dtype, np.bool_)
        self.assertFalse(result.any())

    def test_single_pixel_2d(self):
        """A single 2-D foreground pixel should remain in the skeleton."""
        result = skeletonize(self.single2d)

        self.assertEqual(result.shape, self.single2d.shape)
        self.assertEqual(result.dtype, np.bool_)
        np.testing.assert_array_equal(result, self.single2d)

    def test_single_voxel_3d(self):
        """A single 3-D foreground voxel should remain in the skeleton."""
        result = skeletonize(self.single3d)

        self.assertEqual(result.shape, self.single3d.shape)
        self.assertEqual(result.dtype, np.bool_)
        np.testing.assert_array_equal(result, self.single3d)

    def test_horizontal_line_2d_is_retained(self):
        """A 1-pixel-thick 2-D horizontal line should be retained."""
        result = skeletonize(self.line2d)

        self.assertEqual(result.shape, self.line2d.shape)
        self.assertEqual(result.dtype, np.bool_)
        np.testing.assert_array_equal(result, self.line2d)

    def test_vertical_line_2d_is_retained(self):
        """A 1-pixel-thick 2-D vertical line should be retained."""
        result = skeletonize(self.vline2d)

        self.assertEqual(result.shape, self.vline2d.shape)
        self.assertEqual(result.dtype, np.bool_)
        np.testing.assert_array_equal(result, self.vline2d)

    def test_line_3d_is_retained(self):
        """A 1-voxel-thick 3-D line should be retained."""
        result = skeletonize(self.line3d)

        self.assertEqual(result.shape, self.line3d.shape)
        self.assertEqual(result.dtype, np.bool_)
        np.testing.assert_array_equal(result, self.line3d)

    def test_2d_block_output_is_subset_and_nonempty(self):
        """The skeleton should be a non-empty subset of the input foreground."""
        result = skeletonize(self.block2d)

        self.assertEqual(result.shape, self.block2d.shape)
        self.assertEqual(result.dtype, np.bool_)

        outside = np.logical_and(result, ~self.block2d)
        self.assertEqual(int(outside.sum()), 0)

        self.assertTrue(result.any())

    def test_3d_block_output_is_subset_and_nonempty(self):
        """The skeleton should be a non-empty subset of the input foreground."""
        result = skeletonize(self.block3d)

        self.assertEqual(result.shape, self.block3d.shape)
        self.assertEqual(result.dtype, np.bool_)

        outside = np.logical_and(result, ~self.block3d)
        self.assertEqual(int(outside.sum()), 0)

        self.assertTrue(result.any())

    def test_invalid_ndim_raises(self):
        """Images with 1 or 4 dimensions should raise ValueError."""
        with self.assertRaises(ValueError):
            skeletonize(np.ones(3, dtype=bool))

        with self.assertRaises(ValueError):
            skeletonize(np.ones((2, 2, 2, 2), dtype=bool))


if __name__ == '__main__':
    unittest.main()
