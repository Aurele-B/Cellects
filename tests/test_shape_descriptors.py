#!/usr/bin/env python3
"""Test suite for image shape descriptors.

This test module validates implementation of shape analysis features in the `ShapeDescriptors` class.
Tests cover atomic geometric properties (area, perimeter) and derived statistics (circularity, convexity),
as well as intermediate calculations like moment-of-inertia axes and contour detection. Includes unit tests
for individual descriptor methods and validation of edge case behavior across binary image inputs.
"""
import unittest
from cellects.image_analysis.shape_descriptors import *
from tests._base import CellectsUnitTest


class TestShapeDescriptors(CellectsUnitTest):
    """Test suite for ShapeDescriptors class."""
    binary_image = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    sd = ShapeDescriptors(binary_image, [])

    def test_empty_descriptor_list(self):
        """Ensure class handles empty descriptor list gracefully."""
        self.assertEqual(self.sd.descriptors, {})

    def test_moments(self):
        """Test get mo."""
        self.sd.get_mo()
        self.assertEqual(len(self.sd.mo), 24)

    def test_get_area(self):
        """Test get area."""
        self.sd.get_area()
        self.assertEqual(self.sd.area, 9)

    def test_get_contours(self):
        """Test get contours."""
        self.sd.get_contours()
        self.assertEqual(len(self.sd.contours), 8)

    def test_get_min_bounding_rectangle(self):
        """Test get_min_bounding_rectangle."""
        self.sd.get_min_bounding_rectangle()
        self.assertEqual(len(self.sd.min_bounding_rectangle), 3)

    def test_get_inertia_axes(self):
        """Test get_inertia_axes."""
        self.sd.get_inertia_axes()
        self.assertEqual(self.sd.axes_orientation, 0.)

    def test_get_standard_deviations(self):
        """Test get_standard_deviations."""
        self.sd.get_standard_deviations()
        self.assertTrue(self.sd.sx > 0)

    def test_get_skewness(self):
        """Test get_skewness."""
        self.sd.get_skewness()
        self.assertTrue(self.sd.skx == 0.)

    def test_get_kurtosis(self):
        """Test get_kurtosis."""
        self.sd.get_kurtosis()
        self.assertTrue(self.sd.kx > 0)

    def test_get_convex_hull(self):
        """Test get_convex_hull."""
        self.sd.get_convex_hull()
        self.assertTrue(len(self.sd.convex_hull) == 4)

    def test_get_perimeter(self):
        """Test get_perimeter."""
        self.sd.get_perimeter()
        self.assertTrue(self.sd.perimeter == 8)

    def test_get_circularity(self):
        """Test get_circularity."""
        self.sd.get_circularity()
        self.assertTrue(self.sd.circularity > 1)

    def test_get_circularity_type(self):
        """Test get_circularity."""
        self.sd.get_circularity()
        self.assertTrue(isinstance(self.sd.circularity, float))

    def test_get_rectangularity(self):
        """Test get_rectangularity."""
        self.sd.get_rectangularity()
        self.assertTrue(isinstance(self.sd.rectangularity, float))

    def test_get_total_hole_area(self):
        """Test get_total_hole_area."""
        self.sd.get_total_hole_area()
        self.assertTrue(self.sd.total_hole_area == 0)

    def test_get_solidity(self):
        """Test get_solidity."""
        self.sd.get_solidity()
        self.assertTrue(self.sd.solidity == 1.)

    def test_get_convexity(self):
        """Test get_convexity."""
        self.sd.get_convexity()
        self.assertTrue(self.sd.convexity == 1.)

    def test_get_eccentricity(self):
        """Test get_eccentricity."""
        self.sd.get_eccentricity()
        self.assertTrue(self.sd.eccentricity == 0.)

    def test_get_euler_number(self):
        """Test get_euler_number."""
        self.sd.get_euler_number()
        self.assertTrue(self.sd.euler_number == 0)

    def test_get_major_axis_len(self):
        """Test get_major_axis_len."""
        self.sd.get_major_axis_len()
        self.assertTrue(isinstance(self.sd.major_axis_len, float))

    def test_get_minor_axis_len(self):
        """Test get_minor_axis_len."""
        self.sd.get_minor_axis_len()
        self.assertTrue(isinstance(self.sd.minor_axis_len, float))

    def test_get_axes_orientation(self):
        """Test get_axes_orientation."""
        self.sd.get_axes_orientation()
        self.assertTrue(isinstance(self.sd.axes_orientation, float))


if __name__ == '__main__':
    unittest.main()