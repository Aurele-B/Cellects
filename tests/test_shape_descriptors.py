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
    ones_image = np.ones((3, 3), dtype=np.uint8)
    zeros_image = np.zeros((3, 3), dtype=np.bool_)
    sd1 = ShapeDescriptors(ones_image, [])
    sd0 = ShapeDescriptors(zeros_image, [])

    def test_empty_descriptor_list(self):
        """Ensure class handles empty descriptor list gracefully."""
        self.assertEqual(self.sd1.descriptors, {})

    def test_ones_moments(self):
        """Test get mo."""
        self.sd1.get_mo()
        self.assertEqual(len(self.sd1.mo), 24)

    def test_zeros_moments(self):
        """Test get mo."""
        self.sd0.get_mo()
        self.assertEqual(len(self.sd0.mo), 24)

    def test_ones_get_area(self):
        """Test get area."""
        self.sd1.get_area()
        self.assertEqual(self.sd1.area, 9)

    def test_zeros_get_area(self):
        """Test get area."""
        self.sd0.get_area()
        self.assertEqual(self.sd0.area, 0)

    def test_ones_get_contours(self):
        """Test get contours."""
        self.sd1.get_contours()
        self.assertEqual(len(self.sd1.contours), 8)

    def test_zeros_get_contours(self):
        """Test get contours."""
        self.sd0.get_contours()
        self.assertEqual(len(self.sd0.contours), 0)

    def test_ones_get_min_bounding_rectangle(self):
        """Test get_min_bounding_rectangle."""
        self.sd1.contours = None
        self.sd1.get_min_bounding_rectangle()
        self.assertEqual(len(self.sd1.min_bounding_rectangle), 3)

    def test_zeros_get_min_bounding_rectangle(self):
        """Test get_min_bounding_rectangle."""
        self.sd0.get_min_bounding_rectangle()
        self.assertEqual(len(self.sd0.min_bounding_rectangle), 0)

    def test_get_inertia_axes(self):
        """Test get_inertia_axes."""
        self.sd1.get_inertia_axes()
        self.assertEqual(self.sd1.axes_orientation, 0.)

    def test_get_standard_deviations(self):
        """Test get_standard_deviations."""
        self.sd1.sx = None
        self.sd1.axes_orientation = None
        self.sd1.get_standard_deviations()
        self.assertTrue(self.sd1.sx > 0)

    def test_get_skewness(self):
        """Test get_skewness."""
        self.sd1.sx = None
        self.sd1.get_skewness()
        self.assertTrue(self.sd1.skx == 0.)

    def test_get_kurtosis(self):
        """Test get_kurtosis."""
        self.sd1.get_kurtosis()
        self.assertTrue(self.sd1.kx > 0)

    def test_ones_get_convex_hull(self):
        """Test get_convex_hull."""
        self.sd1.get_convex_hull()
        self.assertTrue(len(self.sd1.convex_hull) == 4)

    def test_zeros_get_convex_hull(self):
        """Test get_convex_hull."""
        self.sd0.get_convex_hull()
        self.assertTrue(len(self.sd0.convex_hull) == 0)

    def test_ones_get_perimeter(self):
        """Test get_perimeter."""
        self.sd1.get_perimeter()
        self.assertTrue(self.sd1.perimeter == 8)

    def test_zeros_get_perimeter(self):
        """Test get_perimeter."""
        self.sd0.get_perimeter()
        self.assertTrue(self.sd0.perimeter == 0)

    def test_get_circularity(self):
        """Test get_circularity."""
        self.sd1.get_circularity()
        self.assertTrue(self.sd1.circularity > 1)

    def test_get_circularity_null_perimeter(self):
        """Test get_circularity."""
        self.sd0.get_circularity()
        self.assertTrue(self.sd0.circularity == 0)

    def test_get_circularity_type(self):
        """Test get_circularity."""
        self.sd1.get_circularity()
        self.assertTrue(isinstance(self.sd1.circularity, float))

    def test_get_rectangularity(self):
        """Test get_rectangularity."""
        self.sd1.get_rectangularity()
        self.assertTrue(isinstance(self.sd1.rectangularity, float))

    def test_get_rectangularity_null_rectangle(self):
        """Test get_rectangularity."""
        sd = ShapeDescriptors(np.zeros((3, 3), np.uint8), ["rectangularity"])
        self.assertTrue(sd.descriptors['rectangularity'] == 0)

    def test_get_total_hole_area_no_holes(self):
        """Test get_total_hole_area."""
        self.sd1.get_total_hole_area()
        self.assertEqual(self.sd1.total_hole_area, 0)
    def test_get_total_hole_area_two_holes(self):
        """Test get_total_hole_area."""
        im = np.ones((7, 7), np.uint8)
        im[0, :] = 0
        im[-1, :] = 0
        im[:, 0] = 0
        im[:, -1] = 0
        im[2, 2:4] = 0
        im[4, 3:5] = 0
        sd = ShapeDescriptors(im, [])
        sd.get_total_hole_area()
        self.assertEqual(sd.total_hole_area, 4)

    def test_ones_get_solidity(self):
        """Test get_solidity."""
        self.sd1.get_solidity()
        self.assertTrue(self.sd1.solidity == 1.)

    def test_zeros_get_solidity(self):
        """Test get_solidity."""
        self.sd0.get_solidity()
        self.assertTrue(self.sd0.solidity == 0.)

    def test_get_convexity(self):
        """Test get_convexity."""
        self.sd1.perimeter = None
        self.sd1.convex_hull = None
        self.sd1.get_convexity()
        self.assertTrue(self.sd1.convexity == 1.)

    def test_get_eccentricity(self):
        """Test get_eccentricity."""
        self.sd1.get_eccentricity()
        self.assertTrue(self.sd1.eccentricity == 0.)

    def test_get_euler_number(self):
        """Test get_euler_number."""
        self.sd1.contours = None
        self.sd1.get_euler_number()
        self.assertTrue(self.sd1.euler_number == 0)

    def test_get_major_axis_len(self):
        """Test get_major_axis_len."""
        self.sd1.major_axis_len = None
        self.sd1.get_major_axis_len()
        self.assertTrue(isinstance(self.sd1.major_axis_len, float))

    def test_get_minor_axis_len(self):
        """Test get_minor_axis_len."""
        self.sd1.minor_axis_len = None
        self.sd1.get_minor_axis_len()
        self.assertTrue(isinstance(self.sd1.minor_axis_len, float))

    def test_get_axes_orientation(self):
        """Test get_axes_orientation."""
        self.sd1.axes_orientation = None
        self.sd1.get_axes_orientation()
        self.assertTrue(isinstance(self.sd1.axes_orientation, float))

    def test_all_descriptors(self):
        """Test get_axes_orientation."""
        all_descriptors = list(from_shape_descriptors_class.keys())
        all_descriptors.append("mo")
        all_descriptors.append("area")
        all_descriptors.append("contours")
        all_descriptors.append("min_bounding_rectangle")
        all_descriptors.append("convex_hull")
        sd = ShapeDescriptors(self.ones_image, all_descriptors)
        self.assertTrue(isinstance(sd.descriptors, dict))

if __name__ == '__main__':
    unittest.main()