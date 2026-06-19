#!/usr/bin/env python3
"""
Test module for the `display` module in Cellects.

This test suite covers image, video, and plot displays.

Notes
-----
All test classes inherit from CellectsUnitTest.
"""
import unittest
from tests._base import CellectsUnitTest
from matplotlib.figure import Figure
from cellects.display.image import *
from cellects.display.param import *
from cellects.display.plot import *


class TestShow(CellectsUnitTest):
    """Test suite for the `show` function."""
    def test_show_with_interactive_mode_on(self):
        """Test if returns valid object."""
        img = np.random.rand(100, 100)
        fig, ax = show(img, interactive=False, show=False)
        self.assertTrue(isinstance(fig, Figure))

    def tearDown(self):
        """Close all figures."""
        plt.close("all")

class TestGetMplColormap(CellectsUnitTest):
    """Test suite for get_mpl_colormap."""

    def test_get_mpl_colormap_valid_input(self):
        """Verify correct output shape and data type with valid cmap name."""
        result = get_mpl_colormap("viridis")

        # Validate array structure
        self.assertEqual(result.shape, (256, 1, 3))
        self.assertTrue(np.issubdtype(result.dtype, np.integer))

        # Ensure all values are within byte range [0-255]
        self.assertTrue(np.all((result >= 0) & (result <= 255)))

    def test_get_mpl_colormap_invalid_cmap_name(self):
        """Ensure ValueError is raised for non-existent colormap names."""
        with self.assertRaises(ValueError):
            get_mpl_colormap("invalid_cmap_name")

    def test_get_mpl_colormap_alpha_exclusion(self):
        """Confirm RGBA -> RGB conversion in output array."""
        result = get_mpl_colormap("gray")

        # Test shape after alpha channel removal
        self.assertEqual(result.shape, (256, 1, 3))


class TestDisplayBoxes(CellectsUnitTest):
    """Test suite for display_boxes function."""

    def test_display_boxes(self):
        """Test if returns valid object."""
        binary_image = np.random.rand(10, 10)
        line_nb = display_boxes(binary_image, box_diameter=2, show=False)
        self.assertTrue(line_nb == 12)

    def tearDown(self):
        """Close all figures."""
        plt.close("all")


if __name__ == '__main__':
    unittest.main()