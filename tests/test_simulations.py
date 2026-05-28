#!/usr/bin/env python3
"""
Test suite for the cellects simulation module.

This module provides unit tests for the ``growing_colonies`` and
``moving_cells`` functions. Tests verify output shape and dtype,
ensure generated videos contain foreground pixels, check deterministic
behaviour for identical parameters, and confirm that cell positions
change over successive frames.
"""
import unittest
import numpy as np
from cellects.simulation import colonies
from cellects.simulation.migration import moving_cells
from tests._base import CellectsUnitTest


class TestGrowingColonies(CellectsUnitTest):
    """Test suite for the growing_colonies simulation function."""

    def test_output_shape_and_dtype(self):
        """Verify that the generated video has the expected shape and dtype."""
        video = colonies.growing_colonies()
        self.assertEqual((20, 1000, 1000, 3), video.shape)
        self.assertEqual(np.uint8, video.dtype)

    def test_video_contains_foreground_pixels(self):
        """Ensure that the video contains at least one non‑zero pixel (foreground)."""
        video = colonies.growing_colonies()
        self.assertTrue(video.any())


class TestMovingCells(CellectsUnitTest):
    """Test suite for moving_cells function."""

    def test_video_shape_and_dtype(self):
        """Video should have correct shape and dtype."""
        video = moving_cells(
            im_size=100,
            cell_size=10,
            cell_nb=2,
            frame_nb=3,
            delta=5,
            display=False,
        )
        expected_shape = (3, 100, 100, 3)
        self.assertEqual(expected_shape, video.shape)
        self.assertEqual(np.uint8, video.dtype)

    def test_deterministic_output(self):
        """Repeated calls with same parameters should yield identical video."""
        video1 = moving_cells(
            im_size=80,
            cell_size=8,
            cell_nb=1,
            frame_nb=2,
            delta=3,
            display=False,
        )
        video2 = moving_cells(
            im_size=80,
            cell_size=8,
            cell_nb=1,
            frame_nb=2,
            delta=3,
            display=False,
        )
        np.testing.assert_array_equal(video1, video2)

    def test_cells_move_over_time(self):
        """Frames should show movement of cells (pixel differences between first and last)."""
        video = moving_cells(
            im_size=50,
            cell_size=5,
            cell_nb=1,
            frame_nb=4,
            delta=4,
            display=False,
        )
        # Ensure at least one pixel changes between first and last frame
        diff_exists = np.any(video[0] != video[-1])
        self.assertTrue(diff_exists)

if __name__ == '__main__':
    unittest.main()