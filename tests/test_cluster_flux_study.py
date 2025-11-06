#!/usr/bin/env python3
"""
This script contains tests for ClusterFluxStudy class.
"""
from tests._base import CellectsUnitTest
from cellects.image_analysis.cluster_flux_study import ClusterFluxStudy
from cellects.image_analysis.morphological_operations import image_borders
import numpy as np
import cv2
from numba.typed import Dict as TDict


class TestClusterFluxStudy(CellectsUnitTest):
    """Test suite for ClusterFluxStudy class."""

    def setUp(self):
        """Set up test fixtures."""
        # Typical dimensions for testing (2D image of size 100x100)
        self.dims = (5, 10, 10)  # (channels, height, width)
        self.video = np.random.choice([0, 1, 2], 500, p=[0.9, 0.05, 0.05]).reshape((5, 10, 10)).astype(np.uint8)
        for t in range(self.dims[0]):
            self.video[t, ...] = cv2.dilate(self.video[t, ...], np.ones((3, 3), dtype=np.uint8), iterations=2)
        self.video[t, ...] = 0
        self.video = np.random.randint(0, 2, (5, 10, 10), dtype=np.uint8)  # (channels, height, width)
        self.cluster_study = ClusterFluxStudy(self.dims)
        self.period_tracking = np.zeros(self.dims[1:], dtype=np.uint32)
        self.clusters_final_data = np.empty((0, 6), dtype=np.float32)

    def test_initialization(self):
        """Test that initialization sets up correct data structures."""
        # Verify dimensions are stored correctly
        self.assertEqual(self.cluster_study.dims, self.dims)

        # Verify pixels_data is empty
        self.assertTrue(np.array_equal(self.cluster_study.pixels_data,
                                       np.empty((4, 0), dtype=np.uint32)))

        # Verify clusters_id is zeros
        expected_clusters = np.zeros((10, 10), dtype=np.uint32)
        self.assertTrue(np.array_equal(self.cluster_study.clusters_id,
                                       expected_clusters))

        # Verify cluster counter starts at 0
        self.assertEqual(self.cluster_study.cluster_total_number, 0)

    def test_update_flux_no_clusters(self):
        """Test update_flux with no clusters (all zeros)."""
        # Create test inputs
        t = 1.0
        contours = 1 - image_borders(self.dims[1:])
        current_flux = np.ones((10, 10), dtype=np.uint32)

        # Execute update_flux
        result_period, result_clusters = self.cluster_study.update_flux(
            t, contours, current_flux, self.period_tracking, self.clusters_final_data
        )

        # Verify nothing changes when no clusters exist
        self.assertTrue(np.array_equal(self.period_tracking, result_period))
        self.assertEqual(result_clusters.shape[0], 0)  # No clusters to archive
        self.assertEqual(self.cluster_study.cluster_total_number, 0)

    def test_update_flux(self):
        """Test update_flux with a single new cluster appearing."""

        contours = 1 - image_borders(self.dims[1:])
        for t in range(self.dims[0]):
            current_flux = self.video[t, :, :]

            # Execute update_flux
            self.period_tracking, self.clusters_final_data = self.cluster_study.update_flux(
                t, contours, current_flux, self.period_tracking, self.clusters_final_data)
        self.assertIsInstance(self.period_tracking, np.ndarray)
        self.assertIsInstance(self.clusters_final_data, np.ndarray)

        current_flux = np.zeros(self.dims[1:], dtype=np.uint8)
        self.period_tracking, self.clusters_final_data = self.cluster_study.update_flux(
            t, contours, current_flux, self.period_tracking, self.clusters_final_data)
        self.assertIsInstance(self.clusters_final_data, np.ndarray)