#!/usr/bin/env python3
"""Analyze oscillating clusters in 2D video data through flux tracking.

This module implements a class to track cluster dynamics by analyzing pixel flux changes over time.
The core functionality updates cluster identifiers, tracks periods of activity, and archives final data for completed clusters based on morphological analysis and contour boundaries.

Classes
ClusterFluxStudy : Updates flux information and tracks oscillating clusters in 2D space

Functions
update_flux : Processes flux changes to update cluster tracking and archive completed clusters

Notes
Uses cv2.connectedComponentsWithStats and custom distance calculations for boundary analysis
Maintains cumulative pixel data for active clusters during time-lapse processing
"""
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from cellects.image_analysis.morphological_operations import cross_33, get_minimal_distance_between_2_shapes


# def detect_oscillations_dynamics():



class ClusterFluxStudy:
    """
    This class updates the flux information and tracking of oscillating clusters.
    """
    def __init__(self, dims: Tuple):
        """
        This initializes with dimensions and manages
        pixel data, cluster IDs, and the total number of clusters.

        """
        self.dims = dims
        self.pixels_data = np.empty((4, 0), dtype=np.uint32)
        self.clusters_id = np.zeros(self.dims[1:], dtype=np.uint32)
        # self.alive_clusters_in_flux = np.empty(0, dtype=np.uint32)#list()
        self.cluster_total_number = 0

    def update_flux(self, t, contours: NDArray[np.uint8], current_flux: NDArray, period_tracking: NDArray[np.uint32], clusters_final_data: NDArray[np.float32]) -> Tuple[NDArray[np.uint32], NDArray[np.float32]]:
        """

        Update the flux information and track periods for cluster data.

        This function updates the tracking of clusters based on their flux, handles lost pixels,
        and archives data for clusters that are no longer active.

        Args:
            t: Time point at which the update is being performed.
            contours (NDArray[np.uint8]): Contour data representing the boundaries of clusters.
            current_flux (NDArray): Array containing the current flux information for each pixel.
            period_tracking (NDArray[np.uint32]): Array used to track the periods of clusters.
            clusters_final_data (NDArray[np.float32]): Array storing final data for archived clusters.

        Returns:
            Tuple containing updated period_tracking and clusters_final_data arrays.
                    - period_tracking (NDArray[np.uint32]): Updated tracking of periods for clusters.
                    - clusters_final_data (NDArray[np.float32]): Updated final data of archived clusters.

        """
        # Save the data from pixels that are not anymore in efflux
        lost = np.greater(self.clusters_id > 0, current_flux > 0)
        # Some pixels of that cluster faded, save their data
        lost_data = np.nonzero(lost)
        lost_data = np.array((period_tracking[lost],  # lost_coord[0], lost_coord[1],
                      self.clusters_id[lost], lost_data[0], lost_data[1]), dtype=np.uint32)
        # Add this to the array containing the data of each cluster that are still alive
        self.pixels_data = np.append(self.pixels_data, lost_data, axis=1)
        # Stop considering these pixels in period_tracking because they switched
        period_tracking[lost] = 0
        current_period_tracking = np.zeros(self.dims[1:], dtype=bool)
        for curr_clust_id in np.unique(current_flux)[1:]:
            # Get all pixels that were in the same flux previously
            curr_clust = current_flux == curr_clust_id
            already = self.clusters_id * curr_clust
            new = np.greater(curr_clust, self.clusters_id > 0)

            if not np.any(already):
                # It is an entirely new cluster:
                cluster_pixels = new
                self.cluster_total_number += 1
                cluster_name = self.cluster_total_number
            else:
                # Check whether parts of that cluster correspond to several clusters in clusters_id
                cluster_names = np.unique(already)[1:]
                # keep only one cluster name to gather clusters that just became connected
                cluster_name = np.min(cluster_names)
                # Put the same cluster name for new ones and every pixels that were
                # a part of a cluster touching the current cluster
                cluster_pixels = np.logical_or(np.isin(self.clusters_id, cluster_names), new)
                # If they are more than one,
                if len(cluster_names) > 1:
                    # Update these cluster names in pixels_data
                    self.pixels_data[1, np.isin(self.pixels_data[1, :], cluster_names)] = cluster_name
            # Update clusters_id
            self.clusters_id[cluster_pixels] = cluster_name
            # Update period_tracking
            current_period_tracking[curr_clust] = True

        period_tracking[current_period_tracking] += 1
        # Remove lost pixels from clusters_id
        self.clusters_id[lost] = 0
        # Find out which clusters are still alive or not
        still_alive_clusters = np.isin(self.pixels_data[1, :], np.unique(self.clusters_id))
        clusters_to_archive = np.unique(self.pixels_data[1, np.logical_not(still_alive_clusters)])
        # store their data in clusters_final_data
        clusters_data = np.zeros((len(clusters_to_archive), 6), dtype=np.float32)
        for clust_i, cluster in enumerate(clusters_to_archive):
            cluster_bool = self.pixels_data[1, :] == cluster
            cluster_size = np.sum(cluster_bool)
            cluster_img = np.zeros(self.dims[1:], dtype=np.uint8)
            cluster_img[self.pixels_data[2, cluster_bool], self.pixels_data[3, cluster_bool]] = 1
            nb, im, stats, centro = cv2.connectedComponentsWithStats(cluster_img)
            if np.any(cv2.dilate(cluster_img, kernel=cross_33, borderType=cv2.BORDER_CONSTANT, borderValue=0) * contours):
                minimal_distance = 1
            else:
                if cluster_size > 200:
                    eroded_cluster_img = cv2.erode(cluster_img, cross_33)
                    cluster_img = np.nonzero(cluster_img - eroded_cluster_img)
                    contours[cluster_img] = 2
                else:
                    contours[self.pixels_data[2, cluster_bool], self.pixels_data[3, cluster_bool]] = 2
                # Get the minimal distance between the border of the cell(s) (noted 1 in contours)
                # and the border of the cluster in the cell(s) (now noted 2 in contours)
                minimal_distance = get_minimal_distance_between_2_shapes(contours)
            data_to_save = np.array([[np.mean(self.pixels_data[0, cluster_bool]), t,
                                   cluster_size, minimal_distance, centro[1, 0], centro[1, 1]]], dtype=np.float32)
            clusters_data[clust_i,:] = data_to_save
        # and remove their data from pixels_data
        clusters_final_data = np.append(clusters_final_data, clusters_data, axis=0)
        self.pixels_data = self.pixels_data[:, still_alive_clusters]

        return period_tracking, clusters_final_data


