#!/usr/bin/env python3
"""
This script contains the class for studying oscillating clusters on videos in 2D
"""

from cv2 import (
    connectedComponents, connectedComponentsWithStats, MORPH_CROSS,
    getStructuringElement, CV_16U, erode, dilate, morphologyEx, MORPH_OPEN,
    MORPH_CLOSE, MORPH_GRADIENT, BORDER_CONSTANT, resize, imshow, waitKey,
    FONT_HERSHEY_SIMPLEX, putText)
from numpy import (
    append, float32, sum, mean, zeros, empty, array, nonzero, unique,
    isin, logical_or, logical_not, greater, uint8,
    uint32, min, any)
from cellects.image_analysis.morphological_operations import cross_33, get_minimal_distance_between_2_shapes


class ClusterFluxStudy:
    def __init__(self, dims):
        self.dims = dims

        self.pixels_data = empty((4, 0), dtype=uint32)
        self.clusters_id = zeros(self.dims[1:], dtype=uint32)
        # self.alive_clusters_in_flux = empty(0, dtype=uint32)#list()
        self.cluster_total_number = 0

    def update_flux(self, t, contours, current_flux, period_tracking, clusters_final_data):
        # flux_dir_changed = logical_xor(current_flux, self.clusters_id)
        # Save the data from pixels that are not anymore in efflux
        lost = greater(self.clusters_id > 0, current_flux > 0)
        # lost = logical_not(equal(current_flux > 0, self.clusters_id > 0))
        # lost = flux_dir_changed * (self.clusters_id > 0)
        # lost_coord = nonzero(lost)
        # if any(lost):
        # Some pixels of that cluster faded, save their data
        lost_data = nonzero(lost)
        lost_data = array((period_tracking[lost],  # lost_coord[0], lost_coord[1],
                      self.clusters_id[lost], lost_data[0], lost_data[1]), dtype=uint32)
        # Add this to the array containing the data of each cluster that are still alive
        self.pixels_data = append(self.pixels_data, lost_data, axis=1)
        # Stop considering these pixels in period_tracking because they switched
        period_tracking[lost] = 0
        current_period_tracking = zeros(self.dims[1:], dtype=bool)
        for curr_clust_id in unique(current_flux)[1:]:
            # Get all pixels that were in the same flux previously
            curr_clust = current_flux == curr_clust_id
            already = self.clusters_id * curr_clust
            new = greater(curr_clust, self.clusters_id > 0)
            # new = flux_dir_changed * (current_flux == curr_clust_id)

            if not any(already):
                # It is an entirely new cluster:
                cluster_pixels = new
                self.cluster_total_number += 1
                cluster_name = self.cluster_total_number
                # self.alive_clusters_in_flux = append(self.alive_clusters_in_flux, cluster_name)
            else:
                # Check whether parts of that cluster correspond to several clusters in clusters_id
                cluster_names = unique(already)[1:]
                # keep only one cluster name to gather clusters that just became connected
                cluster_name = min(cluster_names)
                # Put the same cluster name for new ones and every pixels that were
                # a part of a cluster touching the current cluster
                cluster_pixels = logical_or(isin(self.clusters_id, cluster_names), new)
                # new = self.clusters_id == cluster_names
                # If they are more than one,
                if len(cluster_names) > 1:
                    # Update these cluster names in pixels_data
                    self.pixels_data[1, isin(self.pixels_data[1, :], cluster_names)] = cluster_name
                    # self.pixels_data[self.pixels_data[1, :] == cluster_names] = cluster_name
                    # Update these cluster names in alive_clusters_in_flux: remove names that are not used anymore
                    # self.alive_clusters_in_flux = delete(self.alive_clusters_in_flux, isin(self.alive_clusters_in_flux, cluster_names[cluster_names != cluster_name]))
                    # cluster_names_to_remove = cluster_names.copy()
                    # cluster_names_to_remove = delete(cluster_names_to_remove,
                    #                                  nonzero(cluster_names_to_remove == cluster_name))
                    # # Remove the deleted clusters from the alive cluster list
                    # [self.alive_clusters_in_flux.remove(i) for i in cluster_names_to_remove if i in self.alive_clusters_in_flux]
            # Update clusters_id
            self.clusters_id[cluster_pixels] = cluster_name
            # Update period_tracking
            current_period_tracking[curr_clust] = True

        period_tracking[current_period_tracking] += 1
        # Remove lost pixels from clusters_id
        self.clusters_id[lost] = 0
        # self.alive_clusters_in_flux = self.alive_clusters_in_flux[isin(self.alive_clusters_in_flux, unique(self.clusters_id))]

        # Find out which clusters are still alive or not
        # still_alive_clusters = isin(self.pixels_data[1, :], self.alive_clusters_in_flux)
        still_alive_clusters = isin(self.pixels_data[1, :], unique(self.clusters_id))
        clusters_to_archive = unique(self.pixels_data[1, logical_not(still_alive_clusters)])
        # store their data in clusters_final_data
        for cluster in clusters_to_archive:
            cluster_bool = self.pixels_data[1, :] == cluster
            cluster_size = sum(cluster_bool)
            cluster_img = zeros(self.dims[1:], dtype=uint8)
            cluster_img[self.pixels_data[2, cluster_bool], self.pixels_data[3, cluster_bool]] = 1
            nb, im, stats, centro = connectedComponentsWithStats(cluster_img)
            if any(dilate(cluster_img, kernel=cross_33, borderType=BORDER_CONSTANT, borderValue=0) * contours):
                minimal_distance = 1
            else:
                if cluster_size > 200:
                    cluster_img = nonzero(morphologyEx(cluster_img, MORPH_GRADIENT, cross_33))
                    contours[cluster_img] = 2
                else:
                    contours[self.pixels_data[2, cluster_bool], self.pixels_data[3, cluster_bool]] = 2
                # Get the minimal distance between the border of the cell(s) (noted 1 in contours)
                # and the border of the cluster in the cell(s) (now noted 2 in contours)
                minimal_distance = get_minimal_distance_between_2_shapes(contours)
            data_to_save = array([[mean(self.pixels_data[0, cluster_bool]), t,
                                   cluster_size, minimal_distance, centro[1, 0], centro[1, 1]]], dtype=float32)
            # data_to_save = array([[mean(self.pixels_data[0, cluster_bool]), t,
            #                        cluster_size, minimal_distance]], dtype=float32)
            clusters_final_data = append(clusters_final_data, data_to_save,
                                         axis=0)  # ["mean_pixel_period", "total_size", "death_time"]
        # and remove their data from pixels_data
        self.pixels_data = self.pixels_data[:, still_alive_clusters]

        return period_tracking, clusters_final_data


