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
import logging
from cellects.image_analysis.morphological_operations import cross_33, get_minimal_distance_between_2_shapes, cc, get_contours, CompareNeighborsWithValue
from cellects.utils.utilitarian import smallest_memory_array, PercentAndTimeTracker
from psutil import virtual_memory


def detect_oscillations_dynamics(converted_video: NDArray, binary: NDArray[np.uint8], arena_label: int,
                                 starting_time: int, expected_oscillation_period: int,
                                 time_interval: int, minimal_oscillating_cluster_size:int,
                                 min_ram_free: float=1., lose_accuracy_to_save_memory: bool=False,
                                 save_coord_thickening_slimming: bool=True):
    """
    Detects oscillatory dynamics in a labeled arena from processed video data

    Parameters
    ----------
    converted_video : NDArray
        Processed intensity values of the input video as 3D/4D array (t,y,x[,c])
    binary : NDArray[np.uint8]
        Binary segmentation mask with 1 for active region and 0 otherwise
    arena_label : int
        Label identifier for the specific arena being analyzed in binary mask
    starting_time : int
        Timepoint index to start oscillation analysis from (earlier frames are ignored)
    expected_oscillation_period : int
        Expected average period of oscillations in seconds
    time_interval : int
        Sampling interval between consecutive video frames in seconds
    minimal_oscillating_cluster_size : int
        Minimum number of pixels required for a cluster to be considered an oscillation feature
    min_ram_free : float, optional (default=1.0)
        Minimum free RAM in GB that must remain available during processing
    lose_accuracy_to_save_memory : bool, optional (default=False)
        If True, uses low-precision calculations to reduce memory usage at the cost of accuracy
    save_coord_thickening_slimming : bool, optional (default=True)
        If True, saves detected cluster coordinates as .npy files

    Returns
    -------
    NDArray[np.int8]
        3D array where each pixel is labeled with 1=influx region, 2=efflux region, or 0=no oscillation

    Notes
    -----
    - Processes video data by calculating intensity gradients to detect directional oscillations
    - Memory-intensive operations use float16 when available RAM would otherwise be exceeded
    - Saves coordinate arrays if requested, which may consume significant disk space for large datasets
    """
    logging.info(f"Arena n°{arena_label}. Starting oscillation analysis.")
    dims = converted_video.shape
    oscillations_video = None
    if dims[0] > 1:
        period_in_frame_nb = int(expected_oscillation_period / time_interval)
        if period_in_frame_nb < 2:
            period_in_frame_nb = 2
        necessary_memory = dims[0] * dims[1] * dims[2] * 64 * 4 * 1.16415e-10
        available_memory = (virtual_memory().available >> 30) - min_ram_free
    if len(dims) == 4:
        converted_video = converted_video[:, :, :, 0]
        average_intensities = np.mean(converted_video, (1, 2))
        if lose_accuracy_to_save_memory or (necessary_memory > available_memory):
            oscillations_video = np.zeros(dims, dtype=np.float16)
            for cy in np.arange(dims[1]):
                for cx in np.arange(dims[2]):
                    oscillations_video[:, cy, cx] = np.round(
                        np.gradient(converted_video[:, cy, cx, ...] / average_intensities, period_in_frame_nb), 3).astype(np.float16)
        else:
            oscillations_video = np.gradient(converted_video / average_intensities[:, None, None], period_in_frame_nb, axis=0)
        oscillations_video = np.sign(oscillations_video)
        oscillations_video = oscillations_video.astype(np.int8)
        oscillations_video[binary == 0] = 0

        for t in np.arange(starting_time, dims[0]):
            oscillations_image = np.zeros(dims[1:], np.uint8)
            # Add in or ef if a pixel has at least 4 neighbor in or ef
            neigh_comp = CompareNeighborsWithValue(oscillations_video[t, :, :], connectivity=8, data_type=np.int8)
            neigh_comp.is_inf(0, and_itself=False)
            neigh_comp.is_sup(0, and_itself=False)
            # Not verified if influx is really influx (resp efflux)
            influx = neigh_comp.sup_neighbor_nb
            efflux = neigh_comp.inf_neighbor_nb

            # Only keep pixels having at least 4 positive (resp. negative) neighbors
            influx[influx <= 4] = 0
            efflux[efflux <= 4] = 0
            influx[influx > 4] = 1
            efflux[efflux > 4] = 1
            if np.any(influx) or np.any(efflux):
                influx, in_stats, in_centroids = cc(influx)
                efflux, ef_stats, ef_centroids = cc(efflux)
                # Only keep clusters larger than 'minimal_oscillating_cluster_size' pixels (smaller are considered as noise
                in_smalls = np.nonzero(in_stats[:, 4] < minimal_oscillating_cluster_size)[0]
                if len(in_smalls) > 0:
                    influx[np.isin(influx, in_smalls)] = 0
                ef_smalls = np.nonzero(ef_stats[:, 4] < minimal_oscillating_cluster_size)[0]
                if len(ef_smalls) > 0:
                    efflux[np.isin(efflux, ef_smalls)] = 0
                oscillations_image[influx > 0] = 1
                oscillations_image[efflux > 0] = 2
            oscillations_video[t, :, :] = oscillations_image
        oscillations_video[:starting_time, :, :] = 0
        if save_coord_thickening_slimming:
            np.save(
                f"coord_thickening{arena_label}_t{dims[0]}_y{dims[1]}_x{dims[2]}.npy",
                smallest_memory_array(np.nonzero(oscillations_video == 1), "uint"))
            np.save(
                f"coord_slimming{arena_label}_t{dims[0]}_y{dims[1]}_x{dims[2]}.npy",
                smallest_memory_array(np.nonzero(oscillations_video == 2), "uint"))
    return oscillations_video


