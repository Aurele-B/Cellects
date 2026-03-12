#!/usr/bin/env python3
"""Tracks connected components in binary video sequences with shape analysis.

This module implements a framework for analyzing time-lapse binary images by identifying,
labeling, and characterizing individual connected components (e.g., colonies, cells, or any blob)
across multiple frames. It combines OpenCV-based component detection with custom descriptor
computation to generate time-resolved morphological measurements. Key operations include
component ID persistence across frames, centroid tracking, and conversion of raw binary data
into structured pandas DataFrames for downstream analysis.

Classes
-------
ConnectedComponentsTracking : Tracks connected components in 3D binary video arrays,
    computes shape descriptors per component, and generates time-resolved output tables.

Notes
-----
- Uses OpenCV's connected components detection with statistical properties (area, position)
- Relies on external shape descriptor computation from `cellects.image_analysis.shape_descriptors`
- Requires numpy for array operations and pandas for result organization
"""
import cv2
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from cellects.image_analysis.shape_descriptors import initialize_descriptor_computation, scale_descriptors, ShapeDescriptors
from cellects.utils.formulas import get_newly_explored_area



class ConnectedComponentsTracking:
    """
    Tracks connected components in 3D binary video arrays

    Include a method optimized for this unique task (track_cc) and a method for computing centroids and shape
    descriptors for each component (compute_one_descriptor_per_cc).
    """
    def __init__(self, binary_vid: NDArray[np.uint8], min_component_size: int):
        """
        Initialize connected components tracking with a binary video and minimum component size.

        Parameters
        ----------
        binary_vid : NDArray[np.uint8]
            3D binary image array (time x height x width) representing object masks.
        min_component_size : int
            Minimum number of pixels required for a valid connected component.

        """
        self.binary_vid = binary_vid
        self.dims = binary_vid.shape
        self.min_component_size = min_component_size
        self.descriptors_dict = None
        self.descriptors_table = None
        self.to_compute_from_sd = None
        self.length_measures = None
        self.area_measures = None
        self.current_components = None
        self.cc_final_number = None
        self.cc_id_matrix = None
        self.cc_coord = None
        self.cc_centroids = None
        self.current_cc_id = None
        self.centers = None
        self.updated_cc_names = None
        self.current_cc_img = None
        self.cc_names = None
        self.current_cc = None
        self.t = None
        
    def track_cc(self):
        """
        Track connected component evolution across all time frames in the binary video.

        This method processes each frame sequentially, identifying and tracking individual
        connected components while handling new colonies and divisions. Updates ID matrices
        and coordination data for visualization/tracking.

        Examples
        --------
        >>> tracker = ConnectedComponentsTracking(binary_video, min_size=10)
        >>> tracker.track_cc()
        """
        self.init_cc_tracking()
        for self.t in np.arange(self.dims[0]):
            self.get_current_connected_components()
            for self.current_cc in self.current_components:
                self.identify_current_cc()
            self.update_cc_id_matrix()
            
    def compute_one_descriptor_per_cc(self, arena_label: int, timings: NDArray,
                                      descriptors_dict: dict, output_in_mm: bool, pixel_size: float,
                                      do_fading: bool, save_coord_specimen: bool):
        """
        Compute and store shape descriptors for each tracked connected component.

        Parameters
        ----------
        arena_label : int
            Identifier for the experimental arena.
        timings : NDArray
            Time stamps corresponding to each frame in `binary_vid`.
        descriptors_dict : dict
            Mapping of descriptor names to ShapeDescriptors computation functions.
        output_in_mm : bool
            Whether to scale results to real-world units using pixel_size.
        pixel_size : float
            Conversion factor from pixels to millimeters (if `output_in_mm` is True).
        do_fading : bool
            Whether to compute newly explored area as a growth metric.
        save_coord_specimen : bool
            Whether to save component coordinates in CSV format.

        Returns
        -------
        NDArray | pandas.DataFrame
            Aggregated results with one row per time frame and descriptors per colony.

        Examples
        --------
        >>> descriptor_dict = {"area": compute_area, "perimeter": compute_perimeter}
        >>> results = tracker.compute_one_descriptor_per_cc(arena_label=1,
        ...                                                 timings=np.arange(10),
        ...                                                 descriptors_dict=descriptor_dict,
        ...                                                 output_in_mm=True,
        ...                                                 pixel_size=2.5,
        ...                                                 do_fading=False,
        ...                                                 save_coord_specimen=True)
        """
        self.descriptors_dict = descriptors_dict
        self.init_descriptors_table()
        self.init_cc_tracking()
        for self.t in np.arange(self.dims[0]):
            self.get_current_connected_components()
            for self.current_cc in self.current_components:
                self.identify_current_cc()
                self.get_cc_centroid()
                self.get_cc_descriptors(output_in_mm, pixel_size)
            self.update_cc_id_matrix()
        one_row_per_frame = self.format_and_save_results(arena_label, timings, output_in_mm, pixel_size,
                                      do_fading, save_coord_specimen)
        return one_row_per_frame
        
    def init_descriptors_table(self):
        """
        Initialize a matrix to store shape descriptors for all tracked components.

        This pre-allocates memory based on worst-case scenario (maximum possible colonies)
        to avoid dynamic resizing during tracking iterations.

        Notes
        -----
        Performance: Uses pre-allocation for efficiency in large-scale analysis.
        """
        # Create a matrix with 4 columns (time, y, x, colony) containing the coordinates of all colonies against time
        all_descriptors, self.to_compute_from_sd, self.length_measures, self.area_measures = initialize_descriptor_computation(
            self.descriptors_dict)
        max_colonies = 0
        for t in np.arange(self.dims[0]):
            nb, shapes = cv2.connectedComponents(self.binary_vid[t, :, :])
            max_colonies = np.max((max_colonies, nb))

        self.descriptors_table = np.zeros((self.dims[0], len(self.to_compute_from_sd) * max_colonies * self.dims[0]),
                                          dtype=np.float32)

    def init_cc_tracking(self):
        """
        Reset tracking data structures before processing a new binary video.

        Initializes ID matrices, coordination lists, and colony statistics counters
        to prepare for fresh component tracking.
        """
        self.cc_final_number = 0
        self.cc_id_matrix = np.zeros(self.dims[1:], dtype=np.uint64)
        self.cc_coord = []
        self.cc_centroids = []
        
    def get_current_connected_components(self):
        """
        Identify valid connected components in the current time frame.

        Uses OpenCV's `connectedComponentsWithStats` to extract component properties,
        filtering out small objects below `min_component_size`.

        Notes
        -----
        Performance: Avoids redundant computation by directly using binary mask data.
        """
        # We rank colonies in increasing order to make sure that the larger colony issued from a colony division
        # keeps the previous colony name.
        nb, self.current_cc_id, stats, self.centers = cv2.connectedComponentsWithStats(self.binary_vid[self.t, :, :])
        self.current_components = np.nonzero(stats[:, 4] >= self.min_component_size)[0][1:]
        # Consider that shapes bellow 3 pixels are noise. The loop will stop at nb and not compute them
        self.updated_cc_names = np.zeros(1, dtype=np.uint32)
        
    def identify_current_cc(self):
        """
        Assign unique IDs and track evolution of the current connected component.

        Matches colonies between frames to handle continuity, new formations,
        and divisions. Updates ID matrices and coordination records accordingly.

        Notes
        -----
        Caveat: Assumes larger colonies from divisions take priority in ID assignment.
        """
        self.current_cc_img = self.current_cc_id == self.current_cc
        self.current_cc_img = self.current_cc_img.astype(np.uint8)

        # I/ Find out which names the current colony had at t-1
        cc_previous_names = np.unique(self.current_cc_img * self.cc_id_matrix)
        cc_previous_names = cc_previous_names[cc_previous_names != 0]
        # II/ Find out if the current colony name had already been analyzed at t
        # If there no match with the saved self.cc_id_matrix, assign colony ID
        if self.t == 0 or len(cc_previous_names) == 0:
            # logging.info("New colony")
            self.cc_final_number += 1
            self.cc_names = [self.cc_final_number]
        # If there is at least 1 match with the saved self.cc_id_matrix, we keep the colony_previous_name(s)
        else:
            self.cc_names = cc_previous_names.tolist()
        # Handle colony division if necessary
        if np.any(np.isin(self.updated_cc_names, self.cc_names)):
            self.cc_final_number += 1
            self.cc_names = [self.cc_final_number]

        # Update colony ID matrix for the current frame
        coords = np.nonzero(self.current_cc_img)
        self.cc_id_matrix[coords[0], coords[1]] = self.cc_names[0]

        # Add coordinates to self.cc_coord
        time_column = np.full(coords[0].shape, self.t, dtype=np.uint32)
        cc_column = np.full(coords[0].shape, self.cc_names[0], dtype=np.uint32)
        self.cc_coord.append(np.column_stack((time_column, cc_column, coords[0], coords[1])))

        self.updated_cc_names = np.append(self.updated_cc_names, self.cc_names)

    def get_cc_centroid(self):
        """
        Compute the centroid coordinates of the current connected component.

        Stores results as (time, colony_id, y, x) tuples for later analysis or visualization.
        """
        # Calculate centroid and add to centroids list
        centroid_x, centroid_y = self.centers[self.current_cc, :]
        self.cc_centroids.append((self.t, self.cc_names[0], centroid_y, centroid_x))

    def get_cc_descriptors(self, output_in_mm: bool, pixel_size: float):
        """
        Calculate and store shape descriptors for the current connected component.

        Parameters
        ----------
        output_in_mm : bool
            Whether to scale results to real-world units using `pixel_size`.
        pixel_size : float
            Conversion factor from pixels to millimeters (if `output_in_mm` is True).

        Notes
        -----
        Dependency: Requires initialized ShapeDescriptors instance for computation.
        """
        # Compute shape descriptors
        SD = ShapeDescriptors(self.current_cc_img, self.to_compute_from_sd)
        descriptors = SD.descriptors
        # Adjust descriptors if output_in_mm is specified
        if output_in_mm:
            descriptors = scale_descriptors(descriptors, pixel_size, self.length_measures, self.area_measures)
        # Store descriptors in self.descriptors_table
        descriptor_index = (self.cc_names[0] - 1) * len(self.to_compute_from_sd)
        self.descriptors_table[self.t, descriptor_index:(descriptor_index + len(descriptors))] = list(
            descriptors.values())

    def update_cc_id_matrix(self):
        """
        Prepare the ID matrix for the next time frame by clearing obsolete data.

        Overwrites previous frame's IDs with zeros to prevent carryover artifacts,
        maintaining only current binary mask regions.
        """
        # Reset self.cc_id_matrix for the next frame
        self.cc_id_matrix *= self.binary_vid[self.t, :, :]
            
    def format_and_save_results(self, arena_label:int, timings: NDArray, output_in_mm: bool, pixel_size: float,
                                      do_fading: bool, save_coord_specimen: bool):
        """
        Format and export tracking results to structured data files.

        Parameters
        ----------
        arena_label : int
            Identifier for the experimental arena.
        timings : NDArray
            Time stamps corresponding to each frame in `binary_vid`.
        output_in_mm : bool
            Whether to scale results to real-world units using pixel_size.
        pixel_size : float
            Conversion factor from pixels to millimeters (if `output_in_mm` is True).
        do_fading : bool
            Whether to compute newly explored area as a growth metric.
        save_coord_specimen : bool
            Whether to save component coordinates in CSV format.

        Returns
        -------
        pandas.DataFrame
            Aggregated results with one row per time frame and descriptors per colony.

        Notes
        -----
        Performance: Uses efficient column concatenation for large descriptor sets.
        """
        if len(self.cc_centroids) > 0:
            self.cc_centroids = np.array(self.cc_centroids, dtype=np.float32)
        else:
            self.cc_centroids = np.zeros((0, 4), dtype=np.float32)
        self.descriptors_table = self.descriptors_table[:, :(self.cc_final_number * len(self.to_compute_from_sd))]
        if len(self.cc_coord) > 0:
            self.cc_coord = np.vstack(self.cc_coord)
            if save_coord_specimen:
                self.cc_coord = pd.DataFrame(self.cc_coord, columns=["time", "colony", "y", "x"])
                self.cc_coord.to_csv(
                    f"self.cc_coord{arena_label}_{self.cc_final_number}col_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.csv",
                    sep=';', index=False, lineterminator='\n')

        self.cc_centroids = pd.DataFrame(self.cc_centroids, columns=["time", "colony", "y", "x"])
        self.cc_centroids.to_csv(
            f"colony_centroids{arena_label}_{self.cc_final_number}col_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.csv",
            sep=';', index=False, lineterminator='\n')

        # Format the final dataframe to have one row per time frame, and one column per descriptor_colony_name
        one_row_per_frame = pd.DataFrame({'arena': arena_label, 'time': timings,
                                          'area_total': self.binary_vid.sum((1, 2)).astype(np.float64)})

        if do_fading:
            one_row_per_frame['newly_explored_area'] = get_newly_explored_area(self.binary_vid)
        if output_in_mm:
            one_row_per_frame = scale_descriptors(one_row_per_frame, pixel_size)
        if len(self.to_compute_from_sd) > 0:
            column_names = np.char.add(np.repeat(self.to_compute_from_sd, self.cc_final_number),
                                       np.tile((np.arange(self.cc_final_number) + 1).astype(str), len(self.to_compute_from_sd)))
            self.descriptors_table = pd.DataFrame(self.descriptors_table, columns=column_names)
            one_row_per_frame = pd.concat([one_row_per_frame, self.descriptors_table], axis=1)
        return one_row_per_frame
    