"""Module for computing shape descriptors from binary images.

This module provides a framework for calculating various geometric and statistical
descriptors of shapes in binary images through configurable dictionaries and a core class.
Supported metrics include area, perimeter, axis lengths, orientation, and more.
Descriptor computation is controlled via category dictionaries (e.g., `descriptors_categories`)
and implemented as methods in the ShapeDescriptors class.

Classes
-------
ShapeDescriptors : Class to compute various descriptors for a binary image

Notes
-----
Relies on OpenCV and NumPy for image processing operations.
Shape descriptors: The following names, lists and computes all the variables describing a shape in a binary image.
If you want to allow the software to compute another variable:
1) In the following dicts and list, you need to:
        add the variable name and whether to compute it (True/False) by default
2) In the ShapeDescriptors class:
        add a method to compute that variable
3) In the init method of the ShapeDescriptors class
    attribute a None value to the variable that store it
    add a if condition in the for loop to compute that variable when its name appear in the wanted_descriptors_list
"""
import cv2
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from cellects.utils.utilitarian import translate_dict, smallest_memory_array
from cellects.utils.formulas import (get_inertia_axes, get_standard_deviations, get_skewness, get_kurtosis,
                                     get_newly_explored_area)

descriptors_categories = {'area': True, 'perimeter': False, 'circularity': False, 'rectangularity': False,
                          'total_hole_area': False, 'solidity': False, 'convexity': False, 'eccentricity': False,
                          'euler_number': False, 'standard_deviation_xy': False, 'skewness_xy': False,
                          'kurtosis_xy': False, 'major_axes_len_and_angle': True, 'iso_digi_analysis': False,
                          'oscilacyto_analysis': False,
                          'fractal_analysis': False
                          }

descriptors_names_to_display = ['Area', 'Perimeter', 'Circularity', 'Rectangularity', 'Total hole area',
                                'Solidity', 'Convexity', 'Eccentricity', 'Euler number', 'Standard deviation xy',
                                'Skewness xy', 'Kurtosis xy', 'Major axes lengths and angle',
                                'Growth transitions', 'Oscillations',
                                'Minkowski dimension'
                                ]

from_shape_descriptors_class = {'area': True, 'perimeter': False, 'circularity': False, 'rectangularity': False,
               'total_hole_area': False, 'solidity': False, 'convexity': False, 'eccentricity': False,
               'euler_number': False, 'standard_deviation_y': False, 'standard_deviation_x': False,
               'skewness_y': False, 'skewness_x': False, 'kurtosis_y': False, 'kurtosis_x': False,
               'major_axis_len': True, 'minor_axis_len': True, 'axes_orientation': True
                               }

length_descriptors = ['perimeter', 'major_axis_len', 'minor_axis_len']
area_descriptors = ['area', 'area_total', 'total_hole_area', 'newly_explored_area', 'final_area']

descriptors = deepcopy(from_shape_descriptors_class)
descriptors.update({'minkowski_dimension': False})

def compute_one_descriptor_per_frame(binary_vid: NDArray[np.uint8], arena_label: int, timings: NDArray,
                                     descriptors_dict: dict, output_in_mm: bool, pixel_size: float,
                                     do_fading: bool, save_coord_specimen:bool):
    """
    Computes descriptors for each frame in a binary video and returns them as a DataFrame.

    Parameters
    ----------
    binary_vid : NDArray[np.uint8]
        The binary video data where each frame is a 2D array.
    arena_label : int
        Label for the arena in the video.
    timings : NDArray
        Array of timestamps corresponding to each frame.
    descriptors_dict : dict
        Dictionary containing the descriptors to be computed.
    output_in_mm : bool, optional
        Flag indicating if output should be in millimeters. Default is False.
    pixel_size : float, optional
        Size of a pixel in the video when `output_in_mm` is True. Default is None.
    do_fading : bool, optional
        Flag indicating if the fading effect should be applied. Default is False.
    save_coord_specimen : bool, optional
        Flag indicating if the coordinates of specimens should be saved. Default is False.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the descriptors for each frame in the video.

    Notes
    -----
    For large inputs, consider pre-allocating memory for efficiency.
    The `save_coord_specimen` flag will save coordinate data to a file.

    Examples
    --------
    >>> binary_vid = np.ones((10, 640, 480), dtype=np.uint8)
    >>> timings = np.arange(10)
    >>> descriptors_dict = {'area': True, 'perimeter': True}
    >>> result = compute_one_descriptor_per_frame(binary_vid, 1, timings, descriptors_dict)
    >>> print(result.head())
       arena  time  area  perimeter
    0      1     0     0          0
    1      1     1     0          0
    2      1     2     0          0
    3      1     3     0          0
    4      1     4     0          0

    >>> binary_vid = np.ones((5, 640, 480), dtype=np.uint8)
    >>> timings = np.arange(5)
    >>> descriptors_dict = {'area': True, 'perimeter': True}
    >>> result = compute_one_descriptor_per_frame(binary_vid, 2, timings,
    ...                                            descriptors_dict,
    ...                                            output_in_mm=True,
    ...                                            pixel_size=0.1)
    >>> print(result.head())
       arena  time  area  perimeter
    0      2     0    0         0.0
    1      2     1    0         0.0
    2      2     2    0         0.0
    3      2     3    0         0.0
    4      2     4    0         0.0
    """
    dims = binary_vid.shape
    all_descriptors, to_compute_from_sd, length_measures, area_measures = initialize_descriptor_computation(descriptors_dict)
    one_row_per_frame = pd.DataFrame(np.zeros((dims[0], 2 + len(all_descriptors))),
                                          columns=['arena', 'time'] + all_descriptors)
    one_row_per_frame['arena'] = [arena_label] * dims[0]
    one_row_per_frame['time'] = timings
    for t in np.arange(dims[0]):
        SD = ShapeDescriptors(binary_vid[t, :, :], to_compute_from_sd)
        for descriptor in to_compute_from_sd:
            one_row_per_frame.loc[t, descriptor] = SD.descriptors[descriptor]
    if save_coord_specimen:
        np.save(f"coord_specimen{arena_label}_t{dims[0]}_y{dims[1]}_x{dims[2]}.npy",
                smallest_memory_array(np.nonzero(binary_vid), "uint"))
    # Adjust descriptors scale if output_in_mm is specified
    if do_fading:
        one_row_per_frame['newly_explored_area'] = get_newly_explored_area(binary_vid)
    if output_in_mm:
        one_row_per_frame = scale_descriptors(one_row_per_frame, pixel_size,
                                                   length_measures, area_measures)
    return one_row_per_frame


def compute_one_descriptor_per_colony(binary_vid: NDArray[np.uint8], arena_label: int, timings: NDArray,
                                      descriptors_dict: dict, output_in_mm: bool, pixel_size: float,
                                      do_fading: bool, min_colony_size: int, save_coord_specimen: bool):
    dims = binary_vid.shape
    all_descriptors, to_compute_from_sd, length_measures, area_measures = initialize_descriptor_computation(
        descriptors_dict)
    # Objective: create a matrix with 4 columns (time, y, x, colony) containing the coordinates of all colonies
    # against time
    max_colonies = 0
    for t in np.arange(dims[0]):
        nb, shapes = cv2.connectedComponents(binary_vid[t, :, :])
        max_colonies = np.max((max_colonies, nb))

    time_descriptor_colony = np.zeros((dims[0], len(to_compute_from_sd) * max_colonies * dims[0]),
                                      dtype=np.float32)  # Adjust max_colonies
    colony_number = 0
    colony_id_matrix = np.zeros(dims[1:], dtype=np.uint64)
    coord_colonies = []
    centroids = []

    # pat_tracker = PercentAndTimeTracker(dims[0], compute_with_elements_number=True)
    for t in tqdm(np.arange(dims[0])):
        # We rank colonies in increasing order to make sure that the larger colony issued from a colony division
        # keeps the previous colony name.
        # shapes, stats, centers = cc(binary_vid[t, :, :])
        nb, shapes, stats, centers = cv2.connectedComponentsWithStats(binary_vid[t, :, :])
        true_colonies = np.nonzero(stats[:, 4] >= min_colony_size)[0][1:]
        # Consider that shapes bellow 3 pixels are noise. The loop will stop at nb and not compute them

        # current_percentage, eta = pat_tracker.get_progress(t, element_number=nb)
        # logging.info(f"Arena n°{arena_label}, Colony descriptors computation: {current_percentage}%{eta}")

        updated_colony_names = np.zeros(1, dtype=np.uint32)
        for colon_i in true_colonies:  # 120)):# #92
            current_colony_img = shapes == colon_i
            if current_colony_img.sum() >= 4:
                current_colony_img = current_colony_img.astype(np.uint8)

                # I/ Find out which names the current colony had at t-1
                colony_previous_names = np.unique(current_colony_img * colony_id_matrix)
                colony_previous_names = colony_previous_names[colony_previous_names != 0]
                # II/ Find out if the current colony name had already been analyzed at t
                # If there no match with the saved colony_id_matrix, assign colony ID
                if t == 0 or len(colony_previous_names) == 0:
                    # logging.info("New colony")
                    colony_number += 1
                    colony_names = [colony_number]
                # If there is at least 1 match with the saved colony_id_matrix, we keep the colony_previous_name(s)
                else:
                    colony_names = colony_previous_names.tolist()
                # Handle colony division if necessary
                if np.any(np.isin(updated_colony_names, colony_names)):
                    colony_number += 1
                    colony_names = [colony_number]

                # Update colony ID matrix for the current frame
                coords = np.nonzero(current_colony_img)
                colony_id_matrix[coords[0], coords[1]] = colony_names[0]

                # Add coordinates to coord_colonies
                time_column = np.full(coords[0].shape, t, dtype=np.uint32)
                colony_column = np.full(coords[0].shape, colony_names[0], dtype=np.uint32)
                coord_colonies.append(np.column_stack((time_column, colony_column, coords[0], coords[1])))

                # Calculate centroid and add to centroids list
                centroid_x, centroid_y = centers[colon_i, :]
                centroids.append((t, colony_names[0], centroid_y, centroid_x))

                # Compute shape descriptors
                SD = ShapeDescriptors(current_colony_img, to_compute_from_sd)
                # descriptors = list(SD.descriptors.values())
                descriptors = SD.descriptors
                # Adjust descriptors if output_in_mm is specified
                if output_in_mm:
                    descriptors = scale_descriptors(descriptors, pixel_size, length_measures, area_measures)
                # Store descriptors in time_descriptor_colony
                descriptor_index = (colony_names[0] - 1) * len(to_compute_from_sd)
                time_descriptor_colony[t, descriptor_index:(descriptor_index + len(descriptors))] = list(
                    descriptors.values())

                updated_colony_names = np.append(updated_colony_names, colony_names)

        # Reset colony_id_matrix for the next frame
        colony_id_matrix *= binary_vid[t, :, :]
    if len(centroids) > 0:
        centroids = np.array(centroids, dtype=np.float32)
    else:
        centroids = np.zeros((0, 4), dtype=np.float32)
    time_descriptor_colony = time_descriptor_colony[:, :(colony_number * len(to_compute_from_sd))]
    if len(coord_colonies) > 0:
        coord_colonies = np.vstack(coord_colonies)
        if save_coord_specimen:
            coord_colonies = pd.DataFrame(coord_colonies, columns=["time", "colony", "y", "x"])
            coord_colonies.to_csv(
                f"coord_colonies{arena_label}_{colony_number}col_t{dims[0]}_y{dims[1]}_x{dims[2]}.csv",
                sep=';', index=False, lineterminator='\n')

    centroids = pd.DataFrame(centroids, columns=["time", "colony", "y", "x"])
    centroids.to_csv(
        f"colony_centroids{arena_label}_{colony_number}col_t{dims[0]}_y{dims[1]}_x{dims[2]}.csv",
        sep=';', index=False, lineterminator='\n')

    # Format the final dataframe to have one row per time frame, and one column per descriptor_colony_name
    one_row_per_frame = pd.DataFrame({'arena': arena_label, 'time': timings,
                                      'area_total': binary_vid.sum((1, 2)).astype(np.float64)})

    if do_fading:
        one_row_per_frame['newly_explored_area'] = get_newly_explored_area(binary_vid)
    if output_in_mm:
        one_row_per_frame = scale_descriptors(one_row_per_frame, pixel_size)

    column_names = np.char.add(np.repeat(to_compute_from_sd, colony_number),
                               np.tile((np.arange(colony_number) + 1).astype(str), len(to_compute_from_sd)))
    time_descriptor_colony = pd.DataFrame(time_descriptor_colony, columns=column_names)
    one_row_per_frame = pd.concat([one_row_per_frame, time_descriptor_colony], axis=1)

    return one_row_per_frame

def initialize_descriptor_computation(descriptors_dict: dict) -> Tuple[list, list, list, list]:
    """

    Initialize descriptor computation based on available and requested descriptors.

    Parameters
    ----------
    descriptors_dict : dict
        A dictionary where keys are descriptor names and values are booleans indicating whether
        to compute the corresponding descriptor.

    Returns
    -------
    tuple
        A tuple containing four lists:
        - all_descriptors: List of all requested descriptor names.
        - to_compute_from_sd: Array of descriptor names that need to be computed from the shape descriptors class.
        - length_measures: Array of descriptor names that are length measures and need to be computed.
        - area_measures: Array of descriptor names that are area measures and need to be computed.

    Examples
    --------
    >>> descriptors_dict = {'perimeter': True, 'area': False}
    >>> all_descriptors, to_compute_from_sd, length_measures, area_measures = initialize_descriptor_computation(descriptors_dict)
    >>> print(all_descriptors, to_compute_from_sd, length_measures, area_measures)
    ['length'] ['length'] ['length'] []

    """
    available_descriptors_in_sd = list(from_shape_descriptors_class.keys())
    all_descriptors = []
    to_compute_from_sd = []
    for name, do_compute in descriptors_dict.items():
        if do_compute:
            all_descriptors.append(name)
            if np.isin(name, available_descriptors_in_sd):
                to_compute_from_sd.append(name)
    to_compute_from_sd = np.array(to_compute_from_sd)
    length_measures = to_compute_from_sd[np.isin(to_compute_from_sd, length_descriptors)]
    area_measures = to_compute_from_sd[np.isin(to_compute_from_sd, area_descriptors)]

    return all_descriptors, to_compute_from_sd, length_measures, area_measures

def scale_descriptors(descriptors_dict, pixel_size: float, length_measures: NDArray[str]=None, area_measures: NDArray[str]=None):
    """
    Scale the spatial descriptors in a dictionary based on pixel size.

    Parameters
    ----------
    descriptors_dict : dict
        Dictionary containing spatial descriptors.
    pixel_size : float
        Pixel size used for scaling.
    length_measures : numpy.ndarray, optional
        Array of descriptors that represent lengths. If not provided,
        they will be initialized.
    area_measures : numpy.ndarray, optional
        Array of descriptors that represent areas. If not provided,
        they will be initialized.

    Returns
    -------
    dict
        Dictionary with scaled spatial descriptors.

    Examples
    --------
    >>> from numpy import array as ndarray
    >>> descriptors_dict = {'length': ndarray([1, 2]), 'area': ndarray([3, 4])}
    >>> pixel_size = 0.5
    >>> scaled_dict = scale_descriptors(descriptors_dict, pixel_size)
    >>> print(scaled_dict)
    {'length': array([0.5, 1.]), 'area': array([1.58421369, 2.])}
    """
    if length_measures is None or area_measures is None:
        to_compute_from_sd = np.array(list(descriptors_dict.keys()))
        length_measures = to_compute_from_sd[np.isin(to_compute_from_sd, length_descriptors)]
        area_measures = to_compute_from_sd[np.isin(to_compute_from_sd, area_descriptors)]
    for descr in length_measures:
        descriptors_dict[descr] *= pixel_size
    for descr in area_measures:
        descriptors_dict[descr] *= np.sqrt(pixel_size)
    return descriptors_dict


class ShapeDescriptors:
    """
        This class takes :
        - a binary image of 0 and 1 drawing one shape
        - a list of descriptors to calculate from that image
        ["area", "perimeter", "circularity", "rectangularity", "total_hole_area", "solidity", "convexity",
         "eccentricity", "euler_number",

        "standard_deviation_y", "standard_deviation_x", "skewness_y", "skewness_x", "kurtosis_y", "kurtosis_x",
        "major_axis_len", "minor_axis_len", "axes_orientation",

        "mo", "contours", "min_bounding_rectangle", "convex_hull"]

        Be careful! mo, contours, min_bounding_rectangle, convex_hull,
        standard_deviations, skewness and kurtosis are not atomics
    https://www.researchgate.net/publication/27343879_Estimators_for_Orientation_and_Anisotropy_in_Digitized_Images
    """

    def __init__(self, binary_image, wanted_descriptors_list):
        """
        Class to compute various descriptors for a binary image.

        Parameters
        ----------
        binary_image : ndarray
            Binary image used to compute the descriptors.
        wanted_descriptors_list : list
            List of strings with the names of the wanted descriptors.

        Attributes
        ----------
        binary_image : ndarray
            The binary image.
        descriptors : dict
            Dictionary containing the computed descriptors.
        mo : float or None, optional
            Moment of inertia (default is `None`).
        area : int or None, optional
            Area of the object (default is `None`).
        contours : ndarray or None, optional
            Contours of the object (default is `None`).
        min_bounding_rectangle : tuple or None, optional
            Minimum bounding rectangle of the object (default is `None`).
        convex_hull : ndarray or None, optional
            Convex hull of the object (default is `None`).
        major_axis_len : float or None, optional
            Major axis length of the object (default is `None`).
        minor_axis_len : float or None, optional
            Minor axis length of the object (default is `None`).
        axes_orientation : float or None, optional
            Orientation of the axes (default is `None`).
        sx : float or None, optional
            Standard deviation in x-axis (default is `None`).
        kx : float or None, optional
            Kurtosis in x-axis (default is `None`).
        skx : float or None, optional
            Skewness in x-axis (default is `None`).
        perimeter : float or None, optional
            Perimeter of the object (default is `None`).
        convexity : float or None, optional
            Convexity of the object (default is `None`).

        Examples
        --------
        >>> binary_image = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        >>> wanted_descriptors_list = ["area", "perimeter"]
        >>> SD = ShapeDescriptors(binary_image, wanted_descriptors_list)
        >>> SD.descriptors
        {'area': np.uint64(9), 'perimeter': 8.0}
        """
        # Give a None value to each parameters whose presence is assessed before calculation (less calculus for speed)
        self.mo = None
        self.area = None
        self.contours = None
        self.min_bounding_rectangle = None
        self.convex_hull = None
        self.major_axis_len = None
        self.minor_axis_len = None
        self.axes_orientation = None
        self.sx = None
        self.kx = None
        self.skx = None
        self.perimeter = None
        self.convexity = None

        self.binary_image = binary_image
        if self.binary_image.dtype == 'bool':
            self.binary_image = self.binary_image.astype(np.uint8)

        self.descriptors = {i: np.empty(0, dtype=np.float64) for i in wanted_descriptors_list}
        self.get_area()

        for name in self.descriptors.keys():
            if name == "mo":
                self.get_mo()
                self.descriptors[name] = self.mo
            elif name == "area":
                self.descriptors[name] = self.area
            elif name == "contours":
                self.get_contours()
                self.descriptors[name] = self.contours
            elif name == "min_bounding_rectangle":
                self.get_min_bounding_rectangle()
                self.descriptors[name] = self.min_bounding_rectangle
            elif name == "major_axis_len":
                self.get_major_axis_len()
                self.descriptors[name] = self.major_axis_len
            elif name == "minor_axis_len":
                self.get_minor_axis_len()
                self.descriptors[name] = self.minor_axis_len
            elif name == "axes_orientation":
                self.get_inertia_axes()
                self.descriptors[name] = self.axes_orientation
            elif name == "standard_deviation_y":
                self.get_standard_deviations()
                self.descriptors[name] = self.sy
            elif name == "standard_deviation_x":
                self.get_standard_deviations()
                self.descriptors[name] = self.sx
            elif name == "skewness_y":
                self.get_skewness()
                self.descriptors[name] = self.sky
            elif name == "skewness_x":
                self.get_skewness()
                self.descriptors[name] = self.skx
            elif name == "kurtosis_y":
                self.get_kurtosis()
                self.descriptors[name] = self.ky
            elif name == "kurtosis_x":
                self.get_kurtosis()
                self.descriptors[name] = self.kx
            elif name == "convex_hull":
                self.get_convex_hull()
                self.descriptors[name] = self.convex_hull
            elif name == "perimeter":
                self.get_perimeter()
                self.descriptors[name] = self.perimeter
            elif name == "circularity":
                self.get_circularity()
                self.descriptors[name] = self.circularity
            elif name == "rectangularity":
                self.get_rectangularity()
                self.descriptors[name] = self.rectangularity
            elif name == "total_hole_area":
                self.get_total_hole_area()
                self.descriptors[name] = self.total_hole_area
            elif name == "solidity":
                self.get_solidity()
                self.descriptors[name] = self.solidity
            elif name == "convexity":
                self.get_convexity()
                self.descriptors[name] = self.convexity
            elif name == "eccentricity":
                self.get_eccentricity()
                self.descriptors[name] = self.eccentricity
            elif name == "euler_number":
                self.get_euler_number()
                self.descriptors[name] = self.euler_number

    """
        The following methods can be called to compute parameters for descriptors requiring it
    """

    def get_mo(self):
        """
        Get moments of a binary image.

        Calculate the image moments for a given binary image using OpenCV's
        `cv2.moments` function and then translate these moments into a formatted
        dictionary.

        Notes
        -----
        This function assumes the binary image has already been processed and is in a
        suitable format for moment calculation.

        Returns
        -------
        None

        Examples
        --------
       >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["mo"])
        >>> print(SD.mo["m00"])
        9.0
        """
        self.mo = translate_dict(cv2.moments(self.binary_image))

    def get_area(self):
        """
        Calculate the area of a binary image by summing its pixel values.

        This function computes the area covered by white pixels (value 1) in a binary image,
        which is equivalent to counting the number of 'on' pixels.

        Notes
        -----
        Sums values in `self.binary_image` and stores the result in `self.area`.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["area"])
        >>> print(SD.area)
        9.0
        """
        self.area = self.binary_image.sum()

    def get_contours(self):
        """
        Find and process the largest contour in a binary image.

        Retrieves contours from a binary image, calculates the Euler number,
        and identifies the largest contour based on its length.

        Notes
        -----
        This function modifies the internal state of the `self` object to store
        the largest contour and Euler number.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["euler_number"])
        >>> print(len(SD.contours))
        8
        """
        if self.area == 0:
            self.euler_number = 0.
            self.contours = np.array([], np.uint8)
        else:
            contours, hierarchy = cv2.findContours(self.binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            nb, shapes = cv2.connectedComponents(self.binary_image, ltype=cv2.CV_16U)
            self.euler_number = (nb - 1) - len(contours)
            self.contours = contours[0]
            if len(contours) > 1:
                all_lengths = np.zeros(len(contours))
                for i, contour in enumerate(contours):
                    all_lengths[i] = len(contour)
                self.contours = contours[np.argmax(all_lengths)]

    def get_min_bounding_rectangle(self):
        """
        Retrieve the minimum bounding rectangle from the contours of an image.

        This method calculates the smallest area rectangle that can enclose
        the object outlines present in the image, which is useful for
        object detection and analysis tasks.

        Notes
        -----
        - The bounding rectangle is calculated only if contours are available.
          If not, they will be retrieved first before calculating the rectangle.

        Raises
        ------
        RuntimeError
            If the contours are not available and cannot be retrieved,
            indicating a problem with the image or preprocessing steps.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["rectangularity"])
        >>> print(len(SD.min_bounding_rectangle))
        3
        """
        if self.area == 0:
            self.min_bounding_rectangle = np.array([], np.uint8)
        else:
            if self.contours is None:
                self.get_contours()
            if len(self.contours) == 0:
                self.min_bounding_rectangle = np.array([], np.uint8)
            else:
                self.min_bounding_rectangle = cv2.minAreaRect(self.contours)  # ((cx, cy), (width, height), angle)

    def get_inertia_axes(self):
        """
        Calculate and set the moments of inertia properties of an object.

        This function computes the centroid, major axis length,
        minor axis length, and axes orientation for an object. It
        first ensures that the moments of inertia (`mo`) attribute is available,
        computing them if necessary, before using the `get_inertia_axes` function.

        Returns
        -------
        None

            This method sets the following attributes:
            - `cx` : float
                The x-coordinate of the centroid.
            - `cy` : float
                The y-coordinate of the centroid.
            - `major_axis_len` : float
                The length of the major axis.
            - `minor_axis_len` : float
                The length of the minor axis.
            - `axes_orientation` : float
                The orientation angle of the axes.

        Raises
        ------
        ValueError
            If there is an issue with the moments of inertia computation.

        Notes
        -----
        This function modifies in-place the object's attributes related to its geometry.

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["major_axis_len"])
        >>> print(SD.axes_orientation)
        0.0
        """
        if self.mo is None:
            self.get_mo()
        if self.area == 0:
            self.cx, self.cy, self.major_axis_len, self.minor_axis_len, self.axes_orientation = 0, 0, 0, 0, 0
        else:
            self.cx, self.cy, self.major_axis_len, self.minor_axis_len, self.axes_orientation = get_inertia_axes(self.mo)

    def get_standard_deviations(self):
        """
        Calculate and store standard deviations along x and y (sx, sy).

        Notes
        -----
        Requires centroid and moments; values are stored in `self.sx` and `self.sy`.

        Returns
        -------
        None

        Examples
        --------
       >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["standard_deviation_x", "standard_deviation_y"])
        >>> print(SD.sx, SD.sy)
        0.816496580927726 0.816496580927726
        """
        if self.sx is None:
            if self.axes_orientation is None:
                self.get_inertia_axes()
            self.sx, self.sy = get_standard_deviations(self.mo, self.binary_image, self.cx, self.cy)

    def get_skewness(self):
        """
        Calculate and store skewness along x and y (skx, sky).

        This function computes the skewness about the x-axis and y-axis of
        an image. Skewness is a measure of the asymmetry of the probability
        distribution of values in an image.

        Notes
        -----
        Requires standard deviations; values are stored in `self.skx` and `self.sky`.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["skewness_x", "skewness_y"])
        >>> print(SD.skx, SD.sky)
        0.0 0.0
        """
        if self.skx is None:
            if self.sx is None:
                self.get_standard_deviations()

            self.skx, self.sky = get_skewness(self.mo, self.binary_image, self.cx, self.cy, self.sx, self.sy)

    def get_kurtosis(self):
        """
        Calculates the kurtosis of the image moments.

        Kurtosis is a statistical measure that describes the shape of
        a distribution's tails in relation to its overall shape. It is
        used here in the context of image moments analysis.

        Notes
        -----
        This function first checks if the kurtosis values have already been calculated.
        If not, it calculates them using the `get_kurtosis` function.

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["kurtosis_x", "kurtosis_y"])
        >>> print(SD.kx, SD.ky)
        1.5 1.5
        """
        if self.kx is None:
            if self.sx is None:
                self.get_standard_deviations()

            self.kx, self.ky = get_kurtosis(self.mo, self.binary_image, self.cx, self.cy, self.sx, self.sy)

    def get_convex_hull(self):
        """
        Compute and store the convex hull of the object's contour.

        Notes
        -----
        Stores the result in `self.convex_hull`. Computes contours if needed.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["solidity"])
        >>> print(len(SD.convex_hull))
        4
        """
        if self.area == 0:
            self.convex_hull = np.array([], np.uint8)
        else:
            if self.contours is None:
                self.get_contours()
            self.convex_hull = cv2.convexHull(self.contours)

    def get_perimeter(self):
        """
        Compute and store the contour perimeter length.

        Notes
        -----
        Computes contours if needed and stores the length in `self.perimeter`.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["perimeter"])
        >>> print(SD.perimeter)
        8.0
        """
        if self.area == 0:
            self.perimeter = 0.
        else:
            if self.contours is None:
                self.get_contours()
            if len(self.contours) == 0:
                self.perimeter = 0.
            else:
                self.perimeter = cv2.arcLength(self.contours, True)

    def get_circularity(self):
        """
        Compute and store circularity: 4πA / P².

        Notes
        -----
        Uses `self.area` and `self.perimeter`; stores result in `self.circularity`.

        Returns
        -------
        None

        Examples
        --------
         >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["circularity"])
        >>> print(SD.circularity)
        1.7671458676442586
        """
        if self.area == 0:
            self.circularity = 0.
        else:
            if self.perimeter is None:
                self.get_perimeter()
            if self.perimeter == 0:
                self.circularity = 0.
            else:
                self.circularity = (4 * np.pi * self.binary_image.sum()) / np.square(self.perimeter)

    def get_rectangularity(self):
        """
        Compute and store rectangularity: area / bounding-rectangle-area.

        Notes
        -----
        Uses `self.binary_image` and `self.min_bounding_rectangle`. Computes the MBR if needed.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["rectangularity"])
        >>> print(SD.rectangularity)
        2.25
        """
        if self.area == 0:
            self.rectangularity = 0.
        else:
            if self.min_bounding_rectangle is None:
                self.get_min_bounding_rectangle()
            bounding_rectangle_area = self.min_bounding_rectangle[1][0] * self.min_bounding_rectangle[1][1]
            if bounding_rectangle_area == 0:
                self.rectangularity = 0.
            else:
                self.rectangularity = self.binary_image.sum() / bounding_rectangle_area

    def get_total_hole_area(self):
        """
        Calculate the total area of holes in a binary image.

        This function uses connected component labeling to detect and
        measure the area of holes in a binary image.

        Returns
        -------
        float
            The total area of all detected holes in the binary image.

        Notes
        -----
        This function assumes that the binary image has been pre-processed
        and that holes are represented as connected components of zero
        pixels within the foreground

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["total_hole_area"])
        >>> print(SD.total_hole_area)
        0
        """
        nb, new_order = cv2.connectedComponents(1 - self.binary_image)
        if nb > 2:
            self.total_hole_area = (new_order > 1).sum()
        else:
            self.total_hole_area = 0.

    def get_solidity(self):
        """
        Compute and store solidity: contour area / convex hull area.

        Extended Summary
        ----------------
        The solidity is a dimensionless measure that compares the area of a shape to
        its convex hull. A solidity of 1 means the contour is fully convex, while a
        value less than 1 indicates concavities.

        Notes
        -----
        If the convex hull area is 0 or absent, solidity is set to 0.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["solidity"])
        >>> print(SD.solidity)
        1.0
        """
        if self.area == 0:
            self.solidity = 0.
        else:
            if self.convex_hull is None:
                self.get_convex_hull()
            if len(self.convex_hull) == 0:
                self.solidity = 0.
            else:
                hull_area = cv2.contourArea(self.convex_hull)
                if hull_area == 0:
                    self.solidity = 0.
                else:
                    self.solidity = cv2.contourArea(self.contours) / hull_area

    def get_convexity(self):
        """
        Compute and store convexity: convex hull perimeter / contour perimeter.

        Notes
        -----
        Requires `self.perimeter` and `self.convex_hull`.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["convexity"])
        >>> print(SD.convexity)
        1.0
        """
        if self.perimeter is None:
            self.get_perimeter()
        if self.convex_hull is None:
            self.get_convex_hull()
        if self.perimeter == 0 or len(self.convex_hull) == 0:
            self.convexity = 0.
        else:
            self.convexity = cv2.arcLength(self.convex_hull, True) / self.perimeter

    def get_eccentricity(self):
        """
        Compute and store eccentricity from major and minor axis lengths.

        Notes
        -----
        Calls `get_inertia_axes()` if needed and stores result in `self.eccentricity`.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["eccentricity"])
        >>> print(SD.eccentricity)
        0.0
        """
        self.get_inertia_axes()
        if self.major_axis_len == 0:
            self.eccentricity = 0.
        else:
            self.eccentricity = np.sqrt(1 - np.square(self.minor_axis_len / self.major_axis_len))

    def get_euler_number(self):
        """
        Ensure contours are computed; stores Euler number in `self.euler_number` via `get_contours()`.

        Returns
        -------
        None

        Notes
        -----
        Euler number is computed in `get_contours()` as `(components - 1) - len(contours)`.
        """
        if self.contours is None:
            self.get_contours()

    def get_major_axis_len(self):
        """
        Ensure the major axis length is computed and stored in `self.major_axis_len`.

        Returns
        -------
        None

        Notes
        -----
        Triggers `get_inertia_axes()` if needed.

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8), ["major_axis_len"])
        >>> print(SD.major_axis_len)
        2.8284271247461907
        """
        if self.major_axis_len is None:
            self.get_inertia_axes()

    def get_minor_axis_len(self):
        """
        Ensure the minor axis length is computed and stored in `self.minor_axis_len`.

        Returns
        -------
        None

        Notes
        -----
        Triggers `get_inertia_axes()` if needed.

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8), ["minor_axis_len"])
        >>> print(SD.minor_axis_len)
        0.0
        """
        if self.minor_axis_len is None:
            self.get_inertia_axes()

    def get_axes_orientation(self):
        """
        Ensure the axes orientation angle is computed and stored in `self.axes_orientation`.

        Returns
        -------
        None

        Notes
        -----
        Calls `get_inertia_axes()` if orientation is not yet computed.

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8), ["axes_orientation"])
        >>> print(SD.axes_orientation)
        1.5707963267948966
        """
        if self.axes_orientation is None:
            self.get_inertia_axes()

