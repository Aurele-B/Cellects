#!/usr/bin/env python3
"""
This module provides methods to analyze and modify shapes in binary images.
It includes functions for comparing neighboring pixels, generating shape descriptors,
and performing morphological operations like expanding shapes and filling holes.

Classes
---------
CompareNeighborsWithValue : Class to compare neighboring pixels to a specified value

Functions
---------------
cc : Sort connected components according to size
make_gravity_field : Create a gradient field around shapes
find_median_shape : Generate median shape from multiple inputs
make_numbered_rays : Create numbered rays for analysis
CompareNeighborsWithFocal : Compare neighboring pixels to focal values
ShapeDescriptors : Generate shape descriptors using provided functions
get_radius_distance_against_time : Calculate radius distances over time
expand_until_one : Expand shapes until a single connected component remains
expand_and_rate_until_one : Expand and rate shapes until one remains
expand_until_overlap : Expand shapes until overlap occurs
dynamically_expand_to_fill_holes : Dynamically expand to fill holes in shapes
expand_smalls_toward_biggest : Expand smaller shapes toward largest component
change_thresh_until_one : Change threshold until one connected component remains
create_ellipse : Generate ellipse shape descriptors
get_rolling_window_coordinates_list : Get coordinates for rolling window operations

"""
import logging
from copy import deepcopy
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from cellects.utils.decorators import njit
from cellects.image_analysis.shape_descriptors import ShapeDescriptors
from cellects.utils.formulas import moving_average, bracket_to_uint8_image_contrast
from skimage.filters import threshold_otsu
from skimage.measure import label
from scipy.stats import linregress
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


cross_33 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
square_33 = np.ones((3, 3), np.uint8)


class CompareNeighborsWithValue:
    """
    CompareNeighborsWithValue class to summarize each pixel by comparing its neighbors to a value.

    This class analyzes pixels in a 2D array, comparing each pixel's neighbors
    to a specified value. The comparison can be equality, superiority,
    or inferiority, and neighbors can be the 4 or 8 nearest pixels based on
    the connectivity parameter.
    """
    def __init__(self, array: np.ndarray, connectivity: int=None, data_type: np.dtype=np.int8):
        """
        Initialize a class for array connectivity processing.

        This class processes arrays based on given connectivities, creating
        windows around the original data for both 1D and 2D arrays. Depending on
        the connectivity value (4 or 8), it creates different windows with borders.

        Parameters
        ----------
        array : ndarray
            Input array to process, can be 1D or 2D.
        connectivity : int, optional
            Connectivity type for processing (4 or 8), by default None.
        data_type : dtype, optional
            Data type for the array elements, by default np.int8.

        Attributes
        ----------
        array : ndarray
            The processed array based on the given data type.
        connectivity : int
            Connectivity value used for processing.
        on_the_right : ndarray
            Array with shifted elements to the right.
        on_the_left : ndarray
            Array with shifted elements to the left.
        on_the_bot : ndarray, optional
            Array with shifted elements to the bottom (for 2D arrays).
        on_the_top : ndarray, optional
            Array with shifted elements to the top (for 2D arrays).
        on_the_topleft : ndarray, optional
            Array with shifted elements to the top left (for 2D arrays).
        on_the_topright : ndarray, optional
            Array with shifted elements to the top right (for 2D arrays).
        on_the_botleft : ndarray, optional
            Array with shifted elements to the bottom left (for 2D arrays).
        on_the_botright : ndarray, optional
            Array with shifted elements to the bottom right (for 2D arrays).
        """
        array = array.astype(data_type)
        self.array = array
        self.connectivity = connectivity
        if len(self.array.shape) == 1:
            self.on_the_right = np.append(array[1:], array[-1])
            self.on_the_left = np.append(array[0], array[:-1])
        else:
            # Build 4 window of the original array, each missing one of the four borders
            # Grow each window with a copy of the last border at the opposite of the side a border have been deleted
            if self.connectivity == 4 or self.connectivity == 8:
                self.on_the_right = np.column_stack((array[:, 1:], array[:, -1]))
                self.on_the_left = np.column_stack((array[:, 0], array[:, :-1]))
                self.on_the_bot = np.vstack((array[1:, :], array[-1, :]))
                self.on_the_top = np.vstack((array[0, :], array[:-1, :]))
            if self.connectivity != 4:
                self.on_the_topleft = array[:-1, :-1]
                self.on_the_topright = array[:-1, 1:]
                self.on_the_botleft = array[1:, :-1]
                self.on_the_botright = array[1:, 1:]

                self.on_the_topleft = np.vstack((self.on_the_topleft[0, :], self.on_the_topleft))
                self.on_the_topleft = np.column_stack((self.on_the_topleft[:, 0], self.on_the_topleft))

                self.on_the_topright = np.vstack((self.on_the_topright[0, :], self.on_the_topright))
                self.on_the_topright = np.column_stack((self.on_the_topright, self.on_the_topright[:, -1]))

                self.on_the_botleft = np.vstack((self.on_the_botleft, self.on_the_botleft[-1, :]))
                self.on_the_botleft = np.column_stack((self.on_the_botleft[:, 0], self.on_the_botleft))

                self.on_the_botright = np.vstack((self.on_the_botright, self.on_the_botright[-1, :]))
                self.on_the_botright = np.column_stack((self.on_the_botright, self.on_the_botright[:, -1]))

    def is_equal(self, value, and_itself: bool=False):
        """
        Check equality of neighboring values in an array.

        This method compares the neighbors of each element in `self.array` to a given value.
        Depending on the dimensionality and connectivity settings, it checks different neighboring
        elements.

        Parameters
        ----------
        value : int or float
            The value to check equality with neighboring elements.
        and_itself : bool, optional
            If True, also check equality with the element itself. Defaults to False.

        Returns
        -------
        None

        Attributes (not standard Qt properties)
        --------------------------------------
        equal_neighbor_nb : ndarray of uint8
            Array that holds the number of equal neighbors for each element.

        Examples
        --------
        >>> matrix = np.array([[9, 0, 4, 6], [4, 9, 1, 3], [7, 2, 1, 4], [9, 0, 8, 5]], dtype=np.int8)
        >>> compare = CompareNeighborsWithValue(matrix, connectivity=4)
        >>> compare.is_equal(1)
        >>> print(compare.equal_neighbor_nb)
        [[0 0 1 0]
        [0 1 1 1]
        [0 1 1 1]
        [0 0 1 0]]
        """

        if len(self.array.shape) == 1:
            self.equal_neighbor_nb = np.sum((np.equal(self.on_the_right, value), np.equal(self.on_the_left, value)), axis=0)
        else:
            if self.connectivity == 4:
                self.equal_neighbor_nb =  np.dstack((np.equal(self.on_the_right, value), np.equal(self.on_the_left, value),
                                                 np.equal(self.on_the_bot, value), np.equal(self.on_the_top, value)))
            elif self.connectivity == 8:
                self.equal_neighbor_nb =  np.dstack(
                    (np.equal(self.on_the_right, value), np.equal(self.on_the_left, value),
                     np.equal(self.on_the_bot, value), np.equal(self.on_the_top, value),
                     np.equal(self.on_the_topleft, value), np.equal(self.on_the_topright, value),
                     np.equal(self.on_the_botleft, value), np.equal(self.on_the_botright, value)))
            else:
                self.equal_neighbor_nb =  np.dstack(
                    (np.equal(self.on_the_topleft, value), np.equal(self.on_the_topright, value),
                     np.equal(self.on_the_botleft, value), np.equal(self.on_the_botright, value)))
            self.equal_neighbor_nb = np.sum(self.equal_neighbor_nb, 2, dtype=np.uint8)

        if and_itself:
            self.equal_neighbor_nb[np.not_equal(self.array, value)] = 0

    def is_sup(self, value, and_itself=False):
        """
        Determine if pixels have more neighbors with higher values than a given threshold.

        This method computes the number of neighboring pixels that have values greater
        than a specified `value` for each pixel in the array. Optionally, it can exclude
        the pixel itself if its value is less than or equal to `value`.

        Parameters
        ----------
        value : int
            The threshold value used to determine if a neighboring pixel's value is greater.
        and_itself : bool, optional
            If True, exclude the pixel itself if its value is less than or equal to `value`.
            Defaults to False.

        Examples
        --------
        >>> matrix = np.array([[9, 0, 4, 6], [4, 9, 1, 3], [7, 2, 1, 4], [9, 0, 8, 5]], dtype=np.int8)
        >>> compare = CompareNeighborsWithValue(matrix, connectivity=4)
        >>> compare.is_sup(1)
        >>> print(compare.sup_neighbor_nb)
        [[3 3 2 4]
         [4 2 3 3]
         [4 2 3 3]
         [3 3 2 4]]
        """
        if len(self.array.shape) == 1:
            self.sup_neighbor_nb = (self.on_the_right > value).astype(self.array.dtype) + (self.on_the_left > value).astype(self.array.dtype)
        else:
            if self.connectivity == 4:
                self.sup_neighbor_nb =  np.dstack((self.on_the_right > value, self.on_the_left > value,
                                               self.on_the_bot > value, self.on_the_top > value))
            elif self.connectivity == 8:
                self.sup_neighbor_nb =  np.dstack((self.on_the_right > value, self.on_the_left > value,
                                               self.on_the_bot > value, self.on_the_top > value,
                                               self.on_the_topleft > value, self.on_the_topright > value,
                                               self.on_the_botleft > value, self.on_the_botright > value))
            else:
                self.sup_neighbor_nb =  np.dstack((self.on_the_topleft > value, self.on_the_topright > value,
                                               self.on_the_botleft > value, self.on_the_botright > value))

            self.sup_neighbor_nb = np.sum(self.sup_neighbor_nb, 2, dtype=np.uint8)
        if and_itself:
            self.sup_neighbor_nb[np.less_equal(self.array, value)] = 0

    def is_inf(self, value, and_itself=False):
        """
        is_inf(value and_itself=False)

        Determine the number of neighbors that are infinitely small relative to a given value,
        considering optional connectivity and exclusion of the element itself.

        Parameters
        ----------
        value : numeric
            The value to compare neighbor elements against.
        and_itself : bool, optional
            If True, excludes the element itself from being counted. Default is False.

        Examples
        --------
        >>> matrix = np.array([[9, 0, 4, 6], [4, 9, 1, 3], [7, 2, 1, 4], [9, 0, 8, 5]], dtype=np.int8)
        >>> compare = CompareNeighborsWithValue(matrix, connectivity=4)
        >>> compare.is_inf(1)
        >>> print(compare.inf_neighbor_nb)
        [[1 1 1 0]
         [0 1 0 0]
         [0 1 0 0]
         [1 1 1 0]]
        """
        if len(self.array.shape) == 1:
            self.inf_neighbor_nb = (self.on_the_right < value).astype(self.array.dtype) + (self.on_the_left < value).astype(self.array.dtype)
        else:
            if self.connectivity == 4:
                self.inf_neighbor_nb =  np.dstack((self.on_the_right < value, self.on_the_left < value,
                                               self.on_the_bot < value, self.on_the_top < value))
            elif self.connectivity == 8:
                self.inf_neighbor_nb =  np.dstack((self.on_the_right < value, self.on_the_left < value,
                                               self.on_the_bot < value, self.on_the_top < value,
                                               self.on_the_topleft < value, self.on_the_topright < value,
                                               self.on_the_botleft < value, self.on_the_botright < value))
            else:
                self.inf_neighbor_nb =  np.dstack((self.on_the_topleft < value, self.on_the_topright < value,
                                               self.on_the_botleft < value, self.on_the_botright < value))

            self.inf_neighbor_nb = np.sum(self.inf_neighbor_nb, 2, dtype=np.uint8)
        if and_itself:
            self.inf_neighbor_nb[np.greater_equal(self.array, value)] = 0


def cc(binary_img: NDArray[np.uint8]) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Processes a binary image to reorder and label connected components.

    This function takes a binary image, analyses the connected components,
    reorders them by size, ensures background is correctly labeled as 0,
    and returns the new ordered labels along with their statistics and centers.

    Parameters
    ----------
    binary_img : ndarray of uint8
        Input binary image with connected components.

    Returns
    -------
    new_order : ndarray of uint8, uint16 or uint32
        Image with reordered labels for connected components.
    stats : ndarray of ints
        Statistics for each component (x, y, width, height, area).
    centers : ndarray of floats
        Centers for each component (x, y).

    Examples
    --------
    >>> binary_img = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    >>> new_order, stats, centers = cc(binary_img)
    >>> print(stats)
    array([[0, 0, 3, 2, 4],
       [1, 0, 2, 2, 2]], dtype=int32)
    """
    number, img, stats, centers = cv2.connectedComponentsWithStats(binary_img, ltype=cv2.CV_16U)
    if number > 255:
        img_dtype = np.uint16
        if number > 65535:
            img_dtype = np.uint32
    else:
        img_dtype = np.uint8
    stats[:, 2] = stats[:, 0] + stats[:, 2]
    stats[:, 3] = stats[:, 1] + stats[:, 3]
    sorted_idx = np.argsort(stats[:, 4])[::-1]

    # Make sure that the first connected component (labelled 0) is the background and not the main shape
    size_ranked_stats = stats[sorted_idx, :]
    background = (size_ranked_stats[:, 0] == 0).astype(np.uint8) + (size_ranked_stats[:, 1] == 0).astype(np.uint8) + (
            size_ranked_stats[:, 2] == img.shape[1]).astype(np.uint8) + (
                         size_ranked_stats[:, 3] == img.shape[0]).astype(np.uint8)

    # background = ((size_ranked_stats[:, 0] == 0) & (size_ranked_stats[:, 1] == 0) & (size_ranked_stats[:, 2] == img.shape[1]) & (size_ranked_stats[:, 3] == img.shape[0]))

    touch_borders = np.nonzero(background > 2)[0]
    # if not isinstance(touch_borders, np.int64):
    #     touch_borders = touch_borders[0]
    # Most of the time, the background should be the largest shape and therefore has the index 0,
    # Then, if there is at least one shape touching more than 2 borders and having not the index 0, solve:
    if np.any(touch_borders != 0):
        # If there is only one shape touching borders, it means that background is not at its right position (i.e. 0)
        if len(touch_borders) == 1:
            # Then exchange that shape position with background position
            shape = sorted_idx[0]  # Store shape position in the first place
            back = sorted_idx[touch_borders[0]]  # Store back position in the first place
            sorted_idx[touch_borders[0]] = shape  # Put shape position at the previous place of back and conversely
            sorted_idx[0] = back
        # If there are two shapes, it means that the main shape grew sufficiently to reach at least 3 borders
        # We assume that it grew larger than background
        else:
            shape = sorted_idx[0]
            back = sorted_idx[1]
            sorted_idx[1] = shape
            sorted_idx[0] = back
            # Put shape position at the previous place of back and conversely
            

    stats = stats[sorted_idx, :]
    centers = centers[sorted_idx, :]

    new_order = np.zeros_like(binary_img, dtype=img_dtype)

    for i, val in enumerate(sorted_idx):
        new_order[img == val] = i
    return new_order, stats, centers


spot_size_coefficients = np.arange(0.75, 0.00, - 0.05)
spot_shapes = np.tile(["circle", "rectangle"], len(spot_size_coefficients))
spot_sizes = np.repeat(spot_size_coefficients, 2)


def shape_selection(binary_image:NDArray, several_blob_per_arena: bool, true_shape_number: int=None,
                    horizontal_size: int=None, spot_shape: str=None, bio_mask:NDArray=None, back_mask:NDArray=None):
    """
    Process the binary image to identify and validate shapes.

    This method processes a binary image to detect connected components,
    validate their sizes, and handle bio and back masks if specified.
    It ensures that the number of validated shapes matches the expected
    sample number or applies additional filtering if necessary.

    Args:
        use_bio_and_back_masks (bool): Whether to use bio and back masks
            during the processing. Default is False.

    Selects and validates the shapes of stains based on their size and shape.

    This method performs two main tasks:
    1. Removes stains whose horizontal size varies too much from a reference value.
    2. Determines the shape of each remaining stain and only keeps those that correspond to a reference shape.

    The method first removes stains whose horizontal size is outside the specified confidence interval. Then, it identifies shapes that do not correspond to a predefined reference shape and removes them as well.

    Args:
        horizontal_size (int): The expected horizontal size of the stains to use as a reference.
        shape (str): The shape type ('circle' or 'rectangle')
            that the stains should match. Other shapes are not currently supported.
        confint (float): The confidence interval as a decimal
            representing the percentage within which the size of the stains should fall.
        do_not_delete (NDArray, optional): An array of stain indices that should not be deleted.
            Default is None.

    """

    shape_number, shapes, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    do_not_delete = None
    if bio_mask is not None or back_mask is not None:
        if back_mask is not None:
            if np.any(shapes[back_mask]):
                shapes[np.isin(shapes, np.unique(shapes[back_mask]))] = 0
                shape_number, shapes, stats, centroids = cv2.connectedComponentsWithStats(
                    (shapes > 0).astype(np.uint8), connectivity=8)
        if bio_mask is not None:
            if np.any(shapes[bio_mask]):
                do_not_delete = np.unique(shapes[bio_mask])
                do_not_delete = do_not_delete[do_not_delete != 0]
    shape_number -= 1

    if not several_blob_per_arena and horizontal_size is not None:
        ordered_shapes = shapes.copy()
        if spot_shape is None:
            c_spot_shapes = spot_shapes
            c_spot_sizes = spot_sizes
        else:
            if spot_shape == 'circle':
                c_spot_shapes = spot_shapes[::2]
            else:
                c_spot_shapes = spot_shapes[::-2]
            c_spot_sizes = spot_sizes[::2]

        # shape_number = stats.shape[0]
        counter = 0
        while shape_number != true_shape_number and counter < len(spot_size_coefficients):
            shape = c_spot_shapes[counter]
            confint = c_spot_sizes[counter]
            # counter+=1;horizontal_size = self.spot_size; shape = self.parent.spot_shapes[counter];confint = self.parent.spot_size_confints[::-1][counter]
            # stats columns contain in that order:
            # - x leftmost coordinate of boundingbox
            # - y topmost coordinate of boundingbox
            # - The horizontal size of the bounding box.
            # - The vertical size of the bounding box.
            # - The total area (in pixels) of the connected component.

            # First, remove each stain which horizontal size varies too much from reference
            size_interval = [horizontal_size * (1 - confint), horizontal_size * (1 + confint)]
            cc_to_remove = np.argwhere(np.logical_or(stats[:, 2] < size_interval[0], stats[:, 2] > size_interval[1]))

            if do_not_delete is None:
                ordered_shapes[np.isin(ordered_shapes, cc_to_remove)] = 0
            else:
                ordered_shapes[np.logical_and(np.isin(ordered_shapes, cc_to_remove),
                                              np.logical_not(np.isin(ordered_shapes, do_not_delete)))] = 0

            # Second, determine the shape of each stain to only keep the ones corresponding to the reference shape
            validated_shapes = np.zeros(ordered_shapes.shape, dtype=np.uint8)
            validated_shapes[ordered_shapes > 0] = 1
            nb_components, ordered_shapes, stats, centroids = cv2.connectedComponentsWithStats(validated_shapes,
                                                                                               connectivity=8)
            if nb_components > 1:
                if shape == 'circle':
                    surf_interval = [np.pi * np.square(horizontal_size // 2) * (1 - confint),
                                     np.pi * np.square(horizontal_size // 2) * (1 + confint)]
                    cc_to_remove = np.argwhere(
                        np.logical_or(stats[:, 4] < surf_interval[0], stats[:, 4] > surf_interval[1]))
                elif shape == 'rectangle':
                    # If the smaller side is the horizontal one, use the user provided horizontal side
                    if np.argmin((np.mean(stats[1:, 2]), np.mean(stats[1:, 3]))) == 0:
                        surf_interval = [np.square(horizontal_size) * (1 - confint),
                                         np.square(horizontal_size) * (1 + confint)]
                        cc_to_remove = np.argwhere(
                            np.logical_or(stats[:, 4] < surf_interval[0], stats[:, 4] > surf_interval[1]))
                    # If the smaller side is the vertical one, use the median vertical length shape
                    else:
                        surf_interval = [np.square(np.median(stats[1:, 3])) * (1 - confint),
                                         np.square(np.median(stats[1:, 3])) * (1 + confint)]
                        cc_to_remove = np.argwhere(
                            np.logical_or(stats[:, 4] < surf_interval[0], stats[:, 4] > surf_interval[1]))
                else:
                    logging.info("Original blob shape not well written")

                if do_not_delete is None:
                    ordered_shapes[np.isin(ordered_shapes, cc_to_remove)] = 0
                else:
                    ordered_shapes[np.logical_and(np.isin(ordered_shapes, cc_to_remove),
                                                  np.logical_not(np.isin(ordered_shapes, do_not_delete)))] = 0
                # There was only that before:
                validated_shapes = np.zeros(ordered_shapes.shape, dtype=np.uint8)
                validated_shapes[np.nonzero(ordered_shapes)] = 1
                nb_components, ordered_shapes, stats, centroids = cv2.connectedComponentsWithStats(validated_shapes,
                                                                                                   connectivity=8)

            shape_number = nb_components - 1
            counter += 1
        
        if shape_number == true_shape_number:
            shapes = ordered_shapes
    if true_shape_number is None or shape_number == true_shape_number:
        validated_shapes = np.zeros(shapes.shape, dtype=np.uint8)
        validated_shapes[shapes > 0] = 1
    else:
        max_size = binary_image.size * 0.75
        min_size = 10
        cc_to_remove = np.argwhere(np.logical_or(stats[1:, 4] < min_size, stats[1:, 4] > max_size)) + 1
        shapes[np.isin(shapes, cc_to_remove)] = 0
        validated_shapes = np.zeros(shapes.shape, dtype=np.uint8)
        validated_shapes[shapes > 0] = 1
        shape_number, shapes, stats, centroids = cv2.connectedComponentsWithStats(validated_shapes, connectivity=8)
        if not several_blob_per_arena and true_shape_number is not None and shape_number > true_shape_number:
            # Sort shapes by size and compare the largest with the second largest
            # If the difference is too large, remove that largest shape.
            cc_to_remove = np.array([], dtype=np.uint8)
            to_remove = np.array([], dtype=np.uint8)
            stats = stats[1:, :]
            while stats.shape[0] > true_shape_number and to_remove is not None:
                # 1) rank by height
                sorted_height = np.argsort(stats[:, 2])
                # and only consider the number of shapes we want to detect
                standard_error = np.std(stats[sorted_height, 2][-true_shape_number:])
                differences = np.diff(stats[sorted_height, 2])
                # Look for very big changes from one height to the next
                if differences.any() and np.max(differences) > 2 * standard_error:
                    # Within these, remove shapes that are too large
                    to_remove = sorted_height[np.argmax(differences)]
                    cc_to_remove = np.append(cc_to_remove, to_remove + 1)
                    stats = np.delete(stats, to_remove, 0)

                else:
                    to_remove = None
            shapes[np.isin(shapes, cc_to_remove)] = 0
            validated_shapes = np.zeros(shapes.shape, dtype=np.uint8)
            validated_shapes[shapes > 0] = 1
            shape_number, shapes, stats, centroids = cv2.connectedComponentsWithStats(validated_shapes, connectivity=8)

        shape_number -= 1
    return validated_shapes, shape_number, stats, centroids
    
    

def rounded_inverted_distance_transform(original_shape: NDArray[np.uint8], max_distance: int=None, with_erosion: int=0) -> NDArray[np.uint32]:
    """
    Perform rounded inverted distance transform on a binary image.

    This function computes the inverse of the Euclidean distance transform,
    where each pixel value represents its distance to the nearest zero
    pixel. The operation can include erosion and will stop either at a given
    max distance or until no further expansion is needed.

    Parameters
    ----------
    original_shape : ndarray of uint8
        Input binary image to be processed.
    max_distance : int, optional
        Maximum distance for the expansion. If None, no limit is applied.
    with_erosion : int, optional
        Number of erosion iterations to apply before the transform. Default is 0.

    Returns
    -------
    out : ndarray of uint32
        Output image containing the rounded inverted distance transform.

    Examples
    --------
    >>> segmentation = np.zeros((4, 4), dtype=np.uint8)
    >>> segmentation[1:3, 1:3] = 1
    >>> gravity = rounded_inverted_distance_transform(segmentation, max_distance=2)
    >>> print(gravity)
    [[1 2 2 1]
     [2 0 0 2]
     [2 0 0 2]
     [1 2 2 1]]
    """
    if with_erosion > 0:
        original_shape = cv2.erode(original_shape, cross_33, iterations=with_erosion, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    expand = deepcopy(original_shape)
    if max_distance is not None:
        if max_distance > np.max(original_shape.shape):
            max_distance = np.max(original_shape.shape).astype(np.uint32)
        gravity_field = np.zeros(original_shape.shape , np.uint32)
        for gravi in np.arange(max_distance):
            expand = cv2.dilate(expand, cross_33, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            gravity_field[np.logical_xor(expand, original_shape)] += 1
    else:
        gravity_field = np.zeros(original_shape.shape , np.uint32)
        while np.any(np.equal(original_shape + expand, 0)):
            expand = cv2.dilate(expand, cross_33, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            gravity_field[np.logical_xor(expand, original_shape)] += 1
    return gravity_field


def inverted_distance_transform(original_shape: NDArray[np.uint8], max_distance: int=None, with_erosion: int=0) -> NDArray[np.uint32]:
    """
    Calculate the distance transform around ones in a binary image, with optional erosion.

    This function computes the Euclidean distance transform where zero values
    represent the background and ones represent the foreground. Optionally,
    it erodes the input image before computing the distance transform, and
    limits distances based on a maximum value.

    Parameters
    ----------
    original_shape : ndarray of uint8
        Input binary image where ones represent the foreground.
    max_distance : int, optional
        Maximum distance value to threshold. If None (default), no thresholding is applied.
    with_erosion : int, optional
        Number of iterations for erosion. If 0 (default), no erosion is applied.

    Returns
    -------
    out : ndarray of uint32
        Distance transform array where each element represents the distance
        to the nearest zero value in the input image.

    See also
    --------
    rounded_distance_transform : less precise (outputs int) and faster for small max_distance values.

    Examples
    --------
    >>> segmentation = np.zeros((4, 4), dtype=np.uint8)
    >>> segmentation[1:3, 1:3] = 1
    >>> gravity = inverted_distance_transform(segmentation, max_distance=2)
    >>> print(gravity)
    [[1.         1.41421356 1.41421356 1.        ]
     [1.41421356 0.         0.         1.41421356]
     [1.41421356 0.         0.         1.41421356]
     [1.         1.41421356 1.41421356 1.        ]]
    """
    if with_erosion:
        original_shape = cv2.erode(original_shape, cross_33, iterations=with_erosion, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    gravity_field = distance_transform_edt(1 - original_shape)
    if max_distance is not None:
        if max_distance > np.min(original_shape.shape) / 2:
            max_distance = (np.min(original_shape.shape) // 2).astype(np.uint32)
        gravity_field[gravity_field >= max_distance] = 0
    gravity_field[gravity_field > 0] = 1 + gravity_field.max() - gravity_field[gravity_field > 0]
    return gravity_field


@njit()
def get_line_points(start, end) -> NDArray[int]:
    """
    Get line points between two endpoints using Bresenham's line algorithm.

    This function calculates all the integer coordinate points that form a
    line between two endpoints using Bresenham's line algorithm. It is
    optimized for performance using Numba's just-in-time compilation.

    Parameters
    ----------
    start : tuple of int
        The starting point coordinates (y0, x0).
    end : tuple of int
        The ending point coordinates (y1, x1).

    Returns
    -------
    out : ndarray of int
        Array of points representing the line, with shape (N, 2), where N is
        the number of points on the line.

    Examples
    --------
    >>> start = (0, 0)
    >>> end = (1, 2)
    >>> points = get_line_points(start, end)
    >>> print(points)
    [[0 0]
    [0 1]
    [1 2]]
    """
    y0, x0 = start
    y1, x1 = end

    # Calculate differences
    dx = np.abs(x1 - x0)
    dy = np.abs(y1 - y0)

    # Determine step direction
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    # Initialize
    err = dx - dy
    points = []
    x, y = x0, y0

    while True:
        points.append([y, x])

        # Check if we've reached the end
        if x == x1 and y == y1:
            break

        # Calculate error for next step
        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x += sx

        if e2 < dx:
            err += dx
            y += sy

    return np.array(points)


def get_all_line_coordinates(start_point: NDArray[int], end_points: NDArray[int]) -> NDArray[int]:
    """
    Get all line coordinates between start point and end points.

    This function computes the coordinates of lines connecting a
    start point to multiple end points, converting input arrays to float
    if necessary before processing.

    Parameters
    ----------
    start_point : NDArray[float]
        Starting coordinate point for the lines. Can be of any numeric type,
        will be converted to float if needed.
    end_points : NDArray[float]
        Array of end coordinate points for the lines. Can be of any
        numeric type, will be converted to float if needed.

    Returns
    -------
    out : List[NDArray[int]]
        A list of numpy arrays containing the coordinates of each line
        as integer values.

    Examples
    --------
    >>> start_point = np.array([0, 0])
    >>> end_points = np.array([[1, 2], [3, 4]])
    >>> get_all_line_coordinates(start_point, end_points)
    [array([[0, 0],
       [0, 1],
       [1, 2]], dtype=uint64), array([[0, 0],
       [1, 1],
       [1, 2],
       [2, 3],
       [3, 4]], dtype=uint64)]
    """
    lines = []
    for end_point in end_points:
        line_coords = get_line_points(start_point, end_point)
        lines.append(np.array(line_coords, dtype=np.uint64))
    return lines


def draw_me_a_sun(main_shape: NDArray, ray_length_coef: int=4) -> Tuple[NDArray, NDArray]:
    """
    Draw a sun-shaped pattern on an image based on the main shape and ray length coefficient.

    This function takes an input binary image (main_shape) and draws sun rays
    from the perimeter of that shape. The length of the rays is controlled by a coefficient.
    The function ensures that rays do not extend beyond the image borders.

    Parameters
    ----------
    main_shape : ndarray of bool or int
        Binary input image where the main shape is defined.
    ray_length_coef : float, optional
        Coefficient to control the length of sun rays. Defaults to 2.

    Returns
    -------
    rays : ndarray
        Indices of the rays drawn.
    sun : ndarray
        Image with sun rays drawn on it.

    Examples
    --------
    >>> main_shape = np.zeros((10, 10), dtype=np.uint8)
    >>> main_shape[4:7, 3:6] = 1
    >>> rays, sun = draw_me_a_sun(main_shape)
    >>> print(sun)

    """
    nb, shapes, stats, center = cv2.connectedComponentsWithStats(main_shape)
    sun = np.zeros(main_shape.shape, np.uint32)
    rays = []
    r = 0
    for i in range(1, nb):
        shape_i = cv2.dilate((shapes == i).astype(np.uint8), kernel=cross_33)
        # shape_i = (shapes == i).astype(np.uint8)
        contours = get_contours(shape_i)
        first_ring_idx = np.nonzero(contours)
        centroid = np.round((center[i, 1], center[i, 0])).astype(np.int64)
        second_ring_y = centroid[0] + ((first_ring_idx[0] - centroid[0]) * ray_length_coef)
        second_ring_x = centroid[1] + ((first_ring_idx[1] - centroid[1]) * ray_length_coef)

        second_ring_y[second_ring_y < 0] = 0
        second_ring_x[second_ring_x < 0] = 0

        second_ring_y[second_ring_y > main_shape.shape[0] - 1] = main_shape.shape[0] - 1
        second_ring_x[second_ring_x > main_shape.shape[1] - 1] = main_shape.shape[1] - 1
        for j in range(len(second_ring_y)):
            r += 1
            fy, fx, sy, sx = first_ring_idx[0][j], first_ring_idx[1][j], second_ring_y[j], second_ring_x[j]
            line = get_line_points((fy, fx), (sy, sx))
            sun[line[:, 1], line[:, 0]] = r
            rays.append(r)
    return np.array(rays), sun


def find_median_shape(binary_3d_matrix: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Find the median shape from a binary 3D matrix.

    This function computes the median 2D slice of a binary (0/1) 3D matrix
    by finding which voxels appear in at least half of the slices.

    Parameters
    ----------
    binary_3d_matrix : ndarray of uint8
        Input 3D binary matrix where each slice is a 2D array.

    Returns
    -------
    ndarray of uint8
        Median shape as a 2D binary matrix where the same voxels
        that appear in at least half of the input slices are set to 1.

    Examples
    --------
    >>> binary_3d_matrix = np.random.randint(0, 2, (10, 5, 5), dtype=np.uint8)
    >>> median_shape = find_median_shape(binary_3d_matrix)
    >>> print(median_shape)
    """
    binary_2d_matrix = np.apply_along_axis(np.sum, 0, binary_3d_matrix)
    median_shape = np.zeros(binary_2d_matrix.shape, dtype=np.uint8)
    median_shape[np.greater_equal(binary_2d_matrix, binary_3d_matrix.shape[0] // 2)] = 1
    return median_shape


@njit()
def reduce_image_size_for_speed(image_of_2_shapes: NDArray[np.uint8]) -> Tuple[Tuple, Tuple]:
    """
    Reduces the size of an image containing two shapes for faster processing.

    The function iteratively divides the image into quadrants and keeps only
    those that contain both shapes until a minimal size is reached.

    Parameters
    ----------
    image_of_2_shapes : ndarray of uint8
        The input image containing two shapes.

    Returns
    -------
    out : tuple of tuples
        The indices of the first and second shape in the reduced image.

    Examples
    --------
    >>> image_of_2_shapes = np.zeros((10, 10), dtype=np.uint8)
    >>> image_of_2_shapes[1:3, 1:3] = 1
    >>> image_of_2_shapes[1:3, 4:6] = 2
    >>> shape1_idx, shape2_idx = reduce_image_size_for_speed(image_of_2_shapes)
    >>> print(shape1_idx)
    (array([1, 1, 2, 2]), array([1, 2, 1, 2]))
    """
    sub_image = image_of_2_shapes.copy()
    y_size, x_size = sub_image.shape
    images_list = [sub_image]
    good_images = [0]
    sub_image = images_list[good_images[0]]
    while (len(good_images) == 1 or len(good_images) == 2) and y_size > 3 and x_size > 3:
        y_size, x_size = sub_image.shape
        images_list = []
        images_list.append(sub_image[:((y_size // 2) + 1), :((x_size // 2) + 1)])
        images_list.append(sub_image[:((y_size // 2) + 1), (x_size // 2):])
        images_list.append(sub_image[(y_size // 2):, :((x_size // 2) + 1)])
        images_list.append(sub_image[(y_size // 2):, (x_size // 2):])
        good_images = []
        for idx, image in enumerate(images_list):
            if np.any(image == 2):
                if np.any(image == 1):
                    good_images.append(idx)
        if len(good_images) == 0:
            break
        elif len(good_images) == 2:
            if good_images == [0, 1]:
                sub_image = np.concatenate((images_list[good_images[0]], images_list[good_images[1]]), axis=1)
            elif good_images == [0, 2]:
                sub_image = np.concatenate((images_list[good_images[0]], images_list[good_images[1]]), axis=0)
            elif good_images == [1, 3]:
                sub_image = np.concatenate((images_list[good_images[0]], images_list[good_images[1]]), axis=0)
            elif good_images == [2, 3]:
                sub_image = np.concatenate((images_list[good_images[0]], images_list[good_images[1]]), axis=1)
            else:
                pass
        else:
            sub_image = images_list[good_images[0]]

    shape1_idx = np.nonzero(sub_image == 1)
    shape2_idx = np.nonzero(sub_image == 2)
    return shape1_idx, shape2_idx


def get_minimal_distance_between_2_shapes(image_of_2_shapes: NDArray[np.uint8], increase_speed: bool=True) -> float:
    """
    Get the minimal distance between two shapes in an image.

    This function calculates the minimal Euclidean distance between
    two different shapes represented by binary values 1 and 2 in a given image.
    It can optionally reduce the image size for faster processing.

    Parameters
    ----------
    image_of_2_shapes : ndarray of int8
        Binary image containing two shapes to measure distance between.
    increase_speed : bool, optional
        Flag to reduce image size for faster computation. Default is True.

    Returns
    -------
    min_distance : float64
        The minimal Euclidean distance between the two shapes.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.array([[1, 0], [0, 2]])
    >>> distance = get_minimal_distance_between_2_shapes(image)
    >>> print(distance)
    expected output
    """
    if increase_speed:
        shape1_idx, shape2_idx = reduce_image_size_for_speed(image_of_2_shapes)
    else:
        shape1_idx, shape2_idx = np.nonzero(image_of_2_shapes == 1), np.nonzero(image_of_2_shapes == 2)
    t = KDTree(np.transpose(shape1_idx))
    dists, nns = t.query(np.transpose(shape2_idx), 1)
    return np.min(dists)

def get_min_or_max_euclidean_pair(coords, min_or_max: str="max") -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the pair of points in a given set with the minimum or maximum Euclidean distance.

    Parameters
    ----------
    coords : Union[np.ndarray, Tuple]
        An Nx2 numpy array or a tuple of two arrays, each containing the x and y coordinates of points.
    min_or_max : str, optional
        Whether to find the 'min' or 'max' distance pair. Default is 'max'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the coordinates of the two points that form the minimum or maximum distance pair.

    Raises
    ------
    ValueError
        If `min_or_max` is not 'min' or 'max'.

    Notes
    -----
    - The function first computes all pairwise distances in condensed form using `pdist`.
    - Then, it finds the index of the minimum or maximum distance.
    - Finally, it maps this index to the actual point indices using a binary search method.

    Examples
    --------
    >>> coords = np.array([[0, 1], [2, 3], [4, 5]])
    >>> point1, point2 = get_min_or_max_euclidean_pair(coords, min_or_max="max")
    >>> print(point1)
    [0 1]
    >>> print(point2)
    [4 5]
    >>> coords = (np.array([0, 2, 4, 8, 1, 5]), np.array([0, 2, 4, 8, 0, 5]))
    >>> point1, point2 = get_min_or_max_euclidean_pair(coords, min_or_max="min")
    >>> print(point1)
    [0 0]
    >>> print(point2)
    [1 0]

    """
    if isinstance(coords, Tuple):
        coords = np.column_stack(coords)
    N = coords.shape[0]
    if N <= 1:
        return (coords[0], coords[0]) if N == 1 else None

    # Step 1: Compute all pairwise distances in condensed form
    distances = pdist(coords)

    # Step 2: Find the index of the maximum distance
    if min_or_max == "max":
        idx = np.argmax(distances)
    elif min_or_max == "min":
        idx = np.argmin(distances)
    else:
        raise ValueError

    # Step 3: Map this index to (i, j) using a binary search method

    def get_pair_index(k):
        low, high = 0, N
        while low < high:
            mid = (low + high) // 2
            total = mid * (2 * N - mid - 1) // 2
            if total <= k:
                low = mid + 1
            else:
                high = mid

        i = low - 1
        prev_sum = i * (2 * N - i - 1) // 2
        j_index_in_row = k - prev_sum
        return i, i + j_index_in_row + 1  # Ensure j > i

    i, j = get_pair_index(idx)
    return coords[i], coords[j]

def find_major_incline(vector: NDArray, natural_noise: float) -> Tuple[int, int]:
    """
    Find the major incline section in a vector.

    This function identifies the segment of a vector that exhibits
    the most significant change in values, considering a specified
    natural noise level. It returns the left and right indices that
    define this segment.

    Parameters
    ----------
    vector : ndarray of float64
        Input data vector where the incline needs to be detected.
    natural_noise : float
        The acceptable noise level for determining the incline.

    Returns
    -------
    Tuple[int, int]
        A tuple containing two integers: the left and right indices
        of the major incline section in the vector.

    Examples
    --------
    >>> vector = np.array([3, 5, 7, 9, 10])
    >>> natural_noise = 2.5
    >>> left, right = find_major_incline(vector, natural_noise)
    >>> (left, right)
    (0, 1)
    """
    left = 0
    right = 1
    ref_length = np.max((5, 2 * natural_noise))
    vector = moving_average(vector, 5)
    ref_extent = np.ptp(vector)
    extent = ref_extent
    # Find the left limit:
    while len(vector) > ref_length and extent > (ref_extent - (natural_noise / 4)):
        vector = vector[1:]
        extent = np.ptp(vector)
        left += 1
    # And the right one:
    extent = ref_extent
    while len(vector) > ref_length and extent > (ref_extent - natural_noise / 2):
        vector = vector[:-1]
        extent = np.ptp(vector)
        right += 1
    # And the left again, with stronger stringency:
    extent = ref_extent
    while len(vector) > ref_length and extent > (ref_extent - natural_noise):
        vector = vector[1:]
        extent = np.ptp(vector)
        left += 1
    # When there is no incline, put back left and right to 0
    if len(vector) <= ref_length:
        left = 0
        right = 1
    return left, right


def rank_from_top_to_bottom_from_left_to_right(binary_image: NDArray[np.uint8], y_boundaries: NDArray[int], get_ordered_image: bool=False) -> Tuple:
    """
    Rank components in a binary image from top to bottom and from left to right.

    This function processes a binary image to rank its components based on
    their centroids. It first sorts the components row by row and then orders them
    within each row from left to right. If the ordering fails, it attempts an alternative
    algorithm and returns the ordered statistics and centroids.

    Parameters
    ----------
    binary_image : ndarray of uint8
        The input binary image to process.
    y_boundaries : ndarray of int
        Boundary information for the y-coordinates.
    get_ordered_image : bool, optional
        If True, returns an ordered image in addition to the statistics and centroids.
        Default is False.

    Returns
    -------
    tuple
        If `get_ordered_image` is True, returns a tuple containing:
        - ordered_stats : ndarray of int
            Statistics for the ordered components.
        - ordered_centroids : ndarray of float64
            Centroids for the ordered components.
        - ordered_image : ndarray of uint8
            The binary image with ordered component labels.

        If `get_ordered_image` is False, returns a tuple containing:
        - ordered_stats : ndarray of int
            Statistics for the ordered components.
        - ordered_centroids : ndarray of float64
            Centroids for the ordered components.
    """
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_image.astype(np.uint8),
                                                                               connectivity=8)

    centroids = centroids[1:, :]
    final_order = np.zeros(centroids.shape[0], dtype=np.uint8)
    sorted_against_y = np.argsort(centroids[:, 1])
    # row_nb = (y_boundaries == 1).sum()
    row_nb = np.max(((y_boundaries == 1).sum(), (y_boundaries == - 1).sum()))
    if row_nb > 0:
        component_per_row = int(np.ceil((nb_components - 1) / row_nb))
        for row_i in range(row_nb):
            row_i_start = row_i * component_per_row
            if row_i == (row_nb - 1):
                sorted_against_x = np.argsort(centroids[sorted_against_y[row_i_start:], 0])
                final_order[row_i_start:] = sorted_against_y[row_i_start:][sorted_against_x]
            else:
                row_i_end = (row_i + 1) * component_per_row
                sorted_against_x = np.argsort(centroids[sorted_against_y[row_i_start:row_i_end], 0])
                final_order[row_i_start:row_i_end] = sorted_against_y[row_i_start:row_i_end][sorted_against_x]
    else:
        final_order = np.argsort(centroids[:, 0])
    ordered_centroids = centroids[final_order, :]
    ordered_stats = stats[1:, :]
    ordered_stats = ordered_stats[final_order, :]

    if get_ordered_image:
        old_to_new = np.zeros_like(final_order)
        old_to_new[final_order] = np.arange(len(final_order))
        mapping_array = np.zeros(binary_image.shape, dtype=np.uint8)
        mapping_array[output != 0] = old_to_new[output[output != 0] - 1] + 1
        ordered_image = mapping_array.copy()
        return ordered_stats, ordered_centroids, ordered_image
    else:
        return ordered_stats, ordered_centroids


def get_largest_connected_component(segmentation: NDArray[np.uint8]) -> Tuple[np.int64, NDArray[bool]]:
    """
    Find the largest connected component in a segmentation image.

    This function labels all connected components in a binary
    segmentation image, determines the size of each component,
    and returns information about the largest connected component.

    Parameters
    ----------
    segmentation : ndarray of uint8
        Binary segmentation image where different integer values represent
        different connected components.

    Returns
    -------
    Tuple[int, ndarray of bool]
        A tuple containing:
        - The size of the largest connected component.
        - A boolean mask representing the largest connected
          component in the input segmentation image.

    Examples
    --------
    >>> segmentation = np.zeros((10, 10), dtype=np.uint8)
    >>> segmentation[2:6, 2:5] = 1
    >>> segmentation[6:9, 6:9] = 1
    >>> size, mask = get_largest_connected_component(segmentation)
    >>> print(size)
    12
    """
    labels = label(segmentation)
    assert(labels.max() != 0) # assume at least 1 CC
    con_comp_sizes = np.bincount(labels.flat)[1:]
    largest_idx = np.argmax(con_comp_sizes)
    largest_connected_component = labels == largest_idx + 1
    return con_comp_sizes[largest_idx], largest_connected_component


def expand_until_neighbor_center_gets_nearer_than_own(shape_to_expand: NDArray[np.uint8], without_shape_i: NDArray[np.uint8],
                                                      shape_original_centroid: NDArray,
                                                      ref_centroids: NDArray, kernel: NDArray) -> NDArray[np.uint8]:
    """
    Expand a shape until its neighbor's centroid is closer than its own.

    This function takes in several numpy arrays representing shapes and their
    centroids, and expands the input shape until the distance to the nearest
    neighboring centroid is less than or equal to the distance between the shape's
    contour and its own centroid.

    Parameters
    ----------
    shape_to_expand : ndarray of uint8
        The binary shape to be expanded.
    without_shape_i : ndarray of uint8
        A binary array representing the area without the shape.
    shape_original_centroid : ndarray
        The centroid of the original shape.
    ref_centroids : ndarray
        Reference centroids to compare distances with.
    kernel : ndarray
        The kernel for dilation operation.

    Returns
    -------
    ndarray of uint8
        The expanded shape.
    """
    # shape_to_expand=test_shape
    # shape_i=0
    # shape_original_centroid=ordered_centroids[shape_i, :]
    # ref_centroids=np.delete(ordered_centroids, shape_i, axis=0)
    # kernel=self.small_kernels
    previous_shape_to_expand = shape_to_expand.copy()
    without_shape = deepcopy(without_shape_i)
    if ref_centroids.shape[0] > 1:
        # Calculate the distance between the focal shape centroid and its 10% nearest neighbor centroids
        centroid_distances = np.sqrt(np.square(ref_centroids[1:, 0] - shape_original_centroid[0]) + np.square(
            ref_centroids[1:, 1] - shape_original_centroid[1]))
        nearest_shapes = np.where(np.greater_equal(np.quantile(centroid_distances, 0.1), centroid_distances))[0]

        # Use the nearest neighbor distance as a maximal reference to get the minimal distance between the border of the shape and the neighboring centroids
        neighbor_mindist = np.min(centroid_distances)
        idx = np.nonzero(shape_to_expand)
        for shape_j in nearest_shapes:
            neighbor_mindist = np.minimum(neighbor_mindist, np.min(
                np.sqrt(np.square(ref_centroids[shape_j, 0] - idx[1]) + np.square(ref_centroids[shape_j, 1] - idx[0]))))
        neighbor_mindist *= 0.5
        # Get the maximal distance of the focal shape between its contour and its centroids
        itself_maxdist = np.max(
            np.sqrt(np.square(shape_original_centroid[0] - idx[1]) + np.square(shape_original_centroid[1] - idx[0])))
    else:
        itself_maxdist = np.max(shape_to_expand.shape)
        neighbor_mindist = itself_maxdist
        nearest_shapes = []
    # Put 1 at the border of the reference image in order to be able to stop the while loop once border reached
    without_shape[0, :] = 1
    without_shape[:, 0] = 1
    without_shape[without_shape.shape[0] - 1, :] = 1
    without_shape[:, without_shape.shape[1] - 1] = 1

    # Compare the distance between the contour of the shape and its centroid with this contour with the centroids of neighbors
    # Continue as the distance made by the shape (from its centroid) keeps being smaller than its distance with the nearest centroid.
    while np.logical_and(np.any(np.less_equal(itself_maxdist, neighbor_mindist)),
                         np.count_nonzero(shape_to_expand * without_shape) == 0):
        previous_shape_to_expand = shape_to_expand.copy()
        # Dilate the shape by the kernel size
        shape_to_expand = cv2.dilate(shape_to_expand, kernel, iterations=1,
                                     borderType=cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED)
        # Extract the new connected component
        shape_nb, shape_to_expand = cv2.connectedComponents(shape_to_expand, ltype=cv2.CV_16U)
        shape_to_expand = shape_to_expand.astype(np.uint8)
        # Use the nex shape coordinates to calculate the new distances of the shape with its centroid and with neighboring centroids
        idx = np.nonzero(shape_to_expand)
        for shape_j in nearest_shapes:
            neighbor_mindist = np.minimum(neighbor_mindist, np.min(
                np.sqrt(np.square(ref_centroids[shape_j, 0] - idx[1]) + np.square(ref_centroids[shape_j, 1] - idx[0]))))
        itself_maxdist = np.max(
            np.sqrt(np.square(shape_original_centroid[0] - idx[1]) + np.square(shape_original_centroid[1] - idx[0])))
    return previous_shape_to_expand


def image_borders(dimensions: tuple, shape: str="rectangular") -> NDArray[np.uint8]:
    """
    Create an image with borders, either rectangular or circular.

    Parameters
    ----------
    dimensions : tuple
        The dimensions of the image (height, width).
    shape : str, optional
        The shape of the borders. Options are "rectangular" or "circular".
        Defaults to "rectangular".

    Returns
    -------
    out : ndarray of uint8
        The image with borders. If the shape is "circular", an ellipse border;
        if "rectangular", a rectangular border.

    Examples
    --------
    >>> borders = image_borders((3, 3), "rectangular")
    >>> print(borders)
    [[0 0 0]
     [0 1 0]
     [0 0 0]]
    """
    if shape == "circular":
        borders = create_ellipse(dimensions[0], dimensions[0])
        img_contours = image_borders(dimensions)
        borders = borders * img_contours
    else:
        borders = np.ones(dimensions, dtype=np.uint8)
        borders[0, :] = 0
        borders[:, 0] = 0
        borders[- 1, :] = 0
        borders[:, - 1] = 0
    return borders


def get_radius_distance_against_time(binary_video: NDArray[np.uint8], field) -> Tuple[NDArray[np.float32], int, int]:
    """
    Calculate the radius distance against time from a binary video and field.

    This function computes the change in radius distances over time
    by analyzing a binary video and mapping it to corresponding field values.

    Parameters
    ----------
    binary_video : ndarray of uint8
        Binary video data.
    field : ndarray
        Field values to analyze the radius distances against.

    Returns
    -------
    distance_against_time : ndarray of float32
        Radius distances over time.
    time_start : int
        Starting time index where the radius distance measurement begins.
    time_end : int
        Ending time index where the radius distance measurement ends.

    Examples
    --------
    >>> binary_video = np.ones((10, 5, 5), dtype=np.uint8)

    >>> distance_against_time, time_start, time_end = get_radius_distance_against_time(binary_video, field)
    """
    pixel_start = np.max(field[field > 0])
    pixel_end = np.min(field[field > 0])
    time_span = np.arange(binary_video.shape[0])
    time_start = 0
    time_end = time_span[-1]
    start_not_found: bool = True
    for t in time_span:
        if start_not_found:
            if np.any((field == pixel_start) * binary_video[t, :, :]):
                start_not_found = False
                time_start = t
        if np.any((field == pixel_end) * binary_video[t, :, :]):
            time_end = t
            break
    distance_against_time = np.linspace(pixel_start, pixel_end, (time_end - time_start + 1))
    distance_against_time = np.round(distance_against_time).astype(np.float32)
    return distance_against_time, time_start, time_end


def close_holes(binary_img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Close holes in a binary image using connected components analysis.

    This function identifies and closes small holes within the foreground objects of a binary image. It uses connected component analysis to find and fill holes that are smaller than the main object.

    Parameters
    ----------
    binary_img : ndarray of uint8
        Binary input image where holes need to be closed.

    Returns
    -------
    out : ndarray of uint8
        Binary image with closed holes.

    Examples
    --------
    >>> binary_img = np.zeros((10, 10), dtype=np.uint8)
    >>> binary_img[2:8, 2:8] = 1
    >>> binary_img[4:6, 4:6] = 0  # Creating a hole
    >>> result = close_holes(binary_img)
    >>> print(result)
    [[0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0]
     [0 0 1 1 1 1 1 1 0 0]
     [0 0 1 1 1 1 1 1 0 0]
     [0 0 1 1 1 1 1 1 0 0]
     [0 0 1 1 1 1 1 1 0 0]
     [0 0 1 1 1 1 1 1 0 0]
     [0 0 1 1 1 1 1 1 0 0]
     [0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0]]
    """
    #### Third version ####
    nb, new_order = cv2.connectedComponents(1 - binary_img)
    if nb > 2:
        binary_img[new_order > 1] = 1
    return binary_img


def dynamically_expand_to_fill_holes(binary_video: NDArray[np.uint8], holes: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], int, NDArray[np.float32]]:
    """
    Fill the holes in a binary video by progressively expanding the shape made of ones.

    Parameters
    ----------
    binary_video : ndarray of uint8
        The binary video where holes need to be filled.
    holes : ndarray of uint8
        Array representing the holes in the binary video.

    Returns
    -------
    out : tuple of ndarray of uint8, int, and ndarray of float32
        The modified binary video with filled holes,
        the end time when all holes are filled, and
        an array of distances against time used to fill the holes.

    Examples
    --------
    >>> binary_video = np.zeros((10, 640, 480), dtype=np.uint8)
    >>> binary_video[:, 300:400, 220:240] = 1
    >>> holes = np.zeros((640, 480), dtype=np.uint8)
    >>> holes[340:360, 228:232] = 1
    >>> filled_video, end_time, distances = dynamically_expand_to_fill_holes(binary_video, holes)
    >>> print(filled_video.shape)  # Should print (10, 640, 480)
    (10, 640, 480)
    """
    #first move should be the time at wich the first pixel hole could have been covered
    #it should ask how much time the shape made to cross a distance long enough to overlap all holes
    holes_contours = cv2.dilate(holes, cross_33, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    field = rounded_inverted_distance_transform(binary_video[0, :, :], (binary_video.shape[0] - 1))
    field2 = inverted_distance_transform(binary_video[0, :, :], (binary_video.shape[0] - 1))
    holes_contours = holes_contours * field * binary_video[- 1, :, :]
    holes[np.nonzero(holes)] = field[np.nonzero(holes)]
    if np.any(holes_contours):
        # Find the relationship between distance and time
        distance_against_time, holes_time_start, holes_time_end = get_radius_distance_against_time(binary_video, holes_contours)
        # Use that vector to progressively fill holes at the same speed as shape grows
        for t in np.arange(len(distance_against_time)):
            new_order, stats, centers = cc((holes >= distance_against_time[t]).astype(np.uint8))
            for comp_i in np.arange(1, stats.shape[0]):
                past_image = deepcopy(binary_video[holes_time_start + t, :, :])
                with_new_comp = new_order == comp_i
                past_image[with_new_comp] = 1
                nb_comp, image_garbage = cv2.connectedComponents(past_image)
                if nb_comp == 2:
                    binary_video[holes_time_start + t, :, :][with_new_comp] = 1
        # Make sure that holes remain filled from holes_time_end to the end of the video
        for t in np.arange((holes_time_end + 1), binary_video.shape[0]):
            past_image = binary_video[t, :, :]
            past_image[holes >= distance_against_time[-1]] = 1
            binary_video[t, :, :] = past_image
    else:
        holes_time_end = None
        distance_against_time = np.array([1, 2], dtype=np.float32)

    return binary_video, holes_time_end, distance_against_time


@njit()
def create_ellipse(vsize: int, hsize: int, min_size: int=0) -> NDArray[np.uint8]:
    """
    Create a 2D array representing an ellipse with given vertical and horizontal sizes.

    This function generates a NumPy boolean array where each element is `True` if the point lies within or on
    the boundary of an ellipse defined by its vertical and horizontal radii. The ellipse is centered at the center
    of the array, which corresponds to the midpoint of the given dimensions.

    Parameters
    ----------
    vsize : int
        Vertical size (number of rows) in the output 2D array.
    hsize : int
        Horizontal size (number of columns) in the output 2D array.

    Returns
    -------
    NDArray[bool]
        A boolean NumPy array of shape `(vsize, hsize)` where `True` indicates that a pixel lies within or on
        the boundary of an ellipse centered at the image's center with radii determined by half of the dimensions.
    """
    # Use default values if input sizes are zero
    vsize = min_size if vsize == 0 else vsize
    hsize = min_size if hsize == 0 else hsize

    # Compute radii (half of each size)
    vr = hsize // 2
    hr = vsize // 2

    result = np.empty((vsize, hsize), dtype=np.bool_)
    if vr > 0 and hr > 0:
        for i in range(vsize):
            for j in range(hsize):
                x = i
                y = j
                lhs = ((x - hr) ** 2 / (hr ** 2)) + ((y - vr) ** 2 / (vr ** 2))
                result[i, j] = lhs <= 1
    else:
        result[hr, vr] = True
    return result

rhombus_55 = create_ellipse(5, 5).astype(np.uint8)

def get_contours(binary_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Find and return the contours of a binary image.

    This function erodes the input binary image using a 3x3 cross-shaped
    structuring element and then subtracts the eroded image from the original to obtain the contours.

    Parameters
    ----------
    binary_image : ndarray of uint8
        Input binary image from which to extract contours.

    Returns
    -------
    out : ndarray of uint8
        Image containing only the contours extracted from `binary_image`.

    Examples
    --------
    >>> binary_image = np.zeros((10, 10), dtype=np.uint8)
    >>> binary_image[2:8, 2:8] = 1
    >>> result = get_contours(binary_image)
    >>> print(result)
    [[0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0]
     [0 0 1 1 1 1 1 1 0 0]
     [0 0 1 0 0 0 0 1 0 0]
     [0 0 1 0 0 0 0 1 0 0]
     [0 0 1 0 0 0 0 1 0 0]
     [0 0 1 0 0 0 0 1 0 0]
     [0 0 1 1 1 1 1 1 0 0]
     [0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0]]
    """
    if not isinstance(binary_image.dtype, np.uint8):
        binary_image = binary_image.astype(np.uint8)
    if np.all(binary_image):
        contours = 1 - image_borders(binary_image.shape)
    elif np.any(binary_image):
        eroded_binary = cv2.erode(binary_image, cross_33, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        contours = binary_image - eroded_binary
    else:
        contours = binary_image
    return contours


def get_quick_bounding_boxes(binary_image: NDArray[np.uint8], ordered_image: NDArray, ordered_stats: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Compute bounding boxes for shapes in a binary image.

    Parameters
    ----------
    binary_image : NDArray[np.uint8]
        A 2D array representing the binary image.
    ordered_image : NDArray
        An array containing the ordered image data.
    ordered_stats : NDArray
        A 2D array with statistics about the shapes in the image.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray, NDArray]
        A tuple containing four arrays:
        - top: Array of y-coordinates for the top edge of bounding boxes.
        - bot: Array of y-coordinates for the bottom edge of bounding boxes.
        - left: Array of x-coordinates for the left edge of bounding boxes.
        - right: Array of x-coordinates for the right edge of bounding boxes.

    Examples
    --------
    >>> binary_image = np.array([[0, 1], [0, 0], [1, 0]], dtype=np.uint8)
    >>> ordered_image = np.array([[0, 1], [0, 0], [2, 0]], dtype=np.uint8)
    >>> ordered_stats = np.array([[1, 0, 1, 1, 1], [0, 2, 1, 1, 1]], dtype=np.int32)
    >>> top, bot, left, right = get_quick_bounding_boxes(binary_image, ordered_image, ordered_stats)
    >>> print(top)
    [-1  1]
    >>> print(bot)
    [2 4]
    >>> print(left)
    [0 -1]
    >>> print(right)
    [3 2]
    """
    shapes = get_contours(binary_image)
    x_min = ordered_stats[:, 0]
    y_min = ordered_stats[:, 1]
    x_max = ordered_stats[:, 0] + ordered_stats[:, 2]
    y_max = ordered_stats[:, 1] + ordered_stats[:, 3]
    x_min_dist = shapes.shape[1]
    y_min_dist = shapes.shape[0]

    shapes *= ordered_image
    shape_nb = (len(np.unique(shapes)) - 1)
    i = 0
    a_indices, b_indices = np.triu_indices(shape_nb, 1)
    a_indices, b_indices = a_indices + 1, b_indices + 1
    all_distances = np.zeros((len(a_indices), 3), dtype=float)
    # For every pair of components, find the minimal distance
    for (a, b) in zip(a_indices, b_indices):
        x_dist = np.absolute(x_max[a - 1] - x_min[b - 1])
        y_dist = np.absolute(y_max[a - 1] - y_min[b - 1])
        if x_dist < 2 * x_min_dist and y_dist < 2 * y_min_dist:
            sub_shapes = np.logical_or(shapes == a, shapes == b) * shapes
            sub_shapes = sub_shapes[np.min((y_min[a - 1], y_min[b - 1])):np.max((y_max[a - 1], y_max[b - 1])),
                         np.min((x_min[a - 1], x_min[b - 1])):np.max((x_max[a - 1], x_max[b - 1]))]
            sub_shapes[sub_shapes == a] = 1
            sub_shapes[sub_shapes == b] = 2
            if np.any(sub_shapes == 1) and np.any(sub_shapes == 2):
                all_distances[i, :] = a, b, get_minimal_distance_between_2_shapes(sub_shapes, False)

                if x_dist > y_dist:
                    x_min_dist = np.min((x_min_dist, x_dist))
                else:
                    y_min_dist = np.min((y_min_dist, y_dist))
                i += 1
    shape_number = ordered_stats.shape[0]
    top = np.zeros(shape_number, dtype=np.int64)
    bot = np.repeat(binary_image.shape[0], shape_number)
    left = np.zeros(shape_number, dtype=np.int64)
    right = np.repeat(binary_image.shape[1], shape_number)
    for shape_i in np.arange(1, shape_nb + 1):
        # Get where the shape i appear in pairwise comparisons
        idx = np.nonzero(np.logical_or(all_distances[:, 0] == shape_i, all_distances[:, 1] == shape_i))
        # Compute the minimal distance related to shape i and divide by 2
        if len(all_distances[idx, 2]) > 0:
            dist = all_distances[idx, 2].min() // 2
        else:
            dist = 1
            # Save the coordinates of the arena around shape i
        top[shape_i - 1] = y_min[shape_i - 1] - dist.astype(np.int64)
        bot[shape_i - 1] = y_max[shape_i - 1] + dist.astype(np.int64)
        left[shape_i - 1] = x_min[shape_i - 1] - dist.astype(np.int64)
        right[shape_i - 1] = x_max[shape_i - 1] + dist.astype(np.int64)
    return top, bot, left, right



def get_bb_with_moving_centers(motion_list: list, all_specimens_have_same_direction: bool,
                                original_shape_hsize: int, binary_image: NDArray,
                                y_boundaries: NDArray):
    """
    Get the bounding boxes with moving centers.

    Parameters
    ----------
    motion_list : list
        List of binary images representing the motion frames.
    all_specimens_have_same_direction : bool
        Boolean indicating if all specimens move in the same direction.
    original_shape_hsize : int or None
        Original height size of the shape. If `None`, a default kernel size is used.
    binary_image : NDArray
        Binary image of the initial frame.
    y_boundaries : NDArray
        Array defining the y-boundaries for ranking shapes.

    Returns
    -------
    tuple
        A tuple containing:
        - top : NDArray
            Array of top coordinates for each bounding box.
        - bot : NDArray
            Array of bottom coordinates for each bounding box.
        - left : NDArray
            Array of left coordinates for each bounding box.
        - right : NDArray
            Array of right coordinates for each bounding box.
        - ordered_image_i : NDArray
            Updated binary image with the final ranked shapes.

    Notes
    -----
    This function processes each frame to expand and confirm shapes, updating centroids if necessary.
    It uses morphological operations like dilation to detect shape changes over frames.

    Examples
    --------
    >>> top, bot, left, right, ordered_image = _get_bb_with_moving_centers(motion_frames, True, None, binary_img, y_bounds)
    >>> print("Top coordinates:", top)
    >>> print("Bottom coordinates:", bot)
    """
    print("Read and segment each sample image and rank shapes from top to bot and from left to right")
    k_size = 3
    if original_shape_hsize is not None:
        k_size = int((np.ceil(original_shape_hsize / 5) * 2) + 1)
    big_kernel = create_ellipse(k_size, k_size, min_size=3).astype(np.uint8)

    ordered_stats, ordered_centroids, ordered_image = rank_from_top_to_bottom_from_left_to_right(
        binary_image, y_boundaries, get_ordered_image=True)
    blob_number = ordered_stats.shape[0]

    ordered_image_i = deepcopy(ordered_image)
    logging.info("For each frame, expand each previously confirmed shape to add area to its maximal bounding box")
    for step_i in np.arange(1, len(motion_list)):
        previously_ordered_centroids = deepcopy(ordered_centroids)
        new_image_i = motion_list[step_i].copy()
        detected_shape_number = blob_number + 1
        c = 0
        while c < 5 and detected_shape_number == blob_number + 1:
            c += 1
            image_i = new_image_i
            new_image_i = cv2.dilate(image_i, cross_33, iterations=1)
            detected_shape_number, _ = cv2.connectedComponents(new_image_i, connectivity=8)
        if c == 0:
            break
        else:
            for shape_i in range(blob_number):
                shape_to_expand = np.zeros(image_i.shape, dtype=np.uint8)
                shape_to_expand[ordered_image_i == (shape_i + 1)] = 1
                without_shape_i = ordered_image_i.copy()
                without_shape_i[ordered_image_i == (shape_i + 1)] = 0
                if k_size != 3:
                    test_shape = expand_until_neighbor_center_gets_nearer_than_own(shape_to_expand, without_shape_i,
                                                                                   ordered_centroids[shape_i, :],
                                                                                   np.delete(ordered_centroids, shape_i,
                                                                                             axis=0), big_kernel)
                else:
                    test_shape = shape_to_expand
                test_shape = expand_until_neighbor_center_gets_nearer_than_own(test_shape, without_shape_i,
                                                                               ordered_centroids[shape_i, :],
                                                                               np.delete(ordered_centroids, shape_i,
                                                                                         axis=0), cross_33)
                confirmed_shape = test_shape * image_i
                ordered_image_i[confirmed_shape > 0] = shape_i + 1


            mask_to_display = np.zeros(image_i.shape, dtype=np.uint8)
            mask_to_display[ordered_image_i > 0] = 1

            # If the blob moves enough to drastically change its gravity center,
            # update the ordered centroids at each frame.
            detected_shape_number, mask_to_display = cv2.connectedComponents(mask_to_display,
                                                                             connectivity=8)

            mask_to_display = mask_to_display.astype(np.uint8)
            while np.logical_and(detected_shape_number - 1 != blob_number,
                                 np.sum(mask_to_display > 0) < mask_to_display.size):
                mask_to_display = cv2.dilate(mask_to_display, cross_33, iterations=1)
                detected_shape_number, mask_to_display = cv2.connectedComponents(mask_to_display,
                                                                                 connectivity=8)
                mask_to_display[np.nonzero(mask_to_display)] = 1
                mask_to_display = mask_to_display.astype(np.uint8)
            ordered_stats, ordered_centroids = rank_from_top_to_bottom_from_left_to_right(mask_to_display, y_boundaries)

            if all_specimens_have_same_direction:
                # Adjust each centroid position according to the maximal centroid displacement.
                x_diffs = ordered_centroids[:, 0] - previously_ordered_centroids[:, 0]
                if np.mean(x_diffs) > 0: # They moved left, we add to x
                    add_to_x = np.max(x_diffs) - x_diffs
                else: #They moved right, we remove from x
                    add_to_x = np.min(x_diffs) - x_diffs
                ordered_centroids[:, 0] = ordered_centroids[:, 0] + add_to_x

                y_diffs = ordered_centroids[:, 1] - previously_ordered_centroids[:, 1]
                if np.mean(y_diffs) > 0:  # They moved down, we add to y
                    add_to_y = np.max(y_diffs) - y_diffs
                else:  # They moved up, we remove from y
                    add_to_y = np.min(y_diffs) - y_diffs
                ordered_centroids[:, 1] = ordered_centroids[:, 1] + add_to_y

            ordered_image_i = mask_to_display

    # Save each bounding box
    top = np.zeros(blob_number, dtype=np.int64)
    bot = np.repeat(binary_image.shape[0], blob_number)
    left = np.zeros(blob_number, dtype=np.int64)
    right = np.repeat(binary_image.shape[1], blob_number)
    for shape_i in range(blob_number):
        shape_i_indices = np.where(ordered_image_i == shape_i + 1)
        left[shape_i] = np.min(shape_i_indices[1])
        right[shape_i] = np.max(shape_i_indices[1])
        top[shape_i] = np.min(shape_i_indices[0])
        bot[shape_i] = np.max(shape_i_indices[0])
    return top, bot, left, right, ordered_image_i


def prepare_box_counting(binary_image: NDArray[np.uint8], min_im_side: int=128, min_mesh_side: int=8, zoom_step: int=0, contours: bool=True)-> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Prepare box counting parameters for image analysis.

    Prepares parameters for box counting method based on binary
    image input. Adjusts image size, computes side lengths, and applies
    contour extraction if specified.

    Parameters
    ----------
    binary_image : ndarray of uint8
        Binary image for analysis.
    min_im_side : int, optional
        Minimum side length threshold. Default is 128.
    min_mesh_side : int, optional
        Minimum mesh side length. Default is 8.
    zoom_step : int, optional
        Zoom step for side lengths computation. Default is 0.
    contours : bool, optional
        Whether to apply contour extraction. Default is True.

    Returns
    -------
    out : tuple of ndarray of uint8, ndarray (or None)
        Cropped binary image and computed side lengths.

    Examples
    --------
    >>> binary_image = np.zeros((10, 10), dtype=np.uint8)
    >>> binary_image[2:4, 2:6] = 1
    >>> binary_image[7:9, 4:7] = 1
    >>> binary_image[4:7, 5] = 1
    >>> cropped_img, side_lengths = prepare_box_counting(binary_image, min_im_side=2, min_mesh_side=2)
    >>> print(cropped_img), print(side_lengths)
    [[0 0 0 0 0 0 0]
     [0 1 1 1 1 0 0]
     [0 1 1 1 1 0 0]
     [0 0 0 0 1 0 0]
     [0 0 0 0 1 0 0]
     [0 0 0 0 1 0 0]
     [0 0 0 1 0 1 0]
     [0 0 0 1 1 1 0]
     [0 0 0 0 0 0 0]]
    [4 2]
    """
    side_lengths = None
    zoomed_binary = binary_image
    binary_idx = np.nonzero(binary_image)
    if binary_idx[0].size:
        min_y = np.min(binary_idx[0])
        min_y = np.max((min_y - 1, 0))

        min_x = np.min(binary_idx[1])
        min_x = np.max((min_x - 1, 0))

        max_y = np.max(binary_idx[0])
        max_y = np.min((max_y + 1, binary_image.shape[0] - 1))

        max_x = np.max(binary_idx[1])
        max_x = np.min((max_x + 1, binary_image.shape[1] - 1))

        zoomed_binary = deepcopy(binary_image[min_y:(max_y + 1), min_x: (max_x + 1)])
        min_side = np.min(zoomed_binary.shape)
        if min_side >= min_im_side:
            if contours:
                zoomed_binary = get_contours(zoomed_binary)
            if zoom_step == 0:
                max_power = int(np.floor(np.log2(min_side)))  # Largest integer power of 2
                side_lengths = 2 ** np.arange(max_power, int(np.log2(min_mesh_side // 2)), -1)
            else:
                side_lengths = np.arange(min_mesh_side, min_side, zoom_step)
    return zoomed_binary, side_lengths


def box_counting_dimension(zoomed_binary: NDArray[np.uint8], side_lengths: NDArray, display: bool=False) -> Tuple[float, float, float]:
    """
    Box counting dimension calculation.

    This function calculates the box-counting dimension of a binary image by analyzing the number
    of boxes (of varying sizes) that contain at least one pixel of the image. The function also
    provides the R-squared value from linear regression and the number of boxes used.

    Parameters
    ----------
    zoomed_binary : NDArray[np.uint8]
        Binary image (0 or 255 values) for which the box-counting dimension is calculated.
    side_lengths : NDArray
        Array of side lengths for the boxes used in the box-counting calculation.
    display : bool, optional
        If True, displays a scatter plot of the log-transformed box counts and diameters,
        along with the linear regression fit. Default is False.

    Returns
    -------
    out : Tuple[float, float, float]
        A tuple containing the calculated box-counting dimension (`d`), R-squared value (`r_value`),
        and the number of boxes used (`box_nb`).

    Examples
    --------
    >>> binary_image = np.zeros((10, 10), dtype=np.uint8)
    >>> binary_image[2:4, 2:6] = 1
    >>> binary_image[7:9, 4:7] = 1
    >>> binary_image[4:7, 5] = 1
    >>> zoomed_binary, side_lengths = prepare_box_counting(binary_image, min_im_side=2, min_mesh_side=2)
    >>> dimension, r_value, box_nb = box_counting_dimension(zoomed_binary, side_lengths)
    >>> print(dimension, r_value, box_nb)
    (np.float64(1.1699250014423126), np.float64(0.9999999999999998), 2)
    """
    dimension:float = 0.
    r_value:float = 0.
    box_nb:float = 0.
    if side_lengths is not None:
        box_counts = np.zeros(len(side_lengths), dtype=np.uint64)
        # Loop through side_lengths and compute block counts
        for idx, side_length in enumerate(side_lengths):
            S = np.add.reduceat(
                np.add.reduceat(zoomed_binary, np.arange(0, zoomed_binary.shape[0], side_length), axis=0),
                np.arange(0, zoomed_binary.shape[1], side_length),
                axis=1
            )
            box_counts[idx] = len(np.where(S > 0)[0])

        valid_indices = box_counts > 0
        if valid_indices.sum() >= 2:
            log_box_counts = np.log(box_counts)
            log_reciprocal_lengths = np.log(1 / side_lengths)
            slope, intercept, r_value, p_value, stderr = linregress(log_reciprocal_lengths, log_box_counts)
            # coefficients = np.polyfit(log_reciprocal_lengths, log_box_counts, 1)
            dimension = slope
            box_nb = len(side_lengths)
            if display:
                plt.scatter(log_reciprocal_lengths, log_box_counts, label="Box counting")
                plt.plot([0, log_reciprocal_lengths.min()], [intercept, intercept + slope * log_reciprocal_lengths.min()], label="Linear regression")
                plt.plot([], [], ' ', label=f"D = {slope:.2f}")
                plt.plot([], [], ' ', label=f"R2 = {r_value:.6f}")
                plt.plot([], [], ' ', label=f"p-value = {p_value:.2e}")
                plt.legend(loc='best')
                plt.xlabel(f"log(1/Diameter) | Diameter ⊆ [{side_lengths[0]}:{side_lengths[-1]}] (n={box_nb})")
                plt.ylabel(f"log(Box number) | Box number ⊆ [{box_counts[0]}:{box_counts[-1]}]")
                plt.show()
                # plt.close()

    return dimension, r_value, box_nb


def keep_shape_connected_with_ref(all_shapes: NDArray[np.uint8], reference_shape: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Keep shape connected with reference.

    This function analyzes the connected components of a binary image represented by `all_shapes`
    and returns the first component that intersects with the `reference_shape`.
    If no such component is found, it returns None.

    Parameters
    ----------
    all_shapes : ndarray of uint8
        Binary image containing all shapes to analyze.
    reference_shape : ndarray of uint8
        Binary reference shape used for intersection check.

    Returns
    -------
    out : ndarray of uint8 or None
        The first connected component that intersects with the reference shape,
        or None if no such component is found.

    Examples
    -------
    >>> all_shapes = np.zeros((5, 5), dtype=np.uint8)
    >>> reference_shape = np.zeros((5, 5), dtype=np.uint8)
    >>> reference_shape[3, 3] = 1
    >>> all_shapes[0:2, 0:2] = 1
    >>> all_shapes[3:4, 3:4] = 1
    >>> res = keep_shape_connected_with_ref(all_shapes, reference_shape)
    >>> print(res)
    [[0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 1 0]
     [0 0 0 0 0]]
    """
    number, order = cv2.connectedComponents(all_shapes, ltype=cv2.CV_16U)
    expanded_shape = None
    if number > 1:
        for i in np.arange(1, number):
            expanded_shape_test = np.zeros(order.shape, np.uint8)
            expanded_shape_test[order == i] = 1
            if np.any(expanded_shape_test * reference_shape):
                break
        if np.any(expanded_shape_test * reference_shape):
            expanded_shape = expanded_shape_test
        else:
            expanded_shape = reference_shape
    return expanded_shape


@njit()
def keep_largest_shape(indexed_shapes: NDArray[np.int32]) -> NDArray[np.uint8]:
    """
    Keep the largest shape from an array of indexed shapes.

    This function identifies the most frequent non-zero shape in the input
    array and returns a binary mask where elements matching this shape are set to 1,
    and others are set to 0. The function uses NumPy's bincount to count occurrences
    of each shape and assumes that the first element (index 0) is not part of any
    shape classification.

    Parameters
    ----------
    indexed_shapes : ndarray of int32
        Input array containing indexed shapes.

    Returns
    -------
    out : ndarray of uint8
        Binary mask where the largest shape is marked as 1.

    Examples
    --------
    >>> indexed_shapes = np.array([0, 2, 2, 3, 1], dtype=np.int32)
    >>> keep_largest_shape(indexed_shapes)
    array([0, 1, 1, 0, 0], dtype=uint8)
    """
    label_counts = np.bincount(indexed_shapes.flatten())
    largest_label = 1 + np.argmax(label_counts[1:])
    return (indexed_shapes == largest_label).astype(np.uint8)


def keep_one_connected_component(binary_image: NDArray[np.uint8])-> NDArray[np.uint8]:
    """
    Keep only one connected component in a binary image.

    This function filters out all but the largest connected component in
    a binary image, effectively isolating it from other noise or objects.
    The function ensures the input is in uint8 format before processing.

    Parameters
    ----------
    binary_image : ndarray of uint8
        Binary image containing one or more connected components.

    Returns
    -------
    ndarray of uint8
        Image with only the largest connected component retained.

    Examples
    -------
    >>> all_shapes = np.zeros((5, 5), dtype=np.uint8)
    >>> all_shapes[0:2, 0:2] = 1
    >>> all_shapes[3:4, 3:4] = 1
    >>> res = keep_one_connected_component(all_shapes)
    >>> print(res)
    [[1 1 0 0 0]
     [1 1 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]]
    """
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8)
    num_labels, sh = cv2.connectedComponents(binary_image)
    if num_labels <= 1:
        return binary_image.astype(np.uint8)
    else:
        return keep_largest_shape(sh)

def create_mask(dims: Tuple, minmax: Tuple, shape: str):
    """

    Create a boolean mask based on given dimensions and min/max coordinates.

    Parameters
    ----------
    dims : Tuple[int, int]
        The dimensions of the mask (height and width).
    minmax : Tuple[int, int, int, int]
        The minimum and maximum coordinates for the mask (x_min, x_max,
        y_min, y_max).
    shape : str
        The shape of the mask. Should be either 'circle' or any other value for a rectangular mask.

    Returns
    -------
    np.ndarray[bool]
        A boolean NumPy array with the same dimensions as `dims`, initialized to False,
        where the specified region (or circle) is set to True.

    Raises
    ------
    ValueError
        If the shape is 'circle' and the ellipse creation fails.

    Notes
    -----
    If `shape` is not 'circle', a rectangular mask will be created. The ellipse
    creation method used may have specific performance considerations.

    Examples
    --------
    >>> mask = create_mask((5, 6), (0, 5, 1, 5), 'circle')
    >>> print(mask)
    [[False False False  True False False]
     [False False  True  True  True False]
     [False  True  True  True  True False]
     [False False  True  True  True False]
     [False False False  True False False]]
     """
    mask = np.zeros(dims[:2], dtype=bool)
    if shape == 'circle':
        ellipse = create_ellipse(minmax[1] - minmax[0], minmax[3] - minmax[2])
        mask[minmax[0]:minmax[1], minmax[2]:minmax[3], ...] = ellipse
    else:
        mask[minmax[0]:minmax[1], minmax[2]:minmax[3]] = 1
    return mask

def draw_img_with_mask(img:NDArray, dims: Tuple, minmax: Tuple, shape: str, drawing: Tuple, only_contours: bool=False,
                       dilate_mask: int=0) -> NDArray:
    """

    Draw an image with a mask and optional contours.

    Draws a subregion of the input image using a specified shape (circle or rectangle),
    which can be dilated. The mask can be limited to contours only, and an optional
    drawing (overlay) can be applied within the masked region.

    Parameters
    ----------
    img : NDArray
        The input image to draw on.
    dims : Tuple[int, int]
        Dimensions of the subregion (width, height).
    minmax : Tuple[int, int, int, int]
        Coordinates of the subregion (x_start, x_end, y_start, y_end).
    shape : str
        Shape of the mask to draw ('circle' or 'rectangle').
    drawing : Tuple[NDArray, NDArray, NDArray]
        Optional drawing (overlay) to apply within the masked region.
    only_contours : bool, optional
        If True, draw only the contours of the shape. Default is False.
    dilate_mask : int, optional
        Number of iterations for dilating the mask. Default is 0.

    Returns
    -------
    NDArray
        The modified image with the applied mask and drawing.

    Notes
    -----
    This function assumes that the input image is in BGR format (OpenCV style).

    Examples
    --------
    >>> dim = (100, 100, 3)
    >>> img = np.zeros(dim)
    >>> result = draw_img_with_mask(img, dim, (50, 75, 50, 75), 'circle', (0, 255, 0))
    >>> print((result == 255).sum())
    441
    """
    if shape == 'circle':
        mask = create_ellipse(minmax[1] - minmax[0], minmax[3] - minmax[2]).astype(np.uint8)
        if only_contours:
            mask = get_contours(mask)
    else:
        if only_contours:
            mask = 1 - image_borders((minmax[1] - minmax[0], minmax[3] - minmax[2]))
        else:
            mask = np.ones((minmax[1] - minmax[0], minmax[3] - minmax[2]), np.uint8)
    if dilate_mask:
        mask = cv2.dilate(mask, kernel=cross_33, iterations=dilate_mask)
    anti_mask = 1 - mask
    img[minmax[0]:minmax[1], minmax[2]:minmax[3], 0] *= anti_mask
    img[minmax[0]:minmax[1], minmax[2]:minmax[3], 1] *= anti_mask
    img[minmax[0]:minmax[1], minmax[2]:minmax[3], 2] *= anti_mask
    if isinstance(drawing, np.ndarray):
        if drawing.dtype != np.uint8:
            drawing = bracket_to_uint8_image_contrast(drawing)
        drawing = [drawing[:, :, 0], drawing[:, :, 1], drawing[:, :, 2]]
    img[minmax[0]:minmax[1], minmax[2]:minmax[3], 0] += mask * drawing[0]
    img[minmax[0]:minmax[1], minmax[2]:minmax[3], 1] += mask * drawing[1]
    img[minmax[0]:minmax[1], minmax[2]:minmax[3], 2] += mask * drawing[2]
    return img
