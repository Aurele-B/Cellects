#!/usr/bin/env python3
"""
This script contains methods to compare and modify shapes in binary images
It contains the following functions:
    - cc : Sort connected components according to sizes
    - make_gravity_field : put a gradient of decreasing numbers around a shape
    - find_median_shape : sum shapes and keem a median image of them
    - make_numbered_rays
    - CompareNeighborsWithFocal : get the number of neighbors having an
                                  equal/sup/inf value than each cell
    - CompareNeighborsWithValue : get the number of neighbors having an
                                  equal/sup/inf value than a given value
    - ShapeDescriptors
    - get_radius_distance_against_time : 3D, get a vector of radius distances
                                         with idx as time
    - expand_until_one
    - expand_and_rate_until_one
    - expand_until_overlap
    - expand_to_fill_holes
    - expand_smalls_toward_biggest
    - change_thresh_until_one
    - Ellipse
    - get_rolling_window_coordinates_list
"""
import logging
from copy import deepcopy
import cv2
import numpy as np
from scipy.spatial import KDTree
from numba import njit
from cellects.utils.formulas import moving_average
from skimage.filters import threshold_otsu
from skimage.measure import label


cross_33 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
square_33 = np.ones((3, 3), np.uint8)


class CompareNeighborsWithValue:
    def __init__(self, array, connectivity=None, data_type=np.int8):
        """
        Summarize each pixel (cell) of a 2D array by comparing its neighbors to a value.
        This comparison can be equal, inferior or superior.
        Neighbors can be the 4 or the 8 nearest pixels based on the value of connectivity.
        :param array: a 1 or 2D array
        :type array: must be less permissive than data_type
        :param connectivity: 4 or 8, if different, only compute diagonal
        :type connectivity: uint8
        :param data_type: the data type used for computation
        :type data_type: type
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

    def is_equal(self, value, and_itself=False):
        """
        Give, for each pixel, the number neighboring pixels having the value "value"
        :param value: any number. The equal_neighbor_nb matrix will contain, for each pixel,
        the number of neighboring pixels having that value.
        :param and_itself: When False, the resulting number of neighbors fitting the condition is displayed normally.
        When True, when the focal pixel does not fit the condition, it receives the value 0.
        In other words, the resulting number of neighbors fitting the condition is displayed
        if and only if the focal pixel ALSO fit the condition, otherwise, it will have the value 0.
        :type and_itself: bool
        :return: each cell of equal_neighbor_nb is the number of neighboring pixels having the value "value"
        :rtype: uint8
        """

        if len(self.array.shape) == 1:
            self.equal_neighbor_nb = self.on_the_right + self.on_the_left
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
        Give, for each pixel, the number neighboring pixels having a higher value than "value"
        :param value: any number. The sup_neighbor_nb matrix will contain, for each pixel,
        the number of number neighboring pixels having a higher value than "value".
        :param and_itself: When False, the resulting number of neighbors fitting the condition is displayed normally.
        When True, when the focal pixel does not fit the condition, it receives the value 0.
        In other words, the resulting number of neighbors fitting the condition is displayed
        if and only if the focal pixel ALSO fit the condition, otherwise, it will have the value 0.
        :type and_itself: bool
        :return: each cell of sup_neighbor_nb is the number of neighboring pixels having a value higher than "value"
        :rtype: uint8
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
        Give, for each pixel, the number neighboring pixels having a lower value than "value"
        :param value: any number. The inf_neighbor_nb matrix will contain, for each pixel,
        the number of number neighboring pixels having a lower value than "value".
        :param and_itself: When False, the resulting number of neighbors fitting the condition is displayed normally.
        When True, when the focal pixel does not fit the condition, it receives the value 0.
        In other words, the resulting number of neighbors fitting the condition is displayed
        if and only if the focal pixel ALSO fit the condition, otherwise it receives the value 0.
        :type and_itself: bool
        :return: each cell of sup_neighbor_nb is the number of neighboring pixels having a value lower than "value"
        :rtype: uint8
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


def cc(binary_img):
    """
    This method find and order the numbering of connected components according to their sizes
    stats columns order: left, top, right, bot
    Sort connected components according to sizes
    The shape touching more than 2 borders is considered as the background.
    If another shape touches more than 2 borders, the second larger shape is considered as background
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


def get_largest_connected_component(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0) # assume at least 1 CC
    con_comp = np.bincount(labels.flat)[1:]
    largest_connected_component = labels == np.argmax(con_comp) + 1
    return len(con_comp), largest_connected_component


def make_gravity_field(original_shape, max_distance=None, with_erosion=0):
    """
    gravity_field = scipy.ndimage.distance_transform_edt(1 - original_shape)
    Create a field containing a gradient around a shape
    :param original_shape: a binary matrix containing one connected component
    :type original_shape: uint8
    :param max_distance: maximal distance from the original shape the field can reach. 
    :type max_distance: uint64
    :param with_erosion: Tells how the original shape should be eroded before creating the field
    :type with_erosion: uint8
    
    :return: a matrix of the gravity field
    """
    kernel = cross_33
    if with_erosion > 0:
        original_shape = cv2.erode(original_shape, kernel, iterations=with_erosion, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    expand = deepcopy(original_shape)
    if max_distance is not None:
        if max_distance > np.min(original_shape.shape) / 2:
            max_distance = (np.min(original_shape.shape) // 2).astype(np.uint32)
        gravity_field = np.zeros(original_shape.shape , np.uint32)
        for gravi in np.arange(max_distance):
            expand = cv2.dilate(expand, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            gravity_field[np.logical_xor(expand, original_shape)] += 1
    else:
        gravity_field = np.zeros(original_shape.shape , np.uint32)
        while np.any(np.equal(original_shape + expand, 0)):
            expand = cv2.dilate(expand, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            gravity_field[np.logical_xor(expand, original_shape)] += 1
    return gravity_field


@njit()
def get_line_points(start, end):
    """
    Get all integer coordinates along a line from start to end.
    Uses a simple line drawing algorithm similar to Bresenham's.
    start, end = start_point, end_point
    Args:
        start: tuple (x, y) - starting point
        end: tuple (x, y) - ending point

    Returns:
        numpy array of shape (n, 2) with all integer coordinates
    """
    x0, y0 = start
    x1, y1 = end

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
        points.append([x, y])

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


def get_all_line_coordinates(start_point, end_points):
    """
    Get coordinates for lines from one point to many points.
    Automatically determines the right number of points for continuous lines.
    start_point, end_points = origin_centroid, skel_coord
    Args:
        start_point: tuple (x, y) - starting point
        end_points: list of tuples - ending points

    Returns:
        list of numpy arrays, each containing coordinates for one line
    """
    if not isinstance(start_point.dtype, float):
        start_point = start_point.astype(float)
    if not isinstance(end_points.dtype, float):
        end_points = end_points.astype(float)

    lines = []
    for end_point in end_points:
        line_coords = get_line_points(start_point, end_point)
        lines.append(np.array(line_coords, dtype =np.uint64))
    return lines


def get_every_coord_between_2_points(point_A, point_B):
    """
    Only work in a 2D space
    First check if the segment is vertical (first if) or horizontal (elif), in this case computation takes 3 rows
    Else, determine an approximation of what would be that continuous line in this discrete space.
    :param point_A: y and x coordinates of the first point in a 2D space
    :param point_B: y and x coordinates of the second point in a 2D space
    :return: a matrix of the y and x coordinates of all points forming the segment between point_A and point_B
    """
    xa = point_A[1]
    ya = point_A[0]
    xb = point_B[1]
    yb = point_B[0]

    if np.equal(xa, xb):
        sorted_y = np.sort((ya, yb))
        y_values = np.arange(sorted_y[0], sorted_y[1] + 1).astype(np.uint64)
        segment = np.vstack((y_values, np.repeat(xa, len(y_values)).astype(np.uint64)))
    elif np.equal(ya, yb):
        sorted_x = np.sort((xa, xb))
        x_values = np.arange(sorted_x[0], sorted_x[1] + 1).astype(np.uint64)
        segment = np.vstack((np.repeat(ya, len(x_values)).astype(np.uint64), x_values))
    else:
        # First, create vectors of integers linking the coordinates of two points
        X = np.arange(np.min((xa, xb)), np.max((xa, xb)) + 1).astype(np.uint64)
        Y = np.arange(np.min((ya, yb)), np.max((ya, yb)) + 1).astype(np.uint64)
        # If X is longer than Y, we make Y grow
        new_X = X
        new_Y = Y
        if len(X) > len(Y):
            new_Y = np.repeat(Y, np.round((len(X) / len(Y))))
        if len(X) < len(Y):
            new_X = np.repeat(X, np.round((len(Y) / len(X))))

        # Duplicate the last Y value until Y length reaches X length
        count = 0
        while len(new_X) > len(new_Y):
            new_Y = np.append(new_Y, new_Y[-1])
            count = count + 1
        while len(new_X) < len(new_Y):
            new_X = np.append(new_X, new_X[-1])
            count = count + 1

        if np.logical_or(np.logical_and(xb < xa, yb > ya), np.logical_and(xb > xa, yb < ya)):
            segment = np.vstack((new_Y, np.sort(new_X)[::-1]))
        else:
            segment = np.vstack((new_Y, new_X))
    return segment


def draw_me_a_sun(main_shape, cross_33, ray_length_coef=2):
    """
    Draw numbered rays around one shape. These rays are perpendicular to the tangent of the contour of the shape 
    :param main_shape: a binary matrix containing one connected component
    :param cross_33: A 3*3 matrix containing a binary cross
    :param ray_length_coef: coefficient telling the distance of the rays of the sun
    :return: a vector of the number of each ray, the shape with the numbered rays
    """
    img, stats, center = cc(main_shape)
    main_center = center[1, :]
    dilated_main_shape = cv2.dilate(main_shape, cross_33)
    dilated_main_shape -= main_shape
    first_ring_idx = np.nonzero(dilated_main_shape)
    second_ring_y = main_center[1] + ((first_ring_idx[0] - main_center[1]) * ray_length_coef)
    second_ring_x = main_center[0] + ((first_ring_idx[1] - main_center[0]) * ray_length_coef)
    # Make sure that no negative value try to make rays go beyond the image border
    while np.logical_and(np.logical_or(np.min(second_ring_y, 0) < 0, np.min(second_ring_x, 0) < 0),
                         ray_length_coef > 1):
        ray_length_coef -= 0.1
        second_ring_y = main_center[1] + ((first_ring_idx[0] - main_center[1]) * ray_length_coef)
        second_ring_x = main_center[0] + ((first_ring_idx[1] - main_center[0]) * ray_length_coef)

    second_ring_idx = ((np.round(second_ring_y).astype(np.uint64), np.round(second_ring_x).astype(np.uint64)))
    sun = np.zeros(main_shape.shape, np.uint32)
    # sun[second_ring_idx[0], second_ring_idx[1]]=1
    rays = np.arange(len(first_ring_idx[0])) + 2
    for ray in rays:
        segment = get_every_coord_between_2_points((first_ring_idx[0][ray - 2], first_ring_idx[1][ray - 2]),
                                    (second_ring_idx[0][ray - 2], second_ring_idx[1][ray - 2]))
        try:
            sun[segment[0], segment[1]] = ray
        except IndexError:
            logging.error("The algorithm allowing to correct errors around initial shape partially failed. The initial shape may be too close from the borders of the current arena")

    return rays, sun


def find_median_shape(binary_3d_matrix):
    """
    Sum along the first axis of a 3D matrix binary and create a binary matrix 
    of the pixels that are true half of the time.
    :param binary_3d_matrix:
    :type binary_3d_matrix: uint8
    :return: a 2D binary matrix
    :rtype: uint8
    """
    binary_2d_matrix = np.apply_along_axis(np.sum, 0, binary_3d_matrix)
    median_shape = np.zeros(binary_2d_matrix.shape, dtype=np.uint8)
    median_shape[np.greater_equal(binary_2d_matrix, binary_3d_matrix.shape[0] // 2)] = 1
    return median_shape


@njit()
def reduce_image_size_for_speed(image_of_2_shapes):
    """
    Divide the image into 4 rectangles.
    If the two shapes can be found in one of these, divide this rectangle into 4
    Repeat the above algorithm until image slicing separate the image

    :param image_of_2_shapes: a uint8 numpy array with 0 as background, 1 for one shape and 2 for the other
    :return: Return the smallest rectangle containing 1 and 2 simultaneously
    """
    sub_image = image_of_2_shapes.copy()
    y_size, x_size = sub_image.shape
    images_list = [sub_image]
    good_images = [0]
    sub_image = images_list[good_images[0]]
    while (len(good_images) == 1 | len(good_images) == 2) & y_size > 3 & x_size > 3:
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
        if len(good_images) == 2:
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


def get_minimal_distance_between_2_shapes(image_of_2_shapes, increase_speed=True):
    """
    Fast function to get the minimal distance between two shapes
    :param image_of_2_shapes: binary image
    :type image_of_2_shapes: uint8
    :param increase_speed: 
    :type increase_speed: bool
    :return: 
    """
    if increase_speed:
        shape1_idx, shape2_idx = reduce_image_size_for_speed(image_of_2_shapes)
    else:
        shape1_idx, shape2_idx = np.nonzero(image_of_2_shapes == 1), np.nonzero(image_of_2_shapes == 2)
    t = KDTree(np.transpose(shape1_idx))
    dists, nns = t.query(np.transpose(shape2_idx), 1)
    return np.min(dists)


def find_major_incline(vector, natural_noise):
    """
    Find the major incline of a curve given a certain level of noise
    :param vector: Values drawing a curve containing one major incline
    :param natural_noise: The extent of a curve containing no major incline in the same conditions
    :type natural_noise: uint64
    :return:
    vector = self.converted_video[self.start:self.substantial_time, subst_idx[0][index], subst_idx[1][index]]
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


def rank_from_top_to_bottom_from_left_to_right(binary_image, y_boundaries, get_ordered_image=False):
    """binary_image=self.first_image.validated_shapes; y_boundaries=self.first_image.y_boundaries; get_ordered_image=True
    :param binary_image: Zeros 2D array with ones where shapes have been detected and validated
    :param y_boundaries: Zeros 1D array of the vertical size of the image with 1 and -1 where rows start and end
    :param get_ordered_image: Boolean telling if output should contain a zeros 2D image with shapes drew with integer
    from 1 to the number of shape according to their position
    """
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_image.astype(np.uint8),
                                                                               connectivity=8)

    centroids = centroids[1:, :]
    final_order = np.zeros(centroids.shape[0], dtype=np.uint8)
    sorted_against_y = np.argsort(centroids[:, 1])
    # row_nb = (y_boundaries == 1).sum()
    row_nb = np.max(((y_boundaries == 1).sum(), (y_boundaries == - 1).sum()))
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
    ordered_centroids = centroids[final_order, :]
    ordered_stats = stats[1:, :]
    ordered_stats = ordered_stats[final_order, :]

    # If it fails, use another algo
    if (final_order == 0).sum() > 1:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_image.astype(np.uint8),
                                                                               connectivity=8)
        # First order according to x: from left to right
        # Remove the background and order centroids along x axis
        centroids = centroids[1:, :]
        x_order = np.argsort(centroids[:, 0])
        centroids = centroids[x_order, :]


        # Then use the boundaries of each Y peak to sort these shapes row by row
        if y_boundaries is not None:
            binary_image = deepcopy(output)
            binary_image[np.nonzero(binary_image)] = 1
            y_starts, y_ends = np.argwhere(y_boundaries == - 1), np.argwhere(y_boundaries == 1)

            margins_ci = np.array((0.5, 0.4, 0.3, 0.2, 0.1))
            for margin in margins_ci:
                ranking_success: bool = True
                y_order = np.zeros(centroids.shape[0], dtype=np.uint8)
                count: np.uint8 = 0
                y_margins = (y_ends - y_starts) * margin# 0.3
                # Loop and try to fill each row with all components, fail if the final number is wrong
                for y_interval in np.arange(len(y_starts)):
                    for patch_i in np.arange(nb_components - 1):
                        # Compare the y coordinate of the centroid with the detected y intervals with
                        # an added margin in order to order coordinates
                        if np.logical_and(centroids[patch_i, 1] >= (y_starts[y_interval] - y_margins[y_interval]),
                                          centroids[patch_i, 1] <= (y_ends[y_interval] + y_margins[y_interval])):
                            try:
                                y_order[count] = patch_i
                                count = count + 1
                            except IndexError as exc:
                                ranking_success = False

                if ranking_success:
                    break
        else:
            ranking_success = False
        # if that all tested margins failed, do not rank_from_top_to_bottom_from_left_to_right, i.e. keep automatic ranking
        if not ranking_success:
            y_order = np.arange(centroids.shape[0])


        # Second order according to y: from top to bottom
        ordered_centroids = centroids[y_order, :]
        ordered_stats = stats[1:, :]
        ordered_stats = ordered_stats[x_order, :]
        ordered_stats = ordered_stats[y_order, :]

    if get_ordered_image:
        ordered_image = np.zeros(binary_image.shape, dtype=np.uint8)
        for patch_j in np.arange(centroids.shape[0]):
            sub_output = output[ordered_stats[patch_j, 1]: (ordered_stats[patch_j, 1] + ordered_stats[patch_j, 3]), ordered_stats[patch_j, 0]: (ordered_stats[patch_j, 0] + ordered_stats[patch_j, 2])]
            sub_output = np.sort(np.unique(sub_output))
            if len(sub_output) == 1:
                ordered_image[output == sub_output[0]] = patch_j + 1
            else:
                ordered_image[output == sub_output[1]] = patch_j + 1


        return ordered_stats, ordered_centroids, ordered_image
    else:
        return ordered_stats, ordered_centroids


def expand_until_neighbor_center_gets_nearer_than_own(shape_to_expand, without_shape_i, shape_original_centroid,
                                                      ref_centroids, kernel):
    """
    Expand one shape until its border becomes nearer to the center of neighboring cells than to its own center
    :param shape_to_expand: Binary image containing the focal shape only
    :param shape_original_centroid: The centroid coordinates of the focal shape at the true beginning
    :param without_shape_i: Binary image all shapes except the focal one
    :param kernel: Binary matrix containing a circle of 1, copying roughly a slime mold growth
    :return: Binary image containing the focal shape only, but expanded until it reach a border
    or got too close from a neighbor
    """

    without_shape = deepcopy(without_shape_i)
    # Calculate the distance between the focal shape centroid and its 10% nearest neighbor centroids
    centroid_distances = np.sqrt(np.square(ref_centroids[1:, 0] - shape_original_centroid[0]) + np.square(
        ref_centroids[1:, 1] - shape_original_centroid[1]))
    nearest_shapes = np.where(np.greater(np.quantile(centroid_distances, 0.1), centroid_distances))[0]

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

    # Put 1 at the border of the reference image in order to be able to stop the while loop once border reached
    without_shape[0, :] = 1
    without_shape[:, 0] = 1
    without_shape[without_shape.shape[0] - 1, :] = 1
    without_shape[:, without_shape.shape[1] - 1] = 1

    # Compare the distance between the contour of the shape and its centroid with this contout with the centroids of neighbors
    # Continue as the distance made by the shape (from its centroid) keeps being smaller than its distance with the nearest centroid.
    previous_shape_to_expand = deepcopy(shape_to_expand)
    while np.logical_and(np.any(np.less_equal(itself_maxdist, neighbor_mindist)),
                         np.count_nonzero(shape_to_expand * without_shape) == 0):
        previous_shape_to_expand = deepcopy(shape_to_expand)
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


def image_borders(dimensions):
    """
    Create a matrix of dimensions "dimensions" containing ones everywhere except at borders (0)
    :param dimensions:
    :return:
    :rtype: uint8
    """
    borders = np.ones(dimensions, dtype=np.uint8)
    borders[0, :] = 0
    borders[:, 0] = 0
    borders[- 1, :] = 0
    borders[:, - 1] = 0
    return borders


def get_radius_distance_against_time(binary_video, field):
    """
    Find the relationship between distance in a gravity field and growth speed of a binary shape in a video
    :param binary_video: a binary video having a growing/moving shape
    :type binary_video: uint8
    :param field: a gravity field around an initial shape
    :return:
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


def expand_to_fill_holes(binary_video, holes):
    #first move should be the time at wich the first pixel hole could have been covered
    #it should ask how much time the shape made to cross a distance long enough to overlap all holes
    holes_contours = cv2.dilate(holes, cross_33, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    field = make_gravity_field(binary_video[0, :, :], (binary_video.shape[0] - 1))
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
        distance_against_time = [1, 2]

    return binary_video, holes_time_end, distance_against_time


def change_thresh_until_one(grayscale_image, binary_image, lighter_background):
    coord = np.nonzero(binary_image)
    min_cx = np.min(coord[0])
    max_cx = np.max(coord[0])
    min_cy = np.min(coord[1])
    max_cy = np.max(coord[1])
    gray_img = grayscale_image[min_cy:max_cy, min_cx:max_cx]
    # threshold = get_otsu_threshold(gray_img)
    threshold = threshold_otsu(gray_img)
    bin_img = (gray_img < threshold).astype(np.uint8)
    detected_shape_number, bin_img = cv2.connectedComponents(bin_img, ltype=cv2.CV_16U)
    while (detected_shape_number > 2) and (0 < threshold < 255):
        if lighter_background:
            threshold += 1
            bin_img = (gray_img < threshold).astype(np.uint8)
        else:
            threshold -= 1
            bin_img = (gray_img < threshold).astype(np.uint8)
        detected_shape_number, bin_img = cv2.connectedComponents(bin_img, ltype=cv2.CV_16U)
    binary_image = np.zeros_like(binary_image, np.uint8)
    binary_image[min_cy:max_cy, min_cx:max_cx] = bin_img
    return binary_image


class Ellipse:
    def __init__(self, sizes):
        """
        Usage: Ellipse
        """
        self.vsize = sizes[0]
        self.hsize = sizes[1]
        self.vr = self.hsize // 2
        self.hr = self.vsize // 2

    def ellipse_fun(self, x, y):
        """
        Create an ellipse of size x and y in a 2D array of size vsize and hsize
        :param x: ellipse size on x axis
        :param y: ellipse size on y axis
        :return: a binary image containing the ellipse
        """
        return (((x - self.hr) ** 2) / (self.hr ** 2)) + (((y - self.vr) ** 2) / (self.vr ** 2)) <= 1

    def create(self):
        # if self.hsize % 2 == 0:
        #     self.hsize += 1
        # if self.vsize % 2 == 0:
        #     self.vsize += 1
        return np.fromfunction(self.ellipse_fun, (self.vsize, self.hsize))


def get_rolling_window_coordinates_list(height, width, side_length, window_step, allowed_pixels=None):
    y_remain = height % side_length
    x_remain = width % side_length
    y_nb = height // side_length
    x_nb = width // side_length
    y_coord = np.arange(y_nb + 1, dtype =np.uint64) * side_length
    x_coord = np.arange(x_nb + 1, dtype =np.uint64) * side_length
    y_coord[-1] += y_remain
    x_coord[-1] += x_remain
    window_coords = []
    for y_i in range(len(y_coord) - 1):
        for x_i in range(len(x_coord) - 1):
            for add_to_y in np.arange(0, side_length, window_step, dtype =np.uint64):
                y_max = np.min((height, y_coord[y_i + 1] + add_to_y)).astype(np.uint64)
                for add_to_x in np.arange(0, side_length, window_step, dtype =np.uint64):
                    x_max = np.min((width, x_coord[x_i + 1] + add_to_x)).astype(np.uint64)
                    if allowed_pixels is None or np.any(allowed_pixels[(y_coord[y_i] + add_to_y):y_max, (x_coord[x_i] + add_to_x):x_max]):
                        window_coords.append([y_coord[y_i] + add_to_y, y_max, x_coord[x_i] + add_to_x, x_max])
    return window_coords


def get_contours(binary_image):
    eroded_binary = cv2.erode(binary_image, cross_33)
    contours = binary_image - eroded_binary
    return contours
