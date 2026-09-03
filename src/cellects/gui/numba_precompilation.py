#!/usr/bin/env python3
"""
This module precompile all njit decorated functions used by Cellects.
"""
import numpy as np
from numba.typed import  Dict, List
from cellects.image.image_segmentation import otsu_thresholding, combine_color_spaces, _get_counts_jit
from cellects.image.morphological_operations import get_line_points, reduce_image_size_for_speed, keep_largest_shape
from cellects.image.network_functions import nonzero_to_set
from cellects.utils.formulas import sum_of_abs_differences, bracket_to_uint8_image_contrast, get_power_dists, get_var, \
    get_skewness_kurtosis, get_inertia_axes, get_newly_explored_area
from cellects.utils.utilitarian import greater_along_first_axis, less_along_first_axis


def warming_up_numba_functions(loading):
    """
    This function calls all njit decorated functions used by Cellects.

    It also update a loading object to display a progress bar during launching.
    """
    integer = np.uint8(1)
    vect = np.zeros((1), dtype=np.float64)
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    b_img = np.zeros((1, 1), dtype=np.uint8)
    b_vid = np.zeros((1, 1, 1), dtype=np.uint8)
    c_space_dict = Dict()
    c_space_dict['bgr'] = List([0, 0, 0])
    all_c_spaces = Dict()
    all_c_spaces['bgr'] = img
    subtract_background = img[:, :, 0]
    mo = Dict()
    mo['m00'] = 1.0
    mo["m10"] = 2.
    mo["m01"] = 3.
    mo["m20"] = 4.
    mo["m02"] = 1.
    mo["m11"] = 5.

    loading.add_progress()
    _ = combine_color_spaces(c_space_dict, all_c_spaces, subtract_background)

    loading.add_progress()
    _ = otsu_thresholding(img)

    loading.add_progress()
    _ = sum_of_abs_differences(vect, vect)

    loading.add_progress()
    _ = bracket_to_uint8_image_contrast(img)

    loading.add_progress()
    _, _ = get_power_dists(img, 5.0, 5.0, 2)

    loading.add_progress()
    _, _ = get_var(mo, img[:, :, 0], vect, vect)

    loading.add_progress()
    _, _ = get_skewness_kurtosis(1.5, 2.0, 0.5, 0.75, 3)

    loading.add_progress()
    _, _, _, _, _ = get_inertia_axes(mo)

    loading.add_progress()
    _ = get_newly_explored_area(b_vid)

    loading.add_progress()
    _, _ = _get_counts_jit(integer, vect, vect)

    loading.add_progress()
    get_line_points((0, 0), (1, 0))

    loading.add_progress()
    _ = reduce_image_size_for_speed(b_img)

    loading.add_progress()
    indexed_shapes = np.array([1, 0, 1], dtype=np.int32)

    loading.add_progress()
    _ = keep_largest_shape(indexed_shapes)

    loading.add_progress()
    _ = nonzero_to_set(b_img)

    loading.add_progress()
    _ = greater_along_first_axis(b_img, vect)

    loading.add_progress()
    _ = less_along_first_axis(b_img, vect)

    # Not yet in:
    # linear_model
    # _linregress
    # _cluster_means
    # _slope_shifts
    # _fill_r_squared_matrices