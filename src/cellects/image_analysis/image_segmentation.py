#!/usr/bin/env python3
"""Module for image segmentation operations including filtering, color space conversion, thresholding, and quality assessment.

This module provides tools to process images through grayscale conversion, apply various filters (e.g., Gaussian, Median, Butterworth), perform thresholding methods like Otsu's algorithm, combine color spaces for enhanced segmentation, and evaluate binary image quality. Key functionalities include dynamic background subtraction, rolling window segmentation with localized thresholds, and optimization of segmentation masks using shape descriptors.

Functions
apply_filter : Apply skimage or OpenCV-based filters to grayscale images.
get_color_spaces : Convert BGR images into specified color space representations (e.g., LAB, HSV).
combine_color_spaces : Merge multiple color channels with coefficients to produce a segmented image.
generate_color_space_combination : Create custom grayscale combinations using two sets of channel weights and backgrounds.
otsu_thresholding : Binarize an image using histogram-based Otsu thresholding.
segment_with_lum_value : Segment video frames using luminance thresholds adjusted for background variation.
rolling_window_segmentation : Apply localized Otsu thresholding across overlapping patches to improve segmentation accuracy.
binary_quality_index : Calculate a quality metric based on perimeter and connected components in binary images.
find_threshold_given_mask : Binary search optimization to determine optimal threshold between masked regions.

Notes
Uses Numba's @njit decorator for JIT compilation of performance-critical functions like combine_color_spaces and _get_counts_jit.
"""
import numpy as np
import cv2
from tqdm import tqdm
from numba.typed import Dict
from cellects.utils.decorators import njit
from numpy.typing import NDArray
from typing import Tuple
from cellects.utils.utilitarian import less_along_first_axis, greater_along_first_axis, translate_dict, split_dict
from cellects.utils.formulas import bracket_to_uint8_image_contrast
from cellects.image_analysis.morphological_operations import get_largest_connected_component
from skimage.measure import perimeter
from scipy.optimize import minimize
from skimage.filters import (threshold_otsu, gaussian, butterworth, farid, frangi, hessian, laplace, median, meijering,
                             prewitt, roberts, sato, scharr, sobel)


filter_dict = {"": {'': {}},
               "Gaussian": {'Param1': {'Name': 'Sigma:', 'Minimum': 0., 'Maximum': 1000., 'Default': 1.}},
               "Median": {'': {}},
               "Butterworth": {'Param1': {'Name': 'Cutoff fr:', 'Minimum': 0., 'Maximum': .5, 'Default': .005},
                               'Param2': {'Name': 'Order:', 'Minimum': 0., 'Maximum': 1000., 'Default': 2.}},
               "Frangi": {'Param1': {'Name': 'Sigma min:', 'Minimum': 0., 'Maximum': 1000., 'Default': .5},
                          'Param2': {'Name': 'Sigma max:', 'Minimum': 0., 'Maximum': 1000., 'Default': 5.}},
               "Sato": {'Param1': {'Name': 'Sigma min:', 'Minimum': 0., 'Maximum': 1000., 'Default': .5},
                          'Param2': {'Name': 'Sigma max:', 'Minimum': 0., 'Maximum': 1000., 'Default': 5.}},
               "Meijering": {'Param1': {'Name': 'Sigma min:', 'Minimum': 0., 'Maximum': 1000., 'Default': 1.},
                          'Param2': {'Name': 'Sigma max:', 'Minimum': 0., 'Maximum': 1000., 'Default': 10.}},
               "Hessian": {'Param1': {'Name': 'Sigma min:', 'Minimum': 0., 'Maximum': 1000., 'Default': 1.},
                          'Param2': {'Name': 'Sigma max:', 'Minimum': 0., 'Maximum': 1000., 'Default': 10.}},
               "Laplace": {'Param1': {'Name': 'Ksize:', 'Minimum': 3, 'Maximum': 100, 'Default': 5}},
               "Sharpen": {'': {}},
               "Mexican hat": {'': {}},
               "Farid": {'': {}},
               "Prewitt": {'': {}},
               "Scharr": {'': {}},
               "Sobel": {'': {}},
               }


def apply_filter(image: NDArray, filter_type: str, param, rescale_to_uint8=False) -> NDArray:
    """
    Apply various filters to an image based on the specified filter type.

    This function applies a filter to the input image according to the
    specified `filter_type` and associated parameters. Supported filters
    include Gaussian, Median, Butterworth, Frangi, Sato, Meijering,
    Hessian, Laplace, Mexican hat, Farid, Prewitt, Roberts, Scharr, and Sobel.
    Except from Sharpen and Mexican hat, these filters are implemented using the skimage.filters module.
    Additionally, the function can rescale the output image to uint8
    format if specified.

    Parameters
    ----------
    image : NDArray
        The input image to which the filter will be applied.
    filter_type : str
        The type of filter to apply. Supported values include:
        "Gaussian", "Median", "Butterworth", "Frangi",
        "Sato", "Meijering", "Hessian", "Laplace", "Mexican hat",
        "Sharpen", "Farid", "Prewitt", "Roberts", "Scharr", and "Sobel".
    param : list or tuple
        Parameters specific to the filter type. The structure of `param`
        depends on the chosen filter.
    rescale_to_uint8 : bool, optional
        Whether to rescale the output image to uint8 format. Default is False.

    Notes
    -----
    The Sharpen filter is implemented through:
    cv2.filter2D(image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    The Maxican hat filter is implemented through:
    cv2.filter2D(image, -1, np.array(
            [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]))
    All other filters are skimage filters.

    Returns
    -------
    NDArray
        The filtered image. If `rescale_to_uint8` is True and the output
        image's dtype is not uint8, it will be rescaled accordingly.

    Examples
    --------
    >>> image = np.zeros((3, 3))
    >>> image[1, 1] = 1
    >>> filtered_image = apply_filter(image, "Gaussian", [1.0])
    >>> print(filtered_image)
    [[0.05855018 0.09653293 0.05855018]
     [0.09653293 0.15915589 0.09653293]
     [0.05855018 0.09653293 0.05855018]]
    Filtered image with Gaussian filter.

    >>> image = np.zeros((3, 3))
    >>> image[1, 1] = 1
    >>> filtered_image = apply_filter(image, "Median", [])
    >>> print(filtered_image)
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    Filtered image with Median filter.

    >>> image = np.zeros((3, 3))
    >>> image[1, 1] = 1
    >>> filtered_image = apply_filter(image, "Butterworth", [0.005, 2])
    >>> print(filtered_image)
    [[-0.1111111  -0.11111111 -0.1111111 ]
    [-0.11111111  0.88888886 -0.11111111]
    [-0.1111111  -0.11111111 -0.1111111 ]]
    Filtered image with Butterworth filter.
    """
    if filter_type == "Gaussian":
        image = gaussian(image, sigma=param[0])
    elif filter_type == "Median":
        image = median(image)
    elif filter_type == "Butterworth":
        image = butterworth(image, cutoff_frequency_ratio=param[0], order=param[1])
    elif filter_type == "Frangi":
        image = frangi(image, sigmas=np.linspace(param[0], param[1], num=3))
    elif filter_type == "Sato":
        image = sato(image, sigmas=np.linspace(param[0], param[1], num=3))
    elif filter_type == "Meijering":
        image = meijering(image, sigmas=np.linspace(param[0], param[1], num=3))
    elif filter_type == "Hessian":
        image = hessian(image, sigmas=np.linspace(param[0], param[1], num=3))
    elif filter_type == "Laplace":
        image = laplace(image, ksize=np.max((3, int(np.ceil(param[0])))))
    elif filter_type == "Sharpen":
        image = cv2.filter2D(image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    elif filter_type == "Mexican hat":
        image = cv2.filter2D(image, -1, np.array(
            [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]))
    elif filter_type == "Farid":
        image = farid(image)
    elif filter_type == "Prewitt":
        image = prewitt(image)
    elif filter_type == "Roberts":
        image = roberts(image)
    elif filter_type == "Scharr":
        image = scharr(image)
    elif filter_type == "Sobel":
        image = sobel(image)
    if rescale_to_uint8 and image.dtype != np.uint8:
        image = bracket_to_uint8_image_contrast(image)
    return image


def get_color_spaces(bgr_image: NDArray[np.uint8], space_names: list="") -> Dict:
    """
    Convert a BGR image into various color spaces.

    Converts the input BGR image to specified color spaces and returns them
    as a dictionary. If no space names are provided, converts to all default
    color spaces (LAB, HSV, LUV, HLS, YUV). If 'logical' is in the space names,
    it will be removed before conversion.

    Parameters
    ----------
    bgr_image : ndarray of uint8
        Input image in BGR color space.
    space_names : list of str, optional
        List of color spaces to convert the image to. Defaults to none.

    Returns
    -------
    out : dict
        Dictionary with keys as color space names and values as the converted images.

    Examples
    --------
    >>> bgr_image = np.zeros((5, 5, 3), dtype=np.uint8)
    >>> c_spaces = get_color_spaces(bgr_image, ['lab', 'hsv'])
    >>> print(list(c_spaces.keys()))
    ['bgr', 'lab', 'hsv']
    """
    if 'logical' in space_names:
        space_names.pop(np.nonzero(np.array(space_names, dtype=str) == 'logical')[0][0])
    c_spaces = Dict()
    c_spaces['bgr'] = bgr_image.astype(np.float64)
    if len(space_names) == 0:
        c_spaces['lab'] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB).astype(np.float64)
        c_spaces['hsv'] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float64)
        c_spaces['luv'] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LUV).astype(np.float64)
        c_spaces['hls'] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS).astype(np.float64)
        c_spaces['yuv'] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV).astype(np.float64)
    else:
        if np.isin('lab', space_names):
            c_spaces['lab'] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB).astype(np.float64)
        if np.isin('hsv', space_names):
            c_spaces['hsv'] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float64)
        if np.isin('luv', space_names):
            c_spaces['luv'] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LUV).astype(np.float64)
        if np.isin('hls', space_names):
            c_spaces['hls'] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS).astype(np.float64)
        if np.isin('yuv', space_names):
            c_spaces['yuv'] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV).astype(np.float64)
    return c_spaces


@njit()
def combine_color_spaces(c_space_dict: Dict, all_c_spaces: Dict, subtract_background: NDArray=None) -> NDArray:
    """
    Combine color spaces from a dictionary and generate an analyzable image.

    This function processes multiple color spaces defined in `c_space_dict`, combines
    them according to given coefficients, and produces a normalized image that can be
    converted to uint8. Optionally subtracts background from the resultant image.

    Parameters
    ----------
    c_space_dict : dict
        Dictionary containing color spaces and their respective coefficients.
    all_c_spaces : Dict
        Dictionary of all available color spaces in the image.
    subtract_background : NDArray, optional
        Background image to subtract from the resultant image. Defaults to None.

    Returns
    -------
    out : NDArray
        Processed and normalized image in float64 format, ready for uint8 conversion.

    Examples
    --------
    >>> c_space_dict = Dict()
    >>> c_space_dict['hsv'] = np.array((0, 1, 1))
    >>> all_c_spaces = Dict()
    >>> all_c_spaces['bgr'] = np.random.rand(5, 5, 3)
    >>> all_c_spaces['hsv'] = np.random.rand(5, 5, 3)
    >>> background = np.zeros((5, 5))
    >>> result = combine_color_spaces(c_space_dict, all_c_spaces)
    >>> print(result.shape)
    (5, 5)
    """
    image = np.zeros((all_c_spaces['bgr'].shape[0], all_c_spaces['bgr'].shape[1]), dtype=np.float64)
    for space, channels in c_space_dict.items():
        image += c_space_dict[space][0] * all_c_spaces[space][:, :, 0] + c_space_dict[space][1] * \
                 all_c_spaces[space][:, :, 1] + c_space_dict[space][2] * all_c_spaces[space][:, :, 2]
    if subtract_background is not None:
        # add (resp. subtract) the most negative (resp. smallest) value to the whole matrix to get a min = 0
        image -= np.min(image)
        # Make analysable this image by bracketing its values between 0 and 255 and converting it to uint8
        max_im = np.max(image)
        if max_im != 0:
            image = 255 * (image / np.max(image))
        if image.sum() > subtract_background.sum():
            image -= subtract_background
        else:
            image = subtract_background - image
    # add (resp. subtract) the most negative (resp. smallest) value to the whole matrix to get a min = 0
    image -= np.min(image)
    # Make analysable this image by bracketing its values between 0 and 255
    max_im = np.max(image)
    if max_im != 0:
        image = 255 * (image / max_im)
    return image
# c_space_dict=first_dict; all_c_spaces=self.all_c_spaces; subtract_background=background


def generate_color_space_combination(bgr_image: NDArray[np.uint8], c_spaces: list, first_dict: Dict, second_dict: Dict={}, background: NDArray=None, background2: NDArray=None, convert_to_uint8: bool=False, all_c_spaces: dict={}) -> NDArray[np.uint8]:
    """
    Generate color space combinations for an input image.

    This function generates a grayscale image by combining multiple color spaces
    from an input BGR image and provided dictionaries. Optionally, it can also generate
    a second grayscale image using another dictionary.

    Parameters
    ----------
    bgr_image : ndarray of uint8
        The input image in BGR color space.
    c_spaces : list
        List of color spaces to consider for combination.
    first_dict : Dict
        Dictionary containing color space and transformation details for the first grayscale image.
    second_dict : Dict, optional
        Dictionary containing color space and transformation details for the second grayscale image.
    background : ndarray, optional
        Background image to be used. Default is None.
    background2 : ndarray, optional
        Second background image to be used for the second grayscale image. Default is None.
    convert_to_uint8 : bool, optional
        Flag indicating whether to convert the output images to uint8. Default is False.

    Returns
    -------
    out : tuple of ndarray of uint8
        A tuple containing the first and second grayscale images.

    Examples
    --------
    >>> bgr_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> c_spaces = ['bgr', 'hsv']
    >>> first_dict = Dict()
    >>> first_dict['bgr'] = np.array((0, 1, 1))
    >>> second_dict = Dict()
    >>> second_dict['hsv'] = np.array((0, 0, 1))
    >>> greyscale_image1, greyscale_image2 = generate_color_space_combination(bgr_image, c_spaces, first_dict, second_dict)
    >>> print(greyscale_image1.shape)
    (100, 100)
    """
    greyscale_image2 = None
    first_pc_vector = None
    if "PCA" in c_spaces:
        greyscale_image, var_ratio, first_pc_vector = extract_first_pc(bgr_image)
    else:
        if len(all_c_spaces) == 0:
            all_c_spaces = get_color_spaces(bgr_image, c_spaces)
        try:
            greyscale_image = combine_color_spaces(first_dict, all_c_spaces, background)
        except:
            first_dict = translate_dict(first_dict)
            greyscale_image = combine_color_spaces(first_dict, all_c_spaces, background)
        if len(second_dict) > 0:
            greyscale_image2 = combine_color_spaces(second_dict, all_c_spaces, background2)

    if convert_to_uint8:
        greyscale_image = bracket_to_uint8_image_contrast(greyscale_image)
        if greyscale_image2 is not None and len(second_dict) > 0:
            greyscale_image2 = bracket_to_uint8_image_contrast(greyscale_image2)
    return greyscale_image, greyscale_image2, all_c_spaces, first_pc_vector


@njit()
def get_otsu_threshold(image: NDArray):
    """
    Calculate the optimal threshold value for an image using Otsu's method.

    This function computes the Otsu's thresholding which automatically
    performs histogram shape analysis for threshold selection.

    Parameters
    ----------
    image : NDArray
        The input grayscale image, represented as a NumPy array.

    Returns
    -------
    int or float
        The computed Otsu's threshold value.
    """
    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    if weight1.all():
        mean1 = np.cumsum(hist * bin_mids) / weight1
    else:
        mean1 = np.zeros_like(bin_mids)

    # Get the class means mu1(t)
    if weight2.all():
        mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    else:
        mean2 = np.zeros_like(bin_mids)

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    return threshold


@njit()
def otsu_thresholding(image: NDArray) -> NDArray[np.uint8]:
    """
    Apply Otsu's thresholding to a grayscale image.

    This function calculates the optimal threshold using
    Otsu's method and applies it to binarize the input image.
    The output is a binary image where pixel values are either
    0 or 1.

    Parameters
    ----------
    image : ndarray
        Input grayscale image with any kind of value.

    Returns
    -------
    out : ndarray of uint8
        Binarized image with pixel values 0 or 1.

    Examples
    --------
    >>> image = np.array([10, 20, 30])
    >>> result = otsu_thresholding(image)
    >>> print(result)
    [1 0 0]
    """
    threshold = get_otsu_threshold(image)
    binary_image = (image > threshold)
    if binary_image.sum() < binary_image.size / 2:
        return binary_image.astype(np.uint8)
    else:
        return np.logical_not(binary_image).astype(np.uint8)


def segment_with_lum_value(converted_video: NDArray, basic_bckgrnd_values: NDArray, l_threshold, lighter_background: bool) -> Tuple[NDArray, NDArray]:
    """
    Segment video frames based on luminance threshold.

    This function segments the input video frames by comparing against a dynamic
    luminance threshold. The segmentation can be based on either lighter or darker
    background.

    Parameters
    ----------
    converted_video : ndarray
        The input video frames in numpy array format.

    basic_bckgrnd_values : ndarray
        Array containing background values for each frame.

    l_threshold : int or float
        The luminance threshold value for segmentation.

    lighter_background : bool, optional
        If True, the segmentation is done assuming a lighter background.
        Defaults to False.

    Returns
    -------
    segmentation : ndarray
        Array containing the segmented video frames.
    l_threshold_over_time : ndarray
        Computed threshold over time for each frame.

    Examples
    --------
    >>> converted_video = np.array([[[100, 120], [130, 140]], [[160, 170], [180, 200]]], dtype=np.uint8)
    >>> basic_bckgrnd_values = np.array([100, 120])
    >>> lighter_background = False
    >>> l_threshold = 130
    >>> segmentation, threshold_over_time = segment_with_lum_value(converted_video, basic_bckgrnd_values, l_threshold, lighter_background)
    >>> print(segmentation)
    [[[0 1]
      [1 1]]
     [[1 1]
      [1 1]]]

    """
    # segmentation = None
    if lighter_background:
        l_threshold_over_time = l_threshold - (basic_bckgrnd_values[-1] - basic_bckgrnd_values)
        if np.all(np.logical_and(0 <= l_threshold_over_time, l_threshold_over_time <= 255)):
            segmentation = less_along_first_axis(converted_video, l_threshold_over_time)
        else:
            segmentation = np.zeros_like(converted_video)
            if l_threshold > 255:
                l_threshold = 255
            segmentation += converted_video > l_threshold
    else:
        l_threshold_over_time = l_threshold - (basic_bckgrnd_values[-1] - basic_bckgrnd_values)
        if np.all(np.logical_and(0 <= l_threshold_over_time, l_threshold_over_time <= 255)):
            segmentation = greater_along_first_axis(converted_video, l_threshold_over_time)
        else:
            segmentation = np.zeros_like(converted_video)
            if l_threshold > 255:
                l_threshold = 255
            segmentation += converted_video > l_threshold
    return segmentation, l_threshold_over_time



def kmeans(greyscale: NDArray, greyscale2: NDArray=None, kmeans_clust_nb: int=2,
           bio_mask: NDArray[np.uint8]=None, back_mask: NDArray[np.uint8]=None, logical: str='None',
           bio_label=None, bio_label2=None, previous_binary_image: NDArray[np.uint8]=None):
    """

    Perform K-means clustering on a greyscale image to generate binary images.

    Extended Description
    --------------------
    This function applies the K-means algorithm to a greyscale image or pair of images to segment them into binary images. It supports optional masks and previous segmentation labels for refining the clustering.

    Parameters
    ----------
    greyscale : NDArray
        The input greyscale image to segment.
    greyscale2 : NDArray, optional
        A second greyscale image for logical operations. Default is `None`.
    kmeans_clust_nb : int, optional
        Number of clusters for K-means. Default is `2`.
    bio_mask : NDArray[np.uint8], optional
        Mask for selecting biological objects. Default is `None`.
    back_mask : NDArray[np.uint8], optional
        Mask for selecting background regions. Default is `None`.
    logical : str, optional
        Logical operation flag to enable processing of the second image. Default is `'None'`.
    bio_label : int, optional
        Label for biological objects in the first segmentation. Default is `None`.
    bio_label2 : int, optional
        Label for biological objects in the second segmentation. Default is `None`.
    previous_binary_image : NDArray[np.uint8], optional
        Previous binary image for refinement. Default is `None`.

    Other Parameters
    ----------------
    **greyscale2, logical, bio_label2**: Optional parameters for processing a second image with logical operations.

    Returns
    -------
    tuple
        A tuple containing:
        - `binary_image`: Binary image derived from the first input.
        - `binary_image2`: Binary image for the second input if processed, else `None`.
        - `new_bio_label`: New biological label for the first segmentation.
        - `new_bio_label2`: New biological label for the second segmentation, if applicable.

    Notes
    -----
    - The function performs K-means clustering with random centers.
    - If `logical` is not `'None'`, both images are processed.
    - Default clustering uses 2 clusters, modify `kmeans_clust_nb` for different needs.

    """
    if isinstance(bio_mask, np.ndarray):
        bio_mask = np.nonzero(bio_mask)
    if isinstance(back_mask, np.ndarray):
        back_mask = np.nonzero(back_mask)
    new_bio_label = None
    new_bio_label2 = None
    binary_image2 = None
    image = greyscale.reshape((-1, 1))
    image = np.float32(image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, label, center = cv2.kmeans(image, kmeans_clust_nb, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    kmeans_image = np.uint8(label.flatten().reshape(greyscale.shape[:2]))
    sum_per_label = np.zeros(kmeans_clust_nb)
    binary_image = np.zeros(greyscale.shape[:2], np.uint8)
    if previous_binary_image is not None:
        binary_images = []
        image_scores = np.zeros(kmeans_clust_nb, np.uint64)
        for i in range(kmeans_clust_nb):
            binary_image_i = np.zeros(greyscale.shape[:2], np.uint8)
            binary_image_i[kmeans_image == i] = 1
            image_scores[i] = (binary_image_i * previous_binary_image).sum()
            binary_images.append(binary_image_i)
        binary_image[kmeans_image == np.argmax(image_scores)] = 1
    elif bio_label is not None:
        binary_image[kmeans_image == bio_label] = 1
        new_bio_label = bio_label
    else:
        if bio_mask is not None:
            all_labels = kmeans_image[bio_mask[0], bio_mask[1]]
            for i in range(kmeans_clust_nb):
                sum_per_label[i] = (all_labels == i).sum()
            new_bio_label = np.argsort(sum_per_label)[1]
        elif back_mask is not None:
            all_labels = kmeans_image[back_mask[0], back_mask[1]]
            for i in range(kmeans_clust_nb):
                sum_per_label[i] = (all_labels == i).sum()
            new_bio_label = np.argsort(sum_per_label)[-2]
        else:
            for i in range(kmeans_clust_nb):
                sum_per_label[i] = (kmeans_image == i).sum()
            new_bio_label = np.argsort(sum_per_label)[-2]
        binary_image += np.isin(kmeans_image, new_bio_label)

    if logical != 'None' and greyscale2 is not None:
        image = greyscale2.reshape((-1, 1))
        image = np.float32(image)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, label, center = cv2.kmeans(image, kmeans_clust_nb, None, criteria, attempts=10,
                                                flags=cv2.KMEANS_RANDOM_CENTERS)
        kmeans_image = np.uint8(label.flatten().reshape(greyscale.shape[:2]))
        sum_per_label = np.zeros(kmeans_clust_nb)
        binary_image2 = np.zeros(greyscale.shape[:2], np.uint8)
        if previous_binary_image is not None:
            binary_images = []
            image_scores = np.zeros(kmeans_clust_nb, np.uint64)
            for i in range(kmeans_clust_nb):
                binary_image_i = np.zeros(greyscale.shape[:2], np.uint8)
                binary_image_i[kmeans_image == i] = 1
                image_scores[i] = (binary_image_i * previous_binary_image).sum()
                binary_images.append(binary_image_i)
            binary_image2[kmeans_image == np.argmax(image_scores)] = 1
        elif bio_label2 is not None:
            binary_image2[kmeans_image == bio_label2] = 1
            new_bio_label2 = bio_label2
        else:
            if bio_mask is not None:
                all_labels = kmeans_image[bio_mask[0], bio_mask[1]]
                for i in range(kmeans_clust_nb):
                    sum_per_label[i] = (all_labels == i).sum()
                new_bio_label2 = np.argsort(sum_per_label)[1]
            elif back_mask is not None:
                all_labels = kmeans_image[back_mask[0], back_mask[1]]
                for i in range(kmeans_clust_nb):
                    sum_per_label[i] = (all_labels == i).sum()
                new_bio_label2 = np.argsort(sum_per_label)[-2]
            else:
                for i in range(kmeans_clust_nb):
                    sum_per_label[i] = (kmeans_image == i).sum()
                new_bio_label2 = np.argsort(sum_per_label)[-2]
            binary_image2[kmeans_image == new_bio_label2] = 1
    return binary_image, binary_image2, new_bio_label, new_bio_label2


def windowed_thresholding(image:NDArray, lighter_background: bool=None, side_length: int=None, step: int=None, min_int_var: float=None):
    """
    Perform grid segmentation on the image.

    This method applies a sliding window approach to segment the image into
    a grid-like pattern based on intensity variations and optionally uses a mask.
    The segmented regions are stored in `self.binary_image`.

    Args:
        lighter_background (bool): If True, areas lighter than the Otsu threshold are considered;
            otherwise, darker areas are considered.
        side_length (int, optional): The size of each grid square. Default is None.
        step (int, optional): The step size for the sliding window. Default is None.
        min_int_var (int, optional): Threshold for intensity variation within a grid.
            Default is 20.
        mask (NDArray, optional): A binary mask to restrict the segmentation area. Default is None.
    """
    if lighter_background is None:
        binary_image = otsu_thresholding(image)
        lighter_background = binary_image.sum() > (binary_image.size / 2)
    if min_int_var is None:
        min_int_var = np.ptp(image).astype(np.float64) * 0.1
    if side_length is None:
        side_length = int(np.min(image.shape) // 10)
    if step is None:
        step = side_length // 2
    grid_image = np.zeros(image.shape, np.uint64)
    homogeneities = np.zeros(image.shape, np.uint64)
    mask = np.ones(image.shape, np.uint64)
    for to_add in np.arange(0, side_length, step):
        y_windows = np.arange(0, image.shape[0], side_length)
        x_windows = np.arange(0, image.shape[1], side_length)
        y_windows += to_add
        x_windows += to_add
        for y_start in y_windows:
            if y_start < image.shape[0]:
                y_end = y_start + side_length
                if y_end < image.shape[0]:
                    for x_start in x_windows:
                        if x_start < image.shape[1]:
                            x_end = x_start + side_length
                            if x_end < image.shape[1]:
                                if np.any(mask[y_start:y_end, x_start:x_end]):
                                    potential_detection = image[y_start:y_end, x_start:x_end]
                                    if np.any(potential_detection):
                                        if np.ptp(potential_detection[np.nonzero(potential_detection)]) < min_int_var:
                                            homogeneities[y_start:y_end, x_start:x_end] += 1
                                        threshold = get_otsu_threshold(potential_detection)
                                        if lighter_background:
                                            net_coord = np.nonzero(potential_detection < threshold)
                                        else:
                                            net_coord = np.nonzero(potential_detection > threshold)
                                        grid_image[y_start + net_coord[0], x_start + net_coord[1]] += 1

    binary_image = (grid_image >= (side_length // step)).astype(np.uint8)
    binary_image[homogeneities >= (((side_length // step) // 2) + 1)] = 0
    return binary_image


def _network_perimeter(threshold, img: NDArray):
    """
    Calculate the negative perimeter of a binary image created from an input image based on a threshold.

    This function takes an image and a threshold value to create a binary
    image, then calculates the negative perimeter of that binary image.

    Parameters
    ----------
    threshold : float
        The threshold value to apply to the input image.
    img : ndarray
        The input grayscale image as a NumPy array.

    Returns
    -------
    out : float
        The negative perimeter of the binary image created from the input
        image and threshold.

    Examples
    --------
    >>> img = np.array([[1, 2, 1, 1], [1, 3, 4, 1], [2, 4, 3, 1], [2, 1, 2, 1]])
    >>> _network_perimeter(threshold=2.5, img=img)
    -4
    """
    binary_img = img > threshold
    return -perimeter(binary_img)


def rolling_window_segmentation(greyscale_image: NDArray, possibly_filled_pixels: NDArray, patch_size: tuple=(10, 10)) -> NDArray[np.uint8]:
    """
    Perform rolling window segmentation on a greyscale image, using potentially filled pixels and a specified patch size.

    The function divides the input greyscale image into overlapping patches defined by `patch_size`,
    and applies Otsu's thresholding method to each patch. The thresholds can be optionally
    refined using a minimization algorithm.

    Parameters
    ----------
    greyscale_image : ndarray of uint8
        The input greyscale image to segment.
    possibly_filled_pixels : ndarray of uint8
        An array indicating which pixels are possibly filled.
    patch_size : tuple, optional
        The dimensions of the patches to segment. Default is (10, 10).
        Must be superior to (1, 1).

    Returns
    -------
    output : ndarray of uint8
        The segmented binary image where the network is marked as True.

    Examples
    --------
    >>> greyscale_image = np.array([[1, 2, 1, 1], [1, 3, 4, 1], [2, 4, 3, 1], [2, 1, 2, 1]])
    >>> possibly_filled_pixels = greyscale_image > 1
    >>> patch_size = (2, 2)
    >>> result = rolling_window_segmentation(greyscale_image, possibly_filled_pixels, patch_size)
    >>> print(result)
    [[0 1 0 0]
     [0 1 1 0]
     [0 1 1 0]
     [0 0 1 0]]
    """
    patch_centers = [
        np.floor(np.linspace(
            p // 2, s - p // 2, int(np.ceil(s / (p // 2))) - 1
        )).astype(int)
        for s, p in zip(greyscale_image.shape, patch_size)
    ]
    patch_centers = np.transpose(np.meshgrid(*patch_centers), (1, 2, 0)).reshape((-1, 2))

    patch_slices = [
        tuple(slice(c - p // 2, c + p // 2, 1)
              for c, p in zip(p_c, patch_size)) for p_c in patch_centers
    ]
    maximize_parameter = False

    network_patches = []
    patch_thresholds = []
    # for patch in tqdm(patch_slices):
    for patch in patch_slices:
        v = greyscale_image[patch] * possibly_filled_pixels[patch]
        if v.max() > 0 and np.ptp(v) > 0.5:
            t = threshold_otsu(v)

            if maximize_parameter:
                res = minimize(_network_perimeter, x0=t, args=(v,), method='Nelder-Mead')
                t = res.x[0]

            network_patches.append(v > t)
            patch_thresholds.append(t)
        else:
            network_patches.append(np.zeros_like(v))
            patch_thresholds.append(0)

    network_img = np.zeros(greyscale_image.shape, dtype=np.float64)
    count_img = np.zeros_like(greyscale_image)
    for patch, network_patch, t in zip(patch_slices, network_patches, patch_thresholds):
        network_img[patch] += network_patch
        count_img[patch] += np.ones_like(network_patch)

    # Safe in-place division: zeros remain where count_img == 0
    np.divide(network_img, count_img, out=network_img, where=count_img != 0)

    return (network_img > 0.5).astype(np.uint8)

def binary_quality_index(binary_img: NDArray[np.uint8]) -> float:
    """
    Calculate the binary quality index for a binary image.

    The binary quality index is computed based on the perimeter of the largest
    connected component in the binary image, normalized by the total number of
    pixels.

    Parameters
    ----------
    binary_img : ndarray of uint8
        Input binary image array.

    Returns
    -------
    out : float
        The binary quality index value.
    """
    if np.any(binary_img):
        # SD = ShapeDescriptors(binary_img, ["euler_number"])
        # index = - SD.descriptors['euler_number']
        size, largest_cc = get_largest_connected_component(binary_img)
        index = np.square(perimeter(largest_cc)) / binary_img.sum()
        # index = (largest_cc.sum() * perimeter(largest_cc)) / binary_img.sum()
    else:
        index = 0.
    return index


def find_threshold_given_mask(greyscale: NDArray[np.uint8], mask: np.uint8, min_threshold: np.uint8=0) -> np.uint8:
    """
    Find the optimal threshold value for a greyscale image given a mask.

    This function performs a binary search to find the optimal threshold
    that maximizes the separation between two regions defined by the mask.
    The search is bounded by a minimum threshold value.

    Parameters
    ----------
    greyscale : ndarray of uint8
        The greyscale image array.
    mask : ndarray of uint8
        The binary mask array where positive values define region A and zero values define region B.
    min_threshold : uint8, optional
        The minimum threshold value for the search. Defaults to 0.

    Returns
    -------
    out : uint8
        The optimal threshold value found.

    Examples
    --------
    >>> greyscale = np.array([[255, 128, 54], [0, 64, 20]], dtype=np.uint8)
    >>> mask = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    >>> find_threshold_given_mask(greyscale, mask)
    54
    """
    region_a = greyscale[mask > 0]
    if len(region_a) == 0:
        return np.uint8(255)
    region_b = greyscale[mask == 0]
    if len(region_b) == 0:
        return min_threshold
    else:
        low = min_threshold
        high = 255
        best_thresh = low

        while 0 <= low <= high:
            mid = (low + high) // 2
            count_a, count_b = _get_counts_jit(mid, region_a, region_b)

            if count_a > count_b:
                # Try to find a lower threshold that still satisfies the condition
                best_thresh = mid
                high = mid - 1
            else:
                if count_a == 0 and count_b == 0:
                    best_thresh = greyscale.mean()
                    break
                # Need higher threshold
                low = mid + 1
    return best_thresh


@njit()
def _get_counts_jit(thresh: np.uint8, region_a: NDArray[np.uint8], region_b: NDArray[np.uint8]) -> Tuple[int, int]:
    """
    Get counts of values in two regions above a threshold using Just-In-Time compilation.

    Count the number of elements greater than `thresh` in both `region_a`
    and `region_b`, returning the counts as a tuple. This function utilizes
    Numba's JIT compilation for performance optimization.

    Parameters
    ----------
    thresh : uint8
        The threshold value to compare against.
    region_a : ndarray of uint8
        First region array containing values to be compared with `thresh`.
    region_b : ndarray of uint8
        Second region array containing values to be compared with `thresh`.

    Returns
    -------
    out : tuple of int, int
        A tuple containing the count of elements greater than `thresh` in
        `region_a` and `region_b`, respectively.

    Examples
    --------
    >>> import numpy as np
    >>> region_a = np.array([1, 250, 3], dtype=np.uint8)
    >>> region_b = np.array([4, 250, 6], dtype=np.uint8)
    >>> thresh = np.uint8(100)
    >>> _get_counts_jit(thresh, region_a, region_b)
    (1, 1)
    """
    count_a = 0
    count_b = 0
    for val in region_a:
        if val > thresh:
            count_a += 1
    for val in region_b:
        if val > thresh:
            count_b += 1
    return count_a, count_b


def extract_first_pc(bgr_image: np.ndarray, standardize: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Extract the first principal component from a BGR image.

    Parameters
    ----------
    bgr_image : numpy.ndarray
        A 3D or 2D array representing the BGR image. Expected shape is either
        (height, width, 3) or (3, height, width).
    standardize : bool, optional
        If True, standardizes the image pixel values by subtracting the mean and
        dividing by the standard deviation before computing the principal
        components. Default is True.

    Returns
    -------
    numpy.ndarray
        The first principal component image, reshaped to the original image height and width.
    float
        The explained variance ratio of the first principal component.
    numpy.ndarray
        The first principal component vector.

    Notes
    -----
    The principal component analysis (PCA) is performed using Singular Value Decomposition (SVD).
    Standardization helps in scenarios where the pixel values have different scales.
    Pixels with zero standard deviation are handled by setting their standardization
    denominator to 1.0 to avoid division by zero.

    Examples
    --------
    >>> bgr_image = np.random.rand(100, 200, 3)  # Example BGR image
    >>> first_pc_image, explained_variance_ratio, first_pc_vector = extract_first_pc(bgr_image)
    >>> print(first_pc_image.shape)
    (100, 200)
    >>> print(explained_variance_ratio)
    0.339
    >>> print(first_pc_vector.shape)
    (3,)
    """
    height, width, channels = bgr_image.shape
    pixels = bgr_image.reshape(-1, channels)  # Flatten to Nx3 matrix

    if standardize:
        mean = np.mean(pixels, axis=0)
        std = np.std(pixels, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        pixels_scaled = (pixels - mean) / std
    else:
        pixels_scaled = pixels

    # Perform SVD on standardized data to get principal components
    U, d, Vt = np.linalg.svd(pixels_scaled, full_matrices=False)

    # First PC is the projection of each pixel onto first right singular vector (Vt[0])
    first_pc_vector = Vt[0]  # Shape: (3,)
    eigen = d ** 2
    total_variance = np.sum(eigen)

    explained_variance_ratio = np.zeros_like(eigen)
    np.divide(eigen, total_variance, out=explained_variance_ratio, where=total_variance != 0)

    # Compute first principal component scores
    first_pc_scores = U[:, 0] * d[0]

    # Reshape to image shape and threshold negative values
    first_pc_image = first_pc_scores.reshape(height, width)
    # first_pc_thresholded = np.maximum(first_pc_image, 0)

    return first_pc_image, explained_variance_ratio[0], first_pc_vector


def convert_subtract_and_filter_video(video: NDArray, color_space_combination: dict, background: NDArray=None,
                                      background2: NDArray=None, lose_accuracy_to_save_memory:bool=False,
                                      filter_spec: dict=None) -> Tuple[NDArray, NDArray]:
    """
    Convert a video to grayscale, subtract the background, and apply filters.

    Parameters
    ----------
    video : NDArray
        The input video as a 4D NumPy array.
    color_space_combination : dict
        A dictionary containing the combinations of color space transformations.
    background : NDArray, optional
        The first background image for subtraction. If `None`, no subtraction is performed.
    background2 : NDArray, optional
        The second background image for subtraction. If `None`, no subtraction is performed.
    lose_accuracy_to_save_memory : bool
        Flag to reduce accuracy and save memory by using `uint8` instead of `float64`.
    filter_spec : dict
        A dictionary containing the specifications for filters to apply.

    Returns
    -------
    Tuple[NDArray, NDArray]
        A tuple containing:
        - `converted_video`: The converted grayscale video.
        - `converted_video2`: The second converted grayscale video if logical operation is not 'None'.

    Notes
    -----
        - The function reduces accuracy of the converted video when `lose_accuracy_to_save_memory` is set to True.
        - If `color_space_combination['logical']` is not 'None', a second converted video will be created.
        - This function uses the `generate_color_space_combination` and `apply_filter` functions internally.
    """

    converted_video2 = None
    if len(video.shape) == 3:
        converted_video = video
    else:
        if lose_accuracy_to_save_memory:
            array_type = np.uint8
        else:
            array_type = np.float64
        first_dict, second_dict, c_spaces = split_dict(color_space_combination)
        if 'PCA' in first_dict:
            greyscale_image, var_ratio, first_pc_vector = extract_first_pc(video[0])
            first_dict = Dict()
            first_dict['bgr'] = bracket_to_uint8_image_contrast(first_pc_vector)
            c_spaces = ['bgr']
        if 'PCA' in second_dict:
            greyscale_image, var_ratio, first_pc_vector = extract_first_pc(video[0])
            second_dict = Dict()
            second_dict['bgr'] = bracket_to_uint8_image_contrast(first_pc_vector)
            c_spaces = ['bgr']

        converted_video = np.zeros(video.shape[:3], dtype=array_type)
        if color_space_combination['logical'] != 'None':
            converted_video2 = converted_video.copy()
        for im_i in range(video.shape[0]):
            if im_i == 0 and background is not None:
                # when doing background subtraction, the first and the second image are equal
                image_i = video[1, ...]
            else:
                image_i = video[im_i, ...]
            results = generate_color_space_combination(image_i, c_spaces, first_dict, second_dict, background,
                                                       background2, lose_accuracy_to_save_memory)
            greyscale_image, greyscale_image2, all_c_spaces, first_pc_vector = results
            if filter_spec is not None and filter_spec['filter1_type'] != "":
                greyscale_image = apply_filter(greyscale_image, filter_spec['filter1_type'],
                                               filter_spec['filter1_param'],lose_accuracy_to_save_memory)
                if greyscale_image2 is not None and filter_spec['filter2_type'] != "":
                    greyscale_image2 = apply_filter(greyscale_image2,
                                                    filter_spec['filter2_type'], filter_spec['filter2_param'],
                                                    lose_accuracy_to_save_memory)
            converted_video[im_i, ...] = greyscale_image
            if color_space_combination['logical'] != 'None':
                converted_video2[im_i, ...] = greyscale_image2
    return converted_video, converted_video2
