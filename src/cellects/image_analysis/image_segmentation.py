#!/usr/bin/env python3
"""
This script contains functions to convert bgr images into grayscale and grayscale images into binary
"""
import threading
import logging
from fnmatch import translate

from tqdm import tqdm
from numba.typed import Dict as TDict
from numpy import min, max, all, floor, any, linspace, ceil, transpose, meshgrid, ones_like, zeros_like, ptp, logical_and, pi, square, mean, median, float32, histogram, cumsum, logical_not, float64, array, zeros, std, sum, uint8, round, isin, append, delete, argmax, diff, argsort, argwhere, logical_or, unique, nonzero
from cv2 import TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, kmeans, KMEANS_RANDOM_CENTERS, filter2D, cvtColor, COLOR_BGR2LAB, COLOR_BGR2HSV, COLOR_BGR2LUV, COLOR_BGR2HLS, COLOR_BGR2YUV, connectedComponents, connectedComponentsWithStats
from numba import njit
from cellects.utils.utilitarian import less_along_first_axis, greater_along_first_axis, translate_dict
from cellects.utils.formulas import bracket_to_uint8_image_contrast
from cellects.image_analysis.morphological_operations import get_largest_connected_component
from skimage.measure import perimeter
from scipy.optimize import minimize
from skimage.filters import frangi, sato, threshold_otsu


def get_color_spaces(bgr_image, space_names=""):
    """
    Create a typed dictonary containing the bgr image converted into:
    lab, hsv, luv, hls and yuv
    :param bgr_image: 3D matrix of a bgr image, the two first dims are coordinates, the last is color.
    :return: dict[str, float64]
    """
    if 'logical' in space_names:
        space_names.pop(nonzero(array(space_names, dtype=str) == 'logical')[0][0])
    c_spaces = TDict()
    c_spaces['bgr'] = bgr_image.astype(float64)
    if len(space_names) == 0:
        c_spaces['lab'] = cvtColor(bgr_image, COLOR_BGR2LAB).astype(float64)
        c_spaces['hsv'] = cvtColor(bgr_image, COLOR_BGR2HSV).astype(float64)
        c_spaces['luv'] = cvtColor(bgr_image, COLOR_BGR2LUV).astype(float64)
        c_spaces['hls'] = cvtColor(bgr_image, COLOR_BGR2HLS).astype(float64)
        c_spaces['yuv'] = cvtColor(bgr_image, COLOR_BGR2YUV).astype(float64)
    else:
        if isin('lab', space_names):
            c_spaces['lab'] = cvtColor(bgr_image, COLOR_BGR2LAB).astype(float64)
        if isin('hsv', space_names):
            c_spaces['hsv'] = cvtColor(bgr_image, COLOR_BGR2HSV).astype(float64)
        if isin('luv', space_names):
            c_spaces['luv'] = cvtColor(bgr_image, COLOR_BGR2LUV).astype(float64)
        if isin('hls', space_names):
            c_spaces['hls'] = cvtColor(bgr_image, COLOR_BGR2HLS).astype(float64)
        if isin('yuv', space_names):
            c_spaces['yuv'] = cvtColor(bgr_image, COLOR_BGR2YUV).astype(float64)
    return c_spaces


@njit()
def combine_color_spaces(c_space_dict, all_c_spaces, subtract_background=None):
    """
    Compute a linear combination of some channels of some color spaces.
    Subtract background if needed.
    Standardize values in the image so that they range between 0 and 255

    :param c_space_dict: The linear combination of channels to compute
    :param all_c_spaces: All converted versions of the image
    :type all_c_spaces: TDict[str: float64]
    :param subtract_background: array of the background to subtract
    :return: the grayscale image resulting from the linear combination of the selected channels
    """
    image = zeros((all_c_spaces['bgr'].shape[0], all_c_spaces['bgr'].shape[1]), dtype=float64)
    for space, channels in c_space_dict.items():
        image += c_space_dict[space][0] * all_c_spaces[space][:, :, 0] + c_space_dict[space][1] * \
                 all_c_spaces[space][:, :, 1] + c_space_dict[space][2] * all_c_spaces[space][:, :, 2]
    if subtract_background is not None:
        # add (resp. subtract) the most negative (resp. smallest) value to the whole matrix to get a min = 0
        image -= min(image)
        # Make analysable this image by bracketing its values between 0 and 255 and converting it to uint8
        max_im = max(image)
        if max_im != 0:
            image = 255 * (image / max(image))
        if image.sum() > subtract_background.sum():
            image -= subtract_background
        else:
            image = subtract_background - image
    # add (resp. subtract) the most negative (resp. smallest) value to the whole matrix to get a min = 0
    image -= min(image)
    # Make analysable this image by bracketing its values between 0 and 255 and converting it to uint8
    max_im = max(image)
    if max_im != 0:
        image = 255 * (image / max_im)
    return image
# c_space_dict=first_dict; all_c_spaces=self.all_c_spaces; subtract_background=background


def generate_color_space_combination(bgr_image, c_spaces, first_dict, second_dict={}, background=None, background2=None, convert_to_uint8=False):
    all_c_spaces = get_color_spaces(bgr_image, c_spaces)
    try:
        greyscale_image = combine_color_spaces(first_dict, all_c_spaces, background)
    except:
        first_dict = translate_dict(first_dict)
        greyscale_image = combine_color_spaces(first_dict, all_c_spaces, background)
    if convert_to_uint8:
        greyscale_image = bracket_to_uint8_image_contrast(greyscale_image)
    greyscale_image2 = None
    if len(second_dict) > 0:
        greyscale_image2 = combine_color_spaces(second_dict, all_c_spaces, background2)
        if convert_to_uint8:
            greyscale_image2 = bracket_to_uint8_image_contrast(greyscale_image2)
    return greyscale_image, greyscale_image2


def filter_mexican_hat(image):
    """
    Create a mexican hat filter.
    Other filters are:
    - GaussianBlur(image, (size, size), BORDER_DEFAULT)
    - medianBlur(image, ksize=size)
    - Sharpen an image = filter2D(image, -1, array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    - A blurring that conserve edges = bilateralFilter(image, d=size, sigmaColor=sigma, sigmaSpace=sigma)# 5, 75, 75 9, 150, 150
    - Extract edges = Laplacian(image, ddepth=depth, ksize=(size, size))
    :param image:
    :type image: uint8
    :return: the filtered image
    :rtype image: uint8
    """
    return filter2D(image, -1, array(
        [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]))


@njit()
def get_otsu_threshold(image):
    """
    Compute the otso threshold of an image. Function from Anastasia, see:
    https://github.com/spmallick/learnopencv/blob/master/otsu-method/otsu_implementation.py

    :param image:
    :return:
    """
    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = histogram(image, bins=bins_num)

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = cumsum(hist)
    weight2 = cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    return threshold


@njit()
def otsu_thresholding(image):
    """
    Segment a grayscale image into a binary image using otsu thresholding

    Contrary to cv2.threshold(image, 0, 1, cv2.THRESH_OTSU),
    This method does not require image to be uint8.
    Hence, does not require any rounding if image has been float64.
    Consequently, the binary image obtained contains less noise:
    sum((binary_image_from_rounded_image - binary_image) == 255) returns 0
    sum((binary_image - binary_image_from_rounded_image) == 255) returns 13171 (test)
    :param image: Image of any type and any dimension
    :return: a uint8 binary image in which 1 are less numerous than 0
    --> A usual assumption for segmentation, especially for the first image of
    the time lapse of a growing cell.
    :param image:
    :return:
    """
    threshold = get_otsu_threshold(image)
    binary_image = (image > threshold)
    binary_image2 = logical_not(binary_image)
    if binary_image.sum() < binary_image2.sum():
        return binary_image.astype(uint8)
    else:
        return binary_image2.astype(uint8)


@njit()
def segment_with_lum_value(converted_video, basic_bckgrnd_values, l_threshold, lighter_background):
    """
    Use an uint8 value as a threshold to segment the image: split it into two categories, 0 and 1
    :param converted_video: a 3D matrix with time, y_coord, x_coord
    :param basic_bckgrnd_values: a vector of typical background values of each frame (t)
    :param l_threshold: an average luminosity threshold
    :param lighter_background: True if the background of the image is lighter than the shape to detect
    :type lighter_background: bool
    :return: the resulting video of the segmentation, a vector of the luminosity threshold of each frame (t)
    """
    # segmentation = None
    if lighter_background:
        l_threshold_over_time = l_threshold - (basic_bckgrnd_values[-1] - basic_bckgrnd_values)
        if all(logical_and(0 <= l_threshold_over_time, l_threshold_over_time <= 255)):
            segmentation = less_along_first_axis(converted_video, l_threshold_over_time)
        else:
            segmentation = zeros_like(converted_video)
            if l_threshold > 255:
                l_threshold = 255
            segmentation += converted_video > l_threshold
    else:
        l_threshold_over_time = l_threshold - (basic_bckgrnd_values[-1] - basic_bckgrnd_values)
        if all(logical_and(0 <= l_threshold_over_time, l_threshold_over_time <= 255)):
            segmentation = greater_along_first_axis(converted_video, l_threshold_over_time)
        else:
            segmentation = zeros_like(converted_video)
            if l_threshold > 255:
                l_threshold = 255
            segmentation += converted_video > l_threshold
            # segmentation = (converted_video > l_threshold).astype(uint8)
    return segmentation, l_threshold_over_time


def _network_perimeter(threshold, img):
    binary_img = img > threshold
    return -perimeter(binary_img)


def rolling_window_segmentation(greyscale_image, possibly_filled_pixels, patch_size=(80, 80)):
    patch_centers = [
        floor(linspace(
            p // 2, s - p // 2, int(ceil(s / (p // 2))) - 1
        )).astype(int)
        for s, p in zip(greyscale_image.shape, patch_size)
    ]
    patch_centers = transpose(meshgrid(*patch_centers), (1, 2, 0)).reshape((-1, 2))

    patch_slices = [
        tuple(slice(c - p // 2, c + p // 2, 1)
              for c, p in zip(p_c, patch_size)) for p_c in patch_centers
    ]
    maximize_parameter = False

    network_patches = []
    patch_thresholds = []
    for patch in tqdm(patch_slices):
        v = greyscale_image[patch] * possibly_filled_pixels[patch]
        if v.max() > 0 and ptp(v) > 0.5:
            t = threshold_otsu(v)

            if maximize_parameter:
                res = minimize(_network_perimeter, x0=t, args=(v,), method='Nelder-Mead')
                t = res.x[0]

            network_patches.append(v > t)
            patch_thresholds.append(t)
        else:
            network_patches.append(zeros_like(v))
            patch_thresholds.append(0)

    network_img = zeros_like(greyscale_image)
    count_img = zeros_like(greyscale_image)
    for patch, network_patch, t in zip(patch_slices, network_patches, patch_thresholds):
        network_img[patch] += network_patch
        count_img[patch] += ones_like(network_patch)
    network_img /= count_img
    return network_img

def binary_quality_index(binary_img):
    from cellects.image_analysis.shape_descriptors import ShapeDescriptors

    if any(binary_img):
        # SD = ShapeDescriptors(binary_img, ["euler_number"])
        # index = - SD.descriptors['euler_number']
        nb, largest_cc = get_largest_connected_component(binary_img)
        index = square(perimeter(largest_cc)) / binary_img.sum()
        # index = (largest_cc.sum() * perimeter(largest_cc)) / binary_img.sum()
    else:
        index = 0.
    return index