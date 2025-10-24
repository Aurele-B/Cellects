#!/usr/bin/env python3
"""
This script contains functions to convert bgr images into grayscale and grayscale images into binary
"""
import numpy as np
import cv2
from tqdm import tqdm
from numba.typed import Dict as TDict
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
        space_names.pop(np.nonzero(np.array(space_names, dtype=str) == 'logical')[0][0])
    c_spaces = TDict()
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
def combine_color_spaces(c_space_dict, all_c_spaces, subtract_background=None):
    """
    Compute a linear combination of some channels of some color spaces.
    Subtract background if needed.
    Standardize values in the image so that they range between 0 and 255

    :param c_space_dict: The linear combination of channels to compute
    :param all_c_spaces: All converted versions of the image
    :type all_c_spaces: TDict[str: =np.float64]
    :param subtract_background: array of the background to subtract
    :return: the grayscale image resulting from the linear combination of the selected channels
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
    # Make analysable this image by bracketing its values between 0 and 255 and converting it to uint8
    max_im = np.max(image)
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
    - Sharpen an image = cv2.filter2D(image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    - A blurring that conserve edges = bilateralFilter(image, d=size, sigmaColor=sigma, sigmaSpace=sigma)# 5, 75, 75 9, 150, 150
    - Extract edges = Laplacian(image, ddepth=depth, ksize=(size, size))
    :param image:
    :type image: uint8
    :return: the filtered image
    :rtype image: uint8
    """
    return cv2.filter2D(image, -1, np.array(
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
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    return threshold


@njit()
def otsu_thresholding(image):
    """
    Segment a grayscale image into a binary image using otsu thresholding

    Contrary to cv2.threshold(image, 0, 1, cv2.THRESH_OTSU),
    This method does not require image to be uint8.
    Hence, does not require any rounding if image has been =np.float64.
    Consequently, the binary image obtained contains less noise:
    :param image: Image of any type and any dimension
    :return: a uint8 binary image in which 1 are less numerous than 0
    --> A usual assumption for segmentation, especially for the first image of
    the time lapse of a growing cell.
    :param image:
    :return:
    """
    threshold = get_otsu_threshold(image)
    binary_image = (image > threshold)
    binary_image2 = np.logical_not(binary_image)
    if binary_image.sum() < binary_image2.sum():
        return binary_image.astype(np.uint8)
    else:
        return binary_image2.astype(np.uint8)


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


def _network_perimeter(threshold, img):
    binary_img = img > threshold
    return -perimeter(binary_img)


def rolling_window_segmentation(greyscale_image, possibly_filled_pixels, patch_size=(80, 80)):
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
    for patch in tqdm(patch_slices):
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

    network_img = np.zeros_like(greyscale_image)
    count_img = np.zeros_like(greyscale_image)
    for patch, network_patch, t in zip(patch_slices, network_patches, patch_thresholds):
        network_img[patch] += network_patch
        count_img[patch] += np.ones_like(network_patch)
    network_img /= count_img
    return network_img

def binary_quality_index(binary_img):
    from cellects.image_analysis.shape_descriptors import ShapeDescriptors

    if np.any(binary_img):
        # SD = ShapeDescriptors(binary_img, ["euler_number"])
        # index = - SD.descriptors['euler_number']
        nb, largest_cc = get_largest_connected_component(binary_img)
        index = np.square(perimeter(largest_cc)) / binary_img.sum()
        # index = (largest_cc.sum() * perimeter(largest_cc)) / binary_img.sum()
    else:
        index = 0.
    return index