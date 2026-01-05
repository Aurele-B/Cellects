#!/usr/bin/env python3
"""
Module providing tools for single-image color space analysis and segmentation.

The OneImageAnalysis class offers comprehensive image processing capabilities including
color space conversion (RGB, HSV, LAB, LUV, HLS, YUV), filtering (Gaussian, median, bilateral),
segmentation (Otsu thresholding, k-means clustering), and shape-based validation. It supports
multi-step optimization of color channel combinations to maximize contrast between organisms
and background through automated selection workflows involving logical operations on segmented regions.

Classes
OneImageAnalysis : Analyze images using multiple color spaces for optimal segmentation

Notes
Uses QThread for background operations during combination processing.
"""

import logging
import os
from copy import deepcopy
import numpy as np
import cv2  # named opencv-python
import multiprocessing.pool as mp
from numba.typed import List
from numba.typed import Dict
from numpy.typing import NDArray
from typing import Tuple
import pandas as pd
from scipy.stats import rankdata
from skimage.measure import perimeter
from cellects.image_analysis.morphological_operations import cross_33, create_ellipse, spot_size_coefficients
from cellects.image_analysis.image_segmentation import generate_color_space_combination, get_color_spaces, filter_dict, extract_first_pc, combine_color_spaces, apply_filter, otsu_thresholding, get_otsu_threshold, kmeans, windowed_thresholding
from cellects.image_analysis.one_image_analysis_threads import ProcessImage
from cellects.image_analysis.network_functions import NetworkDetection
from cellects.utils.formulas import bracket_to_uint8_image_contrast
from cellects.utils.utilitarian import split_dict, translate_dict

def init_params():
    params = {}
    # User set variables:
    params['is_first_image']: bool = False
    params['several_blob_per_arena']: bool = True
    params['blob_nb']: int = None
    params['blob_shape']: str = None
    params['blob_size']: int = None
    params['kmeans_clust_nb']: int = None
    params['arenas_mask']: NDArray = None
    params['ref_image']: NDArray = None
    params['bio_mask']: Tuple = None
    params['back_mask']: Tuple = None
    params['filter_spec']: dict = {'filter1_type': "", 'filter1_param': [.5, 1.], 'filter2_type': "", 'filter2_param': [.5, 1.]}
    # Computed before OneImageAnalysis usage:

    # Computed in OneImageAnalysis usage:
    params['con_comp_extent']: list = None
    params['max_blob_size']: int = None
    params['total_surface_area']: int = None
    params['out_of_arenas_mask']: NDArray = None
    params['subtract_background']: NDArray = None
    params['are_zigzag']: str = None
    return params

def make_one_dict_per_channel():
    colorspace_list = ["bgr", "lab", "hsv", "luv", "hls", "yuv"]
    one_dict_per_channel = List()
    channels = np.array((1, 1, 1), dtype=np.int8)
    csc_dict = Dict()
    csc_dict["bgr"] = channels
    one_dict_per_channel.append(csc_dict)
    for c_space in colorspace_list:
        for j in np.arange(3):
            channels = np.array((0, 0, 0), dtype=np.int8)
            channels[j] = 1
            csc_dict = Dict()
            csc_dict[c_space] = channels
            one_dict_per_channel.append(csc_dict)
    return one_dict_per_channel

one_dict_per_channel = make_one_dict_per_channel()

class OneImageAnalysis:
    """
        This class takes a 3D matrix (2 space and 1 color [BGR] dimensions),
        Its methods allow image
        - conversion to any bgr/hsv/lab channels
        - croping
        - rotating
        - filtering using some of the mainly used techniques:
            - Gaussian, Median, Bilateral, Laplacian, Mexican hat
        - segmenting using thresholds or kmeans
        - shape selection according to horizontal size or shape ('circle' vs 'quadrilateral')

        ps: A viewing method displays the image before and after the most advanced modification made in instance
    """
    def __init__(self, image, shape_number: int=1):
        self.image = image
        if len(self.image.shape) == 2:
            self.already_greyscale = True
        else:
            self.already_greyscale = False
        self.image2 = None
        self.binary_image2 = None
        self.drift_correction_already_adjusted: bool = False
        # Create empty variables to fill in the following functions
        self.binary_image = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.previous_binary_image = None
        self.validated_shapes = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.centroids = 0
        self.shape_number = shape_number
        self.concomp_stats = 0
        self.y_boundaries = None
        self.x_boundaries = None
        self.crop_coord = None
        self.cropped: bool = False
        self.subtract_background = None
        self.subtract_background2 = None
        self.im_combinations = None
        self.bgr = image
        self.colorspace_list = List(("bgr", "lab", "hsv", "luv", "hls", "yuv"))
        self.spot_shapes = None
        self.all_c_spaces = Dict()
        self.hsv = None
        self.hls = None
        self.lab = None
        self.luv = None
        self.yuv = None
        self.greyscale = None
        self.greyscale2 = None
        self.first_pc_vector = None
        self.drift_mask_coord = None
        self.saved_csc_nb = 0

    def convert_and_segment(self, c_space_dict: dict, color_number=2, bio_mask: NDArray[np.uint8]=None,
                            back_mask: NDArray[np.uint8]=None, subtract_background: NDArray=None,
                            subtract_background2: NDArray=None, rolling_window_segmentation: dict=None,
                            lighter_background: bool=None,
                            allowed_window: NDArray=None, filter_spec: dict=None):
        """
        Convert an image to grayscale and segment it based on specified parameters.

        This method converts the given color space dictionary into grayscale
        images, combines them with existing color spaces and performs segmentation.
        It has special handling for images that are already in grayscale.

        **Args:**

        - `c_space_dict` (dict): Dictionary containing color spaces.
        - `color_number` (int, optional): Number of colors to use in segmentation. Defaults to 2.
        - `bio_mask` (NDArray[np.uint8], optional): Biomask for segmentation. Defaults to None.
        - `back_mask` (NDArray[np.uint8], optional): Backmask for segmentation. Defaults to None.
        - `subtract_background` (NDArray, optional): Background to subtract. Defaults to None.
        - `subtract_background2` (NDArray, optional): Second background to subtract. Defaults to None.
        - rolling_window_segmentation (dict, optional): Flag for grid segmentation. Defaults to None.
        - `lighter_background` (bool, optional): Flag for lighter background. Defaults to None.
        - `mask` (NDArray, optional): Additional mask for segmentation. Defaults to None.
        - `filter_spec` (dict, optional): Filter specifications. Defaults to None.

        **Attributes:**

        - `self.already_greyscale` (bool): Indicates whether the image is already greyscale.
        - `self.all_c_spaces` (list): List of color spaces.

        """
        if not self.already_greyscale:
            first_dict, second_dict, c_spaces = split_dict(c_space_dict)
            self.image, self.image2, all_c_spaces, self.first_pc_vector = generate_color_space_combination(self.bgr, c_spaces, first_dict, second_dict, subtract_background, subtract_background2)
            if len(all_c_spaces) > len(self.all_c_spaces):
                self.all_c_spaces = all_c_spaces

        self.segmentation(logical=c_space_dict['logical'], color_number=color_number, bio_mask=bio_mask,
                          back_mask=back_mask, rolling_window_segmentation=rolling_window_segmentation,
                          lighter_background=lighter_background, allowed_window=allowed_window, filter_spec=filter_spec)


    def segmentation(self, logical: str='None', color_number: int=2, bio_mask: NDArray[np.uint8]=None,
                     back_mask: NDArray[np.uint8]=None, bio_label=None, bio_label2=None,
                     rolling_window_segmentation: dict=None, lighter_background: bool=None, allowed_window: Tuple=None,
                     filter_spec: dict=None):
        """
        Implement segmentation on the image using various methods and parameters.

        Args:
            logical (str): Logical operation to perform between two binary images.
                           Options are 'Or', 'And', 'Xor'. Default is 'None'.
            color_number (int): Number of colors to use in segmentation. Must be greater than 2
                                for kmeans clustering. Default is 2.
            bio_mask (NDArray[np.uint8]): Binary mask for biological areas. Default is None.
            back_mask (NDArray[np.uint8]): Binary mask for background areas. Default is None.
            bio_label (Any): Label for biological features. Default is None.
            bio_label2 (Any): Secondary label for biological features. Default is None.
            rolling_window_segmentation (dict): Whether to perform grid segmentation. Default is None.
            lighter_background (bool): Indicates if the background is lighter than objects.
                                       Default is None.
            allowed_window (Tuple): Mask to apply during segmentation. Default is None.
            filter_spec (dict): Dictionary of filters to apply on the image before segmentation.

        """
        # 1. Check valid pixels for segmentation (e.g. when there is a drift correction)
        if allowed_window is None:
            min_y, max_y, min_x, max_x = 0, self.image.shape[0] + 1, 0, self.image.shape[1] + 1
        else:
            min_y, max_y, min_x, max_x = allowed_window
        greyscale = self.image[min_y:max_y, min_x:max_x].copy()
        # 2. Apply filter on the greyscale images
        if filter_spec is not None and filter_spec["filter1_type"] != "":
            greyscale = apply_filter(greyscale, filter_spec["filter1_type"], filter_spec["filter1_param"])

        greyscale2 = None
        if logical != 'None':
            greyscale2 = self.image2[min_y:max_y, min_x:max_x].copy()
            if filter_spec is not None and filter_spec["filter2_type"] != "":
                greyscale2 = apply_filter(greyscale2, filter_spec["filter2_type"], filter_spec["filter2_param"])

        # 3. Do one of the three segmentation algorithms: kmeans, otsu, windowed
        if color_number > 2:
            binary_image, binary_image2, self.bio_label, self.bio_label2  = kmeans(greyscale, greyscale2, color_number, bio_mask, back_mask, logical, bio_label, bio_label2)
        elif rolling_window_segmentation is not None and rolling_window_segmentation['do']:
            binary_image = windowed_thresholding(greyscale, lighter_background, rolling_window_segmentation['side_len'],
            rolling_window_segmentation['step'], rolling_window_segmentation['min_int_var'])
        else:
            binary_image = otsu_thresholding(greyscale)
        if logical != 'None' and color_number == 2:
            if rolling_window_segmentation is not None and rolling_window_segmentation['do']:
                binary_image2 = windowed_thresholding(greyscale2, lighter_background, rolling_window_segmentation['side_len'],
                rolling_window_segmentation['step'], rolling_window_segmentation['min_int_var'])
            else:
                binary_image2 = otsu_thresholding(greyscale2)

        # 4. Use previous_binary_image to make sure that the specimens are labelled with ones and the background zeros
        if self.previous_binary_image is not None:
            previous_binary_image = self.previous_binary_image[min_y:max_y, min_x:max_x]
            if not (binary_image * previous_binary_image).any() or (binary_image[0, :].all() and binary_image[-1, :].all() and binary_image[:, 0].all() and binary_image[:, -1].all()):
                # if (binary_image * (1 - previous_binary_image)).sum() > (binary_image * previous_binary_image).sum() + perimeter(binary_image):
                # Ones of the binary image have more in common with the background than with the specimen
                binary_image = 1 - binary_image
            if logical != 'None':
                if (binary_image2 * (1 - previous_binary_image)).sum() > (binary_image2 * previous_binary_image).sum():
                    binary_image2 = 1 - binary_image2

        # 5. Give back the image their original size and combine binary images (optional)
        self.binary_image = np.zeros(self.image.shape, dtype=np.uint8)
        self.binary_image[min_y:max_y, min_x:max_x] = binary_image
        self.greyscale = np.zeros(self.image.shape, dtype=np.uint8)
        self.greyscale[min_y:max_y, min_x:max_x] = greyscale
        if logical != 'None':
            self.binary_image2 = np.zeros(self.image.shape, dtype=np.uint8)
            self.binary_image2[min_y:max_y, min_x:max_x] = binary_image2
            self.greyscale2 = np.zeros(self.image.shape, dtype=np.uint8)
            self.greyscale2[min_y:max_y, min_x:max_x] = greyscale2
        if logical != 'None':
            if logical == 'Or':
                self.binary_image = np.logical_or(self.binary_image, self.binary_image2)
            elif logical == 'And':
                self.binary_image = np.logical_and(self.binary_image, self.binary_image2)
            elif logical == 'Xor':
                self.binary_image = np.logical_xor(self.binary_image, self.binary_image2)
            self.binary_image = self.binary_image.astype(np.uint8)

    def _get_all_color_spaces(self):
        """Generate and store all supported color spaces for the image."""
        if len(self.all_c_spaces) < 6 and not self.already_greyscale:
            self.all_c_spaces = get_color_spaces(self.bgr)

    def generate_subtract_background(self, c_space_dict: dict, drift_corrected: bool=False):
        """
        Generate a background-subtracted image using specified color space dictionary.

        This method first checks if color spaces have already been generated or
        if the image is greyscale. If not, it generates color spaces from the BGR
        image. It then converts and segments the image using the provided color space
        dictionary without grid segmentation. A disk-shaped structuring element is
        created and used to perform a morphological opening operation on the image,
        resulting in a background-subtracted version. If there is a second image
        (see Also: image2), the same operation is performed on it.

        Args:
            c_space_dict (dict): Dictionary containing color space specifications
                for the segmentation process.

        Attributes:
            disk_size: Radius of the disk-shaped structuring element
                used for morphological operations, calculated based on image dimensions.
            subtract_background: Background-subtracted version of `image` obtained
                after morphological operations with the disk-shaped structuring element.
            subtract_background2: Background-subtracted version of `image2` obtained
                after morphological operations with the disk-shaped structuring element,
                if `image2` is present."""
        logging.info("Generate background using the generate_subtract_background method of OneImageAnalysis class")
        self._get_all_color_spaces()
        if drift_corrected:
            # self.adjust_to_drift_correction(c_space_dict['logical'])
            self.check_if_image_border_attest_drift_correction()
        self.convert_and_segment(c_space_dict, rolling_window_segmentation=None, allowed_window=self.drift_mask_coord)
        disk_size = int(np.floor(np.sqrt(np.min(self.bgr.shape[:2])) / 2))
        disk = create_ellipse(disk_size, disk_size, min_size=3).astype(np.uint8)
        self.subtract_background = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, disk)
        if self.image2 is not None:
            self.subtract_background2 = cv2.morphologyEx(self.image2, cv2.MORPH_OPEN, disk)

    def check_if_image_border_attest_drift_correction(self) -> bool:
        """
        Check if the given binary image requires border attenuation and drift correction.

        In order to determine the need for border attenuation or drift correction, this function
        evaluates the borders of a binary image. If any two opposite borders are fully black,
        it assumes that there is an issue requiring correction.

        Returns:
            bool: True if border attenuation or drift correction is required, False otherwise.

        """
        t = np.all(self.binary_image[0, :])
        b = np.all(self.binary_image[-1, :])
        l = np.all(self.binary_image[:, 0])
        r = np.all(self.binary_image[:, -1])
        self.drift_mask_coord = None
        if (t and b) or (t and r) or (t and l) or (t and r) or (b and l) or (b and r) or (l and r):
            cc_nb, shapes = cv2.connectedComponents(self.binary_image)
            if cc_nb > 1:
                if cc_nb == 2:
                    drift_mask_coord = np.nonzero(1 - self.binary_image)
                else:
                    back = np.unique(np.concatenate((shapes[0, :], shapes[-1, :],  shapes[:, 0], shapes[:, -1]), axis=0))
                    drift_mask_coord = np.nonzero(np.logical_or(1 - self.binary_image, 1 - np.isin(shapes, back[back != 0])))
                drift_mask_coord = (np.min(drift_mask_coord[0]), np.max(drift_mask_coord[0]) + 1,
                                    np.min(drift_mask_coord[1]), np.max(drift_mask_coord[1]) + 1)
                self.drift_mask_coord = drift_mask_coord
                return True
            else:
                return False
        else:
            return False

    def adjust_to_drift_correction(self, logical: str):
        """
        Adjust the image and binary image to correct for drift.

        This method applies a drift correction by dilating the binary image, calculating
        the mean value of the drifted region and applying it back to the image. After this,
        it applies Otsu's thresholding method to determine a new binary image and adjusts
        the second image if present. The logical operation specified is then applied to the
        binary images.

        Args:
            logical (str): Logical operation ('Or', 'And', 'Xor') to apply to the binary
                images."""
        if not self.drift_correction_already_adjusted:
            self.drift_correction_already_adjusted = True

            mask = cv2.dilate(self.binary_image, kernel=cross_33)
            mask -= self.binary_image
            mask = np.nonzero(mask)
            drift_correction = np.mean(self.image[mask[0], mask[1]])
            self.image[np.nonzero(self.binary_image)] = drift_correction
            threshold = get_otsu_threshold(self.image)
            binary = (self.image > threshold)
            self.binary_image = binary.astype(np.uint8)

            if self.image2 is not None:
                drift_correction2 = np.mean(self.image2[mask[0], mask[1]])
                self.image2[np.nonzero(self.binary_image)] = drift_correction2
                threshold = get_otsu_threshold(self.image2)
                binary1 = (self.image2 > threshold)
                binary2 = np.logical_not(binary1)
                if binary1.sum() < binary2.sum():
                    binary = binary1
                else:
                    binary = binary2
                while np.any(binary * self.binary_image2) and threshold > 1:
                    threshold -= 1
                    binary1 = (self.image2 > threshold)
                    binary2 = np.logical_not(binary1)
                    if binary1.sum() < binary2.sum():
                        binary = binary1
                    else:
                        binary = binary2
                self.binary_image2 = binary.astype(np.uint8)
                if logical == 'Or':
                    self.binary_image = np.logical_or(self.binary_image, self.binary_image2)
                elif logical == 'And':
                    self.binary_image = np.logical_and(self.binary_image, self.binary_image2)
                elif logical == 'Xor':
                    self.binary_image = np.logical_xor(self.binary_image, self.binary_image2)
                self.binary_image = self.binary_image.astype(np.uint8)

    def init_combinations_lists(self):
        self.im_combinations = []
        self.saved_images_list = List()
        self.converted_images_list = List()
        self.saved_color_space_list = list()
        self.saved_csc_nb = 0

    def find_color_space_combinations(self, params: dict=None, only_bgr: bool = False):
        logging.info(f"Start automatic finding of color space combinations...")
        self.init_combinations_lists()
        if self.image.any():
            # 1. Set all params
            if params is None:
                params = init_params()
            if params['arenas_mask'] is not None:
                params['out_of_arenas_mask'] = 1 - params['arenas_mask']
            if params['ref_image'] is not None:
                params['ref_image'] = cv2.dilate(params['ref_image'], cross_33)
            if params['several_blob_per_arena']:
                params['con_comp_extent'] = [1, self.binary_image.size // 50]
            else:
                params['con_comp_extent'] = [params['blob_nb'], np.max((params['blob_nb'], self.binary_image.size // 100))]
            im_size = self.image.shape[0] * self.image.shape[1]

            if not params['several_blob_per_arena'] and params['blob_nb'] is not None and params['blob_nb'] > 1 and params['are_zigzag'] is not None:
                if params['are_zigzag'] == "columns":
                    inter_dist = np.mean(np.diff(np.nonzero(self.y_boundaries)))
                elif params['are_zigzag'] == "rows":
                    inter_dist = np.mean(np.diff(np.nonzero(self.x_boundaries)))
                else:
                    dist1 = np.mean(np.diff(np.nonzero(self.y_boundaries)))
                    dist2 = np.mean(np.diff(np.nonzero(self.x_boundaries)))
                    inter_dist = np.max(dist1, dist2)
                if params['blob_shape'] == "rectangle":
                    params['max_blob_size'] = np.square(2 * inter_dist)
                else:
                    params['max_blob_size'] = np.pi * np.square(inter_dist)
                params['total_surface_area'] = params['max_blob_size'] * self.sample_number
            else:
                params['max_blob_size'] = .9 * im_size
                params['total_surface_area'] = .99 * im_size

            # 2. Get color_space_dictionaries
            if only_bgr:
                if not 'bgr' in self.all_c_spaces:
                    self.all_c_spaces['bgr'] = self.bgr
            else:
                self._get_all_color_spaces()

            # 3. Init combination_features table
            unaltered_blob_nb_idx, blob_number_idx, blob_shape_idx, blob_size_idx, total_area_idx, width_std_idx, height_std_idx, area_std_idx, out_of_arenas_idx, in_arena_idx, common_with_ref_idx, bio_sum_idx, back_sum_idx, score_idx = np.arange(3, 17)
            self.factors = ['unaltered_blob_nb', 'blob_nb', 'total_area', 'width_std', 'height_std', 'area_std', 'out_of_arenas', 'in_arenas', 'common_with_ref', 'bio_sum', 'back_sum', 'score']
            self.combination_features = pd.DataFrame(np.zeros((100, len(self.factors)), dtype=np.float64), columns=self.factors)

            # 4. Test every channel separately
            process = 'one'
            for csc_dict in one_dict_per_channel:
                ProcessImage([self, params, process, csc_dict])
            # If the blob number is known, try applying filters to improve detection
            if params['blob_nb'] is not None and (params['filter_spec'] is None or params['filter_spec']['filter1_type'] == ''):
                if not (self.combination_features['blob_nb'].iloc[:self.saved_csc_nb] == params['blob_nb']).any():
                    tested_filters = ['Gaussian', 'Median', 'Mexican hat', 'Laplace', '']
                    for tested_filter in tested_filters:
                        self.init_combinations_lists()
                        params['filter_spec'] = {'filter1_type': tested_filter, 'filter1_param': [.5, 1.], 'filter2_type': "", 'filter2_param': [.5, 1.]}
                        if 'Param1' in filter_dict[tested_filter]:
                            params['filter_spec']['filter1_param'] = [filter_dict[tested_filter]['Param1']['Default']]
                            if 'Param2' in filter_dict[tested_filter]:
                                params['filter_spec']['filter1_param'].append(filter_dict[tested_filter]['Param2']['Default'])
                        for csc_dict in one_dict_per_channel:
                            ProcessImage([self, params, process, csc_dict])
                        if (self.combination_features['blob_nb'].iloc[:self.saved_csc_nb] == params['blob_nb']).any():
                            break

            self.score_combination_features()
            # 5. Try adding each valid channel with one another
            # 5.1. Generate an index vector containing, for each color space, the channel maximizing the score
            possibilities = []
            self.all_combined = Dict()
            different_color_spaces = np.unique(self.saved_color_space_list)
            for color_space in different_color_spaces:
                indices = np.nonzero(np.isin(self.saved_color_space_list, color_space))[0]
                csc_idx = indices[0] + np.argmax(self.combination_features.loc[indices, 'score'])
                possibilities.append(csc_idx)
                for k, v in self.saved_color_space_list[csc_idx].items():
                    self.all_combined[k] = v

            # 5.2. Try combining each selected channel with every other in all possible order
            params['possibilities'] = possibilities
            pool = mp.ThreadPool(processes=os.cpu_count() - 1)
            process = 'add'
            list_args = [[self, params, process, i] for i in possibilities]
            for process_i in pool.imap_unordered(ProcessImage, list_args):
                pass

            # 6. Take a combination of all selected channels and try to remove each color space one by one
            ProcessImage([self, params, 'subtract', 0])

            # 7. Add PCA:
            ProcessImage([self, params, 'PCA', None])

            # 8. Make logical operations between pairs of segmentation result
            coverage = np.argsort(self.combination_features['total_area'].iloc[:self.saved_csc_nb])

            # 8.1 Try a logical And between the most covered images
            most1, most2 = coverage.values[-1], coverage.values[-2]
            operation = {0: most1, 1: most2, 'logical': 'And'}
            ProcessImage([self, params, 'logical', operation])

            # 8.2 Try a logical Or between the least covered images
            least1, least2 = coverage.values[0], coverage.values[1]
            operation = {0: least1, 1: least2, 'logical': 'Or'}
            ProcessImage([self, params, 'logical', operation])


            # 8.3 Try a logical And between the best bio_mask images
            if params['bio_mask'] is not None:
                bio_sort = np.argsort(self.combination_features['bio_sum'].iloc[:self.saved_csc_nb])
                bio1, bio2 = bio_sort.values[-1], bio_sort.values[-2]
                operation = {0: bio1, 1: bio2, 'logical': 'And'}
                ProcessImage([self, params, 'logical', operation])

            # 8.4 Try a logical And between the best back_mask images
            if params['back_mask'] is not None:
                back_sort = np.argsort(self.combination_features['back_sum'].iloc[:self.saved_csc_nb])
                back1, back2 = back_sort.values[-1], back_sort.values[-2]
                operation = {0: back1, 1: back2, 'logical': 'And'}
                ProcessImage([self, params, 'logical', operation])

            # 8.5 Try a logical Or between the best bio_mask and the best back_mask images
            if params['bio_mask'] is not None and params['back_mask'] is not None:
                operation = {0: bio1, 1: back1, 'logical': 'Or'}
                ProcessImage([self, params, 'logical', operation])

            # 9. Order all saved features
            self.combination_features = self.combination_features.iloc[:self.saved_csc_nb, :]
            self.score_combination_features()
            if params['is_first_image'] and params['blob_nb'] is not None:
                distances = np.abs(self.combination_features['blob_nb'] - params['blob_nb'])
                cc_efficiency_order = np.argsort(distances)
            else:
                cc_efficiency_order = np.argsort(self.combination_features['score'])
                cc_efficiency_order = cc_efficiency_order.max() - cc_efficiency_order

            # 7. Save and return a dictionary containing the selected color space combinations
            # and their corresponding binary images
            self.im_combinations = []
            for saved_csc in cc_efficiency_order:
                if len(self.saved_color_space_list[saved_csc]) > 0:
                    self.im_combinations.append({})
                    self.im_combinations[len(self.im_combinations) - 1]["csc"] = {}
                    self.im_combinations[len(self.im_combinations) - 1]["csc"]['logical'] = 'None'
                    for k, v in self.saved_color_space_list[saved_csc].items():
                        self.im_combinations[len(self.im_combinations) - 1]["csc"][k] = v
                    self.im_combinations[len(self.im_combinations) - 1]["binary_image"] = self.saved_images_list[saved_csc]
                    self.im_combinations[len(self.im_combinations) - 1]["converted_image"] = np.round(self.converted_images_list[
                        saved_csc]).astype(np.uint8)
                    self.im_combinations[len(self.im_combinations) - 1]["shape_number"] = int(self.combination_features['blob_nb'].iloc[saved_csc])
                    self.im_combinations[len(self.im_combinations) - 1]['filter_spec']= params['filter_spec']
            self.saved_color_space_list = []
            del self.saved_images_list
            del self.converted_images_list
            del self.all_combined

    def save_combination_features(self, process_i: object):
        """
        Saves the combination features of a given processed image.

        Args:
            process_i (object): The processed image object containing various attributes
                such as validated_shapes, image, csc_dict, unaltered_concomp_nb,
                shape_number, total_area, stats, bio_mask, and back_mask.

            Attributes:
                processed image object
                    validated_shapes (array-like): The validated shapes of the processed image.
                    image (array-like): The image data.
                    csc_dict (dict): Color space conversion dictionary
        """
        if process_i.validated_shapes.any():
            saved_csc_nb = self.saved_csc_nb
            self.saved_csc_nb += 1
            self.saved_images_list.append(process_i.validated_shapes)
            self.converted_images_list.append(bracket_to_uint8_image_contrast(process_i.greyscale))
            self.saved_color_space_list.append(process_i.csc_dict)
            self.combination_features.iloc[saved_csc_nb, :] = process_i.fact

    def score_combination_features(self):
        for to_minimize in ['unaltered_blob_nb', 'blob_nb', 'area_std', 'width_std', 'height_std', 'back_sum', 'out_of_arenas']:
            values = rankdata(self.combination_features[to_minimize], method='dense')
            self.combination_features['score'] += values.max() - values
        for to_maximize in ['bio_sum', 'in_arenas', 'common_with_ref']:
            values = rankdata(self.combination_features[to_maximize], method='dense') - 1
            self.combination_features['score'] += values

    def update_current_images(self, current_combination_id: int):
        """
        Update the current images based on a given combination ID.

        This method updates two attributes of the instance: `image` and
        `validated_shapes`. The `image` attribute is set to the value of the key
        "converted_image" from a dictionary in `im_combinations` which is
        indexed by the provided `current_combination_id`. Similarly, the
        `validated_shapes` attribute is set to the value of the key "binary_image"
        from the same dictionary.

        Args:
            current_combination_id (int): The ID of the combination whose
                images should be set as the current ones.

        """
        self.image = self.im_combinations[current_combination_id]["converted_image"]
        self.validated_shapes = self.im_combinations[current_combination_id]["binary_image"]

    def network_detection(self, arenas_mask: NDArray=None, pseudopod_min_size: int=50, csc_dict: dict=None, lighter_background: bool= None, bio_mask=None, back_mask=None):
        """
        Network Detection Function

        Perform network detection and pseudopod analysis on an image.

        Parameters
        ----------
        arenas_mask : NDArray, optional
            The mask indicating the arena regions in the image.
        pseudopod_min_size : int, optional
            The minimum size for pseudopods to be detected.
        csc_dict : dict, optional
            A dictionary containing color space conversion parameters. If None,
            defaults to {'bgr': np.array((1, 1, 1), np.int8), 'logical': 'None'}
        lighter_background : bool, optional
            Whether the background is lighter or not
        bio_mask : NDArray, optional
            The mask for biological objects in the image.
        back_mask : NDArray, optional
            The background mask.

        Notes
        -----
        This function modifies the object's state by setting `self.im_combinations`
        with the results of network detection and pseudopod analysis.
        """
        logging.info(f"Start automatic detection of network(s) in the last image")
        if len(self.bgr.shape) == 3:
            if csc_dict is None:
                csc_dict = {'bgr': np.array((1, 1, 1), np.int8), 'logical': 'None'}
            self._get_all_color_spaces()
            # csc_dict = translate_dict(csc_dict)
            # self.image = combine_color_spaces(csc_dict, self.all_c_spaces)
            first_dict, second_dict, c_spaces = split_dict(csc_dict)
            self.image, _, _, first_pc_vector = generate_color_space_combination(self.bgr, c_spaces, first_dict, second_dict, all_c_spaces=self.all_c_spaces)
            # if first_pc_vector is not None:
            #     csc_dict = {"bgr": first_pc_vector, "logical": 'None'}
        greyscale = self.image
        NetDet = NetworkDetection(greyscale, possibly_filled_pixels=arenas_mask)
        NetDet.get_best_network_detection_method()
        if lighter_background is None:
            lighter_background = True
            if arenas_mask.any() and not arenas_mask.all():
                lighter_background = NetDet.greyscale_image[arenas_mask > 0].mean() < NetDet.greyscale_image[arenas_mask == 0].mean()
        NetDet.detect_pseudopods(lighter_background, pseudopod_min_size=pseudopod_min_size, only_one_connected_component=False)
        NetDet.merge_network_with_pseudopods()
        cc_efficiency_order = np.argsort(NetDet.quality_metrics)
        self.im_combinations = []
        for _i in cc_efficiency_order:
            res_i = NetDet.all_results[_i]
            self.im_combinations.append({})
            self.im_combinations[len(self.im_combinations) - 1]["csc"] = csc_dict
            self.im_combinations[len(self.im_combinations) - 1]["converted_image"] = bracket_to_uint8_image_contrast(res_i['filtered'])
            self.im_combinations[len(self.im_combinations) - 1]["binary_image"] = res_i['binary']
            self.im_combinations[len(self.im_combinations) - 1]['filter_spec']= {'filter1_type': res_i['filter'], 'filter1_param': [np.min(res_i['sigmas']), np.max(res_i['sigmas'])], 'filter2_type': "", 'filter2_param': [1., 1.]}
            self.im_combinations[len(self.im_combinations) - 1]['rolling_window']= res_i['rolling_window']

    def get_crop_coordinates(self):
        """
        Get the crop coordinates for image processing.

        This function projects the image on both x and y axes to detect rows
        and columns of arenas, calculates the boundaries for cropping,
        and determines if the arenas are zigzagged.-

        """
        logging.info("Project the image on the y axis to detect rows of arenas")
        self.y_boundaries, y_max_sum = self.projection_to_get_peaks_boundaries(axis=1)
        logging.info("Project the image on the x axis to detect columns of arenas")
        self.x_boundaries, x_max_sum = self.projection_to_get_peaks_boundaries(axis=0)
        logging.info("Get crop coordinates using the get_crop_coordinates method of OneImageAnalysis class")
        row_number = len(np.nonzero(self.y_boundaries)[0]) // 2
        col_number = len(np.nonzero(self.x_boundaries)[0]) // 2
        are_zigzag = None
        if col_number > 0 and row_number > 0:
            if (x_max_sum / col_number) * 2 < (y_max_sum / row_number):
                are_zigzag = "columns"
            elif (x_max_sum / col_number) > (y_max_sum / row_number) * 2:
                are_zigzag = "rows"
        # here automatically determine if are zigzag
        x_boundary_number = (self.x_boundaries == 1).sum()
        if x_boundary_number > 1:
            if x_boundary_number < 4:
                x_interval = np.absolute(np.max(np.diff(np.where(self.x_boundaries == 1)[0]))) // 2
            else:
                if are_zigzag == "columns":
                    x_interval = np.absolute(np.max(np.diff(np.where(self.x_boundaries == 1)[0][::2]))) // 2
                else:
                    x_interval = np.absolute(np.max(np.diff(np.where(self.x_boundaries == 1)[0]))) // 2
            cx_min = np.where(self.x_boundaries == - 1)[0][0] - x_interval.astype(int)
            cx_max = np.where(self.x_boundaries == 1)[0][col_number - 1] + x_interval.astype(int)
            if cx_min < 0: cx_min = 0
            if cx_max > len(self.x_boundaries): cx_max = len(self.x_boundaries) - 1
        else:
            cx_min = 0
            cx_max = len(self.x_boundaries)# - 1

        y_boundary_number = (self.y_boundaries == 1).sum()
        if y_boundary_number > 1:
            if y_boundary_number < 4:
                y_interval = np.absolute(np.max(np.diff(np.where(self.y_boundaries == 1)[0]))) // 2
            else:
                if are_zigzag == "rows":
                    y_interval = np.absolute(np.max(np.diff(np.where(self.y_boundaries == 1)[0][::2]))) // 2
                else:
                    y_interval = np.absolute(np.max(np.diff(np.where(self.y_boundaries == 1)[0]))) // 2
            cy_min = np.where(self.y_boundaries == - 1)[0][0] - y_interval.astype(int)
            cy_max = np.where(self.y_boundaries == 1)[0][row_number - 1] + y_interval.astype(int)
            if cy_min < 0: cy_min = 0
            if cy_max > len(self.y_boundaries): cy_max = len(self.y_boundaries) - 1
        else:
            cy_min = 0
            cy_max = len(self.y_boundaries)# - 1

        self.crop_coord = [cy_min, cy_max, cx_min, cx_max]

    def projection_to_get_peaks_boundaries(self, axis: int) -> Tuple[NDArray, int]:
        """

        Projection to get peaks' boundaries.

        Calculate the projection of an array along a specified axis and
        identify the boundaries of non-zero peaks.

        Args:
            axis: int,
                The axis along which to calculate the projection and identify
                peaks' boundaries.

        Returns:
            Tuple[NDArray, int]:
                A tuple containing two elements: an array representing the slopes
                of peaks' boundaries and an integer representing the maximum sum
                along the specified axis.

        """
        sums = np.sum(self.validated_shapes, axis)
        slopes = np.greater(sums, 0)
        slopes = np.append(0, np.diff(slopes))
        coord = np.nonzero(slopes)[0]
        for ci in np.arange(len(coord)):
            if ci % 2 == 0:
                slopes[coord[ci]] = - 1
        return slopes, sums.max()

    def automatically_crop(self, crop_coord):
        """
        Automatically crops the image using the given crop coordinates.

        This method crops various attributes of the image such as the main image,
        binary image, and color spaces. It also updates internal states related to
        cropping.

        Args:
            crop_coord (tuple): The coordinates for cropping in the format
                (start_y, end_y, start_x, end_x), representing the bounding box region
                to crop from the image.

        """
        if not self.cropped and crop_coord is not None:
            logging.info("Crop using the automatically_crop method of OneImageAnalysis class")
            self.cropped = True
            self.image = self.image[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            self.bgr = deepcopy(self.bgr[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...])
            self._get_all_color_spaces()
            if self.im_combinations is not None:
                for i in np.arange(len(self.im_combinations)):
                    self.im_combinations[i]["binary_image"] = self.im_combinations[i]["binary_image"][crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3]]
                    self.im_combinations[i]["converted_image"] = self.im_combinations[i]["converted_image"][crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3]]
            self.binary_image = self.binary_image[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3]]
            if self.greyscale is not None:
                self.greyscale = self.greyscale[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            if self.greyscale2 is not None:
                self.greyscale2 = self.greyscale2[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            if self.image2 is not None:
                self.image2 = self.image2[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            if self.binary_image2 is not None:
                self.binary_image2 = self.binary_image2[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            if self.subtract_background is not None:
                self.subtract_background = self.subtract_background[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            if self.subtract_background2 is not None:
                self.subtract_background2 = self.subtract_background2[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            self.validated_shapes = self.validated_shapes[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3]]

            self.y_boundaries, y_max_sum = self.projection_to_get_peaks_boundaries(axis=1)
            self.x_boundaries, x_max_sum = self.projection_to_get_peaks_boundaries(axis=0)

